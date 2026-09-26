"""Escalation policies: the part of a cascade that decides whether the cheap answer stands."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models.base import Action, ModelResponse
from .pricing import Usage
from .suite import Task
from .tools import TOOL_NAMES, Workspace


@dataclass
class StepObservation:
    """Everything a policy is allowed to look at before deciding."""

    task: Task
    step: int
    prompt_tokens: int
    response: ModelResponse
    workspace: Workspace
    history: list[dict[str, Any]]

    @property
    def action(self) -> Action:
        return self.response.action


@dataclass
class Decision:
    escalate: bool
    reason: str
    usage: Usage | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalate": self.escalate,
            "reason": self.reason,
            "score": self.score,
            "usage": self.usage.to_dict() if self.usage else None,
        }


class Policy(Protocol):
    name: str

    def decide(self, obs: StepObservation) -> Decision: ...


# ---------------------------------------------------------------------------
# Structural validation. Free, and the floor any other policy has to beat.
# ---------------------------------------------------------------------------


def validation_errors(action: Action, task: Task, workspace: Workspace) -> list[str]:
    """Everything wrong with an action that can be seen without knowing the answer."""
    errors: list[str] = []
    if action.tool not in TOOL_NAMES:
        errors.append(f"unknown tool {action.tool!r}")
        return errors
    args = action.args
    if action.tool == "read_file":
        path = args.get("path")
        if not path:
            errors.append("read_file without path")
        elif path not in workspace.files:
            errors.append(f"read_file on missing path {path!r}")
    elif action.tool == "grep":
        if not args.get("pattern"):
            errors.append("grep without pattern")
    elif action.tool == "set_value":
        path, key = args.get("path"), args.get("key")
        if not path:
            errors.append("set_value without path")
        elif path not in workspace.files:
            errors.append(f"set_value on missing path {path!r}")
        elif workspace.get_raw(path, str(key)) is None:
            errors.append(f"set_value on missing key {key!r} in {path!r}")
        if "value" not in args:
            errors.append("set_value without value")
        if path and str(path).startswith("examples/"):
            errors.append("set_value targets documentation under examples/")
    elif action.tool == "finish":
        errors.extend(_finish_errors(args, task))
    return errors


def _finish_errors(args: dict[str, Any], task: Task) -> list[str]:
    spec = next((c["summary"] for c in task.checks if "summary" in c), None)
    raw = args.get("summary")
    if spec is None:
        return []
    if not raw:
        return ["finish without the summary this task requires"]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ["finish summary is not valid JSON"]
    if not isinstance(parsed, dict):
        return ["finish summary is not a JSON object"]
    want = set(spec.get("exact_keys", []))
    if want and set(parsed) != want:
        return [f"finish summary keys {sorted(parsed)} != {sorted(want)}"]
    return []


def features(obs: StepObservation) -> dict[str, float]:
    """The feature vector the learned policy is fitted on. Uses nothing a live run lacks."""
    action = obs.action
    errors = validation_errors(action, obs.task, obs.workspace)
    requires_summary = any("summary" in c for c in obs.task.checks)
    return {
        "bias": 1.0,
        "confidence": float(action.confidence if action.confidence is not None else 0.5),
        "schema_invalid": 1.0 if errors else 0.0,
        "is_write": 1.0 if action.tool == "set_value" else 0.0,
        "is_finish": 1.0 if action.tool == "finish" else 0.0,
        "step_frac": min(1.0, obs.step / 8.0),
        "prompt_ktokens": obs.prompt_tokens / 1000.0,
        "empty_summary_when_required": 1.0
        if (requires_summary and action.tool == "finish" and len(str(action.args.get("summary", ""))) < 8)
        else 0.0,
        "n_prior_writes": min(1.0, sum(1 for h in obs.history if h.get("tool") == "set_value") / 4.0),
    }


FEATURE_NAMES = (
    "bias",
    "confidence",
    "schema_invalid",
    "is_write",
    "is_finish",
    "step_frac",
    "prompt_ktokens",
    "empty_summary_when_required",
    "n_prior_writes",
)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass
class AlwaysCheap:
    """Never escalate. The cost floor, and the success floor with it."""

    name: str = "always_cheap"

    def decide(self, obs: StepObservation) -> Decision:
        return Decision(escalate=False, reason="policy never escalates")


@dataclass
class AlwaysStrong:
    """Escalate unconditionally. Equivalent to the single-strong-model baseline plus a wasted cheap call."""

    name: str = "always_strong"

    def decide(self, obs: StepObservation) -> Decision:
        return Decision(escalate=True, reason="policy always escalates")


@dataclass
class SelfConfidence:
    """Escalate when the cheap model's own reported probability is below a threshold."""

    threshold: float = 0.75
    name: str = "self_confidence"

    def decide(self, obs: StepObservation) -> Decision:
        conf = obs.action.confidence
        if conf is None:
            return Decision(escalate=True, reason="no confidence reported", score=None)
        return Decision(
            escalate=conf < self.threshold,
            reason=f"confidence {conf:.3f} {'<' if conf < self.threshold else '>='} {self.threshold:.2f}",
            score=conf,
        )


@dataclass
class SchemaValidation:
    """Escalate only when the cheap action is structurally impossible. Costs nothing to run."""

    name: str = "schema_validation"

    def decide(self, obs: StepObservation) -> Decision:
        errors = validation_errors(obs.action, obs.task, obs.workspace)
        return Decision(
            escalate=bool(errors),
            reason="; ".join(errors) if errors else "action validates",
            score=0.0 if errors else 1.0,
        )


class Verifier(Protocol):
    """A second model asked whether the cheap action should stand."""

    tier: str

    def judge(self, obs: StepObservation) -> tuple[bool, str, Usage]: ...


@dataclass
class VerifierModel:
    """Escalate when a separate, mid-tier model rejects the cheap action.

    The verifier is not free, which is the whole difficulty: it has to be cheaper than
    the escalations it prevents, and it has to be right more often than a free
    structural check.
    """

    verifier: Verifier
    name: str = "verifier_model"

    def decide(self, obs: StepObservation) -> Decision:
        accept, reason, usage = self.verifier.judge(obs)
        return Decision(escalate=not accept, reason=f"verifier: {reason}", usage=usage, score=1.0 if accept else 0.0)


@dataclass
class LearnedThreshold:
    """Escalate when a logistic model fitted on recorded shadow runs says the step is likely wrong."""

    weights: dict[str, float]
    threshold: float = 0.5
    name: str = "learned_threshold"

    @property
    def raw_name(self) -> str:
        return f"learned_threshold@{self.threshold:.2f}"

    def p_wrong(self, obs: StepObservation) -> float:
        f = features(obs)
        z = sum(self.weights.get(k, 0.0) * v for k, v in f.items())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def decide(self, obs: StepObservation) -> Decision:
        p = self.p_wrong(obs)
        return Decision(
            escalate=p >= self.threshold,
            reason=f"p(wrong)={p:.3f} vs threshold {self.threshold:.2f}",
            score=p,
        )


@dataclass
class OraclePolicy:
    """Escalates exactly when the cheap action is actually wrong. Not deployable; it is the ceiling.

    Every other policy is a lossy approximation of this one, so the gap between a
    policy and the oracle is the honest measure of how much routing headroom is left.
    """

    name: str = "oracle"
    gold_lookup: Any = None

    def decide(self, obs: StepObservation) -> Decision:
        gold = (self.gold_lookup or (lambda o: None))(obs)
        if gold is None:
            return Decision(escalate=False, reason="no gold action for this step")
        same = obs.action.tool == gold["tool"] and _args_equal(obs.action.args, gold.get("args", {}))
        return Decision(escalate=not same, reason="cheap action matches gold" if same else "cheap action differs from gold")


def _args_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = set(a) | set(b)
    keys.discard("confidence")
    return all(str(a.get(k)) == str(b.get(k)) for k in keys)


POLICY_ORDER = (
    "always_cheap",
    "schema_validation",
    "self_confidence",
    "verifier_model",
    "learned_threshold",
    "always_strong",
)


@dataclass
class PolicyStats:
    """What a policy actually did, so its cost can be attributed to its own decisions."""

    decisions: int = 0
    escalations: int = 0
    usd_overhead: float = 0.0
    overhead_calls: int = 0
    latency_overhead_ms: float = 0.0
    scores: list[float] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.decisions if self.decisions else 0.0

    def record(self, decision: Decision) -> None:
        self.decisions += 1
        self.escalations += int(decision.escalate)
        if decision.usage is not None:
            self.usd_overhead += decision.usage.usd
            self.overhead_calls += 1
            self.latency_overhead_ms += decision.usage.latency_ms
        if decision.score is not None:
            self.scores.append(decision.score)

    def merge(self, other: PolicyStats) -> None:
        self.decisions += other.decisions
        self.escalations += other.escalations
        self.usd_overhead += other.usd_overhead
        self.overhead_calls += other.overhead_calls
        self.latency_overhead_ms += other.latency_overhead_ms
        self.scores.extend(other.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "escalations": self.escalations,
            "escalation_rate": round(self.escalation_rate, 4),
            "usd_overhead": self.usd_overhead,
            "overhead_calls": self.overhead_calls,
            "latency_overhead_ms": round(self.latency_overhead_ms, 1),
        }
