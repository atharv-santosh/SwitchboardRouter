"""Verifier models: a second opinion on the cheap model's action, simulated and real."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..policies import StepObservation, validation_errors
from ..pricing import Usage, tier
from .simulated import _unit, estimate_tokens

VERIFIER_INSTRUCTION = """You are reviewing one step of another agent's run before it executes.

Decide whether the proposed action is the right next action for the stated task, given
the manual, the steps already taken and the current state. Accept only if executing it
moves the task towards completion correctly. Reject a plausible-looking action whose
value, path or arithmetic is wrong. Reject a `finish` that leaves required work undone
or omits a summary the task asked for.

Answer with the `verdict` tool and nothing else.
"""

VERDICT_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "verdict",
        "description": "Accept or reject the proposed action.",
        "parameters": {
            "type": "object",
            "properties": {
                "accept": {"type": "boolean"},
                "reason": {"type": "string", "description": "One short sentence."},
            },
            "required": ["accept", "reason"],
        },
    }
]


def render_review(obs: StepObservation) -> str:
    """The volatile half of the verifier's prompt."""
    lines = [
        f"Task: {obs.task.prompt}",
        "",
        "Steps already taken:",
    ]
    if not obs.history:
        lines.append("  (none)")
    for i, h in enumerate(obs.history, 1):
        lines.append(f"  {i}. {h['tool']}({json.dumps(h['args'], sort_keys=True)}) -> {h['outcome']}")
    lines += [
        "",
        "Current workspace:",
    ]
    for path, text in sorted(obs.workspace.files.items()):
        lines.append(f"--- {path}\n{text.rstrip()}")
    lines += [
        "",
        f"Proposed action: {obs.action.tool}({json.dumps(obs.action.args, sort_keys=True)})",
        f"The proposing model reported confidence {obs.action.confidence}.",
    ]
    return "\n".join(lines)


@dataclass
class SimulatedVerifier:
    """A noisy oracle: it knows whether the action was right and reports that imperfectly.

    Simulating a verifier by asking a simulated model to reason would only measure the
    simulation. What matters for routing is a verifier's accuracy, so that is what is
    parameterised: the verifier sees ground truth and gets it right with a probability
    set by its tier. Its cost and latency are charged the same way a real one's are.
    """

    tier: str
    seed: int = 0
    accuracy: float | None = None

    def __post_init__(self) -> None:
        self._spec = tier(self.tier)
        if self.accuracy is None:
            # A mid-tier verifier is right about four times in five; a stronger one more often.
            self.accuracy = 0.55 + 0.40 * self._spec.sim_skill

    def judge(self, obs: StepObservation) -> tuple[bool, str, Usage]:
        gold = obs.task.gold_plan[min(obs.step, len(obs.task.gold_plan) - 1)]
        truly_correct = obs.action.tool == gold["tool"] and all(
            str(obs.action.args.get(k)) == str(v) for k, v in (gold.get("args") or {}).items()
        )
        key = (self.seed, self.tier, obs.task.id, obs.step, obs.action.tool)
        assert self.accuracy is not None
        right = _unit("verify", *key) < self.accuracy
        accept = truly_correct if right else not truly_correct

        prompt_tokens = estimate_tokens(VERIFIER_INSTRUCTION + render_review(obs))
        out_tokens = 24
        usage = Usage(
            tier=self.tier,
            model=self._spec.model,
            tokens_in=prompt_tokens,
            tokens_out=out_tokens,
            tokens_thought=int(out_tokens * 1.4),
            latency_ms=self._spec.sim_latency_ms * 0.6 * (0.8 + 0.4 * _unit("vlat", *key)),
        )
        reason = ("looks correct" if accept else "does not follow from the task") + (
            "" if right else " (verifier was wrong here)"
        )
        return accept, reason, usage


@dataclass
class LiveVerifier:
    """Asks a real model for a verdict, with the structural check folded in as a free prior."""

    client: Any
    tier: str

    def judge(self, obs: StepObservation) -> tuple[bool, str, Usage]:
        from .base import ModelRequest

        # A structurally impossible action needs no model call to reject, and paying for
        # one anyway is the most common way a verifier stops being worth its price.
        errors = validation_errors(obs.action, obs.task, obs.workspace)
        if errors:
            return False, f"structurally invalid: {errors[0]}", Usage(tier=self.tier, model="", latency_ms=0.0)

        request = ModelRequest(
            stable_prefix=VERIFIER_INSTRUCTION,
            volatile_suffix=render_review(obs),
            tools=VERDICT_SCHEMA,
            max_output_tokens=128,
            meta={"purpose": "verify", "task_id": obs.task.id, "step": obs.step},
        )
        response = self.client.complete(request)
        args = response.action.args
        accept = bool(args.get("accept", True))
        reason = str(args.get("reason", ""))[:120]
        return accept, reason, response.usage
