"""The cascade: a cheap attempt, a decision about whether it stands, and an escalation if not."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .budget import BudgetEnforcer, Degradation
from .cache import PrefixCache
from .context import Context, layout as make_layout
from .events import Recorder
from .models.base import ModelRequest, ModelResponse
from .models.simulated import estimate_tokens
from .policies import (
    AlwaysCheap,
    AlwaysStrong,
    Decision,
    LearnedThreshold,
    OraclePolicy,
    Policy,
    PolicyStats,
    SchemaValidation,
    SelfConfidence,
    StepObservation,
    VerifierModel,
    features,
)
from .prefetch import Prefetcher
from .pricing import LADDER, Spend, VERIFIER_TIER
from .suite import Grade, Suite, Task, grade
from .tools import TOOL_SCHEMAS, ToolResult, Workspace, execute, render_workspace_listing

OUTPUT_CONVENTIONS = """Output conventions.

Reply with exactly one tool call per turn and no prose outside it. Do not batch
several edits into one call. Report `confidence` on `finish` as an honest
probability that the whole task is now correctly done.
"""


class ModelFactory(Protocol):
    """Produces a client for a named tier; the runner never constructs clients itself."""

    def make(self, tier: str) -> Any: ...


@dataclass
class RunConfig:
    """One row of the frontier: which tiers, which policy, which layout, which levers on."""

    name: str
    tiers: tuple[str, ...] = LADDER
    policy: str = "schema_validation"
    layout: str = "cache_aware"
    cache_enabled: bool = True
    prefetch_enabled: bool = False
    max_steps: int = 10
    tool_latency_ms: float = 0.0
    tool_usd: float = 0.0
    run_cap_usd: float | None = None
    confidence_threshold: float = 0.75
    learned_threshold: float = 0.5
    learned_weights: dict[str, float] = field(default_factory=dict)
    # Shadow mode calls both tiers every step, executes the strong action, and labels the
    # cheap one by whether it agreed. It is how the learned policy's training data is
    # collected, and how you would collect it in production: sample a slice of traffic.
    shadow: bool = False

    @property
    def cascade(self) -> bool:
        return len(self.tiers) > 1

    @property
    def cheap_tier(self) -> str:
        return self.tiers[0]

    @property
    def strong_tier(self) -> str:
        return self.tiers[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tiers": list(self.tiers),
            "policy": self.policy if self.cascade else "none",
            "layout": self.layout,
            "cache_enabled": self.cache_enabled,
            "prefetch_enabled": self.prefetch_enabled,
            "max_steps": self.max_steps,
            "tool_latency_ms": self.tool_latency_ms,
            "run_cap_usd": self.run_cap_usd,
            "confidence_threshold": self.confidence_threshold,
            "learned_threshold": self.learned_threshold,
        }


@dataclass
class StepRecord:
    step: int
    tier_used: str
    action: dict[str, Any]
    escalated: bool
    decision: dict[str, Any] | None
    usages: list[dict[str, Any]]
    tool_result: dict[str, Any] | None
    prefetch_hit: bool
    critical_path_ms: float


@dataclass
class TaskRun:
    run_id: str
    task_id: str
    band: str
    config: str
    solved: bool
    failures: list[str]
    usd: float
    critical_path_ms: float
    steps: list[StepRecord]
    spend: Spend
    escalations: int
    policy_stats: PolicyStats
    degradations: list[dict[str, Any]]
    summary: str | None
    stopped_by_budget: bool
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "band": self.band,
            "config": self.config,
            "solved": self.solved,
            "failures": self.failures,
            "usd": self.usd,
            "critical_path_ms": round(self.critical_path_ms, 1),
            "n_steps": len(self.steps),
            "escalations": self.escalations,
            "degradations": self.degradations,
            "stopped_by_budget": self.stopped_by_budget,
            "tokens_in": self.spend.tokens_in,
            "tokens_cached_in": self.spend.tokens_cached_in,
            "tokens_out": self.spend.tokens_out,
            "calls": self.spend.calls,
            "usd_by_tier": self.spend.by_tier,
        }


def build_context(suite: Suite, task: Task, ws: Workspace) -> Context:
    """Split the prompt into the task-independent half and the rest.

    The task statement goes in the volatile half even though it is constant within a
    run. Putting it in the prefix would give a cache that only one task can use; leaving
    it out gives one cache per tier that every task in the workload shares, which is the
    difference between a cache that pays for its storage and one that does not.
    """
    ctx = Context()
    ctx.add("role", "You are an infrastructure agent editing service configuration.", stable=True)
    ctx.add("manual", suite.manual, stable=True)
    ctx.add("tool_schemas", json.dumps(TOOL_SCHEMAS, indent=1, sort_keys=True), stable=True)
    ctx.add("workspace_listing", "Workspace paths:\n" + render_workspace_listing(ws), stable=True)
    ctx.add("conventions", OUTPUT_CONVENTIONS, stable=True, optional=True)
    ctx.add("task", f"Task: {task.prompt}", stable=False)
    return ctx


def render_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No steps taken yet."
    lines = ["Steps so far:"]
    for i, h in enumerate(history):
        args = json.dumps(h["args"], sort_keys=True)
        outcome = h["outcome"]
        lines.append(f"{i + 1}. {h['tool']}({args}) -> {outcome}")
    return "\n".join(lines)


def summarize_tool_result(result: ToolResult) -> str:
    if not result.ok:
        return f"ERROR: {result.error}"
    payload = result.payload
    if "paths" in payload:
        return "paths: " + ", ".join(payload["paths"])
    if "content" in payload and "key" not in payload:
        return "content:\n" + payload["content"].rstrip()
    if "matches" in payload:
        return "matches:\n" + ("\n".join(payload["matches"]) if payload["matches"] else "(none)")
    if "key" in payload:
        return f"wrote {payload['key']}={payload['value']} in {payload['path']}"
    return json.dumps(payload, sort_keys=True)[:400]


class CascadeRunner:
    """Executes one task under one configuration and reports what it cost."""

    def __init__(
        self,
        suite: Suite,
        factory: ModelFactory,
        *,
        prefix_cache: PrefixCache | None = None,
        prefetcher: Prefetcher | None = None,
        verifier_factory: Callable[[str], Any] | None = None,
        tenant_ledger: Any = None,
    ) -> None:
        self.suite = suite
        self.factory = factory
        self.prefix_cache = prefix_cache or PrefixCache(enabled=False)
        self.prefetcher = prefetcher
        self.verifier_factory = verifier_factory
        self.tenant_ledger = tenant_ledger
        self._clients: dict[str, Any] = {}

    def client(self, tier: str) -> Any:
        if tier not in self._clients:
            self._clients[tier] = self.factory.make(tier)
        return self._clients[tier]

    def _policy(self, config: RunConfig, task: Task) -> Policy:
        name = config.policy
        if name == "always_cheap":
            return AlwaysCheap()
        if name == "always_strong":
            return AlwaysStrong()
        if name == "schema_validation":
            return SchemaValidation()
        if name == "self_confidence":
            return SelfConfidence(threshold=config.confidence_threshold)
        if name == "learned_threshold":
            return LearnedThreshold(weights=config.learned_weights, threshold=config.learned_threshold)
        if name == "verifier_model":
            if self.verifier_factory is None:
                raise ValueError("verifier_model policy needs a verifier_factory")
            return VerifierModel(verifier=self.verifier_factory(VERIFIER_TIER))
        if name == "oracle":
            return OraclePolicy(gold_lookup=lambda obs: _gold_at(obs.task, obs.step))
        raise KeyError(f"unknown policy {name!r}")

    def run(self, task: Task, config: RunConfig, seed: int = 0) -> TaskRun:
        run_id = f"{config.name}::{task.id}::seed{seed}"
        recorder = Recorder(run_id=run_id)
        ws = self.suite.workspace.clone()
        initial = ws.clone()
        layout = make_layout(config.layout)
        ctx = build_context(self.suite, task, ws)
        policy: Policy = AlwaysStrong() if config.shadow else (
            self._policy(config, task) if config.cascade else AlwaysCheap()
        )
        budget = BudgetEnforcer(run_cap_usd=config.run_cap_usd, ledger=self.tenant_ledger)
        spend = Spend()
        policy_stats = PolicyStats()
        history: list[dict[str, Any]] = []
        steps: list[StepRecord] = []
        critical_path_ms = 0.0
        summary: str | None = None
        escalations = 0
        stopped_by_budget = False

        recorder.emit(
            "run_start",
            {
                "task_id": task.id,
                "band": task.band,
                "config": config.to_dict(),
                "seed": seed,
                "tenant": getattr(self.tenant_ledger, "tenant", None),
            },
        )

        for step in range(config.max_steps):
            for transition in budget.refresh():
                recorder.emit("decision", {"kind": "budget_degradation", **transition})
            if budget.stopped:
                stopped_by_budget = True
                recorder.emit(
                    "decision",
                    {
                        "kind": "budget_stop",
                        "detail": "cap reached; finishing with the work completed so far rather than truncating silently",
                        "run_spent_usd": budget.run_spent_usd,
                    },
                )
                summary = json.dumps({"budget_exhausted": True, "steps_completed": len(steps)})
                break

            active_ctx = ctx.drop_optional() if budget.shrink_context else ctx
            step_ctx = Context(blocks=list(active_ctx.blocks))
            step_ctx.add("history", render_history(history), stable=False)
            prefix, suffix = layout.render(step_ctx, step)

            tier_used = config.cheap_tier if config.cascade else config.tiers[0]
            request = ModelRequest(
                stable_prefix=prefix,
                volatile_suffix=suffix,
                tools=TOOL_SCHEMAS,
                meta={
                    "task_id": task.id,
                    "step": step,
                    "attempt": 0,
                    "difficulty": task.difficulty,
                    "plan_len": task.plan_len,
                    "gold_action": _gold_at(task, step),
                    "decoy_path": task.decoy_path,
                },
            )
            prompt_tokens = estimate_tokens(request.prompt())

            entry = self.prefix_cache.lookup_or_create(
                tier_used, prefix, TOOL_SCHEMAS, request.prefix_hash, estimate_tokens(prefix)
            ) if config.cache_enabled else None
            cached_tokens = entry.tokens if entry else 0

            if not budget.can_afford(tier_used, prompt_tokens, cached_tokens, request.max_output_tokens):
                budget.level = Degradation.STOP
                stopped_by_budget = True
                recorder.emit(
                    "decision",
                    {
                        "kind": "budget_refusal",
                        "detail": f"next {tier_used} call would exceed the cap; stopping cleanly",
                        "estimated_usd": budget.estimate_usd(
                            tier_used, prompt_tokens, cached_tokens, request.max_output_tokens
                        ),
                        "headroom_usd": budget.headroom(),
                    },
                )
                summary = json.dumps({"budget_exhausted": True, "steps_completed": len(steps)})
                break

            recent_tools = [h["tool"] for h in history]
            attempt = (
                self.prefetcher.maybe_start(ws, task.id, task.band, recent_tools)
                if (self.prefetcher and config.prefetch_enabled)
                else None
            )

            recorder.emit(
                "llm_call",
                {
                    "tier": tier_used,
                    "request_hash": request.request_hash,
                    "prefix_hash": request.prefix_hash,
                    "prompt_tokens_est": prompt_tokens,
                    "cache_handle": bool(entry),
                },
            )
            response = self._complete(tier_used, request, entry)
            spend.add(response.usage)
            budget.charge(response.usage)
            usages = [response.usage.to_dict()]
            step_ms = response.usage.latency_ms
            recorder.emit("llm_result", {"tier": tier_used, **response.to_dict()})

            decision: Decision | None = None
            if config.cascade:
                obs = StepObservation(
                    task=task,
                    step=step,
                    prompt_tokens=prompt_tokens,
                    response=response,
                    workspace=ws,
                    history=history,
                )
                active_policy = policy
                if config.policy == "verifier_model" and not config.shadow and not budget.allow_verifier:
                    active_policy = SchemaValidation()
                decision = active_policy.decide(obs)
                policy_stats.record(decision)
                if decision.usage is not None:
                    spend.add(decision.usage)
                    budget.charge(decision.usage)
                    usages.append(decision.usage.to_dict())
                    step_ms += decision.usage.latency_ms
                recorder.emit("decision", {"kind": "escalation", "policy": active_policy.name, **decision.to_dict()})

            action = response.action
            escalated = False
            if decision is not None and decision.escalate:
                if not budget.allow_escalation:
                    recorder.emit(
                        "decision",
                        {
                            "kind": "escalation_suppressed",
                            "detail": "budget degradation has disabled escalation; keeping the cheap action",
                        },
                    )
                else:
                    strong = config.strong_tier
                    strong_request = ModelRequest(
                        stable_prefix=prefix,
                        volatile_suffix=suffix,
                        tools=TOOL_SCHEMAS,
                        meta={**request.meta, "attempt": 1},
                    )
                    strong_entry = self.prefix_cache.lookup_or_create(
                        strong, prefix, TOOL_SCHEMAS, strong_request.prefix_hash, estimate_tokens(prefix)
                    ) if config.cache_enabled else None
                    if budget.can_afford(
                        strong,
                        prompt_tokens,
                        strong_entry.tokens if strong_entry else 0,
                        strong_request.max_output_tokens,
                    ):
                        recorder.emit(
                            "llm_call",
                            {
                                "tier": strong,
                                "request_hash": strong_request.request_hash,
                                "prefix_hash": strong_request.prefix_hash,
                                "prompt_tokens_est": prompt_tokens,
                                "cache_handle": bool(strong_entry),
                            },
                        )
                        strong_response = self._complete(strong, strong_request, strong_entry)
                        spend.add(strong_response.usage)
                        budget.charge(strong_response.usage)
                        usages.append(strong_response.usage.to_dict())
                        step_ms += strong_response.usage.latency_ms
                        recorder.emit("llm_result", {"tier": strong, **strong_response.to_dict()})
                        if config.shadow:
                            recorder.emit(
                                "decision",
                                {
                                    "kind": "shadow_label",
                                    "features": features(obs),
                                    "cheap_action": response.action.to_dict(),
                                    "strong_action": strong_response.action.to_dict(),
                                    "cheap_was_wrong": int(
                                        not _actions_agree(response.action, strong_response.action)
                                    ),
                                },
                            )
                        action = strong_response.action
                        tier_used = strong
                        escalated = True
                        escalations += 1
                    else:
                        recorder.emit(
                            "decision",
                            {
                                "kind": "escalation_refused",
                                "detail": "escalation would exceed the cap; keeping the cheap action",
                            },
                        )

            recorder.emit("tool_call", {"request": {"tool": action.tool, "args": action.args}})
            adopted = (
                self.prefetcher.resolve(attempt, action.tool, action.args, response.usage.latency_ms)
                if self.prefetcher
                else None
            )
            if adopted is not None:
                result = adopted
                prefetch_hit = True
            else:
                result = execute(ws, action.tool, action.args, config.tool_latency_ms)
                prefetch_hit = False
                step_ms += result.latency_ms
            recorder.emit("tool_result", {"response": result.to_dict(), "prefetch_hit": prefetch_hit})

            history.append(
                {"tool": action.tool, "args": action.args, "outcome": summarize_tool_result(result)}
            )
            critical_path_ms += step_ms
            steps.append(
                StepRecord(
                    step=step,
                    tier_used=tier_used,
                    action=action.to_dict(),
                    escalated=escalated,
                    decision=decision.to_dict() if decision else None,
                    usages=usages,
                    tool_result=result.to_dict(),
                    prefetch_hit=prefetch_hit,
                    critical_path_ms=step_ms,
                )
            )

            if action.tool == "finish" and result.ok:
                summary = result.payload.get("summary")
                break
        else:
            recorder.emit("error", {"message": f"step limit {config.max_steps} reached without finish"})

        verdict: Grade = grade(task, initial, ws, summary)
        recorder.emit(
            "run_end",
            {
                "solved": verdict.solved,
                "failures": verdict.failures,
                "usd": spend.usd,
                "critical_path_ms": round(critical_path_ms, 1),
                "escalations": escalations,
                "budget": budget.to_dict(),
                "policy": policy_stats.to_dict() if config.cascade else None,
            },
        )
        return TaskRun(
            run_id=run_id,
            task_id=task.id,
            band=task.band,
            config=config.name,
            solved=verdict.solved,
            failures=verdict.failures,
            usd=spend.usd,
            critical_path_ms=critical_path_ms,
            steps=steps,
            spend=spend,
            escalations=escalations,
            policy_stats=policy_stats,
            degradations=budget.transitions,
            summary=summary,
            stopped_by_budget=stopped_by_budget,
            events=[e.to_dict() for e in recorder.events],
        )

    def _complete(self, tier: str, request: ModelRequest, entry: Any) -> ModelResponse:
        client = self.client(tier)
        started = time.perf_counter()
        response = client.complete(request, cache_handle=entry.handle if entry else None)
        if response.usage.latency_ms == 0.0:
            response.usage.latency_ms = (time.perf_counter() - started) * 1000.0
        return response


def _actions_agree(a: Any, b: Any) -> bool:
    """Whether the cheap action is the one the strong model would have taken."""
    if a.tool != b.tool:
        return False
    keys = (set(a.args) | set(b.args)) - {"confidence"}
    return all(str(a.args.get(k)) == str(b.args.get(k)) for k in keys)


def _gold_at(task: Task, step: int) -> dict[str, Any] | None:
    if step < len(task.gold_plan):
        return task.gold_plan[step]
    return task.gold_plan[-1] if task.gold_plan else None


__all__ = ["CascadeRunner", "RunConfig", "StepRecord", "TaskRun", "build_context", "render_history"]
