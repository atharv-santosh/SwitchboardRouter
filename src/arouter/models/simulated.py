"""A deterministic stand-in for a real model, so the routing logic is testable with no credentials.

This is an oracle-driven stub, not a model. The task suite hands it the correct action
for the step; per-tier `sim_skill` decides whether it returns that action or a corrupted
one, and the corruption modes are chosen so that some failures are structurally
detectable and some are not. That distinction is the whole point: a schema-validation
escalation policy that caught every simulated failure would tell us nothing about the
real one.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from ..pricing import Usage, tier
from .base import Action, ModelRequest, ModelResponse

# Failures that a structural validator can see, and failures it cannot.
DETECTABLE_CORRUPTIONS = ("bad_path", "missing_arg", "unknown_tool")
SILENT_CORRUPTIONS = ("wrong_value", "decoy_target", "premature_finish")
CORRUPTIONS = DETECTABLE_CORRUPTIONS + SILENT_CORRUPTIONS

CHARS_PER_TOKEN = 3.9
# How sharply a tier's skill advantage over a task's difficulty turns into success.
SKILL_GAIN = 9.0


def _unit(*parts: Any) -> float:
    """A stable pseudo-random float in [0, 1) derived from the given parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass
class SimulatedModel:
    """Deterministic model whose competence and latency are set per tier."""

    tier: str
    seed: int = 0
    sleep: bool = False
    latency_scale: float = 1.0

    def __post_init__(self) -> None:
        self._spec = tier(self.tier)

    def complete(self, request: ModelRequest, cache_handle: str | None = None) -> ModelResponse:
        meta = request.meta
        gold: dict[str, Any] = meta.get("gold_action") or {"tool": "finish", "args": {}}
        difficulty = float(meta.get("difficulty", 0.5))
        key = (self.seed, self.tier, meta.get("task_id"), meta.get("step"), meta.get("attempt", 0))

        # Difficulty is calibrated at the task level, so the per-step probability is the
        # plan-length root of it. Setting it per step instead would make a seven-step task
        # impossible for every tier regardless of difficulty, which is an artefact of the
        # simulation rather than anything about routing.
        margin = self._spec.sim_skill - difficulty
        p_task = 1.0 / (1.0 + pow(2.718281828, -SKILL_GAIN * margin))
        p_step = p_task ** (1.0 / max(1, int(meta.get("plan_len", 1))))
        correct = _unit("correct", *key) < p_step

        if correct:
            action = Action(tool=gold["tool"], args=dict(gold.get("args", {})))
            confidence = 0.62 + 0.36 * _unit("conf_ok", *key)
        else:
            action = self._corrupt(gold, key, meta)
            confidence = 0.40 + 0.52 * _unit("conf_bad", *key)
        action.confidence = round(confidence, 4)

        latency_ms = self._spec.sim_latency_ms * (0.75 + 0.5 * _unit("lat", *key)) * self.latency_scale
        if self.sleep:
            time.sleep(latency_ms / 1000.0)

        prompt_tokens = estimate_tokens(request.prompt())
        cache_hit = cache_handle is not None
        cached = estimate_tokens(request.stable_prefix) if cache_hit else 0
        out_tokens = max(8, int(20 + 60 * _unit("out", *key)))
        thought = int(out_tokens * 1.4) if self._spec.sim_skill > 0.5 else 0

        usage = Usage(
            tier=self.tier,
            model=self._spec.model,
            tokens_in=prompt_tokens,
            tokens_cached_in=cached,
            tokens_out=out_tokens,
            tokens_thought=thought,
            latency_ms=latency_ms,
        )
        return ModelResponse(action=action, usage=usage, cache_hit=cache_hit)

    def _corrupt(self, gold: dict[str, Any], key: tuple[Any, ...], meta: dict[str, Any]) -> Action:
        """Turn the correct action into a wrong one, in one of six documented ways."""
        mode = CORRUPTIONS[int(_unit("mode", *key) * len(CORRUPTIONS))]
        args = dict(gold.get("args", {}))
        toolname = gold["tool"]

        if mode == "bad_path" and "path" in args:
            return Action(tool=toolname, args={**args, "path": f"config/{args['path']}.bak"})
        if mode == "missing_arg" and args:
            dropped = sorted(args)[0]
            return Action(tool=toolname, args={k: v for k, v in args.items() if k != dropped})
        if mode == "unknown_tool":
            return Action(tool="inspect_service", args=args)
        if mode == "decoy_target" and meta.get("decoy_path"):
            return Action(tool=toolname, args={**args, "path": meta["decoy_path"]})
        if mode == "premature_finish":
            return Action(tool="finish", args={"summary": "{}"})
        # wrong_value, and the fallback when the mode above did not apply.
        for k, v in args.items():
            if isinstance(v, str) and v.isdigit():
                return Action(tool=toolname, args={**args, k: str(int(v) + 1)})
            if isinstance(v, int):
                return Action(tool=toolname, args={**args, k: v + 1})
            if isinstance(v, str) and k in ("content", "value", "summary"):
                return Action(tool=toolname, args={**args, k: v.replace("2", "3", 1) + " "})
        return Action(tool="finish", args={"summary": "{}"})
