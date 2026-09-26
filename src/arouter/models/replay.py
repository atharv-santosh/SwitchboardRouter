"""Replaying committed trajectories, so every real-model measurement reproduces with no credentials."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

from ..events import iter_runs
from ..policies import StepObservation
from ..pricing import Usage
from .base import Action, ModelRequest, ModelResponse


class ReplayMiss(KeyError):
    """A request the fixtures do not contain, which means the code path changed since recording."""


@dataclass
class FixtureIndex:
    """Model responses and escalation verdicts recovered from recorded JSONL trajectories."""

    calls: dict[tuple[str, str], Deque[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    verdicts: dict[str, Deque[dict[str, Any]]] = field(default_factory=lambda: defaultdict(deque))
    run_ids: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, directory: str | Path) -> FixtureIndex:
        index = cls()
        for run_id, events in iter_runs(directory):
            index.run_ids.append(run_id)
            pending: tuple[str, str] | None = None
            for event in events:
                if event.type == "llm_call":
                    pending = (event.payload["tier"], event.payload["request_hash"])
                elif event.type == "llm_result" and pending is not None:
                    index.calls[pending].append(event.payload)
                    pending = None
                elif event.type == "decision" and event.payload.get("kind") == "escalation":
                    index.verdicts[run_id].append(event.payload)
        return index

    def __len__(self) -> int:
        return sum(len(q) for q in self.calls.values())


@dataclass
class ReplayModel:
    """Serves a recorded response for a request whose hash matches, and refuses otherwise."""

    tier: str
    index: FixtureIndex

    def complete(self, request: ModelRequest, cache_handle: str | None = None) -> ModelResponse:
        key = (self.tier, request.request_hash)
        queue = self.index.calls.get(key)
        if not queue:
            raise ReplayMiss(
                f"no recorded {self.tier} call for request {request.request_hash[:12]}; "
                "the prompt changed since these fixtures were recorded, so re-record rather than trusting a replay"
            )
        # Reuse the last recorded response once the queue is drained, which happens when a
        # run revisits an identical state; an empty queue is never silently a fresh guess.
        payload = queue.popleft() if len(queue) > 1 else queue[0]
        raw_action = payload["action"]
        raw_usage = payload["usage"]
        action = Action(
            tool=raw_action["tool"],
            args=dict(raw_action.get("args", {})),
            confidence=raw_action.get("confidence"),
        )
        usage = Usage(
            tier=raw_usage["tier"],
            model=raw_usage["model"],
            tokens_in=int(raw_usage["tokens_in"]),
            tokens_cached_in=int(raw_usage["tokens_cached_in"]),
            tokens_out=int(raw_usage["tokens_out"]),
            tokens_thought=int(raw_usage.get("tokens_thought", 0)),
            latency_ms=float(raw_usage["latency_ms"]),
        )
        return ModelResponse(
            action=action, usage=usage, cache_hit=bool(payload.get("cache_hit")), error=payload.get("error")
        )


@dataclass
class ReplayFactory:
    index: FixtureIndex

    def make(self, tier: str) -> ReplayModel:
        return ReplayModel(tier=tier, index=self.index)


@dataclass
class ReplayVerifier:
    """Replays the verdicts a live verifier gave, in the order it gave them for this run."""

    tier: str
    run_id: str
    index: FixtureIndex

    def judge(self, obs: StepObservation) -> tuple[bool, str, Usage]:
        queue = self.index.verdicts.get(self.run_id)
        if not queue:
            raise ReplayMiss(f"no recorded verifier verdict for {self.run_id} step {obs.step}")
        payload = queue.popleft() if len(queue) > 1 else queue[0]
        raw = payload.get("usage") or {}
        usage = Usage(
            tier=raw.get("tier", self.tier),
            model=raw.get("model", ""),
            tokens_in=int(raw.get("tokens_in", 0)),
            tokens_cached_in=int(raw.get("tokens_cached_in", 0)),
            tokens_out=int(raw.get("tokens_out", 0)),
            tokens_thought=int(raw.get("tokens_thought", 0)),
            latency_ms=float(raw.get("latency_ms", 0.0)),
        )
        return not payload["escalate"], str(payload.get("reason", "replayed")), usage
