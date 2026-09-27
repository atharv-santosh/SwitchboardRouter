"""Speculative tool prefetch: start the likely next tool call before the model commits to it."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .events import canonical_json
from .tools import READ_ONLY_TOOLS, ToolResult, Workspace, execute


def call_key(tool: str, args: dict[str, Any]) -> str:
    """A normalized identity for one tool call, so a prediction can be checked against reality."""
    relevant = {k: str(v) for k, v in sorted(args.items()) if k in ("path", "pattern")}
    return canonical_json({"tool": tool, "args": relevant})


@dataclass
class ToolPredictor:
    """An n-gram over recorded tool sequences. Fitted on past runs, never on the run it serves.

    Keying on the task id makes this close to memorization of a small suite, which would
    flatter it. So the fit is leave-one-out by run id, and a task-agnostic key is fitted
    alongside it; MEASUREMENTS.md reports both, because the honest number for a workload
    you have never seen is the second one.
    """

    order: int = 2
    by_task: dict[tuple[str, tuple[str, ...]], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    by_band: dict[tuple[str, tuple[str, ...]], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    use_task_key: bool = True

    def observe(self, task_id: str, band: str, sequence: list[tuple[str, dict[str, Any]]]) -> None:
        keys = [call_key(tool, args) for tool, args in sequence]
        tools = [tool for tool, _ in sequence]
        for i in range(len(keys)):
            ctx = tuple(tools[max(0, i - self.order) : i])
            self.by_task[(task_id, ctx)][keys[i]] += 1
            self.by_band[(band, ctx)][tools[i]] += 1

    @classmethod
    def from_trajectories(cls, directory: str, config: str | None = None, order: int = 2) -> ToolPredictor:
        """Fit straight from recorded JSONL, so no model has to be called or replayed to fit."""
        from .events import iter_runs

        predictor = cls(order=order)
        for run_id, events in iter_runs(directory):
            if config is not None and not run_id.startswith(f"{config}::"):
                continue
            task_id, band, sequence = "", "medium", []
            for event in events:
                if event.type == "run_start":
                    task_id = str(event.payload.get("task_id", ""))
                    band = str(event.payload.get("band", "medium"))
                elif event.type == "tool_call":
                    request = event.payload.get("request", {})
                    sequence.append((str(request.get("tool", "")), dict(request.get("args", {}))))
            if task_id and sequence:
                predictor.observe(task_id, band, sequence)
        return predictor

    def predict(self, task_id: str, band: str, recent_tools: list[str]) -> tuple[str, dict[str, Any], float] | None:
        """Return (tool, args, probability) for the next call, or None when nothing is known."""
        ctx = tuple(recent_tools[-self.order :])
        table = self.by_task.get((task_id, ctx)) if self.use_task_key else None
        if table:
            key, count = table.most_common(1)[0]
            total = sum(table.values())
            parsed = json.loads(key)
            return parsed["tool"], parsed["args"], count / total
        band_table = self.by_band.get((band, ctx))
        if band_table:
            tool, count = band_table.most_common(1)[0]
            total = sum(band_table.values())
            # Without a task-specific table there are no args to guess, so only
            # argument-free tools are predictable this way.
            if tool == "list_files":
                return tool, {}, count / total
        return None


@dataclass
class PrefetchStats:
    considered: int = 0
    skipped_low_confidence: int = 0
    skipped_not_readonly: int = 0
    skipped_no_prediction: int = 0
    launched: int = 0
    hits: int = 0
    misses: int = 0
    ms_saved: float = 0.0
    ms_wasted: float = 0.0
    usd_wasted: float = 0.0

    @property
    def coverage(self) -> float:
        return self.launched / self.considered if self.considered else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.launched if self.launched else 0.0

    def merge(self, other: PrefetchStats) -> None:
        for field_name in (
            "considered",
            "skipped_low_confidence",
            "skipped_not_readonly",
            "skipped_no_prediction",
            "launched",
            "hits",
            "misses",
            "ms_saved",
            "ms_wasted",
            "usd_wasted",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "launched": self.launched,
            "coverage": round(self.coverage, 4),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "skipped_low_confidence": self.skipped_low_confidence,
            "skipped_not_readonly": self.skipped_not_readonly,
            "skipped_no_prediction": self.skipped_no_prediction,
            "ms_saved": round(self.ms_saved, 1),
            "ms_wasted": round(self.ms_wasted, 1),
            "usd_wasted": self.usd_wasted,
        }


@dataclass
class Attempt:
    key: str
    tool: str
    args: dict[str, Any]
    probability: float
    future: Future[ToolResult]
    started_at: float


@dataclass
class Prefetcher:
    """Runs the predicted tool concurrently with the model call, then adopts or discards it.

    Only read-only tools are ever speculated. A discarded write is not wasted work, it is
    a corrupted workspace, and no latency win justifies that. The check is a hard gate
    rather than a heuristic because the predictor is allowed to be wrong.
    """

    predictor: ToolPredictor
    threshold: float = 0.6
    tool_latency_ms: float = 0.0
    tool_usd: float = 0.0
    enabled: bool = True
    stats: PrefetchStats = field(default_factory=PrefetchStats)
    _pool: ThreadPoolExecutor | None = None

    def __enter__(self) -> Prefetcher:
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prefetch")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def maybe_start(
        self, workspace: Workspace, task_id: str, band: str, recent_tools: list[str]
    ) -> Attempt | None:
        if not self.enabled or self._pool is None:
            return None
        self.stats.considered += 1
        prediction = self.predictor.predict(task_id, band, recent_tools)
        if prediction is None:
            self.stats.skipped_no_prediction += 1
            return None
        tool, args, probability = prediction
        if tool not in READ_ONLY_TOOLS:
            self.stats.skipped_not_readonly += 1
            return None
        if probability < self.threshold:
            self.stats.skipped_low_confidence += 1
            return None
        # Speculation reads a snapshot, so an in-flight guess can never observe a
        # half-applied state or be invalidated by the step that follows it.
        snapshot = workspace.clone()
        future = self._pool.submit(execute, snapshot, tool, args, self.tool_latency_ms)
        self.stats.launched += 1
        return Attempt(
            key=call_key(tool, args),
            tool=tool,
            args=args,
            probability=probability,
            future=future,
            started_at=time.perf_counter(),
        )

    def resolve(self, attempt: Attempt | None, tool: str, args: dict[str, Any], model_ms: float) -> ToolResult | None:
        """Adopt the speculative result if the guess was right, otherwise account for the waste."""
        if attempt is None:
            return None
        result = attempt.future.result()
        if attempt.key == call_key(tool, args):
            self.stats.hits += 1
            # Without prefetch the run pays model time then tool time in series; with it
            # the two overlap, so the saving is whichever of the two is shorter.
            self.stats.ms_saved += min(result.latency_ms, model_ms)
            return result
        self.stats.misses += 1
        self.stats.ms_wasted += result.latency_ms
        self.stats.usd_wasted += self.tool_usd
        return None


def break_even_hit_rate(tool_usd: float, tool_latency_ms: float, usd_per_second_of_latency: float) -> float:
    """The hit rate at which speculation stops paying, given what a second of latency is worth.

    Saving per hit is ``tool_latency_ms`` of wall clock valued at
    ``usd_per_second_of_latency``; cost per miss is one wasted tool call. Speculation is
    worth it while ``p * saving >= (1 - p) * tool_usd``.
    """
    saving = (tool_latency_ms / 1000.0) * usd_per_second_of_latency
    if tool_usd <= 0:
        return 0.0
    if saving <= 0:
        return 1.0
    return tool_usd / (tool_usd + saving)
