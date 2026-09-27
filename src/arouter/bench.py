"""Running configurations over the suite, and reducing the runs to the metrics that matter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .budget import Ledger
from .cache import CacheStats, PrefixCache
from .cascade import CascadeRunner, RunConfig, TaskRun
from .events import TrajectoryEvent, write_jsonl
from .models.simulated import SimulatedModel
from .models.verifier import SimulatedVerifier
from .policies import PolicyStats
from .prefetch import Prefetcher, PrefetchStats, ToolPredictor
from .pricing import BASELINE_TIER, LADDER, PRICE_TABLE_DATE, Spend, TIERS, VERIFIER_TIER
from .suite import Suite, load_suite


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. With a handful of runs per config, interpolation would invent precision."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[rank]


@dataclass
class ConfigResult:
    """Everything measured for one configuration across the whole suite."""

    config: RunConfig
    runs: list[TaskRun]
    cache: CacheStats
    prefetch: PrefetchStats
    policy: PolicyStats

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.runs if r.solved)

    @property
    def success_rate(self) -> float:
        return self.solved / self.n if self.n else 0.0

    @property
    def usd_total(self) -> float:
        return sum(r.usd for r in self.runs)

    @property
    def usd_per_task(self) -> float:
        return self.usd_total / self.n if self.n else 0.0

    @property
    def usd_per_solved_task(self) -> float:
        """The headline. Infinite when nothing was solved, which is the correct answer."""
        return self.usd_total / self.solved if self.solved else float("inf")

    @property
    def latencies(self) -> list[float]:
        return [r.critical_path_ms for r in self.runs]

    def solved_by_band(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for run in self.runs:
            bucket = out.setdefault(run.band, [0, 0])
            bucket[0] += int(run.solved)
            bucket[1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_tasks": self.n,
            "solved": self.solved,
            "success_rate": round(self.success_rate, 4),
            "usd_total": self.usd_total,
            "usd_per_task": self.usd_per_task,
            "usd_per_solved_task": self.usd_per_solved_task if self.solved else None,
            "p50_ms": round(percentile(self.latencies, 0.50), 1),
            "p95_ms": round(percentile(self.latencies, 0.95), 1),
            "mean_ms": round(sum(self.latencies) / self.n, 1) if self.n else 0.0,
            "tokens_in": sum(r.spend.tokens_in for r in self.runs),
            "tokens_cached_in": sum(r.spend.tokens_cached_in for r in self.runs),
            "tokens_out": sum(r.spend.tokens_out for r in self.runs),
            "calls": sum(r.spend.calls for r in self.runs),
            "escalations": sum(r.escalations for r in self.runs),
            "usd_by_tier": _merge_tier_spend(self.runs),
            "solved_by_band": {k: list(v) for k, v in self.solved_by_band().items()},
            "cache": self.cache.to_dict(),
            "prefetch": self.prefetch.to_dict(),
            "policy": self.policy.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
        }


def _merge_tier_spend(runs: Iterable[TaskRun]) -> dict[str, float]:
    total: dict[str, float] = {}
    for run in runs:
        for name, usd in run.spend.by_tier.items():
            total[name] = total.get(name, 0.0) + usd
    return total


@dataclass
class SimulatedFactory:
    seed: int = 0
    sleep: bool = False

    def make(self, tier: str) -> SimulatedModel:
        return SimulatedModel(tier=tier, seed=self.seed, sleep=self.sleep)


@dataclass
class BenchOptions:
    seed: int = 0
    trajectory_dir: Path | None = None
    tenant_cap_usd: float | None = None
    tenant: str = "default"
    cache_ttl_seconds: int = 900
    cache_backend: Any = None
    sleep: bool = False
    predictor: ToolPredictor | None = None
    verifier_factory: Callable[[str, str], Any] | None = None
    factory: Any = None


def run_config(suite: Suite, config: RunConfig, options: BenchOptions) -> ConfigResult:
    """Run every task in the suite once under one configuration."""
    factory = options.factory or SimulatedFactory(seed=options.seed, sleep=options.sleep)
    verifier_factory = options.verifier_factory or (
        lambda tier_name, run_id: SimulatedVerifier(tier=tier_name, seed=options.seed)
    )
    prefix_cache = PrefixCache(
        backend=options.cache_backend,
        ttl_seconds=options.cache_ttl_seconds,
        enabled=config.cache_enabled,
    )
    ledger = Ledger(tenant=options.tenant, cap_usd=options.tenant_cap_usd) if options.tenant_cap_usd else None
    predictor = options.predictor or ToolPredictor()
    prefetcher = Prefetcher(
        predictor=predictor,
        tool_latency_ms=config.tool_latency_ms,
        tool_usd=config.tool_usd,
        enabled=config.prefetch_enabled,
    )

    runs: list[TaskRun] = []
    policy_stats = PolicyStats()
    with prefetcher:
        runner = CascadeRunner(
            suite,
            factory,
            prefix_cache=prefix_cache,
            prefetcher=prefetcher,
            verifier_factory=verifier_factory,
            tenant_ledger=ledger,
        )
        for task in suite.tasks:
            run = runner.run(task, config, seed=options.seed)
            runs.append(run)
            policy_stats.merge(run.policy_stats)
            if options.trajectory_dir:
                events = [TrajectoryEvent.from_dict(e) for e in run.events]
                path = Path(options.trajectory_dir) / config.name / f"{task.id}.jsonl"
                write_jsonl(path, events)
    prefix_cache.close()

    return ConfigResult(
        config=config,
        runs=runs,
        cache=prefix_cache.stats,
        prefetch=prefetcher.stats,
        policy=policy_stats,
    )


def fit_predictor_from(results: Iterable[ConfigResult], exclude_config: str | None = None) -> ToolPredictor:
    """Build the prefetch predictor from runs other than the one it will serve."""
    predictor = ToolPredictor()
    for result in results:
        if exclude_config and result.config.name == exclude_config:
            continue
        for run in result.runs:
            sequence = [(s.action["tool"], s.action["args"]) for s in run.steps]
            predictor.observe(run.task_id, run.band, sequence)
    return predictor


def default_configs(
    *,
    learned_weights: dict[str, float] | None = None,
    learned_threshold: float = 0.5,
    confidence_threshold: float = 0.75,
    tool_latency_ms: float = 0.0,
    max_steps: int = 10,
) -> list[RunConfig]:
    """The frontier's points: three single-model baselines, then one per escalation policy.

    The single-model rows are not padding. A cascade has to beat the best single model on
    dollars per solved task, not just beat the strongest one, and a routing result that
    omits the middle tier is hiding the comparison that matters.
    """
    configs = [
        RunConfig(
            name=f"single_{t}",
            tiers=(t,),
            policy="none",
            tool_latency_ms=tool_latency_ms,
            max_steps=max_steps,
        )
        for t in ("cheap", "mid", "mid2", "strong")
    ]
    for policy in ("schema_validation", "self_confidence", "verifier_model", "learned_threshold", "oracle"):
        configs.append(
            RunConfig(
                name=f"cascade_{policy}",
                tiers=LADDER[:1] + LADDER[-1:],
                policy=policy,
                confidence_threshold=confidence_threshold,
                learned_threshold=learned_threshold,
                learned_weights=learned_weights or {},
                tool_latency_ms=tool_latency_ms,
                max_steps=max_steps,
            )
        )
    return configs


def pareto_front(results: list[ConfigResult]) -> list[str]:
    """Config names that nothing else beats on both cost per task and success rate."""
    front: list[str] = []
    for a in results:
        dominated = any(
            b is not a
            and b.success_rate >= a.success_rate
            and b.usd_per_task <= a.usd_per_task
            and (b.success_rate > a.success_rate or b.usd_per_task < a.usd_per_task)
            for b in results
        )
        if not dominated:
            front.append(a.config.name)
    return front


@dataclass
class BenchReport:
    """The whole measurement, in the form the HTML report and MEASUREMENTS.md read from."""

    backend: str
    suite_path: str
    results: list[ConfigResult]
    baseline: str = f"single_{BASELINE_TIER}"
    notes: dict[str, Any] = field(default_factory=dict)

    def by_name(self, name: str) -> ConfigResult | None:
        return next((r for r in self.results if r.config.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        base = self.by_name(self.baseline)
        rows = []
        for result in self.results:
            row = result.to_dict()
            if base and base.solved and result.solved:
                row["usd_per_solved_vs_baseline"] = round(
                    result.usd_per_solved_task / base.usd_per_solved_task, 4
                )
                row["success_delta_vs_baseline"] = round(result.success_rate - base.success_rate, 4)
            rows.append(row)
        return {
            "backend": self.backend,
            "suite": self.suite_path,
            "baseline": self.baseline,
            "price_table_date": PRICE_TABLE_DATE,
            "tiers": {name: {"model": t.model, "usd_per_mtok_in": t.usd_per_mtok_in,
                             "usd_per_mtok_cached_in": t.usd_per_mtok_cached_in,
                             "usd_per_mtok_out": t.usd_per_mtok_out} for name, t in TIERS.items()},
            "verifier_tier": VERIFIER_TIER,
            "pareto_front": pareto_front(self.results),
            "notes": self.notes,
            "configs": rows,
        }

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str))


def load_bench_suite(path: str | Path | None) -> Suite:
    return load_suite(path)


def total_spend(results: Iterable[ConfigResult]) -> Spend:
    total = Spend()
    for result in results:
        for run in result.runs:
            total.merge(run.spend)
    return total


def pareto_front_of_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Pareto front over already-serialized config rows, for merging runs done separately."""
    front: list[str] = []
    for a in rows:
        dominated = any(
            b is not a
            and b["success_rate"] >= a["success_rate"]
            and b["usd_per_task"] <= a["usd_per_task"]
            and (b["success_rate"] > a["success_rate"] or b["usd_per_task"] < a["usd_per_task"])
            for b in rows
        )
        if not dominated:
            front.append(a["config"]["name"])
    return front


def merge_reports(payloads: list[dict[str, Any]], baseline: str = f"single_{BASELINE_TIER}") -> dict[str, Any]:
    """Combine reports produced by separate invocations into one frontier.

    Quota made one process per configuration the only way to finish, so the frontier is
    assembled from the rows that actually completed. A configuration that never ran is
    absent rather than filled in.
    """
    if not payloads:
        raise ValueError("nothing to merge")
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload["configs"]:
            rows[row["config"]["name"]] = row
    ordered = sorted(rows.values(), key=lambda r: r["usd_per_task"])
    base = rows.get(baseline)
    for row in ordered:
        row.pop("usd_per_solved_vs_baseline", None)
        row.pop("success_delta_vs_baseline", None)
        if base and base.get("usd_per_solved_task") and row.get("usd_per_solved_task"):
            row["usd_per_solved_vs_baseline"] = round(
                row["usd_per_solved_task"] / base["usd_per_solved_task"], 4
            )
            row["success_delta_vs_baseline"] = round(row["success_rate"] - base["success_rate"], 4)
    first = payloads[0]
    notes = {"merged_from": len(payloads), "configs_present": [r["config"]["name"] for r in ordered]}
    for payload in payloads:
        notes.update(payload.get("notes", {}))
    return {
        "backend": first["backend"],
        "suite": first["suite"],
        "baseline": baseline,
        "price_table_date": first["price_table_date"],
        "tiers": first["tiers"],
        "verifier_tier": first.get("verifier_tier", VERIFIER_TIER),
        "pareto_front": pareto_front_of_rows(ordered),
        "notes": notes,
        "configs": ordered,
    }
