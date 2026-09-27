"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .bench import (
    BenchOptions,
    BenchReport,
    ConfigResult,
    SimulatedFactory,
    default_configs,
    fit_predictor_from,
    merge_reports,
    percentile,
    run_config,
)
from .budget import Ledger
from .cache import LocalCacheBackend, PrefixCache
from .cascade import CascadeRunner, RunConfig
from .events import TrajectoryEvent, write_jsonl
from .learned import feature_importance, fit as fit_learned, load_shadow_rows
from .models.replay import FixtureIndex, ReplayFactory, ReplayVerifier
from .models.simulated import SimulatedModel
from .models.verifier import LiveVerifier, SimulatedVerifier
from .prefetch import Prefetcher, ToolPredictor, break_even_hit_rate
from .pricing import LADDER, PRICE_TABLE_DATE, TIERS
from .report import write_report
from .suite import load_suite


def _fmt(value: float) -> str:
    return "n/a" if value == float("inf") else f"${value:.5f}"


@click.group()
@click.version_option(package_name="agent-router")
def main() -> None:
    """Cost-aware execution for agents: cascades, cache-aware layout, prefetch, budgets."""


@main.command()
def tiers() -> None:
    """Print the capability ladder and the prices every dollar figure derives from."""
    click.echo(f"prices as published {PRICE_TABLE_DATE}, USD per 1M tokens, Vertex AI\n")
    click.echo(f"{'tier':8} {'model':26} {'in':>8} {'cached':>8} {'out':>8} {'sim skill':>10}")
    for name, t in sorted(TIERS.items(), key=lambda kv: kv[1].usd_per_mtok_in):
        click.echo(
            f"{name:8} {t.model:26} {t.usd_per_mtok_in:8.3f} {t.usd_per_mtok_cached_in:8.3f} "
            f"{t.usd_per_mtok_out:8.3f} {t.sim_skill:10.2f}"
        )
    click.echo(f"\ncascade ladder: {' -> '.join(LADDER)}")


def _factory_and_cache(backend: str, fixtures: str | None, seed: int, sleep: bool) -> tuple[Any, Any, Any]:
    """Return (model factory, cache backend, verifier factory) for a backend name."""
    if backend == "simulated":
        return (
            SimulatedFactory(seed=seed, sleep=sleep),
            LocalCacheBackend(),
            lambda tier_name: SimulatedVerifier(tier=tier_name, seed=seed),
        )
    if backend == "vertex":
        from .models.vertex import VertexFactory

        vf = VertexFactory()
        return vf, vf.cache_backend(), lambda tier_name: LiveVerifier(client=vf.make(tier_name), tier=tier_name)
    if backend == "replay":
        if not fixtures:
            raise click.ClickException("--fixtures is required with --backend replay")
        # A real factory is built per configuration inside the bench loop; this only
        # validates that the directory holds anything at all.
        if not len(FixtureIndex.load(fixtures)):
            raise click.ClickException(f"no recorded llm calls under {fixtures}")
        return None, None, None
    raise click.ClickException(f"unknown backend {backend}")


@main.command()
@click.option("--suite", "suite_path", type=click.Path(exists=True), default=None)
@click.option("--backend", type=click.Choice(["simulated", "vertex", "replay"]), default="simulated")
@click.option("--fixtures", type=click.Path(), default=None, help="Trajectory directory for --backend replay.")
@click.option("--out", type=click.Path(), default="out/bench.json", show_default=True)
@click.option("--html", "html_out", type=click.Path(), default=None, help="Also write the HTML frontier here.")
@click.option("--trajectories", type=click.Path(), default=None, help="Record every run as JSONL here.")
@click.option("--learned-fit", type=click.Path(exists=True), default=None, help="Fit produced by `arouter fit`.")
@click.option("--configs", "only", default=None, help="Comma-separated config names to run.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--tool-latency-ms", type=float, default=0.0, show_default=True)
@click.option("--max-steps", type=int, default=10, show_default=True)
@click.option("--confidence-threshold", type=float, default=0.75, show_default=True)
@click.option("--prefetch/--no-prefetch", default=False, show_default=True)
@click.option("--cache/--no-cache", "cache_enabled", default=True, show_default=True)
@click.option("--cache-ttl", type=int, default=900, show_default=True)
@click.option("--sleep/--no-sleep", default=False, help="Make the simulated model actually wait its latency.")
@click.option("--tenant-cap-usd", type=float, default=None)
def bench(
    suite_path: str | None,
    backend: str,
    fixtures: str | None,
    out: str,
    html_out: str | None,
    trajectories: str | None,
    learned_fit: str | None,
    only: str | None,
    seed: int,
    tool_latency_ms: float,
    max_steps: int,
    confidence_threshold: float,
    prefetch: bool,
    cache_enabled: bool,
    cache_ttl: int,
    sleep: bool,
    tenant_cap_usd: float | None,
) -> None:
    """Run every configuration over the suite and write the frontier."""
    suite = load_suite(suite_path)
    factory, cache_backend, verifier_factory = _factory_and_cache(backend, fixtures, seed, sleep)

    weights: dict[str, float] = {}
    threshold = 0.5
    if learned_fit:
        fit_data = json.loads(Path(learned_fit).read_text())
        weights = {k: float(v) for k, v in fit_data["weights"].items()}
        threshold = float(fit_data["threshold"])

    configs = default_configs(
        learned_weights=weights,
        learned_threshold=threshold,
        confidence_threshold=confidence_threshold,
        tool_latency_ms=tool_latency_ms,
        max_steps=max_steps,
    )
    for config in configs:
        config.cache_enabled = cache_enabled
        config.prefetch_enabled = prefetch
    if only:
        wanted = {name.strip() for name in only.split(",")}
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            raise click.ClickException(f"no configs matched {sorted(wanted)}")
    if not weights:
        configs = [c for c in configs if c.policy != "learned_threshold"]
        click.echo("no --learned-fit given, skipping the learned_threshold config", err=True)

    results: list[ConfigResult] = []
    predictor: ToolPredictor | None = None
    for config in configs:
        if backend == "replay":
            assert fixtures is not None
            index = FixtureIndex.load(fixtures, config=config.name)
            if not len(index):
                click.echo(f"{config.name:30} SKIPPED: no recorded calls for it under {fixtures}", err=True)
                continue
            factory = ReplayFactory(index=index)
            verifier_factory = (
                lambda tier_name, _i=index, _c=config.name: ReplayVerifier(
                    tier=tier_name, run_id=f"{_c}::", index=_i
                )
            )
        options = BenchOptions(
            seed=seed,
            trajectory_dir=Path(trajectories) if trajectories else None,
            tenant_cap_usd=tenant_cap_usd,
            cache_ttl_seconds=cache_ttl,
            cache_backend=cache_backend,
            sleep=sleep,
            predictor=predictor,
            verifier_factory=verifier_factory,
            factory=factory,
        )
        try:
            result = run_config(suite, config, options)
        except Exception as exc:  # noqa: BLE001 - one unreachable tier must not lose the whole frontier
            click.echo(f"{config.name:30} FAILED: {type(exc).__name__}: {exc}", err=True)
            continue
        results.append(result)
        if predictor is None and prefetch:
            predictor = fit_predictor_from(results)
        click.echo(
            f"{config.name:30} solved {result.solved:2}/{result.n:2}  "
            f"{_fmt(result.usd_per_task)}/task  {_fmt(result.usd_per_solved_task)}/solved  "
            f"p50 {percentile(result.latencies, 0.5):7.0f}ms  p95 {percentile(result.latencies, 0.95):7.0f}ms"
        )

    if not results:
        raise click.ClickException("every configuration failed; nothing to report")

    report = BenchReport(
        backend=backend,
        suite_path=str(suite_path or "benchmarks/suite.yaml"),
        results=results,
        notes={
            "seed": seed,
            "tool_latency_ms": tool_latency_ms,
            "prefetch": prefetch,
            "cache_enabled": cache_enabled,
            "cache_ttl_seconds": cache_ttl,
            "configs_requested": len(configs),
            "configs_completed": len(results),
        },
    )
    report.write(out)
    click.echo(f"\nwrote {out}")
    if html_out:
        write_report(report.to_dict(), html_out)
        click.echo(f"wrote {html_out}")


@main.command()
@click.option("--suite", "suite_path", type=click.Path(exists=True), default=None)
@click.option("--backend", type=click.Choice(["simulated", "vertex"]), default="simulated")
@click.option("--out", type=click.Path(), default="out/shadow", show_default=True)
@click.option("--seeds", default="0,1,2", show_default=True, help="Comma-separated seeds to collect.")
@click.option("--max-steps", type=int, default=10, show_default=True)
def shadow(suite_path: str | None, backend: str, out: str, seeds: str, max_steps: int) -> None:
    """Collect labelled training data by calling both tiers on every step."""
    suite = load_suite(suite_path)
    factory, cache_backend, verifier_factory = _factory_and_cache(backend, None, 0, False)
    total_rows = 0
    for raw_seed in seeds.split(","):
        seed = int(raw_seed.strip())
        config = RunConfig(
            name=f"shadow_seed{seed}",
            tiers=LADDER[:1] + LADDER[-1:],
            policy="always_strong",
            shadow=True,
            max_steps=max_steps,
        )
        options = BenchOptions(
            seed=seed,
            trajectory_dir=Path(out),
            cache_backend=cache_backend,
            verifier_factory=verifier_factory,
            factory=factory if backend != "simulated" else SimulatedFactory(seed=seed),
        )
        result = run_config(suite, config, options)
        labelled = sum(
            1
            for run in result.runs
            for event in run.events
            if event["type"] == "decision" and event["payload"].get("kind") == "shadow_label"
        )
        total_rows += labelled
        click.echo(f"seed {seed}: {result.n} runs, {labelled} labelled steps, {_fmt(result.usd_total)} spent")
    click.echo(f"\nwrote {total_rows} labelled steps under {out}")


@main.command()
@click.argument("shadow_dir", type=click.Path(exists=True))
@click.option("--out", type=click.Path(), default="out/fit.json", show_default=True)
@click.option("--holdout-fraction", type=float, default=0.3, show_default=True)
def fit(shadow_dir: str, out: str, holdout_fraction: float) -> None:
    """Fit the learned escalation policy on recorded shadow runs."""
    rows = load_shadow_rows(shadow_dir)
    if not rows:
        raise click.ClickException(f"no shadow_label events under {shadow_dir}; run `arouter shadow` first")
    result = fit_learned(rows, holdout_fraction=holdout_fraction)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    click.echo(f"{len(rows)} labelled steps, base rate cheap-was-wrong {result.base_rate:.1%}")
    click.echo(f"train n={result.n_train}  holdout n={result.n_holdout}  threshold {result.threshold:.2f}")
    for split, metrics in (("train", result.train_metrics), ("holdout", result.holdout_metrics)):
        if metrics["n"]:
            click.echo(
                f"  {split:8} acc {metrics['accuracy']:.3f}  auc {metrics['auc']:.3f}  "
                f"recall-on-wrong {metrics['recall_on_wrong']:.3f}  "
                f"false-escalation {metrics['false_escalation_rate']:.3f}"
            )
    click.echo("\nfeature importance (weight x sd):")
    for name, value in feature_importance(result.weights, rows):
        click.echo(f"  {name:32} {value:+.4f}")
    click.echo(f"\nwrote {out}")


@main.command("cache-study")
@click.option("--suite", "suite_path", type=click.Path(exists=True), default=None)
@click.option("--backend", type=click.Choice(["simulated", "vertex"]), default="simulated")
@click.option("--out", type=click.Path(), default="out/cache-study.json", show_default=True)
@click.option("--trajectories", type=click.Path(), default=None)
@click.option("--cache-ttl", type=int, default=900, show_default=True)
@click.option("--tier", default="cheap", show_default=True)
def cache_study(
    suite_path: str | None, backend: str, out: str, trajectories: str | None, cache_ttl: int, tier: str
) -> None:
    """Measure what the cache-aware layout wins and what reordering context costs."""
    suite = load_suite(suite_path)
    factory, cache_backend, verifier_factory = _factory_and_cache(backend, None, 0, False)
    rows: list[dict[str, Any]] = []
    for layout, cache_on in (("cache_aware", True), ("naive", True), ("cache_aware", False)):
        name = f"{layout}_{'cached' if cache_on else 'uncached'}"
        config = RunConfig(name=name, tiers=(tier,), policy="none", layout=layout, cache_enabled=cache_on)
        options = BenchOptions(
            trajectory_dir=Path(trajectories) if trajectories else None,
            cache_backend=cache_backend if cache_on else None,
            cache_ttl_seconds=cache_ttl,
            verifier_factory=verifier_factory,
            factory=factory,
        )
        result = run_config(suite, config, options)
        stats = result.cache.to_dict()
        rows.append(
            {
                "name": name,
                "layout": layout,
                "cache_enabled": cache_on,
                "usd_total": result.usd_total,
                "solved": result.solved,
                "n": result.n,
                "tokens_in": sum(r.spend.tokens_in for r in result.runs),
                "tokens_cached_in": sum(r.spend.tokens_cached_in for r in result.runs),
                **stats,
            }
        )
        click.echo(f"{name:24} {result.cache.summary_line()}  spend {_fmt(result.usd_total)}")
    baseline = next(r for r in rows if r["name"] == "cache_aware_uncached")
    for row in rows:
        row["usd_vs_uncached"] = row["usd_total"] - baseline["usd_total"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2, sort_keys=True))
    click.echo(f"\nwrote {out}")


@main.command("prefetch-study")
@click.option("--suite", "suite_path", type=click.Path(exists=True), default=None)
@click.option("--out", type=click.Path(), default="out/prefetch-study.json", show_default=True)
@click.option("--latencies", default="40,120,400,1200", show_default=True, help="Tool latencies in ms to sweep.")
@click.option("--thresholds", default="0.3,0.5,0.7,0.9", show_default=True)
@click.option("--tool-usd", type=float, default=0.0002, show_default=True, help="Cost of one tool call.")
@click.option("--usd-per-second", type=float, default=0.002, show_default=True, help="What a second of latency is worth.")
@click.option("--tier", default="cheap", show_default=True)
def prefetch_study(
    suite_path: str | None,
    out: str,
    latencies: str,
    thresholds: str,
    tool_usd: float,
    usd_per_second: float,
    tier: str,
) -> None:
    """Sweep tool latency and prediction threshold to find where speculation stops paying."""
    suite = load_suite(suite_path)
    warmup = RunConfig(name="warmup", tiers=(tier,), policy="none")
    warm = run_config(suite, warmup, BenchOptions(factory=SimulatedFactory(seed=0)))
    predictor = fit_predictor_from([warm])

    rows: list[dict[str, Any]] = []
    for raw_latency in latencies.split(","):
        latency = float(raw_latency)
        for raw_threshold in thresholds.split(","):
            threshold = float(raw_threshold)
            prefetcher = Prefetcher(
                predictor=predictor, threshold=threshold, tool_latency_ms=latency, tool_usd=tool_usd
            )
            config = RunConfig(
                name=f"pf_l{latency:.0f}_t{threshold:.2f}",
                tiers=(tier,),
                policy="none",
                prefetch_enabled=True,
                tool_latency_ms=latency,
                tool_usd=tool_usd,
            )
            with prefetcher:
                runner = CascadeRunner(
                    suite,
                    SimulatedFactory(seed=0),
                    prefix_cache=PrefixCache(enabled=False),
                    prefetcher=prefetcher,
                )
                runs = [runner.run(task, config, seed=0) for task in suite.tasks]
            stats = prefetcher.stats
            break_even = break_even_hit_rate(tool_usd, latency, usd_per_second)
            value_of_saving = (stats.ms_saved / 1000.0) * usd_per_second
            rows.append(
                {
                    "tool_latency_ms": latency,
                    "threshold": threshold,
                    "launched": stats.launched,
                    "hits": stats.hits,
                    "hit_rate": round(stats.hit_rate, 4),
                    "coverage": round(stats.coverage, 4),
                    "ms_saved": round(stats.ms_saved, 1),
                    "ms_wasted": round(stats.ms_wasted, 1),
                    "usd_wasted": round(stats.usd_wasted, 6),
                    "usd_value_of_saving": round(value_of_saving, 6),
                    "net_usd": round(value_of_saving - stats.usd_wasted, 6),
                    "break_even_hit_rate": round(break_even, 4),
                    "pays_off": stats.hit_rate >= break_even,
                    "solved": sum(1 for r in runs if r.solved),
                }
            )
            click.echo(
                f"tool {latency:6.0f}ms  thr {threshold:.2f}  launched {stats.launched:3}  "
                f"hit {stats.hit_rate:6.1%}  saved {stats.ms_saved:8.0f}ms  wasted {stats.ms_wasted:8.0f}ms  "
                f"net {value_of_saving - stats.usd_wasted:+.6f} USD  break-even {break_even:.1%}"
            )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2, sort_keys=True))
    click.echo(f"\nwrote {out}")


@main.command("budget-demo")
@click.option("--suite", "suite_path", type=click.Path(exists=True), default=None)
@click.option("--run-caps", default="1.0,0.0012,0.0006,0.0002", show_default=True)
@click.option("--tenant-cap-usd", type=float, default=0.004, show_default=True)
@click.option("--out", type=click.Path(), default="out/budget-demo.json", show_default=True)
def budget_demo(suite_path: str | None, run_caps: str, tenant_cap_usd: float, out: str) -> None:
    """Show the degradation ladder under tightening per-run caps and one shared tenant cap."""
    suite = load_suite(suite_path)
    rows: list[dict[str, Any]] = []
    for raw_cap in run_caps.split(","):
        cap = float(raw_cap)
        config = RunConfig(
            name=f"cap_{cap}",
            tiers=LADDER[:1] + LADDER[-1:],
            policy="schema_validation",
            run_cap_usd=cap,
        )
        result = run_config(suite, config, BenchOptions(seed=0, factory=SimulatedFactory(seed=0)))
        levels: dict[str, int] = {}
        for run in result.runs:
            for transition in run.degradations:
                levels[str(transition["level"])] = levels.get(str(transition["level"]), 0) + 1
        rows.append(
            {
                "run_cap_usd": cap,
                "solved": result.solved,
                "n": result.n,
                "usd_total": result.usd_total,
                "escalations": sum(r.escalations for r in result.runs),
                "runs_stopped_by_budget": sum(1 for r in result.runs if r.stopped_by_budget),
                "runs_over_cap": sum(1 for r in result.runs if r.usd > cap),
                "degradations": levels,
            }
        )
        click.echo(
            f"run cap {cap:<9.5f} solved {result.solved:2}/{result.n:2}  spend {_fmt(result.usd_total)}  "
            f"escalations {sum(r.escalations for r in result.runs):3}  "
            f"stopped {sum(1 for r in result.runs if r.stopped_by_budget):2}  "
            f"over cap {sum(1 for r in result.runs if r.usd > cap)}  {levels or 'no degradation'}"
        )

    ledger = Ledger(tenant="acme", cap_usd=tenant_cap_usd)
    config = RunConfig(name="tenant", tiers=LADDER[:1] + LADDER[-1:], policy="schema_validation")
    runner = CascadeRunner(
        suite, SimulatedFactory(seed=0), prefix_cache=PrefixCache(enabled=False), tenant_ledger=ledger,
        verifier_factory=lambda t: SimulatedVerifier(tier=t),
    )
    tenant_rows = []
    for task in suite.tasks:
        run = runner.run(task, config, seed=0)
        tenant_rows.append(
            {
                "task": task.id,
                "solved": run.solved,
                "usd": run.usd,
                "tenant_spent_usd": ledger.spent_usd,
                "level": run.degradations[-1]["level"] if run.degradations else "NONE",
                "stopped": run.stopped_by_budget,
            }
        )
    click.echo(f"\nshared tenant cap {_fmt(tenant_cap_usd)}: spend stopped at {_fmt(ledger.spent_usd)} "
               f"over {len(tenant_rows)} runs; {sum(1 for r in tenant_rows if r['stopped'])} runs degraded to STOP")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"per_run_caps": rows, "shared_tenant": tenant_rows}, indent=2, sort_keys=True))
    click.echo(f"wrote {out}")


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--out", type=click.Path(), default="out/bench.json", show_default=True)
@click.option("--html", "html_out", type=click.Path(), default=None)
def merge(inputs: tuple[str, ...], out: str, html_out: str | None) -> None:
    """Combine bench results produced by separate runs into one frontier."""
    payloads = [json.loads(Path(p).read_text()) for p in inputs]
    merged = merge_reports(payloads)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(merged, indent=2, sort_keys=True, default=str))
    base = next((c for c in merged["configs"] if c["config"]["name"] == merged["baseline"]), None)
    for c in merged["configs"]:
        ratio = c.get("usd_per_solved_vs_baseline")
        click.echo(
            f"{c['config']['name']:30} solved {c['solved']:2}/{c['n_tasks']:2}  "
            f"{_fmt(c['usd_per_task'])}/task  {_fmt(c['usd_per_solved_task'] or float('inf'))}/solved  "
            f"{(f'{ratio:.2f}x' if ratio else '   n/a'):>7}  "
            f"p50 {c['p50_ms']:7.0f}ms  p95 {c['p95_ms']:7.0f}ms"
        )
    if base is None:
        click.echo(f"\nbaseline {merged['baseline']} is absent, so no ratios were computed", err=True)
    click.echo(f"\npareto front: {', '.join(merged['pareto_front'])}")
    click.echo(f"wrote {out}")
    if html_out:
        write_report(merged, html_out)
        click.echo(f"wrote {html_out}")


@main.command()
@click.argument("results", type=click.Path(exists=True))
@click.option("--out", type=click.Path(), default="out/report.html", show_default=True)
def report(results: str, out: str) -> None:
    """Regenerate the HTML frontier from a bench results file."""
    write_report(json.loads(Path(results).read_text()), out)
    click.echo(f"wrote {out}")


@main.command()
@click.argument("fixtures", type=click.Path(exists=True))
def verify(fixtures: str) -> None:
    """Check that committed fixtures parse and account for what they recorded."""
    index = FixtureIndex.load(fixtures)
    if not len(index):
        raise click.ClickException(f"no llm calls found under {fixtures}")
    usd = 0.0
    tokens_in = tokens_cached = tokens_out = 0
    for queue in index.calls.values():
        for payload in queue:
            usage = payload["usage"]
            usd += float(usage["usd"])
            tokens_in += int(usage["tokens_in"])
            tokens_cached += int(usage["tokens_cached_in"])
            tokens_out += int(usage["tokens_out"]) + int(usage.get("tokens_thought", 0))
    click.echo(f"{len(index.run_ids)} runs, {len(index)} recorded calls")
    click.echo(f"tokens in {tokens_in:,} (cached {tokens_cached:,}), out {tokens_out:,}")
    click.echo(f"recorded spend {_fmt(usd)}")


if __name__ == "__main__":
    main()
