from __future__ import annotations

import json

import pytest

from arouter.bench import (
    BenchOptions,
    BenchReport,
    ConfigResult,
    SimulatedFactory,
    default_configs,
    fit_predictor_from,
    pareto_front,
    percentile,
    run_config,
)
from arouter.cache import LocalCacheBackend
from arouter.cascade import RunConfig
from arouter.pricing import LADDER
from arouter.report import render
from arouter.suite import Suite


def test_percentile_uses_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0
    assert percentile([], 0.5) == 0.0


def test_percentile_is_order_independent() -> None:
    assert percentile([5.0, 1.0, 3.0], 0.5) == percentile([1.0, 3.0, 5.0], 0.5)


def _result(suite: Suite, config: RunConfig) -> ConfigResult:
    return run_config(
        suite, config, BenchOptions(seed=0, cache_backend=LocalCacheBackend(), factory=SimulatedFactory(seed=0))
    )


def test_the_headline_metric_is_spend_over_tasks_actually_solved(suite: Suite) -> None:
    result = _result(suite, RunConfig(name="strong", tiers=("strong",), policy="none"))
    assert result.solved > 0
    assert result.usd_per_solved_task == pytest.approx(result.usd_total / result.solved)
    assert result.usd_per_solved_task > result.usd_per_task


def test_solving_nothing_gives_an_infinite_cost_per_solved_task(suite: Suite) -> None:
    result = ConfigResult(
        config=RunConfig(name="x"),
        runs=[],
        cache=type(_result(suite, RunConfig(name="y", tiers=("cheap",), policy="none")).cache)(),
        prefetch=type(_result(suite, RunConfig(name="z", tiers=("cheap",), policy="none")).prefetch)(),
        policy=type(_result(suite, RunConfig(name="w", tiers=("cheap",), policy="none")).policy)(),
    )
    assert result.usd_per_solved_task == float("inf")
    assert result.to_dict()["usd_per_solved_task"] is None


def test_a_stronger_tier_costs_more_and_solves_more(suite: Suite) -> None:
    cheap = _result(suite, RunConfig(name="cheap", tiers=("cheap",), policy="none"))
    strong = _result(suite, RunConfig(name="strong", tiers=("strong",), policy="none"))
    assert strong.solved > cheap.solved
    assert strong.usd_total > cheap.usd_total


def test_bands_are_reported_separately(suite: Suite) -> None:
    result = _result(suite, RunConfig(name="cheap", tiers=("cheap",), policy="none"))
    bands = result.solved_by_band()
    assert set(bands) == {"easy", "medium", "hard"}
    assert sum(attempted for _, attempted in bands.values()) == result.n


def test_a_dominated_configuration_is_off_the_front(suite: Suite) -> None:
    cheap = _result(suite, RunConfig(name="cheap", tiers=("cheap",), policy="none"))
    strong = _result(suite, RunConfig(name="strong", tiers=("strong",), policy="none"))
    hot = _result(suite, RunConfig(name="hot", tiers=LADDER[:1] + LADDER[-1:], policy="always_strong"))
    front = pareto_front([cheap, strong, hot])
    # always_strong pays for a cheap call it always discards, so it cannot dominate.
    assert "hot" not in front or hot.success_rate > strong.success_rate
    assert "cheap" in front


def test_the_front_is_never_empty(suite: Suite) -> None:
    results = [_result(suite, RunConfig(name=f"single_{t}", tiers=(t,), policy="none")) for t in ("cheap", "strong")]
    assert pareto_front(results)


def test_default_configs_cover_every_policy_and_every_tier() -> None:
    configs = default_configs(learned_weights={"bias": 0.0})
    names = {c.name for c in configs}
    assert {"single_cheap", "single_mid", "single_mid2", "single_strong"} <= names
    for policy in ("schema_validation", "self_confidence", "verifier_model", "learned_threshold", "oracle"):
        assert f"cascade_{policy}" in names


def test_the_predictor_can_be_fitted_from_other_configs_runs(suite: Suite) -> None:
    a = _result(suite, RunConfig(name="a", tiers=("cheap",), policy="none"))
    b = _result(suite, RunConfig(name="b", tiers=("strong",), policy="none"))
    predictor = fit_predictor_from([a, b], exclude_config="b")
    assert predictor.by_task
    assert predictor.predict(suite.tasks[0].id, suite.tasks[0].band, []) is not None


def test_the_report_dict_round_trips_through_json(suite: Suite) -> None:
    results = [_result(suite, RunConfig(name=f"single_{t}", tiers=(t,), policy="none")) for t in ("cheap", "strong")]
    report = BenchReport(backend="simulated", suite_path="benchmarks/suite.yaml", results=results)
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert payload["baseline"] == "single_strong"
    assert len(payload["configs"]) == 2
    assert payload["price_table_date"]


def test_the_report_states_the_ratio_against_the_baseline(suite: Suite) -> None:
    results = [_result(suite, RunConfig(name=f"single_{t}", tiers=(t,), policy="none")) for t in ("cheap", "strong")]
    report = BenchReport(backend="simulated", suite_path="s", results=results).to_dict()
    cheap = next(c for c in report["configs"] if c["config"]["name"] == "single_cheap")
    assert cheap["usd_per_solved_vs_baseline"] < 1.0
    assert cheap["success_delta_vs_baseline"] < 0.0


def test_the_html_report_is_self_contained(suite: Suite, tmp_path) -> None:
    results = [_result(suite, RunConfig(name=f"single_{t}", tiers=(t,), policy="none")) for t in ("cheap", "strong")]
    html = render(BenchReport(backend="simulated", suite_path="s", results=results).to_dict())
    assert html.startswith("<!doctype html>")
    for forbidden in ("http://", "https://", "<script", "src="):
        assert forbidden not in html
    assert "<svg" in html and "single_strong" in html


def test_the_html_report_survives_a_config_that_solved_nothing(suite: Suite) -> None:
    result = _result(suite, RunConfig(name="tiny", tiers=("cheap",), policy="none", max_steps=1))
    html = render(BenchReport(backend="simulated", suite_path="s", results=[result]).to_dict())
    assert "<svg" in html
