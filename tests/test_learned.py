from __future__ import annotations

import random

import pytest

from arouter.learned import (
    Row,
    auc,
    choose_threshold,
    feature_importance,
    fit,
    fit_logistic,
    metrics_at,
    predict,
    split_rows,
)
from arouter.policies import FEATURE_NAMES


def _rows(n: int = 240, seed: int = 0) -> list[Row]:
    """Rows where schema_invalid and low confidence genuinely predict a wrong action."""
    rng = random.Random(seed)
    out: list[Row] = []
    for i in range(n):
        invalid = rng.random() < 0.3
        wrong = invalid or rng.random() < 0.25
        confidence = rng.uniform(0.2, 0.6) if wrong else rng.uniform(0.6, 0.95)
        out.append(
            Row(
                features={
                    "bias": 1.0,
                    "confidence": confidence,
                    "schema_invalid": 1.0 if invalid else 0.0,
                    "is_write": float(rng.random() < 0.4),
                    "is_finish": float(rng.random() < 0.2),
                    "step_frac": rng.random(),
                    "prompt_ktokens": rng.uniform(2.0, 3.0),
                    "empty_summary_when_required": 0.0,
                    "n_prior_writes": rng.random(),
                },
                label=int(wrong),
                run_id=f"r{i % 20}",
                task_id=f"task{i % 12}",
            )
        )
    return out


def test_the_split_shares_no_task_between_train_and_holdout() -> None:
    train, holdout = split_rows(_rows(), holdout_fraction=0.3)
    assert train and holdout
    assert not ({r.task_id for r in train} & {r.task_id for r in holdout})


def test_the_split_keeps_every_row() -> None:
    rows = _rows()
    train, holdout = split_rows(rows)
    assert len(train) + len(holdout) == len(rows)


def test_a_single_task_cannot_be_split() -> None:
    rows = [Row(features={"bias": 1.0}, label=0, run_id="r", task_id="only")]
    train, holdout = split_rows(rows)
    assert train == rows and holdout == []


def test_fitting_learns_the_signal_that_is_there() -> None:
    rows = _rows()
    weights = fit_logistic(rows)
    assert weights["schema_invalid"] > 0.5
    assert weights["confidence"] < 0.0
    assert auc(weights, rows) > 0.8


def test_fitting_beats_the_base_rate_on_held_out_tasks() -> None:
    rows = _rows(n=400)
    result = fit(rows)
    base_rate = max(result.base_rate, 1 - result.base_rate)
    assert result.holdout_metrics["accuracy"] > base_rate
    assert result.holdout_metrics["auc"] > 0.75


def test_a_fit_on_pure_noise_does_not_claim_skill() -> None:
    rng = random.Random(1)
    rows = [
        Row(
            features={name: (1.0 if name == "bias" else rng.random()) for name in FEATURE_NAMES},
            label=rng.randint(0, 1),
            run_id="r",
            task_id=f"task{i % 10}",
        )
        for i in range(300)
    ]
    result = fit(rows)
    assert 0.35 < result.holdout_metrics["auc"] < 0.65


def test_all_features_are_fitted_even_when_absent_from_a_row() -> None:
    weights = fit_logistic([Row(features={"bias": 1.0}, label=1, run_id="r", task_id="t")])
    assert set(weights) == set(FEATURE_NAMES)


def test_predictions_stay_in_range_under_extreme_weights() -> None:
    for weight in (-1e6, 1e6):
        p = predict({"bias": weight}, {"bias": 1.0})
        assert 0.0 <= p <= 1.0


def test_metrics_partition_every_row() -> None:
    rows = _rows()
    weights = fit_logistic(rows)
    m = metrics_at(weights, rows, 0.5)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(rows)


def test_a_higher_threshold_escalates_less() -> None:
    rows = _rows()
    weights = fit_logistic(rows)
    low = metrics_at(weights, rows, 0.2)
    high = metrics_at(weights, rows, 0.8)
    assert low["tp"] + low["fp"] >= high["tp"] + high["fp"]
    assert low["recall_on_wrong"] >= high["recall_on_wrong"]


def test_the_chosen_threshold_is_inside_the_swept_range() -> None:
    weights = fit_logistic(_rows())
    assert 0.05 <= choose_threshold(weights, _rows()) <= 0.95


def test_feature_importance_is_scaled_by_spread_and_excludes_the_bias() -> None:
    rows = _rows()
    ranking = feature_importance(fit_logistic(rows), rows)
    assert "bias" not in dict(ranking)
    assert ranking[0][0] in ("schema_invalid", "confidence")
    assert abs(ranking[0][1]) >= abs(ranking[-1][1])


def test_auc_is_a_half_when_one_class_is_missing() -> None:
    rows = [Row(features={"bias": 1.0}, label=1, run_id="r", task_id="t")]
    assert auc({"bias": 1.0}, rows) == 0.5


def test_fitting_nothing_returns_zero_weights() -> None:
    assert set(fit_logistic([]).values()) == {0.0}


def test_the_shadow_pipeline_produces_a_usable_fit(tmp_path, suite) -> None:
    """End to end: record shadow runs, load the labels, fit, and get a real holdout number."""
    from arouter.bench import BenchOptions, SimulatedFactory, run_config
    from arouter.cascade import RunConfig
    from arouter.learned import load_shadow_rows
    from arouter.pricing import LADDER

    for seed in (0, 1, 2):
        config = RunConfig(
            name=f"shadow{seed}", tiers=LADDER[:1] + LADDER[-1:], policy="always_strong", shadow=True
        )
        run_config(suite, config, BenchOptions(seed=seed, trajectory_dir=tmp_path,
                                               factory=SimulatedFactory(seed=seed)))
    rows = load_shadow_rows(tmp_path)
    assert len(rows) > 100
    assert 0.1 < sum(r.label for r in rows) / len(rows) < 0.9
    result = fit(rows)
    assert result.n_holdout > 0
    assert result.holdout_metrics["auc"] > 0.7
