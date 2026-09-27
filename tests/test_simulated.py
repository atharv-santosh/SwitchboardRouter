from __future__ import annotations

import time

import pytest

from arouter.models.base import ModelRequest
from arouter.models.simulated import (
    CORRUPTIONS,
    DETECTABLE_CORRUPTIONS,
    SILENT_CORRUPTIONS,
    SimulatedModel,
    estimate_tokens,
)
from arouter.policies import validation_errors
from arouter.pricing import LADDER, TIERS, tier
from arouter.suite import Suite


def _request(task_id: str = "t", difficulty: float = 0.5, step: int = 0, plan_len: int = 3) -> ModelRequest:
    return ModelRequest(
        stable_prefix="S" * 4000,
        volatile_suffix="V" * 200,
        meta={
            "task_id": task_id,
            "step": step,
            "attempt": 0,
            "difficulty": difficulty,
            "plan_len": plan_len,
            "gold_action": {"tool": "set_value", "args": {"path": "services/api.yaml", "key": "replicas", "value": "5"}},
            "decoy_path": "examples/web.yaml",
        },
    )


def test_the_same_request_gives_the_same_answer_every_time() -> None:
    model = SimulatedModel(tier="cheap", seed=3)
    first = model.complete(_request())
    second = model.complete(_request())
    assert first.action.to_dict() == second.action.to_dict()
    assert first.usage.to_dict() == second.usage.to_dict()


def test_a_different_seed_gives_a_different_run() -> None:
    a = [SimulatedModel(tier="cheap", seed=0).complete(_request(step=i)).action.to_dict() for i in range(12)]
    b = [SimulatedModel(tier="cheap", seed=1).complete(_request(step=i)).action.to_dict() for i in range(12)]
    assert a != b


def test_per_tier_quality_is_ordered_by_the_configured_skill() -> None:
    """The whole point of the stub is that tier quality is a knob, so the ordering must hold."""
    rates: dict[str, float] = {}
    for name in LADDER:
        model = SimulatedModel(tier=name, seed=0)
        hits = sum(
            model.complete(_request(task_id=f"t{i}", difficulty=0.6, step=i % 4)).action.args.get("value") == "5"
            for i in range(200)
        )
        rates[name] = hits / 200
    ordered = [rates[name] for name in LADDER]
    assert ordered == sorted(ordered)
    assert ordered[0] < ordered[-1]


def test_a_harder_task_is_solved_less_often_at_every_tier() -> None:
    for name in LADDER:
        model = SimulatedModel(tier=name, seed=0)
        easy = sum(
            model.complete(_request(task_id=f"e{i}", difficulty=0.15)).action.args.get("value") == "5"
            for i in range(150)
        )
        hard = sum(
            model.complete(_request(task_id=f"h{i}", difficulty=0.9)).action.args.get("value") == "5"
            for i in range(150)
        )
        assert easy > hard, name


def test_difficulty_is_calibrated_per_task_not_per_step() -> None:
    """A long plan must not be impossible for every tier; that would be a simulation artefact."""
    model = SimulatedModel(tier="strong", seed=0)
    short = sum(
        model.complete(_request(task_id=f"s{i}", difficulty=0.5, plan_len=2)).action.args.get("value") == "5"
        for i in range(150)
    )
    long = sum(
        model.complete(_request(task_id=f"l{i}", difficulty=0.5, plan_len=8)).action.args.get("value") == "5"
        for i in range(150)
    )
    # Per-step success rises with plan length, because the task-level target is fixed.
    assert long >= short


def test_per_tier_latency_is_ordered_and_scalable() -> None:
    latencies = {
        name: SimulatedModel(tier=name, seed=0).complete(_request()).usage.latency_ms for name in ("cheap", "strong")
    }
    assert latencies["cheap"] < latencies["strong"]
    scaled = SimulatedModel(tier="cheap", seed=0, latency_scale=4.0).complete(_request()).usage.latency_ms
    assert scaled == pytest.approx(latencies["cheap"] * 4.0)


def test_latency_is_reported_without_sleeping_unless_asked() -> None:
    started = time.perf_counter()
    response = SimulatedModel(tier="strong", seed=0).complete(_request())
    assert (time.perf_counter() - started) * 1000 < response.usage.latency_ms
    assert response.usage.latency_ms > 100


def test_sleeping_actually_waits() -> None:
    model = SimulatedModel(tier="cheap", seed=0, sleep=True, latency_scale=0.05)
    started = time.perf_counter()
    response = model.complete(_request())
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms >= response.usage.latency_ms * 0.8


def test_a_cache_handle_turns_the_prefix_into_cached_tokens() -> None:
    request = _request()
    uncached = SimulatedModel(tier="cheap", seed=0).complete(request)
    cached = SimulatedModel(tier="cheap", seed=0).complete(request, cache_handle="local://x")
    assert uncached.usage.tokens_cached_in == 0
    assert cached.usage.tokens_cached_in == estimate_tokens(request.stable_prefix)
    assert cached.usage.usd < uncached.usage.usd
    assert cached.cache_hit and not uncached.cache_hit


def test_confidence_is_reported_and_correlates_with_being_right() -> None:
    model = SimulatedModel(tier="mid", seed=0)
    right, wrong = [], []
    for i in range(300):
        response = model.complete(_request(task_id=f"c{i}", difficulty=0.6))
        assert response.action.confidence is not None
        (right if response.action.args.get("value") == "5" else wrong).append(response.action.confidence)
    assert right and wrong
    assert sum(right) / len(right) > sum(wrong) / len(wrong)


def test_confidence_distributions_overlap_so_a_threshold_has_a_real_tradeoff() -> None:
    model = SimulatedModel(tier="mid", seed=0)
    right, wrong = [], []
    for i in range(300):
        response = model.complete(_request(task_id=f"o{i}", difficulty=0.6))
        assert response.action.confidence is not None
        (right if response.action.args.get("value") == "5" else wrong).append(response.action.confidence)
    assert min(right) < max(wrong)


def test_the_corruption_modes_split_into_detectable_and_silent() -> None:
    assert set(CORRUPTIONS) == set(DETECTABLE_CORRUPTIONS) | set(SILENT_CORRUPTIONS)
    assert not set(DETECTABLE_CORRUPTIONS) & set(SILENT_CORRUPTIONS)


def test_both_kinds_of_failure_actually_occur(suite: Suite) -> None:
    """A simulation whose every failure is structurally visible would flatter one policy."""
    model = SimulatedModel(tier="cheap", seed=0)
    task = suite.task("decoy_web_replicas")
    detectable = silent = 0
    for i in range(400):
        response = model.complete(_request(task_id=f"d{i}", difficulty=0.95))
        if response.action.args.get("value") == "5" and response.action.tool == "set_value":
            continue
        if validation_errors(response.action, task, suite.workspace):
            detectable += 1
        else:
            silent += 1
    assert detectable > 20
    assert silent > 20


def test_every_tier_in_the_price_table_can_be_simulated() -> None:
    for name in TIERS:
        response = SimulatedModel(tier=name, seed=0).complete(_request())
        assert response.usage.model == tier(name).model
        assert response.usage.usd > 0
