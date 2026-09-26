from __future__ import annotations

import pytest

from arouter.prefetch import Prefetcher, ToolPredictor, break_even_hit_rate, call_key
from arouter.suite import Suite
from arouter.tools import READ_ONLY_TOOLS


def _predictor(sequence: list[tuple[str, dict]], task_id: str = "t", band: str = "easy") -> ToolPredictor:
    predictor = ToolPredictor()
    predictor.observe(task_id, band, sequence)
    return predictor


def test_call_key_ignores_irrelevant_arguments() -> None:
    assert call_key("read_file", {"path": "a"}) == call_key("read_file", {"path": "a", "confidence": 0.9})
    assert call_key("read_file", {"path": "a"}) != call_key("read_file", {"path": "b"})


def test_the_predictor_returns_nothing_before_it_has_seen_anything() -> None:
    assert ToolPredictor().predict("t", "easy", []) is None


def test_the_predictor_recovers_a_sequence_it_observed() -> None:
    predictor = _predictor([("list_files", {}), ("read_file", {"path": "a"}), ("finish", {})])
    tool, args, probability = predictor.predict("t", "easy", [])  # type: ignore[misc]
    assert (tool, args) == ("list_files", {})
    assert probability == 1.0
    tool, args, _ = predictor.predict("t", "easy", ["list_files"])  # type: ignore[misc]
    assert (tool, args) == ("read_file", {"path": "a"})


def test_probability_falls_when_the_same_context_led_somewhere_else() -> None:
    predictor = ToolPredictor()
    predictor.observe("t", "easy", [("list_files", {}), ("read_file", {"path": "a"})])
    predictor.observe("t", "easy", [("list_files", {}), ("read_file", {"path": "b"})])
    _, _, probability = predictor.predict("t", "easy", ["list_files"])  # type: ignore[misc]
    assert probability == pytest.approx(0.5)


def test_the_task_agnostic_key_only_predicts_argument_free_tools() -> None:
    predictor = _predictor([("list_files", {}), ("read_file", {"path": "a"})], task_id="seen")
    predictor.use_task_key = False
    assert predictor.predict("unseen", "easy", [])[0] == "list_files"  # type: ignore[index]
    assert predictor.predict("unseen", "easy", ["list_files"]) is None


def test_a_write_is_never_speculated(suite: Suite) -> None:
    """A discarded write is a corrupted workspace, so this gate is not a heuristic."""
    predictor = _predictor([("set_value", {"path": "services/api.yaml", "key": "replicas", "value": "5"})])
    with Prefetcher(predictor=predictor, threshold=0.0) as prefetcher:
        assert prefetcher.maybe_start(suite.workspace.clone(), "t", "easy", []) is None
    assert prefetcher.stats.skipped_not_readonly == 1
    assert prefetcher.stats.launched == 0


def test_every_speculable_tool_is_read_only() -> None:
    assert "set_value" not in READ_ONLY_TOOLS and "finish" not in READ_ONLY_TOOLS


def test_a_prediction_below_the_threshold_is_not_launched(suite: Suite) -> None:
    predictor = ToolPredictor()
    predictor.observe("t", "easy", [("read_file", {"path": "services/api.yaml"})])
    predictor.observe("t", "easy", [("read_file", {"path": "services/web.yaml"})])
    with Prefetcher(predictor=predictor, threshold=0.9) as prefetcher:
        assert prefetcher.maybe_start(suite.workspace.clone(), "t", "easy", []) is None
    assert prefetcher.stats.skipped_low_confidence == 1


def test_a_correct_guess_is_adopted_and_credits_the_shorter_of_the_two_times(suite: Suite) -> None:
    predictor = _predictor([("read_file", {"path": "services/api.yaml"})])
    with Prefetcher(predictor=predictor, threshold=0.5, tool_latency_ms=30.0) as prefetcher:
        attempt = prefetcher.maybe_start(suite.workspace.clone(), "t", "easy", [])
        assert attempt is not None
        result = prefetcher.resolve(attempt, "read_file", {"path": "services/api.yaml"}, model_ms=5.0)
    assert result is not None and result.ok
    assert prefetcher.stats.hits == 1
    assert prefetcher.stats.ms_saved == pytest.approx(5.0, abs=1.0)


def test_a_wrong_guess_is_discarded_and_charged(suite: Suite) -> None:
    predictor = _predictor([("read_file", {"path": "services/api.yaml"})])
    with Prefetcher(predictor=predictor, threshold=0.5, tool_latency_ms=20.0, tool_usd=0.001) as prefetcher:
        attempt = prefetcher.maybe_start(suite.workspace.clone(), "t", "easy", [])
        result = prefetcher.resolve(attempt, "read_file", {"path": "services/web.yaml"}, model_ms=50.0)
    assert result is None
    assert prefetcher.stats.misses == 1
    assert prefetcher.stats.ms_wasted > 0
    assert prefetcher.stats.usd_wasted == pytest.approx(0.001)
    assert prefetcher.stats.ms_saved == 0.0


def test_speculation_reads_a_snapshot_and_cannot_mutate_the_live_workspace(suite: Suite) -> None:
    predictor = _predictor([("read_file", {"path": "services/api.yaml"})])
    live = suite.workspace.clone()
    before = live.digest()
    with Prefetcher(predictor=predictor, threshold=0.5) as prefetcher:
        attempt = prefetcher.maybe_start(live, "t", "easy", [])
        prefetcher.resolve(attempt, "read_file", {"path": "services/api.yaml"}, model_ms=1.0)
    assert live.digest() == before


def test_a_disabled_prefetcher_does_nothing(suite: Suite) -> None:
    predictor = _predictor([("read_file", {"path": "services/api.yaml"})])
    with Prefetcher(predictor=predictor, enabled=False) as prefetcher:
        assert prefetcher.maybe_start(suite.workspace.clone(), "t", "easy", []) is None
    assert prefetcher.stats.considered == 0


def test_resolving_nothing_is_safe() -> None:
    with Prefetcher(predictor=ToolPredictor()) as prefetcher:
        assert prefetcher.resolve(None, "read_file", {}, model_ms=1.0) is None


def test_break_even_rises_as_the_tool_gets_cheaper_to_wait_for() -> None:
    fast = break_even_hit_rate(tool_usd=0.002, tool_latency_ms=20, usd_per_second_of_latency=0.002)
    slow = break_even_hit_rate(tool_usd=0.002, tool_latency_ms=600, usd_per_second_of_latency=0.002)
    assert fast > slow
    assert fast == pytest.approx(0.002 / (0.002 + 0.00004))


def test_a_free_tool_always_pays_and_a_worthless_second_never_does() -> None:
    assert break_even_hit_rate(0.0, 500, 0.002) == 0.0
    assert break_even_hit_rate(0.002, 500, 0.0) == 1.0
