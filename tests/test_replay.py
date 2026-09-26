from __future__ import annotations

import pytest

from arouter.bench import BenchOptions, SimulatedFactory, run_config
from arouter.cache import LocalCacheBackend
from arouter.cascade import CascadeRunner, RunConfig
from arouter.models.base import ModelRequest
from arouter.models.replay import FixtureIndex, ReplayFactory, ReplayMiss, ReplayModel
from arouter.pricing import LADDER
from arouter.suite import Suite


def _record(suite: Suite, tmp_path, policy: str = "schema_validation") -> RunConfig:
    config = RunConfig(name="rec", tiers=LADDER[:1] + LADDER[-1:], policy=policy)
    run_config(
        suite,
        config,
        BenchOptions(seed=0, trajectory_dir=tmp_path, cache_backend=LocalCacheBackend(),
                    factory=SimulatedFactory(seed=0)),
    )
    return config


def test_a_recorded_run_replays_to_the_same_outcome(suite: Suite, tmp_path) -> None:
    config = _record(suite, tmp_path)
    live = run_config(
        suite, config, BenchOptions(seed=0, cache_backend=LocalCacheBackend(), factory=SimulatedFactory(seed=0))
    )
    index = FixtureIndex.load(tmp_path)
    replayed = run_config(
        suite, config, BenchOptions(seed=0, cache_backend=LocalCacheBackend(), factory=ReplayFactory(index=index))
    )
    assert replayed.solved == live.solved
    assert replayed.usd_total == pytest.approx(live.usd_total)
    assert [r.task_id for r in replayed.runs] == [r.task_id for r in live.runs]
    for a, b in zip(replayed.runs, live.runs):
        assert [s.action for s in a.steps] == [s.action for s in b.steps]


def test_the_index_finds_every_recorded_call(suite: Suite, tmp_path) -> None:
    _record(suite, tmp_path)
    index = FixtureIndex.load(tmp_path)
    assert len(index.run_ids) == len(suite.tasks)
    assert len(index) > len(suite.tasks)


def test_replay_refuses_a_request_it_never_saw(suite: Suite, tmp_path) -> None:
    _record(suite, tmp_path)
    model = ReplayModel(tier="cheap", index=FixtureIndex.load(tmp_path))
    with pytest.raises(ReplayMiss, match="re-record"):
        model.complete(ModelRequest(stable_prefix="a prompt nobody recorded", volatile_suffix=""))


def test_replay_is_keyed_by_tier_as_well_as_request(suite: Suite, tmp_path) -> None:
    config = _record(suite, tmp_path)
    index = FixtureIndex.load(tmp_path)
    key_tiers = {tier for tier, _ in index.calls}
    assert "cheap" in key_tiers
    assert config.strong_tier in key_tiers


def test_a_verifier_run_records_its_verdicts(suite: Suite, tmp_path) -> None:
    _record(suite, tmp_path, policy="verifier_model")
    index = FixtureIndex.load(tmp_path)
    assert index.verdicts
    verdict = next(iter(index.verdicts.values()))[0]
    assert "escalate" in verdict and "usage" in verdict


def test_replayed_token_counts_are_the_recorded_ones(suite: Suite, tmp_path) -> None:
    runner = CascadeRunner(suite, SimulatedFactory(seed=0))
    config = RunConfig(name="one", tiers=("cheap",), policy="none")
    run = runner.run(suite.task("api_replicas"), config)
    from arouter.events import TrajectoryEvent, write_jsonl

    write_jsonl(tmp_path / "one" / "api_replicas.jsonl", [TrajectoryEvent.from_dict(e) for e in run.events])
    index = FixtureIndex.load(tmp_path)
    replayed = CascadeRunner(suite, ReplayFactory(index=index)).run(suite.task("api_replicas"), config)
    assert replayed.spend.tokens_in == run.spend.tokens_in
    assert replayed.spend.tokens_out == run.spend.tokens_out
    assert replayed.usd == pytest.approx(run.usd)
