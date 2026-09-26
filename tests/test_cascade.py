from __future__ import annotations

import pytest

from arouter.cascade import CascadeRunner, RunConfig, build_context, render_history
from arouter.context import CacheAwareLayout, NaiveLayout
from arouter.events import EVENT_TYPES
from arouter.pricing import LADDER
from arouter.suite import Suite


def test_a_run_starts_and_ends_with_the_right_events(runner: CascadeRunner, suite: Suite, cascade_config) -> None:
    run = runner.run(suite.task("api_replicas"), cascade_config)
    assert run.events[0]["type"] == "run_start"
    assert run.events[-1]["type"] == "run_end"
    assert all(e["type"] in EVENT_TYPES for e in run.events)
    assert [e["seq"] for e in run.events] == list(range(len(run.events)))
    assert {e["run_id"] for e in run.events} == {run.run_id}


def test_every_llm_call_is_followed_by_a_result(runner: CascadeRunner, suite: Suite, cascade_config) -> None:
    run = runner.run(suite.task("cap_and_report"), cascade_config)
    types = [e["type"] for e in run.events]
    for i, name in enumerate(types):
        if name == "llm_call":
            assert types[i + 1] == "llm_result"
        if name == "tool_call":
            assert types[i + 1] == "tool_result"


def test_the_stable_prefix_never_changes_during_a_run(suite: Suite) -> None:
    ctx = build_context(suite, suite.task("halve_large_memory"), suite.workspace)
    lay = CacheAwareLayout()
    prefixes = set()
    for step in range(8):
        step_ctx = type(ctx)(blocks=list(ctx.blocks))
        step_ctx.add("history", render_history([{"tool": "read_file", "args": {}, "outcome": "x" * step}]), stable=False)
        prefixes.add(lay.render(step_ctx, step)[0])
    assert len(prefixes) == 1


def test_the_naive_layout_breaks_that_invariant(suite: Suite) -> None:
    ctx = build_context(suite, suite.task("halve_large_memory"), suite.workspace)
    lay = NaiveLayout()
    assert len({lay.render(ctx, step)[0] for step in range(8)}) == 8


def test_the_task_statement_stays_out_of_the_cached_prefix(suite: Suite) -> None:
    """One cache per tier has to serve every task, which it cannot if the task is in it."""
    a = build_context(suite, suite.task("api_replicas"), suite.workspace)
    b = build_context(suite, suite.task("cap_and_report"), suite.workspace)
    lay = CacheAwareLayout()
    assert lay.render(a, 0)[0] == lay.render(b, 0)[0]
    assert lay.render(a, 0)[1] != lay.render(b, 0)[1]


def test_a_single_tier_config_never_escalates(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(name="solo", tiers=("cheap",), policy="none")
    run = runner.run(suite.task("cap_and_report"), config)
    assert run.escalations == 0
    assert {s.tier_used for s in run.steps} == {"cheap"}


def test_always_strong_escalates_every_step(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(name="hot", tiers=LADDER[:1] + LADDER[-1:], policy="always_strong")
    run = runner.run(suite.task("api_replicas"), config)
    assert run.escalations == len(run.steps)
    assert all(s.escalated for s in run.steps)
    assert all(len(s.usages) >= 2 for s in run.steps)


def test_an_escalated_step_costs_more_than_an_accepted_one(runner: CascadeRunner, suite: Suite) -> None:
    cheap = runner.run(suite.task("api_replicas"), RunConfig(name="a", tiers=("cheap",), policy="none"))
    both = runner.run(
        suite.task("api_replicas"), RunConfig(name="b", tiers=LADDER[:1] + LADDER[-1:], policy="always_strong")
    )
    assert both.usd > cheap.usd


def test_the_oracle_cascade_solves_at_least_as_much_as_the_strong_model_alone(runner: CascadeRunner, suite: Suite) -> None:
    oracle = RunConfig(name="oracle", tiers=LADDER[:1] + LADDER[-1:], policy="oracle")
    solo = RunConfig(name="solo", tiers=("strong",), policy="none")
    a = sum(runner.run(t, oracle).solved for t in suite.tasks)
    b = sum(runner.run(t, solo).solved for t in suite.tasks)
    assert a >= b


def test_the_cheap_tier_solves_some_tasks_and_fails_others(runner: CascadeRunner, suite: Suite) -> None:
    """Without both outcomes present the cascade has nothing to route."""
    config = RunConfig(name="cheap_only", tiers=("cheap",), policy="none")
    solved = [runner.run(task, config).solved for task in suite.tasks]
    assert any(solved) and not all(solved)


def test_easy_tasks_are_easier_for_the_cheap_tier_than_hard_ones(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(name="cheap_only", tiers=("cheap",), policy="none")
    rates: dict[str, list[bool]] = {}
    for task in suite.tasks:
        rates.setdefault(task.band, []).append(runner.run(task, config).solved)
    easy = sum(rates["easy"]) / len(rates["easy"])
    hard = sum(rates["hard"]) / len(rates["hard"])
    assert easy > hard


def test_runs_are_deterministic_for_a_fixed_seed(suite: Suite, cascade_config) -> None:
    from arouter.bench import SimulatedFactory
    from arouter.cache import LocalCacheBackend, PrefixCache
    from arouter.models.verifier import SimulatedVerifier

    def once() -> dict:
        runner = CascadeRunner(
            suite,
            SimulatedFactory(seed=7),
            prefix_cache=PrefixCache(backend=LocalCacheBackend()),
            verifier_factory=lambda t: SimulatedVerifier(tier=t, seed=7),
        )
        run = runner.run(suite.task("effective_footprint"), cascade_config, seed=7)
        return {"solved": run.solved, "usd": run.usd, "actions": [s.action for s in run.steps]}

    assert once() == once()


def test_a_run_never_touches_the_shared_workspace(runner: CascadeRunner, suite: Suite, cascade_config) -> None:
    before = suite.workspace.digest()
    for task in suite.tasks:
        runner.run(task, cascade_config)
    assert suite.workspace.digest() == before


def test_shadow_mode_labels_every_step_it_escalates(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(name="shadow", tiers=LADDER[:1] + LADDER[-1:], policy="always_strong", shadow=True)
    run = runner.run(suite.task("cpu_consistency"), config)
    labels = [e for e in run.events if e["type"] == "decision" and e["payload"].get("kind") == "shadow_label"]
    assert len(labels) == len(run.steps)
    assert all(label["payload"]["cheap_was_wrong"] in (0, 1) for label in labels)
    assert all("confidence" in label["payload"]["features"] for label in labels)


def test_the_step_limit_is_reported_as_an_error_not_a_hang(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(name="tiny", tiers=("cheap",), policy="none", max_steps=1)
    run = runner.run(suite.task("halve_large_memory"), config)
    assert len(run.steps) == 1
    assert any(e["type"] == "error" for e in run.events)


def test_a_run_cap_walks_the_ladder_then_stops_without_raising(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(name="capped", tiers=("strong",), policy="none", run_cap_usd=0.006)
    run = runner.run(suite.task("halve_large_memory"), config)
    assert run.stopped_by_budget
    ladder = ["NO_ESCALATION", "NO_VERIFIER", "SHRINK_CONTEXT", "STOP"]
    levels = [d["level"] for d in run.degradations]
    # Which rungs a run lands on depends on how fast it spends, but it may never skip
    # backwards and it must finish at STOP.
    assert levels == sorted(levels, key=ladder.index)
    assert levels[0] == "NO_ESCALATION" and levels[-1] == "STOP"
    assert run.summary and "budget_exhausted" in run.summary
    kinds = {e["payload"].get("kind") for e in run.events if e["type"] == "decision"}
    assert "budget_degradation" in kinds and "budget_refusal" in kinds


def test_a_capped_run_never_spends_more_than_its_cap(runner: CascadeRunner, suite: Suite) -> None:
    for cap in (0.004, 0.005, 0.006, 0.008):
        config = RunConfig(name=f"cap{cap}", tiers=("strong",), policy="none", run_cap_usd=cap)
        run = runner.run(suite.task("halve_large_memory"), config)
        assert run.usd <= cap, f"cap {cap} overspent to {run.usd}"


def test_degradation_suppresses_escalation_before_it_stops_the_run(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(
        name="squeezed", tiers=LADDER[:1] + LADDER[-1:], policy="always_strong", run_cap_usd=0.006
    )
    run = runner.run(suite.task("halve_large_memory"), config)
    assert [d["level"] for d in run.degradations] == ["NO_ESCALATION"]
    assert 0 < run.escalations < len(run.steps)
    assert any(
        e["payload"].get("kind") == "escalation_suppressed" for e in run.events if e["type"] == "decision"
    )


@pytest.mark.parametrize("policy", ["always_cheap", "schema_validation", "self_confidence", "verifier_model", "oracle"])
def test_every_policy_runs_end_to_end(runner: CascadeRunner, suite: Suite, policy: str) -> None:
    config = RunConfig(name=policy, tiers=LADDER[:1] + LADDER[-1:], policy=policy)
    run = runner.run(suite.task("replica_cap"), config)
    assert run.policy_stats.decisions == len(run.steps)
    assert run.usd > 0


def test_an_unknown_policy_is_refused(runner: CascadeRunner, suite: Suite) -> None:
    config = RunConfig(name="bogus", tiers=LADDER[:1] + LADDER[-1:], policy="vibes")
    with pytest.raises(KeyError, match="unknown policy"):
        runner.run(suite.task("api_replicas"), config)
