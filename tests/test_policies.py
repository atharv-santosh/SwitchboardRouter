from __future__ import annotations

import pytest

from arouter.models.base import Action, ModelResponse
from arouter.models.verifier import SimulatedVerifier
from arouter.policies import (
    AlwaysCheap,
    AlwaysStrong,
    LearnedThreshold,
    OraclePolicy,
    PolicyStats,
    SchemaValidation,
    SelfConfidence,
    StepObservation,
    VerifierModel,
    features,
    validation_errors,
)
from arouter.pricing import Usage
from arouter.suite import Suite


def _obs(suite: Suite, task_id: str, action: Action, step: int = 0) -> StepObservation:
    task = suite.task(task_id)
    return StepObservation(
        task=task,
        step=step,
        prompt_tokens=2000,
        response=ModelResponse(action=action, usage=Usage(tier="cheap", model="m")),
        workspace=suite.workspace.clone(),
        history=[],
    )


def test_validation_catches_an_unknown_tool(suite: Suite) -> None:
    errors = validation_errors(Action(tool="inspect_service"), suite.task("api_replicas"), suite.workspace)
    assert errors and "unknown tool" in errors[0]


def test_validation_catches_a_missing_path_and_a_missing_value(suite: Suite) -> None:
    task, ws = suite.task("api_replicas"), suite.workspace
    assert validation_errors(Action(tool="read_file", args={"path": "services/api.yaml.bak"}), task, ws)
    assert validation_errors(Action(tool="set_value", args={"path": "services/api.yaml", "key": "replicas"}), task, ws)


def test_validation_refuses_a_write_to_documentation(suite: Suite) -> None:
    errors = validation_errors(
        Action(tool="set_value", args={"path": "examples/web.yaml", "key": "replicas", "value": "6"}),
        suite.task("decoy_web_replicas"),
        suite.workspace,
    )
    assert any("examples/" in e for e in errors)


def test_validation_cannot_see_a_wrong_number(suite: Suite) -> None:
    """The failure mode routing most needs to catch is the one structure cannot see."""
    wrong = Action(tool="set_value", args={"path": "services/api.yaml", "key": "replicas", "value": "6"})
    assert validation_errors(wrong, suite.task("api_replicas"), suite.workspace) == []


def test_validation_requires_the_summary_a_task_asked_for(suite: Suite) -> None:
    task = suite.task("summary_schema")
    assert validation_errors(Action(tool="finish", args={"summary": "{}"}), task, suite.workspace)
    good = '{"changed": [], "total_memory_mi": 5120, "ok": true}'
    assert validation_errors(Action(tool="finish", args={"summary": good}), task, suite.workspace) == []


def test_validation_ignores_summaries_no_task_asked_for(suite: Suite) -> None:
    task = suite.task("api_replicas")
    assert validation_errors(Action(tool="finish", args={"summary": "{}"}), task, suite.workspace) == []


def test_fixed_policies_do_what_their_names_say(suite: Suite) -> None:
    obs = _obs(suite, "api_replicas", Action(tool="list_files", confidence=0.1))
    assert AlwaysCheap().decide(obs).escalate is False
    assert AlwaysStrong().decide(obs).escalate is True


@pytest.mark.parametrize("confidence,expected", [(0.9, False), (0.75, False), (0.74, True), (0.1, True)])
def test_self_confidence_respects_its_threshold(suite: Suite, confidence: float, expected: bool) -> None:
    obs = _obs(suite, "api_replicas", Action(tool="list_files", confidence=confidence))
    assert SelfConfidence(threshold=0.75).decide(obs).escalate is expected


def test_self_confidence_escalates_when_no_confidence_was_reported(suite: Suite) -> None:
    obs = _obs(suite, "api_replicas", Action(tool="list_files", confidence=None))
    assert SelfConfidence().decide(obs).escalate is True


def test_schema_validation_costs_nothing(suite: Suite) -> None:
    obs = _obs(suite, "api_replicas", Action(tool="inspect_service"))
    decision = SchemaValidation().decide(obs)
    assert decision.escalate is True and decision.usage is None


def test_verifier_policy_charges_for_its_opinion(suite: Suite) -> None:
    obs = _obs(suite, "api_replicas", Action(tool="read_file", args={"path": "services/api.yaml"}))
    decision = VerifierModel(verifier=SimulatedVerifier(tier="mid", seed=0)).decide(obs)
    assert decision.usage is not None and decision.usage.usd > 0


def test_a_perfect_verifier_accepts_the_gold_action_and_rejects_others(suite: Suite) -> None:
    verifier = SimulatedVerifier(tier="mid", seed=0, accuracy=1.0)
    gold = suite.task("api_replicas").gold_plan[0]
    good = _obs(suite, "api_replicas", Action(tool=gold["tool"], args=dict(gold["args"])))
    assert verifier.judge(good)[0] is True
    bad = _obs(suite, "api_replicas", Action(tool="finish", args={"summary": "{}"}))
    assert verifier.judge(bad)[0] is False


def test_oracle_escalates_exactly_on_a_wrong_action(suite: Suite) -> None:
    policy = OraclePolicy(gold_lookup=lambda o: o.task.gold_plan[o.step])
    gold = suite.task("api_replicas").gold_plan[0]
    assert policy.decide(_obs(suite, "api_replicas", Action(tool=gold["tool"], args=dict(gold["args"])))).escalate is False
    assert policy.decide(_obs(suite, "api_replicas", Action(tool="grep", args={"pattern": "x"}))).escalate is True


def test_learned_policy_moves_with_its_weights(suite: Suite) -> None:
    obs = _obs(suite, "api_replicas", Action(tool="inspect_service", confidence=0.2))
    never = LearnedThreshold(weights={"bias": -10.0}, threshold=0.5)
    always = LearnedThreshold(weights={"bias": 10.0}, threshold=0.5)
    assert never.decide(obs).escalate is False
    assert always.decide(obs).escalate is True


def test_learned_policy_weights_the_structural_signal(suite: Suite) -> None:
    weights = {"bias": -2.0, "schema_invalid": 6.0}
    policy = LearnedThreshold(weights=weights, threshold=0.5)
    invalid = _obs(suite, "api_replicas", Action(tool="inspect_service"))
    valid = _obs(suite, "api_replicas", Action(tool="list_files"))
    assert policy.p_wrong(invalid) > policy.p_wrong(valid)


def test_features_are_finite_and_named(suite: Suite) -> None:
    from arouter.policies import FEATURE_NAMES

    f = features(_obs(suite, "summary_schema", Action(tool="finish", args={"summary": ""}, confidence=0.5)))
    assert set(f) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) and v == v for v in f.values())
    assert f["empty_summary_when_required"] == 1.0


def test_policy_stats_attributes_overhead_to_the_policy() -> None:
    stats = PolicyStats()
    from arouter.policies import Decision

    stats.record(Decision(escalate=True, reason="", usage=Usage(tier="mid", model="m", tokens_in=1000)))
    stats.record(Decision(escalate=False, reason=""))
    assert stats.decisions == 2 and stats.escalations == 1
    assert stats.overhead_calls == 1 and stats.usd_overhead > 0
    assert stats.escalation_rate == 0.5
