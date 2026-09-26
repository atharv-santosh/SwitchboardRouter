from __future__ import annotations

import pytest

from arouter.budget import BudgetEnforcer, BudgetExceeded, Degradation, Ledger
from arouter.pricing import Usage


def _charge(enforcer: BudgetEnforcer, usd: float) -> None:
    """Charge an exact dollar amount by choosing token counts that produce it."""
    enforcer.run_spent_usd += usd
    if enforcer.ledger is not None:
        enforcer.ledger.charge(usd)


def test_no_cap_means_no_degradation() -> None:
    enforcer = BudgetEnforcer()
    _charge(enforcer, 1000.0)
    assert enforcer.refresh() == []
    assert enforcer.level is Degradation.NONE
    assert enforcer.headroom() == float("inf")


def test_the_ladder_is_walked_in_order_and_never_skips_a_rung() -> None:
    enforcer = BudgetEnforcer(run_cap_usd=1.0)
    _charge(enforcer, 0.95)
    transitions = enforcer.refresh()
    assert [t["level"] for t in transitions] == ["NO_ESCALATION", "NO_VERIFIER", "SHRINK_CONTEXT"]
    assert enforcer.level is Degradation.SHRINK_CONTEXT


def test_each_rung_disables_exactly_what_it_should() -> None:
    enforcer = BudgetEnforcer(run_cap_usd=1.0)
    assert (enforcer.allow_escalation, enforcer.allow_verifier, enforcer.shrink_context) == (True, True, False)
    _charge(enforcer, 0.72)
    enforcer.refresh()
    assert not enforcer.allow_escalation and enforcer.allow_verifier
    _charge(enforcer, 0.15)
    enforcer.refresh()
    assert not enforcer.allow_verifier and not enforcer.shrink_context
    _charge(enforcer, 0.08)
    enforcer.refresh()
    assert enforcer.shrink_context and not enforcer.stopped
    _charge(enforcer, 0.10)
    enforcer.refresh()
    assert enforcer.stopped


def test_reaching_the_cap_degrades_rather_than_raising() -> None:
    enforcer = BudgetEnforcer(run_cap_usd=0.001)
    _charge(enforcer, 0.01)
    enforcer.refresh()
    assert enforcer.stopped
    assert enforcer.transitions[-1]["level"] == "STOP"


def test_strict_mode_is_the_only_way_to_get_an_exception() -> None:
    enforcer = BudgetEnforcer(run_cap_usd=0.001, strict=True)
    _charge(enforcer, 0.01)
    with pytest.raises(BudgetExceeded):
        enforcer.refresh()


def test_a_call_is_refused_in_advance_not_regretted_afterwards() -> None:
    enforcer = BudgetEnforcer(run_cap_usd=0.00001)
    assert not enforcer.can_afford("strong", prompt_tokens=20_000, cached_tokens=0, max_output_tokens=640)
    assert enforcer.run_spent_usd == 0.0


def test_strict_mode_estimate_is_a_hard_upper_bound() -> None:
    enforcer = BudgetEnforcer(strict_bound=True)
    estimate = enforcer.estimate_usd("strong", prompt_tokens=5000, cached_tokens=0, max_output_tokens=640)
    worst_case = Usage(tier="strong", model="m", tokens_in=5000, tokens_out=640, tokens_thought=640).usd
    assert estimate >= worst_case


def test_the_default_estimate_trades_the_hard_bound_for_a_usable_cap() -> None:
    """Documented consequence: a cap can be overshot by at most one call's cost."""
    calibrated = BudgetEnforcer()
    strict = BudgetEnforcer(strict_bound=True)
    args = ("strong", 5000, 0, 640)
    assert calibrated.estimate_usd(*args) < strict.estimate_usd(*args)
    typical = Usage(tier="strong", model="m", tokens_in=5000, tokens_out=110, tokens_thought=160).usd
    assert calibrated.estimate_usd(*args) > typical


def test_a_cached_prefix_makes_the_same_call_affordable() -> None:
    enforcer = BudgetEnforcer(run_cap_usd=0.002)
    assert enforcer.estimate_usd("strong", 40_000, 39_000, 640) < enforcer.estimate_usd("strong", 40_000, 0, 640)


def test_a_tenant_cap_binds_across_runs() -> None:
    ledger = Ledger(tenant="acme", cap_usd=0.01)
    first = BudgetEnforcer(ledger=ledger)
    first.charge(Usage(tier="strong", model="m", tokens_in=10_000_000))
    assert ledger.spent_usd == pytest.approx(7.5)
    second = BudgetEnforcer(ledger=ledger)
    assert second.headroom() == 0.0
    second.refresh()
    assert second.stopped


def test_the_tighter_of_the_two_caps_binds() -> None:
    ledger = Ledger(tenant="acme", cap_usd=100.0)
    enforcer = BudgetEnforcer(run_cap_usd=0.5, ledger=ledger)
    assert enforcer.binding_cap == pytest.approx(0.5)
    ledger.spent_usd = 99.9
    assert BudgetEnforcer(run_cap_usd=0.5, ledger=ledger).binding_cap == pytest.approx(0.1)


def test_a_ledger_without_a_cap_is_unbounded() -> None:
    assert Ledger(tenant="acme").remaining() == float("inf")
