"""Per-run and per-tenant spend caps, and the ladder a run walks down instead of crashing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .pricing import Usage, tier


class Degradation(IntEnum):
    """Steps down, cheapest capability to lose first.

    The order is chosen so that each rung sheds spend without invalidating the run's
    output. Dropping escalation loses accuracy on hard steps but still answers. Dropping
    the verifier loses the signal that decides escalation, so it comes after. Shrinking
    context is last before stopping, because it is the only rung that changes what the
    model was told, and it is deliberately arranged to leave the cached prefix intact so
    that degrading does not also destroy the cache hit rate.
    """

    NONE = 0
    NO_ESCALATION = 1
    NO_VERIFIER = 2
    SHRINK_CONTEXT = 3
    STOP = 4


class BudgetExceeded(Exception):
    """Raised only when a caller explicitly asks for strict mode instead of degradation."""


@dataclass
class Ledger:
    """Shared spend across every run belonging to one tenant."""

    tenant: str
    cap_usd: float | None = None
    spent_usd: float = 0.0

    def remaining(self) -> float:
        return float("inf") if self.cap_usd is None else max(0.0, self.cap_usd - self.spent_usd)

    def charge(self, usd: float) -> None:
        self.spent_usd += usd


@dataclass
class BudgetEnforcer:
    """Decides, before each call, whether it fits; degrades the run when it does not.

    A cap is enforced by refusing calls in advance, using an upper-bound estimate of
    what the call will cost. Checking afterwards is not enforcement, it is reporting,
    and a cap that can be blown through by one large call is not a cap.
    """

    run_cap_usd: float | None = None
    ledger: Ledger | None = None
    strict: bool = False
    # Fractions of the binding cap at which each rung engages.
    thresholds: dict[Degradation, float] = field(
        default_factory=lambda: {
            Degradation.NO_ESCALATION: 0.70,
            Degradation.NO_VERIFIER: 0.85,
            Degradation.SHRINK_CONTEXT: 0.92,
            Degradation.STOP: 1.00,
        }
    )
    run_spent_usd: float = 0.0
    level: Degradation = Degradation.NONE
    transitions: list[dict[str, object]] = field(default_factory=list)

    @property
    def binding_cap(self) -> float | None:
        caps = [c for c in (self.run_cap_usd, self.ledger.remaining() + self.run_spent_usd if self.ledger else None) if c is not None]
        return min(caps) if caps else None

    def headroom(self) -> float:
        cap = self.binding_cap
        return float("inf") if cap is None else max(0.0, cap - self.run_spent_usd)

    def fraction_used(self) -> float:
        cap = self.binding_cap
        if cap is None or cap <= 0:
            return 0.0
        return self.run_spent_usd / cap

    def estimate_usd(self, tier_name: str, prompt_tokens: int, cached_tokens: int, max_output_tokens: int) -> float:
        """An upper bound on a call's cost, assuming the output budget is fully spent."""
        spec = tier(tier_name)
        uncached = max(0, prompt_tokens - cached_tokens)
        # Thinking tokens are billed as output, so budget for them before they arrive.
        out = max_output_tokens * 2.5
        return (
            uncached * spec.usd_per_mtok_in
            + cached_tokens * spec.usd_per_mtok_cached_in
            + out * spec.usd_per_mtok_out
        ) / 1_000_000

    def can_afford(self, tier_name: str, prompt_tokens: int, cached_tokens: int, max_output_tokens: int) -> bool:
        return self.estimate_usd(tier_name, prompt_tokens, cached_tokens, max_output_tokens) <= self.headroom()

    def charge(self, usage: Usage) -> None:
        self.run_spent_usd += usage.usd
        if self.ledger is not None:
            self.ledger.charge(usage.usd)

    def refresh(self) -> list[dict[str, object]]:
        """Advance the degradation level to match current spend; returns the new transitions."""
        cap = self.binding_cap
        if cap is None:
            return []
        used = self.fraction_used()
        target = Degradation.NONE
        for rung, fraction in sorted(self.thresholds.items(), key=lambda kv: kv[1]):
            if used >= fraction:
                target = max(target, rung)
        new: list[dict[str, object]] = []
        while self.level < target:
            self.level = Degradation(self.level + 1)
            record = {
                "level": self.level.name,
                "fraction_used": round(used, 4),
                "run_spent_usd": self.run_spent_usd,
                "cap_usd": cap,
                "reason": f"spend reached {used:.1%} of the binding cap",
            }
            self.transitions.append(record)
            new.append(record)
        if self.strict and self.level >= Degradation.STOP:
            raise BudgetExceeded(f"run spend {self.run_spent_usd:.6f} exceeded cap {cap:.6f}")
        return new

    @property
    def allow_escalation(self) -> bool:
        return self.level < Degradation.NO_ESCALATION

    @property
    def allow_verifier(self) -> bool:
        return self.level < Degradation.NO_VERIFIER

    @property
    def shrink_context(self) -> bool:
        return self.level >= Degradation.SHRINK_CONTEXT

    @property
    def stopped(self) -> bool:
        return self.level >= Degradation.STOP

    def to_dict(self) -> dict[str, object]:
        return {
            "run_cap_usd": self.run_cap_usd,
            "tenant": self.ledger.tenant if self.ledger else None,
            "tenant_cap_usd": self.ledger.cap_usd if self.ledger else None,
            "run_spent_usd": self.run_spent_usd,
            "level": self.level.name,
            "transitions": self.transitions,
        }
