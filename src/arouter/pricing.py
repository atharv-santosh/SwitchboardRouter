"""Model tiers and the published price list every dollar figure in this repo derives from."""

from __future__ import annotations

from dataclasses import dataclass, field

# Prices are USD per 1M tokens, Vertex AI, as published on 2026-09-25. See
# docs/pricing-sources.md for where each row came from and what is still unverified.
PRICE_TABLE_DATE = "2026-09-25"


@dataclass(frozen=True)
class Tier:
    """One rung of the capability ladder, with the prices used to cost it."""

    name: str
    model: str
    usd_per_mtok_in: float
    usd_per_mtok_cached_in: float
    usd_per_mtok_out: float
    usd_per_mtok_hour_cache_storage: float = 1.00
    # Quality/latency knobs the simulated backend uses; ignored by the real backend.
    sim_skill: float = 0.5
    sim_latency_ms: float = 200.0


TIERS: dict[str, Tier] = {
    "cheap": Tier(
        name="cheap",
        model="gemini-2.5-flash-lite",
        usd_per_mtok_in=0.10,
        usd_per_mtok_cached_in=0.01,
        usd_per_mtok_out=0.40,
        sim_skill=0.34,
        sim_latency_ms=620.0,
    ),
    "mid": Tier(
        name="mid",
        model="gemini-2.5-flash",
        usd_per_mtok_in=0.30,
        usd_per_mtok_cached_in=0.03,
        usd_per_mtok_out=2.50,
        sim_skill=0.62,
        sim_latency_ms=1150.0,
    ),
    "mid2": Tier(
        name="mid2",
        model="gemini-3.5-flash-lite",
        usd_per_mtok_in=0.30,
        usd_per_mtok_cached_in=0.03,
        usd_per_mtok_out=2.50,
        sim_skill=0.74,
        sim_latency_ms=900.0,
    ),
    "strong": Tier(
        name="strong",
        model="gemini-3.6-flash",
        usd_per_mtok_in=0.75,
        usd_per_mtok_cached_in=0.075,
        usd_per_mtok_out=3.75,
        sim_skill=0.92,
        sim_latency_ms=1900.0,
    ),
}

# The cascade's rungs in ascending cost order, and the single model the baseline uses.
LADDER = ("cheap", "mid2", "strong")
BASELINE_TIER = "strong"
VERIFIER_TIER = "mid"


def tier(name: str) -> Tier:
    if name not in TIERS:
        raise KeyError(f"unknown tier {name!r}; known tiers: {sorted(TIERS)}")
    return TIERS[name]


@dataclass
class Usage:
    """Token counts and wall-clock for one model call, plus what it cost."""

    tier: str
    model: str
    tokens_in: int = 0
    tokens_cached_in: int = 0
    tokens_out: int = 0
    tokens_thought: int = 0
    latency_ms: float = 0.0
    cache_storage_token_hours: float = 0.0

    @property
    def tokens_uncached_in(self) -> int:
        return max(0, self.tokens_in - self.tokens_cached_in)

    @property
    def usd(self) -> float:
        t = tier(self.tier)
        billed_out = self.tokens_out + self.tokens_thought
        return (
            self.tokens_uncached_in * t.usd_per_mtok_in
            + self.tokens_cached_in * t.usd_per_mtok_cached_in
            + billed_out * t.usd_per_mtok_out
            + self.cache_storage_token_hours * t.usd_per_mtok_hour_cache_storage
        ) / 1_000_000

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "tier": self.tier,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_cached_in": self.tokens_cached_in,
            "tokens_out": self.tokens_out,
            "tokens_thought": self.tokens_thought,
            "latency_ms": round(self.latency_ms, 2),
            "cache_storage_token_hours": round(self.cache_storage_token_hours, 4),
            "usd": self.usd,
        }


@dataclass
class Spend:
    """Running total over many calls."""

    usd: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_cached_in: int = 0
    tokens_out: int = 0
    by_tier: dict[str, float] = field(default_factory=dict)

    def add(self, usage: Usage) -> None:
        self.usd += usage.usd
        self.calls += 1
        self.tokens_in += usage.tokens_in
        self.tokens_cached_in += usage.tokens_cached_in
        self.tokens_out += usage.tokens_out + usage.tokens_thought
        self.by_tier[usage.tier] = self.by_tier.get(usage.tier, 0.0) + usage.usd

    def merge(self, other: Spend) -> None:
        self.usd += other.usd
        self.calls += other.calls
        self.tokens_in += other.tokens_in
        self.tokens_cached_in += other.tokens_cached_in
        self.tokens_out += other.tokens_out
        for name, usd in other.by_tier.items():
            self.by_tier[name] = self.by_tier.get(name, 0.0) + usd
