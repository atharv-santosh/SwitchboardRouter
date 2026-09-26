"""Prefix cache bookkeeping: which prefixes are live, how often they hit, what that saved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .pricing import Usage, tier

# Vertex AI refuses to create an explicit cache below this many tokens, so a prefix
# smaller than the floor is not worth arranging the whole prompt around.
MIN_CACHEABLE_TOKENS = 1024


class CacheBackend(Protocol):
    """What the cache manager needs from a provider to hold a prefix."""

    def create(self, model: str, prefix: str, tools: list[dict[str, Any]], ttl_seconds: int) -> tuple[str, int]: ...

    def delete(self, handle: str) -> None: ...


@dataclass
class CacheEntry:
    handle: str
    tokens: int
    model: str


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    creates: int = 0
    skipped_too_small: int = 0
    cached_tokens_served: int = 0
    uncached_prefix_tokens_paid: int = 0
    storage_token_hours: float = 0.0
    usd_saved_on_reads: float = 0.0
    usd_storage: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def usd_net_saving(self) -> float:
        return self.usd_saved_on_reads - self.usd_storage

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "creates": self.creates,
            "skipped_too_small": self.skipped_too_small,
            "hit_rate": round(self.hit_rate, 4),
            "cached_tokens_served": self.cached_tokens_served,
            "uncached_prefix_tokens_paid": self.uncached_prefix_tokens_paid,
            "storage_token_hours": round(self.storage_token_hours, 3),
            "usd_saved_on_reads": self.usd_saved_on_reads,
            "usd_storage": self.usd_storage,
            "usd_net_saving": self.usd_net_saving,
        }

    def merge(self, other: CacheStats) -> None:
        self.lookups += other.lookups
        self.hits += other.hits
        self.misses += other.misses
        self.creates += other.creates
        self.skipped_too_small += other.skipped_too_small
        self.cached_tokens_served += other.cached_tokens_served
        self.uncached_prefix_tokens_paid += other.uncached_prefix_tokens_paid
        self.storage_token_hours += other.storage_token_hours
        self.usd_saved_on_reads += other.usd_saved_on_reads
        self.usd_storage += other.usd_storage


@dataclass
class PrefixCache:
    """Maps a prefix hash to a live provider cache, and accounts for what that costs.

    Storage is charged for the whole TTL at creation, not for the time actually used,
    because that is what the provider bills. That is why a prefix reused twice can be
    a net loss and the same prefix reused thirty times is a large win, and why the
    net figure matters more than the hit rate.
    """

    backend: CacheBackend | None = None
    ttl_seconds: int = 900
    enabled: bool = True
    stats: CacheStats = field(default_factory=CacheStats)
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def lookup_or_create(
        self,
        tier_name: str,
        prefix: str,
        tools: list[dict[str, Any]],
        prefix_hash: str,
        estimated_tokens: int,
    ) -> CacheEntry | None:
        """Return a live cache for this prefix, creating one if the prefix is big enough."""
        if not self.enabled:
            return None
        self.stats.lookups += 1
        key = f"{tier_name}:{prefix_hash}"
        entry = self.entries.get(key)
        if entry is not None:
            self.stats.hits += 1
            self.stats.cached_tokens_served += entry.tokens
            spec = tier(tier_name)
            self.stats.usd_saved_on_reads += (
                entry.tokens * (spec.usd_per_mtok_in - spec.usd_per_mtok_cached_in) / 1_000_000
            )
            return entry

        self.stats.misses += 1
        self.stats.uncached_prefix_tokens_paid += estimated_tokens
        if estimated_tokens < MIN_CACHEABLE_TOKENS:
            self.stats.skipped_too_small += 1
            return None
        if self.backend is None:
            return None

        spec = tier(tier_name)
        handle, tokens = self.backend.create(spec.model, prefix, tools, self.ttl_seconds)
        entry = CacheEntry(handle=handle, tokens=tokens, model=spec.model)
        self.entries[key] = entry
        self.stats.creates += 1
        token_hours = tokens * (self.ttl_seconds / 3600.0)
        self.stats.storage_token_hours += token_hours
        self.stats.usd_storage += token_hours * spec.usd_per_mtok_hour_cache_storage / 1_000_000
        return None  # the creating call itself pays full input price for the prefix

    def close(self) -> None:
        """Release provider caches; a no-op offline."""
        if self.backend is None:
            return
        for entry in self.entries.values():
            try:
                self.backend.delete(entry.handle)
            except Exception:  # noqa: BLE001 - TTL expiry makes this best-effort by design
                pass
        self.entries.clear()


def counterfactual_usd_without_cache(usages: list[Usage]) -> float:
    """What the same token counts would have cost billed entirely at the uncached rate."""
    total = 0.0
    for usage in usages:
        spec = tier(usage.tier)
        total += (
            usage.tokens_in * spec.usd_per_mtok_in
            + (usage.tokens_out + usage.tokens_thought) * spec.usd_per_mtok_out
        ) / 1_000_000
    return total
