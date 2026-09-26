from __future__ import annotations

import pytest

from arouter.cache import MIN_CACHEABLE_TOKENS, LocalCacheBackend, PrefixCache, counterfactual_usd_without_cache
from arouter.context import CacheAwareLayout, Context, NaiveLayout, layout, prefix_hash
from arouter.pricing import Usage, tier


def _context() -> Context:
    ctx = Context()
    ctx.add("role", "R" * 40, stable=True)
    ctx.add("manual", "M" * 400, stable=True)
    ctx.add("tools", "T" * 200, stable=True)
    ctx.add("conventions", "C" * 60, stable=True, optional=True)
    ctx.add("task", "do the thing", stable=False)
    return ctx


def test_cache_aware_prefix_is_identical_across_steps() -> None:
    ctx, lay = _context(), CacheAwareLayout()
    prefixes = {lay.render(ctx, step)[0] for step in range(12)}
    assert len(prefixes) == 1


def test_naive_prefix_changes_on_every_step() -> None:
    ctx, lay = _context(), NaiveLayout()
    prefixes = [lay.render(ctx, step)[0] for step in range(4)]
    assert len(set(prefixes)) == 4


def test_both_layouts_send_the_same_bytes_only_in_a_different_order() -> None:
    ctx = _context()
    aware = "".join(CacheAwareLayout().render(ctx, 3))
    naive = "".join(NaiveLayout().render(ctx, 3))
    assert sorted(aware.replace(" ", "")) != [] and len(naive) >= len(aware)
    for block in ctx.blocks:
        assert block.text in aware and block.text in naive


def test_no_volatile_block_precedes_a_stable_one_in_the_prefix() -> None:
    ctx = _context()
    prefix, suffix = CacheAwareLayout().render(ctx, 0)
    for block in ctx.blocks:
        assert (block.text in prefix) is block.stable
        assert (block.text in suffix) is (not block.stable)


def test_drop_optional_removes_only_optional_blocks() -> None:
    ctx = _context()
    shrunk = ctx.drop_optional()
    assert len(shrunk.blocks) == len(ctx.blocks) - 1
    assert all(not b.optional for b in shrunk.blocks)


def test_prefix_hash_depends_on_tool_schemas_too() -> None:
    assert prefix_hash("p", [{"name": "a"}]) != prefix_hash("p", [{"name": "b"}])


def test_unknown_layout_raises() -> None:
    with pytest.raises(KeyError, match="unknown layout"):
        layout("creative")


def test_cache_records_a_miss_then_hits_forever() -> None:
    cache = PrefixCache(backend=LocalCacheBackend(), ttl_seconds=3600)
    big = "x" * (MIN_CACHEABLE_TOKENS * 5)
    h = prefix_hash(big, [])
    first = cache.lookup_or_create("cheap", big, [], h, 5 * MIN_CACHEABLE_TOKENS)
    assert first is None  # the creating call still pays full input price
    for _ in range(9):
        assert cache.lookup_or_create("cheap", big, [], h, 5 * MIN_CACHEABLE_TOKENS) is not None
    assert cache.stats.creates == 1
    assert cache.stats.hits == 9
    assert cache.stats.hit_rate == pytest.approx(0.9)


def test_a_prefix_below_the_floor_is_never_cached() -> None:
    cache = PrefixCache(backend=LocalCacheBackend())
    h = prefix_hash("tiny", [])
    for _ in range(3):
        assert cache.lookup_or_create("cheap", "tiny", [], h, MIN_CACHEABLE_TOKENS - 1) is None
    assert cache.stats.creates == 0
    assert cache.stats.skipped_too_small == 3


def test_caches_are_keyed_per_tier() -> None:
    cache = PrefixCache(backend=LocalCacheBackend())
    big = "y" * (MIN_CACHEABLE_TOKENS * 5)
    h = prefix_hash(big, [])
    cache.lookup_or_create("cheap", big, [], h, 5 * MIN_CACHEABLE_TOKENS)
    cache.lookup_or_create("strong", big, [], h, 5 * MIN_CACHEABLE_TOKENS)
    assert cache.stats.creates == 2


def test_storage_cost_scales_with_ttl() -> None:
    big = "z" * (MIN_CACHEABLE_TOKENS * 5)
    h = prefix_hash(big, [])
    short = PrefixCache(backend=LocalCacheBackend(), ttl_seconds=60)
    long = PrefixCache(backend=LocalCacheBackend(), ttl_seconds=3600)
    for cache in (short, long):
        cache.lookup_or_create("cheap", big, [], h, 5 * MIN_CACHEABLE_TOKENS)
    assert long.stats.usd_storage == pytest.approx(short.stats.usd_storage * 60, rel=1e-6)


def test_net_saving_can_be_negative_when_a_cache_is_barely_reused() -> None:
    big = "w" * (MIN_CACHEABLE_TOKENS * 5)
    h = prefix_hash(big, [])
    cache = PrefixCache(backend=LocalCacheBackend(), ttl_seconds=3600)
    cache.lookup_or_create("cheap", big, [], h, 5 * MIN_CACHEABLE_TOKENS)
    cache.lookup_or_create("cheap", big, [], h, 5 * MIN_CACHEABLE_TOKENS)
    assert cache.stats.usd_net_saving < 0


def test_disabled_cache_records_nothing() -> None:
    cache = PrefixCache(enabled=False)
    assert cache.lookup_or_create("cheap", "p" * 9000, [], "h", 9000) is None
    assert cache.stats.lookups == 0


def test_counterfactual_prices_cached_tokens_at_the_full_rate() -> None:
    usage = Usage(tier="cheap", model="m", tokens_in=100_000, tokens_cached_in=90_000)
    without = counterfactual_usd_without_cache([usage])
    assert without == pytest.approx(100_000 * tier("cheap").usd_per_mtok_in / 1e6)
    assert without > usage.usd
