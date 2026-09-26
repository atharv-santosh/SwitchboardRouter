from __future__ import annotations

import pytest

from arouter.pricing import LADDER, Spend, TIERS, Usage, tier


def test_ladder_is_strictly_increasing_in_input_price() -> None:
    prices = [tier(name).usd_per_mtok_in for name in LADDER]
    assert prices == sorted(prices)
    assert len(set(prices)) == len(prices)


def test_cached_input_is_cheaper_than_uncached_for_every_tier() -> None:
    for spec in TIERS.values():
        assert spec.usd_per_mtok_cached_in < spec.usd_per_mtok_in


def test_cost_splits_cached_and_uncached_input() -> None:
    usage = Usage(tier="cheap", model="m", tokens_in=1_000_000, tokens_cached_in=900_000, tokens_out=0)
    # 100k at $0.10/M plus 900k at $0.01/M.
    assert usage.usd == pytest.approx(0.01 + 0.009)
    assert usage.tokens_uncached_in == 100_000


def test_thinking_tokens_are_billed_as_output() -> None:
    without = Usage(tier="strong", model="m", tokens_out=1000)
    with_thought = Usage(tier="strong", model="m", tokens_out=1000, tokens_thought=1000)
    assert with_thought.usd == pytest.approx(without.usd * 2)


def test_cache_storage_is_charged_in_token_hours() -> None:
    usage = Usage(tier="mid", model="m", cache_storage_token_hours=1_000_000)
    assert usage.usd == pytest.approx(tier("mid").usd_per_mtok_hour_cache_storage)


def test_spend_accumulates_per_tier() -> None:
    spend = Spend()
    spend.add(Usage(tier="cheap", model="m", tokens_in=1000))
    spend.add(Usage(tier="strong", model="m", tokens_in=1000))
    assert spend.calls == 2
    assert set(spend.by_tier) == {"cheap", "strong"}
    assert spend.by_tier["strong"] > spend.by_tier["cheap"]


def test_spend_merge_is_additive() -> None:
    a, b = Spend(), Spend()
    a.add(Usage(tier="cheap", model="m", tokens_in=500))
    b.add(Usage(tier="cheap", model="m", tokens_in=500))
    a.merge(b)
    assert a.calls == 2 and a.tokens_in == 1000


def test_unknown_tier_raises() -> None:
    with pytest.raises(KeyError, match="unknown tier"):
        tier("gigantic")
