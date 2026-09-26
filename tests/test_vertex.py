"""Live Vertex tests. Every one of these skips cleanly with no credentials present."""

from __future__ import annotations

import os

import pytest

from arouter.cache import MIN_CACHEABLE_TOKENS, PrefixCache
from arouter.models.base import ModelRequest
from arouter.pricing import TIERS
from arouter.tools import TOOL_SCHEMAS

pytest.importorskip("google.genai", reason="the gemini extra is not installed")

LIVE = pytest.mark.skipif(
    not os.environ.get("GOOGLE_CLOUD_PROJECT"),
    reason="set GOOGLE_CLOUD_PROJECT to run live Vertex tests",
)


def test_the_client_module_imports_without_credentials() -> None:
    from arouter.models.vertex import VertexFactory, VertexModel

    model = VertexModel(tier="cheap")
    assert model.model == TIERS["cheap"].model
    assert VertexFactory() is not None


def test_constructing_a_client_without_a_project_fails_with_a_useful_message(monkeypatch) -> None:
    from arouter.models.vertex import _client

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        _client()


@LIVE
@pytest.mark.parametrize("tier_name", sorted(TIERS))
def test_every_tier_answers_a_function_call(tier_name: str) -> None:
    from arouter.models.vertex import VertexModel

    model = VertexModel(tier=tier_name)
    request = ModelRequest(
        stable_prefix="You edit service configuration. Reply with exactly one tool call.\n",
        volatile_suffix="Task: list every file in the workspace.\n",
        tools=TOOL_SCHEMAS,
        max_output_tokens=256,
    )
    response = model.complete(request)
    assert response.error is None, response.error
    assert response.action.tool in {s["name"] for s in TOOL_SCHEMAS}
    assert response.usage.tokens_in > 0
    assert response.usage.latency_ms > 0
    assert response.usage.usd > 0


@LIVE
@pytest.mark.parametrize("tier_name", ["cheap", "mid", "strong"])
def test_an_explicit_cache_is_reported_as_cached_tokens(tier_name: str) -> None:
    """Implicit caching reported nothing on this project, so the saving is arranged explicitly."""
    from arouter.models.vertex import VertexFactory

    factory = VertexFactory()
    prefix = ("Reference line: every handler is idempotent and retried at most three times.\n"
              * 400)
    cache = PrefixCache(backend=factory.cache_backend(), ttl_seconds=120)
    request = ModelRequest(stable_prefix=prefix, volatile_suffix="Say OK.", max_output_tokens=32)
    entry = cache.lookup_or_create(tier_name, prefix, [], request.prefix_hash, MIN_CACHEABLE_TOKENS * 4)
    assert entry is None and cache.stats.creates == 1
    entry = cache.lookup_or_create(tier_name, prefix, [], request.prefix_hash, MIN_CACHEABLE_TOKENS * 4)
    assert entry is not None
    try:
        response = factory.make(tier_name).complete(request, cache_handle=entry.handle)
        assert response.usage.tokens_cached_in > 0
        assert response.cache_hit
        assert response.usage.tokens_cached_in <= response.usage.tokens_in
    finally:
        cache.close()
