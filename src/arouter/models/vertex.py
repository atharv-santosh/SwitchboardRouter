"""Vertex AI Gemini client, reporting real tokens, real cached tokens and real latency."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..pricing import Usage, tier
from .base import Action, ModelRequest, ModelResponse

DEFAULT_LOCATION = "global"
RETRYABLE = ("429", "500", "502", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE")


def _tool(types: Any, declarations: list[dict[str, Any]]) -> Any:
    """Wrap raw function declarations; the SDK accepts dicts but its stubs insist on models."""
    return types.Tool(function_declarations=declarations)


def _any_tool_config(types: Any) -> Any:
    """Force a function call rather than prose, so a step always produces an action."""
    return types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)
    )


def _client(project: str | None = None, location: str | None = None) -> Any:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("install the gemini extra: uv pip install -e '.[gemini]'") from exc
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("set GOOGLE_CLOUD_PROJECT (this is only needed for live runs)")
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
    return genai.Client(vertexai=True, project=project, location=location)


@dataclass
class VertexModel:
    """One capability tier, bound to the Gemini model id the price table quotes."""

    tier: str
    project: str | None = None
    location: str | None = None
    temperature: float = 0.0
    max_retries: int = 5
    shared_client: Any = None

    def __post_init__(self) -> None:
        self._spec = tier(self.tier)
        self._c = self.shared_client

    @property
    def model(self) -> str:
        return self._spec.model

    def _get(self) -> Any:
        if self._c is None:
            self._c = _client(self.project, self.location)
        return self._c

    def complete(self, request: ModelRequest, cache_handle: str | None = None) -> ModelResponse:
        from google.genai import types

        client = self._get()
        if cache_handle:
            # The prefix lives in the cache object, so only the volatile half is sent.
            contents: list[Any] = [{"role": "user", "parts": [{"text": request.volatile_suffix}]}]
        else:
            contents = [{"role": "user", "parts": [{"text": request.prompt()}]}]

        # Vertex rejects a request that sets tools or tool_config alongside cached content:
        # both belong to the cache object, which is created with them. So a cached call
        # sends neither, and the forced-function-calling mode comes from the cache.
        send_tools = bool(request.tools) and not cache_handle
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=request.max_output_tokens,
            tools=[_tool(types, request.tools)] if send_tools else None,
            tool_config=_any_tool_config(types) if send_tools else None,
            cached_content=cache_handle or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        started = time.perf_counter()
        response = self._call_with_retry(client, contents, config)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = self._usage(response, latency_ms)
        action, text, error = self._action(response)
        return ModelResponse(
            action=action,
            usage=usage,
            raw_text=text,
            cache_hit=usage.tokens_cached_in > 0,
            error=error,
        )

    def _call_with_retry(self, client: Any, contents: Any, config: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return client.models.generate_content(model=self.model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001 - classified below and re-raised if permanent
                last = exc
                if not any(m in f"{exc}" for m in RETRYABLE) or attempt == self.max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 30))
        raise last  # type: ignore[misc]

    def _usage(self, response: Any, latency_ms: float) -> Usage:
        u = getattr(response, "usage_metadata", None)
        return Usage(
            tier=self.tier,
            model=self.model,
            tokens_in=int(getattr(u, "prompt_token_count", 0) or 0),
            tokens_cached_in=int(getattr(u, "cached_content_token_count", 0) or 0),
            tokens_out=int(getattr(u, "candidates_token_count", 0) or 0),
            tokens_thought=int(getattr(u, "thoughts_token_count", 0) or 0),
            latency_ms=latency_ms,
        )

    def _action(self, response: Any) -> tuple[Action, str, str | None]:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return Action(tool="finish", args={"summary": "{}"}), "", "no candidates"
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            reason = getattr(candidate, "finish_reason", "UNKNOWN")
            return Action(tool="finish", args={"summary": "{}"}), "", f"empty candidate ({reason})"

        texts: list[str] = []
        for part in parts:
            call = getattr(part, "function_call", None)
            if call is not None:
                args = dict(call.args) if call.args else {}
                confidence = args.pop("confidence", None)
                action = Action(tool=call.name, args=args)
                if isinstance(confidence, (int, float)):
                    action.confidence = float(confidence)
                return action, "", None
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        joined = "".join(texts)
        return Action(tool="finish", args={"summary": joined[:400] or "{}"}), joined, "no function call"


@dataclass
class VertexCacheBackend:
    """Creates and deletes explicit Vertex context caches for a stable prefix."""

    project: str | None = None
    location: str | None = None
    shared_client: Any = None
    created: list[str] = field(default_factory=list)

    def _get(self) -> Any:
        if self.shared_client is None:
            self.shared_client = _client(self.project, self.location)
        return self.shared_client

    def create(self, model: str, prefix: str, tools: list[dict[str, Any]], ttl_seconds: int) -> tuple[str, int]:
        from google.genai import types

        client = self._get()
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                contents=[{"role": "user", "parts": [{"text": prefix}]}],
                tools=[_tool(types, tools)] if tools else None,
                ttl=f"{int(ttl_seconds)}s",
            ),
        )
        tokens = int(getattr(cache.usage_metadata, "total_token_count", 0) or 0)
        self.created.append(cache.name)
        return cache.name, tokens

    def delete(self, handle: str) -> None:
        self._get().caches.delete(name=handle)


@dataclass
class VertexFactory:
    """Builds one client per tier over a single shared genai client."""

    project: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        self._shared: Any = None

    def _client(self) -> Any:
        if self._shared is None:
            self._shared = _client(self.project, self.location)
        return self._shared

    def make(self, tier_name: str) -> VertexModel:
        return VertexModel(tier=tier_name, shared_client=self._client())

    def cache_backend(self) -> VertexCacheBackend:
        return VertexCacheBackend(shared_client=self._client())
