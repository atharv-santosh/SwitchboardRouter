"""The one interface the cascade talks to, so simulated, live and replayed models are interchangeable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..events import content_hash
from ..pricing import Usage


@dataclass
class ModelRequest:
    """A step's prompt, already split into the part that should be cached and the part that should not."""

    stable_prefix: str
    volatile_suffix: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_output_tokens: int = 640
    temperature: float = 0.0
    # Opaque, never sent to the model; lets the simulated backend know what it is being asked.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prefix_hash(self) -> str:
        return content_hash({"prefix": self.stable_prefix, "tools": self.tools})

    @property
    def request_hash(self) -> str:
        return content_hash(
            {
                "prefix": self.stable_prefix,
                "suffix": self.volatile_suffix,
                "tools": self.tools,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
        )

    def prompt(self) -> str:
        return self.stable_prefix + self.volatile_suffix


@dataclass
class Action:
    """What the model decided to do this step."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "confidence": self.confidence}


@dataclass
class ModelResponse:
    action: Action
    usage: Usage
    raw_text: str = ""
    cache_hit: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "usage": self.usage.to_dict(),
            "cache_hit": self.cache_hit,
            "error": self.error,
        }


class ModelClient(Protocol):
    """Anything that can turn a ModelRequest into a ModelResponse at a known tier."""

    tier: str

    def complete(self, request: ModelRequest, cache_handle: str | None = None) -> ModelResponse: ...
