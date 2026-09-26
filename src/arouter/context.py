"""Context assembly, arranged so the cacheable prefix stays byte-identical across a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .events import content_hash


@dataclass(frozen=True)
class Block:
    """One labelled chunk of prompt, marked by whether it can change during a run."""

    name: str
    text: str
    stable: bool
    # Optional blocks are the first thing dropped when the budget enforcer degrades a run.
    optional: bool = False


@dataclass
class Context:
    blocks: list[Block] = field(default_factory=list)

    def add(self, name: str, text: str, *, stable: bool, optional: bool = False) -> None:
        self.blocks.append(Block(name=name, text=text, stable=stable, optional=optional))

    def drop_optional(self) -> Context:
        """A copy with optional blocks removed; used by graceful degradation."""
        return Context(blocks=[b for b in self.blocks if not b.optional])


class Layout(Protocol):
    name: str

    def render(self, context: Context, step: int) -> tuple[str, str]: ...


@dataclass
class CacheAwareLayout:
    """Every stable block first, in a fixed order, so the prefix is bit-identical each step.

    The invariant this enforces is not subtle but it is easy to lose: nothing that
    varies within a run may appear before anything that does not. A single timestamp,
    step counter or reordered tool list in the prefix region moves the cache boundary
    to zero.
    """

    name: str = "cache_aware"
    separator: str = "\n\n"

    def render(self, context: Context, step: int) -> tuple[str, str]:
        stable = [b for b in context.blocks if b.stable]
        volatile = [b for b in context.blocks if not b.stable]
        prefix = self.separator.join(b.text for b in stable)
        if prefix:
            prefix += self.separator
        suffix = self.separator.join(b.text for b in volatile)
        return prefix, suffix


@dataclass
class NaiveLayout:
    """Assembles context the way people actually do it, and loses the cache for it.

    Two habits, both common and both fatal to prefix caching: rotate the stable
    blocks as the run proceeds (what you get from assembling context out of an
    unordered mapping, or from a "most recently touched first" ordering), and let a
    step counter into the prefix region. The bytes before the first difference are
    all the cache can keep, so both habits drive the hit rate to zero while the
    prompt stays semantically identical.
    """

    name: str = "naive"
    separator: str = "\n\n"

    def render(self, context: Context, step: int) -> tuple[str, str]:
        stable = [b for b in context.blocks if b.stable]
        volatile = [b for b in context.blocks if not b.stable]
        if stable:
            pivot = step % len(stable)
            stable = stable[pivot:] + stable[:pivot]
        header = f"[step {step}]"
        prefix = self.separator.join([header] + [b.text for b in stable])
        if prefix:
            prefix += self.separator
        suffix = self.separator.join(b.text for b in volatile)
        return prefix, suffix


LAYOUTS: dict[str, type[CacheAwareLayout] | type[NaiveLayout]] = {
    "cache_aware": CacheAwareLayout,
    "naive": NaiveLayout,
}


def layout(name: str) -> CacheAwareLayout | NaiveLayout:
    if name not in LAYOUTS:
        raise KeyError(f"unknown layout {name!r}; known: {sorted(LAYOUTS)}")
    return LAYOUTS[name]()


def prefix_hash(prefix: str, tools: object) -> str:
    return content_hash({"prefix": prefix, "tools": tools})
