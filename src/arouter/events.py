"""Trajectory event schema shared with the sibling agent projects in this portfolio."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

EVENT_TYPES = (
    "run_start",
    "llm_call",
    "llm_result",
    "tool_call",
    "tool_result",
    "decision",
    "human_gate",
    "error",
    "run_end",
)


@dataclass(frozen=True)
class TrajectoryEvent:
    """One line of a JSONL trajectory."""

    run_id: str
    seq: int
    ts: float
    type: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}; expected one of {EVENT_TYPES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": self.ts,
            "type": self.type,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrajectoryEvent:
        return cls(
            run_id=raw["run_id"],
            seq=int(raw["seq"]),
            ts=float(raw["ts"]),
            type=raw["type"],
            payload=raw.get("payload", {}),
        )


def canonical_json(obj: Any) -> str:
    """The one canonical encoding every sibling project hashes against."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


@dataclass
class Recorder:
    """Accumulates events for one run and assigns sequence numbers."""

    run_id: str
    events: list[TrajectoryEvent] = field(default_factory=list)
    clock: Any = time.time

    def emit(self, type: str, payload: dict[str, Any] | None = None) -> TrajectoryEvent:
        event = TrajectoryEvent(
            run_id=self.run_id,
            seq=len(self.events),
            ts=float(self.clock()),
            type=type,
            payload=payload or {},
        )
        self.events.append(event)
        return event

    def of_type(self, type: str) -> list[TrajectoryEvent]:
        return [e for e in self.events if e.type == type]


def write_jsonl(path: str | Path, events: list[TrajectoryEvent]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for event in events:
            fh.write(canonical_json(event.to_dict()) + "\n")


def read_jsonl(path: str | Path) -> list[TrajectoryEvent]:
    out: list[TrajectoryEvent] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(TrajectoryEvent.from_dict(json.loads(line)))
    return out


def iter_runs(directory: str | Path) -> Iterator[tuple[str, list[TrajectoryEvent]]]:
    """Yield (run_id, events) for every ``*.jsonl`` under ``directory``, sorted by path."""
    for path in sorted(Path(directory).rglob("*.jsonl")):
        events = read_jsonl(path)
        if events:
            yield events[0].run_id, events
