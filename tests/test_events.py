from __future__ import annotations

import pytest

from arouter.events import (
    EVENT_TYPES,
    Recorder,
    TrajectoryEvent,
    canonical_json,
    content_hash,
    read_jsonl,
    write_jsonl,
)


def test_all_nine_shared_types_are_accepted() -> None:
    for name in EVENT_TYPES:
        TrajectoryEvent(run_id="r", seq=0, ts=0.0, type=name, payload={})
    assert len(EVENT_TYPES) == 9


def test_unknown_type_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown event type"):
        TrajectoryEvent(run_id="r", seq=0, ts=0.0, type="llm_maybe", payload={})


def test_recorder_assigns_monotonic_sequence_numbers() -> None:
    recorder = Recorder(run_id="r", clock=lambda: 1.0)
    for _ in range(5):
        recorder.emit("decision", {})
    assert [e.seq for e in recorder.events] == [0, 1, 2, 3, 4]


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert content_hash({"b": 1, "a": 2}) == content_hash({"a": 2, "b": 1})


def test_jsonl_round_trip_preserves_every_field(tmp_path) -> None:
    events = [
        TrajectoryEvent(run_id="r", seq=0, ts=1.5, type="run_start", payload={"task_id": "t"}),
        TrajectoryEvent(run_id="r", seq=1, ts=2.5, type="run_end", payload={"solved": True}),
    ]
    path = tmp_path / "nested" / "r.jsonl"
    write_jsonl(path, events)
    assert [e.to_dict() for e in read_jsonl(path)] == [e.to_dict() for e in events]


def test_one_event_per_line(tmp_path) -> None:
    events = [TrajectoryEvent(run_id="r", seq=i, ts=0.0, type="decision", payload={"i": i}) for i in range(4)]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, events)
    assert len(path.read_text().strip().splitlines()) == 4
