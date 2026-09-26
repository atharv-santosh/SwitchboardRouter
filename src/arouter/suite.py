"""Loading the benchmark suite and grading a finished run against final workspace state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .tools import Workspace

DEFAULT_SUITE = Path(__file__).resolve().parents[2] / "benchmarks" / "suite.yaml"
DEFAULT_MANUAL = Path(__file__).resolve().parents[2] / "benchmarks" / "manual.md"


@dataclass
class Task:
    id: str
    prompt: str
    difficulty: float
    band: str
    gold_plan: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    decoy_path: str | None = None

    @property
    def plan_len(self) -> int:
        return max(1, len(self.gold_plan))


@dataclass
class Suite:
    workspace: Workspace
    tasks: list[Task]
    manual: str

    def task(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(f"no task {task_id!r} in suite")

    @property
    def bands(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for t in self.tasks:
            out.setdefault(t.band, []).append(t.id)
        return out


def load_suite(path: str | Path | None = None, manual_path: str | Path | None = None) -> Suite:
    suite_path = Path(path) if path else DEFAULT_SUITE
    raw = yaml.safe_load(suite_path.read_text())
    tasks = [
        Task(
            id=t["id"],
            prompt=" ".join(str(t["prompt"]).split()),
            difficulty=float(t["difficulty"]),
            band=t.get("band", "medium"),
            gold_plan=t["gold_plan"],
            checks=t["checks"],
            decoy_path=t.get("decoy_path"),
        )
        for t in raw["tasks"]
    ]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids in suite")
    manual = Path(manual_path).read_text() if manual_path else DEFAULT_MANUAL.read_text()
    return Suite(workspace=Workspace(files=dict(raw["workspace"])), tasks=tasks, manual=manual)


@dataclass
class Grade:
    solved: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"solved": self.solved, "failures": self.failures}


def grade(task: Task, initial: Workspace, final: Workspace, summary: str | None) -> Grade:
    """A task is solved only if every check passes; there is no partial credit."""
    failures: list[str] = []
    for check in task.checks:
        if "summary" in check:
            failures.extend(_grade_summary(check["summary"], summary))
            continue
        path, key = check["path"], check["key"]
        try:
            actual = final.get_raw(path, key)
        except Exception as exc:  # noqa: BLE001 - a deleted or renamed file is a failed check
            failures.append(f"{path}:{key} unreadable ({exc})")
            continue
        if check.get("unchanged"):
            expected = initial.get_raw(path, key)
            if actual != expected:
                failures.append(f"{path}:{key} changed from {expected!r} to {actual!r}")
        else:
            want = str(check["equals"])
            if actual != want:
                failures.append(f"{path}:{key} is {actual!r}, want {want!r}")
    # Nothing outside services/ and policy/ may ever be written.
    for path, text in final.files.items():
        if path.startswith(("examples/", "runbook")) and text != initial.files.get(path):
            failures.append(f"{path} was modified and must not be")
    return Grade(solved=not failures, failures=failures)


def _grade_summary(spec: dict[str, Any], summary: str | None) -> list[str]:
    if summary is None:
        return ["no summary produced"]
    try:
        parsed = json.loads(summary)
    except (json.JSONDecodeError, TypeError):
        return [f"summary is not JSON: {summary[:80]!r}"]
    if not isinstance(parsed, dict):
        return ["summary JSON is not an object"]
    failures: list[str] = []
    if "exact_keys" in spec:
        want_keys = set(spec["exact_keys"])
        if set(parsed) != want_keys:
            failures.append(f"summary keys {sorted(parsed)} != {sorted(want_keys)}")
    for key, want in (spec.get("values") or {}).items():
        got = parsed.get(key)
        if isinstance(want, bool):
            if got is not want:
                failures.append(f"summary[{key}] is {got!r}, want {want!r}")
        elif isinstance(want, int):
            if not _as_int(got) == want:
                failures.append(f"summary[{key}] is {got!r}, want {want}")
        elif isinstance(want, list):
            if list(got or []) != list(want):
                failures.append(f"summary[{key}] is {got!r}, want {want!r}")
        elif str(got) != str(want):
            failures.append(f"summary[{key}] is {got!r}, want {want!r}")
    return failures


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
