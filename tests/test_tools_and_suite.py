from __future__ import annotations

import pytest

from arouter.suite import Suite, grade, load_suite
from arouter.tools import READ_ONLY_TOOLS, TOOL_NAMES, ToolError, Workspace, execute


def test_suite_has_a_real_difficulty_ladder(suite: Suite) -> None:
    bands = suite.bands
    assert set(bands) == {"easy", "medium", "hard"}
    assert all(len(ids) >= 3 for ids in bands.values())
    difficulties = sorted(t.difficulty for t in suite.tasks)
    assert difficulties[0] < 0.25 and difficulties[-1] > 0.8


def test_every_task_has_a_gold_plan_that_ends_in_finish(suite: Suite) -> None:
    for task in suite.tasks:
        assert task.gold_plan[-1]["tool"] == "finish"
        assert task.checks


def test_the_gold_plan_solves_every_task(suite: Suite) -> None:
    """If the reference solution did not pass the grader the suite would measure nothing."""
    for task in suite.tasks:
        ws = suite.workspace.clone()
        summary = None
        for step in task.gold_plan:
            result = execute(ws, step["tool"], dict(step.get("args", {})))
            assert result.ok, f"{task.id}: gold step {step} failed with {result.error}"
            if step["tool"] == "finish":
                summary = result.payload["summary"]
        verdict = grade(task, suite.workspace, ws, summary)
        assert verdict.solved, f"{task.id} not solved by its own gold plan: {verdict.failures}"


def test_doing_nothing_solves_only_the_no_op_task(suite: Suite) -> None:
    solved = []
    for task in suite.tasks:
        ws = suite.workspace.clone()
        if grade(task, suite.workspace, ws, "{}").solved:
            solved.append(task.id)
    assert solved == ["no_op_trap"]


def test_read_only_tools_are_a_strict_subset_of_all_tools() -> None:
    assert READ_ONLY_TOOLS < TOOL_NAMES
    assert "set_value" not in READ_ONLY_TOOLS
    assert "finish" not in READ_ONLY_TOOLS


def test_set_value_rewrites_only_the_named_key() -> None:
    ws = Workspace(files={"a.yaml": "name: a\nreplicas: 2\ncpu_m: 3\n"})
    execute(ws, "set_value", {"path": "a.yaml", "key": "replicas", "value": "9"})
    assert ws.get_int("a.yaml", "replicas") == 9
    assert ws.get_int("a.yaml", "cpu_m") == 3
    assert ws.get_raw("a.yaml", "name") == "a"


def test_set_value_on_a_missing_key_fails_rather_than_appending() -> None:
    ws = Workspace(files={"a.yaml": "name: a\n"})
    result = execute(ws, "set_value", {"path": "a.yaml", "key": "replicas", "value": "9"})
    assert not result.ok and "replicas" in (result.error or "")
    assert ws.files["a.yaml"] == "name: a\n"


def test_missing_path_and_unknown_tool_both_report_cleanly() -> None:
    ws = Workspace(files={"a.yaml": "name: a\n"})
    assert not execute(ws, "read_file", {"path": "nope"}).ok
    assert not execute(ws, "inspect_service", {}).ok


def test_grep_searches_every_file_including_documentation(suite: Suite) -> None:
    result = execute(suite.workspace.clone(), "grep", {"pattern": "replicas:"})
    matches = result.payload["matches"]
    assert any("examples/web.yaml" in m for m in matches)
    assert any("services/api.yaml" in m for m in matches)


def test_bad_regex_is_an_error_not_a_crash() -> None:
    assert not execute(Workspace(files={}), "grep", {"pattern": "("}).ok


def test_clone_isolates_writes(suite: Suite) -> None:
    a = suite.workspace.clone()
    execute(a, "set_value", {"path": "services/api.yaml", "key": "replicas", "value": "77"})
    assert suite.workspace.get_int("services/api.yaml", "replicas") == 2


def test_grading_rejects_a_summary_with_extra_keys(suite: Suite) -> None:
    task = suite.task("summary_schema")
    ws = suite.workspace.clone()
    execute(ws, "set_value", {"path": "services/api.yaml", "key": "replicas", "value": "4"})
    bad = '{"changed": ["services/api.yaml"], "total_memory_mi": 5120, "ok": true, "extra": 1}'
    assert not grade(task, suite.workspace, ws, bad).solved


def test_grading_rejects_non_json_and_missing_summaries(suite: Suite) -> None:
    task = suite.task("summary_schema")
    ws = suite.workspace.clone()
    execute(ws, "set_value", {"path": "services/api.yaml", "key": "replicas", "value": "4"})
    assert not grade(task, suite.workspace, ws, "all done").solved
    assert not grade(task, suite.workspace, ws, None).solved


def test_editing_documentation_fails_a_task_that_never_mentioned_it(suite: Suite) -> None:
    task = suite.task("api_replicas")
    ws = suite.workspace.clone()
    execute(ws, "set_value", {"path": "services/api.yaml", "key": "replicas", "value": "5"})
    assert grade(task, suite.workspace, ws, "{}").solved
    execute(ws, "set_value", {"path": "examples/web.yaml", "key": "replicas", "value": "5"})
    verdict = grade(task, suite.workspace, ws, "{}")
    assert not verdict.solved and "examples/web.yaml" in verdict.failures[0]


def test_grading_has_no_partial_credit(suite: Suite) -> None:
    task = suite.task("two_edits_one_file")
    ws = suite.workspace.clone()
    execute(ws, "set_value", {"path": "services/api.yaml", "key": "cpu_m", "value": "300"})
    assert not grade(task, suite.workspace, ws, "{}").solved


def test_duplicate_task_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        "workspace: {a.yaml: 'k: 1\n'}\n"
        "tasks:\n"
        "  - {id: x, difficulty: 0.1, band: easy, prompt: p, gold_plan: [{tool: finish, args: {}}],"
        " checks: [{path: a.yaml, key: k, equals: 1}]}\n"
        "  - {id: x, difficulty: 0.1, band: easy, prompt: p, gold_plan: [{tool: finish, args: {}}],"
        " checks: [{path: a.yaml, key: k, equals: 1}]}\n"
    )
    with pytest.raises(ValueError, match="duplicate task ids"):
        load_suite(path)
