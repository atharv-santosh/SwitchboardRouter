from __future__ import annotations

import json

from click.testing import CliRunner

from arouter.cli import main


def _run(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(main, list(args))
    return result.exit_code, result.output + (str(result.exception) if result.exception else "")


def test_tiers_prints_the_ladder_and_the_price_date() -> None:
    code, out = _run("tiers")
    assert code == 0
    assert "gemini-2.5-flash-lite" in out and "gemini-3.6-flash" in out
    assert "cascade ladder" in out


def test_bench_runs_offline_and_writes_json_and_html(tmp_path) -> None:
    out, html = tmp_path / "b.json", tmp_path / "b.html"
    code, output = _run("bench", "--out", str(out), "--html", str(html), "--max-steps", "6")
    assert code == 0, output
    payload = json.loads(out.read_text())
    assert payload["backend"] == "simulated"
    assert payload["pareto_front"]
    assert any(c["config"]["name"] == "single_strong" for c in payload["configs"])
    assert "<svg" in html.read_text()


def test_bench_can_be_narrowed_to_named_configs(tmp_path) -> None:
    out = tmp_path / "b.json"
    code, output = _run("bench", "--out", str(out), "--configs", "single_cheap,single_strong")
    assert code == 0, output
    names = {c["config"]["name"] for c in json.loads(out.read_text())["configs"]}
    assert names == {"single_cheap", "single_strong"}


def test_bench_rejects_a_config_name_that_does_not_exist(tmp_path) -> None:
    code, output = _run("bench", "--out", str(tmp_path / "b.json"), "--configs", "nonexistent")
    assert code != 0
    assert "no configs matched" in output


def test_replay_backend_needs_fixtures(tmp_path) -> None:
    code, output = _run("bench", "--backend", "replay", "--out", str(tmp_path / "b.json"))
    assert code != 0 and "--fixtures is required" in output


def test_shadow_then_fit_then_bench_with_the_learned_policy(tmp_path) -> None:
    shadow_dir, fit_path, bench_path = tmp_path / "shadow", tmp_path / "fit.json", tmp_path / "b.json"
    code, output = _run("shadow", "--out", str(shadow_dir), "--seeds", "0,1")
    assert code == 0, output
    assert "labelled steps" in output

    code, output = _run("fit", str(shadow_dir), "--out", str(fit_path))
    assert code == 0, output
    fit_data = json.loads(fit_path.read_text())
    assert fit_data["n_holdout"] > 0
    assert 0.0 < fit_data["threshold"] < 1.0
    assert fit_data["holdout"]["auc"] > 0.5

    code, output = _run(
        "bench", "--out", str(bench_path), "--learned-fit", str(fit_path), "--configs", "cascade_learned_threshold"
    )
    assert code == 0, output
    configs = json.loads(bench_path.read_text())["configs"]
    assert configs[0]["config"]["policy"] == "learned_threshold"
    assert configs[0]["escalations"] > 0


def test_fit_refuses_a_directory_with_no_labels(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    code, output = _run("fit", str(tmp_path / "empty"))
    assert code != 0 and "no shadow_label events" in output


def test_bench_skips_the_learned_config_when_no_fit_is_supplied(tmp_path) -> None:
    out = tmp_path / "b.json"
    code, output = _run("bench", "--out", str(out))
    assert code == 0
    names = {c["config"]["name"] for c in json.loads(out.read_text())["configs"]}
    assert "cascade_learned_threshold" not in names


def test_cache_study_reports_the_cost_of_reordering_context(tmp_path) -> None:
    out = tmp_path / "cs.json"
    code, output = _run("cache-study", "--out", str(out))
    assert code == 0, output
    rows = {r["name"]: r for r in json.loads(out.read_text())}
    assert rows["cache_aware_cached"]["hit_rate"] > rows["naive_cached"]["hit_rate"]
    assert rows["cache_aware_cached"]["creates"] < rows["naive_cached"]["creates"]
    assert rows["cache_aware_cached"]["usd_total"] < rows["cache_aware_uncached"]["usd_total"]


def test_prefetch_study_reports_a_break_even_hit_rate(tmp_path) -> None:
    out = tmp_path / "pf.json"
    code, output = _run("prefetch-study", "--out", str(out), "--latencies", "10", "--thresholds", "0.5")
    assert code == 0, output
    row = json.loads(out.read_text())[0]
    assert row["launched"] > 0
    assert 0.0 <= row["break_even_hit_rate"] <= 1.0
    assert "pays_off" in row


def test_budget_demo_never_exceeds_a_cap(tmp_path) -> None:
    out = tmp_path / "bd.json"
    code, output = _run("budget-demo", "--out", str(out), "--run-caps", "0.006,0.02")
    assert code == 0, output
    payload = json.loads(out.read_text())
    for row in payload["per_run_caps"]:
        assert row["runs_over_cap"] == 0
    assert payload["shared_tenant"]


def test_report_regenerates_html_from_a_results_file(tmp_path) -> None:
    results, html = tmp_path / "b.json", tmp_path / "r.html"
    _run("bench", "--out", str(results), "--configs", "single_cheap")
    code, output = _run("report", str(results), "--out", str(html))
    assert code == 0, output
    assert html.read_text().startswith("<!doctype html>")


def test_verify_accounts_for_committed_fixtures(tmp_path) -> None:
    trajectories = tmp_path / "t"
    _run("bench", "--out", str(tmp_path / "b.json"), "--configs", "single_cheap", "--trajectories", str(trajectories))
    code, output = _run("verify", str(trajectories))
    assert code == 0, output
    assert "recorded calls" in output and "recorded spend" in output


def test_verify_refuses_an_empty_directory(tmp_path) -> None:
    (tmp_path / "nothing").mkdir()
    code, output = _run("verify", str(tmp_path / "nothing"))
    assert code != 0 and "no llm calls found" in output
