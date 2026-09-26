"""A single self-contained HTML page: the cost/success frontier and the levers behind it."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

W, H = 720, 400
PAD_L, PAD_R, PAD_T, PAD_B = 62, 26, 22, 48


def _fmt_usd(value: Any) -> str:
    if value is None:
        return "n/a"
    v = float(value)
    if v == 0:
        return "$0"
    if v < 0.01:
        return f"${v:.5f}"
    return f"${v:.4f}"


def _scatter(configs: list[dict[str, Any]], front: list[str]) -> str:
    points = [(c["usd_per_task"], c["success_rate"], c["config"]["name"]) for c in configs if c["n_tasks"]]
    if not points:
        return "<p>No configurations to plot.</p>"
    xs = [p[0] for p in points]
    x_max = max(xs) * 1.12 or 1.0
    x_min = 0.0

    def sx(x: float) -> float:
        return PAD_L + (x - x_min) / (x_max - x_min) * (W - PAD_L - PAD_R)

    def sy(y: float) -> float:
        return H - PAD_B - y * (H - PAD_T - PAD_B)

    parts: list[str] = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Success rate against cost per task">']
    parts.append(
        f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" class="axis"/>'
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" class="axis"/>'
    )
    for i in range(0, 6):
        y = i / 5
        parts.append(f'<line x1="{PAD_L}" y1="{sy(y):.1f}" x2="{W - PAD_R}" y2="{sy(y):.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD_L - 9}" y="{sy(y) + 4:.1f}" class="tick end">{y:.0%}</text>')
    for i in range(0, 5):
        x = x_max * i / 4
        parts.append(f'<text x="{sx(x):.1f}" y="{H - PAD_B + 17:.1f}" class="tick mid">{_fmt_usd(x)}</text>')

    front_points = sorted([p for p in points if p[2] in front], key=lambda p: p[0])
    if len(front_points) > 1:
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in front_points)
        parts.append(f'<polyline points="{path}" class="front"/>')

    for x, y, name in sorted(points, key=lambda p: p[0]):
        on_front = name in front
        cls = "dot front-dot" if on_front else "dot"
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{5.5 if on_front else 4}" class="{cls}"/>')
        anchor = "end" if sx(x) > W * 0.72 else "start"
        dx = -9 if anchor == "end" else 9
        parts.append(
            f'<text x="{sx(x) + dx:.1f}" y="{sy(y) - 8:.1f}" class="label {anchor}">{html.escape(name)}</text>'
        )

    parts.append(
        f'<text x="{(W + PAD_L) / 2:.0f}" y="{H - 6}" class="axis-title mid">dollars per task attempted</text>'
        f'<text x="14" y="{(H) / 2:.0f}" class="axis-title" transform="rotate(-90 14 {(H) / 2:.0f})">tasks solved</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render(report: dict[str, Any]) -> str:
    configs = report["configs"]
    front = report.get("pareto_front", [])
    baseline_name = report["baseline"]
    baseline = next((c for c in configs if c["config"]["name"] == baseline_name), None)

    main_rows: list[list[str]] = []
    for c in sorted(configs, key=lambda c: c["usd_per_task"]):
        name = c["config"]["name"]
        mark = " *" if name in front else ""
        ratio = c.get("usd_per_solved_vs_baseline")
        delta = c.get("success_delta_vs_baseline")
        main_rows.append(
            [
                f"<code>{html.escape(name)}{mark}</code>",
                f"{c['solved']}/{c['n_tasks']}",
                f"{c['success_rate']:.0%}",
                _fmt_usd(c["usd_per_task"]),
                _fmt_usd(c["usd_per_solved_task"]),
                f"{ratio:.2f}x" if ratio is not None else "n/a",
                f"{delta:+.0%}" if delta is not None else "n/a",
                f"{c['p50_ms']:.0f}",
                f"{c['p95_ms']:.0f}",
                str(c["escalations"]),
            ]
        )

    band_names = sorted({b for c in configs for b in c["solved_by_band"]})
    band_rows = [
        [f"<code>{html.escape(c['config']['name'])}</code>"]
        + [
            (lambda v: f"{v[0]}/{v[1]}")(c["solved_by_band"][b]) if b in c["solved_by_band"] else "-"
            for b in band_names
        ]
        for c in sorted(configs, key=lambda c: c["usd_per_task"])
    ]

    cache_rows = []
    for c in configs:
        cache = c["cache"]
        if not cache["lookups"]:
            continue
        cache_rows.append(
            [
                f"<code>{html.escape(c['config']['name'])}</code>",
                html.escape(c["config"]["layout"]),
                str(cache["lookups"]),
                f"{cache['hit_rate']:.0%}",
                f"{cache['cached_tokens_served']:,}",
                _fmt_usd(cache["usd_saved_on_reads"]),
                _fmt_usd(cache["usd_storage"]),
                _fmt_usd(cache["usd_net_saving"]),
            ]
        )

    prefetch_rows = []
    for c in configs:
        pf = c["prefetch"]
        if not pf["launched"]:
            continue
        prefetch_rows.append(
            [
                f"<code>{html.escape(c['config']['name'])}</code>",
                str(pf["considered"]),
                str(pf["launched"]),
                f"{pf['hit_rate']:.0%}",
                f"{pf['ms_saved']:.0f}",
                f"{pf['ms_wasted']:.0f}",
                _fmt_usd(pf["usd_wasted"]),
            ]
        )

    tier_rows = [
        [
            f"<code>{html.escape(name)}</code>",
            f"<code>{html.escape(t['model'])}</code>",
            _fmt_usd(t["usd_per_mtok_in"] / 1),
            _fmt_usd(t["usd_per_mtok_cached_in"]),
            _fmt_usd(t["usd_per_mtok_out"]),
        ]
        for name, t in sorted(report["tiers"].items(), key=lambda kv: kv[1]["usd_per_mtok_in"])
    ]

    headline = ""
    if baseline and baseline["solved"]:
        best = min(
            (c for c in configs if c["solved"] and c["config"]["name"] != baseline_name),
            key=lambda c: c["usd_per_solved_task"],
            default=None,
        )
        if best:
            headline = (
                f"Baseline <code>{html.escape(baseline_name)}</code> solved "
                f"{baseline['solved']}/{baseline['n_tasks']} at {_fmt_usd(baseline['usd_per_solved_task'])} "
                f"per solved task. The cheapest configuration that solves anything is "
                f"<code>{html.escape(best['config']['name'])}</code> at "
                f"{_fmt_usd(best['usd_per_solved_task'])} per solved task, solving "
                f"{best['solved']}/{best['n_tasks']}."
            )

    notes = report.get("notes", {})
    note_items = "".join(
        f"<div class='note'><span>{html.escape(str(k))}</span><code>{html.escape(json.dumps(v, default=str))}</code></div>"
        for k, v in notes.items()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agent-router frontier</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #fbfbfa; --fg: #17171a; --mut: #6a6a72; --line: #dedeD8;
  --accent: #2f5d50; --accent2: #a8563c; --card: #ffffff;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #131316; --fg: #ececee; --mut: #9a9aa2; --line: #2c2c31;
           --accent: #7fc4ae; --accent2: #e39877; --card: #1a1a1e; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg); padding: 0 16px 64px;
  font: 14px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 860px; margin: 0 auto; }}
h1 {{ font-size: 21px; font-weight: 600; margin: 40px 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 15px; font-weight: 600; margin: 40px 0 8px; }}
p {{ color: var(--fg); margin: 8px 0; max-width: 68ch; }}
.sub {{ color: var(--mut); font-size: 13px; }}
code {{ font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }}
figure {{ margin: 16px 0 0; background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 8px 4px 0; overflow-x: auto; }}
svg {{ display: block; width: 100%; height: auto; min-width: 560px; }}
.axis {{ stroke: var(--line); stroke-width: 1; }}
.grid {{ stroke: var(--line); stroke-width: 1; stroke-dasharray: 2 4; opacity: .7; }}
.tick, .axis-title {{ fill: var(--mut); font-size: 10px; font-family: ui-monospace, monospace; }}
.axis-title {{ font-size: 10.5px; font-family: inherit; }}
.mid {{ text-anchor: middle; }} .end {{ text-anchor: end; }} .start {{ text-anchor: start; }}
.dot {{ fill: var(--mut); }}
.front-dot {{ fill: var(--accent); }}
.front {{ fill: none; stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 5 3; }}
.label {{ fill: var(--fg); font-size: 10px; font-family: ui-monospace, monospace; }}
.wrap {{ overflow-x: auto; margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; min-width: 520px; }}
caption {{ text-align: left; color: var(--mut); font-size: 12px; padding: 0 0 6px; }}
th, td {{ text-align: right; padding: 5px 9px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--mut); font-weight: 500; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .04em; }}
.note {{ display: flex; gap: 10px; justify-content: space-between; border-bottom: 1px solid var(--line);
  padding: 4px 0; font-size: 12.5px; }}
.note span {{ color: var(--mut); }}
</style></head>
<body><main>
<h1>Cost-aware execution frontier</h1>
<p class="sub">backend <code>{html.escape(report['backend'])}</code> &middot; suite
<code>{html.escape(report['suite'])}</code> &middot; prices as published
{html.escape(report['price_table_date'])} &middot; starred rows are on the Pareto front</p>
<p>{headline}</p>

<h2>Success against cost</h2>
<figure>{_scatter(configs, front)}</figure>
<p class="sub">Up and to the left is better. A point below and to the left of the baseline is a
cheaper system that solves fewer tasks, which is only an improvement if the tasks it drops
were worth less than the money it saved.</p>

<h2>Dollars per solved task</h2>
<div class="wrap">{_table(
    ["config", "solved", "success", "$/task", "$/solved", "vs baseline", "success delta", "p50 ms", "p95 ms", "escalations"],
    main_rows,
)}</div>

<h2>Where the cheap model actually fails</h2>
<div class="wrap">{_table(["config"] + band_names, band_rows,
    "Solved out of attempted, by difficulty band. If these columns look the same, the suite is not measuring routing.")}</div>

<h2>Prompt cache</h2>
<div class="wrap">{_table(
    ["config", "layout", "lookups", "hit rate", "cached tokens", "read saving", "storage cost", "net"],
    cache_rows or [["<em>no cache lookups recorded</em>"] + [""] * 7],
    "Storage is charged for the full TTL at creation, so the net column is the one that decides whether caching paid.",
)}</div>

<h2>Speculative prefetch</h2>
<div class="wrap">{_table(
    ["config", "considered", "launched", "hit rate", "ms saved", "ms wasted", "wasted spend"],
    prefetch_rows or [["<em>prefetch not enabled in any configuration</em>"] + [""] * 6],
)}</div>

<h2>Price table</h2>
<div class="wrap">{_table(["tier", "model", "in /Mtok", "cached in /Mtok", "out /Mtok"], tier_rows)}</div>

<h2>Run notes</h2>
{note_items or "<p class='sub'>none recorded</p>"}
</main></body></html>
"""


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(report))
    return p
