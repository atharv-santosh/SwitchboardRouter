# agent-router

Cost-aware execution for LLM agents, measured in dollars per solved task rather than tokens
saved.

Most agent deployments send every step to the strongest model available. That is how you get
a demo that works and a bill that does not survive production. Most steps do not need the
strong model, and the engineering problem is deciding which ones do without quietly trading
away task success. The only honest way to report the answer is success rate against cost,
never cost alone.

A cascade attempts each step on a cheap model and a pluggable policy decides whether that
answer stands or escalates. Four policies are compared: the cheap model's own reported
confidence, a separate verifier model, free structural validation of the proposed action,
and a logistic model fitted on recorded shadow runs. An oracle that escalates exactly when
the cheap action is wrong is included as a ceiling, so the gap between a real policy and the
best possible one is visible rather than implied.

Context is laid out so the cacheable prefix stays byte-identical across every step and every
task, which lets one prompt cache serve the whole workload. A deliberately naive layout is
measured beside it. When the next tool call is predictable it starts before the model has
finished deciding, and only read-only tools are ever speculated. Per-run and per-tenant
spend caps are enforced before each call rather than regretted after it, and a run that
reaches its cap walks down a degradation ladder instead of crashing or truncating.

## The numbers

Simulated, 20 tasks, against a single-strong-model baseline of `gemini-3.6-flash` that
solved 17 of 20 at 0.00450 USD per solved task, p50 8282 ms, p95 13274 ms.

The learned policy solved the same 17 of 20 at 0.00275 USD per solved task, 0.61x the
baseline, with p50 down 18%. The oracle reached 0.54x, so the learned policy captures about
83% of the available headroom. The verifier model was worse on both axes at once, 1.19x the
cost for 4 fewer tasks solved, because a paid verifier has to beat a free structural check by
more than its own price. Free structural validation alone reached 0.62x per solved task while
solving 7 tasks instead of 17, which is the result that looks like a win until the success
column sits next to it.

Caching the stable prefix cut spend 81%, from 0.02053 to 0.00388 USD. The naive layout still
hit 84.6% but needed 8 caches instead of 1 and spent 59% more for the same tokens.
Speculative prefetch saved wall clock at every tool latency tested and went net negative only
once the tool itself cost money and was fast: at 0.002 USD per call the crossing sits between
60 and 200 ms of tool latency.

Full tables, the real-model runs with model ids and dates, and what the first real run
changed about the benchmark are in `MEASUREMENTS.md`. `NOTES.md` is the build log.

## The benchmark

Twenty configuration-editing tasks, graded on exact final state with no partial credit,
deliberately built so the cheap tier lands in the middle. A suite it passes everywhere makes
every escalation a waste; one it fails everywhere makes every acceptance a loss; either way
the policies become indistinguishable. The first real run solved 11 of the original 14, too
easy to measure anything, so six harder tasks were added targeting what it actually got
wrong. CI fails the build if the cheap and strong tiers converge again.

## Running it

```
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
arouter shadow --out out/shadow --seeds 0,1,2,3,4
arouter fit out/shadow --out out/fit.json
arouter bench --learned-fit out/fit.json --out out/bench.json --html out/frontier.html
```

None of that needs credentials. `arouter cache-study`, `arouter prefetch-study` and
`arouter budget-demo` produce the other three tables, and `arouter tiers` prints the price
table every dollar figure derives from.

For real models, run `gcloud auth application-default login`, set `GOOGLE_CLOUD_PROJECT`,
install the `gemini` extra and pass `--backend vertex`. Recorded runs replay with
`--backend replay --fixtures fixtures/real/frontier`, so the real-model measurements
reproduce offline with no spend.

Trajectories are JSONL in the event schema shared with the sibling projects in this
portfolio, documented in `docs/trajectory-schema.md`.
