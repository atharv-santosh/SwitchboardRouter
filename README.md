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

Real Vertex AI, 20 tasks, against a single-strong-model baseline of `gemini-3.6-flash` that
solved 19 of 20 (95%) at 0.00864 USD per solved task, p50 10638 ms, p95 19857 ms. The
headline result: a learned escalation policy, fitted on simulated shadow data and never
retrained on a real model, matches that 19-of-20 success rate at 0.00351 USD per solved
task, 0.41x the baseline's cost. An oracle that escalates exactly when the cheap step is
wrong solves 18 of 20 (90%) at 0.62x, worse on both axes than the learned policy, because
some failures survive a perfect escalation decision. A paid verifier model solves 17 of 20
(85%) at 0.44x; free structural validation alone also solves 17 of 20 at 0.22x, which is
why dollars per solved task is never reported without the success rate next to it.

Caching the stable prefix cut real spend 59%, from 0.03607 to 0.01486 USD, at a 98.9% hit
rate. A deliberately naive layout that lets a step counter into the prefix still hit 90.3%
but needed 9 caches instead of 1 and spent 21% more for fewer served tokens. Speculative
prefetch, replayed against real recorded tool-call sequences, hit 57 to 72% (below the
93 to 100% a simulated stub gives, because two real models take different routes through
the same task) and still paid off at every latency tested once the tool itself is free;
the simulated finding that a fast, billed tool is where you should not speculate stands.

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
