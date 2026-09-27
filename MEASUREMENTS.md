# Measurements

Every number here came out of running the code in this repository. Nothing is estimated,
extrapolated or interpolated. Token counts, cached-token counts and latencies are measured;
dollar figures are those counts multiplied by the price table in `docs/pricing-sources.md`,
which is quoted from published rates rather than read off an invoice.

Two kinds of run are reported and never mixed. Simulated runs use the deterministic
oracle-driven stub in `arouter/models/simulated.py` and need no credentials, so a fresh
virtualenv reproduces them exactly. Real-model runs called live Vertex AI Gemini; each one
names its models and its date, and every call is committed under `fixtures/` so the numbers
replay offline.

Reproduce the simulated numbers:

```
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
arouter shadow --out out/shadow --seeds 0,1,2,3,4
arouter fit out/shadow --out out/fit.json
arouter bench --learned-fit out/fit.json --out out/bench.json --html out/frontier.html
arouter cache-study --out out/cache-study.json
arouter prefetch-study --out out/prefetch-study.json
arouter budget-demo --out out/budget-demo.json
```

Reproduce the real-model numbers offline, with no credentials and no spend, from the
committed fixtures:

```
arouter verify fixtures/real
arouter merge fixtures/real/bench-singles.json fixtures/real/bench-cascade_*.json \
  --out out/real-frontier.json --html out/real-frontier.html
arouter prefetch-study --backend replay --fixtures fixtures/real/frontier \
  --replay-config single_cheap --fit-on single_mid --out out/prefetch-real.json
```

## Test suite

```
$ pytest -q
189 passed, 7 skipped in 4.01s
```

189 tests pass with no credentials present. The 7 skips are the live Vertex tests in
`tests/test_vertex.py`, which skip on a missing `GOOGLE_CLOUD_PROJECT`. `mypy src/arouter`
reports "Success: no issues found in 21 source files".

## The suite is built so the cheap tier lands in the middle

This is a design requirement, not an observation. A suite the cheap tier passes everywhere
makes every escalation a waste; one it fails everywhere makes every acceptance a loss; in
either case the escalation policies become indistinguishable and the cascade measures
nothing. The 20 tasks are graded on exact final workspace state with no partial credit.

Simulated, seed 0, 20 tasks, solved out of attempted by difficulty band:

| tier                            | easy | medium | hard  | total |
|---------------------------------|------|--------|-------|-------|
| cheap (gemini-2.5-flash-lite)   | 2/4  | 2/6    | 2/10  | 6/20  |
| mid (gemini-2.5-flash)          | 4/4  | 4/6    | 3/10  | 11/20 |
| mid2 (gemini-3.5-flash-lite)    | 4/4  | 5/6    | 4/10  | 13/20 |
| strong (gemini-3.6-flash)       | 4/4  | 6/6    | 7/10  | 17/20 |

The first real run showed the same requirement being violated and is why the suite has 20
tasks rather than 14. See the real-model section below.

## Simulated frontier

Backend `simulated`, seed 0, 20 tasks, prompt caching on, prefetch off, no budget cap.
Baseline is `single_strong`, one model for every step. Latency is critical-path: model
latency plus tool latency, minus any overlap prefetch won.

| config                      | solved | success | $/task  | $/solved | vs baseline | success delta | p50 ms | p95 ms | escalations |
|-----------------------------|--------|---------|---------|----------|-------------|---------------|--------|--------|-------------|
| single_cheap                | 6/20   | 30%     | 0.00019 | 0.00065  | 0.14x       | -55%          | 1195   | 4810   | 0           |
| cascade_schema_validation   | 7/20   | 35%     | 0.00098 | 0.00281  | 0.62x       | -50%          | 2099   | 11523  | 16          |
| cascade_self_confidence     | 12/20  | 60%     | 0.00163 | 0.00272  | 0.60x       | -25%          | 5309   | 10058  | 32          |
| single_mid                  | 11/20  | 55%     | 0.00165 | 0.00300  | 0.67x       | -30%          | 3755   | 9130   | 0           |
| single_mid2                 | 13/20  | 65%     | 0.00169 | 0.00260  | 0.58x       | -20%          | 3198   | 6313   | 0           |
| cascade_oracle              | 19/20  | 95%     | 0.00229 | 0.00241  | 0.54x       | +10%          | 7260   | 13068  | 46          |
| cascade_learned_threshold   | 17/20  | 85%     | 0.00234 | 0.00275  | 0.61x       | 0%            | 6752   | 12800  | 47          |
| cascade_verifier_model      | 13/20  | 65%     | 0.00347 | 0.00533  | 1.19x       | -20%          | 8758   | 19468  | 43          |
| single_strong (baseline)    | 17/20  | 85%     | 0.00382 | 0.00450  | 1.00x       | 0%            | 8282   | 13274  | 0           |

The one row that is an unambiguous win is `cascade_learned_threshold`: the same 17 of 20
tasks as the baseline, at 0.61x the cost per solved task, with p50 down 18% and p95 down 4%.
The oracle shows 0.54x is the ceiling for this ladder, so the learned policy captures about
83% of the available headroom.

The rows that look like wins and are not matter more. `cascade_schema_validation` costs
0.62x per solved task, which reads well until you see it solves 7 tasks instead of 17.
`single_cheap` costs 0.14x per solved task and solves 6. Dollars per solved task alone would
rank both above the baseline, which is why the success column is next to it and why the
frontier is plotted rather than tabulated.

`cascade_verifier_model` is worse than the baseline on both axes at once: 1.19x the cost per
solved task for 4 fewer tasks solved. It pays a mid-tier model on every step and then acts
on its verdict, and its verdicts are wrong often enough to both escalate steps that were
fine and accept steps that were not. A paid verifier has to beat a free structural check by
more than its own price, and here it does not.

Pareto front as reported: `single_cheap`, `single_mid2`, `cascade_schema_validation`,
`cascade_self_confidence`, `cascade_oracle`. `cascade_learned_threshold` is off it only
because the oracle dominates it, and the oracle is a ceiling rather than something you can
deploy. Excluding the oracle, the learned policy is on the front and dominates the baseline.

## Learned escalation policy

Fitted on 485 labelled steps from 5 simulated shadow seeds. Shadow mode calls both tiers on
every step, executes the strong action, and labels the cheap one by whether the strong model
agreed. The split is by task, not by row, because adjacent steps of one run share almost
everything and a row split leaks.

```
485 labelled steps, base rate cheap-was-wrong 42.7%
train n=321  holdout n=164  threshold 0.44
  train    acc 0.891  auc 0.923  recall-on-wrong 0.813  false-escalation 0.061
  holdout  acc 0.933  auc 0.960  recall-on-wrong 0.917  false-escalation 0.050
```

Feature importance, weight times feature standard deviation:

```
schema_invalid                   +1.8458
is_finish                        +1.0855
confidence                       -0.4861
is_write                         +0.4124
n_prior_writes                   -0.1780
step_frac                        -0.1562
empty_summary_when_required      -0.0836
prompt_ktokens                   +0.0083
```

Self-reported confidence carries real signal in the right direction but is fourth by
magnitude and far behind structure. The simulated model is deliberately built with imperfect
calibration, and the confidence-only policy's 60% success against the learned policy's 85%
at similar cost is the consequence.

## Prompt cache

Simulated, cheap tier, 20 tasks, 900-second TTL, one stable prefix shared by every task.
The task statement is deliberately kept out of the prefix: putting it in would give a cache
only one task can use, and leaving it out gives one cache per tier that the whole workload
shares.

| layout      | cache | hit rate | caches created | cached tokens served | net saving | total spend |
|-------------|-------|----------|----------------|----------------------|------------|-------------|
| cache_aware | on    | 98.1%    | 1              | 202,011              | +0.01719   | 0.00388     |
| naive       | on    | 84.6%    | 8              | 174,372              | +0.00777   | 0.00616     |
| cache_aware | off   | 0%       | 0              | 0                    | 0          | 0.02053     |

Caching the stable prefix cut total spend by 81%, from 0.02053 to 0.00388. The naive layout,
which rotates the stable blocks as the run proceeds and lets a step counter into the prefix,
spends 59% more than the cache-aware one for exactly the same tokens.

The naive layout does not lose the cache entirely, which is not what I expected. Its step
counter recurs across tasks, so every task's step 0 shares a prefix with every other task's
step 0, and it still hits 84.6%. What it loses is the cache population: 8 caches rather than
1, so 8 lots of storage charged for the full TTL, and each one reused less. A respectable
hit rate can sit on top of a cache population that costs more than it saves, which is the
argument for reporting the net figure next to the hit rate and never the hit rate alone.

## Speculative prefetch

Simulated, cheap tier, 20 tasks, predictor fitted on a prior run of the same suite. Only
read-only tools are ever speculated; a discarded write would corrupt the workspace, so the
gate is hard rather than probabilistic. That gate, not prediction accuracy, bounds coverage:
about a third of steps in this benchmark are reads, so coverage tops out near 37%.

Free tool, latency valued at 0.002 USD per second:

| tool latency | threshold | launched | hit rate | ms saved | ms wasted | net USD   | break-even |
|--------------|-----------|----------|----------|----------|-----------|-----------|------------|
| 40 ms        | 0.30      | 14       | 92.9%    | 561      | 44        | +0.000922 | 71.4%      |
| 40 ms        | 0.70      | 12       | 100%     | 523      | 0         | +0.001046 | 71.4%      |
| 120 ms       | 0.30      | 14       | 92.9%    | 1616     | 121       | +0.003032 | 45.5%      |
| 400 ms       | 0.30      | 14       | 92.9%    | 5250     | 400       | +0.010301 | 20.0%      |
| 1200 ms      | 0.30      | 14       | 92.9%    | 8483     | 1205      | +0.016766 | 7.7%       |

With a tool that costs nothing, speculation is a pure win and the only remaining question is
thread pressure. It stops paying when the tool itself costs money. At 0.002 USD per tool
call, threshold 0.30, and the same valuation of latency:

| tool latency | hit rate | ms saved | net USD   | break-even | pays off |
|--------------|----------|----------|-----------|------------|----------|
| 20 ms        | 92.9%    | 313      | -0.001374 | 98.0%      | no       |
| 60 ms        | 92.9%    | 821      | -0.000358 | 94.3%      | no       |
| 200 ms       | 92.9%    | 2703     | +0.003406 | 83.3%      | yes      |
| 600 ms       | 92.9%    | 7619     | +0.013238 | 62.5%      | yes      |

The crossing is between 60 and 200 ms of tool latency. A fast, expensive tool is the case
where you should not speculate, and that is a statement about the tool rather than about the
model or the predictor. Raising the threshold to 0.70 removes every miss on this suite and
so pays at every latency, at the cost of 2 fewer launches out of 14.

## Budget enforcement

Simulated, cheap-to-strong cascade with schema validation, 20 tasks. Caps are enforced
before each call from an upper-bound estimate, so no run exceeds its cap in any row below.

| run cap  | solved | total spend | escalations | runs stopped | runs over cap | degradation rungs engaged                              |
|----------|--------|-------------|-------------|--------------|---------------|--------------------------------------------------------|
| 1.00     | 7/20   | 0.07632     | 16          | 0            | 0             | none                                                   |
| 0.012    | 7/20   | 0.06858     | 14          | 1            | 0             | NO_ESCALATION 2, NO_VERIFIER 1, SHRINK_CONTEXT 1, STOP 1 |
| 0.008    | 7/20   | 0.05188     | 9           | 0            | 0             | NO_ESCALATION 2                                        |
| 0.006    | 7/20   | 0.04782     | 8           | 2            | 0             | NO_ESCALATION 4, NO_VERIFIER 2, SHRINK_CONTEXT 2, STOP 2 |
| 0.004    | 6/20   | 0.02053     | 0           | 0            | 0             | NO_ESCALATION 1                                        |

Tightening the cap from unlimited to 0.004 cut spend by 73% and cost one solved task. At
0.004 no escalation is ever affordable, so the cascade degrades to cheap-only rather than
failing. A shared tenant cap of 0.02 USD across all 20 runs stopped at 0.01980 spent, with
9 runs degrading to STOP; each of those finished with a `budget_exhausted` summary and a
recorded `budget_refusal` decision event rather than raising or truncating silently.

The caveat is stated in `arouter/budget.py` and worth repeating: the default pre-flight
reserve is a calibrated p95 of observed output rather than the full output budget, so a cap
can be overshot by at most one call's cost. Reserving the true worst case makes any
realistic cap refuse its own first call. `strict_bound=True` restores the hard guarantee.

## Real-model runs

All real runs used Vertex AI, project `research-501307`, location `global`, ADC, temperature
0, `max_output_tokens` 640, `max_steps` 10, explicit prompt caching on with a 900-second TTL.
Every call is committed as a JSONL trajectory under `fixtures/real/`, and
`arouter verify fixtures/real` re-derives the token counts and spend from them.

### First real run, 2026-09-25, and why the suite changed

`gemini-2.5-flash-lite`, one model for every step, on the original 14-task suite: solved
11/14 (easy 4/4, medium 5/6, hard 2/4) for 0.00048 USD per task and 0.00062 USD per solved
task, p50 3326 ms, p95 11632 ms, 53 calls, 121,467 input tokens of which 113,731 were served
from cache.

A cheap tier that solves 79% of the suite leaves a cascade three tasks of headroom, and with
three tasks of headroom no escalation policy can be distinguished from another. That run is
the reason the suite has 20 tasks. The six added tasks target what it actually got wrong,
which was not what I had guessed: it handled multi-file arithmetic and failed at aggregating
across all four services before deciding, at ordering two edits whose inputs change under
them, and at producing an exact output schema alongside the work.

The same run produced the first real cache measurement: 98.1% hit rate, one cache created,
113,731 of 121,467 input tokens served from cache, 0.01009 USD saved on reads against
0.00054 USD of storage, for a net 0.00955 USD on a run that cost 0.00669 USD in total. Read
at full input price those tokens would have cost about 2.5 times as much.

### Explicit cache token counts, 2026-09-26

The stable prefix, created as an explicit cache on each tier:

| tier   | model                 | cache tokens | floor | cached |
|--------|-----------------------|--------------|-------|--------|
| cheap  | gemini-2.5-flash-lite | 4050         | 1024  | yes    |
| mid    | gemini-2.5-flash      | 4050         | 1024  | yes    |
| mid2   | gemini-3.5-flash-lite | 4104         | 4096  | yes    |
| strong | gemini-3.6-flash      | 4104         | 4096  | yes    |

Identical input text. The 2.5 and 3.x tokenizers disagree by 54 tokens on 4,050, and the 3.x
explicit-caching floor is 4,096, so the two most expensive tiers clear their floor by 8
tokens. Before the manual was expanded the prefix was 2,210 tokens and Vertex refused to
cache it on `gemini-3.6-flash` with "the minimum token count to start explicit caching is
4096". The lever is real but it has a size threshold that differs by model generation.

### Real shadow run, 2026-09-26

`gemini-2.5-flash-lite` shadowed by `gemini-3.6-flash`, both tiers called on every step,
20 tasks, seed 0: 105 labelled steps, 0.18237 USD spent. The cheap tier disagreed with the
strong one on 36.2% of steps.

```
105 labelled steps, base rate cheap-was-wrong 36.2%
train n=72  holdout n=33  threshold 0.46
  train    acc 0.750  auc 0.790  recall-on-wrong 0.633  false-escalation 0.167
  holdout  acc 0.758  auc 0.790  recall-on-wrong 0.375  false-escalation 0.120
```

Feature importance on real data, next to the simulated ordering:

```
real                              simulated
is_finish            +1.3401      schema_invalid       +1.8458
step_frac            -0.5974      is_finish            +1.0855
is_write             +0.3708      confidence           -0.4861
confidence           -0.3405      is_write             +0.4124
schema_invalid       +0.2588      n_prior_writes       -0.1780
```

The learned policy is weaker on real data, holdout AUC 0.790 against 0.960, and the reason
is the more useful finding. My simulated model's failure modes include structurally
impossible actions, and a real `gemini-2.5-flash-lite` almost never emits one. It emits a
well-formed call with the wrong number in it, or finishes early having done part of the work.
`schema_invalid` therefore falls from first to fifth. Had I only run the simulation I would
have concluded that free structural validation is most of what a cascade needs, and that
conclusion would have been wrong.

### Real frontier, all nine configs, 2026-09-26

All 20 tasks, every configuration, against Vertex. `arouter verify fixtures/real` replays
every trajectory below to matching token counts and spend; the table itself comes from
`arouter merge fixtures/real/bench-singles.json fixtures/real/bench-cascade_*.json`, and the
same numbers reproduce with `arouter bench --backend replay --fixtures fixtures/real/frontier
--learned-fit fixtures/real/fit.json`.

| config                      | solved | success | $/task  | $/solved | vs baseline | p50 ms | p95 ms | escalations |
|-----------------------------|--------|---------|---------|----------|-------------|--------|--------|-------------|
| single_cheap                | 15/20  | 75%     | 0.00074 | 0.00099  | 0.11x       | 6652   | 13632  | 0           |
| cascade_schema_validation   | 17/20  | 85%     | 0.00159 | 0.00187  | 0.22x       | 6928   | 15255  | 4           |
| single_mid2                 | 16/20  | 80%     | 0.00168 | 0.00210  | 0.24x       | 4545   | 10170  | 0           |
| single_mid                  | 17/20  | 85%     | 0.00306 | 0.00360  | 0.42x       | 6902   | 13284  | 0           |
| cascade_verifier_model      | 17/20  | 85%     | 0.00320 | 0.00376  | 0.44x       | 12549  | 25368  | 4           |
| cascade_learned_threshold   | 19/20  | 95%     | 0.00333 | 0.00351  | 0.41x       | 9672   | 25556  | 27          |
| cascade_oracle              | 18/20  | 90%     | 0.00486 | 0.00540  | 0.62x       | 12294  | 27486  | 51          |
| cascade_self_confidence     | 18/20  | 90%     | 0.00734 | 0.00816  | 0.94x       | 15892  | 32215  | 84          |
| single_strong (baseline)    | 19/20  | 95%     | 0.00821 | 0.00864  | 1.00x       | 10638  | 19857  | 0           |

Pareto front: `single_cheap`, `cascade_schema_validation`, `cascade_learned_threshold`.
`cascade_learned_threshold` matches the baseline's success rate exactly, 19 of 20, at 0.41x
the cost per solved task, which is the headline number for this project: a policy fitted on
485 simulated shadow steps and never retrained on real data still recovers the baseline's
success rate at 41% of its cost against a live model it was not tuned on.

`cascade_oracle` is not the ceiling here that it is in simulation. It escalates exactly when
the cheap step disagrees with the strong one, but a task can still fail for reasons the
escalation decision cannot fix: a wrong strong-model step, or the run exhausting
`max_steps`. That is a real finding rather than a modelling artefact: an oracle over one
decision is a ceiling on that decision, not on the task.

`cascade_verifier_model` needed a genuine bug fix before it would replay at all. Its
verifier factory was constructed with `run_id=f"{config.name}::"`, missing the task id and
seed that every recorded run actually carries, so replaying it against the full fixture
directory always missed the lookup. The fix threads the real run_id through
`verifier_factory` from `CascadeRunner.run` down to every call site (`src/arouter/cascade.py`,
`bench.py`, `cli.py`); nothing about the escalation logic changed, and the numbers above are
what it already was.

This was found while assembling this exact frontier, which is the point of replaying
committed fixtures in CI rather than only trusting the numbers each fixture was recorded
with.

### Real speculative prefetch, 2026-09-26

Replayed the recorded `single_cheap` trajectories under `fixtures/real/frontier` with real
Vertex tool-call sequences: the predictor is fit on the recorded `single_mid` run and
evaluated on `single_cheap`, so it never sees the run it predicts. Free tool, latency valued
at 0.002 USD per second, launched out of 89 real tool calls:

| tool latency | threshold | launched | hit rate | ms saved | ms wasted | net USD   | break-even |
|--------------|-----------|----------|----------|----------|-----------|-----------|------------|
| 40 ms        | 0.30      | 49       | 57.1%    | 1216     | 906       | -0.001767 | 71.4%      |
| 40 ms        | 0.70      | 32       | 71.9%    | 999      | 386       | +0.000198 | 71.4%      |
| 120 ms       | 0.30      | 49       | 57.1%    | 3443     | 2592      | +0.002686 | 45.5%      |
| 120 ms       | 0.70      | 32       | 71.9%    | 2844     | 1112      | +0.003887 | 45.5%      |
| 400 ms       | 0.30      | 49       | 57.1%    | 11307    | 8467      | +0.018414 | 20.0%      |
| 400 ms       | 0.70      | 32       | 71.9%    | 9286     | 3626      | +0.016772 | 20.0%      |
| 1200 ms      | 0.30      | 49       | 57.1%    | 31797    | 25287     | +0.059395 | 7.7%       |
| 1200 ms      | 0.70      | 32       | 71.9%    | 25867    | 10829     | +0.049935 | 7.7%       |

The real hit rate, 57 to 72%, is well below the simulated stub's 93 to 100%. Two different
real models genuinely take different routes through the same task, so a predictor fit on
`gemini-2.5-flash` traffic mispredicts `gemini-2.5-flash-lite`'s next call more often than
the simulation's shared oracle logic ever would. Prefetch still pays at every latency
tested here once the threshold is raised to 0.70, because a free tool means only wasted
compute is at risk, never wasted spend; the simulated finding that speculation stops paying
once the tool itself costs money is unchanged by this, since none of these real tool calls
are billed.

### Real cache study on the full 20-task suite, 2026-09-26

The 14-task cache measurement above predates the six added tasks. Re-run on the current
20-task suite, cheap tier, 900-second TTL:

| layout      | cache | hit rate | caches created | cached tokens served | net saving | total spend |
|-------------|-------|----------|----------------|-----------------------|------------|--------------|
| cache_aware | on    | 98.9%    | 1              | 356,400               | +0.03106   | 0.01486      |
| naive       | on    | 90.3%    | 9              | 340,704               | +0.02154   | 0.01792      |
| cache_aware | off   | 0%       | 0              | 0                     | 0          | 0.03607      |

Caching the stable prefix cuts real spend by 59%, from 0.03607 to 0.01486 USD. The naive
layout still finds a real cache 90.3% of the time, for the same reason the simulated one
does: the step counter it lets into the prefix recurs across tasks. It needs 9 caches
instead of 1 and spends 21% more than the cache-aware layout for a lower hit rate, matching
the simulated and first-real-run findings that a respectable hit rate is not the same claim
as a net saving.
