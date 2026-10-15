# Switchboard Router

Cost aware execution for agents, measured in dollars per solved task rather than
tokens saved.

Most deployments send every step to the strongest model. That gets you a demo that
works and a bill that does not survive production. Most steps do not need it, and the
problem is deciding which ones do without quietly trading away success. So the only
honest way to report it is success against cost, never cost alone.

A cheap model attempts each step and a policy decides whether that stands or escalates.
Four policies compared, plus an oracle that escalates exactly when the cheap action is
wrong, as a ceiling.

## Numbers

Real Vertex runs, 20 tasks, nine configurations:

```
single_strong (baseline)     19/20   $0.00864 per solved   p50 10638ms
cascade_learned_threshold    19/20   $0.00351 per solved   0.41x
cascade_oracle               18/20                         0.62x
cascade_verifier_model       17/20                         0.44x
cascade_schema_validation    17/20                         0.22x
```

The learned policy was fitted on simulated shadow data and never retrained on real
data. The oracle loses to it on both axes, because some failures survive a perfect
escalation decision. The verifier model pays a mid tier model on every step and acts on
verdicts that are wrong often enough to both escalate steps that were fine and accept
steps that were not. A paid verifier has to beat a free structural check by more than
its own price.

Schema validation costs 0.22x and solves 17. Dollars per solved task alone would rank
it near the top, which is why the success column sits next to it.

Cache aware layout hits 98.9% and cuts spend 59%. A naive layout still hits 90.3% but
needs nine caches instead of one and costs 21% more, so hit rate alone hides it.
Prefetch hits 57 to 72% on real recorded tool sequences against the simulated stub's 93
to 100%.

## Running it

```
uv pip install -e ".[gemini]"
arouter bench --learned-fit out/fit.json --out out/bench.json --html out/frontier.html
```

`arouter verify fixtures/real` re-derives the token counts and spend from the committed
trajectories. Routing logic is fully testable offline against a simulated model with
configurable per tier quality and latency.
