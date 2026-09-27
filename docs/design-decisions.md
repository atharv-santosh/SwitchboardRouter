# Design decisions

The choices in this repository that took an argument, and what the argument was.

## Dollars per solved task, not cost reduction

A routing system that cuts spend 60% and drops success from 90% to 70% is a worse system,
and cost reduction in isolation cannot say so. Dividing spend by tasks actually solved can,
because a configuration that solves fewer tasks pays for its saving in the denominator.

It is still not sufficient on its own. A configuration that solves one task very cheaply
has an excellent dollars-per-solved-task figure, which is why the frontier is plotted with
success rate on one axis, why the success column sits next to the cost column in every
table, and why `usd_per_solved_task` is infinite rather than zero when nothing was solved.
`arouter.bench.ConfigResult.usd_per_solved_task` returns `float("inf")` deliberately.

## Single-model rows for every tier are part of the comparison

A cascade has to beat the best single model, not merely the strongest one. It is easy to
show a cheap-to-strong cascade beating the strong model on cost and quietly omit the middle
tier, which may already do the job for less. Here `single_mid2` solves 13 of 20 at 0.00260
USD per solved task, better than three of the four cascades, and leaving it off the chart
would have hidden that.

## An oracle policy that cannot be deployed

`OraclePolicy` escalates exactly when the cheap action differs from the gold action. It is
not a system anyone can run, because it needs the answer. It is on the frontier because
every real policy is a lossy approximation of it, so the distance between a policy and the
oracle is the honest measure of remaining headroom. Without it a 0.61x result has nothing
to be judged against and could be either most of the available win or a third of it.

The consequence is that the oracle sometimes dominates a deployable policy and pushes it
off the reported Pareto front. MEASUREMENTS.md says so where it happens rather than quietly
excluding the oracle to make the table read better.

## The task statement goes in the volatile half of the prompt

It is constant within a run, so the obvious thing is to put it in the cacheable prefix.
That gives a cache only one task can use, amortised over four or five steps. Leaving it out
gives one cache per tier that every task in the workload shares, amortised over every step
of every run inside the TTL. The measured difference is 1 cache created rather than 20, and
since explicit caches are billed for storage over their whole TTL, that difference is most
of the saving.

The general rule this encodes: the prefix should contain what is invariant across your
workload, not what is invariant across one request. Those are different sets and the second
is much less valuable.

## Storage charged for the whole TTL at creation

`PrefixCache` charges the full TTL the moment a cache is created, not the fraction actually
used. That is what the provider bills, and modelling it any other way would make caching
look unconditionally good. It is the reason `CacheStats` reports a net figure, and the
reason a test asserts that a cache reused only twice shows a net loss. A hit rate without
a net figure beside it is the easiest number in this whole project to fool yourself with.

## Only read-only tools are speculated, as a hard gate

The prefetcher is allowed to be wrong; that is what makes it speculation. A wrong guess on
a read is discarded work. A wrong guess on a write is a corrupted workspace, and no latency
win justifies it. So the check is membership in `READ_ONLY_TOOLS`, applied before the work
is submitted, rather than a confidence threshold tuned high enough to feel safe.

Speculation also runs against a snapshot of the workspace rather than the live one, so an
in-flight guess can never observe a half-applied state.

## The predictor is fitted on other runs, and reports both key schemes

Keying an n-gram on the task id, over a 20-task suite, is close to memorisation, and a hit
rate produced that way would flatter the technique. The fit therefore excludes the run it
serves, and a task-agnostic key is fitted alongside the task-specific one. The
task-agnostic key can only predict argument-free tools, because without a task there is no
basis for guessing a path, and that limitation is the honest picture for a workload you
have not seen before.

## Caps refused in advance, from an estimate that is not the worst case

Checking spend after a call is reporting, not enforcement, and a cap one large call can
blow through is not a cap. So `BudgetEnforcer.can_afford` runs before every call.

The only hard upper bound on a call is the full `max_output_tokens` budget plus headroom
for thinking tokens, which are billed as output. With a 640-token budget that reserves
about five times what a single tool call actually emits, so every realistic cap refuses its
own first call and the degradation ladder is unreachable. The default therefore reserves a
calibrated p95 and accepts a bounded overshoot of at most one call's cost, with
`strict_bound=True` available for callers who need the guarantee instead. This is a real
tradeoff; naming it is better than shipping a conservative version that does not work or a
permissive one that pretends to be exact.

## The degradation ladder's order

Escalation goes first because losing it costs accuracy on hard steps but still answers.
The verifier goes second because it is what decides escalation, so dropping it earlier
would be incoherent. Context shrinking goes third because it is the only rung that changes
what the model was told, and it drops only blocks marked optional and only from the end of
the stable region, so the cached prefix survives. Degrading a run should not also destroy
its cache hit rate. Stopping is last, and a stopped run finishes with an explicit
`budget_exhausted` summary and a recorded decision event, never a silent truncation.

## The simulated model is an oracle-driven stub, and says so

It is not a model. The suite hands it the correct action and a per-tier skill parameter
decides whether it returns that action or a corrupted one. Its only job is making the
routing logic testable with no credentials.

Its corruption modes are split into ones a structural validator can see and ones it cannot,
because a simulation where schema validation catches every failure would make that policy
look like the answer. It did anyway: `schema_invalid` is the top feature on simulated data
and fifth on real data, because a real cheap model rarely emits a malformed call and
instead returns a well-formed call with a wrong number in it. The simulation is useful for
testing logic and unreliable for ranking policies, and MEASUREMENTS.md keeps the two kinds
of run in separate sections for that reason.

Difficulty is calibrated at the task level and converted to a per-step probability by the
plan-length root. Setting it per step instead makes an eight-step task nearly impossible
for every tier regardless of its difficulty, which is an artefact of the simulation rather
than anything about routing.

## Replay refuses rather than falling back

`ReplayModel` raises `ReplayMiss` when a request hash is not in the fixtures. The
alternative, calling the live model or returning something plausible, would produce a
number that looks like a replayed measurement and is not one. A miss means the prompt
changed since recording, and the correct response is to re-record.
