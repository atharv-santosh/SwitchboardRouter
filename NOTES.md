# Build log

Problems hit while building this, in the order they came up. These are the actual
failures, not a tidied narrative, and several of them changed the design.

## The build failed before a line of code ran

`uv pip install -e ".[dev]"` failed because `pyproject.toml` had
`readme = "README.md"` and I had not written the README yet, planning to write it last
once there were real numbers to put in it. Hatchling will not build a wheel, even an
editable one, when the `readme` target is missing. Stubbed all three markdown files
immediately and filled them at the end.

## Implicit prompt caching reported nothing, so the lever had to be explicit

The first design assumed Vertex's implicit caching: send the same prefix repeatedly and
read `cached_content_token_count` off the usage metadata. I probed it with an identical
26,320-token prefix sent three times in a row to each of gemini-2.5-flash-lite,
gemini-2.5-flash and gemini-3.6-flash. Every response came back with no
`cached_content_token_count` field at all, on all three models, on all three calls.
Latency did drop after the first call, which suggests something was cached somewhere, but
a saving I cannot measure is not a saving I can report.

So the whole cache lever moved to explicit `client.caches.create`, which returns a token
count and reports `cached_content_token_count` on every read. That is a worse deal in
principle, because explicit caches are billed for storage whether or not you read them
again, and it is the reason `CacheStats` reports a net figure rather than just a hit rate.
A hit rate of 98% on a cache nobody reused enough to cover its storage is a loss.

## Vertex rejects tools sent alongside cached content

First real benchmark run, immediate failure:

```
400 INVALID_ARGUMENT. Tool config, tools and system instruction should not be set
in the request when using cached content.
```

Tools belong to the cache object, not the request. The cache is created with the function
declarations and the forced-function-calling mode attached, and a cached call sends
neither. Obvious once you know, invisible until you try, and it means a cached agent and
an uncached agent do not build their requests the same way.

## The explicit cache floor is 1,024 tokens on 2.5 and 4,096 on 3.x

Next failure:

```
400 INVALID_ARGUMENT. The cached content is of 2210 tokens. The minimum token
count to start explicit caching is 4096.
```

The stable prefix was 2,210 tokens. That clears the floor on the 2.5 models and misses it
on gemini-3.6-flash. So a prefix can be cacheable on one rung of the ladder and not on the
next, which is not something I had considered and which makes the floor per-tier rather
than a constant. `Tier.min_cacheable_tokens` now carries it, and a refused create is
recorded in `CacheStats.create_failures` and degrades to no cache rather than killing the
run.

Then I expanded the operator manual with seven worked examples covering the task shapes
the cheap tier had actually failed, which is content I wanted anyway and which also took
the prefix over the floor. Being straight about the order of events: I wrote the examples
because the cheap tier needed the guidance, and I kept writing until the prefix cleared
4,096 because the cache needed that.

## The same text is 4,050 tokens to one tokenizer and 4,104 to the other

After expanding the manual, the created caches came back as 4,050 tokens on
gemini-2.5-flash-lite and gemini-2.5-flash, and 4,104 on gemini-3.5-flash-lite and
gemini-3.6-flash. Identical input. The 3.x floor is 4,096, so the two expensive tiers
clear it by 8 tokens.

My local character-based estimate said 3,628, which is 10% under the real count. That
mattered: the floor check was refusing to even offer the prefix to the provider, because
3,628 is below 4,096, so both 3.x tiers silently got no cache while the provider would
have accepted them. Fixed by only skipping locally when the estimate is well clear of the
floor, and otherwise letting the provider decide and recording its refusal. Second-guessing
a provider's tokenizer with a character heuristic is how you lose a lever without noticing.

## Reserving the true worst case made every cap unusable

The budget enforcer refuses a call in advance from an upper bound on its cost, because
checking afterwards is reporting rather than enforcement. The only hard bound is the full
`max_output_tokens` budget plus headroom for thinking tokens, which are billed as output.
With a 640-token budget that reserves about 0.006 USD per strong-tier call, roughly five
times what a single tool call actually emits. Every realistic cap therefore refused its
first call, the degradation ladder was unreachable, and `budget-demo` showed nothing but
immediate stops.

The default now reserves a calibrated 320 tokens and the consequence is stated rather than
hidden: a cap can be overshot by at most one call's cost. `strict_bound=True` restores the
hard guarantee for anyone who needs it. This is a real tradeoff and I would rather name it
than pretend the conservative version was usable.

## Two budget bugs the tests found, not the demo

The demo looked fine while both of these were live.

A cap of exactly zero, which is what an exhausted tenant ledger leaves behind, was read as
"no cap applies" instead of "the cap is already reached", so a tenant that had spent its
allowance got unlimited further spend. `fraction_used` returned 0.0 for a non-positive cap.

And a pre-flight refusal set the stop level directly instead of going through the
transition recorder, so a run that was refused its next call looked, in its own record,
like a run that had never degraded at all. The audit trail was wrong in exactly the case
where you would want it.

## The first real run made the suite look pointless

gemini-2.5-flash-lite solved 11 of the original 14 tasks: 4 of 4 easy, 5 of 6 medium, 2 of
4 hard, for 0.00062 USD per solved task. A cheap tier that clears 79% of the suite leaves a
cascade three tasks to argue over, and with three tasks of headroom every escalation
policy looks the same and none of the differences mean anything.

So I added six harder tasks aimed at what it actually got wrong, which was not what I had
guessed. It did the multi-file arithmetic fine. What it failed was aggregating across all
four services before deciding anything, ordering two edits whose inputs change under them,
and producing an exact output schema alongside the work. The new tasks are
`mean_footprint_cpu` (compute a mean, then compare each service to it), `two_constraints`
(one constraint fixes a value, the other is solved against the value you are about to
write), `order_dependent_edits` (re-reading between the two edits gives a plausible wrong
answer), `smallest_footprint_report`, `ignore_the_runbook`, and `already_compliant` (the
condition turns out false and the correct action is none).

The suite is now deliberately built so the cheap tier lands in the middle. That is a design
requirement, not an accident: a suite it passes everywhere makes every escalation a waste,
one it fails everywhere makes every acceptance a loss, and either way the policies become
indistinguishable. `benchmarks/suite.yaml` says so at the top and CI fails the build if the
cheap and strong tiers converge.

## The learned policy is much worse on real data, and for an interesting reason

On simulated runs the fitted policy reaches holdout AUC 0.903 with recall 0.774 on wrong
actions. On the real shadow run it reaches 0.790 and 0.375. Worse, but the feature ordering
is the more informative change. On simulated data `schema_invalid` dominates. On real data
it falls to fifth and `is_finish` dominates.

The reason is that my simulated model's failure modes include structurally impossible
actions such as bad paths and missing arguments, and a real gemini-2.5-flash-lite almost
never emits those. It emits a well-formed call with the wrong number in it, or it finishes
early having done some of the work. Those are exactly the failures a free structural check
cannot see, which is why the schema-validation policy looks much better offline than it
does against a real model. Had I only ever run the simulation I would have concluded that
free validation is most of what a cascade needs, and that conclusion would have been wrong.

## The naive layout did not do what I predicted

I expected the reordering layout to drive the cache hit rate to zero, and wrote the
docstring that way before measuring. It does not. It gets 81.1% against the cache-aware
layout's 97.3%, because the step counter in its prefix recurs across tasks: every task's
step 0 produces the same prefix as every other task's step 0. So it still caches, it just
needs one cache per step index instead of one for the whole workload, which is seven caches
rather than one, seven lots of storage, and 50% more spend for the same tokens.

That is a better finding than the one I expected. A hit rate that looks respectable can sit
on top of a cache population that costs more than it saves, which is the argument for
reporting the net figure rather than the hit rate.

## Prefetch coverage is capped by the read-only rule, not by the predictor

Speculation only ever fires on read-only tools, because a discarded write is not wasted
work, it is a corrupted workspace. That gate, not prediction accuracy, is what bounds
coverage: the predictor is offered a guess on every step but only about a third of steps
in this benchmark are reads, so coverage tops out near 37% while the hit rate on what it
does launch is 93 to 100%.

It also means the honest break-even analysis is not about the predictor at all. With a free
tool, speculation is a pure win and the only question is thread pressure. It stops paying
when the tool itself costs money: at 0.002 USD per call and latency valued at 0.002 USD per
second, the measured 92.9% hit rate clears the break-even at 200ms of tool latency and
falls below it at 60ms and 20ms. A fast, expensive tool is the case where you should not
speculate, and that is a statement about the tool, not the model.

## Things I would do differently

The simulated model is an oracle-driven stub, and its failure modes were my guesses about
how a cheap model fails. They were wrong in a way that flattered one policy. If I were
starting again I would record a real shadow run first, categorise the actual failures, and
only then write the simulation, so that the offline harness is calibrated against observed
behaviour rather than intuition.

I would also not report a cache hit rate anywhere without the net dollar figure next to it.
Separating them was the single easiest way to fool myself in this project.
