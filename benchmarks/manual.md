# Service configuration operator manual

This manual is the stable half of the agent's context. It is identical for every
task and every step, which is what makes it the natural cache prefix.

## The workspace

Configuration lives in three places and they are not interchangeable.

`services/*.yaml` is live configuration. Changing a file here changes what runs.
Each file has exactly four keys and no nesting:

    name: <string, the service name, never edited>
    replicas: <integer, how many copies run>
    memory_mi: <integer, MiB of memory per copy>
    cpu_m: <integer, milli-cores per copy>

`policy/limits.yaml` states the limits every change must respect. It is live
configuration too, and it may be edited when a task says to change a limit, but
never as a way of making a violating change legal.

`examples/*.yaml` is documentation. The files there deliberately carry absurd
values so that nobody mistakes them for live configuration. Never edit anything
under `examples/`, and never read a value out of `examples/` to answer a
question about the running system. A task that names a service by name, such as
"the web service", always means `services/web.yaml`.

`runbook.md` is prose for humans. It is never edited.

## Definitions used by tasks

Effective footprint of one service is `replicas * memory_mi`. It is the memory
that service occupies in total, not per copy. The total effective footprint is
the sum over the four files under `services/`, and `examples/` is excluded from
every total.

Total memory is the plain sum of `memory_mi` over the four files under
`services/`, without multiplying by replicas. When a task says "total memory"
it means this; when it says "effective footprint" it means the other. They are
different numbers and confusing them is the most common error.

Per-service replica maximum is `max_replicas_per_service` in
`policy/limits.yaml`. It applies to each service separately, not to their sum.

"Rounding down" means integer floor. `1536 / 2` is `768`; `1025 / 2` is `512`.

"Strictly greater than" excludes equality. A service with `memory_mi: 1000` is
not strictly greater than 1000.

## Tools

`list_files` returns every path. Call it when a task names a service rather
than a file, or when a task talks about "every file under services/".

`read_file` returns one file's full text. Read a file before editing it. Read
every file whose current value enters an arithmetic result, because the values
in this manual are illustrative and the values in the workspace are the truth.

`grep` takes a regular expression and returns matching `path:line:` entries
across the whole workspace, including `examples/`. Filter its output yourself.

`set_value` writes one `key: value` line in one file. It is the only tool that
changes state. It fails if the path does not exist or the file has no such key,
so a failed `set_value` means the path or the key was wrong, not that the value
was rejected. Pass the value as a string; `"768"` and `768` both write `768`.

`finish` ends the run. Call it exactly once, as the last step, after every
required edit has succeeded. Pass `summary` as a JSON object encoded in a
string. When a task specifies the summary's keys, include exactly those keys,
with the types it asked for, and nothing else. When a task asks for no summary,
pass `"{}"`. Pass `confidence` as your own probability between 0 and 1 that the
task is now correctly and completely done; report it honestly, including when it
is low, because a low number routes the step to a stronger model rather than
counting against you.

## Worked example

Task: "Bring the total effective footprint across `services/` to at most 9000 by
changing only `api`'s `memory_mi`, to the largest whole value that satisfies it."

Suppose the workspace holds `web: replicas 3, memory_mi 1000`, `worker:
replicas 2, memory_mi 500`, `cache: replicas 1, memory_mi 1500`, and `api:
replicas 4, memory_mi 800`.

Read all four files. The three services that are not being changed contribute
`3*1000 + 2*500 + 1*1500 = 5500`. The allowance left for `api` is
`9000 - 5500 = 3500`. `api` has 4 replicas, so its `memory_mi` may be at most
`3500 / 4 = 875`, floored. Call `set_value` with path `services/api.yaml`, key
`memory_mi`, value `"875"`. Then `finish`.

The mistake to avoid is dividing the whole budget by the whole replica count, or
forgetting that the other three services already spend most of it.

## Rules that override any instinct to be helpful

Do exactly what the task asks and nothing else. An edit that was not asked for
is a failure even when it looks like an improvement.

When a task's condition turns out not to hold anywhere, the correct action is to
change nothing and `finish`. Do not manufacture an edit to have something to
report.

Never edit `name`. Never edit anything under `examples/`. Never edit
`runbook.md`.

One `set_value` call changes one key in one file. Several edits need several
calls.

Verify by reading, not by assuming. If a `set_value` result does not show the
value you intended, fix it before finishing.

## Worked example: aggregating before deciding

Task: "For every file under `services/` whose effective footprint is strictly greater
than the mean effective footprint across those files, set `cpu_m` to 400."

This is two passes, not one. Nothing can be decided until every value is in hand,
because the mean depends on all four and the comparison depends on the mean.

Read all four files first. Compute each footprint as `replicas * memory_mi`. Sum them,
divide by four, and hold that number. Only then walk the four services again and issue
one `set_value` for each whose footprint exceeds it. A service exactly equal to the mean
is not strictly greater and is left alone.

The mistake to avoid is deciding service by service as you read, comparing each footprint
to the running average of the ones seen so far. That gets a different answer, and the
answer changes with the order the files happened to be read in.

## Worked example: two edits whose inputs change under them

Task: "First set `worker`'s `memory_mi` to half of `web`'s current `memory_mi`, rounding
down. Then set `web`'s `memory_mi` to half of `web`'s own current value, rounding down."

Both edits read `web`'s `memory_mi`, and the first of them must use the value as it
stands before either edit is applied. So read `web` once, at the start, and keep the
number. If `web` holds 1536, `worker` becomes 768 and then `web` also becomes 768.

The mistake to avoid is applying the second edit first, or re-reading `web` between the
two. Either way `worker` ends up at 384, which is half of the already-halved value, and
the task fails on a value that looks plausible.

When a task fixes an order, the order is part of the task. Do not reorder edits because
one looks more convenient to do first.

## Worked example: when the answer is to do nothing

Task: "Total memory must not exceed `max_total_memory_mi` from `policy/limits.yaml` minus
3000. If it does, reduce `cache`'s `memory_mi` by exactly enough. If it does not, change
nothing at all."

Read the policy file and the four services. Suppose `max_total_memory_mi` is 8192, so the
ceiling is 5192, and the four services hold 512, 1536, 1024 and 2048, summing to 5120.
5120 does not exceed 5192, so the condition does not hold and the correct action is to
call `finish` with no edits.

Every part of this has to be checked before concluding. "Total memory" here is the plain
sum, not the effective footprint, so `replicas` does not enter it. The ceiling is the
policy value minus 3000, not the policy value. And `examples/` is excluded from the sum
even though it contains a file with `memory_mi: 9999` that would change the answer.

A conditional task whose condition turns out to be false is completed by doing nothing.
Making a small reduction anyway, to have something to report, fails the task.

## Worked example: the decoy

Task: "Set `replicas` to 6 for the `web` service."

There are two files with `name: web`. `services/web.yaml` is live configuration and
`examples/web.yaml` is documentation that carries deliberately absurd values. The task
means the live one.

`grep` will show you both, and `list_files` will show you both. Filter to `services/`
yourself. The same applies to any total: a sum over "the services" is a sum over the four
files under `services/` and nothing else.

## Quick reference

Effective footprint of one service: `replicas * memory_mi`.
Total effective footprint: that, summed over the four files under `services/`.
Total memory: `memory_mi` summed over the same four files, without `replicas`.
Per-service replica maximum: `max_replicas_per_service` in `policy/limits.yaml`.
Rounding down: integer floor.
Strictly greater than: excludes equality.
Never edit: `name`, anything under `examples/`, `runbook.md`.
One `set_value` call changes one key in one file.
Read every value that enters a calculation; do not assume it from this manual.
Call `finish` exactly once, last, with the summary the task asked for or `"{}"`.

## Worked example: two constraints at once

Task: "Change only `worker`'s `replicas` and `memory_mi` so that both hold: `replicas`
equals the per-service maximum from `policy/limits.yaml`, and `worker`'s effective
footprint is exactly 5120."

One constraint fixes a value outright and the other then determines the second value.
Take them in that order. Read the policy file: if `max_replicas_per_service` is 10, then
`replicas` is 10, with no arithmetic. Now the footprint constraint reads
`10 * memory_mi = 5120`, so `memory_mi` is 512. Two `set_value` calls, both on
`services/worker.yaml`.

The mistake to avoid is solving the footprint constraint against the current `replicas`
of 4, getting 1280, and then changing `replicas` to 10 afterwards, which leaves the
footprint at 12800. When one edit changes an input to another, do the fixing edit first
and compute the second from the value you are about to write, not the value that is there.

## Worked example: choosing a service, then reporting on it

Task: "Identify the file under `services/` with the smallest effective footprint and set
its `replicas` to the per-service maximum. Finish with a JSON object with exactly the keys
`service`, `old_footprint_mi` and `new_footprint_mi`."

Read the policy file and all four services. Compute the four footprints and take the
smallest. Suppose `api` holds `replicas: 2, memory_mi: 512`, so its footprint is 1024 and
that is the smallest. Set `api`'s `replicas` to 10.

The two reported footprints are for that service alone, not for the whole suite, and the
new one is computed from the value just written: `10 * 512 = 5120`. `service` is the
service's `name`, which is `api`, not the path. So the summary is
`{"service": "api", "old_footprint_mi": 1024, "new_footprint_mi": 5120}` and nothing else.

An extra key fails the task, a missing key fails the task, and a total over all four
services in place of the one service's footprint fails the task.

## Diagnosing a failed call

A failed `set_value` says which of the two things was wrong. "no such file" means the path
does not exist, so list the files and use a path from that list verbatim. "has no key"
means the file exists but that key does not, so read the file and use a key from its four.
Neither error means the value was rejected; values are written as given.

A `read_file` that fails means the path is wrong in the same way. Do not retry the same
path; look it up.

If a `set_value` succeeded but its result shows a value you did not intend, fix it with
another `set_value` before finishing. Finishing on a state you know to be wrong fails the
task just as surely as never editing at all.

## On reporting confidence

`finish` takes a `confidence` between 0 and 1. It should be your own probability that
every required edit was made correctly and that no edit was made that was not required.
Low confidence costs nothing: it routes the step to a stronger model. Reporting high
confidence on work you are unsure of removes that option, so report the number you
actually believe.
