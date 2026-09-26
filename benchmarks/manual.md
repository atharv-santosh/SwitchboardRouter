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
