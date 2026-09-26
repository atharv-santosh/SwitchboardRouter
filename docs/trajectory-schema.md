# Trajectory schema

This is the wire format shared across the agent projects in this portfolio. A recorded
trajectory is a JSON Lines file: one event per line, each line independently parseable.
`arouter` writes it and replays from it, and a sibling project can read it without
knowing anything about routing.

## Event shape

```json
{"run_id": "cascade_oracle::api_replicas::seed0", "seq": 0, "ts": 1790000000.12, "type": "run_start", "payload": {}}
```

| field     | type   | meaning                                                          |
|-----------|--------|------------------------------------------------------------------|
| `run_id`  | string | Identifier shared by every event in one agent run.               |
| `seq`     | int    | Monotonic sequence number within the run, starting at 0.        |
| `ts`      | float  | Unix timestamp in seconds when the event was recorded.          |
| `type`    | string | One of the nine types below.                                     |
| `payload` | object | Event-specific, free-form, JSON-serializable.                     |

Events are strictly ordered by `seq` within a `run_id`. `ts` is not guaranteed monotonic
across a distributed run; `seq` always is. Here `run_id` is
`<config>::<task_id>::seed<n>`, so a whole frontier lands in one directory without
collisions.

## The nine types

- **`run_start`** carries `task_id`, `band`, `seed`, the full `config`, and the tenant.
- **`llm_call`** carries `tier`, `request_hash`, `prefix_hash`, `prompt_tokens_est` and
  whether a cache handle was attached. The request hash is over the canonical JSON of
  the prompt, tools, temperature and output budget, and is what replay matches on. The
  prefix hash is over the stable half alone, and is the cache key.
- **`llm_result`** carries `tier`, the chosen `action`, the full `usage` (input tokens,
  cached input tokens, output tokens, thinking tokens, latency, and the dollar figure
  those imply), and `cache_hit`.
- **`tool_call`** carries `{"request": {"tool": name, "args": {...}}}`.
- **`tool_result`** carries the tool's return value, its measured latency, and
  `prefetch_hit`: whether the result came from work started speculatively before the
  model committed to it.
- **`decision`** is where this project puts everything a cascade decides that is neither
  a model call nor a tool call. Its `kind` distinguishes them:
  `escalation` (a policy's accept or escalate verdict, with its score, reason and any
  cost the policy itself incurred), `escalation_suppressed` and `escalation_refused`
  (the budget would not allow it), `budget_degradation` (a rung of the ladder engaged),
  `budget_refusal` and `budget_stop`, and `shadow_label` (a cheap action labelled by
  whether the strong model agreed, with the feature vector used to fit the learned
  policy).
- **`human_gate`** is unused here; no step in this benchmark waits on a person. It stays
  in the vocabulary so a consumer can read every project's trajectories with one parser.
- **`error`** carries at least a `message`. Reaching the step limit without finishing is
  recorded this way rather than raised.
- **`run_end`** carries `solved`, the grader's `failures`, spend, critical-path latency,
  escalation count, the budget's final state and the policy's own statistics.

## Why the hash and not the raw request

Storing a hash alongside the request lets replay do an O(1) equality check instead of
deep-diffing a 2,500-token prompt on every step, while keeping the raw request for a
readable diff when the hashes differ. `arouter.models.replay.ReplayModel` raises
`ReplayMiss` on an unrecorded hash rather than falling back to a live call, because a
replay that quietly re-derives an answer is not a replay and the measurement it produces
is not the recorded one.

## Producing and consuming it

`arouter.events.TrajectoryEvent` is the canonical representation, `write_jsonl` and
`read_jsonl` handle serialization, and `iter_runs` walks a directory. Any project that
wants to hash a request the same way must use the same canonical encoding:

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
```

`arouter verify <dir>` re-derives the token counts and spend from committed fixtures,
which is the check that a fixture directory still means what its MEASUREMENTS.md rows
say it means.
