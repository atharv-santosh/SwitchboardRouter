# Where the prices come from

Every dollar figure in this repository is a measured token count multiplied by a price
from `arouter/pricing.py`. The token counts are real; the prices are quoted from public
price lists, and that distinction matters when reading MEASUREMENTS.md. If a price here
is wrong, every dollar figure is wrong by the same factor, so the table is one place and
`arouter tiers` prints it.

Retrieved 2026-09-25. USD per 1M tokens, Vertex AI, on-demand.

| tier   | model                  | input | cached input | output |
|--------|------------------------|-------|--------------|--------|
| cheap  | gemini-2.5-flash-lite  | 0.10  | 0.01         | 0.40   |
| mid    | gemini-2.5-flash       | 0.30  | 0.03         | 2.50   |
| mid2   | gemini-3.5-flash-lite  | 0.30  | 0.03         | 2.50   |
| strong | gemini-3.6-flash       | 0.75  | 0.075        | 3.75   |

Cached input is 10% of input on every row, which is the ratio Google applies across these
models.

Context cache storage is charged at 1.00 per 1M tokens per hour for Flash-class models.
This repo charges it for the whole TTL at creation, because that is what is billed
whether or not the cache is read again.

gemini-3.6-flash is on introductory pricing through 2026-12-31. Standard pricing from
2027-01-01 is 1.50 input and 7.50 output, which doubles the baseline's cost and roughly
doubles every ratio reported against it. That is worth knowing before quoting a number
from this repo in 2027.

## Sources

The 2.5 rows and the cached-input column were taken from a published per-model rate table
and cross-checked against a second: <https://benchlm.ai/google/api-pricing>. The
gemini-3.6-flash introductory and standard rates were confirmed separately, as was the
gemini-3.5-flash-lite row. The cache storage rate is the Flash-class figure quoted in
Google Cloud pricing commentary rather than a line item this project read from Google's
own page directly.

## What is not verified

Google's own pricing page did not render its rate table to the fetcher used here, so none
of these rows was read from `cloud.google.com` directly. Two independent third-party
tables agree on every row, which is why the numbers are used, but the honest status is
corroborated rather than authoritative.

Vertex list prices also run above AI Studio for the same model, and per-region pricing can
differ. These runs used `location="global"`.

## Thinking tokens

Gemini 2.5 Flash, 3.5 Flash-Lite and 3.6 Flash return `thoughts_token_count` separately
from `candidates_token_count`. Thinking tokens are billed at the output rate, so
`Usage.usd` adds them to output rather than reporting them as a separate free line. On
this benchmark thinking is a large fraction of the strong tier's output tokens, and a cost
model that ignored it would understate the baseline substantially, which would flatter
every routing result in this repo.
