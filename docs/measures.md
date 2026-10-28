# Measures

Measures use the sessions a query selects. Every number reaches CLI, MCP and
HTTP through the command table. Time and comparison page sessions; facets page
family/value rows through the ordinary query cursor. `--limit` chooses a
page, and `--cursor` continues it without a result cap.

```sh
polylogue 'sessions where repo:morrow | measure time'
polylogue 'sessions where repo:morrow | measure compare:origin,words,wall_span,ends_with'
polylogue 'sessions where repo:morrow | measure facets:deep'
polylogue measure outlook --option example:trailing-7d-mean
polylogue 'sessions | agg p50:wall_span, p90:wall_span, max:wall_span'
polylogue explain derived
```

[Derived data](derived.md) is generated from the declarations. It names each
product's catalog entry, validity kind, inputs, unknown rules, parameters and
executable evidence. `explain derived` serves those declarations; `explain
fields --unit sessions` includes each derived field's declaration.

## Session span and time lanes (K15, K16)

`session_span` keeps the earliest and latest own message times, or event times
when there are no messages. It never uses import time. `wall_span` is their
difference; a session with no timed item has an unknown span. An untimed
message stays untimed.

`clustered_time` is the union of consecutive shown active-path message
intervals at most **5 minutes** apart, extended across paired tool calls to
their results. Fewer than two timed messages leaves it unknown. Reversed
gaps are counted. On Morrow #2 it is 62 minutes with the 5-minute gap and
75 minutes with a 10-minute gap; wall span is 75 minutes. Clustering does
not measure attention.

`tool_time` sums positive paired tool intervals; `tool_time_union` takes
their union. `tool_overlap_ms` is the difference. Missing, equal and reversed
timestamps are unknown durations, counted separately in `tool_untimed`.
Tool-time aggregates include `unmeasured_calls_FIELD` beside the known sum;
default comparisons include the unmeasured-call count. Recorded absence of
calls has an empty sum; this says nothing about tools an origin does not
record. Harness tool-result messages never become human prompts.

`session_time(session_pk, lane, key, ms)` divides every millisecond of the wall
span between tools, waiting for the model, waiting for the human and
unattributed time. A tool lane is keyed by normalized command for shell
calls, action for others. Tools take priority over waiting intervals;
overlapping tools share their union equally, with deterministic integer
rounding. A human prose prompt or tool result starts model waiting; a model
message ending a turn starts human waiting. Missing or reversed transitions
remain unattributed. The lanes sum exactly to wall span.

`ends_with` describes the last recorded item: `human_prompt`, `human_command`,
`model_text`, `tool_call_open`, `tool_result`, `harness`, or `unknown`.
`last_stop_reason` is the source's recorded last model stop reason. Neither
field states whether a task succeeded. Survival reads the same span; an
unanswered-prompt endpoint requires **6 quiet hours** at the query clock.

Both time tables are written together from one parsed-row read by the
`session_span` derivation. Counts (K23) are ordinary typed session columns;
`messages`, `human_prompts`, `human_commands`, `words`, `human_words`,
`model_words`, `thinking_messages`, `compactions`, `tool_calls` and `errors`
read them. They use the same shown-path and script-action definitions as
ordinary queries. Counts of calls by action are grouped over `tool_calls`.

A changed session or derivation version causes its normal derivation pass to
replace these rows. A single-session time or compare read computes a behind
product with the same function. Population summaries leave behind products
unknown and count them as `pending`. Count fields can compute the current
count while behind; their aggregates also report `pending_FIELD`. Stored
products read neither notes, settings nor the clock.

## Facet counts (K29)

Facets group parsed rows at read time; they have no separate store. Each
session counts once per family/value. The default families are origins,
models, repositories, current tags, logical months and `ends_with`. `deep`
adds roles, authors, block kinds, actions and attachment/thinking/tool flags.
A family name requests only that family. Missing values have a null bucket.
Project facets await the directory-attribution product and its required evidence.

Every row has selected and global counts. Global uses the same session unit's
defaults as the query, including an explicitly expanded kind population.
IDF is `ln(global sessions / global sessions with this value)`; lift is
`selected share / global share`. A zero or absent denominator leaves the
weight unknown. Empty selections return zero selected counts with outcome
`empty`; the global counts remain available. Tag revisions and withdrawals
apply on the next read.

## Compare (K30)

Compare accepts comma-separated recorded field names. It returns each
session's values, the fields that vary across the whole selection, unknown
counts per field and pending sessions. These whole-selection figures stay the
same on every page. An explicit `cost_nusd` compares complete own API-equivalent
response costs as integer nano-USD; a partially priced or unknown total stays
null. Structural comparisons read no message bodies or parent histories.

Distributions use query aggregates (`p50`, `p90`, `max`), with nearest-rank
percentiles and unknown counts. A separate weighted similarity measure is
absent: the catalog's related ranking awaits its recorded recall evaluation.

## Billing-cycle outlook (K12)

Set `plan.<provider>` to explicit terms. The provider key is local configuration;
`origin` names the source to count. There are no guessed subscription allowances.
Money uses integer nano-USD (`1 USD = 1000000000 nano-USD`).

```sh
polylogue setting plan.example '{"name":"pro","origin":"codex","cycle_anchor_day":1,"price_nusd":20000000000,"budget":null}'
polylogue measure outlook --option example:linear
```

The setting's recorded time starts its terms. Changes split the cycle; an
as-of query ignores later settings and observations. Cycles follow UTC
calendar dates and clamp anchor days 29–31 to month end. Billing before the
first setting remains unknown. A missing anchor leaves the outlook unavailable.

Each setting span carries projections for API-equivalent money,
provider-reported money and the five token lanes. Responses and independently
recorded usage reports keep separate populations and denominators. Only
confirmed reports enter sums; unconfirmed and untimed reports are counted.
Unknown values remain null, known zero stays zero, and partial sums are labelled
lower bounds, including when response boundaries or report confirmation are
unknown. Missing observed days are listed per unit.

`linear` projects `used / elapsed * total`. `trailing-7d-mean` keeps history
and adds the mean of the last **7 observed UTC billing days** over the remaining
span. Each result names the method; it is an estimate. There is one grouped
response read over all plan spans and one pass over recorded reports. No day
or cycle aggregate is stored.

An optional operator budget is explicitly labelled `your budget`:

```json
{"name":"pro","origin":"codex","cycle_anchor_day":1,"price_nusd":20000000000,"budget":{"population":"responses","unit":"api_nusd","amount":50000000000,"overage_rate_nusd_per_unit":null}}
```

Budget pressure uses that budget's own population and unit. An unobserved
unit stays unknown; a zero budget has no division or projected breach date.
The plan's price is split by its share of the cycle. Subscription price,
API-equivalent money and provider-reported money remain separate figures.
