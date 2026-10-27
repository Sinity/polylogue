# Session measures

Profile, comparison, correlation, distribution, pacing and facet measures use
the sessions selected by a query, including its unit defaults. Usage measures
without conditions cover all sessions, including hidden sessions. CLI, MCP and HTTP use the same measure
registry and Store reader transaction. Unknown measurements serialize as null.

```
polylogue query 'sessions where repo:project | measure facets:deep'
polylogue measure facets --option deep 'sessions where repo:project'
polylogue query 'sessions | measure profiles'
polylogue query 'sessions where wall_duration >= 5m | measure compare'
polylogue query 'sessions | measure correlation:message_count:word_count'
polylogue query 'sessions | measure distribution:median_tool_call_ms'
polylogue query 'sessions | measure pace:60'
```

## Facets

`facets` returns selected and global session counts, message counts, and origin
and tag families. Each session counts once per value, including repeated tags
and repeated messages or actions of one kind. `facets:deep` also includes
repositories from every recorded working directory (root commit identity when
available), roles, authors (`written_by`),
block kinds, actions and flags (thinking, tool calls and attachments).
Repository display labels do not split a known root commit identity.
The default result names the deeper families in `omitted`.

IDF is `ln(global session count / global sessions containing value)`. A value
present everywhere has zero weight. The global population uses the same unit defaults as the selection, before
its conditions. By default both sides count main sessions, excluding stubs,
hidden sessions and excisions. Query conditions that override a unit default
apply the same override to the global population. Tags are lowercase and use
the shared current-note and withdrawal rules. Deep families are derived once
per changed session; `underived_sessions` reports missing derivations. `facets:no-idf` and `facets:deep-no-idf` omit weights.
Omission information is separate from facet families and never receives weights.

## Cost outlook

`outlook:PLAN:METHOD` estimates usage through the end of a configured monthly
billing window. Methods are `linear` (default) and `trailing-7d-mean`. The trailing method adds the mean
of the last seven observed days times only the remaining days to observed use.
At zero elapsed time it preserves observed use and reports zero daily burn.

Configure the `cost.plans` setting with a JSON object keyed by plan name:

```
polylogue setting cost.plans '{"sample":{"origin":"codex","cycle_anchor_day":1,"quota_unit":"output_tokens","quota":1000000,"price_usd":20,"overage_rate_usd_per_unit":0.00001}}'
polylogue measure outlook --option sample:linear
```

These are synthetic example inputs, not provider plan terms. All amounts and
the plan origin come from this setting. There are no built-in plans. Missing
configuration, name or cycle anchor returns an explicit reason. An unknown
configured name is an error. Fields may be null. Quota and overage are unknown
when the configured unit has no observations. Missing quota has its own reason.
A missing overage rate leaves overage cost unknown. The configured fixed price
is shown separately; API-equivalent usage is never presented as an invoice.

The cycle starts at midnight UTC on the anchor day and ends at the next anchor,
exclusive. Anchor days 29–31 use the last day in shorter months. Observations
are bounded by the exact as-of instant before daily aggregation. The logical
day setting does not move billing boundaries. The window, clock, method,
observed-day coverage and missing full UTC days accompany every estimate.
The current partial day is excluded from the coverage denominator. Coverage
means a day had a known usage amount, not that every response or usage unit was
measured. Unknown source times and incomplete usage are reported separately.

Units stay separate: `api_usd`, `reported_usd`, `input_tokens`,
`cached_input_tokens`, `cache_write_tokens`, `output_tokens`, `reasoning_tokens`.
Existing validated independent usage reports add their known amounts alongside
known response rows, with their missing spend-time and completeness information
in `recorded_usage`. Codex has independent readings rather than usage on its
response rows, so adding those readings does not count their usage twice.
`untimed_responses` counts response rows without a source time. The response
series is grouped in one pass over the window; recorded readings take one pass. `api_usd` means API-equivalent USD and `reported_usd` means
tool-reported USD. Plan credits remain unobserved unless source facts establish
them; dollars never substitute for credits. Partial usage projects only recorded
known amounts. Zero is measured zero, while an unobserved unit stays unknown.
No spending recommendations are made.

## Pacing

`pace:THRESHOLD_SECONDS` defaults to `pace:60`.
The query cursor pages sessions. Every returned session contributes all its gaps
and its totals to that page. Gaps are between timestamped
user/assistant messages in source order, excluding harness-written rows. Missing timestamps are counted;
negative gaps are ignored and counted. A gap at or below the threshold is
`active`; a longer gap ending with an assistant is `model_work`; a longer gap
ending with a user is `user_idle`. These names describe observed timing, not
human attention or model compute time. `active_ms` excludes `model_work_ms`,
so their sum corresponds to the predecessor's combined active total.

Within each session page, gaps are ordered by decreasing duration, then session
ID and source position. Each includes both message IDs, source turn positions
and classification. The response's `count` and `next` refer to selected sessions,
with `total_gaps` describing the returned sessions. All paged measures use the
same query cursor and `--limit`/`--cursor` arguments as ordinary queries.

Burstiness is the Goh–Barabási coefficient `B = (σ − μ) / (σ + μ)` of the gaps
between consecutive user messages within a session. σ is the population
standard deviation. At least 10 nonnegative timestamped gaps are required;
otherwise it is unknown. Harness-generated user rows are excluded from the burstiness population. A missing
user timestamp breaks that pair. A population
of zero-length gaps has an undefined coefficient and stays unknown.

## Session profiles

`profiles` combines the recorded title (or the shared derived title), origin/harness,
working directories and their known repository labels/identities, tools, current tag notes, message/word/thinking
counts (including substantive dialogue of at least three words, excluding tool,
thinking and harness messages), dominant model by output tokens, timing facts, workflow shape, final observed state, and existing own and
with-parents usage/cost totals. When parent history is incomplete, `usage.own`
remains available, `usage.with_parents` is null, and `usage.parents` names the
missing history. Usage fields retain their existing units and
missing-cost rules; they are not copied into profile storage. Profile facts are
versioned derivations. Each result stores `rule_version` and `cluster_gap_ms`.

The title uses the same title derivation as session listings: the recorded title
wins, otherwise the first prose prompt a person typed supplies its first line,
masked before truncation. Origin declarations classify harness-written prompts
before derivation. Repository facts use existing observations for every recorded
working directory, with the longest matching repository path and distinct
identities. An unknown directory stays in `working_directories`.

Scalar facts have typed columns in `session_profile`. Bounds are read from
`session_survival`, with the event/session fallback described below. An underived
session returns `facts: null`; `underived_sessions` counts these in each result
page or aggregate population. Retraction must advance `session.changed` in the
same transaction as a changed shown version or deleted parsed rows, so the
ordinary derivation pass recomputes these facts.

Profile bounds prefer message timestamps, then event timestamps, then session
bounds, with `timestamp_source` naming the choice. A source-only session clock
never invents message latency. `timestamped_messages` and `message_count` show
message timestamp coverage. `wall_duration_ms` is the elapsed span of the
chosen clock. `message_clustered_duration_ms` sums message-clustered intervals separated
by gaps of at most five minutes, using interval min/max times even for edited
messages out of chronological order. Without timestamped messages it uses event
times. It does not measure human attention, keyboard focus or operator presence.

`tool_active_duration_ms` sums paired tool-call/result intervals with explicit
timestamps. Overlapping calls each contribute their own interval. Unpaired or
untimestamped calls do not invent durations. A paired 12-minute Bash call has
12 minutes of tool-active time even when message clustering drops that gap.
Ten user messages six minutes apart count as neither clustered nor tool-active
time. Short replies interleaved with two-minute calls can count in both.

Tool latency and response latency each report measured pair count, median, p90
(nearest rank) and maximum. Missing pairs remain unknown and measured zero
survives storage and percentiles. User-to-assistant gaps include intervening
tool execution. Assistant-to-user gaps over 30 minutes are excluded. Negative
response/tool deltas are clamped to zero, as in the predecessor's latency rules.
Unpaired tool starts more than five minutes before the session's last timestamp
contribute to `unpaired_tools_past_gap`.

`workflow_shape` is the predecessor's threshold classifier: delegation takes
precedence (`subagent_dispatch`), then tool density below 0.1 with dialogue
(`chat`), at least three reads with no edits and at most two user turns
(`batch_review`), edits or at least three runs with tool density at least 0.2
and at least ten messages (`agentic_loop`), remaining read/tool activity
(`exploratory`), otherwise `unknown`. Ratios are rounded to four decimals before classification. Input counts and
ratios accompany the
label in `workflow_features`. Shape does not measure quality, correctness,
importance or productivity.

Final observed states are `tool_left`, `question_left`, `error_left`, `refused`,
`truncated` and `unknown`. A later structural tool success clears an earlier
tool error; prose claiming success does not. A clean assistant reply remains
`unknown`: it does not establish task success. The method and final meaningful
message ID accompany the state.

Session query fields `profile_first`, `profile_last`, `wall_duration`,
`message_clustered_duration`, `tool_active_duration`, `workflow_shape`, `final_observed_state`,
and `timestamp_source` can filter, sort or group these facts before
measurement. Existing session query fields keep their existing definitions.

## Comparison and distributions

`profiles` and `compare` page selected sessions through the query cursor.
`compare` returns structural facts and the varying fields within that page;
it reads no parent history. Its optional numeric metric can be selected as
`compare:cost_usd` (CLI `measure compare --metric cost_usd`) to include own
usage cost. Without a metric it compares structural facts and reads no usage. Identity and title are excluded from
varying-field differences. Usage is read for full profiles and only for an
explicit `cost_usd` aggregate metric. `similar` takes a query selecting one
reference session and pages matches from every other session in the archive:

```
polylogue query 'sessions where id:"codex:example" | measure similar'
```

Similarity excludes self and zero scores. Its explicit rule adds 3 for the same
known workflow shape, 1 for origin, 2 for first timestamps within three UTC days
(or 1 within fourteen), and 1 per shared tag. Reasons accompany scores, sorted
by descending score and session ID. Only origin, workflow shape, first time
and current tags enter this read. `underived_sessions` counts missing profiles
in the comparison population; origin and tags remain usable while facts are
underived. This is metadata similarity.

`correlation:X:Y` returns Pearson r rounded to four decimals, selected total, measured pair count, missing
pair count and an explicit reason when fewer than three pairs or zero variance
makes it undefined. It skips sessions missing either metric. `distribution:X`
reports selected/measured/missing sessions, median, nearest-rank p90 and maximum;
zeros participate. Metrics are `message_count`, `word_count`, `thinking_count`,
`tool_call_count`, `wall_duration_ms`, `message_clustered_duration_ms`,
`tool_active_duration_ms`, `median_tool_call_ms`, `median_agent_response_ms`,
`median_user_response_ms`, `substantive_count`, `total_duration_ms` (sum of
reported message durations, unknown when none are reported), and `cost_usd` (complete own API-equivalent cost only).
