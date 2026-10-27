# Session measures

Every measure uses the sessions selected by a query. Without conditions it uses
all sessions, including hidden sessions. CLI, MCP and HTTP use the same measure
registry and Store reader transaction. Unknown measurements serialize as null.

```
polylogue query 'sessions where repo:project | measure facets:deep'
polylogue measure facets --metric deep 'sessions where repo:project'
polylogue query 'sessions | measure profiles'
polylogue query 'sessions where wall_duration >= 5m | measure compare'
polylogue query 'sessions | measure correlation:message_count:word_count'
polylogue query 'sessions | measure distribution:median_tool_call_ms'
polylogue query 'sessions | measure pace:60:0:100'
```

## Facets

`facets` returns selected and global session counts, message counts, and origin
and tag families. Each session counts once per value, including repeated tags
and repeated messages or actions of one kind. `facets:deep` also includes
repositories (root commit identity when available), roles, authors (`written_by`),
block kinds, actions and flags (thinking, tool calls and attachments).
Repository display labels do not split a known root commit identity.
The default result names the deeper families in `omitted`.

IDF is `ln(global session count / global sessions containing value)`. A value
present everywhere has zero weight. The global population includes sessions
outside the selection. `facets:no-idf` and `facets:deep-no-idf` omit weights.
Omission information is separate from facet families and never receives weights.

## Cost outlook

`outlook:PLAN:METHOD` estimates usage through the end of a configured monthly
billing window. Methods are `linear` (default), `trailing-7d-mean`, and
`eom-naive` (the same calculation as linear). The trailing method adds the mean
of the last seven observed days times only the remaining days to observed use.
At zero elapsed time it preserves observed use and reports zero daily burn.

Configure the `cost.plans` setting with a JSON object keyed by plan name:

```
polylogue setting cost.plans '{"sample":{"origin":"codex","cycle_anchor_day":1,"quota_unit":"output_tokens","quota":1000000,"price_usd":20,"overage_rate_usd_per_unit":0.00001}}'
polylogue measure outlook --metric sample:linear
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
means a day had an observation, not that every response or usage unit was
measured. Unknown source times and incomplete usage are reported separately.

Units stay separate: `api_usd`, `reported_usd`, `input_tokens`,
`cached_input_tokens`, `cache_write_tokens`, `output_tokens`, `reasoning_tokens`.
Existing validated independent usage reports add their known amounts alongside
known response rows, with their missing spend-time and completeness information
in `usage`. `api_usd` means API-equivalent USD and `reported_usd` means
tool-reported USD. Plan credits remain unobserved unless source facts establish
them; dollars never substitute for credits. Partial usage projects only recorded
known amounts. Zero is measured zero, while an unobserved unit stays unknown.
No spending recommendations are made.

## Pacing

`pace:THRESHOLD_SECONDS:OFFSET:PAGE_SIZE` defaults to `pace:60:0:100`.
Every selected session contributes to the totals. Gaps are between timestamped
user/assistant messages in source order. Missing timestamps are counted;
negative gaps are ignored and counted. A gap at or below the threshold is
`active`; a longer gap ending with an assistant is `model_work`; a longer gap
ending with a user is `user_idle`. These names describe observed timing, not
human attention or model compute time. `active_ms` excludes `model_work_ms`,
so their sum corresponds to the predecessor's combined active total.

The gap page is ordered by decreasing duration, then session ID and source
position. It includes both message IDs, source turn positions and classification,
with `total_gaps` and `next_offset`. Request the next offset with the same
selection; concurrent writes may change an offset page's population.

Burstiness is the Goh–Barabási coefficient `B = (σ − μ) / (σ + μ)` of the gaps
between consecutive user messages within a session. σ is the population
standard deviation. At least 10 nonnegative timestamped gaps are required;
otherwise it is unknown. Harness-generated user rows are excluded from the burstiness population. A missing
user timestamp breaks that pair. A population
of zero-length gaps has an undefined coefficient and stays unknown.

## Session profiles

`profiles` combines raw title, a deterministic topic label, origin/harness,
repository display and identity, tools, current tag notes, message/word/thinking
counts (including substantive dialogue over ten characters, excluding tool,
thinking and harness messages), dominant model by output tokens, timing facts, workflow shape, final observed state, and existing own and
with-parents usage/cost totals. Usage fields retain their existing units and
missing-cost rules; they are not copied into profile storage. Profile facts are
versioned derivations. Each result stores `rule_version` and `cluster_gap_ms`.

The topic prefers substantive authored prose, strips known context prefixes,
and uses the recorded title only when no prompt supplies a label. Harness
protocol text does not supply the topic. For Codex with repository evidence,
the repository name prefixes the prompt label. Topics use up to 120 characters
(60 prompt characters with a repository prefix). The raw title stays unchanged.
A topic is a scan label, not a semantic summary.

Profile bounds prefer message timestamps, then event timestamps, then session
bounds, with `timestamp_source` naming the choice. A source-only session clock
never invents message latency. `timestamped_messages` and `message_count` show
message timestamp coverage. `wall_duration_ms` is the elapsed span of the
chosen clock. `engaged_duration_ms` sums message-clustered intervals separated
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
Unpaired tool starts at least five minutes before the session's last timestamp
contribute to `stuck_tool_count`.

`workflow_shape` is the predecessor's threshold classifier: delegation takes
precedence (`subagent_dispatch`), then tool density below 0.1 with dialogue
(`chat`), at least three reads with no edits and at most two user turns
(`batch_review`), edits or at least three runs with tool density at least 0.2
and at least ten messages (`agentic_loop`), remaining read/tool activity
(`exploratory`), otherwise `unknown`. Input counts and ratios accompany the
label in `workflow_features`. Shape does not measure quality, correctness,
importance or productivity.

Final observed states are `tool_left`, `question_left`, `error_left`, `refused`,
`truncated` and `unknown`. A later structural tool success clears an earlier
tool error; prose claiming success does not. A clean assistant reply remains
`unknown`: it does not establish task success. The method and final meaningful
message ID accompany the state.

Session query fields `profile_first`, `profile_last`, `wall_duration`,
`engaged_duration`, `tool_active_duration`, `workflow_shape`, `terminal_state`,
`timestamp_source` and `topic` can filter, sort or group these facts before
measurement. Existing session query fields keep their existing definitions.

## Comparison and distributions

`compare` returns selected profiles, varying comparison fields and pairwise
metadata similarity. Identity and title are excluded from varying-field diffs.
Similarity excludes self and zero scores. Its explicit rule adds 3 for the same
known workflow shape, 1 for origin, 2 for first timestamps within three UTC days
(or 1 within fourteen), and 1 per shared tag. Reasons accompany scores, sorted
by score and IDs. This is metadata-shape similarity, not semantic similarity.

`correlation:X:Y` returns Pearson r, selected total, measured pair count, missing
pair count and an explicit reason when fewer than three pairs or zero variance
makes it undefined. It skips sessions missing either metric. `distribution:X`
reports selected/measured/missing sessions, median, nearest-rank p90 and maximum;
zeros participate. Metrics are `message_count`, `word_count`, `thinking_count`,
`tool_call_count`, `wall_duration_ms`, `engaged_duration_ms`,
`tool_active_duration_ms`, `median_tool_call_ms`, `median_agent_response_ms`,
`median_user_response_ms`, `substantive_count`, `total_duration_ms` (sum of
reported message durations, unknown when none are reported), and `cost_usd` (complete own API-equivalent cost only).
