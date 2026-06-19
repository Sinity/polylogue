[← Back to Docs](README.md)

# Insights

Polylogue derives structured read models from raw session data. These are
materialized in the database and queried through the `polylogue ops insights`
subcommand group.

## Overview

| Insight Type | CLI Command | Description |
|-------------|-------------|-------------|
| Session Profiles | `insights profiles` | Per-session evidence, inference, and probabilistic enrichment: repos, tools, costs, durations, message counts, workflow shape, terminal state, summaries |
| Work Events | `insights work-events` | File-level operations detected within sessions |
| Work Phases | `insights phases` | Session segment classification: planning, implementation, verification, exploration |
| Work Threads | `insights threads` | Multi-session groupings by repo and work continuity |
| Session Latency Profiles | API / MCP | Per-session response/tool latency aggregates and stuck-tool counts |
| Tag Rollups | `insights tags` | Tag usage across sessions |
| Day Summaries | `insights day-summaries` | Per-day session and message counts |
| Week Summaries | `insights week-summaries` | Per-week session and message counts |
| Provider Analytics | `insights analytics` | Per-provider message, tool, and thinking stats |
| Session Costs | `insights costs` | Per-session cost estimates with model identification |
| Cost Rollups | `insights cost-rollups` | Provider/model-level cost aggregation |
| Archive Debt | `insights debt` | Maintenance readiness and known debt |

All insight commands support `--limit`, `--offset`, `--format json`, and
inherit root-level filters (`--provider`, `--since`, `--until`).

The per-product evidence / inference / fallback semantics, plus the
`polylogue ops insights audit` CLI, are documented in
[insights-rigor-matrix.md](insights-rigor-matrix.md).

## Session Profiles

Per-session derived aggregates. Materialized in the `session_profiles` table.

```bash
polylogue ops insights profiles
polylogue ops insights profiles --tier merged --sort first-message
polylogue --provider claude-code insights profiles --min-wallclock-seconds 300
```

Fields: session ID, provider, raw title, inferred topic, session date,
first/last session timestamp, timestamp source, wall clock duration, message
count, engaged minutes, workflow shape, terminal state, and folded enrichment
payloads for the merged tier.

`first_message_at` and `last_message_at` prefer provider-supplied message
timestamps. For sources that only preserve session-level timestamps,
session profile materialization falls back to the session created/updated
timestamps and records `evidence.timestamp_source =
session_timestamp_fallback`. That fallback makes time-bucketed analysis
complete without pretending the archive recovered per-message timing.

Optional filters: `--first-message-since`, `--first-message-until`,
`--min-wallclock-seconds`, `--max-wallclock-seconds`, `--workflow-shape`,
`--terminal-state`, `--sort`, `--tier` (`merged`, `evidence`, `inference`),
`--query`.

The merged profile tier also includes probabilistic enrichment fields:
`enrichment.intent_summary`, `enrichment.outcome_summary`,
`enrichment.blockers`, `enrichment.confidence`,
`enrichment.support_level`, and `enrichment.support_signals`. These fields are
derived summaries over the session profile evidence and should be interpreted
through the confidence/support fields, not as raw archive facts. The evidence
and inference tiers intentionally omit the enrichment payload.

The storage column `engaged_duration_ms` is message-clustered wall clock: it
sums phase intervals separated by no more than the fixed five-minute idle
threshold. It does not measure human attention, keyboard focus, or operator
presence. Each materialized phase records `phase_idle_threshold_ms`, currently
300000, so readers do not need to know a hidden global constant to interpret a
phase split.

`tool_active_duration_ms` is session-event tool activity: it sums paired
tool-call start/output events that have explicit timestamps. It does not
measure human attention and it does not invent duration for unpaired or
untimestamped tool events. A 12-minute Bash call with a matching output counts
as 12 minutes of tool-active time even when message clustering drops the
inter-message gap; ten user messages six minutes apart count as neither
message-clustered nor tool-active time; four two-minute tool calls interleaved
with short replies count in both measures.

`inference.inferred_topic` is a deterministic label for search and scan
ergonomics. It is not the provider title and it is not a semantic summary:
materialization prefers the first substantive user turn, strips known context
dump prefixes, and for Codex sessions with repository evidence prefixes the
topic with the first inferred repo name. The raw `title` remains unchanged for
provenance and provider fidelity.

`workflow_shape` is a threshold classifier over observable session features:
tool-call density, read/edit/run/subagent mix, compaction count, thinking ratio,
and user/assistant turn counts. `chat` means a short low-tool session;
`exploratory` means mixed read/search/tool activity without a strong edit loop;
`agentic_loop` means sustained tool use with edits or repeated commands;
`subagent_dispatch` means Task/subagent-style delegation is present;
`batch_review` means read-heavy, low-edit inspection from a small prompt set.
These labels do not measure task importance, agent quality, correctness, or
operator productivity. The input vector is stored as
`evidence.workflow_shape_features` so a reader can audit the rule that fired.

`terminal_state` is a read-only boundary signal. `clean_finish` means the last
meaningful observed message was assistant-side with no trailing tool error or
unpaired provider tool event. `question_left` means the final meaningful
message was user-side. `error_left` means the trailing session event or final
assistant text carried an error marker. `tool_left` means a provider tool start
was observed without a matching output. `agent_hanging` is reserved for future
session-end evidence with an explicit inactivity boundary. These states do not
judge whether abandoning a session was good or bad; they only expose the final
observable archive shape. `evidence.terminal_state_evidence` cites the
message/session event or pending-tool count behind the decision.

MCP exposes two convenience readers over the same materialized rows:
`workflow_shape_distribution(since, until, group_by)` and
`find_abandoned_sessions(since, repo_path, min_severity)`.

## Session Latency Profiles

Session latency profiles are materialized in `session_latency_profiles` and
read through the Python API and MCP. They expose per-session aggregates:
median, p90, and max paired provider tool-call latency; count of provider tool
starts left open beyond the fixed stuck threshold; median user-to-assistant and
assistant-to-user response gaps; and tool-call counts by category.

These fields are runtime-shape signals, not quality judgments. Agent response
latency includes model output delay and any intervening tool execution visible
in the archive. Provider tool latency requires timestamped start/output event
pairs; unpaired starts contribute only to `stuck_tool_count` when the session
end is far enough past the start. User response latency caps long idle gaps so
calendar-scale pauses do not masquerade as sessional latency.

MCP exposes three readers over these same rows:
`session_latency_profile(session_id)`,
`tool_call_latency_distribution(since, until, provider, tool_category)`, and
`find_stuck_sessions(since, limit)`.

## Work Events

Message-range work segments derived from tool calls, message timing, and
coarse text/action signals. These rows are useful for timeline navigation,
file/tool context, and rough event grouping. The `heuristic_label` field is a
weak event label such as `implementation`, `debugging`, `testing`, `research`,
or `review`; it is not a durable workflow taxonomy and should not be treated
as the session-level semantic contract. Use `workflow_shape` and
`terminal_state` on session profiles when the question is about the whole
session.

```bash
polylogue ops insights work-events
polylogue ops insights work-events --heuristic-label implementation
polylogue ops insights work-events --session-id claude-ai:abc123
```

Each event has: start/end time, duration, file paths, tools used, a short
summary, and the heuristic event label plus confidence/evidence. File/tool
categories such as `file_read`, `file_edit`, or `shell` live on action/tool
surfaces, not in `session_work_events.heuristic_label`.

## Work Phases

Sessions are segmented into phases based on activity patterns.

```bash
polylogue ops insights phases
polylogue ops insights phases --kind implementation
```

Phase kinds:

| Kind | Description |
|------|-------------|
| `planning` | Design discussion, requirements gathering |
| `implementation` | Code writing and editing |
| `verification` | Testing, linting, debugging |
| `exploration` | Searching, browsing, reading |

## Work Threads

Multi-session groupings connected by repo, branch relationships, and work
continuity.

```bash
polylogue ops insights threads
```

Fields: thread ID, dominant repo, session count, total messages, depth,
session IDs.

## Day and Week Summaries

Pre-aggregated rollups for calendar views.

```bash
polylogue ops insights day-summaries
polylogue ops insights week-summaries
```

## Provider Analytics

Per-provider message counts, tool use percentages, and thinking block
percentages.

```bash
polylogue ops insights analytics
polylogue ops insights analytics --format json
```

## Cost Tracking

Polylogue includes a 23-model pricing catalog covering Anthropic, OpenAI, and
Google models.

### Session Costs

```bash
polylogue ops insights costs
polylogue ops insights costs --model claude-opus-4-6
polylogue ops insights costs --status exact
```

Cost statuses: `exact` (API-reported tokens), `priced` (inferred tokens),
`partial` (incomplete data), `unavailable` (no usage data).

### Cost Rollups

```bash
polylogue ops insights cost-rollups
polylogue ops insights cost-rollups --model gpt-4o
```

### Pricing Catalog

```
Anthropic: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5,
           claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus/sonnet/haiku
OpenAI:    gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo,
           o1, o1-mini, o3, o3-mini
Google:    gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-pro
```

Cost estimates include input tokens, output tokens, cache-read tokens (for
models that support prompt caching), and cache-write tokens.

## Tags

```bash
polylogue ops insights tags
polylogue ops insights tags --query polylogue
```

Shows tag names with session counts, explicit count (user-assigned), and
auto count (derived from session content).

## Archive Debt

```bash
polylogue ops insights debt
polylogue ops insights debt --category schema
polylogue ops insights debt --only-actionable
```

Tracks maintenance readiness: schema version gaps, rebuild debt, stale
indexes, and unprocessed raw records.
