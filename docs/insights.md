[← Back to Docs](README.md)

# Insights

Polylogue derives structured read models from raw conversation data. These are
materialized in the database and queried through the `polylogue insights`
subcommand group.

## Overview

| Insight Type | CLI Command | Description |
|-------------|-------------|-------------|
| Session Profiles | `insights profiles` | Per-session aggregation: repos, tools, costs, durations, message counts |
| Session Enrichments | `insights enrichments` | Probabilistic session enrichment (support level, enrichment family) |
| Work Events | `insights work-events` | File-level operations detected within sessions |
| Work Phases | `insights phases` | Session segment classification: planning, implementation, verification, exploration |
| Work Threads | `insights threads` | Multi-session groupings by repo and work continuity |
| Session Latency Profiles | API / MCP | Per-session response/tool latency aggregates and stuck-tool counts |
| Tag Rollups | `insights tags` | Tag usage across conversations |
| Day Summaries | `insights day-summaries` | Per-day session and message counts |
| Week Summaries | `insights week-summaries` | Per-week session and message counts |
| Provider Analytics | `insights analytics` | Per-provider message, tool, and thinking stats |
| Session Costs | `insights costs` | Per-session cost estimates with model identification |
| Cost Rollups | `insights cost-rollups` | Provider/model-level cost aggregation |
| Archive Debt | `insights debt` | Maintenance readiness and known debt |

All insight commands support `--limit`, `--offset`, `--format json`, and
inherit root-level filters (`--provider`, `--since`, `--until`).

The per-product evidence / inference / fallback semantics, plus the
`polylogue insights audit` CLI, are documented in
[insights-rigor-matrix.md](insights-rigor-matrix.md).

## Session Profiles

Per-session derived aggregates. Materialized in the `session_profiles` table.

```bash
polylogue insights profiles
polylogue insights profiles --tier merged --sort first-message
polylogue --provider claude-code insights profiles --min-wallclock-seconds 300
```

Fields: conversation ID, provider, raw title, inferred topic, session date,
first/last session timestamp, timestamp source, wall clock duration, message
count, engaged minutes, workflow shape, and terminal state.

`first_message_at` and `last_message_at` prefer provider-supplied message
timestamps. For sources that only preserve conversation-level timestamps,
session profile materialization falls back to the conversation created/updated
timestamps and records `evidence.timestamp_source =
conversation_timestamp_fallback`. That fallback makes time-bucketed analysis
complete without pretending the archive recovered per-message timing.

Optional filters: `--first-message-since`, `--first-message-until`,
`--min-wallclock-seconds`, `--max-wallclock-seconds`, `--workflow-shape`,
`--terminal-state`, `--sort`, `--tier` (`merged`, `evidence`, `inference`),
`--query`.

The storage column `engaged_duration_ms` is message-clustered wall clock: it
sums phase intervals separated by no more than the fixed five-minute idle
threshold. It does not measure human attention, keyboard focus, or operator
presence.

`tool_active_duration_ms` is provider-event tool activity: it sums paired
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
message was user-side. `error_left` means the trailing provider event or final
assistant text carried an error marker. `tool_left` means a provider tool start
was observed without a matching output. `agent_hanging` is reserved for future
session-end evidence with an explicit inactivity boundary. These states do not
judge whether abandoning a session was good or bad; they only expose the final
observable archive shape. `evidence.terminal_state_evidence` cites the
message/provider event or pending-tool count behind the decision.

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
calendar-scale pauses do not masquerade as conversational latency.

MCP exposes three readers over these same rows:
`session_latency_profile(conversation_id)`,
`tool_call_latency_distribution(since, until, provider, tool_category)`, and
`find_stuck_sessions(since, limit)`.

## Work Events

File-level operations detected from tool calls and semantic block types.

```bash
polylogue insights work-events
polylogue insights work-events --kind file_edit
polylogue insights work-events --conversation-id claude-ai:abc123
```

Each event has: kind (`file_read`, `file_write`, `file_edit`, `shell`, etc.),
start/end time, duration, file paths, tools used.

## Work Phases

Sessions are segmented into phases based on activity patterns.

```bash
polylogue insights phases
polylogue insights phases --kind implementation
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
polylogue insights threads
```

Fields: thread ID, dominant repo, session count, total messages, depth,
session IDs.

## Day and Week Summaries

Pre-aggregated rollups for calendar views.

```bash
polylogue insights day-summaries
polylogue insights week-summaries
```

## Provider Analytics

Per-provider message counts, tool use percentages, and thinking block
percentages.

```bash
polylogue insights analytics
polylogue insights analytics --format json
```

## Cost Tracking

Polylogue includes a 23-model pricing catalog covering Anthropic, OpenAI, and
Google models.

### Session Costs

```bash
polylogue insights costs
polylogue insights costs --model claude-opus-4-6
polylogue insights costs --status exact
```

Cost statuses: `exact` (API-reported tokens), `priced` (inferred tokens),
`partial` (incomplete data), `unavailable` (no usage data).

### Cost Rollups

```bash
polylogue insights cost-rollups
polylogue insights cost-rollups --model gpt-4o
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
polylogue insights tags
polylogue insights tags --query polylogue
```

Shows tag names with conversation counts, explicit count (user-assigned), and
auto count (derived from session content).

## Archive Debt

```bash
polylogue insights debt
polylogue insights debt --category schema
polylogue insights debt --only-actionable
```

Tracks maintenance readiness: schema version gaps, migration debt, stale
indexes, and unprocessed raw records.
