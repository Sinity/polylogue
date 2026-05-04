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
| Tag Rollups | `insights tags` | Tag usage across conversations |
| Day Summaries | `insights day-summaries` | Per-day session and message counts |
| Week Summaries | `insights week-summaries` | Per-week session and message counts |
| Provider Analytics | `insights analytics` | Per-provider message, tool, and thinking stats |
| Session Costs | `insights costs` | Per-session cost estimates with model identification |
| Cost Rollups | `insights cost-rollups` | Provider/model-level cost aggregation |
| Archive Debt | `insights debt` | Maintenance readiness and known debt |

All insight commands support `--limit`, `--offset`, `--format json`, and
inherit root-level filters (`--provider`, `--since`, `--until`).

## Session Profiles

Per-session derived aggregates. Materialized in the `session_profiles` table.

```bash
polylogue insights profiles
polylogue insights profiles --tier merged --sort first-message
polylogue --provider claude-code insights profiles --min-wallclock-seconds 300
```

Fields: conversation ID, provider, title, session date, first/last message
timestamp, wall clock duration, message count, engaged minutes.

Optional filters: `--first-message-since`, `--first-message-until`,
`--min-wallclock-seconds`, `--max-wallclock-seconds`, `--sort`, `--tier`
(`merged`, `evidence`, `inference`), `--query`.

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
