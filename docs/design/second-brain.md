# Second Brain

SessionStart hook injection that gives coding agents context from past sessions
without them having to ask for it.

## The Problem

Agents start every session cold. They don't know what you worked on yesterday,
what decisions were made, what patterns were established, or what errors were
hit. CLAUDE.md provides static context, but the dynamic context — what actually
happened in recent sessions — is locked in the archive.

Agents can query polylogue for this context, but they have to know what to ask.
The archive has 7,767 sessions and 3.3M messages. An agent can't search
everything at session start.

## The Solution

A SessionStart hook that injects a compact context brief into the agent's
`additionalContext`. The brief is relevance-ranked, time-decayed, and capped at
250 tokens.

## What Gets Injected

Three sections, each with a token budget:

### 1. Recent Activity (150 tokens)
The last 3 sessions touching the current project, with:
- What was accomplished (from session profiles)
- Key files modified (from action_events.affected_paths)
- Outcome: completed / blocked / abandoned
- Session date and duration

### 2. Active Decisions (50 tokens)
The 2 most recent un-stale Decisions from project memory that reference files
in the current working directory.

### 3. Recurring Errors (50 tokens)
If any error pattern appears in 3+ recent sessions, surface it. Example:
"conftest.py fixture teardown fails on Python 3.11 (3 occurrences, last: 2 days ago)"

## Ranking Formula

```
score = file_match(4.0) + recency_decay(0.9^days) + event_match(2.0)
```

- `file_match`: number of files referenced by the session that match the current
  working directory or files mentioned in the agent's prompt
- `recency_decay`: exponential decay with 7.7-day half-life. Yesterday's session
  scores ~0.9, last week's ~0.5, last month's ~0.04
- `event_match`: bonus for sessions that touched the same repo or tool categories

Top 3 sessions by score are included in the brief. Sessions below a minimum score
threshold are excluded (avoid noise from barely-relevant sessions).

## Data Sources

All data comes from existing tables, zero new collection:

| Information | Source |
|---|---|
| Session summary | `session_profiles` (repo_paths, tool_categories, duration, work_events) |
| Files modified | `action_events.affected_paths` |
| Session outcome | `session_profiles.outcome` or inferred from last message |
| Error patterns | `action_events` filtered by tool failures + compaction_count |
| Project memory entries | `project_memory` table (see project-memory.md) |

## Context Brief Format

```
[Polylogue — recent activity in polylogue/]
2026-05-03: Fixed FTS trigger leak in daemon ingestion (42 min, 5 files)
2026-05-02: Filed #800-#807 from deep audit findings (3h 15min, 12 files)
2026-04-30: Closed assurance-ontology Track A (#506-#520) (2h, 8 files)

Active decisions: Use WAL mode for all SQLite access; Schema is fresh-only.

Recurring: conftest.py teardown fails on 3.11 (3× in 2 weeks, ref
session/abc123)
```

Token count: ~180 tokens. Fits within the 250-token budget with room for the
hook to add framing.

## Implementation

`build_resume_brief()` and `render_resume_brief()` already exist in
`polylogue/insights/resume.py`. They compute the data but aren't wired as a
hook-format output.

What's needed:
1. A `SessionStart` hook handler that calls `build_resume_brief()` with the
   current project path
2. Integration with the project memory table for Decisions section
3. Error pattern detection from `action_events` across recent sessions
4. The brief rendered as `additionalContext` text

The hook runs in-process during daemon operation. Since it's read-only (no
ingestion), it completes in <100ms.

## Non-Goals

- Not a full sessional search. The agent can use MCP search tools for
  deeper queries — this is just the "while you were away" summary.
- Not a replacement for CLAUDE.md. Static conventions belong in CLAUDE.md;
  dynamic context belongs here.
- Not cross-project by default. Sessions are filtered to the current project
  root. Cross-project queries are available via MCP tools.
