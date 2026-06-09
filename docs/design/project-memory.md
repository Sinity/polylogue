# Project Memory

Machine-observed knowledge about a project, accumulated from AI coding sessions.
Complements CLAUDE.md — CLAUDE.md is what humans declare; project memory is what
agents actually do.

## Design Properties

Four properties guide the design:

1. **Proactive** — injected at SessionStart without the agent asking. The agent
   shouldn't need to know what questions to ask.
2. **Visible degradation** — stale entries visibly decay rather than silently
   becoming wrong. Freshness is a first-class property, not an afterthought.
3. **Provenance** — every entry cites the session that produced it. Agents can
   judge credibility from source context.
4. **Queryability** — agents can search project memory via MCP tools, not just
   receive the injected summary.

## Record Types

Five kinds of knowledge, each with a distinct schema:

### Decision
A design choice with rationale and context. Example: "Chose WAL mode over
rollback journal because concurrent reads during ingestion are required."
Fields: `summary`, `rationale`, `alternatives_rejected`, `session_id`,
`recorded_at`, `freshness_decay_days`.

### Trial
Something that was tried and failed. Prevents re-litigation. Example:
"Tried sqlite-vec 0.1.0 but it crashed on vector dims > 1024."
Fields: `summary`, `failure_mode`, `session_id`, `recorded_at`.

### HotFile
A file that consistently causes issues or is central to the architecture.
Example: "archive_tiers/index.py — changing this requires index-tier version bump and
regenerating all test fixtures."
Fields: `path`, `why_hot`, `last_modified_session_id`, `recorded_at`.

### Pattern
A recurring code pattern with its rationale. Example: "All SQLite writes use
BEGIN IMMEDIATE to avoid SQLITE_BUSY on concurrent access."
Fields: `pattern_name`, `description`, `examples`, `session_ids[]`,
`recorded_at`.

### IssueNote
A durable observation about an open issue. Supplements the issue body with
machine-observed context. Example: "The concurrency test fails only on Python
3.11, not 3.12+. Possibly a sqlite3 module bug."
Fields: `issue_number`, `observation`, `session_id`, `recorded_at`.

## Freshness Model

Entries decay on a per-type schedule:

| Type | Default decay | Rationale |
|------|--------------|-----------|
| Decision | 90 days | Architecture decisions age slowly |
| Trial | 180 days | Failed approaches stay failed |
| HotFile | 30 days | File importance shifts with codebase |
| Pattern | 60 days | Patterns persist but can become anti-patterns |
| IssueNote | 14 days | Issue context changes rapidly |

Decay is not deletion. Stale entries remain queryable but are excluded from
SessionStart injection. An entry is "stale" when `recorded_at + decay_days < now`.

Entries can be refreshed: if a new session confirms a Decision or Pattern, its
`recorded_at` is bumped. Conflicts (new session contradicts an old entry) are
surfaced, not silently resolved.

## SessionStart Injection

The hook injects project memory into `additionalContext`:

```
[Project Memory — polylogue]
Recent decisions (2):
- Chose WAL mode over rollback journal (2026-04-15)
- Schema is fresh-only, no in-place upgrade chain (2026-03-10)

Active patterns (3):
- All SQLite writes use BEGIN IMMEDIATE
- Content hash excludes user metadata
- ...

Hot files (3):
- polylogue/storage/sqlite/archive_tiers/index.py — index-tier schema authority
- ...

Stale warnings (1):
- "Use the root batch command for all pipeline operations" — daemon replaced this in #717
```

The injection is capped at 250 tokens. Ranking formula:

```
score = file_match(4.0) + recency_decay(0.9^days) + event_match(2.0)
```

- `file_match`: if the entry references files touched in the current session
- `recency_decay`: exponential decay since `recorded_at`
- `event_match`: if the entry's session touched similar files to current session

Top 3 entries by score are injected.

## Storage

A `project_memory` table in the archive database. Not a separate file — it lives
with the rest of polylogue's data so it benefits from FTS5 search, backup, and
the existing query infrastructure.

```sql
CREATE TABLE project_memory (
    id TEXT PRIMARY KEY,
    project_path TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('decision','trial','hotfile','pattern','issuenote')),
    summary TEXT NOT NULL,
    detail TEXT,
    session_ids TEXT NOT NULL,  -- JSON array
    recorded_at TEXT NOT NULL,
    refreshed_at TEXT,
    freshness_decay_days INTEGER NOT NULL DEFAULT 30,
    file_refs TEXT  -- JSON array of paths
);
```

## MCP Surface

Three tools:

- `get_project_memory(project_path, kind=None)` — returns all entries, optionally filtered by kind
- `search_project_memory(query, project_path)` — FTS5 search across summaries and details
- `record_project_memory(kind, summary, detail, file_refs)` — agent-initiated recording (requires write role)

## Relationship to CLAUDE.md

CLAUDE.md is human-curated and checked into the repo. Project memory is
machine-observed and stored in the archive. They serve different needs:

| | CLAUDE.md | Project Memory |
|---|---|---|
| Author | Human | Agent (observed) |
| Content | Declared conventions | Revealed behavior |
| Freshness | Manual updates | Automatic decay |
| Scope | What should be true | What actually happens |
| Verification | Code review | Session evidence |

Neither replaces the other. CLAUDE.md says "we use WAL mode." Project memory
says "we tried switching to rollback journal in session X and it caused Y."

## Build Order

1. `project_memory` table + backfill
2. MCP tools (read-only first: search + get)
3. SessionStart hook injection with ranking formula
4. Agent-initiated recording (write tool)
5. Freshness decay + conflict detection
6. Automated capture heuristics (detect Decisions from session patterns)
