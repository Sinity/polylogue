# Archive Storytelling

Narrative generation from AI session archives. Stories are not analytics
reports — they are readable narratives that answer "how did that happen" by
tracing the archive for patterns, arcs, and origins.

## Design Properties

Three properties distinguish stories from queries:

1. **Narrative arc** — a story has a beginning, middle, and end. It traces
   change over time, not a snapshot.
2. **Provenance links** — every factual claim in a story links back to the
   session that produced it. The reader can verify by clicking through.
3. **Deterministic** — the same inputs produce the same story. No LLM
   summarization (the provenance guarantee breaks). Pure SQL + template
   rendering.

## Story Types

### Feature Birth

"How did feature X come to be?" Traces a feature from its first mention in
a session through its implementation arc to production.

The story follows this template:

1. **Genesis**: first session where the feature is mentioned. The exact prompt
   or session excerpt. Date, repo context, session link.
2. **Design sessions**: sessions where the feature was discussed, planned, or
   explored but not yet implemented. Key decisions, alternatives rejected.
3. **Implementation arc**: sessions where files were created or modified for
   the feature. File paths, commit references, session-by-session timeline.
4. **Completion**: the session where the work was merged, shipped, or declared
   done. Duration from genesis to completion.
5. **Aftermath**: follow-up fixes, enhancements, documentation sessions within
   2 weeks of completion.

Discovery: the user provides a file path, branch name, or keyword. The story
engine searches `actions.affected_paths` and `session_profiles.repo_paths_json`
for matches, traces the sessions forward in time, and assembles the narrative.

CLI:

```
polylogue story feature-birth --path polylogue/storage/sqlite/archive_tiers/index.py
polylogue story feature-birth --branch feature/feat/time-machine
polylogue story feature-birth --keyword "content hash"
```

### Day in the Life

"What did AI-assisted work look like on a specific day?" A single day's AI
usage rendered as a narrative timeline.

The story template:

1. **Day opening**: first session of the day. What was the first prompt? What
   project was open? (From session profile data.)
2. **Session arc**: each session on that day gets a paragraph with:
   - Time range (first to last message timestamp)
   - Session title or inferred topic
   - Provider used
   - Tool use breakdown (tool calls vs. thinking vs. prose)
   - Cost
   - Key files touched
3. **Day closing**: last session of the day. Was work completed or carried
   forward? (Inferred from session outcome.)
4. **Day stats**: total sessions, total words, total cost, tools vs. prose
   ratio, providers used.

Discovery: the user provides a date. The story engine queries `session_profiles`
with `canonical_session_date = :date`, orders by `first_message_at`, enriches
with `actions.affected_paths` for each session, and renders the template.

CLI:

```
polylogue story day-in-life --date 2026-03-17
polylogue story day-in-life --date yesterday
```

### The Big Refactor

"How did this refactoring unfold across sessions?" Traces a multi-session
refactoring from first touch to completion.

The story template:

1. **Catalyst**: the session that triggered the refactoring. What was broken
   or inadequate?
2. **Scope discovery**: sessions where the refactoring scope expanded,
   contracted, or was renegotiated. Unintended consequences discovered.
3. **Execution phases**: one section per phase, where a phase is a cluster of
   sessions on consecutive days touching the same file set. Each phase lists
   files changed, sessions involved, and the cumulative diff shape.
4. **Completion**: the session where the refactoring was declared done or
   the refactoring branch was merged.
5. **Aftermath**: regressions, follow-up fixes, documentation updates.

Discovery: the user provides a branch name. The story engine:

1. Finds sessions with that branch in `session_profiles.auto_tags_json` or
   in message text.
2. Orders them by date.
3. Detects phase boundaries (gaps >2 days without a session touching the
   branch).
4. For each phase, collects affected paths from `actions`.
5. Computes cumulative word count, tool usage, and cost per phase.
6. Renders the template.

If no branch name is provided, the engine suggests candidate refactoring
branches from the archive (branches with 3+ sessions spanning 3+ days).

CLI:

```
polylogue story refactor --branch feature/refactor/storage-product-splits
polylogue story refactor --suggest
polylogue story refactor --path polylogue/storage/sqlite/
```

### AI Journey

"How did my AI usage evolve over the archive's lifetime?" The macroscope
story — traces the user's entire AI journey in one narrative.

The story template:

1. **First contact**: the first session in the archive. What was the first
   thing the user asked an AI? First provider, first tool use.
2. **Era transitions**: one section per detected era (see time-machine.md),
   describing:
   - Date range and duration
   - Primary provider and model
   - Session density and work cadence
   - Dominant projects and repo context
   - Tool sophistication (tool use ratio, thinking ratio, multi-turn depth)
   - Cost per session trend
3. **Milestones**: notable firsts — first tool use, first multi-hour session,
   first $10+ session, first day with 5+ sessions, first Claude Code session,
   first subagent use, first MCP integration.
4. **Now**: the current era. Recent trends, current provider mix, current
   project focus.
5. **Projection**: based on month-over-month trend, where might usage be in
   3 months? (Simple linear projection, clearly labeled as speculative.)

Discovery: no user input needed — this is the whole-archive story. The engine
computes era boundaries, finds milestones, and renders the full narrative.

CLI:

```
polylogue story ai-journey
polylogue story ai-journey --output markdown  # for sharing
```

## Implementation

All four story types are SQL-driven template renders. No LLM, no vector search,
no external services. Total implementation estimated at ~650 LOC of Python
(template rendering + story assembly + CLI wiring) with ~200 LOC of SQL.

### Query Patterns

**Feature Birth** (core query):

```sql
-- Find sessions touching a file, ordered by first occurrence
SELECT
    sp.session_id, sp.canonical_session_date, sp.title,
    ae.affected_paths
FROM session_profiles sp
JOIN actions ae ON sp.session_id = ae.session_id
WHERE ae.affected_paths LIKE '%' || :file_path || '%'
ORDER BY sp.canonical_session_date ASC;
```

**Day in the Life**:

```sql
-- Single day's sessions with enrichment
SELECT
    sp.session_id, sp.canonical_session_date, sp.title,
    sp.first_message_at, sp.last_message_at,
    sp.provider_name, sp.word_count, sp.tool_use_count,
    sp.thinking_count, sp.total_cost_usd, sp.total_duration_ms,
    sp.total_cost_usd, sp.repo_names_json, sp.tags_json,
    ae.affected_paths
FROM session_profiles sp
LEFT JOIN actions ae ON sp.session_id = ae.session_id
WHERE sp.canonical_session_date = :date
ORDER BY sp.first_message_at ASC;
```

**The Big Refactor** (phase detection):

```sql
-- Sessions spanning 3+ days on the same file set, with gap detection
SELECT
    sp.session_id, sp.canonical_session_date,
    sp.first_message_at, sp.word_count, sp.tool_use_count,
    ae.affected_paths
FROM session_profiles sp
JOIN actions ae ON sp.session_id = ae.session_id
WHERE sp.session_id IN (
    SELECT DISTINCT session_id
    FROM session_profiles
    WHERE auto_tags_json LIKE '%' || :branch || '%'
       OR search_text LIKE '%' || :branch || '%'
)
ORDER BY sp.canonical_session_date ASC;
```

**AI Journey** (era query):

```sql
-- Monthly aggregation for era detection
SELECT
    strftime('%Y-%m', canonical_session_date) AS month,
    provider_name,
    COUNT(*) AS session_count,
    SUM(word_count) AS total_words,
    SUM(tool_use_count) AS total_tool_uses,
    SUM(total_cost_usd) AS total_cost,
    COUNT(DISTINCT json_each.value) FILTER (
        WHERE json_valid(repo_names_json)
    ) AS unique_repos
FROM session_profiles,
     json_each(repo_names_json)
WHERE canonical_session_date IS NOT NULL
GROUP BY month, provider_name
ORDER BY month ASC;
```

### Template Rendering

Stories are rendered as Markdown with embedded links to the web UI or
local session files. No HTML, no JavaScript — the Markdown output is
readable in any editor, viewable on GitHub, and shareable as-is.

Each story section includes a "Verify" link pointing to the session
that sourced the claim. This is the provenance guarantee in action.

Example output (Feature Birth):

```markdown
# Feature Birth: Content Hash

## Genesis — 2025-11-03
First mentioned in a session with Claude Code while debugging
duplicate ingestion. The user asked:

> "Why does ingestion re-import the same session twice?"

[See full session →](polylogue://session/claude-code:abc123)

## Design Sessions
### 2025-11-03 (later same day)
Decision: use SHA-256 over NFC-normalized payload. Rejected
content-addressable blob store as premature.
[See full session →](polylogue://session/claude-code:def456)
...
```

## Verification

Stories can be verified against the archive. A `--verify` flag runs the
source queries and checks that every claim in the story matches current
archive state. If the archive has been re-ingested or sessions have been
deleted, verification will flag stale claims.

This is important because stories are shareable artifacts. A story rendered
in March and shared in May might have drifted. Verification makes drift
visible rather than silent.

## Non-Goals

- Not LLM-generated. The provenance guarantee requires deterministic rendering.
  An LLM summarization pass could be an optional post-processing step, but the
  base story must be SQL-derived.
- Not a charting engine. The story is text, not visualizations. Charts belong
  in the analytics surface.
- Not interactive. Stories are rendered, saved, and shared. Live story
  construction with parameter tweaking is a future enhancement.
- Not cross-archive. Stories are scoped to a single polylogue archive.

## Build Order

1. `polylogue story day-in-life --date` (simplest story type, single-query)
2. `polylogue story feature-birth --path` (file-driven discovery)
3. `polylogue story refactor --branch` (phase detection + gap analysis)
4. `polylogue story ai-journey` (full-archive synthesis)
5. `--verify` mode for drift detection
6. `--output json` for MCP/programmatic consumption
