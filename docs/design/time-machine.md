# Time Machine

Chronological browsing of AI session archives. "What was I working on in
March 2025?" — answered by navigating the archive as a temporal document, not
a query language.

## Problem

Answer "what was I doing then" without constructing a search query. The user
knows *when* but not *what*. Existing search tools require filtering on
content; time-based navigation should require only a date or month.

The data is already there — `created_at` and `updated_at` on sessions,
`canonical_session_date` on session profiles, timestamps on messages. The
missing piece is temporal presentation surfaces that let the user pan and zoom
through their own history.

## Surfaces

### Calendar Heatmap

A GitHub-style contribution grid showing AI usage intensity per day over the
full archive range. Each cell is a day; color intensity maps to session count,
message volume, or cost.

```
       Jan              Feb              Mar
   M  T  W  T  F     M  T  W  T  F     M  T  W  T  F
   1  2  3  4  5     3  4  5  6  7     3  4  5  6  7
   6  7  8  9 10     8  9 10 11 12     8  9 10 11 12
  ...
```

The grid answers "when was I active" in one glance. High-activity weeks jump
out. Gaps are visible. The eye catches patterns — weekend work, crunch weeks,
quiet months — that would be invisible in a list or table.

The heatmap can show three different metrics, toggled with a flag:

| Metric | SQL source | Use |
|--------|-----------|-----|
| Session count | `COUNT(*)` from `session_profiles` per day | Activity volume |
| Message volume | `SUM(word_count)` from `messages` grouped by `canonical_session_date` | Output intensity |
| Cost | `SUM(total_cost_usd)` from `session_profiles` per day | Spending awareness |

Default metric: session count (fastest, most intuitive).

Implementation is a single aggregation query:

```sql
SELECT
    canonical_session_date,
    COUNT(*) AS session_count,
    SUM(word_count) AS total_words,
    SUM(total_cost_usd) AS total_cost
FROM session_profiles
WHERE canonical_session_date BETWEEN :start AND :end
GROUP BY canonical_session_date
ORDER BY canonical_session_date;
```

### Era Detection

An *era* is a contiguous period with a stable activity signature. Polylogue
detects eras automatically from session density and project focus.

Eras answer "what phase of my AI usage was I in" without manual tagging:

| Era | Detection signal |
|-----|------------------|
| "Exploration" | Low density (<3 sessions/week), many different providers, no dominant repo |
| "Settled usage" | Medium density (3-10/week), one dominant provider, one primary repo |
| "Intensive phase" | High density (10+/week), single provider, single repo |
| "Multi-project" | High density but diverse repos, possibly multiple providers |
| "Dormant" | <1 session/week for 3+ consecutive weeks |
| "Claude Code adoption" | Transition from claude-ai to claude-code as dominant provider |

Era boundaries are recalculated on pipeline runs. No manual labeling needed.
The user sees named eras in the timeline, can rename them, and can jump to era
boundaries.

Detection algorithm (Python, ~80 LOC):

1. Pull daily session counts and provider distribution from `session_profiles`.
2. Apply a 7-day rolling window for density classification.
3. Detect provider transitions (e.g., `claude-ai` share drops below 20% while
   `claude-code` share rises above 50%) — mark as era boundary.
4. Detect density plateaus by looking for 14+ day runs at a consistent tier.
5. Name eras heuristically; user can rename.

### This Day in History

"What was I doing on this day last year?" A view that surfaces sessions from
the same calendar day in previous years.

The query:

```sql
SELECT
    sp.session_id,
    sp.title,
    sp.canonical_session_date,
    sp.provider_name,
    sp.word_count,
    sp.repo_names_json
FROM session_profiles sp
WHERE CAST(strftime('%m', sp.canonical_session_date) AS INTEGER) = :month
  AND CAST(strftime('%d', sp.canonical_session_date) AS INTEGER) = :day
  AND sp.canonical_session_date <= date('now', :lookback)
ORDER BY sp.canonical_session_date DESC
LIMIT 10;
```

The `--lookback` flag controls how far back to exclude (default `-1 year`, so
"this day last year or earlier"). The presentation groups results by year,
showing a descending list of anniversaries.

Bonus: "this week in history" variant that groups by ISO week and shows
the top sessions from that week across all past years.

### Session Filmstrip

A horizontal chronological view of sessions as a scrollable strip. Each session
is a card (title, date, duration, repo, cost). The user pans left and right
through time; clicking a card shows the full session.

The filmstrip is the "zoom in" companion to the heatmap's "pan out." The
heatmap shows the year at a glance; the filmstrip shows a month or week in
detail.

Implementation: paginated query over `session_profiles` ordered by
`canonical_session_date`. Each page returns 20 sessions. The UI renders them
as a horizontal strip with Prev/Next pagination.

```sql
SELECT
    session_id, title, canonical_session_date,
    provider_name, word_count, tool_use_count,
    total_cost_usd, total_duration_ms,
    repo_names_json, tags_json
FROM session_profiles
WHERE canonical_session_date BETWEEN :start AND :end
ORDER BY canonical_session_date ASC, first_message_at ASC
LIMIT :limit OFFSET :offset;
```

## CLI Surface

Four commands, all read-only:

```
polylogue calendar [--metric sessions|words|cost] [--year 2025] [--output grid|json]
polylogue eras [--list | --rename ERA NAME]
polylogue this-day [--lookback "1 year"] [--week]
polylogue timeline [--from DATE] [--to DATE] [--page N] [--limit 20]
```

### `polylogue calendar`

Prints the heatmap grid to the terminal. Uses Unicode block characters for
color intensity (no terminal color library dependency needed for the first
cut, though truecolor would be nice). JSON output for programmatic consumers.

```
polylogue calendar --year 2025

2025
    Jan  ░░█░░░█░░░░░░░░░█░░░░░░░█░░  4 sessions, 8,234 words
    Feb  ░░░░░█░████░░░░░░██░░░░█░░░  7 sessions, 18,412 words
    Mar  ███░█░████░░░██░░░█░░█░██░░  12 sessions, 45,011 words
    Apr  ░░░░░░████░░░░░░░██░░░░░░░░  5 sessions, 12,345 words
    ...
```

### `polylogue eras`

Lists detected eras with date ranges, labels, and session counts. Accepts
`--rename` to assign a custom name to an era (stored in session metadata
and honored on subsequent pipeline runs).

### `polylogue this-day`

Shows sessions from this calendar day in previous years, ordered by recency.
The `--week` variant shows the whole ISO week. If today is May 4 and the
lookback is "1 year", it shows May 4 sessions from 2025 and earlier.

### `polylogue timeline`

The filmstrip in CLI form. Prints a paginated list of sessions in date order,
each with title, date, repo context, and cost. Accepts `--from` and `--to`
for date range, `--page` / `--limit` for pagination.

## Data Sources

All queries run over existing tables. Zero new collection, zero new tables.

| Surface | Primary table | Secondary table |
|---------|--------------|-----------------|
| Calendar heatmap | `session_profiles` | `session_stats` (for word counts) |
| Era detection | `session_profiles` | `sessions` (for provider distribution) |
| This day in history | `session_profiles` | — |
| Session filmstrip | `session_profiles` | — |

## Performance Budget

| Surface | Target latency | Data touched |
|---------|---------------|-------------|
| Calendar heatmap | <50ms | Full scan of session_profiles (indexed on date) |
| Era detection | <100ms (recompute) | Full scan with grouping |
| This day in history | <10ms | Indexed point lookup on month+day |
| Session filmstrip | <30ms per page | Indexed range scan with LIMIT |

All within SQLite coprolite tier — single read-only connection, no background
tasks, no caching layer needed for archive sizes under 50K sessions.

## Presentation Tier

The CLI surfaces are the first delivery. Two presentation targets follow:

1. **Web dashboard panel**: the heatmap renders as an SVG or Canvas grid in the
   web UI. Clicking a day opens the filmstrip for that day.
2. **MCP tools**: `polylogue_calendar` and `polylogue_timeline` tools that
   return structured JSON, enabling agents to answer "what was I doing" queries.

## Non-Goals

- Not a full analytics dashboard. The heatmap is a single visualization, not
  a charting library. More complex analytics (cost trends, model comparisons)
  belong in a separate analytics surface.
- Not real-time. Era detection runs during pipeline execution, not on every
  query.
- Not a replacement for the search surface. Time-based browsing complements
  query-based search; it does not replace it.

## Build Order

1. `polylogue calendar` — heatmap query + terminal output (Unicode grid)
2. `polylogue timeline` — filmstrip query + paginated CLI output
3. `polylogue this-day` — anniversary query
4. `polylogue eras` — era detection algorithm + CLI listing + rename
5. Web dashboard heatmap panel
6. MCP tool equivalents
