# Whole Product Vision

Polylogue: your AI memory.

A local archive that remembers every AI session so you can find, trace,
and understand your AI-assisted work across days, months, and years.

## One-Line Description

Polylogue records every AI session, makes it searchable, and surfaces
patterns you didn't know were there — all from a local SQLite archive on your
machine.

## First Experience

### 30 seconds: bare `polylogue`

The user installs polylogue, runs it with no arguments, and sees:

```
$ polylogue

Polylogue — your AI memory
Archive: /home/user/.local/share/polylogue/
Sources configured: 1  (1 active, 0 stale)
Last ingestion: 2 minutes ago

Recent activity (7 days):
  claude-code  28 sessions  142,000 words  $3.42
  chatgpt       2 sessions    3,200 words  $0.00

  Today                                    This week
  polylogue/docs/design/time-machine.md   polylogue/storage/sqlite/archive_tiers/index.py
  ── 2h 15min, 4 tool calls             ── 8 sessions, $1.20

Quick search: polylogue "error handling"
Full help:    polylogue --help
```

This is the zero-configuration landing. The user sees that polylogue is
already working (if sources are configured), understands their usage at a
glance, and gets immediate next actions.

Time budget: <500ms wall clock. Achievable because the status query is a
handful of indexed `COUNT` and `SUM` queries.

### 5 minutes: first search

```
$ polylogue "schema rebuild"
  claude-code:abc123  2026-03-10  Schema is fresh-only, no in-place upgrade chain
  claude-code:def456  2026-03-15  Considered v3->v4 rebuild, accepted
  claude-code:ghi789  2026-04-02  FTS rowid repair — schema impact
  3 results (23ms)
```

The user searches for a concept they remember discussing. Results appear
instantly. Clicking any result (in the web UI) or running `polylogue show`
(in the CLI) opens the full session.

### 15 minutes: first insight

The user runs `polylogue stats` and sees something they didn't know:

```
$ polylogue stats
Archive: 2025-06-12 to 2026-05-04  (327 days)
Sessions: 1,847
Messages: 284,012
Total words: 12,847,234
Total cost: $847.23

Most active month: March 2026 (182 sessions, $92.41)
Most active day: 2026-03-17 (14 sessions, $7.80)
Provider split: claude-code 92%, chatgpt 5%, gemini 3%

Your archive contains 12,847,234 words.
That's roughly the length of 7 copies of "War and Peace."
```

This is the hook. The user didn't know March 2026 was their busiest AI month.
They didn't know they'd spent $847. The "War and Peace" comparison makes the
scale tangible.

## One-Week Habit Formation

### SessionStart hook (immediate value)

The SessionStart hook injects yesterday's summary into the agent's context
before the first prompt. The user doesn't need to run a command — context
arrives automatically.

```
[Polylogue — recent activity in polylogue/]
Yesterday: feature/fix/schema-annotations — 4h, 12 tool calls, $0.87
   Files: schema_inference.py, verification.py, runtime.py
   Outcome: PR #795 opened

3 days ago: fixed FTS trigger leak — 2h, $0.42
```

The user sees this in their first agent interaction of the day. It becomes
background awareness — "oh right, I was working on schema annotations."

### Daily command: `polylogue --since yesterday`

```
$ polylogue --since yesterday

Yesterday — 2026-05-03
  claude-code  4 sessions  18,000 words  $0.87
  Projects: polylogue (4)

  feature/fix/schema-annotations               $0.52  3,200 words
  devtools verify                               $0.21  1,100 words
  PR body writing                               $0.08    400 words
  SessionStart context check                    $0.06    300 words

Cost yesterday: $0.87
Cost this week: $3.42
Cost this month: $0.87
```

A 2-second habit that answers "what did I do yesterday" and "what did it cost."
The same command works with `--since monday`, `--since "last week"`, or any
natural-language time expression.

### Weekly check: `polylogue cost`

```
$ polylogue cost --this-week
Week of 2026-04-28 — 2026-05-04
  claude-code  32 sessions  $18.42
  chatgpt       2 sessions   $0.00
  Total:                    $18.42

Cost trend:
  This week:   $18.42  ████████░░
  Last week:   $22.10  ██████████░
  2 weeks ago: $14.30  ██████░░░░

Projection (month): ~$78 based on current rate.
```

The user develops cost awareness. No surprises at the end of the month.

## One-Month Indispensability

### Project memory (context that persists)

After a month, the user has 30+ project memory entries accumulated across
their repos. The SessionStart hook now injects:

```
[Project Memory — polylogue]
Recent decisions (3):
- Schema is fresh-only, no in-place upgrade chain (2026-03-10, 55 days ago)
- Content hash excludes user metadata (2026-03-15, 50 days ago)
- FTS5 tokenizer is unicode61, no porter stemmer (2026-03-20, 45 days ago)
```

The agent knows these decisions without being told. The user doesn't have to
repeat themselves or rediscover failed approaches.

### Time machine (a year of history)

With a year of archive data, the time machine surfaces patterns:

```
$ polylogue calendar --year 2025

2025
    Jun  █░░░░░░░░░░░░░  1 session   — first contact
    Jul  ███░░░░█████░░  8 sessions  — exploration
    Aug  ███░█░█████░░░  10 sessions
    Sep  ███░█░████░░░░  8 sessions
    Oct  ██░██░████░░░░  9 sessions
    Nov  █████████████░  14 sessions — Claude Code adoption
    Dec  █████░████░░░░  9 sessions
```

The heatmap reveals the adoption arc — from cautious exploration in June to
daily usage by November.

### Session continuity (seamless handoff)

The user finishes a session, opens a new one an hour later. The SessionStart
hook injects:

```
[Polylogue — continuing from 1 hour ago]
Previous session: feature/fix/schema-annotations (42 min)
  Last action: ran devtools verify --quick
  Outcome: format check passed, lint passed, 4 mypy errors remain
  Files in play: polylogue/schemas/inference/semantic/runtime.py,
                 polylogue/schemas/verification.py

4 unresolved tasks from this session:
  - Fix mypy errors in runtime.py (labeling)
  - Add test for new annotation type
  - Update docs/schema-annotations.md
  - Open PR
```

The user picks up exactly where they left off. The agent already knows the
context. Zero time spent re-establishing state.

## Design Values

### Speed

Every interaction must feel instant. The archive is local — there is no
network latency, no server round-trip, no cloud dependency.

| Interaction | Target | Reality |
|-------------|--------|---------|
| Bare `polylogue` | <500ms | Handful of indexed counts |
| FTS5 search | <2s | Single SQLite query, unicode61 tokenizer |
| Heatmap render | <50ms | One aggregation scan |
| Story render | <5s | Multiple queries + template assembly |
| Web UI first paint | <1s | Static files + API calls to local daemon |

If any operation exceeds its target, the performance is treated as a bug.

### Trust

Every number in polylogue has provenance. The user should never wonder
"where did that come from?"

- **Content hashing**: archive writes are idempotent by SHA-256. If a
  session was ingested, its content hasn't been silently modified.
- **Verification dashboard**: `polylogue ops doctor` audits the archive for
  integrity — schema consistency, foreign key violations, orphan records,
  FTS5 index health.
- **Provenance links**: stories and insights link back to source sessions.
  Claims are verifiable.
- **No inference without evidence**: derived data (session profiles, era
  labels) is stored alongside its provenance — materializer version, input
  data hash, compute timestamp. When inputs change, stale derivations are
  visible.

### Discovery

The user should find things without knowing what to search for. The time
machine, stories, and heatmap are discovery surfaces — they reveal patterns
the user wasn't looking for.

| Surface | Discovery mode |
|---------|---------------|
| Calendar heatmap | "I didn't realize I was that active in March" |
| This day in history | "Oh right, I was debugging that exact same thing last year" |
| Era detection | "That was my Claude Code adoption month" |
| Feature Birth | "That feature took 12 sessions across 3 weeks" |
| AI Journey | "I've gone from 1 session/week to 5 sessions/day" |
| Stats | "I've written 12 million words to AI — that's 20 novels" |

No query language required. The user pans and zooms through their own history.

## Surface Matrix

Every feature is available through every surface. The surface determines
presentation, not capability.

| Feature | CLI | MCP | Web |
|---------|-----|-----|-----|
| Search | `polylogue "query"` | `polylogue_search` | Search bar |
| List sessions | `polylogue list --since ...` | `polylogue_list_sessions` | Session list |
| Show session | `polylogue show <id>` | `polylogue_get_session` | Session detail page |
| Stats | `polylogue stats` | `polylogue_get_stats` | Dashboard stats panel |
| Cost | `polylogue cost` | `polylogue_get_cost` | Dashboard cost panel |
| Calendar heatmap | `polylogue calendar` | `polylogue_calendar` | Calendar page |
| Timeline | `polylogue timeline` | `polylogue_timeline` | Timeline page |
| Eras | `polylogue eras` | `polylogue_get_eras` | Era browser |
| This day | `polylogue this-day` | `polylogue_this_day` | "On this day" panel |
| Feature birth | `polylogue story feature-birth` | `polylogue_story_feature_birth` | Story page |
| Day in life | `polylogue story day-in-life` | `polylogue_story_day_in_life` | Story page |
| Refactor story | `polylogue story refactor` | `polylogue_story_refactor` | Story page |
| AI journey | `polylogue story ai-journey` | `polylogue_story_ai_journey` | Story page |
| Project memory | `polylogue memory` | `polylogue_get_project_memory` | Memory page |
| Tags | `polylogue tags` | `polylogue_tag_session` | Tag editor |
| Health check | `polylogue ops doctor` | `polylogue_health` | Health dashboard |
| MCP server | `polylogue mcp` | (self) | — |
| Daemon | `polylogued` | `polylogue_shutdown` | Daemon status |

The CLI is the primary human interface. MCP is the primary agent interface.
Web is the exploration and sharing interface. Same data, same operations,
different presentations.

## Non-Goals

- Not a cloud service. Polylogue data stays on the user's machine. No
  telemetry, no sync, no accounts.
- Not a replacement for CLAUDE.md. Polylogue provides context *about* AI
  usage; CLAUDE.md provides instructions *for* AI usage.
- Not a project management tool. Polylogue surfaces work that happened; it
  does not plan work that should happen.
- Not a code search engine. FTS5 indexes session text, not code
  repositories. Cross-reference between sessions and code is via
  `actions.affected_paths`, not via code indexing.
- Not a general note-taking tool. Polylogue is for AI sessions
  specifically. General notes, todos, and documentation belong in a
  knowledgebase, not in the archive.

## Build Order

This is a 12-18 month vision. The build order prioritizes immediate value:

1. **Ship the core** (current): ingestion, FTS5 search, session profiles.
2. **Surface the basics**: bare `polylogue` status, `polylogue stats`,
   `polylogue cost`, `polylogue --since`.
3. **Add discovery**: calendar heatmap, timeline, this-day-in-history.
4. **Add narratives**: day-in-life story, feature birth story, era detection.
5. **Add persistence**: project memory table, SessionStart hook injection.
6. **Add the big story**: AI journey, refactor story.
7. **Polish the web surface**: dashboard with heatmap, story pages, health panel.
8. **Era detection v2**: machine learning on session embeddings (far future,
   speculative).

Each step delivers value independently. The user doesn't need to wait for
step 8 to benefit from step 1.
