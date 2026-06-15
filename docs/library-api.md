[← Back to README](../README.md)

# Library API

Polylogue is designed library-first. The CLI wraps the Python API.

The public Python surface is split into:

- `polylogue`: archive-core access (`Polylogue`, `SyncPolylogue`, `ArchiveStats`,
  `Session`, `Message`, `SearchResult`)
- `polylogue.archive`: domain-model, query, projection, and semantic helpers
- precise modules for higher-order semantic analysis, storage, and reporting

## Basic Usage

```python
import asyncio

from polylogue import Polylogue


async def main() -> None:
    async with Polylogue.open() as archive:
        # Full-text search (returns list[Session])
        results = await archive.repository.search("error", limit=10)
        for session in results:
            print(f"{session.id}: {session.display_title}")

        # Fluent filter by origin (lazy; terminals are async)
        recent = await archive.filter().origin("claude-ai-export").limit(10).list_summaries()
        for summary in recent:
            print(f"{summary.id}: {summary.display_title}")

        # Single session by id prefix
        session = await archive.get_session("abc123")


asyncio.run(main())
```

`Polylogue.open()` is the async context manager; `archive.repository` is a
`SessionRepository` and `archive.filter()` returns a `SessionFilter` already
bound to the active archive — never construct `SessionFilter` directly.

## Precise Module Imports

Semantic-analysis/reporting helpers are still public, but they are no longer
re-exported from package roots. Import them from their actual modules:

```python
from polylogue.archive.session.session_profile import build_session_profile, infer_auto_tags
from polylogue.archive.session.threads import build_session_threads
```

Durable archive insights are public too:

```python
from polylogue import Polylogue
from polylogue.insights.archive import (
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsightQuery,
    SessionLatencyProfileInsightQuery,
    SessionPhaseInsightQuery,
    SessionProfileInsightQuery,
    SessionTagRollupQuery,
)

async with Polylogue() as archive:
    status = await archive.get_session_insight_status()
    profiles = await archive.list_session_profile_insights(
        SessionProfileInsightQuery(
            provider="claude-code",
            session_date_since="2026-03-16",
            session_date_until="2026-03-16",
            limit=25,
        )
    )
    phases = await archive.list_session_phase_insights(
        SessionPhaseInsightQuery(provider="claude-code", kind="execution", limit=25)
    )
    latency = await archive.list_session_latency_profile_insights(
        SessionLatencyProfileInsightQuery(provider="claude-code", only_stuck=False, limit=25)
    )
    tags = await archive.list_session_tag_rollup_insights(
        SessionTagRollupQuery(provider="claude-code", since="2026-01-01")
    )
    coverage = await archive.list_archive_coverage_insights(
        ArchiveCoverageInsightQuery(provider="claude-code", group_by="day", since="2026-01-01")
    )
    debt = await archive.list_archive_debt_insights(
        ArchiveDebtInsightQuery(only_actionable=True)
    )
```

`SessionProfileInsight` exposes stable session semantics directly:

- `inferred_topic`
- `first_message_at`
- `canonical_session_date`
- `timestamp_source`
- `engaged_duration_ms`
- `engaged_minutes`
- `tool_active_duration_ms`
- `workflow_shape`
- `workflow_shape_confidence`
- `terminal_state`
- `terminal_state_confidence`
- `enrichment`
- `enrichment_provenance`
- `repo_names`
- `repo_paths`

`SessionWorkEventInsight` and `SessionPhaseInsight` expose timestamped timeline
rows that can be queried directly.

`SessionLatencyProfileInsight` exposes per-session runtime-shape signals:

- `median_tool_call_ms`
- `p90_tool_call_ms`
- `max_tool_call_ms`
- `stuck_tool_count`
- `median_agent_response_ms`
- `median_user_response_ms`
- `tool_call_count_by_category`

The latency payload includes a construct-boundary string because these are
archive-observed timing aggregates. They do not measure correctness, human
attention, or total wall-clock productivity.

The merged `SessionProfileInsight` tier exposes probabilistic enrichment:

- `intent_summary`
- `outcome_summary`
- `blockers`
- `confidence`
- `support_level`
- `support_signals`
- `provenance`

The enrichment payload is intentionally folded into session profiles because it
is derived from the same per-session materialization row. Request
`SessionProfileInsightQuery(tier="merged")` when callers need both grounded
profile evidence and probabilistic enrichment in one payload.

Archive coverage and archive debt are public insights too:

- `ArchiveCoverageInsight`: provider/day/week session, message, cost, and activity coverage rollups
- `ArchiveDebtInsight`: governed cleanup/repair debt with maintenance targets plus preview/apply/validation lineage

## Filter Chain API

All examples below assume an open archive, where `archive.filter()` yields a
fresh `SessionFilter` bound to the active archive:

```python
async with Polylogue.open() as archive:
    # Chainable, lazy evaluation (terminals are async)
    results = await (archive.filter()
        .contains("error")
        .contains("python")                          # AND
        .origin("claude-ai-export", "chatgpt-export")  # OR
        .since("2025-01-01")
        .has("thinking")
        .limit(10)
        .list())                        # Terminal: await list(), first(), count(), delete()

    # Exclusion filters
    results = await (archive.filter()
        .contains("error")
        .exclude_text("warning")
        .exclude_origin("gemini-cli-session")
        .exclude_tag("archived")
        .list())

    # Lightweight summaries (no message loading)
    summaries = await (archive.filter()
        .origin("claude-ai-export")
        .since("2025-01-01")
        .list_summaries())              # Returns SessionSummary (no messages)

    # Check if summaries are sufficient
    f = archive.filter().origin("claude-ai-export")
    if f.can_use_summaries():
        results = await f.list_summaries()  # Fast path
    else:
        results = await f.list()            # Loads full sessions

    # Custom predicates
    results = await (archive.filter()
        .where(lambda c: len(c.messages) > 50)
        .list())

    # Sorting and sampling
    results = await (archive.filter()
        .sort("tokens")
        .reverse()
        .sample(10)
        .list())

    # Session structure filters
    roots = await archive.filter().is_root().list()
    continuations = await archive.filter().is_continuation().list()
```

## Available Filter Methods

| Method | Description |
|--------|-------------|
| `.contains(text)` | FTS term (chainable = AND) |
| `.exclude_text(text)` | Exclude FTS term |
| `.origin(*names)` | Include origins (e.g. `claude-ai-export`, `chatgpt-export`) |
| `.exclude_origin(*names)` | Exclude origins |
| `.repo(*names)` | Require repository names |
| `.tag(*tags)` | Include tags |
| `.exclude_tag(*tags)` | Exclude tags |
| `.referenced_path(pattern)` | Require a touched-path substring |
| `.cwd_prefix(prefix)` | Require a working-directory prefix |
| `.action(*kinds)` | Require semantic action kinds |
| `.exclude_action(*kinds)` | Exclude semantic action kinds |
| `.tool(*names)` | Require normalized tool names |
| `.exclude_tool(*names)` | Exclude normalized tool names |
| `.has(*types)` | Content types: `thinking`, `tools`, `summary`, `attachments` |
| `.title(pattern)` | Title contains pattern |
| `.id(prefix)` | ID prefix match |
| `.since(date)` | After date (str or datetime) |
| `.until(date)` | Before date (str or datetime) |
| `.similar(text)` | Semantic similarity (requires vector index) |
| `.sort(field)` | Sort: `date`, `tokens`, `messages`, `words`, `longest`, `random` |
| `.reverse()` | Reverse sort order |
| `.limit(n)` | Max results |
| `.sample(n)` | Random sample |
| `.where(predicate)` | Custom filter predicate |
| `.is_root()` | Root sessions only |
| `.is_continuation()` | Continuation sessions only |
| `.is_sidechain()` | Sidechain sessions only |
| `.has_branches()` | Sessions with branching messages |
| `.parent(id)` | Children of a given parent |

## Terminal Methods

| Method | Description |
|--------|-------------|
| `.list()` | Execute and return `list[Session]` |
| `.list_summaries()` | Execute and return `list[SessionSummary]` (lightweight, no messages) |
| `.first()` | Execute and return first match or `None` |
| `.count()` | Execute and return count (uses SQL fast path when possible) |
| `.delete()` | Delete matching sessions (returns count deleted) |
| `.can_use_summaries()` | Check if `list_summaries()` is valid for current filters |

## Ingestion

```python
import asyncio
from polylogue import Polylogue

async def main():
    async with Polylogue() as archive:
        result = await archive.parse_sources()
        return result.counts

counts = asyncio.run(main())
```

## Async API

Polylogue provides a full async/await facade with concurrent operations:

```python
import asyncio
from polylogue import Polylogue

async def main():
    async with Polylogue() as archive:
        # Concurrent queries
        stats, recent, claude = await asyncio.gather(
            archive.stats(),
            archive.list_sessions(limit=10),
            archive.list_sessions(origin="claude-ai-export"),
        )

        print(f"Total: {stats.session_count} sessions")

        # Parallel batch retrieval (5-10x faster than sequential)
        ids = [c.id for c in recent]
        convs = await archive.get_sessions(ids)

        # Search with evidence snippets
        results = await archive.search("error handling", limit=20)
        for hit in results.hits:
            print(f"{hit.title}: {hit.snippet}")

        # Parse files
        result = await archive.parse_file("chatgpt_export.json")

        # Fluent filter (terminals are async)
        convs = await archive.filter().origin("claude-ai-export").contains("error").limit(10).list()

        # Rebuild search index
        await archive.rebuild_index()

asyncio.run(main())
```

### Async Methods

| Method | Description |
|--------|-------------|
| `get_session(id)` | Get single session by ID |
| `get_sessions(ids)` | Parallel batch fetch (5-10x faster) |
| `list_sessions(origin, limit)` | List with optional filtering |
| `search(query, limit, source, since)` | Search returning evidence snippets; text matches report message evidence and Drive/Gemini `provider_id` / `id` / `fileId` / `driveId` attachment-id matches report attachment evidence |
| `parse_file(path, source_name)` | Parse a single export file |
| `parse_sources(sources, download_assets)` | Parse from configured sources |
| `rebuild_index()` | Rebuild FTS5 search index |
| `stats()` | Archive statistics (returns `ArchiveStats`) |
| `filter()` | Fluent filter builder (sync, reuses `SessionFilter`) |
| `get_session_insight_status()` | Durable insight readiness/freshness summary |
| `get_session_profile_insight(id)` | Get one durable session-profile insight |
| `list_session_profile_insights(query)` | List durable session-profile insights |
| `get_session_latency_profile_insight(id)` | Get one durable session-latency insight |
| `list_session_latency_profile_insights(query)` | List durable session-latency insights |
| `find_stuck_session_latency_profile_insights(query)` | List sessions with stuck tool starts |
| `list_session_work_event_insights(query)` | List durable work-event insights |
| `list_thread_insights(query)` | List durable work-thread insights |
| `list_session_tag_rollup_insights(query)` | List durable tag-rollup insights |
| `list_archive_coverage_insights(query)` | List provider, day, or week archive coverage insights |
| `list_tool_usage_insights(query)` | Per-provider tool usage with explicit coverage gaps |
| `list_archive_debt_insights(query)` | List governed archive-debt insights |

---

**See also:** [CLI Reference](cli-reference.md) · [Data Model](data-model.md) · [Configuration](configuration.md)
