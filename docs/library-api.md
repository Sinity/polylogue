[← Back to README](../README.md)

# Library API

Polylogue is designed library-first. The CLI wraps the Python API.

The public Python surface is split into:

- `polylogue`: archive-core access (`Polylogue`, `SyncPolylogue`, `ArchiveStats`,
  `Conversation`, `Message`, `SearchResult`)
- `polylogue.archive`: domain-model, query, projection, and semantic helpers
- precise modules for higher-order semantic analysis, storage, and reporting

## Basic Usage

```python
from polylogue.archive.filter.filters import ConversationFilter
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.backends.async_sqlite import SQLiteBackend

backend = SQLiteBackend()
repo = ConversationRepository(backend=backend)

# Search
results = await ConversationFilter(repo).contains("error").provider("claude-ai").list()
for conv in results:
    print(f"{conv.id}: {conv.display_title}")

# Single conversation
conv = await ConversationFilter(repo).id("abc123").first()
```

## Precise Module Imports

Semantic-analysis/reporting helpers are still public, but they are no longer
re-exported from package roots. Import them from their actual modules:

```python
from polylogue.archive.session.session_profile import build_session_profile, infer_auto_tags
from polylogue.archive.conversation.threads import build_session_threads
```

Durable archive products are public too:

```python
from polylogue import Polylogue
from polylogue.archive_products import (
    ArchiveDebtProductQuery,
    DaySessionSummaryProductQuery,
    ProviderAnalyticsProductQuery,
    SessionEnrichmentProductQuery,
    SessionPhaseProductQuery,
    SessionProfileProductQuery,
    SessionTagRollupQuery,
)

async with Polylogue() as archive:
    status = await archive.get_session_product_status()
    profiles = await archive.list_session_profile_products(
        SessionProfileProductQuery(
            provider="claude-code",
            session_date_since="2026-03-16",
            session_date_until="2026-03-16",
            limit=25,
        )
    )
    phases = await archive.list_session_phase_products(
        SessionPhaseProductQuery(provider="claude-code", kind="execution", limit=25)
    )
    enrichments = await archive.list_session_enrichment_products(
        SessionEnrichmentProductQuery(
            provider="claude-code",
            limit=25,
        )
    )
    tags = await archive.list_session_tag_rollup_products(
        SessionTagRollupQuery(provider="claude-code", since="2026-01-01")
    )
    days = await archive.list_day_session_summary_products(
        DaySessionSummaryProductQuery(provider="claude-code", since="2026-01-01")
    )
    analytics = await archive.list_provider_analytics_products(
        ProviderAnalyticsProductQuery(provider="claude-code")
    )
    debt = await archive.list_archive_debt_products(
        ArchiveDebtProductQuery(only_actionable=True)
    )
```

`SessionProfileProduct` exposes stable session semantics directly:

- `first_message_at`
- `canonical_session_date`
- `engaged_duration_ms`
- `engaged_minutes`
- `repo_names`
- `repo_paths`

`SessionWorkEventProduct` and `SessionPhaseProduct` expose timestamped timeline
rows that can be queried directly.

`SessionEnrichmentProduct` exposes the separate enrichment tier directly:

- `intent_summary`
- `outcome_summary`
- `blockers`
- `confidence`
- `support_level`
- `support_signals`
- `provenance`

Provider analytics and archive debt are public products too:

- `ProviderAnalyticsProduct`: provider-level conversation/message/tool/thinking metrics
- `ArchiveDebtProduct`: governed cleanup/repair debt with maintenance targets plus preview/apply/validation lineage

## Filter Chain API

```python
# Chainable, lazy evaluation (terminals are async)
results = await (ConversationFilter(repo)
    .contains("error")
    .contains("python")             # AND
    .provider("claude-ai", "chatgpt")  # OR
    .since("2025-01-01")
    .has("thinking")
    .limit(10)
    .list())                        # Terminal: await list(), first(), count(), delete()

# Exclusion filters
results = await (ConversationFilter(repo)
    .contains("error")
    .exclude_text("warning")
    .exclude_provider("gemini")
    .exclude_tag("archived")
    .list())

# Lightweight summaries (no message loading)
summaries = await (ConversationFilter(repo)
    .provider("claude-ai")
    .since("2025-01-01")
    .list_summaries())              # Returns ConversationSummary (no messages)

# Check if summaries are sufficient
f = ConversationFilter(repo).provider("claude-ai")
if f.can_use_summaries():
    results = f.list_summaries()    # Fast path
else:
    results = f.list()              # Loads full conversations

# Custom predicates
results = await (ConversationFilter(repo)
    .where(lambda c: len(c.messages) > 50)
    .list())

# Sorting and sampling
results = await (ConversationFilter(repo)
    .sort("tokens")
    .reverse()
    .sample(10)
    .list())

# Conversation structure filters
roots = await ConversationFilter(repo).is_root().list()
continuations = await ConversationFilter(repo).is_continuation().list()
```

## Available Filter Methods

| Method | Description |
|--------|-------------|
| `.contains(text)` | FTS term (chainable = AND) |
| `.exclude_text(text)` | Exclude FTS term |
| `.provider(*names)` | Include providers |
| `.exclude_provider(*names)` | Exclude providers |
| `.tag(*tags)` | Include tags |
| `.exclude_tag(*tags)` | Exclude tags |
| `.referenced_path(*terms)` | Require touched-path substrings |
| `.action(*kinds)` | Require semantic action kinds |
| `.exclude_action(*kinds)` | Exclude semantic action kinds |
| `.tool(*names)` | Require normalized tool names |
| `.exclude_tool(*names)` | Exclude normalized tool names |
| `.action_sequence(*kinds)` | Require ordered semantic action subsequence |
| `.action_text(*terms)` | Require text inside normalized action evidence |
| `.retrieval_lane(name)` | Choose retrieval lane: `auto`, `dialogue`, `actions`, `hybrid` |
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
| `.is_root()` | Root conversations only |
| `.is_continuation()` | Continuation conversations only |
| `.is_sidechain()` | Sidechain conversations only |
| `.has_branches()` | Conversations with branching messages |
| `.parent(id)` | Children of a given parent |

## Terminal Methods

| Method | Description |
|--------|-------------|
| `.list()` | Execute and return `list[Conversation]` |
| `.list_summaries()` | Execute and return `list[ConversationSummary]` (lightweight, no messages) |
| `.first()` | Execute and return first match or `None` |
| `.count()` | Execute and return count (uses SQL fast path when possible) |
| `.delete()` | Delete matching conversations (returns count deleted) |
| `.pick()` | Interactive picker (TTY) or first match (non-TTY) |
| `.can_use_summaries()` | Check if `list_summaries()` is valid for current filters |

## Pipeline (Run)

```python
import asyncio
from polylogue.config import get_config
from polylogue.pipeline.runner import run_sources

config = get_config()
result = asyncio.run(run_sources(config=config, stage="all"))
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
            archive.list_conversations(limit=10),
            archive.list_conversations(provider="claude-ai"),
        )

        print(f"Total: {stats.conversation_count} conversations")

        # Parallel batch retrieval (5-10x faster than sequential)
        ids = [c.id for c in recent]
        convs = await archive.get_conversations(ids)

        # Search with evidence snippets
        results = await archive.search("error handling", limit=20)
        for hit in results.hits:
            print(f"{hit.title}: {hit.snippet}")

        # Parse files
        result = await archive.parse_file("chatgpt_export.json")

        # Fluent filter (terminals are async)
        convs = await archive.filter().provider("claude-ai").contains("error").limit(10).list()

        # Rebuild search index
        await archive.rebuild_index()

asyncio.run(main())
```

### Async Methods

| Method | Description |
|--------|-------------|
| `get_conversation(id)` | Get single conversation by ID |
| `get_conversations(ids)` | Parallel batch fetch (5-10x faster) |
| `list_conversations(provider, limit)` | List with optional filtering |
| `search(query, limit, source, since)` | Search returning evidence snippets; text matches report message evidence and Drive/Gemini `provider_id` / `id` / `fileId` / `driveId` attachment-id matches report attachment evidence |
| `parse_file(path, source_name)` | Parse a single export file |
| `parse_sources(sources, download_assets)` | Parse from configured sources |
| `rebuild_index()` | Rebuild FTS5 search index |
| `stats()` | Archive statistics (returns `ArchiveStats`) |
| `filter()` | Fluent filter builder (sync, reuses `ConversationFilter`) |
| `get_session_product_status()` | Durable product readiness/freshness summary |
| `get_session_profile_product(id)` | Get one durable session-profile product |
| `list_session_profile_products(query)` | List durable session-profile products |
| `list_session_work_event_products(query)` | List durable work-event products |
| `list_work_thread_products(query)` | List durable work-thread products |
| `list_session_tag_rollup_products(query)` | List durable tag-rollup products |
| `list_day_session_summary_products(query)` | List durable day-summary products |
| `list_week_session_summary_products(query)` | List durable week-summary products |
| `list_maintenance_run_products(query)` | List durable maintenance lineage products |
| `list_provider_analytics_products(query)` | List provider-level analytics products |
| `list_archive_debt_products(query)` | List governed archive-debt products |

---

**See also:** [CLI Reference](cli-reference.md) · [Data Model](data-model.md) · [Configuration](configuration.md)
