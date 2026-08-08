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
    phases = await archive.list_session_phase_insights(SessionPhaseInsightQuery(provider="claude-code", limit=25))
    latency = await archive.list_session_latency_profile_insights(
        SessionLatencyProfileInsightQuery(provider="claude-code", only_stuck=False, limit=25)
    )
    tags = await archive.list_session_tag_rollup_insights(
        SessionTagRollupQuery(provider="claude-code", since="2026-01-01")
    )
    coverage = await archive.list_archive_coverage_insights(
        ArchiveCoverageInsightQuery(provider="claude-code", group_by="day", since="2026-01-01")
    )
    debt = await archive.list_archive_debt_insights(ArchiveDebtInsightQuery(only_actionable=True))
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
- `terminal_state_method`
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
    results = await (
        archive.filter()
        .contains("error")
        .contains("python")  # AND
        .origin("claude-ai-export", "chatgpt-export")  # OR
        .since("2025-01-01")
        .has("thinking")
        .limit(10)
        .list()
    )  # Terminal: await list(), first(), count(), delete()

    # Exclusion filters
    results = await (
        archive.filter()
        .contains("error")
        .exclude_text("warning")
        .exclude_origin("gemini-cli-session")
        .exclude_tag("archived")
        .list()
    )

    # Lightweight summaries (no message loading)
    summaries = await (
        archive.filter().origin("claude-ai-export").since("2025-01-01").list_summaries()
    )  # Returns SessionSummary (no messages)

    # Check if summaries are sufficient
    f = archive.filter().origin("claude-ai-export")
    if f.can_use_summaries():
        results = await f.list_summaries()  # Fast path
    else:
        results = await f.list()  # Loads full sessions

    # Custom predicates
    results = await archive.filter().where(lambda c: len(c.messages) > 50).list()

    # Sorting and sampling
    results = await archive.filter().sort("tokens").reverse().sample(10).list()

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

        # Explain import decisions without writing archive rows
        explain = await archive.explain_import("chatgpt_export.json", source_name="chatgpt")
        print(explain.produced.sessions, explain.entries[0].provider)

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
| `explain_import(path, source_name, limit)` | Explain provider detection, artifact classification, parser mode, produced row counts, skips, and caveats without writing archive rows |
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

<!-- BEGIN GENERATED API OPERATION PARITY -->

## Generated facade operation index

This reference is generated from `polylogue/api/operation_parity.py`. Each live public facade callable is bound to a stable semantic operation ID; exported data models and adapter helpers are listed as intentional exclusions in the committed [machine-readable matrix](generated/api-operation-parity.json).

### Lifecycle and builders

#### `api.lifecycle.construct`

Construct, open, and close a facade bound to one archive runtime.

Route/tier class: `lifecycle`. CLI: Intentional absence: `polylogue-s1kr`. MCP: Intentional absence: `polylogue-s1kr`.

| Python callable | Signature |
|---|---|
| `Polylogue` | Constructed facade builder |
| `Polylogue.__init__` | `(archive_root: 'str | Path | None' = None, db_path: 'str | Path | None' = None, *, runtime: 'ResolvedRuntimeConfig | None' = None, config: 'Config | None' = None) -> 'None'` |
| `Polylogue.open` | `(*, config: 'Config | None' = None, runtime: 'ResolvedRuntimeConfig | None' = None, **kwargs: 'object') -> 'Polylogue'` |
| `Polylogue.__aenter__` | `async (self) -> 'Polylogue'` |
| `Polylogue.__aexit__` | `async (self, exc_type: 'object', exc_val: 'object', exc_tb: 'object') -> 'None'` |
| `Polylogue.close` | `async (self) -> 'None'` |

### Embedding readiness

#### `api.embedding.status`

Read the no-spend embedding readiness state.

Route/tier class: `embedding-status`. CLI: `ops embed status`. MCP: `status`.

| Python callable | Signature |
|---|---|
| `Polylogue.embedding_status` | `(self, *, detail: 'bool' = False) -> 'dict[str, object]'` |

#### `api.embedding.preflight`

Calculate a bounded no-provider-call embedding catch-up window.

Route/tier class: `embedding-preflight`. CLI: `ops embed preflight`. MCP: `status`.

| Python callable | Signature |
|---|---|
| `Polylogue.embedding_preflight` | `(self, *, rebuild: 'bool' = False, max_sessions: 'int | None' = None, max_messages: 'int | None' = None, max_cost_usd: 'float | None' = None) -> 'dict[str, object]'` |

### Embedding retrieval

#### `api.embedding.search`

Search stored session vectors using the embeddings tier.

Route/tier class: `embedding-read`. CLI: `find similar`. MCP: `query`.

| Python callable | Signature |
|---|---|
| `Polylogue.search_similar_sessions` | `async (self, session_id: 'str', *, limit: 'int' = 10, vector_provider: 'VectorProvider | None' = None, voyage_api_key: 'str | None' = None) -> 'dict[str, object]'` |

### Ingestion and derived maintenance

#### `api.ingest.parse`

Parse configured or explicit sources into source and index tiers.

Route/tier class: `source-index-write`. CLI: `import`. MCP: `run`.

| Python callable | Signature |
|---|---|
| `Polylogue.parse_file` | `async (self, path: 'str | Path', *, source_name: 'str | None' = None) -> 'ParseResult'` |
| `Polylogue.parse_sources` | `async (self, sources: 'list[Source] | None' = None, *, download_assets: 'bool' = True) -> 'ParseResult'` |

#### `api.index.rebuild`

Rebuild or update the derived index through the mutation executor.

Route/tier class: `index-write`. CLI: `ops reset --index`. MCP: `maintenance`.

| Python callable | Signature |
|---|---|
| `Polylogue.rebuild_index` | `async (self) -> 'bool'` |
| `Polylogue.update_index` | `async (self, session_ids: 'list[str]') -> 'bool'` |
| `Polylogue.rebuild_insights` | `async (self, session_ids: 'Sequence[str] | None' = None, *, progress_callback: 'ProgressCallback | None' = None) -> 'SessionInsightCounts'` |

### Archive reads

#### `api.archive.session-read`

Read sessions, summaries, messages, actions, and archive statistics from the index tier.

Route/tier class: `index-read`. CLI: `find`, `read`. MCP: `query`, `read`, `get`, `status`.

| Python callable | Signature |
|---|---|
| `Polylogue.get_session` | `async (self, session_id: 'str', *, content_projection: 'ContentProjectionSpec | None' = None) -> 'Session | None'` |
| `Polylogue.get_sessions` | `async (self, session_ids: 'list[str]', *, content_projection: 'ContentProjectionSpec | None' = None) -> 'list[Session]'` |
| `Polylogue.get_actions_batch` | `async (self, session_ids: 'builtins.list[str]') -> 'dict[str, tuple[Action, ...]]'` |
| `Polylogue.list_sessions` | `async (self, origin: 'str | None' = None, limit: 'int | None' = None, content_projection: 'ContentProjectionSpec | None' = None) -> 'list[Session]'` |
| `Polylogue.list_summaries` | `async (self, *, limit: 'int | None' = 50, offset: 'int' = 0, origin: 'str | None' = None) -> 'builtins.list[SessionSummary]'` |
| `Polylogue.list_sessions_for_spec` | `async (self, spec: 'SessionQuerySpec', *, content_projection: 'ContentProjectionSpec | None' = None) -> 'list[Session]'` |
| `Polylogue.search_session_hits` | `async (self, spec: 'SessionQuerySpec') -> 'builtins.list[SessionSearchHit]'` |
| `Polylogue.search` | `async (self, query: 'str', *, limit: 'int' = 100, source: 'str | None' = None, since: 'str | None' = None) -> 'SearchResult'` |
| `Polylogue.search_envelope` | `async (self, query: 'str', *, limit: 'int' = 50, offset: 'int' = 0, origin: 'str | None' = None, since: 'str | None' = None, until: 'str | None' = None, retrieval_lane: 'str' = 'auto', sort: 'str | None' = None, cursor: 'str | None' = None) -> 'SearchEnvelope'` |
| `Polylogue.archive_count_sessions` | `async (self, *, origin: 'str | None' = None, excluded_origins: 'Sequence[str]' = (), tags: 'Sequence[str]' = (), excluded_tags: 'Sequence[str]' = (), repo_names: 'Sequence[str]' = (), project_refs: 'Sequence[str]' = (), has_types: 'Sequence[str]' = (), has_tool_use: 'bool' = False, has_thinking: 'bool' = False, has_paste: 'bool' = False, tool_terms: 'Sequence[str]' = (), excluded_tool_terms: 'Sequence[str]' = (), action_terms: 'Sequence[str]' = (), excluded_action_terms: 'Sequence[str]' = (), action_sequence: 'Sequence[str]' = (), action_text_terms: 'Sequence[str]' = (), referenced_paths: 'Sequence[str]' = (), cwd_prefix: 'str | None' = None, typed_only: 'bool' = False, message_type: 'str | None' = None, title: 'str | None' = None, min_messages: 'int | None' = None, max_messages: 'int | None' = None, min_words: 'int | None' = None, max_words: 'int | None' = None, since: 'str | None' = None, until: 'str | None' = None) -> 'int'` |
| `Polylogue.archive_get_session` | `async (self, session_id: 'str') -> 'ArchiveSessionEnvelope | None'` |
| `Polylogue.get_messages_paginated` | `async (self, session_id: 'str', *, message_role: 'MessageRoleFilter' = (), message_type: 'MessageTypeName | None' = None, material_origin: 'tuple[MaterialOrigin, ...]' = (), limit: 'int' = 50, offset: 'int' = 0, content_projection: 'ContentProjectionSpec | None' = None) -> 'tuple[list[Message], int, LineageCompleteness]'` |
| `Polylogue.iter_messages` | `(self, session_id: 'str', *, message_roles: 'MessageRoleFilter' = (), material_origin: 'tuple[MaterialOrigin, ...]' = (), limit: 'int | None' = None) -> 'AsyncIterator[Message]'` |
| `Polylogue.bulk_get_messages` | `async (self, session_ids: 'Sequence[str]', *, since: 'str | None' = None, until: 'str | None' = None, message_role: 'MessageRoleFilter' = (), material_origin: 'tuple[MaterialOrigin, ...]' = (), content_projection: 'ContentProjectionSpec | None' = None) -> 'dict[str, list[Message]]'` |
| `Polylogue.query_sessions` | `async (self, *, origin: 'str | None' = None, tag: 'str | None' = None, since: 'str | None' = None, until: 'str | None' = None, sort: 'str | None' = None, limit: 'int | None' = None, offset: 'int' = 0, has_tool_use: 'bool' = False, has_thinking: 'bool' = False, has_paste: 'bool' = False, typed_only: 'bool' = False, min_messages: 'int | None' = None, max_messages: 'int | None' = None, min_words: 'int | None' = None, **kwargs: 'object') -> 'builtins.list[dict[str, object]]'` |
| `Polylogue.count_sessions` | `async (self, *, origin: 'str | None' = None, since: 'str | None' = None, until: 'str | None' = None, **kwargs: 'object') -> 'int'` |
| `Polylogue.get_session_summary` | `async (self, session_id: 'str') -> 'SessionSummary | None'` |
| `Polylogue.get_session_stats` | `async (self, session_id: 'str') -> 'dict[str, int]'` |
| `Polylogue.get_stats_by` | `async (self, group_by: 'str' = 'origin') -> 'dict[str, int]'` |
| `Polylogue.get_index_status` | `async (self) -> 'IndexStatus'` |
| `Polylogue.stats` | `async (self) -> 'ArchiveStats'` |
| `Polylogue.storage_stats` | `async (self) -> 'StorageArchiveStats'` |
| `Polylogue.facets` | `async (self, spec: 'SessionQuerySpec | None' = None, *, include_idf: 'bool' = True, include_deferred: 'bool' = True) -> 'FacetsResponse'` |
| `Polylogue.health_check` | `async (self) -> 'ReadinessReport'` |
| `Polylogue.filter` | `(self) -> 'SessionFilter'` |
| `Polylogue.list_read_view_profiles` | `async (self) -> 'list[JSONDocument]'` |

#### `api.archive.query-analysis`

Compile, explain, diagnose, and resolve archive query and reference projections.

Route/tier class: `index-read`. CLI: `find`, `read`, `analyze`. MCP: `query`, `read`, `get`, `explain`.

| Python callable | Signature |
|---|---|
| `Polylogue.explain_query_expression` | `async (self, expression: 'str') -> 'JSONDocument'` |
| `Polylogue.query_units` | `async (self, expression: 'str | None' = None, *, limit: 'int | None' = None, offset: 'int | None' = None, origin: 'str | None' = None, origins: 'tuple[str, ...]' = (), excluded_origins: 'tuple[str, ...]' = (), tag: 'str | None' = None, tags: 'tuple[str, ...]' = (), excluded_tags: 'tuple[str, ...]' = (), repo: 'str | None' = None, repo_names: 'tuple[str, ...]' = (), project: 'str | None' = None, project_refs: 'tuple[str, ...]' = (), has_types: 'tuple[str, ...]' = (), tool_terms: 'tuple[str, ...]' = (), excluded_tool_terms: 'tuple[str, ...]' = (), action_terms: 'tuple[str, ...]' = (), excluded_action_terms: 'tuple[str, ...]' = (), action_sequence: 'tuple[str, ...]' = (), action_text_terms: 'tuple[str, ...]' = (), referenced_paths: 'tuple[str, ...]' = (), cwd_prefix: 'str | None' = None, title: 'str | None' = None, since: 'str | None' = None, until: 'str | None' = None, has_tool_use: 'bool' = False, has_thinking: 'bool' = False, has_paste: 'bool' = False, typed_only: 'bool' = False, min_messages: 'int | None' = None, max_messages: 'int | None' = None, min_words: 'int | None' = None, max_words: 'int | None' = None, message_type: 'str | None' = None, continuation: 'str | None' = None) -> 'QueryUnitResultEnvelope'` |
| `Polylogue.query_completions` | `async (self, kind: 'str', *, incomplete: 'str' = '', unit: 'str | None' = None, field: 'str | None' = None) -> 'JSONDocument'` |
| `Polylogue.diagnose_query_miss` | `async (self, spec: 'SessionQuerySpec', *, full: 'bool' = False) -> 'QueryMissDiagnostics'` |
| `Polylogue.resolve_ref` | `async (self, ref: 'str') -> 'PublicRefResolutionPayload'` |
| `Polylogue.export_otel` | `async (self, *, source_ref: 'str', expressions: 'Sequence[str]', limit: 'int' = 50, include_message_text: 'bool' = False) -> 'OtelProjectionPayload'` |
| `Polylogue.neighbor_candidates` | `async (self, *, session_id: 'str | None' = None, query: 'str | None' = None, origin: 'str | None' = None, limit: 'int' = 10, window_hours: 'int' = 24) -> 'list[SessionNeighborCandidate]'` |
| `Polylogue.neighbor_candidate_payloads` | `async (self, *, session_id: 'str | None' = None, query: 'str | None' = None, origin: 'str | None' = None, limit: 'int' = 10, window_hours: 'int' = 24) -> 'list[JSONDocument]'` |
| `Polylogue.session_correlation_payload` | `async (self, session_id: 'str', *, repo_path: 'str | None' = None, since_hours: 'int' = 2, confidence_threshold: 'float' = 0.3) -> 'JSONDocument | None'` |
| `Polylogue.origin_usage_report` | `async (self, *, origin: 'str | None' = None, limit: 'int | None' = 25, detail: 'str' = 'full') -> 'ProviderUsageReport'` |
| `Polylogue.session_usage_reconciliation` | `async (self, session_id: 'str') -> 'SessionUsageReconciliation'` |
| `Polylogue.resume_brief` | `async (self, session_id: 'str', *, related_limit: 'int' = 6, repo_path: 'str | None' = None, recent_files: 'Sequence[str]' = ()) -> 'ResumeBrief | None'` |
| `Polylogue.find_resume_candidates` | `async (self, *, repo_path: 'str', cwd: 'str | None' = None, recent_files: 'Sequence[str]' = (), limit: 'int' = 10) -> 'tuple[ResumeCandidate, ...]'` |

### Source evidence reads

#### `api.archive.source-evidence-read`

Read raw artifacts and provider-side evidence retained in the durable source tier.

Route/tier class: `source-read`. CLI: `read`, `analyze`. MCP: `read`, `explain`.

| Python callable | Signature |
|---|---|
| `Polylogue.explain_import` | `async (self, path: 'str | Path | None' = None, *, raw_ref: 'str | None' = None, source_path: 'str | None' = None, source_name: 'str' = 'unknown', limit: 'int' = 100, redact_paths: 'bool' = True) -> 'ImportExplainPayload'` |
| `Polylogue.get_raw_artifacts_for_session` | `async (self, session_id: 'str', *, limit: 'int' = 50, offset: 'int' = 0) -> 'tuple[list[dict[str, object]], int]'` |
| `Polylogue.get_hook_event_summary_for_session` | `async (self, session_id: 'str') -> 'dict[str, object] | None'` |
| `Polylogue.get_session_events` | `async (self, session_id: 'str', *, event_type: 'str | None' = None, limit: 'int | None' = None) -> 'list[dict[str, object]] | None'` |
| `Polylogue.get_file_edits` | `async (self, session_id: 'str') -> 'list[dict[str, object]] | None'` |
| `Polylogue.get_web_content_constructs` | `async (self, session_id: 'str', *, construct_type: 'str | None' = None) -> 'list[dict[str, object]] | None'` |
| `Polylogue.get_agent_policies` | `async (self, session_id: 'str') -> 'list[dict[str, object]] | None'` |

### Insights and topology

#### `api.archive.insight-read`

Read materialized archive insights, topology, and derived archive health from the index tier.

Route/tier class: `index-read`. CLI: `analyze`, `read`. MCP: `query`, `get`, `status`, `explain`.

| Python callable | Signature |
|---|---|
| `Polylogue.get_session_insight_status` | `async (self) -> 'SessionInsightStatusSnapshot'` |
| `Polylogue.get_session_profile_insight` | `async (self, session_id: 'str', *, tier: 'str' = 'merged') -> 'SessionProfileInsight | None'` |
| `Polylogue.get_session_profile_record` | `async (self, session_id: 'str') -> 'SessionProfileRecord | None'` |
| `Polylogue.list_session_profile_insights` | `async (self, query: 'SessionProfileInsightQuery | None' = None) -> 'list[SessionProfileInsight]'` |
| `Polylogue.insight_readiness_report` | `async (self, query: 'InsightReadinessQuery | None' = None) -> 'InsightReadinessReport'` |
| `Polylogue.insight_rigor_audit` | `async (self, query: 'InsightRigorAuditQuery | None' = None) -> 'InsightRigorAuditReport'` |
| `Polylogue.archive_debt` | `async (self, *, kinds: 'Iterable[str] | None' = None, only_actionable: 'bool' = False, limit: 'int | None' = None, exact_fts: 'bool' = False) -> 'ArchiveDebtListPayload'` |
| `Polylogue.get_session_work_event_insights` | `async (self, session_id: 'str') -> 'list[SessionWorkEventInsight]'` |
| `Polylogue.list_session_work_event_insights` | `async (self, query: 'SessionWorkEventInsightQuery | None' = None) -> 'list[SessionWorkEventInsight]'` |
| `Polylogue.get_session_phase_insights` | `async (self, session_id: 'str') -> 'list[SessionPhaseInsight]'` |
| `Polylogue.list_session_phase_insights` | `async (self, query: 'SessionPhaseInsightQuery | None' = None) -> 'list[SessionPhaseInsight]'` |
| `Polylogue.get_thread_insight` | `async (self, thread_id: 'str') -> 'ThreadInsight | None'` |
| `Polylogue.list_thread_insights` | `async (self, query: 'ThreadInsightQuery | None' = None) -> 'list[ThreadInsight]'` |
| `Polylogue.list_session_tag_rollup_insights` | `async (self, query: 'SessionTagRollupQuery | None' = None) -> 'list[SessionTagRollupInsight]'` |
| `Polylogue.list_archive_coverage_insights` | `async (self, query: 'ArchiveCoverageInsightQuery | None' = None) -> 'list[ArchiveCoverageInsight]'` |
| `Polylogue.list_tool_usage_insights` | `async (self, query: 'ToolUsageInsightQuery | None' = None) -> 'list[ToolUsageInsight]'` |
| `Polylogue.list_session_cost_insights` | `async (self, query: 'SessionCostInsightQuery | None' = None) -> 'list[SessionCostInsight]'` |
| `Polylogue.get_session_latency_profile_insight` | `async (self, session_id: 'str') -> 'SessionLatencyProfileInsight | None'` |
| `Polylogue.list_session_latency_profile_insights` | `async (self, query: 'SessionLatencyProfileInsightQuery | None' = None) -> 'list[SessionLatencyProfileInsight]'` |
| `Polylogue.find_stuck_session_latency_profile_insights` | `async (self, query: 'SessionLatencyProfileInsightQuery | None' = None) -> 'list[SessionLatencyProfileInsight]'` |
| `Polylogue.list_cost_rollup_insights` | `async (self, query: 'CostRollupInsightQuery | None' = None) -> 'list[CostRollupInsight]'` |
| `Polylogue.list_usage_timeline_insights` | `async (self, query: 'UsageTimelineInsightQuery | None' = None) -> 'list[UsageTimelineInsight]'` |
| `Polylogue.list_archive_debt_insights` | `async (self, query: 'ArchiveDebtInsightQuery | None' = None) -> 'list[ArchiveDebtInsight]'` |
| `Polylogue.cost_outlook` | `async (self, plan_name: 'str', *, now: 'datetime | None' = None, method: 'ProjectionMethod' = <ProjectionMethod.linear: 'linear'>) -> 'CycleOutlook | None'` |
| `Polylogue.aggregate_sessions` | `async (self, *, group_by: 'str' = 'workflow_shape', since: 'str | None' = None, until: 'str | None' = None, origin: 'str | None' = None) -> 'dict[str, object]'` |
| `Polylogue.workflow_shape_distribution` | `async (self, *, group_by: 'str' = 'week', since: 'str | None' = None, until: 'str | None' = None, origin: 'str | None' = None) -> 'dict[str, object]'` |
| `Polylogue.find_abandoned_sessions` | `async (self, *, since: 'str | None' = None, repo_path: 'str | None' = None, min_severity: 'str' = 'question_left', limit: 'int' = 20) -> 'dict[str, object]'` |
| `Polylogue.tool_call_latency_distribution` | `async (self, *, since: 'str | None' = None, until: 'str | None' = None, origin: 'str | None' = None, tool_category: 'str | None' = None, limit: 'int' = 500) -> 'dict[str, object]'` |
| `Polylogue.compare_sessions` | `async (self, session_ids: 'Sequence[str]') -> 'dict[str, object]'` |
| `Polylogue.find_similar_sessions_by_metadata` | `async (self, session_id: 'str', *, limit: 'int' = 10, candidate_pool_limit: 'int' = 200) -> 'dict[str, object] | None'` |
| `Polylogue.correlate_sessions` | `async (self, *, metric_x: 'str', metric_y: 'str', origin: 'str | None' = None, since: 'str | None' = None, until: 'str | None' = None) -> 'dict[str, object]'` |
| `Polylogue.get_session_topology` | `async (self, session_id: 'str') -> 'SessionTopology | None'` |
| `Polylogue.get_ancestors` | `async (self, session_id: 'str') -> 'list[SessionRef]'` |
| `Polylogue.get_descendants` | `async (self, session_id: 'str') -> 'list[SessionRef]'` |
| `Polylogue.get_siblings` | `async (self, session_id: 'str') -> 'list[SessionRef]'` |
| `Polylogue.get_thread` | `async (self, session_id: 'str') -> 'list[SessionRef]'` |
| `Polylogue.get_logical_session` | `async (self, session_id: 'str') -> 'LogicalSession | None'` |
| `Polylogue.get_session_tree` | `async (self, session_id: 'str') -> 'list[Session]'` |
| `Polylogue.postmortem_bundle` | `async (self, spec: 'SessionQuerySpec | None' = None, *, limit: 'int | None' = None) -> 'PostmortemBundle'` |
| `Polylogue.pathology_report` | `async (self, spec: 'SessionQuerySpec | None' = None, *, limit: 'int | None' = None) -> 'PathologyReport'` |
| `Polylogue.portfolio_bundle` | `async (self, spec: 'SessionQuerySpec | None' = None, *, limit: 'int | None' = None, top_n: 'int' = 10) -> 'PortfolioBundle'` |
| `Polylogue.export_insight_bundle` | `async (self, request: 'InsightExportBundleRequest') -> 'InsightExportBundleResult'` |
| `Polylogue.regenerate_private_fable_packet` | `async (self, *, seed: 'str', requested_size: 'int', schema_id: 'str' = 'delegation.discourse', schema_version: 'int' = 1, exact_template_cap: 'int' = 1) -> 'FableDelegationPacket'` |

### Context and evidence

#### `api.context.delivery`

Compile context and record or inspect durable delivery receipts.

Route/tier class: `cross-tier`. CLI: `continue`, `read`. MCP: `context`, `get`, `status`.

| Python callable | Signature |
|---|---|
| `Polylogue.compile_context` | `async (self, spec: 'ContextSpec') -> 'ContextImage'` |
| `Polylogue.context_image_payload` | `async (self, *, project_path: 'str | None' = None, project_repo: 'str | None' = None, since: 'str | None' = None, until: 'str | None' = None, origin: 'str | None' = None, query: 'str | None' = None, max_sessions: 'int' = 5, max_tokens: 'int | None' = None, max_messages_per_session: 'int | None' = 24, max_chars_per_message: 'int | None' = 1800, include_messages: 'bool' = True, include_assertions: 'bool' = True, redact_paths: 'bool' = True, seed_session_id: 'str | None' = None) -> 'ContextImage'` |
| `Polylogue.context_preamble_payload` | `async (self, session_id: 'str', *, related_limit: 'int' = 5) -> 'Any'` |
| `Polylogue.get_context_delivery` | `async (self, snapshot_ref: 'str', *, recipient_ref: 'str') -> 'ArchiveContextDeliveryEnvelope | None'` |
| `Polylogue.list_context_deliveries` | `async (self, *, recipient_ref: 'str | None' = None, assertion_ref: 'str | None' = None, limit: 'int' = 50) -> 'list[ArchiveContextDeliveryEnvelope]'` |
| `Polylogue.record_context_delivery` | `async (self, *, image: 'ContextImage', boundary: 'str', recipient_ref: 'str', delivered_by_ref: 'str', run_ref: 'str | None' = None, inheritance_mode: 'str' = 'explicit') -> 'ArchiveContextDeliveryEnvelope'` |
| `Polylogue.compile_and_record_context` | `async (self, *, recipient_ref: 'str', delivered_by_ref: 'str', boundary: 'str', query: 'str | None' = None, max_sessions: 'int' = 5, max_tokens: 'int | None' = None, include_messages: 'bool' = True, include_assertions: 'bool' = True, redact_paths: 'bool' = True, seed_session_id: 'str | None' = None, run_ref: 'str | None' = None, inheritance_mode: 'str' = 'explicit') -> 'ArchiveContextDeliveryEnvelope'` |
| `Polylogue.correlate_hermes_context_deliveries` | `async (self, hermes_session_native_id: 'str') -> 'tuple[HermesContextDeliveryCorrelation, ...]'` |
| `Polylogue.reconcile_hermes_session_lifecycle` | `async (self, hermes_session_native_id: 'str') -> 'HermesLifecycleReconciliation | None'` |
| `Polylogue.reconcile_codex_spawn_edges` | `async (self) -> 'CodexSpawnEdgeReconciliation | None'` |
| `Polylogue.hermes_integration_health` | `async (self) -> 'HermesIntegrationHealth'` |

### Assertions and judgments

#### `api.assertion.review`

Read, capture, and judge durable assertions and comparative evidence.

Route/tier class: `cross-tier`. CLI: `mark`, `read`. MCP: `write`, `judge`, `read`.

| Python callable | Signature |
|---|---|
| `Polylogue.import_annotation_batch` | `async (self, request: 'AnnotationBatchImportRequest', *, registry: 'AnnotationSchemaRegistry | None' = None) -> 'AnnotationBatchImportResult'` |
| `Polylogue.list_assertion_claims` | `async (self, *, kinds: 'Sequence[str | AssertionKind] | None' = None, target_ref: 'str | None' = None, scope_ref: 'str | None' = None, statuses: 'Sequence[str | AssertionStatus] | None' = ('active', 'candidate'), context_inject: 'bool | None' = None, limit: 'int | None' = None) -> 'list[ArchiveAssertionEnvelope]'` |
| `Polylogue.list_assertion_claim_payloads` | `async (self, *, kinds: 'Sequence[str | AssertionKind] | None' = None, target_ref: 'str | None' = None, scope_ref: 'str | None' = None, statuses: 'Sequence[str | AssertionStatus] | None' = ('active', 'candidate'), context_inject: 'bool | None' = None, limit: 'int | None' = None) -> 'list[AssertionClaimPayload]'` |
| `Polylogue.list_assertion_candidates` | `async (self, *, target_ref: 'str | None' = None, kinds: 'Sequence[str | AssertionKind] | None' = None, limit: 'int | None' = None) -> 'list[AssertionClaimPayload]'` |
| `Polylogue.list_assertion_candidate_reviews` | `async (self, *, target_ref: 'str | None' = None, kinds: 'Sequence[str | AssertionKind] | None' = None, statuses: 'Sequence[str | AssertionStatus] | None' = None, limit: 'int | None' = None) -> 'AssertionCandidateReviewListPayload'` |
| `Polylogue.assertion_candidate_queue_health` | `async (self) -> 'AssertionCandidateQueueHealthPayload'` |
| `Polylogue.judge_assertion_candidate` | `async (self, *, candidate_ref: 'str', decision: 'str', reason: 'str | None' = None, actor_ref: 'str' = 'user:local', inject: 'bool' = False, replacement_kind: 'str | None' = None, replacement_body_text: 'str | None' = None, replacement_value: 'object | None' = None) -> 'AssertionJudgmentResultPayload'` |
| `Polylogue.capture_assertion_candidate` | `async (self, *, body_text: 'str', kind: 'AssertionKind', refs: 'Sequence[str]' = (), scope_refs: 'Sequence[str]' = (), cwd: 'Path | None' = None, author_ref: 'str' = 'user:local', author_kind: 'str' = 'user', idempotency_key: 'str | None' = None, ttl_seconds: 'int | None' = None) -> 'AssertionClaimPayload'` |
| `Polylogue.judge_assertion_candidates` | `async (self, *, items: 'Sequence[Any]') -> 'AssertionBulkJudgmentPayload'` |
| `Polylogue.record_comparative_judgment` | `async (self, judgment: 'ComparativeJudgment', *, author_kind: 'str' = 'user') -> 'ArchiveAssertionEnvelope'` |
| `Polylogue.list_comparative_judgments` | `async (self) -> 'list[ComparativeJudgment]'` |
| `Polylogue.join_typed_annotations` | `async (self, *, schema_id: 'str', schema_version: 'int', statuses: 'Sequence[str | AssertionStatus]', target_kind: 'str | None' = None, group_by: "Sequence[Literal['repo', 'model', 'time', 'origin']]" = (), limit: 'int' = 500, offset: 'int' = 0) -> 'AnnotationStructuralJoinResult'` |

### Archive mutations

#### `api.archive.session-delete`

Delete a session and its archive records through the shared mutation executor.

Route/tier class: `cross-tier`. CLI: `delete`. MCP: `write`.

| Python callable | Signature |
|---|---|
| `Polylogue.delete_session` | `async (self, session_id: 'str') -> 'bool'` |
| `Polylogue.delete_session_safe` | `async (self, session_id: 'str', *, actor: 'str' = 'user:api') -> 'DeleteSessionResult'` |

### Durable user state

#### `api.user-state.read`

Read tags, marks, annotations, views, recall packs, workspaces, corrections, notes, and settings from user.db.

Route/tier class: `user-read`. CLI: `read`, `mark`. MCP: `read`, `get`.

| Python callable | Signature |
|---|---|
| `Polylogue.list_tags` | `async (self, *, origin: 'str | None' = None) -> 'dict[str, int]'` |
| `Polylogue.get_metadata` | `async (self, session_id: 'str') -> 'dict[str, str]'` |
| `Polylogue.list_marks` | `async (self, *, mark_type: 'str | None' = None, session_id: 'str | None' = None, target_type: 'str | None' = None, target_id: 'str | None' = None, message_id: 'str | None' = None) -> 'list[dict[str, str]]'` |
| `Polylogue.get_annotation` | `async (self, annotation_id: 'str') -> 'dict[str, str] | None'` |
| `Polylogue.list_annotations` | `async (self, *, session_id: 'str | None' = None, target_type: 'str | None' = None, target_id: 'str | None' = None, message_id: 'str | None' = None) -> 'list[dict[str, str]]'` |
| `Polylogue.get_view` | `async (self, view_id: 'str') -> 'dict[str, str] | None'` |
| `Polylogue.list_views` | `async (self) -> 'list[dict[str, str]]'` |
| `Polylogue.get_recall_pack` | `async (self, pack_id: 'str') -> 'dict[str, str] | None'` |
| `Polylogue.list_recall_packs` | `async (self) -> 'list[dict[str, str]]'` |
| `Polylogue.get_workspace` | `async (self, workspace_id: 'str') -> 'dict[str, str] | None'` |
| `Polylogue.list_workspaces` | `async (self) -> 'list[dict[str, str]]'` |
| `Polylogue.list_corrections` | `async (self, *, session_id: 'str | None' = None, kind: 'str | None' = None) -> 'list[LearningCorrection]'` |
| `Polylogue.list_blackboard_notes` | `async (self, *, kind: 'str | None' = None, scope_repo: 'str | None' = None, unresolved: 'bool' = False, limit: 'int' = 20) -> 'list[BlackboardNote]'` |
| `Polylogue.get_setting` | `async (self, setting_key: 'str') -> 'ArchiveUserSettingEnvelope | None'` |
| `Polylogue.list_settings` | `async (self) -> 'list[ArchiveUserSettingEnvelope]'` |

#### `api.user-state.write`

Mutate tags, metadata, marks, annotations, views, recall packs, workspaces, corrections, notes, and settings in user.db.

Route/tier class: `user-write`. CLI: `mark`, `delete`. MCP: `write`.

| Python callable | Signature |
|---|---|
| `Polylogue.add_tag` | `async (self, session_id: 'str', tag: 'str', *, author_ref: 'str | None' = None, author_kind: 'str | None' = None) -> 'TagMutationResult'` |
| `Polylogue.remove_tag` | `async (self, session_id: 'str', tag: 'str') -> 'TagMutationResult'` |
| `Polylogue.update_metadata` | `async (self, session_id: 'str', key: 'str', value: 'str') -> 'bool'` |
| `Polylogue.set_metadata` | `async (self, session_id: 'str', key: 'str', value: 'object') -> 'MetadataMutationResult'` |
| `Polylogue.delete_metadata` | `async (self, session_id: 'str', key: 'str') -> 'MetadataMutationResult'` |
| `Polylogue.bulk_tag_sessions` | `async (self, session_ids: 'list[str]', tags: 'list[str]', *, author_ref: 'str | None' = None, author_kind: 'str | None' = None) -> 'BulkTagMutationResult'` |
| `Polylogue.add_mark` | `async (self, session_id: 'str', mark_type: 'str', *, target_type: 'str' = 'session', target_id: 'str | None' = None, message_id: 'str | None' = None) -> 'bool'` |
| `Polylogue.remove_mark` | `async (self, session_id: 'str', mark_type: 'str', *, target_type: 'str' = 'session', target_id: 'str | None' = None, message_id: 'str | None' = None) -> 'bool'` |
| `Polylogue.save_annotation` | `async (self, annotation_id: 'str', session_id: 'str', note_text: 'str', *, target_type: 'str' = 'session', target_id: 'str | None' = None, message_id: 'str | None' = None) -> 'bool'` |
| `Polylogue.delete_annotation` | `async (self, annotation_id: 'str') -> 'bool'` |
| `Polylogue.save_view` | `async (self, view_id: 'str', name: 'str', query_json: 'str') -> 'bool'` |
| `Polylogue.delete_view` | `async (self, view_id: 'str') -> 'bool'` |
| `Polylogue.create_recall_pack` | `async (self, pack_id: 'str', label: 'str', payload_json: 'str') -> 'bool'` |
| `Polylogue.delete_recall_pack` | `async (self, pack_id: 'str') -> 'bool'` |
| `Polylogue.save_workspace` | `async (self, workspace_id: 'str', name: 'str', mode: 'str', open_targets_json: 'str', layout_json: 'str', active_target_json: 'str' = '{}') -> 'bool'` |
| `Polylogue.delete_workspace` | `async (self, workspace_id: 'str') -> 'bool'` |
| `Polylogue.record_correction` | `async (self, session_id: 'str', kind: 'str', payload: 'dict[str, str]', *, note: 'str | None' = None, author_ref: 'str | None' = None, author_kind: 'str | None' = None) -> 'LearningCorrection'` |
| `Polylogue.delete_correction` | `async (self, session_id: 'str', kind: 'str') -> 'bool'` |
| `Polylogue.clear_corrections` | `async (self, session_id: 'str') -> 'int'` |
| `Polylogue.post_blackboard_note` | `async (self, *, kind: 'str', title: 'str', content: 'str', scope_repo: 'str | None' = None, scope_session: 'str | None' = None, scope_issue: 'int | None' = None, scope_path: 'str | None' = None, related_sessions: 'tuple[str, ...]' = (), author_ref: 'str | None' = None, author_kind: 'str' = 'user', evidence_refs: 'tuple[str, ...]' = (), staleness: 'dict[str, object] | None' = None, context_policy: 'dict[str, object] | None' = None) -> 'BlackboardNote'` |
| `Polylogue.set_setting` | `async (self, setting_key: 'str', value: 'object', *, author_ref: 'str' = 'user:local') -> 'ArchiveUserSettingEnvelope'` |

### Intentional exclusions

| Export | Reason | Authority |
|---|---|---|
| `ArchiveStats` | Result data model, not an executable archive operation. | `polylogue-s1kr` |
| `select_pending_embedding_session_window` | Public adapter helper for daemon/CLI window selection. It is intentionally not a facade operation. | `polylogue-s1kr` |
| `Polylogue.__repr__` | Diagnostic representation protocol, not an archive operation. | `polylogue-s1kr` |

<!-- END GENERATED API OPERATION PARITY -->

---

**See also:** [CLI Reference](cli-reference.md) · [Data Model](data-model.md) · [Configuration](configuration.md)
