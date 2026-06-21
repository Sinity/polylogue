[← Back to Docs](README.md)

# Database Schema

Polylogue does not store the archive in one SQLite file. The archive is a **set
of SQLite files split by durability class**, all in WAL mode. Each file (a
"tier") owns a distinct kind of data and carries its own independent schema
version. This page describes the conceptual shape and points at the canonical
DDL; for the deeper invariant detail see [Internals](internals.md) and
[Architecture](architecture.md).

## Authoritative source

The DDL under
[`polylogue/storage/sqlite/archive_tiers/`](../polylogue/storage/sqlite/archive_tiers/)
is the **single source of truth** for tables, columns, CHECK constraints, and
indexes. This document deliberately does not re-list every column, because an
exhaustive column table drifts from the DDL and goes stale. When you need an
exact column, read the tier file named below — do not trust a prose summary
over the `CREATE TABLE` statement.

| Tier file | Tier | Version constant |
|-----------|------|------------------|
| `source.py` | `source.db` | `SOURCE_SCHEMA_VERSION = 1` |
| `index.py` | `index.db` | `INDEX_SCHEMA_VERSION = 4` |
| `embeddings.py` | `embeddings.db` | `EMBEDDINGS_SCHEMA_VERSION = 1` |
| `user.py` | `user.db` | `USER_SCHEMA_VERSION = 3` |
| `ops.py` | `ops.db` | `OPS_SCHEMA_VERSION = 1` |

There is no single global "schema version" number. Each tier is versioned and
bootstrapped independently.

The runtime table inventory per tier is also enumerated in
`polylogue/cli/commands/status.py` (`_ARCHIVE_TIER_TABLES`), which `polylogue
status` uses to report row counts; that dict is a useful cross-check against
the DDL.

## The Five Tiers

### `source.db` — raw acquisition (rebuild-from-source)

Stores acquired bytes and source evidence. Everything downstream is rebuilt
from this tier, so it is the durable record of *what was ingested*.

Core tables: `raw_sessions` (one row per acquired payload, keyed by `raw_id`,
carrying `origin`, `native_id`, `blob_hash`, and parse/validation state),
`raw_artifacts` (artifact-taxonomy classification per acquired file),
`blob_refs` / `pending_blob_refs` / `gc_generations` (content-addressed blob
references and the lease/GC bookkeeping described in
[Internals § Blob Store Model](internals.md#blob-store-model)),
`raw_hook_events`, and `history_sidecars`.

### `index.db` — parsed tree, search, and insights (rebuildable)

The parsed session/message/block tree, full-text search indexes, cross-session
topology, and the materialized read models. Rebuildable from `source.db`.

Core tables: `sessions`, `messages`, `blocks`, plus the `actions` **view**
(not a table — see below), `session_links` (typed cross-session edges),
`threads` / `thread_sessions`, attachment tables (`attachments`,
`attachment_refs`, `attachment_native_ids`), `paste_spans`, cost tables
(`price_catalogs`, `model_prices`, `session_reported_costs`,
`session_model_usage`), the auto-tag side of `session_tags`, and the insight
read models (`session_profiles`, `session_work_events`, `session_phases`,
`session_latency_profiles`, `session_tag_rollups`, `threads`) plus
`insight_materialization` for cache invalidation.

### `embeddings.db` — vectors (rebuildable, expensive)

`message_embeddings` (a `vec0` virtual table, 1024-dimensional float
embeddings), `message_embeddings_meta`, and `embedding_status`. Populated only
when embedding is enabled with a valid Voyage key (see
[Architecture § Embedding Pipeline](architecture.md#embedding-pipeline)).
Rebuildable, but re-embedding costs Voyage API calls.

### `user.db` — irreplaceable human input (back up this one)

The only tier that is not rebuildable from source. Its canonical table is
`assertions`: user marks, annotations, corrections, suppressions, tags,
metadata, saved views, recall packs, workspaces, blackboard notes, candidate
claims, and operator judgments are all assertion rows with lifecycle state.
`index.db.session_tags` remains a rebuildable auto-tag/read-model projection;
human-owned tag and metadata writes do not have separate user-tier tables.

### `ops.db` — disposable daemon telemetry

Ingest cursors (`ingest_cursor`), ingest attempts (`ingest_attempts`),
convergence debt (`convergence_debt`), cursor-lag samples, daemon stage/event
logs, embedding catch-up runs, and OTLP spans/telemetry. Safe to discard; the
daemon repopulates it.

## Identifiers

Primary keys in `index.db` are **generated** from natural components rather than
opaque UUIDs (see the `GENERATED ALWAYS AS (...) STORED` columns in `index.py`):

- `sessions.session_id` = `origin || ':' || native_id`
- `messages.message_id` = `session_id || ':' || COALESCE(native_id, position || '.' || variant_index)`
- `blocks.block_id` = `message_id || ':' || position`

`origin` is a closed vocabulary (`Origin` enum in `polylogue/core/enums.py`,
e.g. `claude-code-session`, `claude-ai-export`, `chatgpt-export`,
`codex-session`, `aistudio-drive`) enforced by a SQL `CHECK`. It is the public
source-origin token; the provider-wire `Provider` enum is used only at the
parsing/schema boundary.

## Timestamps, hashes, and types

- **Timestamps** are stored as integer epoch milliseconds in `*_at_ms` columns
  (e.g. `created_at_ms`, `updated_at_ms`, `occurred_at_ms`).
- **Content hashes** are 32-byte `BLOB` columns (`CHECK(length(...) = 32)`),
  SHA-256 over the NFC-normalized session payload — title, timestamps,
  messages, attachments, and blocks. User metadata (tags, summaries, notes) is
  **excluded** from the hash, so editing it does not trigger re-import. See
  [Internals § Content Hash Model](internals.md#content-hash-model).
- Tables use SQLite `STRICT` mode and closed-enum `CHECK` constraints generated
  from the `StrEnum` vocabularies in `polylogue/core/enums.py`.

## Content blocks are first-class rows

Structured message content lives in the `blocks` table — one row per block,
keyed `(message_id, position)`, with a `block_type` constrained to the
`BlockType` enum (`text`, `thinking`, `reasoning`, `tool_use`, `tool_result`,
`image`, `code`, `document`). Tool calls carry `tool_name`, `tool_id`, and
`tool_input`; `tool_command` and `tool_path` are virtual generated columns
extracted from `tool_input` JSON.

`actions` is a **view**, not a table: it left-joins `tool_use` blocks to their
matching `tool_result` block (by `tool_id`) so tool executions read as paired
records without a separate materialization.

## Full-text search

`index.db` carries three FTS5 virtual tables, all using the `unicode61`
tokenizer (no porter stemmer in this SQLite build):

| Virtual table | Indexes |
|---------------|---------|
| `messages_fts` | Block `search_text` (text + tool name + command/path), contentless (`content=''`, `contentless_delete=1`) |
| `session_work_events_fts` | Work-event search text |
| `threads_fts` | Thread search text |

`messages_fts` is kept in sync with `blocks` by the `messages_fts_ai/ad/au`
triggers. During bulk ingest these triggers are suspended for performance and
restored before commit; see [Internals § FTS5 Model](internals.md#fts5-model)
for the drift-detection and repair behavior.

## Schema Versioning — fresh-only, no upgrade chain

Polylogue has **no in-place schema upgrade chain**. On startup each tier's
on-disk `PRAGMA user_version` is compared against its tier constant:

- **Empty file** (`user_version == 0`): bootstrap fresh.
- **Version match**: open as-is.
- **Anything else** (older or newer): the database is **rejected**.

A version mismatch is resolved by re-acquiring from source, not by patching the
file in place. The operator moves the archive aside and re-ingests:

```bash
polylogue ops reset --database && polylogued run
```

Schema bumps are deletes-then-defines edits of the owning tier DDL, never
deltas. The bootstrap branching that decides fresh-init vs. open vs. reject is
shared across sync and async backends in
`storage/sqlite/schema_bootstrap.py`. See
[Internals § Schema Versioning Model](internals.md#schema-versioning-model) and
[CONTRIBUTING § Schema-Touching Changes](../CONTRIBUTING.md#schema-touching-changes).

## Inspecting an archive

```bash
polylogue ops status                    # daemon/archive snapshot, including per-tier row counts
polylogue ops maintenance archive-plan  # planned archive file set
polylogue ops doctor                     # schema health + referential integrity
polylogue ops doctor --schemas           # provider-schema conformance over raw records
```

The `devtools lab schema roundtrip` command verifies committed provider
schema packages reload and roundtrip cleanly through typed models.

---

**See also:** [Architecture](architecture.md) · [Internals](internals.md) · [Data Model](data-model.md) · [Configuration](configuration.md)
