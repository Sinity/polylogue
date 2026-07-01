# Internals Reference

Working map of the live codebase: invariants, hot files, extension points, and
debugging landmarks. For the conceptual system shape, see
[architecture.md](architecture.md).

## Key Invariants

| Invariant | Enforced in |
| --- | --- |
| Archive writes are idempotent by content hash | `pipeline/ids.py`, `pipeline/prepare_enrichment.py` |
| Content hash excludes user metadata (tags, summaries) | `pipeline/ids.py:session_content_hash()` |
| Content hash uses NFC normalization | `core/hashing.py:hash_text()` |
| Async SQLite is the primary runtime; sync SQLite exists for CLI, schema tooling, and batch-ingest write paths | `storage/sqlite/async_sqlite.py`, `storage/sqlite/connection.py`, `pipeline/services/ingest_batch.py` |
| SQLite read/write tuning is profile-driven, not backend-local | `storage/sqlite/connection_profile.py` |
| FTS tokenizer is `unicode61` (no porter stemmer) | `storage/sqlite/archive_tiers/index.py` |
| Schema bootstrap branching is shared across sync and async backends | `storage/sqlite/schema_bootstrap.py:decide_schema_bootstrap()` |

## Hot Files

### Entry Points

| File | Purpose |
| --- | --- |
| `polylogue/api/__init__.py` | Async library API |
| `polylogue/config.py` | Runtime configuration and XDG resolution |
| `polylogue/cli/click_app.py` | Root query-first CLI dispatch |
| `polylogue/cli/command_inventory.py` | CLI command inventory |
| `polylogue/operations/archive.py` | High-level archive operations |

### Storage

| File | Purpose |
| --- | --- |
| `storage/sqlite/archive_tiers/` | Current split archive DDL and tier versions |
| `storage/sqlite/schema.py` | Shared sync/async fresh-init, version guard, and extension application |
| `storage/sqlite/schema_bootstrap.py` | Shared schema snapshot, bootstrap branching, and extension planning |
| `storage/sqlite/connection_profile.py` | Canonical read/write SQLite timeouts, cache, mmap, and PRAGMA profiles |
| `storage/repository/__init__.py` | Repository facade (9-mixin composition: archive reads/writes, action reads, four insight readers, raw, vectors) |
| `storage/search_providers/fts5.py` | Lexical search |
| `storage/search_providers/hybrid.py` | Hybrid retrieval (RRF fusion) |

### Sources and Pipeline

| File | Purpose |
| --- | --- |
| `sources/dispatch.py` | Provider detection and parser routing |
| `sources/parsers/*.py` | Per-provider parsing |
| `pipeline/ingest_support.py` | Ingest stage definitions and source selection helpers |
| `pipeline/ids.py` | Content hashing and ID generation |
| `pipeline/services/ingest_batch.py` | Batch ingest (largest pipeline file) |

## Extension Points

**Adding a provider**: Start at `sources/dispatch.py:detect_provider()`. Add a
`looks_like()` function in a new parser under `sources/parsers/`. Add a
`Provider` enum variant in `types.py`. Add a provider schema bundle under
`schemas/providers/`. Dispatch is generally strict-before-loose for record-path
and Pydantic-validated checks, but `sources/dispatch.py` runs some structural
sequence detectors before loose code/dict probes. Insert the new check at the
tightness level it deserves or an earlier parser will claim its records first.

**Adding a filter**: Filter chain: `archive/filter/filters.py`. If the filter
needs a stats-table join, update `_needs_stats_join()` in
`storage/sqlite/connection.py`. Add the corresponding CLI flag in
`cli/query.py` and MCP parameter in `mcp/`.

**Adding a CLI command**: Register in `cli/command_inventory.py`. Implementation
goes in `cli/commands/`. The CLI shows fast daemon status on bare invocation
and falls back to archive summary when the daemon is not running.

**Adding a session insight**: Define the insight model in `insights/`. Add
storage in `storage/insights/session/`. Wire rebuild logic and register in
`insights/registry.py`.

**Adding a devtools command**: Add a `CommandSpec` to
`devtools/command_catalog.py`. Implementation goes in `devtools/<name>.py`.
Run `devtools render all` to update the generated catalog in
`docs/devtools.md`.

**Adding any new module/file under `polylogue/`**: regenerate the topology
projection or `verify topology` and `render all --check` will fail on the new
path. Run `devtools render topology-projection && devtools render topology-status`
and commit the updated `docs/plans/topology-target.yaml` and
`docs/topology-status.md` alongside the code.

## Schema Versioning Model

Polylogue has no in-place schema upgrade chain. The runtime knows exactly one
schema shape:

- Tier version constants under `storage/sqlite/archive_tiers/` are the
  authority. The canonical schema is described directly by each tier DDL;
  there are no upgrade plans, no `build_vN_to_vM_*` helpers, and no
  chain-dispatch logic in `schema_bootstrap.py`.
- On startup the on-disk `PRAGMA user_version` is compared against the
  tier constant:
  - **Empty file** (`user_version == 0`): bootstrap fresh.
  - **Version match**: open as-is.
  - **Anything else** (older or newer): the database is rejected.
- **Mismatch → rebuild the affected tier from source.** Polylogue does not
  patch an out-of-band shape into the canonical one. For index-tier schema
  bumps, the operator moves the index database aside with
  `polylogue ops reset --index && polylogued run`, preserving `source.db`,
  `user.db`, and other durable tiers while rebuilding the canonical index.
- Schema bumps are deletes-then-defines, never deltas. A schema change
  edits the owning tier DDL/version and documents the re-ingest expectation.
  No upgrade helpers are added for the bump.
- Index schema version 20 adds `idx_blocks_type_tool`, an expression index on
  `(block_type, COALESCE(NULLIF(LOWER(tool_name), ''), 'unknown'))`, so tool
  family rollups can resolve exact tool names and MCP-server prefixes without
  scanning every `tool_use` block in large archives. The `analyze tools`
  lowerers use range predicates for MCP prefixes to match the index expression.
  Existing index tiers must be rebuilt from source evidence
  (`polylogue ops reset --index && polylogued run`).
- Index schema version 19 adds expression indexes for materialized
  `session_observed_events` tool outcome payload fields. The terminal query DSL
  can now ask questions such as
  `observed-events where kind:tool_finished | group by handler | count`; on
  large archives the v18 shape filtered by `kind` and then built temporary
  B-trees over JSON-extracted `tool_name`, `handler_kind`, or `status` values.
  The v19 indexes key `(kind, COALESCE(NULLIF(json_extract(payload_json,
  '$.<field>'), ''), 'unknown'))` for `tool_name`, `handler_kind`, and
  `status`, matching the SQL lowerer's group expressions. Existing index tiers
  must be rebuilt from source evidence (`polylogue ops reset --index &&
  polylogued run`).
- Index schema version 16 captures structured tool-result outcomes (the
  "keystone"). `blocks` gains `tool_result_is_error` (0/1, nullable) and
  `tool_result_exit_code` (nullable INTEGER); the `actions` view exposes them
  as `is_error` / `exit_code` alongside the paired tool_use row. Previously the
  source's own outcome fields (Claude `is_error` on tool_result content, Codex
  `function_call_output.metadata.exit_code`) were dropped at parse, forcing
  in-session outcomes ("tests passed", "command failed") to be regex-guessed
  from result text. Now they are read from structure: NULL means unknown
  (default), never a fabricated positive. This is additive — existing rows
  read NULL until rebuilt. Rebuild from source evidence
  (`polylogue ops reset --index && polylogued run`).
- Index schema version 15 makes `idx_messages_session_sortkey` an expression
  index — `(session_id, (occurred_at_ms IS NULL), occurred_at_ms, message_id)`
  (#2475 perf audit). The keyset/paginated message reads order by
  `(occurred_at_ms IS NULL), occurred_at_ms, message_id` so NULL-timestamp rows
  sort last; the v14 plain `(session_id, occurred_at_ms, message_id)` index could
  not satisfy that leading `IS NULL` expression, so the planner fell back to
  `USE TEMP B-TREE FOR ORDER BY` and sorted the whole session per chunk
  (expensive on multi-thousand-message sessions). The expression index matches
  the ORDER BY exactly and plans as a covering-index scan with no temp sort
  (verified via EXPLAIN QUERY PLAN). Rebuild from source evidence
  (`polylogue ops reset --index && polylogued run`).
- Index schema version 14 hardens lineage normalization (#2467 audit).
  `session_links.branch_point_message_id` is no longer a FK with `ON DELETE SET
  NULL`: a parent's full-replace re-ingest (`DELETE FROM messages` then re-INSERT)
  would otherwise null the child's branch point during the DELETE step and
  permanently break the child's composition. `message_id` is deterministic, so the
  plain-TEXT reference survives the re-create; reads bail to the child's own tail
  if a branch point ever dangles. Also adds `idx_messages_session_sortkey` so the
  keyset/paginated message reads stop doing a temp B-tree sort per chunk. Rebuild
  from source evidence.
- Index schema version 13 makes attachment bytes honest (#2468). `attachments.blob_hash`
  is now nullable and holds the **true SHA-256 of the stored bytes** when acquired
  (previously a synthetic hash of the attachment id was written with no blob ever
  stored — 0 blobs for 8,425 rows). A new `acquisition_status`
  (`acquired` / `unavailable` / `unfetched`) records whether the bytes were
  fetched. Inline export bytes (e.g. Gemini base64) are written to the
  content-addressed blob store at ingest; other sources stay `unfetched` with the
  source ref preserved for later re-acquisition. Rebuild from source evidence.
- Index schema version 12 normalizes prefix-sharing lineage (#2467). A fork,
  resume, spawned subagent, or auto-compaction copy physically replays the
  parent's leading context; the writer now drops that inherited prefix and keeps
  only the child's divergent tail, recording the relationship on `session_links`
  via two new columns: `branch_point_message_id` (the last inherited parent
  message) and `inheritance` (`prefix-sharing` vs `spawned-fresh`). Reads
  (`get_messages`, `read_archive_session_envelope`) compose the parent transcript
  up to the branch point + the child's own tail, so each real message is stored
  once while the full logical transcript is still served. Existing index tiers
  must be rebuilt from source evidence (`polylogue ops reset --index &&
  polylogued run`); `source.db` is untouched, so this is a derived-index rebuild
  with no user-data impact. (The matching parser detection of Codex
  `forked_from_id`/`thread_spawn` and Claude `agent-acompact-*` shipped
  alongside.)
- Index schema version 11 materializes the run projection into three derived
  read-model tables — `session_runs`, `session_observed_events`, and
  `session_context_snapshots` — recomputed by the session-insight materializer
  (`compile_session_digest(...).run_projection`) exactly like `session_profiles`.
  They give the `run` / `observed-event` / `context-snapshot` query units a
  durable SQL-backed lowerer (terminal rows, `exists`, and sort) instead of the
  per-query session-digest scan. They are derived/rebuildable, not a new
  source of truth; the durable evidence stays `session_events` + `messages` +
  `topology_edges`. Existing index tiers must be rebuilt from source evidence
  (`polylogue ops reset --index && polylogued run`).
- Index schema version 10 adds a role-leading `idx_messages_role` index so
  daemon `/api/facets` can compute global role counts without scanning the
  session-role compound index and spilling to a temp B-tree. Existing index
  tiers must be rebuilt from source evidence or explicitly operator-patched by
  adding the index to a backed-up archive before moving that archive to version
  10 with the matching deployed package.
- Index schema version 9 adds typed `sessions.session_kind` for standard vs
  temporary sessions. Existing index tiers must be rebuilt from source evidence
  or explicitly operator-patched so temporary ChatGPT/Claude/browser-capture
  sessions become schema state instead of only parser auto-tags.
- Index schema version 8 added `web_content_constructs` for provider-native
  web/export constructs such as citations, source references, canvas/artifact
  revisions, asset pointers, token-budget rows, and task/search records.
- Index schema version 7 added JSON-object checks for `blocks.tool_input`
  and `session_provider_usage_events.payload_json`. Existing index tiers must
  be rebuilt from source evidence so the canonical read model is recreated with
  the stricter JSON contract.
- Provider usage accounting is audited as a source-derived read model: exact
  event rows (`session_provider_usage_events`), text-only estimates, unsupported
  origins, source rows acquired but not materialized, and stale
  `session_model_usage` rollups are distinct diagnostic states. Cache read/write
  token lanes remain labelled and are not merged into generic input/output.
- Index schema version 6 added `session_provider_usage_events` for
  provider-reported token usage events and single-model rollup repair from
  those events. Existing index tiers must be rebuilt from source evidence so
  Codex `token_count` records materialize into durable provider usage rows
  instead of disappearing after parse.
- Index schema version 5 added `messages.material_origin` plus authored-user
  aggregate columns on `sessions`. Existing index tiers must be rebuilt from
  source evidence so Claude Code `role=user` protocol/context/generated-pack
  rows move out of authored-user accounting while provider-role counts remain
  available.
- Provider schemas (the parsing/validation surface, distinct from the
  storage schema) are still regenerated fresh via
  `devtools lab schema generate` and promoted via `devtools lab schema promote`.

This design intentionally rejects in-place upgrade-chain complexity (no
Alembic, no forward/reverse upgrade scripts, no partially-applied upgrade
states, no `_apply_version_upgrade_plan` rollback windows). If the configured
archive path is not the current schema, the operator moves it aside and
re-ingests from source. Files that are not configured archive paths are not
classified or handled by the archive runtime.

The only compatibility carve-out is `ops.db`: `archive_tiers/bootstrap.py` may
use narrowly scoped `ALTER TABLE ... ADD COLUMN` helpers for disposable daemon
telemetry, such as ingest-cursor runtime fields and cursor-lag rollups. That
exception is not a migration pattern for `source.db`, `index.db`,
`embeddings.db`, or `user.db`; durable tier changes still require a version
bump and rebuild/reset path.

## Archive Activation

The archive file set is split by durability class:

- `source.db` stores raw acquisition rows and source evidence that
  rebuildable projections depend on.
- `index.db` stores the parsed session/message/block tree, FTS/search
  indexes, graph/topology rows, and derived insight read models.
- `embeddings.db` stores vector rows, embedding status, and embedding
  catch-up metadata; it is rebuildable, but expensive.
- `user.db` stores irreplaceable human input as assertion rows: marks,
  annotations, corrections, user tags, metadata keys, saved views,
  recall packs, workspaces, blackboard notes, candidates, and judgments.
- `ops.db` stores disposable daemon telemetry such as ingest cursors,
  attempts, convergence debt, stage events, embedding catch-up runs,
  and OTLP spans.

The operator flow is explicit:

```bash
polylogue ops maintenance archive-plan
polylogue ops maintenance archive-init --yes
polylogued run
polylogue ops maintenance archive-read --limit 20
```

`archive-init` bootstraps the archive file set. The daemon and explicit
ingest paths populate `source.db` and `index.db` directly from source
artifacts. Root query commands use the active `index.db`.

## Topology Edges (#1258)

`topology_edges` persists every parent reference asserted by a parser as a
typed row, including references whose parent has not yet been ingested
(out-of-order ingestion) or has been hard-deleted. The pre-existing fast
path (`sessions.parent_session_id` set when the parent is in the
prepare cache) is unchanged; the topology table is an additional durable
record that always carries the original provider-native parent id.

- **Identity:** `(src_session_id, dst_provider_native_id, edge_type)`
  with `UNIQUE`. Re-ingesting the same child is idempotent.
- **Closed enums:** `polylogue/archive/topology/edge.py` defines
  `TopologyEdgeType` (continuation / sidechain / subagent / branch / fork /
  resume / repaired) and `TopologyEdgeStatus` (unresolved / resolved /
  repaired). Slice A emits `unresolved` and `resolved` only.
- **Resolve:** every session save runs
  `resolve_topology_edges_for_session` so that an out-of-order child's
  edge flips to `resolved` the moment its parent's native id appears in
  `sessions`.
- **Hash boundary:** topology edges are derived per ingest and are NOT part
  of `sessions.content_hash` — mirrors the same boundary as
  `user_corrections` (#1131) and the blob lease tables.

## Logical Session Identity (#866)

`session_profiles.logical_session_id` materializes the resolved root of a
session's parent chain. For a root session it equals
`session_id`; for continuations, forks, sidechains, and subagents it points
at the root session that represents the logical work session.

Day summaries and tag rollups retain `session_count` as the physical
session count and add `logical_session_count` plus
`logical_session_ids_json` so weekly and cross-provider reducers can count
logical sessions without re-walking parent pointers. The Python API exposes
`get_logical_session(session_id)` as the compact read-pull envelope for
agents and MCP callers; `get_session_topology` remains the full graph view.

## Learning Corrections (Feedback Loop)

User corrections are stored in `user_corrections` and live outside the
content-hash boundary by construction (#1131):

- Keyed by `(session_id, insight_kind)` — at most one correction of
  each kind per session, so deterministic rebuilds always produce the
  same merged insight output.
- Recognized kinds (closed `CorrectionKind` enum):
  `tag_reject`, `tag_accept`, `summary_override`. New kinds are an
  explicit code change.
- Recording or removing a correction never touches
  `sessions.content_hash`. The hash invariant is asserted by
  `tests/unit/insights/test_feedback.py`.
- Insight materialization paths consult corrections after computing
  heuristic suggestions. Auto-tag and summary merge helpers live in
  `polylogue/insights/feedback.py`.
- Surfaces:
  - MCP: `record_correction`, `list_corrections`, `clear_corrections`.
  - Library: `Polylogue.record_correction(...)`,
    `Polylogue.list_corrections(...)`,
    `Polylogue.delete_correction(...)`, `Polylogue.clear_corrections(...)`.
- Storage backed by `polylogue/storage/sqlite/archive_tiers/archive.py`
  (`ArchiveStore.record_correction` / `list_corrections` /
  `delete_correction` / `clear_corrections`) and
  `polylogue/storage/insights/feedback/` (async SQL helpers).

## Text Handling Contracts

Polylogue exposes several text-processing boundaries. Each declares one of
three contracts per edge case; the matrix in
`tests/property/test_encoding_boundary_matrix.py` (#1305) is the executable
record of which contract applies where.

| Boundary | Module | Contract for edge cases |
| --- | --- | --- |
| JSON byte decoding | `polylogue/sources/decoder_json.py:decode_json_bytes` | UTF-8 BOM and BOM-bearing UTF-16 are decoded and the BOM is stripped; raw UTF-16 without a BOM is unsupported by design. |
| Content hash | `polylogue/core/hashing.py:hash_text`, `polylogue/pipeline/ids.py` | NFC normalization is applied to text fields (title, message text) before hashing, so NFC and NFD inputs produce identical `content_hash`. Lone surrogates raise `UnicodeEncodeError` (typed rejection — not silent corruption). |
| FTS5 indexing | `polylogue/storage/sqlite/archive_tiers/index.py` (`messages_fts`, unicode61) | Block search text is stored and indexed unchanged. RTL scripts (Arabic, Hebrew) and Latin-with-diacritics are word-tokenized; CJK runs index as a single token (substring queries against CJK are not supported). Zero-width and bidi characters pass through without crashing indexing. |
| FTS5 query escaping | `polylogue/storage/search/query_support.py:escape_fts5_query` | Every edge-case input produces a `MATCH`-safe query — bidi, zero-width, RTL, CJK, surrogate-pair emoji never raise `OperationalError`. |
| Terminal output | UTF-8 `TextIOWrapper` | All matrix strings pass through unchanged; lone surrogates raise `UnicodeEncodeError`. |

Edge cases covered: UTF-8/UTF-16 BOM, NFC/NFD equivalence, combining marks,
RTL (Arabic, Hebrew), bidi overrides, zero-width joiners and ZWNJ/ZWSP,
non-BMP characters (CJK extension B), ZWJ emoji sequences with skin-tone
modifiers, and unpaired surrogates.

## WAL Management

SQLite WAL behavior for the archive database (see also
`polylogue/storage/sqlite/connection_profile.py`):

- **Journal mode**: WAL on the write profile. Set once per database
  the first time a writer opens it; persists in the file header.
- **Connection cache profiles**: high-throughput ingest writes use the
  full write profile (`128 MiB` SQLite cache, `1 GiB` mmap allowance).
  Long-running daemon/ops writes use the daemon write profile (`16 MiB`
  cache, `64 MiB` mmap allowance) so small telemetry, cursor, heartbeat,
  OTLP, and maintenance writes do not keep batch-sized SQLite page-cache
  pressure charged to `polylogued.service`. Read-only daemon probes use
  the read profile (`query_only=ON`) instead of opening write connections.
- **Autocheckpoint threshold**: `WAL_AUTOCHECKPOINT_PAGES = 10000` =
  40 MiB. When the WAL crosses this, SQLite runs a PASSIVE checkpoint
  inline with the next commit.
- **Size cap** (#1614): `WAL_JOURNAL_SIZE_LIMIT_BYTES = 160 MiB` (4x
  the autocheckpoint threshold). Set via `PRAGMA journal_size_limit`
  on the write profile. When a TRUNCATE checkpoint succeeds (no
  readers blocking), SQLite truncates the WAL file down to the cap.
  A reader-blocked WAL still grows past the cap during the blocked
  window; the first successful TRUNCATE after the reader closes
  shrinks it back.
- **Read profile** (#1614): the canonical read profile sets
  `PRAGMA query_only = ON`. Read connections opened via
  `open_readonly_connection` already use the `mode=ro` URI flag for
  file-level read-only enforcement; the pragma additionally rejects
  accidental writes at SQL parse time instead of contending for the
  write lock and timing out.

### Relationship to #1602 (autocheckpoint revert)

A previous PR (3c5cc1a0) lowered the autocheckpoint threshold to 4 MB
to shrink the symptom of unbounded WAL growth under reader-blocked
TRUNCATE checkpoints. That tightening was reverted in #1602 once the
real fix landed: the autocheckpoint threshold is back to 40 MB
(`WAL_AUTOCHECKPOINT_PAGES = 10000`), and the new
`journal_size_limit` cap from #1614 is what bounds WAL growth
without paying the per-commit cost of a tighter autocheckpoint
threshold. The two work together — autocheckpoint is the routine
trim, journal_size_limit is the post-blocked-window shrink.

### Daemon-side periodic checkpoint

The daemon runs `maybe_checkpoint_wal()` every 5 minutes via the
periodic loop. This explicit TRUNCATE pass is the primary mechanism
that exercises the `journal_size_limit` cap in production — once the
reader contention clears, the next periodic pass shrinks the WAL.

## Content Hash Model

Archive writes are idempotent by content hash:

- SHA-256 over NFC-normalized (Unicode Normalization Form C) session
  payload
- Hashed fields: title, timestamps, messages, attachments, content blocks
- Excluded from hash: user metadata (tags, summaries, notes) — editing these
  does not trigger re-import
- Hash is computed in `pipeline/ids.py:session_content_hash()` and stored
  as `content_hash` on sessions
- On re-ingest, if the content hash matches, the session is skipped
  (idempotency). If it differs, the session is updated and dependent
  insights are rebuilt.

## FTS5 Model

Full-text search uses SQLite FTS5:

- **Tokenizer**: `unicode61` (no porter stemmer — this SQLite build doesn't
  include it). Case-insensitive for ASCII. Unicode-aware tokenization.
- **Content sync**: FTS5 indexes use `content='messages'` to stay in sync with
  the source table. Triggers handle INSERT/UPDATE/DELETE.
- **Trigger suspension (atomic, #1242)**: During bulk operations the writer
  drops the FTS triggers, bulk-writes, then re-creates the triggers and rebuilds
  the index (`INSERT INTO messages_fts(messages_fts) VALUES('rebuild')`) — all
  inside the single ingest transaction (`commit_archive_write_effects`). SQLite
  DDL is transactional, so the drop/restore is atomic with the commit: the
  committed state always has triggers, and a SIGKILL mid-batch rolls back to
  the triggers-present state on the next connection. The dropped-trigger state
  is therefore only ever observable by the writer's own connection mid-batch and
  never lands as committed drift. There is no separate FTS-trigger-drift health
  check or auto-restore loop — that machinery guarded a state nothing can
  commit (removed; the readiness/freshness check below is the real FTS net).
- **Query syntax**: FTS5 boolean operators (AND/OR/NOT), phrase search
  (`"exact phrase"`), prefix search (`prefix*`). Column filters are not
  directly exposed; use CLI/MCP filters instead.

## Blob Store Model

Content-addressed blob storage for large binary data:

- **Content addressing**: SHA-256 hash over raw bytes. The hash IS the address.
  Identical content → identical hash → automatic deduplication.
- **Prefix sharding**: 256 subdirectories (`blob/00/` through `blob/ff/`),
  each containing blobs keyed by the remaining 62 hex characters of the hash.
- **Linking**: `artifact_observations.link_group_key` groups related blobs
  (e.g., all blobs belonging to one session). The blob store itself
  (`polylogue/storage/blob_store.py`) is a pure content-addressed store with
  no notion of grouping; session-to-blob association is recorded as
  rows in `artifact_observations` keyed by `raw_id` with a shared
  `link_group_key`. (There is no separate `blob_links` table; the name is a
  historical alias for this row-group view of `artifact_observations`.)
- **Operations**: Blobs are write-once, read-many. No in-place modification.
  GC identifies unreferenced blobs via link counting.

### GC concurrency model — leases plus snapshot reference check

`run_blob_gc` deletes orphan blob files using two independent safety
invariants combined:

1. **DB reference check** — `_still_referenced` queries
   `raw_sessions` for the blob's `raw_id`. If a raw record points at
   the blob, GC skips it. This is the snapshot/mark-and-sweep view of
   "this blob is in active use right now."
2. **Pending lease check** — `_has_active_lease` queries
   `pending_blob_refs` for an in-flight operation that has *announced*
   it intends to reference the blob but hasn't committed yet. If a write
   path acquired a lease but its transaction hasn't yet inserted the
   `raw_sessions` row, the snapshot check alone would
   misclassify the blob as orphan and delete it.

The lease tables (`pending_blob_refs`, `gc_generations`) are therefore
load-bearing — they exist specifically to bridge the
acquire-blob → write-DB-row commit window. Removing them would
re-open the exact race the lease design was added to close: a GC pass
running between blob materialization and the corresponding archive
write would delete blobs the write was about to reference.

The write path that exercises the leases is
`polylogue/archive/write_effects.py:commit_archive_write_effects` —
calls `acquire_blob_leases(db_path, blob_hashes, operation_id)` on a
separate immediate-commit connection so the lease is visible to a
concurrent GC before the main transaction commits, then calls
`release_operation_leases(conn, operation_id)` after the commit so
the blob is now durably referenced by `raw_sessions` and the
lease can drop.

The acquire/release pair is wrapped in `try/finally` keyed by
`operation_id` (#1746). The lease is committed on its own connection,
so rolling back the main transaction on failure does NOT undo it — if
FTS repair or the commit raises, the `finally` opens a fresh
immediate-commit connection and releases the lease anyway. Without this
the lease row would leak into `pending_blob_refs` permanently and the
blob it named could never be GC'd.

The durable backstop is `sweep_orphaned_blob_leases`, run at daemon
startup (`polylogue/daemon/cli.py:_sweep_orphaned_blob_leases`,
mirroring `CursorStore._mark_interrupted_attempts` for
`live_ingest_attempt`). A writer SIGKILLed between acquire and release
cannot run its `finally`; the sweep deletes any `pending_blob_refs`
row older than `ORPHAN_LEASE_MAX_AGE_S` (3600 s), a bound generous
enough that a slow-but-live ingest is never swept out from under
itself.

`gc_generations` tracks the high-water mark of completed GC runs. The
"defense-in-depth" age gate is enforced in `run_blob_gc` (#1746): a
candidate blob must be older than `max(MIN_AGE_S, now -
prev_generation.completed_at)`. The generation term means a blob
created during the same window as the previous GC pass is never
reclaimed before its eventual reference can land. With no prior
generation recorded, the static `MIN_AGE_S` floor applies on its own.

- **Known issues**: GC has bugs with orphan detection and integrity
  verification ([#818](https://github.com/Sinity/polylogue/issues/818))
  — independent of the lease design audited above.

## Daemon Convergence Evidence

`polylogue ops diagnostics workload` produces a stable, JSON-serializable
snapshot of daemon-relevant state read directly from the archive SQLite
database. The probe is read-only and does not talk to the running daemon.

Raw acquisition, index materialization, and FTS readiness are treated as one
auditable convergence contract. A `source.db.raw_sessions` row that is not
explicitly skipped and has no matching `index.db.sessions` row is
`raw-materialization` archive debt; daemon status exposes it through
`component_readiness.raw_materialization` and
`raw_materialization_readiness` instead of reporting the archive as simply
healthy. FTS readiness is likewise a freshness invariant, not a best-effort
cache: stale or untrusted recorded counts make search readiness non-ready until
the index is demonstrably current.

For #845-style before/after convergence evidence snapshots:

```bash
polylogue ops diagnostics workload --json > before.json
# ...run convergence work (e.g. polylogued runs, ingest, debt drain)...
polylogue ops diagnostics workload --json > after.json
polylogue ops diagnostics workload --compare before.json after.json
polylogue ops diagnostics workload --compare before.json after.json --json > diff.json
```

The report has a stable top-level shape carrying its `report_version`,
`captured_at`, and structured sections that compare diffs arithmetically:

- `attempt_counts` — total/running/completed/failed `live_ingest_attempt`
  rows plus `stale_cursor_writes` and overlapping running source paths.
- `recent_attempts` — most recent attempts with read amplification,
  parse/convergence timings, and source-path bundles.
- `convergence_stage_timings` — min/max/sum/mean parse/convergence/read-
  amplification stats over completed attempts.
- `boundary_table_counts` — cheap planner-estimated counts for the
  daemon-relevant tables by default
  (`raw_sessions`, `sessions`, `messages`, `blocks`,
  `artifact_observations`, `messages_fts_docsize`, `actions`,
  `message_embeddings`, `session_profile`,
  `live_ingest_attempt`, `live_convergence_debt`, `pending_blob_refs`).
  Missing tables surface as `-1` and tables without SQLite planner
  statistics surface as `-2` rather than crashing the probe. Pass
  `--exact-table-counts` when a before/after evidence run needs exact
  arithmetic and the archive can afford the scans.
- `archive_tiers` — archive inventory for `source.db`,
  `index.db`, `embeddings.db`, `user.db`, and `ops.db`: file presence,
  durability/backup policy, `PRAGMA user_version`, missing backup-required
  tiers, and cheap planner-estimated table counts per tier. It does not run SQLite
  `PRAGMA quick_check` by default because that is a full-file integrity scan
  on large archives; pass `--integrity-check` when the workload snapshot should
  include that expensive evidence. It also avoids exact generated-text
  reconciliation by default; pass `--exact-derived-counts` when diagnosing
  FTS/source-row drift and the archive can afford the scan.
- `blob_lease_state` — pending lease count, distinct lease operations,
  oldest `acquired_at`. See the lease/GC concurrency model above.
- `blob_reference_debt` — skipped by default so routine workload snapshots do
  not stat every referenced blob path on large archives. Pass
  `--blob-reference-debt` when diagnosing backup/integrity incidents; the
  section then reports the exact missing referenced-blob count, bounded hash
  sample, reference-source counts, and source/index DB path used.
  For recovery planning, run
  `polylogue ops maintenance blob-reference-debt --output-format json`; that
  read-only classifier groups missing blob refs by table, ref type, origin,
  raw-row joinability, validation/parse state, and source-path availability.
  The paired `polylogue ops maintenance blob-reference-restore-direct`
  command only restores direct-file source paths after exact SHA-256
  verification; archive-member paths remain source re-acquisition work.
- `gc_state` — high-water `gc_generations` row, `last_completed_at`,
  total generation count.
- `fts_trigger_state` — the three expected FTS sync triggers
  (`messages_fts_a{i,d,u}`) with
  `present`, `missing`, and `all_present` fields. A missing trigger means
  FTS index drift risk (suspended during bulk operations and not
  restored, for example).
- `daemon_resource_signal` — RSS / cgroup memory / worker-progress fields
  pulled from the most recent `live_ingest_attempt` row (these are the
  only daemon-RSS signals readable without IPC).
- `source_path_churn`, `convergence_debt`, `query_plans` — source-path
  churn/read amplification, debt-by-stage, and hot-query EXPLAIN evidence.
  On archives, churn is read across `source.db.raw_sessions`
  and `index.db.sessions` so full-vs-append raw rows and unmaterialized raw
  payloads stay visible without requiring `raw_sessions` in `index.db`.

The compare mode refuses incompatible `report_version` inputs loudly and
requires both inputs to be `ok: True`.  Numeric fields produce
`{before, after, delta}` triples; the FTS trigger section reports
`regressed` (newly missing) and `restored` separately so trigger drift is
attributable to a specific convergence cycle.

## Daemon Metrics Endpoint (#1321)

`GET /metrics` returns the Prometheus text exposition format
(`text/plain; version=0.0.4`). Implementation lives in
`polylogue/daemon/metrics.py`; the route is wired in
`polylogue/daemon/http.py` alongside `/healthz/*` so all three scrape
surfaces share the same unauthenticated posture (Prometheus and
kubernetes/docker healthchecks cannot supply credentials, and the
daemon binds to loopback by default).

Series are derived from existing daemon state tables via
`open_readonly_connection` — `live_ingest_attempt` (totals by status,
in-flight gauge, recent-attempt duration min/mean/max), unresolved
`live_convergence_debt` grouped by stage, `pending_blob_refs` (pending
count, distinct operations), and the expected FTS sync triggers
from `active_fts_triggers_sync`. The same scrape also exposes embedding
backlog counts and the latest `embedding_catchup_runs` progress row so
semantic-search catch-up is visible in normal daemon dashboards without
running an operator CLI command. Missing tables degrade to zero samples
rather than 5xx-ing, so a fresh archive still emits the discovery
skeleton.

Archive layout observability is emitted on the same scrape. The
`polylogue_archive_storage_layout` gauge mirrors `polylogue config paths`
with bounded labels for `archive_missing`, `archive_partial`, and
`archive_complete`; `polylogue_archive_tier_count` and
`polylogue_archive_blocker_count` provide compact alerting
totals; the per-tier, per-blocker, and `active_tier_role` gauges give
the drilldown needed to tell whether the daemon is anchored on the
normal `index.db` path and which split-file tier is missing or stale.

Polylogue does not depend on `prometheus_client`; the exposition format
is hand-rolled. The OTLP HTTP receiver from #1224's ambitious-move
section is intentionally deferred and tracked under the same issue.

## Debugging Landmarks

Cross-check adjacent surfaces after changes:

- query: `cli/query*.py` ↔ `archive/filter/filters.py` ↔ `storage/search*.py`
- pipeline: `daemon/` ↔ `pipeline/` ↔ `storage/` ↔ `insights/`
- maintenance: `cli/commands/check.py` ↔ `storage/repair.py` ↔ `health.py`
- publication: `rendering/` ↔ `site/` ↔ `devtools/`
- schema: `schemas/` ↔ `sources/providers/` ↔ `pipeline/services/validation_*`

Drift check:

```bash
devtools render all --check
devtools test tests/unit/cli/test_demo_command.py tests/unit/demo/test_demo_seed_verify.py tests/visual
```

## Local State

- `.cache/`: disposable caches (hypothesis, pytest, mypy, ruff)
- `.local/`: untracked outputs (campaigns, demo artifacts, build artifacts)
- `.local/result`: out-link for `devtools release build-package`
