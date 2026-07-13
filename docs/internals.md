# Internals Reference

Working map of the live codebase: invariants, hot files, extension points, and
debugging landmarks. For the conceptual system shape, see
[architecture.md](architecture.md).

## Key Invariants

| Invariant | Enforced in |
| --- | --- |
| Archive writes are idempotent by content hash | `pipeline/ids.py`, `pipeline/services/ingest_batch/_core.py` |
| Content hash excludes user metadata (tags, summaries) | `pipeline/ids.py:session_content_hash()` |
| Content hash uses NFC normalization | `core/hashing.py:hash_text()` |
| Async SQLite is the primary runtime; sync SQLite exists for CLI, schema tooling, and batch-ingest write paths | `storage/sqlite/async_sqlite.py`, `storage/sqlite/connection.py`, `pipeline/services/ingest_batch/_core.py` |
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
| `storage/repository/__init__.py` | Repository facade (10-mixin composition: archive reads, archive writes, raw, vectors, and six insight readers — profile, run-projection, timeline, thread, summary, topology) |
| `storage/search_providers/fts5.py` | Lexical search |
| `storage/search_providers/hybrid.py` | Hybrid retrieval (RRF fusion) |

### Sources and Pipeline

| File | Purpose |
| --- | --- |
| `sources/dispatch.py` | Provider detection and parser routing |
| `sources/parsers/*.py` | Per-provider parsing |
| `pipeline/ingest_support.py` | Ingest stage definitions and source selection helpers |
| `pipeline/ids.py` | Content hashing and ID generation |
| `pipeline/services/ingest_batch/_core.py` | Batch ingest (largest pipeline file) |

## Extension Points

**Adding a provider**: Start at `sources/dispatch.py:detect_provider()`. Add a
`looks_like()` function in a new parser under `sources/parsers/`. Add a
`Provider` enum variant in `core/enums.py`. Add a provider schema bundle under
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

Polylogue has two schema-evolution regimes, keyed by tier durability.

- Tier version constants under `storage/sqlite/archive_tiers/` are the
  authority. The canonical fresh schema is described directly by each tier DDL.
- **Durable tiers** (`source.db`, `user.db`) may use explicit additive
  migrations. Migration SQL lives under
  `storage/sqlite/migrations/{source,user}/NNN_name.sql`, advances
  `PRAGMA user_version` one step at a time, and requires a verified backup
  manifest containing the affected tier before it runs. Verification restores
  the backup into scratch, checks every included SQLite tier and referenced
  blob, then writes a versioned receipt authenticated by a per-tier HMAC key
  under the Polylogue XDG state directory. The key path is derived independently
  from the live archive path and is never copied into the backup. This prevents
  public backup hashes or a transplanted receipt from impersonating successful
  verification. It is an artifact-integrity boundary, not protection against
  hostile arbitrary code running as the same Unix user, which can read the
  local `0600` key. Every accepted bundle artifact must also be a contained,
  single-link regular file: symlinks, hardlinks, non-regular files, linked blob
  ancestry, and a target tier that aliases the live database are rejected so
  migration cannot mutate its own recovery copy. SQLite `-wal`, `-shm`, and
  rollback-journal sidecars are forbidden in a verified bundle; checkpointed
  copies are read in immutable mode, so no unsigned sidecar can change their
  logical contents. The receipt also binds a closed recursive inventory of
  every directory and file (except the receipt itself), including auxiliary
  debt reports, so no undeclared artifact can appear, disappear, or change.
  Additive means
  `CREATE TABLE`, `CREATE INDEX`, `ADD COLUMN`, and bounded backfills.
  Destructive durable-tier changes require a copy-forward design and explicit
  operator consent.
- **Derived tiers** (`index.db`, `embeddings.db`) do not have in-place migration
  chains. They are rebuildable products over durable source/user evidence:
  schema mismatches are handled by rebuilding or blue-green replacing the
  affected derived tier from source, preserving durable tiers.
- **Disposable tiers** (`ops.db`) may keep narrow bootstrap-time `ALTER TABLE`
  helpers for daemon telemetry because the tier is disposable.
- On startup the on-disk `PRAGMA user_version` is compared against the tier
  constant:
  - **Empty file** (`user_version == 0`): bootstrap fresh.
  - **Version match**: open as-is.
  - **Older durable tier**: refuse ordinary open and require explicit migration
    with a backup manifest.
  - **Derived mismatch or newer durable tier**: reject and require rebuild or a
    newer runtime as appropriate.
- During development, schema changes are triaged before reset/reingest:
  metadata-only, index-only, additive-derived, additive-durable, or
  semantic-reparse-required. Same-tier schema changes from ready Beads should be
  batched before a live rebuild so the active archive is not reset repeatedly.
- `devtools lab policy schema-versioning` enforces the boundary: durable SQL
  migrations are allowed only under the numbered migration resource roots, while
  derived-tier upgrade helpers remain forbidden.

- User schema version 7 adds durable content-addressed `queries`, mutable
  `query_names`, promoted `result_sets`/`result_set_members`, and planner
  emitted `query_edges`. Existing `saved_query` assertions are repointed to
  their canonical `query:<hash>` target in the same verified-backup migration.
  Exact result members are retained only for watch/pinned/finding/cohort
  persistence classes; ordinary execution relations remain ops-tier telemetry.
- User schema version 6 adds immutable, fingerprinted `annotation_schemas`
  definitions and independent `annotation_batches` provenance containers.
  Batch labels remain assertion rows linked through an
  `annotation-batch:<id>` `scope_ref`. Existing v5 tiers migrate additively
  only after an authenticated verified-backup receipt; fresh user tiers create
  the same canonical tables and registered `delegation.discourse@v1` schema.
- User schema version 5 adds `context_deliveries`, an immutable receipt ledger
  for the exact context image delivered to an agent, including recipient,
  actor, run/boundary, included refs, omissions, caveats, and image digest.
  Existing v4 tiers migrate additively after an authenticated verified-backup
  receipt; fresh user tiers create the table directly.
- User schema version 4 adds `user_settings`, a durable key/value table for
  workspace/settings state that is not an epistemic assertion. Existing v3
  user tiers migrate additively after a verified backup manifest; fresh user
  tiers create the table directly.
- Source schema version 3 drops `pending_blob_refs` (polylogue-v7e0). The
  table backed a blob-GC lease mechanism (`acquire_blob_leases`/
  `release_operation_leases`) that a race-window audit found no production
  ingest caller ever populated — the table was provably empty in every real
  deployment (zero writers anywhere in the write path) — see "GC concurrency
  model" below for the current, lease-free contract. Existing v2 source
  tiers migrate via
  `storage/sqlite/migrations/source/003_drop_pending_blob_refs.sql` after a
  verified backup manifest; fresh source tiers never create the table.
- Source schema version 4 adds `blob_publication_reservations`, an exact
  filesystem-publication boundary rather than the removed late write-effects
  lease. Rows are keyed by per-publication receipt ID and indexed by content
  hash, so concurrent publishers of identical bytes remain independent.
  Existing v3 tiers migrate additively through
  `004_blob_publication_reservations.sql` after a verified backup manifest.
- Index schema version 35 adds Polish lexical search-recall folding
  (polylogue-9jsi). The `messages_fts`, `threads_fts`, and
  `session_work_events_fts` tokenizers move from `unicode61` to
  `unicode61 remove_diacritics 2`, which folds ordinary combining-mark
  diacritics (`ó`->`o`, `ż`->`z`, `ą`->`a`, ...) symmetrically for indexed
  and `MATCH` query text. Separately, `ł`/`Ł` (Latin L with stroke) has no
  Unicode decomposition, so neither NFD normalization nor
  `remove_diacritics` can fold it; every write path for the three FTS
  surfaces—fresh-write triggers, full rebuilds, missing-row repair, and
  dangling derived-surface repair—applies an inline `REPLACE`-chain fold
  (`polylogue/storage/fts/pl_fold.py:pl_fold_sql_expr`) to the indexed
  `text` column, and `escape_fts5_query` applies the byte-identical Python
  fold (`pl_fold`) to query text, so `latwo`/`zrobilem` queries find seeded
  `łatwo`/`zrobiłem` content. The fold is deliberately narrow (`ł`/`Ł`
  only) and expressed once from `PL_FOLD_TABLE`; real-route rebuild and
  repair tests lock the write-side calls to query-side normalization.
  Existing index tiers must be
  rebuilt from source evidence (`polylogue ops reset --index && polylogued
  run`) to pick up the new tokenizer and re-fold already-indexed rows.
- Index schema version 34 rebuilds the `delegations` view (polylogue-y964,
  polylogue-4c27). The prior view aliased `session_links.src_session_id`
  (canonically the CHILD) as `parent_session_id` and
  `resolved_dst_session_id` (canonically the PARENT) as `child_session_id` —
  backwards — and joined `branch_point_message_id` (the child's last
  inherited parent message, for lineage composition) as if it were the Task
  dispatch pointer. The rebuilt view spines on parent-side `actions` rows
  (`semantic_type='subagent'`) instead of `session_links`, exposes a
  `mapping_state` (`resolved`/`unresolved`/`ambiguous`/`edge_only`/
  `quarantined`) instead of silently dropping unpaired or quarantined edges,
  and separates model identity into `dispatch_turn_model` (the message that
  authored the dispatch), `requested_model` (an explicit routing override
  from the dispatch tool_input, if the provider recorded one), and
  `parent_session_dominant_model`/`child_session_dominant_model` (the
  session-level dominant-model fallback, explicitly named and excluded from
  turn-level claims) — replacing the old `orchestrator_model`/
  `subagent_model` columns, which conflated a session-wide aggregate with
  per-turn dispatch authorship. Existing index tiers must be rebuilt from
  source evidence (`polylogue ops reset --index && polylogued run`); no
  public reader should keep consuming the old column names.
- Index schema version 33 adds `'provider_usage'` to the
  `insight_materialization.insight_type` CHECK constraint (polylogue-f2qv.5),
  so `session_model_usage` can carry a materializer-version stamp and
  self-heal through the same session-insight rebuild path as
  `session_profile`/`latency`/etc instead of requiring a manual
  `ops reset --index` after a provider-usage materializer fix. Existing index
  tiers must be rebuilt from source evidence
  (`polylogue ops reset --index && polylogued run`) — a `CREATE TABLE IF NOT
  EXISTS` on an already-existing table does not retroactively widen its CHECK
  constraint, so the version bump (not just the DDL edit) is what forces
  existing archives through the fresh-first rebuild path rather than hitting a
  CHECK-constraint violation the first time a session's insights are rebuilt.
- Index schema version 30 makes `session_events` the lossless generic relation
  for every parsed non-message event. It retains open event types and structured
  payloads in original positions while policy and usage tables remain typed
  projections. The original provider-local source-message reference is stored
  independently from its nullable canonical `messages.message_id` resolution,
  so unresolved and lineage-normalized references remain auditable. Existing index tiers must be rebuilt from source evidence
  (`polylogue ops reset --index && polylogued run`).
- Index schema version 28 adds the `delegations` VIEW, derived from
  `session_links` (`link_type='subagent'`) LEFT JOIN'd to the parent's Task
  dispatch `actions` row and both sessions' `session_profiles` — no
  convergence stage needed, matching the `actions` view's own derived-tier
  precedent. `result_status` (`ok`/`error`/`unknown`) is derived only from
  `actions.is_error`/`exit_code`, never guessed.
- Index schema version 27 adds `session_profiles.primary_model_name` and
  `primary_model_family`, the dominant model per session by assistant
  OUTPUT-token share, backing the `delegations` view above.
- Index schema version 26 rewrites the `actions` view to pair `tool_use`/
  `tool_result` blocks by transcript rank within `(session_id, tool_id)`
  instead of a plain `tool_id` equality join, which fanned out N*M when a
  provider re-emitted the same `tool_id` on distinct messages.
- Index schema version 25 adds `blocks.content_hash`, a SHA-256 over
  canonical block evidence (type, text, tool_name, canonical tool_input,
  semantic/media/language, is_error, exit_code — excluding
  session_id/message_id/position/tool_id) so a stored block citation anchor
  survives fork-position shift, re-ingest renumbering, and provider tool-id
  regeneration. See `polylogue/storage/block_anchor.py`.
- Index schema version 24 admits `capture_gap` rows in `session_events`. These
  are narrow lifecycle evidence events emitted when ingest rejects a lower-
  precedence DOM browser-capture fallback because a richer source row already
  owns the session. Existing index tiers must be rebuilt from source evidence
  (`polylogue ops reset --index && polylogued run`).
- Index schema version 23 adds `idx_blocks_search_text_populated`, a partial
  index over text-bearing `blocks` rows, and makes `fts_freshness_state` part of
  the canonical fresh index tier. Message search readiness compares that source
  set with `messages_fts_docsize` and consults the durable freshness marker
  before running user FTS queries; without the partial index and ledger, large
  archives can spend the query budget scanning `blocks` merely to decide whether
  `polylogue find hermes` is allowed to run. Existing index tiers must be
  rebuilt from source evidence (`polylogue ops reset --index && polylogued
  run`).
- Index schema version 22 adds `idx_blocks_tool_result_outcome`, a partial
  index over structured `tool_result` outcome fields. Claim-vs-evidence and
  action-outcome reads anchor on provider-reported `is_error` / non-zero
  `exit_code` rather than assistant prose; without an outcome-leading index,
  large archives must scan the whole tool-result block set before pairing
  actions. Existing index tiers must be rebuilt from source evidence
  (`polylogue ops reset --index && polylogued run`).
- Index schema version 21 adds `idx_messages_embedding_prose`, a partial
  covering index for authored prose messages eligible for paid embeddings:
  standard `message` rows from `user`/`assistant` roles, with
  `human_authored`/`assistant_authored` material origin and positive word
  count. Embedding preflight, status detail, and archive-session embedding
  reads use this index when present so cost windows do not scan unrelated tool,
  protocol, context-pack, or runtime rows in large archives. Existing index
  tiers must be rebuilt from source evidence
  (`polylogue ops reset --index && polylogued run`).
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
  They are derived/rebuildable enrichments, not the only query substrate:
  terminal `run` / `observed-event` / `context-snapshot` queries also synthesize
  cheap local rows directly from `sessions` and `blocks` (main runs,
  `session_started`, tool-finished outcomes, and session-start context
  snapshots). The durable evidence stays `session_events` + `messages` +
  `session_links`, and materialized rows are retained only where they encode
  richer non-local projections that are not cheap to lower directly. Existing
  index tiers must be rebuilt from source evidence (`polylogue ops reset --index
  && polylogued run`).
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

For **derived tiers** (`index.db`, `embeddings.db`) this design intentionally
rejects in-place upgrade-chain complexity (no Alembic, no forward/reverse
upgrade scripts, no partially-applied upgrade states, no
`_apply_version_upgrade_plan` rollback windows): a derived-tier schema mismatch
rebuilds or blue-green-replaces the tier from durable source/user evidence.
Files that are not configured archive paths are not classified or handled by
the archive runtime.

For **durable tiers** (`source.db`, `user.db`) the boundary is different, because
`user.db` holds irreplaceable human assertions that cannot be rebuilt from
source. These tiers use explicit *additive* numbered SQL migrations under
`storage/sqlite/migrations/{source,user}/NNN_*.sql`, applied one `PRAGMA
user_version` step at a time by `migration_runner.py` behind a **verified backup
manifest** for the affected tier. Additive means `CREATE TABLE`/`CREATE INDEX`/
`ADD COLUMN`/bounded backfill; destructive durable-tier changes require a
copy-forward design and explicit operator consent, never a routine migration.

`ops.db` (disposable daemon telemetry) may additionally use narrowly scoped
`ALTER TABLE ... ADD COLUMN` bootstrap helpers in
`storage/sqlite/archive_tiers/bootstrap.py`
(ingest-cursor runtime fields, cursor-lag rollups). The
`devtools lab policy schema-versioning` lint enforces the whole boundary:
numbered durable-tier migrations are allowed; derived-tier upgrade helpers are
forbidden.

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

Source schema version 2 removes the old unique `(origin, native_id)` raw-row
constraint. A native session can have multiple durable source observations
(direct export, browser native capture, DOM fallback, historical ZIPs) while
still coalescing to one canonical indexed session; source evidence must not be
replaced merely because two captures describe the same provider-native id.

## Topology Edges (#1258)

The `session_links` table (index tier) persists every parent reference asserted
by a parser as a typed row, including references whose parent has not yet been
ingested (out-of-order ingestion) or has been hard-deleted. The pre-existing
fast path (`sessions.parent_session_id` set when the parent is in the prepare
cache) is unchanged; `session_links` is an additional durable record that always
carries the original provider-native parent id. The same table also carries the
#2467 prefix-sharing lineage columns (`branch_point_message_id`, `inheritance`).

- **Identity:** primary key
  `(src_session_id, dst_origin, dst_native_id, link_type)`.
  Re-ingesting the same child is idempotent.
- **Closed enums:** `polylogue/core/enums.py` owns `LinkType`
  (continuation / sidechain / subagent / branch / fork / resume / repaired),
  re-exported as `TopologyEdgeType` from
  `polylogue/archive/topology/edge.py`, and `TopologyEdgeStatus`
  (unresolved / resolved / repaired / quarantined). `quarantined` marks a link
  the topology reducer dropped to break a cycle.
- **Resolve:** every session save runs `resolve_session_links_for_session`
  (`polylogue/storage/sqlite/queries/session_links.py`) so that an out-of-order
  child's edge flips to `resolved` the moment its parent's native id appears in
  `sessions`.
- **Hash boundary:** these links are derived per ingest and are NOT part
  of `sessions.content_hash` — mirrors the same boundary as user correction
  assertions (#1131).

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

User corrections live outside the content-hash boundary by construction
(#1131). In the split archive they are `AssertionKind.CORRECTION` rows in the
unified `user.db` `assertions` table; a legacy `user_corrections` table is still
read for pre-split single-file archives (`storage/insights/feedback/`):

- Keyed by session and correction kind — at most one correction of
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
- **Contentless, not external-content**: `messages_fts` is declared
  `content=''` with `contentless_delete=1` — it does NOT sync against a
  `messages` (or any other) source table by re-reading it. The indexed text
  is `blocks.search_text`, a `VIRTUAL` generated column (`trim(text || ' ' ||
  tool_name || ' ' || tool_input.command/file_path/path)`), supplied
  explicitly at insert time by three rowid-keyed triggers
  (`messages_fts_ai`/`_ad`/`_au`) on `blocks`, not read back from any content
  table. Practical consequence: `snippet()`/`highlight()` return `NULL`
  (contentless tables have no stored text for FTS5 to slice), so every read
  path that needs the actual text joins `blocks` by `rowid` and uses
  `b.search_text` directly instead.
- **No trigger-suspension/rebuild loop**: an earlier design (#1242) dropped
  the FTS triggers during bulk writes and rebuilt the index afterward
  (`INSERT INTO messages_fts(messages_fts) VALUES('rebuild')`); that
  auto-restore loop has been removed — it guarded a mid-batch state that
  never actually landed as committed drift (SQLite DDL is transactional, so
  a SIGKILL mid-batch always rolled back to the triggers-present state
  anyway). The real net today is the freshness/repair machinery: the
  `fts_freshness_state` durable marker plus the `idx_blocks_search_text_populated`
  partial index, which message-search readiness consults against
  `messages_fts_docsize` before running a user FTS query (index schema
  version 23, see the schema-version-history note above), and the
  per-session repair path (#1851) that reconciles a specific session's FTS
  rows against `blocks.search_text` on demand.
- **Query syntax**: FTS5 boolean operators (AND/OR/NOT), phrase search
  (`"exact phrase"`), prefix search (`prefix*`). Column filters are not
  directly exposed; use CLI/MCP filters instead.

## Blob Store Model

Content-addressed blob storage for large binary data:

- **Content addressing**: SHA-256 hash over raw bytes. The hash IS the address.
  Identical content → identical hash → automatic deduplication.
- **Prefix sharding**: 256 subdirectories (`blob/00/` through `blob/ff/`),
  each containing blobs keyed by the remaining 62 hex characters of the hash.
- **Linking**: `raw_artifacts.link_group_key` groups related blobs
  (e.g., all blobs belonging to one session). The blob store itself
  (`polylogue/storage/blob_store.py`) is a pure content-addressed store with
  no notion of grouping; session-to-blob association is recorded as
  rows in `raw_artifacts` keyed by `raw_id` with a shared
  `link_group_key`. (There is no separate `blob_links` table; the name is a
  historical alias for this row-group view of `raw_artifacts`.)
- **Operations**: Blobs are write-once, read-many. No in-place modification.
  GC identifies unreferenced blobs via a snapshot reference check plus an
  age floor (see GC concurrency model below), not simple link counting.

### GC concurrency model — publication reservation, reference, and age floor

`run_blob_gc` deletes orphan blob files using three safety
invariants combined:

1. **DB reference check** — `_still_referenced` queries `raw_sessions` for
   the blob's `raw_id`. If a raw record points at the blob, GC skips it.
   This is the snapshot/mark-and-sweep view of "this blob is in active use
   right now." It only sees *committed* rows: a blob whose referencing
   ingest has written the bytes to disk but not yet committed the row is,
   by SQLite's isolation, indistinguishable from a true orphan to this
   check alone.
2. **Publication receipt** — archive orchestration prepares a bounded batch of
   private temporary files, commits every per-publication receipt in one
   source-tier transaction, then publishes every final content-addressed path.
   The exact source-reference transaction consumes its own receipt ID; an
   index-only attachment consumes its receipt only after the index commit.
   Same-hash publishers cannot consume one another. Pure parser/source APIs
   receive an injected writer and remain independent of archive paths/schema.
   GC enumerates outside its lock, then holds the source-tier write lock only
   across the bounded final reference/receipt recheck and unlink. Dry-run is
   read-only.
3. **Age floor** — a candidate must be older than
   `max(MIN_AGE_S, now - prev_generation.completed_at)`
   (`polylogue/storage/blob_gc.py:run_blob_gc_report`). `MIN_AGE_S` is 60
   seconds; `gc_generations` tracks the high-water mark of completed GC
   runs so a blob created during the same window as the previous pass is
   never reclaimed before its eventual reference can land. With no prior
   generation recorded, the static `MIN_AGE_S` floor applies on its own.

The age floor was previously the sole defense against the race the reference
check cannot see. A prior revision carried a late lease mechanism (`pending_blob_refs`,
`acquire_blob_leases`/`release_operation_leases`) meant to make that window
explicit rather than relying on a timing heuristic. It was fully
implemented and unit-tested but **never reachable in production**: the only
call site that could have populated the lease payload keys
(`commit_archive_write_effects`'s `_blob_hashes`/`_operation_id`) was never
given them by any real ingest caller (`_commit_sync_ingest_side_effects`
built its payload without them). A race-window audit
(`docs/audits/2026-07-09-race-window-audit.md`, rows 1a/1b) confirmed zero
production callers across the whole write path, so `has_lease` was always
`False` and the acquire/release calls never ran. The mechanism was removed
rather than left as dead code implying a protection that did not exist
(polylogue-v7e0). A deterministic provider-shaped measurement later proved
the real window structurally unbounded: acquisition prefetched 128 artifacts
and committed references in batches of 500, so a slow following artifact
could age an earlier blob past 60 seconds. Source v4 therefore reserves at
the only boundary that closes the race: before final-path visibility.

Crash reconciliation has no TTL. It classifies and retains receipts by
default, including missing-path receipts: absence is not proof that a paused
publisher died. Automatic clearing requires archive-wide writer exclusion.
Existing blob-without-reference rows remain explicit recoverable acquisition
debt until reacquisition or confirmed operator abandonment through
`polylogue ops maintenance blob-publications`. Age is never treated as proof
that a publisher is dead. Archive backup holds the same writer exclusion and
copies an exact hash/size inventory for the union of durable references and
publication receipts.

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

Session insights are the same class of daemon-enforced derived-model
invariant, alongside FTS and embeddings. Missing or stale session profiles,
work events, phases, threads, and run projections remain convergence debt until
the insights stage has rebuilt them; a bounded false result is retried like the
other derived-model stages rather than becoming an operator-maintenance task.
The rebuild is idempotent by session id and materializer version, and the
daemon records the stage timing and debt evidence through the normal convergence
attempt path.

Insight rebuilds deliberately stay out of the ingest write transaction even
though they are automatic. They can hydrate and summarize large sessions, so
the materializer commits per message-budget chunk to keep WAL and write-lock
scope bounded. They also batch hot, still-changing source files until a quiet
window, avoiding repeated expensive rebuilds while Codex or Claude continues to
append to a session. Finally, the materializer has its own versioned rebuild
contract: changing the derived insight algorithm should refresh the rebuildable
read model without changing the durable source/index write boundary.

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
- `boundary_table_counts` — mixed cheap evidence for the tables in
  `_BOUNDARY_TABLES`/`_OPS_BOUNDARY_TABLES`
  (`devtools/daemon_workload_probe.py`, non-exhaustive here): exact counts for
  small/core archive cardinality tables and maintained rollups such as
  `sessions`, `raw_sessions`, `messages`, and `session_profiles`;
  planner-estimated counts only where exact counting would scan large derived
  tables (`blocks`, `messages_fts_docsize`, `message_embeddings`,
  `session_events`, `session_links`, `repos`, `session_repos`,
  `session_commits`, `live_ingest_attempt`, `convergence_debt`).
  Missing tables surface as `-1` and expensive tables without SQLite planner
  statistics surface as `-2` rather than crashing the probe. Pass
  `--exact-derived-counts` when a before/after evidence run needs exact
  arithmetic and the archive can afford the scans. The companion
  `boundary_table_count_precision` map labels each value as `exact`,
  `estimate`, `unavailable`, or `missing`.
- `archive_tiers` — archive inventory for `source.db`,
  `index.db`, `embeddings.db`, `user.db`, and `ops.db`: file presence,
  durability/backup policy, `PRAGMA user_version`, missing backup-required
  tiers, and the same mixed cheap/exact table counts per tier with a
  `table_count_precision` map for each tier. It does not run SQLite
  `PRAGMA quick_check` by default because that is a full-file integrity scan
  on large archives; pass `--integrity-check` when the workload snapshot should
  include that expensive evidence. It also avoids exact generated-text
  reconciliation by default; pass `--exact-derived-counts` when diagnosing
  FTS/source-row drift and the archive can afford the scan.
- `blob_reference_debt` — skipped by default so routine workload snapshots do
  not stat every referenced blob path on large archives. Pass
  `--blob-reference-debt` when diagnosing backup/integrity incidents; the
  section then reports the exact missing referenced-blob count, bounded hash
  sample, reference-source counts, and source/index DB path used.
  For recovery planning, run
  `polylogue ops maintenance blob-reference-debt --output-format json`; that
  read-only classifier groups missing blob refs by table, ref type, origin,
  raw-row joinability, validation/parse state, and source-path availability.
  During daemon convergence, direct source files whose current bytes still hash
  to a missing blob address are restored automatically before raw
  materialization replay. Container/member paths such as
  `export.zip:conversations.json` remain source re-acquisition work because the
  referenced blob may be an extracted record inside the member, not the member
  file itself.
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
`live_convergence_debt` grouped by stage, and the expected FTS sync triggers
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
