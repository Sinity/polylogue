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
- **Mismatch → re-ingest from source.** Polylogue does not patch an
  out-of-band shape into the canonical one. The operator moves the
  database aside and runs `polylogue ops reset --database && polylogued run`,
  which re-acquires from the source archives and rebuilds the canonical
  archive.
- Schema bumps are deletes-then-defines, never deltas. A schema change
  edits the owning tier DDL/version and documents the re-ingest expectation.
  No upgrade helpers are added for the bump.
- Provider schemas (the parsing/validation surface, distinct from the
  storage schema) are still regenerated fresh via
  `devtools lab schema generate` and promoted via `devtools lab schema promote`.

This design intentionally rejects in-place upgrade-chain complexity (no
Alembic, no forward/reverse upgrade scripts, no partially-applied upgrade
states, no `_apply_version_upgrade_plan` rollback windows). If the configured
archive path is not the current schema, the operator moves it aside and
re-ingests from source. Files that are not configured archive paths are not
classified or handled by the archive runtime.

## Archive Activation

The archive file set is split by durability class:

- `source.db` stores raw acquisition rows and source evidence that
  rebuildable projections depend on.
- `index.db` stores the parsed session/message/block tree, FTS/search
  indexes, graph/topology rows, and derived insight read models.
- `embeddings.db` stores vector rows, embedding status, and embedding
  catch-up metadata; it is rebuildable, but expensive.
- `user.db` stores irreplaceable human input such as marks,
  annotations, corrections, user tags, session metadata, saved views,
  recall packs, workspaces, and blackboard notes.
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
  - CLI: `polylogue user-state feedback {record,list,clear}`.
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
- **Trigger suspension**: During bulk operations, FTS triggers are suspended
  for performance, then re-enabled and the index rebuilt via
  `INSERT INTO messages_fts(messages_fts) VALUES('rebuild')`.
- **Risk**: SIGKILL during trigger suspension leaves FTS out of sync.
  Mitigation: daemon/CLI startup checks and restores FTS triggers.
- **Catch-up health signalling** (#1613): the FAST-tier
  `fts_trigger_drift` health check downgrades from CRITICAL to INFO
  when triggers are missing inside a fresh in-flight bulk attempt
  (`live_ingest_attempt.status='running'`, `phase IN
  ('full_parse','full_worker_wait')`, heartbeat within
  `_BULK_ATTEMPT_FRESHNESS_S` seconds). The writer dropped the triggers
  inside its own transaction; `commit_archive_write_effects` will
  restore them before the commit lands, so other readers never see
  the dropped state — the only entity that observes "15/15 missing"
  is the writer's own connection mid-batch. Stale or orphaned
  in-flight rows (no heartbeat in the freshness window) still escalate
  to CRITICAL because that signature is identical to the SIGKILL leak.
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
count, distinct operations), and the six expected FTS sync triggers
from `_EXPECTED_FTS_TRIGGERS`. The same scrape also exposes embedding
backlog counts and the latest `embedding_catchup_runs` progress row so
semantic-search catch-up is visible in normal daemon dashboards without
running an operator CLI command. Missing tables degrade to zero samples
rather than 5xx-ing, so a fresh archive still emits the discovery
skeleton.

Archive layout observability is emitted on the same scrape. The
`polylogue_archive_storage_layout` gauge mirrors `polylogue ops paths`
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
- publication: `rendering/` ↔ `site/` ↔ `showcase/`
- schema: `schemas/` ↔ `sources/providers/` ↔ `pipeline/services/validation_*`

Drift check:

```bash
devtools render all --check
devtools lab scenario verify-baselines
```

## Local State

- `.cache/`: disposable caches (hypothesis, pytest, mypy, ruff)
- `.local/`: untracked outputs (campaigns, showcases, build artifacts)
- `.local/result`: out-link for `devtools release build-package`
