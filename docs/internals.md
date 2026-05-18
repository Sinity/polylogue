# Internals Reference

Working map of the live codebase: invariants, hot files, extension points, and
debugging landmarks. For the conceptual system shape, see
[architecture.md](architecture.md).

## Key Invariants

| Invariant | Enforced in |
| --- | --- |
| Archive writes are idempotent by content hash | `pipeline/ids.py`, `pipeline/prepare_enrichment.py` |
| Content hash excludes user metadata (tags, summaries) | `pipeline/ids.py:conversation_content_hash()` |
| Content hash uses NFC normalization | `core/hashing.py:hash_text()` |
| Async SQLite is the primary runtime; sync SQLite exists for CLI, schema tooling, and batch-ingest write paths | `storage/sqlite/async_sqlite.py`, `storage/sqlite/connection.py`, `pipeline/services/ingest_batch.py` |
| SQLite read/write tuning is profile-driven, not backend-local | `storage/sqlite/connection_profile.py` |
| FTS tokenizer is `unicode61` (no porter stemmer) | `storage/sqlite/schema_ddl_archive.py` |
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
| `storage/sqlite/schema_ddl.py` | Schema definition and `SCHEMA_VERSION` |
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
Run `devtools render-all` to update the generated catalog in
`docs/devtools.md`.

## Schema Versioning Model

Polylogue uses fresh-first schema versioning, not migration chains:

- `SCHEMA_VERSION` constant in `storage/sqlite/schema_ddl.py` is the authority
- On startup, the version in the database is compared against the constant
- **Version match**: normal operation
- **Version mismatch**: the database is rejected. There is no automatic
  migration. The operator must explicitly run a reviewed in-place upgrade
  script for that exact version transition.
- Schema is regenerated fresh for provider schemas via `devtools schema-generate`
  and promoted via `devtools schema-promote`

This design avoids migration-chain complexity (no Alembic, no forward/reverse
migrations, no partially-applied migration states) at the cost of requiring
explicit version-transition scripts.

## Learning Corrections (Feedback Loop)

User corrections are stored in `user_corrections` and live outside the
content-hash boundary by construction (#1131):

- Keyed by `(conversation_id, insight_kind)` — at most one correction of
  each kind per session, so deterministic rebuilds always produce the
  same merged insight output.
- Recognized kinds (closed `CorrectionKind` enum):
  `classification_override`, `tag_reject`, `tag_accept`,
  `summary_override`. New kinds are an explicit code change.
- Recording or removing a correction never touches
  `conversations.content_hash`. The hash invariant is asserted by
  `tests/unit/insights/test_feedback.py`.
- Insight materialization paths consult corrections after computing the
  heuristic suggestion. `classify_session()` is the first wired consumer
  (`apply_correction_to_classification`); auto-tag and summary merge
  helpers live in `polylogue/insights/feedback.py` for follow-on
  materialization paths.
- Surfaces:
  - CLI: `polylogue feedback {record,list,clear}`.
  - MCP: `record_correction`, `list_corrections`, `clear_corrections`.
  - Library: `Polylogue.record_correction(...)`,
    `Polylogue.list_corrections(...)`,
    `Polylogue.delete_correction(...)`, `Polylogue.clear_corrections(...)`.
- Storage backed by `polylogue/storage/insights/feedback/` (async SQL
  helpers) and `RepositoryWriteMixin.record_correction` /
  `list_corrections` / `delete_correction` / `clear_corrections`.

## Content Hash Model

Archive writes are idempotent by content hash:

- SHA-256 over NFC-normalized (Unicode Normalization Form C) conversation
  payload
- Hashed fields: title, timestamps, messages, attachments, content blocks
- Excluded from hash: user metadata (tags, summaries, notes) — editing these
  does not trigger re-import
- Hash is computed in `pipeline/ids.py:conversation_content_hash()` and stored
  as `content_hash` on conversations
- On re-ingest, if the content hash matches, the conversation is skipped
  (idempotency). If it differs, the conversation is updated and dependent
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
  no notion of grouping; conversation-to-blob association is recorded as
  rows in `artifact_observations` keyed by `raw_id` with a shared
  `link_group_key`. (There is no separate `blob_links` table; the name is a
  historical alias for this row-group view of `artifact_observations`.)
- **Operations**: Blobs are write-once, read-many. No in-place modification.
  GC identifies unreferenced blobs via link counting.

### GC concurrency model — leases plus snapshot reference check

`run_blob_gc` deletes orphan blob files using two independent safety
invariants combined:

1. **DB reference check** — `_still_referenced` queries
   `raw_conversations` for the blob's `raw_id`. If a raw record points at
   the blob, GC skips it. This is the snapshot/mark-and-sweep view of
   "this blob is in active use right now."
2. **Pending lease check** — `_has_active_lease` queries
   `pending_blob_refs` for an in-flight operation that has *announced*
   it intends to reference the blob but hasn't committed yet. If a write
   path acquired a lease but its transaction hasn't yet inserted the
   `raw_conversations` row, the snapshot check alone would
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
the blob is now durably referenced by `raw_conversations` and the
lease can drop.

`gc_generations` tracks the high-water mark of completed GC runs. The
"defense-in-depth" age check requires candidate blobs to be older than
the previous generation plus `MIN_AGE_S` so that a brand-new blob is
never considered for deletion within the same GC run cycle that
created it.

- **Known issues**: GC has bugs with orphan detection and integrity
  verification ([#818](https://github.com/Sinity/polylogue/issues/818))
  — independent of the lease design audited above.

## Daemon Convergence Evidence

`devtools daemon-workload-probe` produces a stable, JSON-serializable
snapshot of daemon-relevant state read directly from the archive SQLite
database. The probe is read-only and does not talk to the running daemon.

For #845-style before/after convergence proofs:

```bash
devtools daemon-workload-probe --json > before.json
# ...run convergence work (e.g. polylogued runs, ingest, debt drain)...
devtools daemon-workload-probe --json > after.json
devtools daemon-workload-probe --compare before.json after.json
devtools daemon-workload-probe --compare before.json after.json --json > diff.json
```

The report has a stable top-level shape carrying its `report_version`,
`captured_at`, and structured sections that compare diffs arithmetically:

- `attempt_counts` — total/running/completed/failed `live_ingest_attempt`
  rows plus `stale_cursor_writes` and overlapping running source paths.
- `recent_attempts` — most recent attempts with read amplification,
  parse/convergence timings, and source-path bundles.
- `convergence_stage_timings` — min/max/sum/mean parse/convergence/read-
  amplification stats over completed attempts.
- `boundary_table_counts` — row counts for the daemon-relevant tables
  (`raw_conversations`, `conversations`, `messages`, `content_blocks`,
  `artifact_observations`, `messages_fts_docsize`, `action_events`,
  `action_events_fts_docsize`, `message_embeddings`, `session_profile`,
  `live_ingest_attempt`, `live_convergence_debt`, `pending_blob_refs`).
  Missing tables surface as `-1` rather than crashing the probe.
- `blob_lease_state` — pending lease count, distinct lease operations,
  oldest `acquired_at`. See the lease/GC concurrency model above.
- `gc_state` — high-water `gc_generations` row, `last_completed_at`,
  total generation count.
- `fts_trigger_state` — the six expected FTS sync triggers
  (`messages_fts_a{i,d,u}`, `action_events_fts_a{i,d,u}`) with
  `present`, `missing`, and `all_present` fields. A missing trigger means
  FTS index drift risk (suspended during bulk operations and not
  restored, for example).
- `daemon_resource_signal` — RSS / cgroup memory / worker-progress fields
  pulled from the most recent `live_ingest_attempt` row (these are the
  only daemon-RSS signals readable without IPC).
- `source_path_churn`, `convergence_debt`, `query_plans` — pre-existing
  read amplification, debt-by-stage, and hot-query EXPLAIN evidence.

The compare mode refuses incompatible `report_version` inputs loudly and
requires both inputs to be `ok: True`.  Numeric fields produce
`{before, after, delta}` triples; the FTS trigger section reports
`regressed` (newly missing) and `restored` separately so trigger drift is
attributable to a specific convergence cycle.

## Debugging Landmarks

Cross-check adjacent surfaces after changes:

- query: `cli/query*.py` ↔ `archive/filter/filters.py` ↔ `storage/search*.py`
- pipeline: `daemon/` ↔ `pipeline/` ↔ `storage/` ↔ `insights/`
- maintenance: `cli/commands/check.py` ↔ `storage/repair.py` ↔ `health.py`
- publication: `rendering/` ↔ `site/` ↔ `showcase/`
- schema: `schemas/` ↔ `sources/providers/` ↔ `pipeline/services/validation_*`

Drift check:

```bash
devtools render-all --check
devtools lab-scenario verify-baselines
```

## Local State

- `.cache/`: disposable caches (hypothesis, pytest, mypy, ruff)
- `.local/`: untracked outputs (campaigns, showcases, build artifacts)
- `.local/result`: out-link for `devtools build-package`
