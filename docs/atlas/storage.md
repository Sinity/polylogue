# Storage

## Area boundary

Six SQLite tiers plus a content-addressed filesystem blob store. Durability, not subject matter, determines tier placement (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:49-85`).

## Tier map

| Tier | Runtime durability | Backup | Primary contents |
| --- | --- | --- | --- |
| `source.db` | `irreplaceable` | required | Raw acquisition records, blob references, publication reservations, GC generations and members, hook events, sidecars (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:50-55`; `polylogue/storage/sqlite/archive_tiers/source.py:416-453`; `polylogue/storage/sqlite/archive_tiers/source.py:692-747`; `polylogue/storage/sqlite/archive_tiers/source.py:797-844`) |
| `index.db` | `rebuildable` | no | Parsed sessions/messages/blocks, action pairs and the `actions` view, lineage links, FTS state, materialized session profiles (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:56-61`; `polylogue/storage/sqlite/archive_tiers/index.py:644-650`; `polylogue/storage/sqlite/archive_tiers/index.py:728-896`; `polylogue/storage/sqlite/archive_tiers/index.py:1038-1090`; `polylogue/storage/sqlite/archive_tiers/index.py:1558-1610`) |
| `embeddings.db` | `expensive_rebuild` | required | `vec0` vector table, metadata, refs, status, derivation state, failures (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:62-67`; `polylogue/storage/sqlite/archive_tiers/embeddings.py:22-73`) |
| `user.db` | `human` | required | Assertions, saved queries/result sets, annotation schemas and batches, settings, context-delivery provenance (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:68-73`; `polylogue/storage/sqlite/archive_tiers/user.py:17-64`; `polylogue/storage/sqlite/archive_tiers/user.py:257-363`) |
| `ops.db` | `disposable` | no | Ingest cursors, convergence debt, daemon stage/lifecycle events, MCP telemetry (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:74-79`; `polylogue/storage/sqlite/archive_tiers/ops.py:106-176`; `polylogue/storage/sqlite/archive_tiers/ops.py:199-287`; `polylogue/storage/sqlite/archive_tiers/ops.py:299-336`) |
| `audit.db` | `irreplaceable` | required | Operation previews, authorizations, runs, targets, attempts, events, continuity head (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:80-85`; `polylogue/storage/sqlite/archive_tiers/audit.py:51-211`; `polylogue/storage/sqlite/archive_tiers/audit.py:212-265`) |

Live page admission first proves that every active index raw reference exists
in source. A process-local healthy certificate starts with a complete check,
then consumes source/index transactional changed-key journals on each page;
it is discarded on restart or tier/schema change
(`polylogue/storage/frontier_existence.py:35-268`;
`polylogue/storage/sqlite/archive_tiers/source.py:481-499`;
`polylogue/storage/sqlite/archive_tiers/index.py:651-676`). Predecessor and
cursor authority are read afresh for the selected path component, including
paths with no retained raw. A raw or live cursor without its stored canonical
path causes an indexed refusal because an alias cannot be matched to a
selected path without inspecting every source row
(`polylogue/storage/raw_retention.py:1182-1380`;
`polylogue/storage/sqlite/archive_tiers/source.py:525-531`;
`polylogue/storage/sqlite/archive_tiers/ops.py:218-225`).

## Baseline construction and durable trains

The immutable fresh archive baseline creates all six tiers at version 1.
`initialize_active_archive_root` records that baseline's pending intent,
format marker and bootstrap receipt before advancing durable tiers through
their declared numbered trains. Source runtime version 2 replaces the two
raw-artifact partition indexes through `002_raw_artifact_failure_identity.sql`;
it retains every row and lets two missing-coordinate refusals at the same
source coordinate remain attached to their distinct raw acquisitions.
The index replacement claim is checked against SQLite's actual keys,
collations, order, uniqueness and complementary literal predicates in both
rehearsal and live execution; exact row values, primary keys, foreign keys
and integrity must survive the owned transaction. Other non-additive changes
still require verified backup authority (`storage/sqlite/migration_runner.py`).
Writable canonical bootstrap admits installed trains before runtime version
validation. Read-only and acquisition-only opens refuse a baseline Source
tier without applying migrations. A crash after baseline publication resumes
the same persisted train rather than restamping the baseline as current
(`storage/sqlite/archive_tiers/bootstrap.py`;
`storage/sqlite/durable_change_train.py`).
The format marker retains immutable baseline birth versions and fingerprints
after migration. Isolated runtime consumer probes and canonical schema census
build their empty connection through the baseline plus the actual numbered
migration runner. File-backed probes declare their owned temporary path;
in-memory probes consume a verified SQLite backup from an owned temporary
file and compare exact schema, rows and version before accepting it. Populated
or attached connections refuse. Probes never release the train
whose consumers they are proving.
Released train admission checks the physical archive identity, installed and
historical schema bindings, version, `quick_check` and `integrity_check`; it
does not count or hash mutable rows on ordinary restart. Interrupted APPLIED
or PROVEN recovery still requires exact typed row and schema equality with
the recorded post-apply evidence. Row proofs encode SQLite storage classes
and literal values, including embedded NUL bytes, in deterministic primary
key or actual rowid order (`storage/sqlite/migration_runner.py`).

Archive backups retain original numbered Source, User and Audit train manifests
only for the durable tiers included in the backup profile. The shared
`durable_train_manifest_paths` enumerator selects numbered history; process
lock files are not durable history. Copied receipts retain their original
physical bindings and bytes. They do not authorize a relocated backup inode
as the original live archive (`operations/archive_backup.py`). Explicit
`maintenance.restore_verified_backup` consumes an authenticated package through
`storage/sqlite/archive_population.py`, creates fresh destination train authority,
and preserves original receipts as detached provenance. Fixture clones use the
same deep owner. Partial durable cores refuse operational restoration; omitted
purchased Embeddings remain unrestored and produce degraded admission.

Literal row evidence streams current durable ROWID table TEXT and BLOB cells
through same-connection incremental handles, including primary keys and invalid
UTF-8. Metadata projections fetch only storage classes and numeric scalars.
Generic synthetic WITHOUT ROWID proofs keep keys in a private SQL ordinal
locator and transfer bounded literal chunks; SQLite may still allocate one
complete cell or sort full keys internally. No admitted durable archive uses
that shape, and this is not a general native SQLite memory bound.

## Identity and generated columns

- `sessions.session_id` is stored-generated as `origin || ':' || native_id` (`SESSIONS_SPEC` in `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:805-813`).
- `messages.message_id` is stored-generated with explicit namespace tags: native identity becomes `session_id || ':n:' || native_id`; content-derived identity becomes `session_id || ':c:' || content_identity || '.' || content_occurrence` -- a digest of the message's own declared semantic fields, so an insertion elsewhere in the export cannot renumber it onto a different message (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:337-342`).
- Content identity (`session_content_hash`, message content identity, revision projections) hashes canonical JSON: absence is `null`, distinct from `""` and every string; only the declared prose fields in `_NFC_TEXT_FIELDS` are NFC-folded, while identifiers, tool arguments, paths, metadata, event payloads, and mapping keys hash exactly, so canonically equivalent keys stay separate slots (`polylogue/pipeline/ids.py`). `messages.content_address`, the branch-point witness, follows the same rules (`_message_content_address` in `polylogue/storage/sqlite/archive_tiers/write.py`).
- `messages.identity_source` records which identity path fired; its index CHECK is generated from the semantic `MessageIdentitySource` Literal (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:360-365`; `polylogue/core/types.py:13-16`).
- `blocks.block_id` is stored-generated as `message_id || ':' || position`; tool command/path and `search_text` projections are virtual generated columns (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:561-566`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:635-650`).
- Sessions, messages, and blocks are `STRICT`; message and block ownership is enforced by cascading FKs (`polylogue/storage/sqlite/archive_tiers/index.py:527-647`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:330-333`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:93`).
- `material_origin` is independently constrained from role, preserving authoredness as a separate axis (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:399-405`).
- `blocks.tool_outcome` is the canonical structural outcome; a deliberate
  `unknown` is only storable together with its parser reason, enforced by a
  table CHECK that also keeps that reason off any other block shape. The
  `actions` view joins paired blocks and derives `result_state` from
  `tool_outcome` alone, so the legacy `is_error`/`exit_code` pair stays exposed
  for compatibility without deciding the state
  (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:606-611`;
  `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:642-652`;
  `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:594-605`;
  `polylogue/storage/sqlite/archive_tiers/index.py:1038-1090`).

## Parsed-session write choke point

Canonical message FTS publication retains the caller's transaction and reads
deletion targets in connection-bounded pages. It first walks the unchanged
blocks by their indexed `(message_id, position)` key, then walks remaining
owned identity residue by `block_id`. Each page cursor closes before companion
deletes; both scans preserve signed rowids and the existing session ownership
predicate. SQL inserts read text directly from blocks. A caller-owned
transaction needs no Python text digest, while publication prepared outside
the transaction retains its optimistic input revalidation
(`storage/fts/derivation.py`; `storage/fts/fts_lifecycle.py`).

- `write_parsed_session_to_archive` computes public origin, stored native identity, session identity, parser fingerprint, and lowering fingerprint before lowering one parsed session (function `write_parsed_session_to_archive` in `polylogue/storage/sqlite/archive_tiers/write.py`).
- It owns its transaction by default; bulk callers pass `manage_transaction=False` and own the surrounding commit to amortize per-commit fsync and WAL churn (the `manage_transaction` argument and transaction context in `write_parsed_session_to_archive`, `polylogue/storage/sqlite/archive_tiers/write.py`).
- It is the parsed-session lowering choke point shared by batch ingest (`polylogue/pipeline/services/ingest_batch/_core.py`) and authoritative revision replay/reindex (`write_with_reparse_receipt` in `polylogue/storage/sqlite/archive_tiers/revision_governance.py`). It is not the only mutation function in the six-tier substrate.

## Blob publication, liveness, and GC

- Blob paths are SHA-256-addressed as `<root>/<first-two-hex>/<remaining-hex>` (`polylogue/storage/blob_store.py:193-197`).
- Every preparation route hashes while writing a private staging file and fsyncs its bytes before publication; publication fsyncs the shard directory after an atomic `os.replace` (`polylogue/storage/blob_store.py:207-237`; `polylogue/storage/blob_store.py:245-274`; `polylogue/storage/blob_store.py:282-295`; `polylogue/storage/blob_store.py:303-316`).
- Archive publication commits durable reservation receipts before exposing final paths; the exact receipt is consumed in the durable-reference transaction (`BlobPublicationReservationStore.reserve_many` in `polylogue/storage/blob_publication.py:125`; `ArchiveBlobPublisher.flush` in the same file at `302-334`; `consume_blob_publication_receipt` at `553-564`).
- Liveness is descriptor-owned. Ordinary `blob_refs.ref_type` values must map unambiguously to one referent relation (`polylogue/storage/blob_liveness.py:90-113`).
- A destructive liveness check returns `LIVE`, `UNREFERENCED`, or typed `BLOCKED`; unavailable or unreadable required tiers block deletion (`polylogue/storage/blob_liveness.py:250-291`).
- GC safety requires no live DB reference, no publication reservation, a final locked recheck across control/source/index, an age floor, and bounded deletion batches (`polylogue/storage/blob_gc.py:7-25`).
- A refusal is not a quiet pass. Every preflight and final-recheck refusal sets `report.blocked_reason` and emits one `storage.blob_gc.refused` event carrying `outcome="refused"` and the phase, so a GC that refused can never be read as a GC that ran and reclaimed nothing (`polylogue/storage/blob_gc.py:1383-1385`; `polylogue/storage/blob_gc.py:796-800`; `polylogue/storage/blob_gc.py:78-104`). Callers that collapse the report into counts must read `blocked_reason` before believing a zero (`polylogue/storage/blob_gc.py:1013-1016`).
- Refusals inside the locked execution window also emit `storage.blob_gc.refused`; `_execute_gc_generation_members` records the reason and returns the deletions already committed in that batch (`polylogue/storage/blob_gc.py:789-800`; `polylogue/storage/blob_gc.py:816-825`; `polylogue/storage/blob_gc.py:883-895`). Trust the report for the partial counts and the event for the refusal phase.

### Two-phase `gc_generations`

1. Commit one generation and every exact member intent as `pending` before any unlink (`gc_generation_members` in `polylogue/storage/sqlite/archive_tiers/source.py:689-704`; `polylogue/storage/blob_gc.py:525-569`).
2. Under `BEGIN IMMEDIATE` on source and index, recheck liveness/reservations, unlink or reconcile each member, commit outcomes, then finalize only when no pending members remain (`polylogue/storage/blob_gc.py:735-900`; `polylogue/storage/blob_gc.py:589-620`).

Pending generations are restartable; a restart resumes their exact member set instead of rediscovering intent from the filesystem, and refuses an intent whose blob namespace was swapped or remounted (`polylogue/storage/blob_gc.py:622-638`; `polylogue/storage/blob_gc.py:640-664`; `polylogue/storage/blob_gc.py:916-951`).

## Lineage storage model

- A prefix-sharing child stores only its divergent tail. The writer resolves the parent, compares composed signatures, records the last inherited message as the branch point, and lowers only the remaining messages (`_prepared_message_context` in `polylogue/storage/sqlite/archive_tiers/write.py:1694-1753`).
- `session_links` stores destination identity, resolved parent, branch point and its content address, inheritance mode, status, parent tool-use block, method, confidence, and evidence (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:1263-1304`).
- Reads plan the composition before materializing it: one iterative walk, bounded only by its visited set, resolves the ancestral prefix into per-session segment lengths, with explicit cycle and dangling-branch-point status instead of silently claiming completeness. No depth cap drops a valid ancestor (`_composed_transcript_plan` in `polylogue/storage/sqlite/archive_tiers/write.py:3399`).
- A write never strands an inheriting child. Before a parent full replace, the writer records each direct prefix-sharing child's inherited rows. Afterwards, a child whose branch point no longer resolves gets that pre-write prefix materialized into its own rows and stops inheriting (`spawned-fresh`, still linked to its parent); descendants anchored in those rows follow them. A child whose branch point still resolves keeps inheriting the parent's current prefix (`_capture_inherited_prefixes` and `_settle_inherited_prefixes` in `polylogue/storage/sqlite/archive_tiers/write.py:11905` and `12038`). A full read materializes every segment; a bounded page fetches only the window the caller asked for, so a deep child's first paint costs the chain depth rather than the composed transcript (`read_archive_session_page` in `polylogue/storage/sqlite/archive_tiers/write.py:3881`).
- Link writes refuse to let parser inference overwrite an existing hook-authoritative edge, rather than losing it to last-writer-wins (`_upsert_session_link` in `polylogue/storage/sqlite/archive_tiers/write.py:7852-7894`).
- Provider usage counters are NOT sliced like messages. A prefix-sharing child keeps its own reported `total_*` lanes verbatim; only a usage event bound to a replayed prefix message is dropped, because the parent already owns that observation (`_provider_usage_event_row` in `polylogue/storage/sqlite/archive_tiers/write.py:9527`; `_reextract_provider_usage_tail_db` in the same file at `13204`).
- Logical-session usage is therefore the chain root's observation plus each prefix-sharing descendant's own, not a root plus deltas (`polylogue/storage/usage.py:1837-1849`).

## Invariants and gotchas

- `branch_point_message_id` is deliberately not an FK. Parent full replacement deletes before reinserting deterministic message IDs; `ON DELETE SET NULL` would fire during the DELETE step and permanently sever the child (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:1255-1274`).
- A failed or unavailable liveness surface is not equivalent to zero references (`polylogue/storage/blob_liveness.py:321-340`; `polylogue/storage/blob_gc.py:9-13`).
- A published blob may legitimately have no durable ref yet; its reservation protects that publication window (`polylogue/storage/blob_publication.py:110-150`; `polylogue/storage/sqlite/archive_tiers/source.py:730-739`).
- GC history counters are summaries derived only after all member outcomes close; member rows are the crash-recovery authority (`gc_generation_members` in `polylogue/storage/sqlite/archive_tiers/source.py:689-704`; `polylogue/storage/blob_gc.py:571-588`).
- A retained agent work event (`append_work_event`) is its own logical source: its `agent-work-event:` raw id is also its logical key and source path, admitted as a byte-proven singleton baseline, so it never joins the byte-revision cohort or accepted head of the transcript it annotates (`write_work_event_raw_and_parsed_result` in `polylogue/storage/sqlite/archive_tiers/revision_governance.py`). Its write is event-only and keeps the session's `raw_id` and `content_hash`. A cold build replays work-event keys after every byte and membership cohort, so the transcript's fresh write never meets a session an event created (`backfill_historical_revision_evidence` in `polylogue/sources/revision_backfill.py`). Excision seeds work-event raws by the session's `(origin, native_id)`, and source conservation counts one as materialized when its session is indexed.
- Rebuildable `index.db` must not become authority for an irreversible durable mutation; blob GC therefore requires source-ledger and active-index checks to agree (`polylogue/storage/blob_gc.py:7-20`; `polylogue/storage/blob_liveness.py:321-359`).

## DISCREPANCIES

- The `docs/architecture.md` ring diagram draws only source, index, embeddings, user, and ops; code has six tiers and includes `audit.db` (`docs/architecture.md:25`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:49-85`).
- `docs/architecture.md` calls embeddings plainly rebuildable; runtime metadata classifies them as `expensive_rebuild` with backup required (`docs/architecture.md:52-55`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:62-67`).

Existing index and ops tiers are admitted by their derived schema identity
before initialization issues any DDL. A mismatch raises typed `SchemaSkew`;
initialization never patches or restamps a foreign derived identity. An
admitted INDEX tier installs only canonical runtime performance indexes before
manifest validation, following its writable sync and async policy; read-only
opens still refuse a missing manifest index without writing. At daemon
startup, exclusive archive ownership and a write lease allow replacement of a
stale disposable ops file before persistent tier handles open. Index
reconvergence keeps its declared reset/rebuild route; purchased embeddings and
durable tiers are never replaced by ops startup.

## Tool-result association

`storage/sqlite/action_pairs.py` owns associations for both materialized and
canonical action reads and observed `tool_finished` events. An orphan result
before the first use of an ID cannot certify that use. A clean alternating
stream supports sequential ID reuse without a many-to-many join. A missing
middle result, duplicate receipt, or overlapping same-ID invocation makes
that invocation and the remaining same-ID suffix unresolved: the action has
`result_state=outcome_unknown`, `outcome_unknown_reason=ambiguous_tool_id_reuse`,
and no attributed result block, output, error flag or exit code. A clean final
use with no result stays `no_result`.

Session IDs isolate fork-local tool IDs. Variant creation order is not a
causal branch identity: when a same-ID stream involves variants, only an
otherwise unambiguous result in the same message is attributed; cross-message
associations remain unknown rather than treating equal variant indexes as a
branch path. These are evidence refusals, not loss of the underlying blocks.
Reingesting complete evidence recomputes the relation at its ordinary write
boundary. The decision does not depend on row insertion order or on whether
a reader uses the materialized or directly computed relation.

Facade settings and context-delivery reads distinguish an absent row in a
readable tier from unavailable authority. Required user state, source events
for Hermes delivery correlation, and the ops injection ledger refuse with
`ArchiveTierUnavailableError` when their tier cannot be inspected. Read calls
never create missing tiers. Named-source freshness reads current cursor progress
only from ops `ingest_cursor`; a missing offset remains unknown, and successful
empty canonical state cannot inherit progress from a retired index table.
