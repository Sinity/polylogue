# Storage

## Area boundary

Six SQLite tiers plus a content-addressed filesystem blob store. Durability, not subject matter, determines tier placement (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:49-85`).

## Tier map

| Tier | Runtime durability | Backup | Primary contents |
| --- | --- | --- | --- |
| `source.db` | `irreplaceable` | required | Raw acquisition records, blob references, publication reservations, GC generations and members, hook events, sidecars (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:50-55`; `polylogue/storage/sqlite/archive_tiers/source.py:561-614`; `polylogue/storage/sqlite/archive_tiers/source.py:666-705`; `polylogue/storage/sqlite/archive_tiers/source.py:728-739`) |
| `index.db` | `rebuildable` | no | Parsed sessions/messages/blocks, action pairs and the `actions` view, lineage links, FTS state, materialized session profiles (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:56-61`; `polylogue/storage/sqlite/archive_tiers/index.py:527-647`; `polylogue/storage/sqlite/archive_tiers/index.py:802-867`; `polylogue/storage/sqlite/archive_tiers/index.py:917-942`; `polylogue/storage/sqlite/archive_tiers/index.py:1341-1411`) |
| `embeddings.db` | `expensive_rebuild` | required | `vec0` vector table, metadata, refs, status, derivation state, failures (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:62-67`; `polylogue/storage/sqlite/archive_tiers/embeddings.py:22-73`) |
| `user.db` | `human` | required | Assertions, saved queries/result sets, annotation schemas and batches, settings, context-delivery provenance (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:68-73`; `polylogue/storage/sqlite/archive_tiers/user.py:17-64`; `polylogue/storage/sqlite/archive_tiers/user.py:257-363`) |
| `ops.db` | `disposable` | no | Ingest cursors, convergence debt, daemon stage/lifecycle events, MCP telemetry (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:74-79`; `polylogue/storage/sqlite/archive_tiers/ops.py:106-176`; `polylogue/storage/sqlite/archive_tiers/ops.py:198-278`; `polylogue/storage/sqlite/archive_tiers/ops.py:290-327`) |
| `audit.db` | `irreplaceable` | required | Operation previews, authorizations, runs, targets, attempts, events, continuity head (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:80-85`; `polylogue/storage/sqlite/archive_tiers/audit.py:51-211`; `polylogue/storage/sqlite/archive_tiers/audit.py:212-265`) |

## Identity and generated columns

- `sessions.session_id` is stored-generated as `origin || ':' || native_id` (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:773-779`).
- `messages.message_id` is stored-generated with explicit namespace tags: native identity becomes `session_id || ':n:' || native_id`; content-derived identity becomes `session_id || ':c:' || content_identity || '.' || content_occurrence` -- a digest of the message's own declared semantic fields, so an insertion elsewhere in the export cannot renumber it onto a different message (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:324-329`).
- `blocks.block_id` is stored-generated as `message_id || ':' || position`; tool command/path and `search_text` projections are virtual generated columns (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:548-553`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:624-636`).
- Sessions, messages, and blocks are `STRICT`; message and block ownership is enforced by cascading FKs (`polylogue/storage/sqlite/archive_tiers/index.py:527-647`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:330-333`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:93`).
- `material_origin` is independently constrained from role, preserving authoredness as a separate axis (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:386-391`).
- `blocks.tool_outcome` is the canonical structural outcome; a deliberate
  `unknown` is only storable together with its parser reason, enforced by a
  table CHECK that also keeps that reason off any other block shape. The
  `actions` view joins paired blocks and derives `result_state` from
  `tool_outcome` alone, so the legacy `is_error`/`exit_code` pair stays exposed
  for compatibility without deciding the state
  (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:606-611`;
  `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:642-652`;
  `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:594-605`;
  `polylogue/storage/sqlite/archive_tiers/index.py:848-867`).

## Parsed-session write choke point

- `write_parsed_session_to_archive` computes public origin, stored native identity, session identity, parser fingerprint, and lowering fingerprint before lowering one parsed session (`polylogue/storage/sqlite/archive_tiers/write.py:1021-1124`).
- It owns its transaction by default; bulk callers pass `manage_transaction=False` and own the surrounding commit to amortize per-commit fsync and WAL churn (`polylogue/storage/sqlite/archive_tiers/write.py:1070-1074`; `polylogue/storage/sqlite/archive_tiers/write.py:1241`).
- It is the parsed-session lowering choke point shared by batch ingest and authoritative revision replay/reindex (`polylogue/storage/sqlite/archive_tiers/write.py:1108`; `polylogue/pipeline/services/ingest_batch/_core.py:1399`; `polylogue/storage/sqlite/archive_tiers/revision_governance.py:442`). It is not the only mutation function in the six-tier substrate.

## Blob publication, liveness, and GC

- Blob paths are SHA-256-addressed as `<root>/<first-two-hex>/<remaining-hex>` (`polylogue/storage/blob_store.py:193-197`).
- Every preparation route hashes while writing a private staging file and fsyncs its bytes before publication; publication fsyncs the shard directory after an atomic `os.replace` (`polylogue/storage/blob_store.py:207-237`; `polylogue/storage/blob_store.py:245-274`; `polylogue/storage/blob_store.py:282-295`; `polylogue/storage/blob_store.py:303-316`).
- Archive publication commits durable reservation receipts before exposing final paths; the exact receipt is consumed in the durable-reference transaction (`polylogue/storage/blob_publication.py:110-150`; `polylogue/storage/blob_publication.py:212-224`; `polylogue/storage/blob_publication.py:270-283`).
- Liveness is descriptor-owned. Ordinary `blob_refs.ref_type` values must map unambiguously to one referent relation (`polylogue/storage/blob_liveness.py:90-113`).
- A destructive liveness check returns `LIVE`, `UNREFERENCED`, or typed `BLOCKED`; unavailable or unreadable required tiers block deletion (`polylogue/storage/blob_liveness.py:250-291`).
- GC safety requires no live DB reference, no publication reservation, a final locked recheck across control/source/index, an age floor, and bounded deletion batches (`polylogue/storage/blob_gc.py:7-25`).
- A refusal is not a quiet pass. Every preflight and final-recheck refusal sets `report.blocked_reason` and emits one `storage.blob_gc.refused` event carrying `outcome="refused"` and the phase, so a GC that refused can never be read as a GC that ran and reclaimed nothing (`polylogue/storage/blob_gc.py:1383-1385`; `polylogue/storage/blob_gc.py:796-800`; `polylogue/storage/blob_gc.py:78-104`). Callers that collapse the report into counts must read `blocked_reason` before believing a zero (`polylogue/storage/blob_gc.py:1013-1016`).
- The refusals inside the locked execution window are the exception: `_execute_gc_generation_members` sets `blocked_reason` and returns without emitting anything, and the per-member return carries the deletions already committed in that batch (`polylogue/storage/blob_gc.py:752`; `polylogue/storage/blob_gc.py:801`; `polylogue/storage/blob_gc.py:886`). Trust the report, not the event stream, for an aborted unlink pass.

### Two-phase `gc_generations`

1. Commit one generation and every exact member intent as `pending` before any unlink (`polylogue/storage/sqlite/archive_tiers/source.py:585-614`; `polylogue/storage/blob_gc.py:525-569`).
2. Under `BEGIN IMMEDIATE` on source and index, recheck liveness/reservations, unlink or reconcile each member, commit outcomes, then finalize only when no pending members remain (`polylogue/storage/blob_gc.py:735-900`; `polylogue/storage/blob_gc.py:589-620`).

Pending generations are restartable; a restart resumes their exact member set instead of rediscovering intent from the filesystem, and refuses an intent whose blob namespace was swapped or remounted (`polylogue/storage/blob_gc.py:622-638`; `polylogue/storage/blob_gc.py:640-664`; `polylogue/storage/blob_gc.py:916-951`).

## Lineage storage model

- A prefix-sharing child stores only its divergent tail. The writer resolves the parent, compares composed signatures, records the last inherited message as the branch point, and lowers only the remaining messages (`polylogue/storage/sqlite/archive_tiers/write.py:804-867`).
- `session_links` stores destination identity, resolved parent, branch point and its content address, inheritance mode, status, parent tool-use block, method, confidence, and evidence (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:1263-1304`).
- Reads plan the composition before materializing it: one recursive walk resolves the ancestral prefix into per-session segment lengths, with explicit depth-limit and dangling-branch-point status instead of silently claiming completeness (`polylogue/storage/sqlite/archive_tiers/write.py:2138-2207`). A full read materializes every segment; a bounded page fetches only the window the caller asked for, so a deep child's first paint costs the chain depth rather than the composed transcript (`polylogue/storage/sqlite/archive_tiers/write.py:2574-2603`).
- Link writes refuse to let parser inference overwrite an existing hook-authoritative edge, rather than losing it to last-writer-wins (`polylogue/storage/sqlite/archive_tiers/write.py:5333-5389`).
- Provider usage counters are NOT sliced like messages. A prefix-sharing child keeps its own reported `total_*` lanes verbatim; only a usage event bound to a replayed prefix message is dropped, because the parent already owns that observation (`polylogue/storage/sqlite/archive_tiers/write.py:6432-6452`; `polylogue/storage/sqlite/archive_tiers/write.py:8605-8630`).
- Logical-session usage is therefore the chain root's observation plus each prefix-sharing descendant's own, not a root plus deltas (`polylogue/storage/usage.py:1837-1849`).

## Invariants and gotchas

- `branch_point_message_id` is deliberately not an FK. Parent full replacement deletes before reinserting deterministic message IDs; `ON DELETE SET NULL` would fire during the DELETE step and permanently sever the child (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:1255-1274`).
- A failed or unavailable liveness surface is not equivalent to zero references (`polylogue/storage/blob_liveness.py:321-340`; `polylogue/storage/blob_gc.py:9-13`).
- A published blob may legitimately have no durable ref yet; its reservation protects that publication window (`polylogue/storage/blob_publication.py:110-150`; `polylogue/storage/sqlite/archive_tiers/source.py:574-584`).
- GC history counters are summaries derived only after all member outcomes close; member rows are the crash-recovery authority (`polylogue/storage/sqlite/archive_tiers/source.py:594-614`; `polylogue/storage/blob_gc.py:571-588`).
- Rebuildable `index.db` must not become authority for an irreversible durable mutation; blob GC therefore requires source-ledger and active-index checks to agree (`polylogue/storage/blob_gc.py:7-20`; `polylogue/storage/blob_liveness.py:321-359`).

## DISCREPANCIES

- The `docs/architecture.md` ring diagram draws only source, index, embeddings, user, and ops; code has six tiers and includes `audit.db` (`docs/architecture.md:25`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:49-85`).
- `docs/architecture.md` calls embeddings plainly rebuildable; runtime metadata classifies them as `expensive_rebuild` with backup required (`docs/architecture.md:52-55`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:62-67`).
