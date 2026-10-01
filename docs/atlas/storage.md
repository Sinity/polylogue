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

## Write custody and connection lifetime

Each outer write lease names its archive root and holds an owned, descriptor-anchored
`.archive-write-custody.lock` before writable archive SQL begins. Nested owners
borrow that custody; a thread receives authority through a single-use grant,
and a task cannot acquire authority by inheriting another task's context.
The lock file remains in place after release. Directory and lock identities
are checked before and after acquisition; replacement is a visible refusal.

An ArchiveStore retains custody while any write transaction or temporary User
writer handle remains unsettled. Commit, rollback and close settle its actual
handles before release. Native connection construction registers each actual
handle before schema or profile SQL; successful setup hands the idle handle to
its caller, while failed construction cleanup retains its original owner.
A failed close retains those handles and exposes the Store in a typed settlement
error. Cleanup must execute on the original SQLite thread. A finished task may
leave cleanup custody on that thread, but cannot leave authority for a new
mutation. Async SQLite close retains its existing worker and raw handle when
raw close fails; it stops the worker only after the raw handle has closed.

If a daemon operation finishes with unsettled SQL, the coordinator retains its
original worker, context and grant. A later admission or shutdown may request
one cleanup attempt on that worker before acquiring new write custody. The
mailbox accepts cleanup only. Failed settlement is retryable and keeps the
actual handles; cancellation of a waiter does not cancel accepted cleanup.
Status reports retained workers and async backends, and shutdown reports idle
only after those owners settle.

The index rebuild lock has a separate lifetime: RebuildLease acquires its
exclusive lock nonblocking under brief write custody, then releases that
acquisition custody. Each accepted write segment takes its own physical custody.
The exclusive rebuild lock survives accepted work and failed SQL settlement;
competing active-writer and rebuild admissions refuse promptly rather than wait
for that lifetime lock. An idle ArchiveStore retains its shared rebuild lock,
but releases physical write custody between mutations.

Native connection caching lasts for one admitted operation. Nested contexts
reuse that operation's handle; a standalone context takes the existing physical
write lease. Outer settlement closes every successfully idle cached handle,
including SQL committed after context exit. An active transaction or failed
close stays pinned and yields a typed settlement refusal. Promotion settles
current-operation cached handles before changing the Index pointer, so later
cached artifact validation opens the current generation. Schema corpus validation
uses its explicit Source connection. It never derives
file identity from a pathname or a separate SQLite metadata descriptor.
Verified-leaf descriptors remain with the same native owner until SQL closes,
including body and profile failures. Sync and async sibling attachment carries
the configured archive root separately from SQLite's resolved generation path,
so promotion retains the intended Source, User and Ops authority.

ReadFrame owns its connection and cursors on the request or page-loop thread.
Its existing process census holds a strong reference until actual connection
close succeeds, including failed construction or cleanup. Close attempts every
cursor and the connection; failed cleanup retains the concrete frame for retry.
Every caller closes through its context or ExitStack. Only declared cancellation
may interrupt from another thread. Independent read frames do not acquire writer
custody; readers inside an admitted async write operation retain that operation's
existing grant until actual SQL close and original worker exit.

Compute and admitted archive-read workers complete their physical Future only after the existing native-owner census settles on the creator thread. Failed close retains the worker, submitter context, reservation or read lease, and backing artifacts; cancellation of an asyncio wrapper does not release them. The existing custody owner can request another cleanup attempt. Archive readers and reference seals bind native children to their actual terminal parent, and healthy parents retire after all SQL and artifact obligations settle. Revision projections close their writer before transfer and close each bounded readonly page before yielding immutable rows (`polylogue/core/sql_settlement.py`, `polylogue/storage/sqlite/connection_profile.py`, `polylogue/pipeline/ids.py`).

## Identity and generated columns

- `sessions.session_id` is stored-generated as `origin || ':' || native_id` (`SESSIONS_SPEC` in `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:805-813`).
- `messages.message_id` is stored-generated with explicit namespace tags: native identity becomes `session_id || ':n:' || native_id`; content-derived identity becomes `session_id || ':c:' || content_identity || '.' || content_occurrence` -- a digest of the message's own declared semantic fields, so an insertion elsewhere in the export cannot renumber it onto a different message (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:337-342`).
- Content hashes use canonical JSON: absence stays `null` and empty strings stay `""`. Only declared prose fields in `_NFC_TEXT_FIELDS` are NFC-folded; identifiers, tool arguments, paths, metadata, event payloads and mapping keys stay exact. Content-derived message IDs use one recursive encoder that preserves ordinary pre-a44 null, empty-string and empty-block preimages. Literal reserved values and lossy typed/key lowering receive a framed typed projection, while a reserved-looking mapping key alone does not trigger an escape. An occurrence ordinal distinguishes repeated identical messages without depending on their position (`polylogue/pipeline/ids.py`).
- `messages.content_address` witnesses the complete declared message hash projection, including block metadata, file edits and web constructs. A parent replacement may retain a branch edge only where that complete witness agrees; a same-ID message with changed content cannot silently become the child's inherited prefix (`_message_content_address` in `polylogue/storage/sqlite/archive_tiers/write.py`).
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

- `write_parsed_session_to_archive` computes public origin, stored native identity, session identity, parser fingerprint, and lowering fingerprint before lowering one parsed session (function `write_parsed_session_to_archive` in `polylogue/storage/sqlite/archive_tiers/write.py`).
- Every producer declares its actual destination and transaction owner. Active archive writes use a durable-reference seal; an owned inactive generation defers the archive-wide proof to promotion. A genuine non-archive memory index declares its standalone transaction explicitly. Missing archive arguments or missing durable tiers do not grant standalone permission.
- Bulk callers reuse one `IndexMutationScope` per commit window. Its disk-backed witness stores original typed lookups, re-resolves their targets before publication, and includes composed descendants affected by parent changes. Inserts, aliases and lineage changes also receive proof: unchanged message IDs alone do not establish preserved lookup or block-position semantics. Generation replacement checks the whole candidate. User/Audit JSON anchor enumeration remains global and unindexed; batching amortizes that census rather than making it independent of archive size (`polylogue/storage/sqlite/reference_seal.py`).
- It is the parsed-session lowering choke point shared by batch ingest (`polylogue/pipeline/services/ingest_batch/_core.py`) and authoritative revision replay/reindex (`write_with_reparse_receipt` in `polylogue/storage/sqlite/archive_tiers/revision_governance.py`). It is not the only mutation function in the six-tier substrate.

## Blob publication, liveness, and GC

- Blob paths are SHA-256-addressed as `<root>/<first-two-hex>/<remaining-hex>` (`polylogue/storage/blob_store.py:193-197`).
- Every preparation route hashes while writing a private staging file and fsyncs its bytes before publication; publication fsyncs the shard directory after an atomic `os.replace` (`polylogue/storage/blob_store.py:207-237`; `polylogue/storage/blob_store.py:245-274`; `polylogue/storage/blob_store.py:282-295`; `polylogue/storage/blob_store.py:303-316`).
- Archive publication commits durable reservation receipts before exposing final paths; the exact receipt is consumed in the durable-reference transaction (`polylogue/storage/blob_publication.py:110-150`; `polylogue/storage/blob_publication.py:212-224`; `polylogue/storage/blob_publication.py:270-283`).
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

- A prefix-sharing child stores only its divergent tail. The writer resolves the parent, compares composed signatures, records the last inherited message as the branch point, and lowers only the remaining messages (`polylogue/storage/sqlite/archive_tiers/write.py:804-867`).
- `session_links` stores destination identity, resolved parent, branch point and its content address, inheritance mode, status, parent tool-use block, method, confidence, and evidence (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:1263-1304`).
- Reads plan the composition before materializing it: one iterative walk, bounded only by its visited set, resolves the ancestral prefix into per-session segment lengths, with explicit cycle and dangling-branch-point status instead of silently claiming completeness. No depth cap drops a valid ancestor (`polylogue/storage/sqlite/archive_tiers/write.py:2540-2625`).
- A write never strands an inheriting child. Before a parent full replace, the writer records each direct prefix-sharing child's inherited rows. Afterwards, a child whose branch point no longer resolves gets that pre-write prefix materialized into its own rows and stops inheriting (`spawned-fresh`, still linked to its parent); descendants anchored in those rows follow them. A child whose branch point still resolves keeps inheriting the parent's current prefix (`polylogue/storage/sqlite/archive_tiers/write.py:10257-10420`). A full read materializes every segment; a bounded page fetches only the window the caller asked for, so a deep child's first paint costs the chain depth rather than the composed transcript (`polylogue/storage/sqlite/archive_tiers/write.py:3024-3060`).
- Link writes refuse to let parser inference overwrite an existing hook-authoritative edge, rather than losing it to last-writer-wins (`polylogue/storage/sqlite/archive_tiers/write.py:5333-5389`).
- Provider usage counters are NOT sliced like messages. A prefix-sharing child keeps its own reported `total_*` lanes verbatim; only a usage event bound to a replayed prefix message is dropped, because the parent already owns that observation (`polylogue/storage/sqlite/archive_tiers/write.py:6432-6452`; `polylogue/storage/sqlite/archive_tiers/write.py:8605-8630`).
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
