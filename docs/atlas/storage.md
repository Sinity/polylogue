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

Large streamed session-event arrays are stored as ordered JSON item rows in
`session_event_array_items`, keyed by session, event position and payload key.
`session_events.payload_json` keeps the other event fields; typed event reads
restore array fields as JSON arrays in the returned payload, preserving the
public event shape. Those public full-event reads intentionally materialize the
arrays. Ingest, hashing, prepared serialization and index writes replay the
ordered item rows without collecting them. The table is part of the rebuildable
index schema and cascades with its event.

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

Raw/index materialization readiness may settle an unmatched raw as a valid
non-session only from a complete typed, non-terminal artifact
(`parse_as_session=0`, not schema-eligible, without decode errors or malformed
JSONL), a current complete parser receipt, and a current zero-member
`raw_membership_census.status='non_session'` receipt whose exact identities
match. A path or payload that merely resembles a sidecar does not settle an
unreceipted raw; decode failures, unsupported/refused artifacts, validation
refusals, mixed session cohorts, and missing or stale receipts remain visible
gaps (`storage/archive_readiness.py`).
Retained replay uses the same membership currency requirement. Missing or older
non-session membership receipts remain pending, and the canonical Source census
publishes the current receipt before replay; a schema exemption cannot skip it.

The bounded daemon status projection reuses this durable receipt and identity
check without opening raw blobs. It keeps the raw/index join count visible
while excluding only receipt-backed non-session raws from its unchecked-gap
count. Every artifact in the cohort must remain raw-only and free of terminal
support, decode, malformed-line, or validation evidence for that fast
classification to apply.

Prepared frontier inspection uses the resident preparation owner and the same
original Source, Index and Ops inputs through publication. Source migration
005 and the derived-tier DDL journal changes to the dependencies of accepted
heads; an Ops coverage mark binds the measured journal watermarks and physical
tier identity. A current mark certifies relational coverage. Explicit
maintenance inspection also checks physical blob fingerprints through the
existing verified-byte cache, because external blob changes have no SQLite
journal entry. Relocating an archive invalidates the path-bound runtime mark;
retained artifact completeness is checked from its census and physical inputs.

Blocker acknowledgement prepares and validates the operator-visible facts
before audited intent begins, physically closes that preparation, then
prepares publication on the same admitted creator after intent is durable.
Recovery uses the same supplied compute owner and short writer admissions; no
Source capability crosses the preparation lifetime.

## Backup readability

Backup preflight preserves SQLite read failures and cancellation through both
check-only and acquisition operations before a package is published. Its
original connection and statement remain owned through physical settlement;
failed close retains them for retry on the creator. Missing required tiers and
disk-space observations remain separate diagnostics. Opaque pre-migration
backup evidence keeps its declared version admission; readability does not
authorize interpretation of current Source coordinates. The daemon runs this
read-only phase before beginning a backup write attempt. Named infrastructure
failures are retryable; deterministic SQLite refusals remain rejected. The
snapshot repeats its admission under writer ownership, and later failures keep
the runtime's effect-aware handling. Disk-space advisories do not refuse an
actual backup merely because they make a check-only observation unsuccessful.

## Write custody and connection lifetime

Reference preparation supports declared configured tier symlinks by opening
exact resolved leaves with no-follow admission. It retains the configured root
and link incarnations and rechecks those names through writer admission and
promotion. Recreating a link to the same leaf invalidates the old proof.
Generation linking and Source snapshots preserve those same configured names
and selected directory incarnations through their operation.

Physical SHA-256 of a live tier file runs in a child process. The parent binds
an anchored directory and selected device/inode, then the child opens the
regular leaf without following links, streams its digest, and checks the
identity and file metadata before and after reading. The parent rechecks the
directory entry and kills and reaps that child on cancellation. This avoids
closing an ordinary descriptor for the database in the process that holds
SQLite's POSIX locks. Schema census and live migration fingerprint checks use
this owner; ordinary hashing of closed backup artifacts remains local.

Each outer write lease names its archive root and holds an owned, descriptor-anchored
`.archive-write-custody.lock` before writable archive SQL begins. Nested owners
borrow that custody; a thread receives authority through a single-use grant,
and a task cannot acquire authority by inheriting another task's context.
The lock file remains in place after release. Directory and lock identities
are checked before and after acquisition; replacement is a visible refusal.

Specialized tier owners admit the configured archive root before their native
open: bootstrap, durable change trains, inactive tuple embedding writes and
embedding checkpoints retain their existing creation or `mode=rw` policy.
Inactive Index bootstrap names its owning root explicitly; the candidate's
parent directory does not identify that authority. Graph publication, User
lifecycle writes and the blob publication fence use the shared isolated writer
factory. Demo augmentation borrows one cached writer for its complete sequence.

Descriptor cleanup attempts every owned binding once and keeps the actual
primary and cleanup errors. Closing the lock descriptor releases flock; it is
never unlocked before a close that could fail without taking effect. Actual
Linux native close errors retire the descriptor, while an ambiguous substituted
or non-Linux failure retains the exact binding on its creator. A numeric slot
is never blindly closed again. Failed async acquisition keeps its existing
worker, task, context and admission alive through physical settlement; an
explicit request reaches that same custody owner. Successful acquisition binds
the returned custody to the loop task, whose failed cleanup likewise keeps
that task alive until settlement. The async backend retains its original cleanup
Task and attempt Future through last-grant retirement. The runtime requires
aiosqlite 0.22.1 or newer: read connections use its `set_authorizer`, and
physical settlement owns its separate worker thread and `stop()` sentinel.
The Python dependency floor and shared Nix package-set pin enforce that same
contract; older connection/thread implementations are not supported. Later backend
or coordinator settlement requests wake the same custody retry and shield that
attempt; they never start a parallel close or join the owner's application Task.

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

Per-operation `sqlite3` readers use `contextlib.closing` or an explicit `finally`
to end the connection lifetime. SQLite's connection context manager commits or
rolls back but does not close. A reader scope consumes its cursors before close;
lazy rows and iterators remain inside the connection owner's scope.

Archive reads submit their exact UTF-8 request-byte demand to the shared bounded compute owner; interactive and scan work use its interactive-read and bulk-candidate classes. The read controller retains only the connection-weight budget. Compute workers complete their physical Future only after the existing native-owner census settles on the creator thread. Failed close retains the worker, submitter context, reservation or read lease, and backing artifacts; cancellation of an asyncio wrapper does not release them. The existing custody owner can request another cleanup attempt. Reset retains a shared compute owner while its physical workers survive; publishing a distinct owner requires that original owner to settle first. Archive readers and reference seals bind native children to their actual terminal parent, and healthy parents retire after all SQL and artifact obligations settle. Revision projections close their writer before transfer and close each bounded readonly page before yielding immutable rows (`polylogue/core/sql_settlement.py`, `polylogue/storage/sqlite/connection_profile.py`, `polylogue/pipeline/ids.py`).
## Baseline construction and durable trains

`ArchiveStore(root, read_only=True)` opens only existing tiers and never bootstraps.
A missing Index raises `ArchiveTierUnavailableError` without creating the root
or tier files. Explicit `initialize=True` with `read_only=True` raises
`ReadOnlyArchiveError`; the constructor defaults initialization to the writable
mode. `open_existing` uses the same constructor contract.

The immutable fresh archive baseline creates all six tiers at version 1.
`initialize_active_archive_root` records that baseline's pending intent,
format marker and bootstrap receipt before advancing durable tiers through
their declared numbered trains. The `polylogue.archive-format.v6` lineage
folded the predecessor's Source slots 002-006 into the Source v1 baseline, so
Source, User and Audit currently have no numbered migrations and a fresh
bootstrap replays no train. The folded baseline's durable schema inventory is
pinned in `tests/unit/storage/test_fresh_archive_format.py`; a later durable
change is a numbered migration, not an edit of the baseline.

A row-preserving index replacement claim is checked against SQLite's actual
keys, collations, order, uniqueness and complementary literal predicates in
both rehearsal and live execution; exact row values, primary keys, foreign
keys and integrity must survive the owned transaction. Other non-additive
changes require verified backup authority (`storage/sqlite/migration_runner.py`).

### Source identity partitions

`raw_artifacts` carries two complementary unique partitions: ordinary
artifacts are unique per source coordinate, while deferred and terminal
refusals (including missing coordinates, missing profile evidence and
unproved retained ZIP membership) are unique per raw acquisition, so two
refusals at one coordinate stay attached to their distinct raws.

`blob_refs` has no single primary key. Attachment references include their
provider coordinate in the unique key, so identical bytes under two file IDs
retain two references; other reference kinds keep their raw or hook owner key.
Raw attribution still owns liveness and retirement; coordinate identity never
encodes a different raw. Deferred, inline and prepared attachment acquisition
use the same provider file coordinate (or the provider attachment ID when no
file ID exists).

`raw_profile_identity_receipts` records a raw's captured profile identity, and
prepared manifest members and accepted source items carry their captured-input
identity. A raw without a receipt reports the explicit profile gap instead of
discovering a qualifier from current source paths.

Browser-native membership may mark an older raw `superseded_by_winner` only
when one later provider-timestamped snapshot independently dominates it by
preserving its provider message and attachment identities. This terminal
Source decision does not claim an ordering among older snapshots; Index replay
records each as `superseded` by the selected winner. Provider-time ties,
missing identity evidence, or competing candidates do not receive this
decision (`archive/session_revision_membership.py` and
`storage/sqlite/archive_tiers/revision_governance.py`).

New ingest acceptance stages physical inputs pagewise through
`prepare_source_manifest` and carries a sealed reference into the audit plan.
After materialization, the ingest writer settles an accepted item only when its
current generation/item binding and exact raw/blob membership match the pinned
receipt and every raw has complete logical publication evidence. Strict
schema-validation refusal remains pending with typed `validation_rejected` evidence;
it keeps the generation census unsealable and the terminal ingest receipt
degraded. Unresolved or untyped raw evidence cannot settle an item.
Immutable pending commands retain their original inline evidence for restart;
the opened single-ZIP acquisition also retains its one-input manifest. Decoder
completion compares streamed coordinates against the caller's uncommitted
Source membership using regular indexed rows and a disk journal in a private
Native-owned scratch database. It never changes the caller's TEMP policy.
Read evidence uses that same enumeration measurement on the supplied Source
snapshot. Input headers are paged separately from their nested raw witnesses;
the daemon streams those witnesses into its existing private receipt spool and
authenticates each historical raw page without retaining all pages. Private
spool files remain under their exact Native lifetime until every opened owner
settles, including failed reader or writer closes.

Every write that can change a raw's existence evidence (the raw row, its
artifacts, memberships, census rows, payload blob refs, and blob receipts or
GC members keyed by its blob) records the raw key in `raw_existence_changes`
through Source triggers in the same transaction, including external writers.

Writable canonical bootstrap admits installed trains before runtime version
validation. Read-only and acquisition-only opens refuse a durable tier below
the runtime version without applying migrations. A crash after baseline
publication resumes the same persisted train rather than restamping the
baseline as current (`storage/sqlite/archive_tiers/bootstrap.py`;
`storage/sqlite/durable_change_train.py`).
The format marker retains immutable baseline birth versions and fingerprints
after migration. Isolated runtime consumer probes and canonical schema census
replay the baseline and installed numbered SQL as empty schema probes, checking
the same canonical inventory and version authority after replay. Consumer
probes carry the authenticated post-apply candidate's schema/version and refuse
an inventory or version mismatch. They perform no archive migration or backup
exemption; the live train already holds its required package authority.
File-backed probes declare their owned temporary path; populated or attached
connections refuse. Probes never release the train whose consumers they prove.
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
physical bindings and bytes. An earlier installed Source schema can be backed
up as opaque evidence when its required bytes are retained. Acquiring missing
bytes from current raw-source coordinates requires the current Source schema
and otherwise raises the ordinary typed schema refusal before that read. Copied
receipts do not authorize a relocated backup inode
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

Incremental TEXT publication uses `literal_cells.write_literal_text` under the
original native transaction owner. Audit keeps its destination allowlist and
verified canonical chunks; capture jobs retain CAPTURE intent and event JSON,
and exact native preparation metadata and asset outcomes, in their existing
TEXT columns. Both consumers enforce SQLite physical limits and retain failed
Blob settlement with its creator. Capture reads detach
the pinned cells to request-owned lazy JSON scratch before the native snapshot
closes. The registry FULL transaction owns durability; transient cell/read/response
scratch is flushed for its reader without artifact fsync. Retained native
prefix and final artifacts keep publication fsync. Collection cardinality does
not require a decoded Python tree; the
largest individual string token and SQLite cell allocation remain physical
memory contributors.

## Identity and generated columns

- `sessions.session_id` is stored-generated as `origin || ':' || native_id` (`SESSIONS_SPEC` in `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:809-815`).
- `messages.message_id` is stored-generated with explicit namespace tags: native identity becomes `session_id || ':n:' || native_id`; content-derived identity becomes `session_id || ':c:' || content_identity || '.' || content_occurrence` -- a digest of the message's own declared semantic fields, so an insertion elsewhere in the export cannot renumber it onto a different message (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:337-342`).
- A parser's `provider_message_id` names exactly one message. A provider value shared by two distinct records is not a message id on its own: Codex tool records carry only `call_id`, which the call and its output both repeat, so each record's native id is that `call_id` qualified by its declared record side (`::call` / `::output`, and `::mcp-call` / `::mcp-output` for an MCP pair), and the session event from the same record names that qualified id (`_codex_tool_record_message_id` in `polylogue/sources/parsers/codex.py`). Position never qualifies a native id.
- Raw export-member identity (`core/content_identity.py`) preserves every decoded string and mapping key exactly, including operational paths and Unicode spelling. JSON key order and integral numeric spellings remain equivalent. The v3 domain applies to fresh acquisition; it has no predecessor identity fallback.
- Content hashes use canonical JSON: absence stays `null` and empty strings stay `""`. Only declared prose fields in `_NFC_TEXT_FIELDS` are NFC-folded; identifiers, tool arguments, paths, metadata, event payloads and mapping keys stay exact. Content-derived message IDs use one recursive encoder that preserves ordinary pre-a44 null, empty-string and empty-block preimages. Literal reserved values and lossy typed/key lowering receive a framed typed projection, while a reserved-looking mapping key alone does not trigger an escape. An occurrence ordinal distinguishes repeated identical messages without depending on their position (`polylogue/pipeline/ids.py`).
- Message row digests and block citation evidence frame optional text and JSON with an absent part for SQL NULL and an `=` prefix for every present value, including the empty string. Parsed writes, block coalescing, copied lineage rows and stored-row rehashing use the same framing. Citation hashes continue to exclude session/message identity, position and tool ID. The omitted `semantic_extra_json` column expands to its canonical empty extras object before hashing; metadata null and an empty object remain distinct inside that JSON. Row digests do not determine embedding freshness, which uses `vector_derivation_hash` over the embedder's actual input (`polylogue/storage/embeddings/identity.py`).
- Full parsed-session replacement explicitly removes attachment-native-ID dependents before their attachment refs, so bulk rebuilds preserve foreign-key integrity while enforcement is disabled. The delete is scoped to the replaced session's refs and leaves other sessions' identities intact (`_clear_session_projection_rows` in `polylogue/storage/sqlite/archive_tiers/write.py`).
- `messages.content_address` witnesses the complete declared message hash projection, including block metadata, file edits and web constructs. A parent replacement may retain a branch edge only where that complete witness agrees; a same-ID message with changed content cannot silently become the child's inherited prefix (`_message_content_address` in `polylogue/storage/sqlite/archive_tiers/write.py`).
- `messages.identity_source` records which identity path fired; its index CHECK is generated from the semantic `MessageIdentitySource` Literal (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:360-365`; `polylogue/core/types.py:13-16`).
- Id-less appends continue above the greatest stored occurrence for each content digest. Materialized-prefix replay preserves existing ordinals, including gaps left by removed tail messages; an append never renumbers those anchors (`_stored_content_occurrences` in `polylogue/storage/sqlite/archive_tiers/write.py`).
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
- Every producer declares its actual destination and transaction owner. Active archive writes use a durable-reference seal; an owned inactive generation defers the archive-wide proof to promotion. A genuine non-archive memory index declares its standalone transaction explicitly. Missing archive arguments or missing durable tiers do not grant standalone permission.
- Manual continuation stores User authority first as a nondeleted HANDOFF assertion targeting `session:<child>`, with `value_json={"_schema":"polylogue.manual-continuation.v1","parent_session_id":"<exact parent session ID>"}`. The manual product and canonical parsed-session writer derive its `continuation` / `spawned-fresh` Index edge through the same projection before graph resolution. Replacement, append and retained generation rebuild consult the scope-owned User reader; deleted assertions remove their projection, absent parents remain unresolved, and missing or malformed required authority refuses. Prose-only handoffs have no topology meaning.
- Assertion listing and CLI assertion export require the operation's attached `user_tier` and query it through the pinned Index connection. Missing, unreadable or foreign User authority is a typed refusal; a present empty tier is a valid empty result. These operations never reopen the User pathname or attach replacement evidence after pinning, and a refused CLI export leaves its destination unchanged.
- Bulk callers reuse one `IndexMutationScope` per commit window. Its disk-backed witness stores original typed lookups, re-resolves their targets before publication, and includes composed descendants affected by parent changes. Inserts, aliases and lineage changes also receive proof: unchanged message IDs alone do not establish preserved lookup or block-position semantics. Generation replacement checks the whole candidate. User/Audit JSON anchor enumeration remains global and unindexed; batching amortizes that census rather than making it independent of archive size (`polylogue/storage/sqlite/reference_seal.py`).
- Audit preview and operation targets retain historical effect identity. The validated delete, identity-reset or excision apply may leave those exact targets absent within its authorized removal closure; it never changes their rows or digests. Every User anchor (tags, metadata, annotations, deliveries, suppression and excision records) may likewise describe an absent session: user.db is durable, the delete is the operator's explicit action, and those rows resolve again when the same source identity is re-imported. A reference may dangle only when every session that owns or scopes it is an authorized, now-absent target (a stored canonical `session:<id>` is judged by that exact row, never by public-token prefix resolution onto a surviving session such as a subagent child); one owned or scoped by a surviving session (for example a child composing a removed parent's prefix) still refuses, as does any disappearance outside an authorized plan. Exact bound excision may remove content-bearing assertion rows whose own targets lie in its declared excised session/message/block closure; surviving rows that reference the excised session keep those references unchanged. Lifecycle history remains retained. This permission is bound to the existing archive custody, creator task/thread and validated plan during apply. Retained reparse/replay and generation promotion retain normal reference proof (`polylogue/storage/sqlite/reference_seal.py`, `polylogue/operations/mutation_transaction.py`).
- Authoritative retained replay keeps identical session and message rows when the semantic hash, provider aliases, parser/lowering fingerprints and canonical prefix representation match the original prepared write. It still settles Source governance and provenance, and applies the existing link and graph updater for current hook evidence. Changed prefix representation or fingerprints require lowering. Work-event raws insert their new events independently of transcript equality; duplicate event IDs retain their original idempotent result.
- It is the parsed-session lowering choke point used by retained Raw publication and authoritative revision replay/reindex (`write_with_reparse_receipt` in `polylogue/storage/sqlite/archive_tiers/revision_governance.py`). The acquired Raw owner prepares and admits batches before this writer runs. It is not the only mutation function in the six-tier substrate.

## Blob publication, liveness, and GC

- Blob paths are SHA-256-addressed as `<root>/<first-two-hex>/<remaining-hex>` (`polylogue/storage/blob_store.py:193-197`).
- Every preparation route hashes while writing a private staging file and fsyncs its bytes before publication; publication fsyncs the shard directory after an atomic `os.replace` (`polylogue/storage/blob_store.py:207-237`; `polylogue/storage/blob_store.py:245-274`; `polylogue/storage/blob_store.py:282-295`; `polylogue/storage/blob_store.py:303-316`). Preparation, publication deduplication and error cleanup admit the same owned private staging root or child directories; symlinks, foreign directories, parent traversal and escaping companion paths refuse before unlink.
- Archive publication commits durable reservation receipts before exposing final paths; the exact receipt is consumed in the durable-reference transaction (`BlobPublicationReservationStore.reserve_many` in `polylogue/storage/blob_publication.py:125`; `ArchiveBlobPublisher.flush` in the same file at `302-334`; `consume_blob_publication_receipt` at `553-564`).
- Liveness is descriptor-owned. Ordinary `blob_refs.ref_type` values must map unambiguously to one referent relation (`polylogue/storage/blob_liveness.py:90-113`).
- A destructive liveness check returns `LIVE`, `UNREFERENCED`, or typed `BLOCKED`; unavailable or unreadable required tiers block deletion (`polylogue/storage/blob_liveness.py:250-291`).
- GC safety requires no live DB reference, no publication reservation, a final locked recheck across control/source/index, an age floor, and bounded deletion batches (`polylogue/storage/blob_gc.py:7-25`).
- A refusal is not a quiet pass. Every preflight and final-recheck refusal sets `report.blocked_reason` and emits one `storage.blob_gc.refused` event carrying `outcome="refused"` and the phase, so a GC that refused can never be read as a GC that ran and reclaimed nothing (`polylogue/storage/blob_gc.py:1383-1385`; `polylogue/storage/blob_gc.py:796-800`; `polylogue/storage/blob_gc.py:78-104`). Callers that collapse the report into counts must read `blocked_reason` before believing a zero (`polylogue/storage/blob_gc.py:1013-1016`).
- Refusals inside the locked execution window also emit `storage.blob_gc.refused`; `_execute_gc_generation_members` records the reason and returns the deletions already committed in that batch (`polylogue/storage/blob_gc.py:789-800`; `polylogue/storage/blob_gc.py:816-825`; `polylogue/storage/blob_gc.py:883-895`). Trust the report for the partial counts and the event for the refusal phase.

### Two-phase `gc_generations`

1. Commit one generation and every exact member intent as `pending` before any unlink (`gc_generation_members` in `polylogue/storage/sqlite/archive_tiers/source.py:706-721`; `polylogue/storage/blob_gc.py:525-569`).
2. Under `BEGIN IMMEDIATE` on source and index, recheck liveness/reservations, unlink or reconcile each member, commit outcomes, then finalize only when no pending members remain (`polylogue/storage/blob_gc.py:735-900`; `polylogue/storage/blob_gc.py:589-620`).

Pending generations are restartable; a restart resumes their exact member set instead of rediscovering intent from the filesystem, and refuses an intent whose blob namespace was swapped or remounted (`polylogue/storage/blob_gc.py:622-638`; `polylogue/storage/blob_gc.py:640-664`; `polylogue/storage/blob_gc.py:916-951`).

## Lineage storage model

The exported synchronous topology adapter discovers both children and outbound links for every fetched node, including ancestors found after the initial target. Its visited queue terminates cycles and includes ancestor siblings and their descendants; the shared topology composition engine retains edge classification and deterministic breadth-first output.

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
- Membership preparation stores accepted session output and attachment claims on the canonical sealed PreparedJsonl artifact. Publication queues the same publisher's closed-page claims before the Source transaction, then reads attachment values and references from that artifact. A private captured copy can survive collection of an unreferenced public blob; losing or changing the sealed private capture refuses publication (`polylogue/sources/prepared_jsonl.py`).
- GC history counters are summaries derived only after all member outcomes close; member rows are the crash-recovery authority (`gc_generation_members` in `polylogue/storage/sqlite/archive_tiers/source.py:706-721`; `polylogue/storage/blob_gc.py:571-588`).
- A retained agent work event (`append_work_event`) is its own logical source: its `agent-work-event:` raw id is also its logical key and source path, admitted as a byte-proven singleton baseline, so it never joins the byte-revision cohort or accepted head of the transcript it annotates (`write_work_event_raw_and_parsed_result` in `polylogue/storage/sqlite/archive_tiers/revision_governance.py`). Its write is event-only and keeps the session's `raw_id` and `content_hash`. A cold build replays work-event keys after every byte and membership cohort, so the transcript's fresh write never meets a session an event created (`apply_prepared_revision_replay` in `polylogue/sources/revision_backfill.py`). Excision seeds work-event raws by the session's `(origin, native_id)`, and source conservation counts one as materialized when its session is indexed.
- Rebuildable `index.db` must not become authority for an irreversible durable mutation; blob GC therefore requires source-ledger and active-index checks to agree (`polylogue/storage/blob_gc.py:7-20`; `polylogue/storage/blob_liveness.py:321-359`).

## DISCREPANCIES

- The `docs/architecture.md` ring diagram draws only source, index, embeddings, user, and ops; code has six tiers and includes `audit.db` (`docs/architecture.md:25`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:49-85`).
- Embeddings are purchased state, require backup, and cannot be replayed from Source. `excision_embedding_completions` certifies the selected paid transaction under its original Source-command binding; it neither authorizes the command nor substitutes for current surviving-reference checks (`polylogue/storage/sqlite/archive_tiers/embeddings.py`; `polylogue/storage/sqlite/reference_seal.py`).

Existing index and ops tiers are admitted by their derived schema identity
before initialization issues any DDL. A mismatch raises typed `SchemaSkew`;
initialization never patches or restamps a foreign derived identity. An
admitted INDEX tier installs only canonical runtime performance indexes before
manifest validation, following its writable sync and async policy; read-only
opens still refuse a missing manifest index without writing. At daemon
startup, exclusive archive ownership and a write lease allow replacement of a
stale disposable ops file before persistent tier handles open. A stale managed
Index, including its anchored regular bootstrap before the first promotion,
with canonical physical DDL may instead be replaced at startup only after
proving all Source custody and parsed/material Index tables
empty, physical blob custody empty, and durable User/Audit reference preservation.
The existing generation owner creates and atomically promotes fresh empty DDL
before tier handles open; the predecessor remains recoverable under normal
promotion and retention, including its normal WAL checkpoint. Durable and
purchased tier schemas and bindings must remain supported and current.
A populated managed Index with canonical physical DDL and a stale fingerprint
is reconstructed at startup from retained Source, before ordinary preflight.
The existing Raw owner performs bounded canonical census, classification and
replay into an owned inactive generation; external originals are unnecessary.
The acquisition snapshot binds raw identities, captured coordinates, verified
payload bytes, blob claims and capture observations to the current Index recipe.
Origin and revision interpretation may refine only through those original
Source phases. ColdBuild retains its separate full revision-authority digest.
Readiness, full replay completion, current Source/User/Audit observers and
previously resolving purchased message references must all pass before the
normal reference-checked promotion. No durable or purchased tier is replaced.
Interrupted MEMORY candidates are discarded through their owner and rebuilt;
a published successor completes its existing promotion tail on restart.
Promotion and restart do not derive durable parse success from Index receipts.
Unacknowledged successful session components remain eligible for ordinary retained
replay, which prepares current evidence and publishes its original Source permit
after the Index outcome. This can require an additional preparation pass after
a cold build; terminal, deferred and non-session dispositions keep their existing
inspection behavior. Unsupported physical DDL, durable schemas,
changed custody or missing reference coverage remain explicit refusals.

## Tool-result association

`core/tool_association.py` lowers the original occurrence and resolved-parent
facts on the caller's existing SQL creator. Eager outcome preparation,
materialized and canonical action reads, append reconciliation and observed
`tool_finished` events share that decision. A unique native parent chain
through tool replies proves the exact invocation, including multiple replies
and replies exported before the call. Traversal terminates cycles with `UNION`;
it imposes no depth limit. Original globally unique message keys permit an
inherited reply intermediary; invocation assignment remains session-scoped.
Replies naming the same native parent with different declared sibling ordinals
are alternatives, not sequential fanout. Every physical reply remains stored.

Without exact parent proof, a clean alternating stream supports sequential
ID reuse. A missing middle result, duplicate receipt or overlapping same-ID
invocation leaves that invocation and the unproved same-ID suffix unknown,
with `outcome_unknown_reason=ambiguous_tool_id_reuse`. Variant creation order
is not causal evidence. A proved invocation is resolved independently of an
unrelated ambiguous window. A final resultless invocation remains `no_result`;
absence of a paired reply does not retract independent execution-sidecar
facts on the original tool-use block.

A proved fanout aggregates error before unknown before success. One invocation
event carries the call and every exact reply reference. Scalar result location,
output, error flag and exit code remain absent for multiple replies rather
than selecting a representative. Tool episodes expose the omission in their
caveat and the existing authority gap `tool_episode_plural_output_omitted`.
A known aggregate verdict does not certify a complete scalar output. Writers
resolve original parent links before reconciling outcomes and refreshing
associations; reads do not depend on insertion order or a populated cache.

Facade settings and context-delivery reads distinguish an absent row in a
readable tier from unavailable authority. Required user state, source events
for Hermes delivery correlation, and the ops injection ledger refuse with
`ArchiveTierUnavailableError` when their tier cannot be inspected. Read calls
never create missing tiers. Named-source freshness reads current cursor progress
only from ops `ingest_cursor`; a missing offset remains unknown, and successful
empty canonical state cannot inherit progress from a retired index table.

## Embedding contract transitions

Acquisition checks all retained producer contracts in the active Embeddings
membership before a provider call; publication checks again under the same
generation's writer lock. Declared compatible recipes retain their actual
producer identities. Incompatible output contracts raise
`EmbeddingContractTransitionRequiredError` before purchasing or writing a
window. Unknown producer provenance refuses independently. A separately
produced candidate must contain the current excision completion relation before
being validated and explicitly promoted through
`EmbeddingGenerationStore.replace`; this admission does not automatically
build or switch generations (`storage/embeddings/generations.py`;
`storage/embeddings/materialization.py`; `storage/embeddings/derivation.py`).

Embedding status measures coverage from Index and the canonical Embeddings tier.
Its separately guarded Ops history reader returns nullable catchup history when
the disposable tier is missing or unreadable; that absence does not abort
otherwise measurable coverage. Supplied pinned connections keep their original
attached snapshot authority. Owned diagnostic readers close even when an
Embeddings attachment refuses (`storage/embeddings/status_payload.py`).

Ordinary full FTS rebuilds clear text and identity residue together and stream
session pages through the existing paired SQL projections. Progress counts
settled sessions, using the exact session total; an empty terminal event follows
both resets. Page publication retains the caller transaction. Async observers
run on their event loop with one settled handoff at a time; observer errors or
cancellation settle the physical worker before returning and emit no later
progress (`storage/fts/fts_lifecycle.py`; `pipeline/services/indexing.py`). The
separately owned resumable bulk generation retains its existing chunk commits.

### Provider usage projection

`storage/usage.py` owns the event fold, provider-inclusive to disjoint token lanes, and catalog pricing decisions. Full writes and append windows stream stored token-count events through that owner. The latest session-global cumulative replaces earlier deltas; append windows retain existing replacement or increment behavior. Unknown models are attributed only when the writer has exactly one measured session model; unresolved events stay retained without a guessed model.

Provider total-only or reasoning-only counters cannot establish the priced input/output/cache lanes. Their cost remains unknown through append and usage-rollup derivation. The same derived `session_model_usage` row carries `provider_lanes_complete`, independently of nullable catalog price. Append composes that bit with the new event window; reads use the same SQLite snapshot and expose incompleteness without rescanning events. An empty model declaration has no unmappable provider evidence, but remains unknown usage rather than a measured zero. Catalog repricing precedes the provider fold during derivation so it cannot erase the refusal. Explicit zero lanes remain measured zero.

Event projection and stored catalog costs retain precision; the origin/model rollup rounds only after summing. Session buckets are replaced directly for cumulative observations without rescanning other sessions. Memory follows the distinct session/model result cardinality, with stored source events streamed. The caller owns the transaction; projection performs no commits. This additive-derived column moves the Index identity and must land before reconvergence; there is no durable migration or live rebuild in this delivery.

## Retained schema coverage

Artifact inspection measures large JSON documents through the same semantic
observations as decoded registry resolution. Its private SQLite spill keeps
input keys and nested schema state on disk; structural hashes stream all
required evidence, including canonical and shipped-order source witnesses.
Eligibility requires complete document validation through EOF. The diagnostic
prefix does not decide support. Memory remains subject to the existing decoder's
one-scalar allocation and SQLite's physical value-length bound.

Source conservation checks retained raw CAS presence independently of original
source availability. A present original with absent retained bytes is blocking
`missing_blob`; an absent original and absent retained bytes is `source_lost`.
An absent original with retained bytes remains nonblocking `source_missing`.
These checks inspect CAS existence, while retained-byte validation owns full
body fidelity.
