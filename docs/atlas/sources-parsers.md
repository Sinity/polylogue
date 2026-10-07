# Sources and Parsers

## Area boundary

Sources acquire bytes and identify their material source. Detection chooses a
provider parser by input shape; the pipeline normalizes provider records into
parsed sessions before the storage writer lowers them
(`polylogue/sources/dispatch.py:1-80`; `CompiledDetectorRegistry.detect` in `polylogue/sources/detection.py:88-105`;
retained preparation in `polylogue/sources/revision_backfill.py`, driven by
`RawObservationDerivation` in `polylogue/operations/raw_observation_derivation.py`).

Document arrays use the same tightness-ordered document predicates for every
member, including streamed detection. An unrelated fragment cannot claim a
later complete document. Lowering streams all accepted documents and refuses
complete documents from a different origin; browser envelopes retain each
member's own declared provider. Gemini checkpoint logs keep their ordered fold,
and Hermes ATOF records keep their grouped reducer. Bundle specs do not retain
the complete decoded cohort.

Retained Codex state material preparation records its complete Source statements
under the original preparation parent before the writer accepts them.
Interruption during preparation accepts no material rows; replay publishes the
complete material set without duplicates. Row and text pages bound work while
preserving all content.

## Source observation and SQLite reads

`source_snapshot.py` publishes a declared root's complete member inventory or
an unavailable result. Byte members and candidate copies read an anchored,
no-follow descriptor matching the enumerated inode; captured append prefixes
keep that descriptor's size, hash and identity together. A spool handoff binds
the new active generation at its creation and observes that exact generation
for carry-forward arrivals.

`sqlite_export.py` owns canonical logical exports, SQLite shape reads and
existing staged-import backups and live Antigravity/Hermes import previews.
Each runs in a fresh reader process. The
parent anchors source-directory metadata and forwards the existing no-follow
walk parent for directory members. The reader uses that descriptor as its
working directory, opens SQLite and sidecars by relative URI, and proves the actual
main descriptor before SQL, binds WAL/SHM descriptors after the first schema
read and rechecks every source descriptor before closing the transaction.
Initially absent sidecars bind only when their opened descriptor matches the
current anchored name. Unknown or unlinked regular descriptors are refused;
no database guard descriptor is opened and closed in the caller's process.

`source_staging.py` owns `SourceInputBinding`, which keeps staged provenance,
declaration, logical-table scope and routing with the accepted main identity.
A narrow fresh-process operation reads provenance before opening SQLite;
the parent never reads an ordinary metadata descriptor that could have been
substituted with a database held by another reader. Present unreadable,
invalid or mismatched provenance refuses acquisition. Only absence permits
the ordinary source route. The actual reader proves the same main descriptor
before SQL and checks the bound metadata before the operation and at its end.
Acquisition results carry the accepted source coordinate through attribution
and profile identity; retained exports use that durable coordinate directly.

The physical canonical coordinate identifies the accepted open input. The
semantic coordinate keeps its captured declared parent and basename, so an
alias named `state.db` retains its declaration even when its target has a
different name. Hermes acquisition separately captures the resolved declared
profile namespace and its shared qualifier before retention. New raw identities
use the captured namespace/member under the v3 domain; retained replay reads
`raw_profile_identity_receipts` and never resolves a current filesystem alias.
An older raw without a receipt reports `terminal_missing_profile_identity`,
distinct from missing physical byte coordinates. Captured source-manifest
members retain physical, semantic and profile evidence together.

ZIP member publication and reacquisition require the captured container/member
receipt and its exact ordinal, split index and addressing mode. Backup,
restoration, integrity, debt and conservation readers never infer that namespace
from a source-path suffix or a Raw ID. Loose filenames containing colons remain
literal paths; a recorded member without its receipt is an explicit refusal.

Staging publishes provenance and database through separate replacements.
The provenance includes the backup owner's actual destination identity, so
readers refuse the intermediate mismatch and a failed second replacement.
A retry publishes a newly proved pair. Explicit stable root aliases resolve
once to the accepted actual root; aliases inside enumerated directories are
refused.

Exports retain their canonical bytes and declared logical-table scope. Pipe
frames stream to the existing sink, with each callback acknowledged before the
reader advances. A failed callback or final binding check leaves an unfinished
operation: the blob writer discards its private staging file, digest callers
raise, and a staged backup is not published. Transport memory is bounded by
chunks; the existing canonical emitter still allocates an individual row and
its encoded cells. No whole-export transport buffer or input limit is added.

Ordinary byte acquisition lends one isolated reader to the caller's existing
bounded input page. Each sequential request transfers its original source and
metadata directory capabilities over a private ancillary channel. The child
opens and proves that request's no-follow source, streams bytes with the same
per-chunk ACK, then closes its file and received directory descriptors before
request completion. It retains no byte or identity cache between requests.
The page returns its frozen input tuple only after final process completion,
EOF and actual reap; any request fault or cancellation kills and reaps that
exact child before pipes, sockets or scratch retire. Empty pages allocate no
reader. Standalone capture and preflight use the same owner for one input;
logical SQLite export and its native custody stay in their separate fresh
process. A retained ZIP allocates its disposition spool only on its first
actual refusal or unselected member, keeping their original ordinal and
completion laws.

The descriptor census uses `/proc/self/fd` where present and otherwise scans
the finite OS descriptor bound. Every regular reader descriptor belongs to
an explicitly bound source role or backup destination. A VFS that opens an
additional regular lock file needs that file's identity bound before the read
can complete. Native Darwin VFS/proxy-lock qualification has not been run;
no unknown descriptor is exempted as a guessed platform lock file.

Import explain and SQLite preflight detect and parse on the same proved
connection and read transaction. Named domain operations return counts,
session references and fidelity evidence only after the final binding proof;
they do not transport another transcript representation. Native live reads
preserve the source's collation, affinity, views and rowid semantics. Retained
logical exports retain ordinary and readable VIRTUAL/STORED generated column
values, evaluated in the same acquisition snapshot with exact SQLite storage
classes. Hidden virtual-table implementation columns remain excluded. Their
private untyped reconstruction stores those acquired values; original DDL is
evidence and generated expressions are never replayed. Reconstruction streams
from the accepted export descriptor. SQLite preflight aggregates every
trajectory through the production positive-conversational evidence gate;
empty/degraded evidence remains a caveat, and a prefix cannot hide a later
admitted session. Hermes verification reads every event/state row without a
consumer row-count refusal. Its preview validates each row through the parser's
existing transforms and aggregates fidelity counters, spilling Python session
keys into a private BINARY-collated grouping database. It does not instantiate
the ledger's parsed-event list. The public parser still returns its declared
session list; the preview transports only its declared references and counts.
Explicit low-level connection-return readers retain their existing semantics;
this guarantee covers the actual acquisition and import-preview operations.

## Detection and parse route

1. Acquisition records raw bytes and source metadata in `source.db`.
2. Detector bindings are declared by each `OriginSpec`, not by `dispatch.py`.
   `compile_detector_registry` validates them and sorts them per
   `DetectionMode` by `(mode_rank or detector_tightness, local_rank,
   binding_id)`; `CompiledDetectorRegistry.detect` returns the first predicate
   that claims the payload (`polylogue/sources/detection.py:88-105`;
   `polylogue/sources/detection.py:196-227`).
3. The selected provider parser emits normalized sessions, messages, blocks,
   tool uses, tool results, and lineage hints.
4. `write_parsed_session_to_archive` computes public origin and identities,
   writes the parsed tree, and resolves asserted parent links
   (`write_parsed_session_to_archive` in
   `polylogue/storage/sqlite/archive_tiers/write.py:2119`).
5. The daemon converger materializes FTS, embeddings, and insight read models.

## Detector tightness order

Lower number runs first. Tightness must be unique among executable
`OriginSpec`s, which is enforced at spec validation
(`OriginSpecRegistry.diagnostics` in
`polylogue/sources/origin_specs.py:1452-1462`). Current executable order:

| Tightness | Origin |
| --- | --- |
| 10 | `gemini-cli-session` |
| 20 | `hermes-session` |
| 30 | `antigravity-session` |
| 50 | `codex-session` |
| 60 | `claude-code-session` |
| 70 | `chatgpt-export` |
| 80 | `claude-ai-export` |
| 82 | `claude-design-session` |
| 85 | `grok-export` |
| 90 | `aistudio-drive` |
| 95 | `otel-genai` |

`beads-issue` (reserved) and `unknown-export` (compatibility-only) carry no
tightness and are not executable detectors. A new detector inserted looser
than it deserves lets an earlier parser claim its records; a binding may also
declare `mode_rank` to take a different precedence in one payload mode than
its origin-wide tightness (`polylogue/sources/detection.py:36-42`).

Complete-stream detector projections live in `sources/detection_projection.py`.
Each detector declares its selected fields and first/all/any folds through its
existing parser and registry. Unselected material is still consumed and syntax
validated; duplicate mapping keys keep their final value. Artifact candidacy
uses the taxonomy's separate declared projection and complete record fold.
Diagnostic schema samples cannot choose a provider, discard a late session, or
prove support for uninspected records. Canonical parsing validates the original
full records and retains their exact decode/partial disposition.

## Identity vocabulary

`Provider` names the older provider-wire family at acquisition/parser/schema
boundaries. `Origin` is the public source-origin token and the only coordinate
public filters accept. `Source` carries richer acquisition identity. The
mapping is non-injective -- `Provider.GEMINI` and `Provider.DRIVE` both map to
`Origin.AISTUDIO_DRIVE` -- so an origin must never be reversed into a guessed
provider (`docs/provider-origin-identity.md:15-30`;
`docs/provider-origin-identity.md:108-115`).

## Invariants

- Native Codex browser-capture envelopes retain the original record array and
  delegate to the ordinary Codex parser before merging envelope attachments;
  the extension does not provide a Codex page adapter.
  Native attachment turns lacking provider IDs use explicit retained ordinals
  with matching native role/text to produce private owner coordinates. The
  parser refuses absent or conflicting evidence instead of inventing an ID.

- Grok native and export human/user sender records establish human authorship,
  including prose that resembles runtime instructions. Textual message type
  stays independently classified; native tool and reasoning blocks retain
  their structural classification.
- Detection is shape-based and ordered by declared tightness, per payload mode.
- Parsing preserves structured tool-result outcome and exit-code fields;
  prose is not an outcome oracle.
- Parser inference cannot overwrite a hook-authoritative lineage edge: when
  `_authoritative_parent_claim` returns a hook-asserted parent, the write
  uses that parent for lineage resolution. A hook parent with no parser parent
  promotes the session to `SessionKind.SUBAGENT`
  (`_prepared_message_context` in
  `polylogue/storage/sqlite/archive_tiers/write.py:1655-1698`).
- Replaying identical normalized content is idempotent by content hash;
  user metadata does not alter import identity.
- All ordinary ingest, replay, and reindex paths share the parsed-session
  write choke point (`write_parsed_session_to_archive` in
  `polylogue/storage/sqlite/archive_tiers/write.py:2119`).
- Full live ingestion acquires durable Raw inputs before the supplied resident
  Raw owner prepares them. First ingestion and retained replay share
  `RawObservationConvergenceOwner` and the original prepared Source and Index
  witnesses. Preparation reads the retained membership, precedence, blob and
  parser evidence before short admitted publication; the matching writer
  consumes attachment reservations with the same prepared receipt. Physical
  cleanup stays with that preparation creator through publication and failure.
- Raw preparation and publication are owned by `RawObservationConvergenceOwner`;
  it validates retained membership and precedence before publication and
  consumes attachment reservations using the prepared receipt. The parsed
  session writer remains the shared lowering choke point for ordinary ingest,
  replay and reindex.

## Gotchas

Provider fixtures are not interchangeable with live captures: acquisition
identity and parser shape can differ. Check the provider completeness matrix
before claiming a mode is supported (`polylogue/sources/provider_completeness.py:1-100`).
When adding a detector, add a real-shaped fixture, precedence coverage, and
replay-parity verification. When changing normalized fields, inspect every
lowering reader and the corresponding schema checks.

## Navigation

Start with `polylogue/sources/origin_specs.py` for the declared detector
bindings and tightness, then `polylogue/sources/dispatch.py` for payload
lowering, then the provider parser and its fixture. Follow the parsed object
into `polylogue/storage/sqlite/archive_tiers/write.py`; do not infer the
durable contract from a surface serializer. The provider guides under
`docs/providers/` explain format-specific caveats.

## Settled decode and partial admission

Live intake and retained census share `terminal_decode_evidence`: a known-provider
JSON document or complete JSONL record that cannot decode settles as
`terminal_corrupt_input`. Unknown-provider decode failures retain their distinct
terminal token. Wrapped source I/O failures and parser defects remain retryable.
Changed source bytes form a new observation rather than retrying the same refusal.

A stable JSONL capture with an unfinished final record admits its complete prefix.
`PartialAdmission` records `truncated_tail`, the complete-record count, the prefix
byte offset and the full acquired size. Acquisition seals the count before writer
entry. Intake, batch byte accounting and dispatcher events expose the partial;
the attempt carries `batch:partial_admission` and its event is degraded. No-session
and settled corrupt observations remain excluded. The literal raw retains the
unfinished tail so a later completed observation can advance normally.

Grok native tool results declare `not_reported` when outcome evidence is absent
and `unsupported_construct` when a supplied outcome has an unsupported shape.
Both reasons remain typed unknown outcomes through storage and readback.
