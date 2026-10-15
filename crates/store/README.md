# Foundation store

The daemon acquires `Writer::open` and holds it until shutdown. The canonical
archive's exclusive file lock rejects competing writers and cold readers,
including aliases. Writer-created read handles retain that authority until
their connections close; they use the active SQLite view without scanning a WAL
being appended. Cold read handles acquire the existing lock without creating or
rewriting it. Callers receive typed handles, never raw SQLite connections.
`put_file_blob` is a foundation byte-storage diagnostic, not provider ingestion,
revisions, parsing or M1 support.

All three schemas are exact SQL-fence extractions from the owning snapshot;
a drift test compares their text. Archive version 1 adds the immutable import
identity tuple; embeddings version remains zero and index version is 18. No
pre-release durable migration is supplied. Existing schemas and versions are checked before database mutations. A newer durable file is
refused unchanged; an index mismatch returns `IndexRebuildRequired`, preserving
it. Retained state without its coordination lock requires controlled recovery;
raw headers may explain a newer-version refusal without opening SQLite.

Before any cold SQLite open, the lock holder checks durable WAL headers,
current frame salts, page numbers and rolling checksums. Memory is bounded by
one page (at most 64 KiB); total IO scales with WAL length. Cancellation is
checked before and between frames. Valid complete uncommitted frames are
allowed. A bad header/frame, partial frame, orphan WAL or uncertain generation
tail returns `RecoveryRequired` before SQLite can discard or checkpoint bytes.
Refusal and scan cancellation preserve retained files. After successful
validation, SQLite may update SHM coordination state during inspection.

This intermediate guard may refuse legitimate torn uncommitted tails and stale
frames from `RESTART` reuse. It needs controlled recovery and does not qualify
full startup availability. Aligned truncation to an earlier valid commit,
older consistent main/WAL pairs, valid-prefix salt/header/history changes and
checksum collisions cannot be excluded without an independent expected head.
There is no external commit ledger. Runtime live-WAL corruption and complete
recovery remain unqualified.

Bundled SQLite 3.53.2 is checked at build time and runtime. Its matching
checksum VFS has one documented local fault callback. Main-database checksum
failures retain `SQLITE_IOERR_DATA` and report the VFS's actual file, page,
offset and page size through operation-scoped thread-local evidence. The VFS
does not wrap WAL reads; the cold guard above is a separate boundary. Index
uses the underlying Unix VFS and is rebuildable.

Fresh archive directories get No_COW on btrfs before children exist. Directories
use 0700 and owned files 0600. Synthetic tests cover actual writes, read-only
mutation denial, range/page corruption, killed-process retained-WAL faults,
legitimate/ambiguous tails, cancellation, lock lifetimes, schema refusal and
privacy. Compile-fail examples deny writer methods and raw connections. These
checks do not qualify coupled cross-file views, ingestion, excision, index
rebuild/swap, full recovery or M0 completion.

Disposable parser state uses typed Writer admission and read-only pinned
snapshots, with no connection or SQL escape. Small state deltas, supplied output and
checkpoint share the final step transaction; the store primitive does not infer
canonical completion from EOF. A per-index monotonic allocator survives head
and input-key removal. Snapshot, stale-result, bounded retirement and actual
child-process crash tests qualify this component, including the real archive
writer lock. The Codex owner uses this provider-state protocol through public import/progress
requests. Canonical message finalization remains unfinished.

Larger provider deltas can be staged in admitted encoded-cell batches, sealed,
and applied in bounded cell/byte steps. New parser snapshots wait while this
work exists; already pinned readers retain the old SQLite view. Interrupted
unsealed preparation can be discarded in bounded steps. Sealed application
resumes after process exit and publishes its head/counts/checkpoint only after
all staged members have been consumed. This protocol changes no canonical
Message, block or search rows. The eventual canonical writer must couple its
message-local selection to final publication. Whole-cell encoding, decoding and
SQLite page work remain explicitly unqualified for arbitrarily large cells.


The forward index SQL is extracted from the pinned owning specification revision
`INDEX_SPEC_COMMIT` and checked against `INDEX_SQL_BLAKE3`, copied Markdown, and
actual SQLite application. Its Message head/body/directory schema replaces the
old mutable Message columns. The inbox index8 executable comparison remains
separate and cannot qualify this forward representation. These schema checks do
not establish canonical publication, public-client restart or giant-value/search
resource acceptance.

Canonical normalized payloads now use `values`: at most64KiB per immutable
append/range read, UTF8 validation and bounded ordinary-BLAKE3 proof steps outside
write transactions. Process restart keeps appended bytes and restarts hashing at
zero; sealing performs fixed metadata/digest work. JSON typed encoding remains
its producer contract. No complete payload is hidden in a final readiness row.

Message directory primitives copy a fixed radix16 path and retain unchanged
children through counted references. Collection drains one at-most16-edge node
per transaction, with pinned readers retaining their historical WAL snapshot.
`ReadHandle::selected_message` pins one SQLite snapshot; block ordinals traverse
at most32 fixed directory nodes and text reads return admitted value ranges.
Unselected preparation is invisible. Current hidden/removal epochs revoke old
snapshot reads; ordinary replacement preserves the pinned incarnation. This read
primitive does not register a
public consumer or qualify election and ready-body retirement.

Pending Message preparation is separate from body selection. `prepare_message`
reserves a committed, basis-pinned body/candidate without making it eligible or
public. A build owns one active normalized value while bytes are appended and
verified; sealing deduplicates and transfers that producer lease atomically.
`write_message_field` transfers it into one source field fact and copies one
fixed-depth directory path. Source changes, removal epochs and archive/index
identity mismatches reject further preparation. Invalidation releases its active
value lease even after the source basis changes. Unpublished invalidated builds drain with `retire_message_turn`: one frontier/root,
one fixed-size directory node, one fact/source row, or final control metadata per
turn. `collect_retiring_value` drains one chunk or final value row; building and
ready unleased generic values are excluded. Candidate reconciliation, body
activation and public-client/whole-path resource acceptance
remain open.

`begin_block`/`write_block_field`/`install_block` stage Text, Thinking, ToolCall
and ToolResult through the same group/local directories; `write_block` is the
atomic convenience over those helpers. A call block owns one private summary
with an opaque caller-proved operation identity. Tool names lease the same ready
JSON field value; `write_tool_command` transfers an independently prepared UTF-8
producer without decoding JSON in the writer. No prepared summary enters serving
FTS. A result retains its own typed outcome facts without invented pairing.
Producer leases, directory roots and the projection cursor commit together.
Old unleased paths release known leaves and each counted payload lease during
bounded node collection. Attached ToolItem versions and external serving owners
retain summaries until their owning cleanup releases its known reference;
this module does not implement global operation pairing or ToolItem cleanup. Field and no-output projection units use
the same cursor CAS. `finish_message_projection` checks fixed roots and records a
source producer's completed projection/evidence metadata. Body and candidate
remain unready and ineligible, so interrupted folding still uses bounded abort.
Restart recovers that projection through `prepare_message`. Source semantic
coverage and the exact evidence proof remain required caller prerequisites.

`retire_message_body` checks selection, active-build ownership and an indexed
eligible-candidate liveness probe. One turn detaches an ineligible candidate's
payload pointer, or deletes one unselected body and releases its fixed roots.
Immutable rank/digest certificates and shared canonical/frontier leaves survive.
Known leaf release reclaims facts/text only after directory and source-body
ownership ends; `collect_message_candidate` drains one locator or its unused
certificate. These APIs are being qualified independently of canonical election
and public consumers. They do not implement privacy/excision closure or giant
search retirement.

Occupied field-conflict leaves pin a shared immutable witness subtree. They
remain distinct from missing fields, contribute no new source-rank facts, and
retain exact alternatives independently of mutable input representatives.
Directory lookup/copy/collection uses the same leases and fixed-depth paths;
no per-body alternative list is copied on an unrelated field change. The pure
model compares canonical typed values one witness at a time. Serving conflict
views, export/import and full canonical activation are still unwired.


Immutable inputs retain their original byte digest and explicit format-owned
interpretation context. `InputId::imported` implements the pinned version-1
framing from spec191adbec; physical CAS remains named by raw byte digest.
Registration/recovery validates that tuple. The primitive accepts explicit
Claude contexts; the public actor accepts self-identifying Codex rollouts with
empty context and explicit Claude declarations. It does not choose a Claude pathname from competing
observations. Any existing excision refuses acquisition until the refusal-aware
collector is wired.

`BlobCapture` holds archive lock authority without SQLite capability. It streams
and verifies bytes on a worker before opaque proof admission writes metadata.
`client_request` and the import observation commit together; an unchanged retry
replays the retained result before opening its provider path. Archive commit
precedes repairable disposable `input_ref` registration. Capture cancellation is
cooperative around decoder reads; seek tables, record/cell allocations and physical
IO work remain outside a complete memory/latency qualification.

Text and thinking preparation can hold one private block while its normalized
field values stream. Each field changes one local field-directory path and the
same source cursor; installation transfers the complete block to its prepared
body directory atomically. Restart retains the active block, and abort releases
its text/field leases before bounded collection. Block-local semantic keys omit
the public ordinal; the mapper proves membership and the selected block root
owns correspondence. Thinking text and encryption may remain unknown. These
primitives do not certify source completeness or select a canonical body.

`select_uncontested_source_message` activates one certified source body only
when an indexed probe finds no competing eligible input candidate. Selection,
index counter and session change marker commit together. A competing candidate
keeps the prior selected body intact and leaves merge work pending. This first
Codex path does not implement the full chain/input fold, fill, conflicts,
operation summaries or search publication.

`message_reference` resolves an opaque Message ID through its stored Session
relation and distinguishes absent from not-yet-selected bodies. `block_at`
uses the same fixed-depth order-statistic traversal as ordinal block reads,
including interleaved separate reasoning IDs. The terminal client streams
selected UTF-8 ranges; it reports degraded coverage rather than treating this
consumer as complete source accounting or export support.

Index 15 pins exact candidate source-span sets. Abort and historical candidate
collection remove one full `(input, chain, start, end)` key per turn, with native
failure rollback. Provider record citations are separate from field selectors
and lazy payload extents. Historical field-origin descriptors and their second
producer lease are the next owning schema/runtime seam, still unimplemented.

Field facts may retain a separate chunked JSON origin descriptor. Preparation owns independent normalized-value and origin-value leases, including two counts when deduplication aliases them. Fact publication transfers both atomically; `FieldView::origin_range` reads the descriptor through the same selected snapshot. Integrity-ready JSON is only a carrier: source selector/transform validation, pinned-span membership and export remain separate unfinished obligations.

Source certificates retain the exact parser generation, step and basis alongside
the preparation digest. Reparse can withdraw only a verified strictly older
generation of the same input/chain/archive revision/logical candidate; a missing
same-generation receipt or inconsistent certificate refuses publication. This
works after old parser cells are collected. Source replacement leaves the selected
body unchanged until explicit selection, and pinned readers retain their old view.
There is no automatic index migration or general rebuild/swap implementation.

The existing parser-cell owner retains a source-row settlement receipt after the
engine audits dispositions/effects and selected source certificates. Publication
binds the exact head, complete-record frontier and input removal state in one
fixed metadata transaction. This producer-supplied semantic proof is distinct
from arbitrary row counts and from Session/query/search completeness. Missing or
corrupt authority refuses; no second progress ledger or automatic migration is
introduced.
