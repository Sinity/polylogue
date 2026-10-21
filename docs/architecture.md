# Architecture

This describes the current Rust module boundaries. The specification and Beads
also contain intended behavior that is not implemented or qualified yet.

## Inputs and stored records

`polylogue-sources` interprets source formats. An origin declaration owns
recognition, paths, framing, parser version and tool capabilities. Opening
an input selects line, export or document parsing; callers do not maintain a second
framing flag. Parsers produce model values, record ranges and accounting. The
shared `Records` sink owns disposition and wire-field counts. Line parsing
supplies an independent framed total; document parsers submit the items they
discover. Finishing either path checks the same accounting invariant.

Each message-source occurrence records its parser version. Within the same
input, revision chain and capture, a newer parser may replace an older
interpretation even if the logical position moves later. Same-version
duplicates keep their earliest occurrence; older-parser replay cannot overwrite
newer coordinates or content. Parser numbers from different inputs do not
compete, and newer revision/capture scopes keep their precedence. Promoting a
remaining source retains that source's parser version.

A completed newer parse retires the previous parser's contributions to that
revision chain, including omitted messages, events, links and session metadata.
Completion requires source EOF and accounting through the selected length; a
fully consumed captured prefix is insufficient. Until then, old contributions
remain available alongside the new rows. Session facts merge within each
input/chain/parser interpretation, then across inputs; obsolete interpretations
cannot supply omitted fields after completion. The writer recomputes affected
source selection and hook-derived live state. A disposable SQLite file holds
affected IDs with a 1 MiB cache and mmap disabled. Derived session-row removal
is shared with excision; it does not change raw bytes, authored rows or embeddings.

`polylogue-engine` discovers changes, retains input bytes and coordinates
parsing. Existing input directories are watched recursively. A missing location
keeps a nonrecursive watch on its nearest existing ancestor; directory creation
advances that watch before collection, so a new source appears without waiting
for the periodic fallback. Read-access events do not trigger collection. Its
worker pool feeds one bounded writer. Source interpretation stays
in sources; SQLite access stays in store. Rebuilding a derived index must not
replace the retained observations with a claim that a provider file is still
available.

Hook settings and reader credentials are replaced using
`atomicwrites`, which owns temporary-file cleanup, file synchronization,
rename and directory synchronization. Each caller still owns its path and
permissions; hook installation refuses managed symlinks.

Browser captures use the provider's existing ChatGPT or Claude parser; Grok
endpoint bundles adapt native responses and attachment metadata to the shared
message model. `POST /v1/browser-captures/inspect` accepts the final capture
envelope and returns its summary and attachment descriptors without a write.
Acquired files travel by value in `session.attachments[].content_base64`;
Claude inline media stays in the raw payload and is linked to retained blobs.
`POST /v1/browser-captures` acknowledges the durable spool write. Each
conversation has one length/digest framed append log; the host syncs its
new envelope before acknowledgement. Exact envelope bytes identify retries: an
older retry is a no-op; a new observation, including a reversion, carries new
observation provenance. Existing single-envelope spools convert on the next
distinct capture. An interrupted append can complete only when every existing
tail byte matches the submitted frame prefix; it appends only the missing
suffix. Constant-only prefixes contain no observation identity. Differing or
otherwise ambiguous tails are preserved and refuse appends.
Appending scans headers for identity without rereading other payloads; the
parser verifies every interpreted payload. Text excision recomputes active
digests while preserving observed digests and source offsets.

Each complete envelope is interpreted once into an ordered source snapshot.
The rebuildable index retains normalized message versions with exact source
ranges, sharing prose in the text store. Ordinary reads, queries and
derivations use current projections. Only explicit complete-session coverage
can retire absent members; current browser acquisitions supply partial
coverage, including native payloads. Session facts merge with earlier supplied
facts; complete population coverage does not clear omitted metadata. Historical
chains replay through the same parser and writer, with chain and snapshot
range ordering preserving the latest projection.

Before selected parse jobs emit rows, the engine supplies their required
format versions to the writer. The index retains those requirements and each
chain's observed EOF, source length and last fully retired parser version.
Replacing one input can update matching records immediately, but obsolete
session-owned records remain until that session's contributing inputs complete
their required parses. A shared input's sessions retire independently; links
wait for their existing endpoints. Pending work survives writer restart and is
shown as `reparsing`. Ready sessions and affected identities are deduplicated
in operation-scoped SQLite, without collecting their bodies.

An event's message anchor participates in its identity. The event table keeps
its shared facts; `event_source` owns each observation's input, revision,
snapshot, parser version, order and byte range. Current reads select a source
by stable input ID, then its newest revision/snapshot. They do not keep the
first arrival's coordinates. Source retirement promotes another observation,
and excision consults the retained observation rows. Repeated identical records
within one observation scope keep their latest order/range; this is not an
occurrence list. A parser replacement resets the order for that scope.
Active-branch selection chooses each input's newest marker by revision, snapshot and record
order. Across inputs, known event time takes precedence and content identity
resolves ties. Database row IDs do not describe observation order.

`GET /v1/reader-location` resolves the running reader's published port and
sign-in fragment after checking the listener owner. The CLI uses the same
lookup. A reader that is unavailable returns an error; this lookup does not
start one.

## Storage and queries

`polylogue-store` owns the archive, index, embeddings and blob access. One
writer serializes archive changes; readers use read transactions. Original
observations and authored values differ from derived indexes. Session reads,
derivation and exports share one row mapping, including repository and mode.
The daemon admits at most `Config.workers` client Readers, with one separate
Reader reserved for convergence so waiting clients cannot starve the loop.
Each Reader owns one index connection with the archive attached and may open
one auxiliary embeddings connection. Daemon reads therefore open at most
`2 * workers + 1` connections; the engine does not open embeddings.

The default `Config.reader_memory_bytes` is a 128 MiB page-cache target for
these readers together. Each Reader gets an equal share, split between the
main index, attached archive and optional embeddings caches (rounded down to
KiB); all three have mmap disabled. A lifetime-bound vector guard retains at
most one auxiliary connection per Reader, including embedding builds. Upgrading
that connection closes the old one before opening another. The engine leaves
its optional embeddings share unused. Workers must be positive and the budget
must provide at least 1 KiB per database per slot. This is SQLite's suggested
page-cache target, not a hard process memory limit: query work space,
statement caches and returned values still use memory. It imposes no row or
content-size refusal. Short-lived setup and maintenance connections retain
their separate 16 MiB setup allocation with mmap disabled.

A full read pool waits for a released slot; opening connections also count
against the cap. Open, BEGIN and transaction-completion failures release slots.
A Reader is thread-affine: acquire, use and drop it within one thread. HTTP
handlers acquire it inside their blocking application worker, outside async
network work. Nested readers on the same thread may use spare slots for snapshot comparisons;
a checkout that would wait while that thread already holds a slot refuses with
`ReadPoolNested`. Callers reuse or drop their reader before retrying. Closing a
pool wakes every waiter, refuses checkouts still opening, releases idle
connections and lets already admitted transactions finish.

Within store, `encoding.rs` owns packed row bodies and block-text separation;
`text_rows.rs` owns text deduplication, pending search entries and removal.
Fixtures use the same text-storage path as ingestion.

Message ordering uses the existing opaque input ID and the record's wire offset.
The index keeps that ID in `message.order_input_id` beside its local input
foreign key. Source selection and every order consumer compare the stable ID,
so parallel arrival and index rebuilds cannot change the message order or a
title derived from its first prompt. Preferred message versions keep their
existing source, completeness, route and content-hash rules.

`store/derive.rs` orders session derivations and records the source-change counter
and rule version for each. A consumer's version includes its complete declared
dependency rules, so an upstream rule change also refreshes unchanged sessions.
Missing, cyclic or incorrectly ordered declarations fail before any rows change. Message costs use the same path, with the complete
catalog digest and its published label in their version. Changing prices does
not invalidate unrelated derivations. The work graph and its orphan cleanup
live under `derive/work.rs`. Provider-reported costs remain recorded facts.
The index schema version and SQL digest are pinned together in a test.

`polylogue-query` binds the query language and compiles parameterized SQL.
`daemon/query.rs` executes it through a store reader, computes the final answer
count and prepares pages. Recall uses this path too. Saved-query and kept-result
storage failures propagate as errors; an unavailable collection is not an
empty collection. Cursors retain the query identity and resolved time; they do
not establish a snapshot that survives between requests.

## Imported bytes

`engine/import.rs` acquires each file or ZIP member through the existing blob
store's streaming plain writer. The original BLAKE3 names the input and its
single retained blob. Before registering the complete revision, acquisition
checks the same open source descriptor's generation and size; ZIP members also
check their declared uncompressed length. An interrupted acquisition publishes
no input or source observation. A completed blob can be orphaned if registration
fails.

Parsing reads that retained representation, so later source replacement or
deletion does not change its identity or parsed content. The verified sequential
plain reader decodes each frame once after one verification pass. Line-framed
inputs and unparsed members use the normal parse pool's part-sized jobs without
retaining duplicate part blobs. One input is imported at a time. Acquisition
uses the blob store's default 4 MiB frames and a 64-byte recognition prefix;
parser buffers scale with the configured chunk and the largest incomplete
record carried between jobs. Other document-framed origins still read and
parse a whole document. Its messages cite that retained document's full byte
range; the writer takes their reading order from the emitted sequence. A
logical message ordinal is never a byte offset. Framed captures cite the
corresponding payload's absolute range. Reimporting a retained input retries missing or outdated parser
progress without adding another revision or copy of its bytes. Restart replay
selects the same sequential reader for a chain retained as one complete file
blob, and obtains route precedence from its durable revision class.

ChatGPT and Claude export arrays use Serde's array visitor and `RawValue` to
frame one conversation at a time. The reader records and verifies the exact
lexeme against the source bytes before accepting its range. The whole outer
array, including its trailing suffix, must validate before any conversation is
published. A private operation-scoped scratch object compresses the elements
into an anonymous temporary file and indexes their ranges and ordering in a
temporary SQLite file (1 MiB cache, no mmap). ChatGPT conversations retain their
global oldest-first ordering and first-owner message deduplication through this
same object; Claude preserves physical array order. Scratch disappears when the
operation ends and is rebuilt from retained bytes on retry.

AI Studio uses the same scratch and commit path for `chunkedPrompt.chunks`.
A Serde object visitor retains the surrounding fields with an empty chunks
array, including metadata appearing after the chunks. After the full input
validates, the parser emits prompt settings and the system message, then one
chunk at a time. Its state is the model name, previous message ID, chunk ordinal
and whether a start time was found. Chunk messages cite exact original chunk
spans; the system message and settings cite the enclosing prompt. Duplicate
fields on the selected array path refuse the input before publication, rather
than combining two ambiguous chunk populations. The Drive listing uses the
same visitor for `files`, emitting session facts one entry at a time. A recorded
Drive title outranks the separate `file_name` fallback from a loose import,
regardless of arrival order. Unknown fields remain accounted and raw bytes
retained.

OpenAI-compatible chat files use the same reader for a bare message array or
an object containing `messages`. The complete wrapper is checked before its
ID, title and model initialize the parser, so fields after the array apply to
earlier messages too. State keeps that session identity/model, the previous
message ID and the raw message ordinal. Unknown/refused records still occupy
their original ordinal; tool replies keep their harness meaning. Each emitted
message cites its original JSON span. Header metadata and one message/output
remain the memory bound, with the same retained-byte retry rules.

Gemini CLI rewrites use that named `messages` reader too. The parser retains
session ID and previous message ID after reading header facts; tool calls and
results stay within their turn. Notices and errors remain events anchored to
the preceding message, with their own original record spans. Watched revisions
are retained before parsing, and retries and rebuilds use those bytes after the
source file disappears. Physical order is the default; ChatGPT uses its recorded conversation time.

OpenRouter uses `sources/export/keyed.rs` for its named message map and
`store/export_items.rs` for disk ordering. Records sort by optional integer
milliseconds, optional message ID and object key; unknown times sort first.
Repeated object keys replace the staged value, its position and its source
range. The selected first record names the room after those replacements,
so an overwritten old value cannot supply its identity. The complete wrapper
validates before header or message output; metadata after messages still applies.
Only ordering fields decode during the scan; message bodies remain raw until
their turn. Room metadata and one message/output remain memory bounds.

OpenTelemetry GenAI exports use that retained export path, with repeated
`resourceSpans/scopeSpans/spans` arrays visited by Serde. One span and its decoded
attributes/output are held at a time. The private SQLite table sorts exact
optional unsigned nanoseconds, span IDs and physical ties; its ownership table
keeps first-seen conversation metadata and message positions. Each event and
message cites its original span rather than an invented wrapper or the whole
file.

A whole JSON document validates before publication. If its syntax is invalid,
the scanner rewinds the same retained view and accounts independent JSONL lines.
Each staged group uses a SQLite savepoint and a compressed-file checkpoint;
a malformed line discards its entire prefix before later lines are scanned.
Input and callback errors abort instead of falling back. The private index uses
a disk rollback journal with durability disabled; journal mode OFF cannot roll
back staged rows. Rewinding reopens retained bytes, never the provider file.
Temporary disk grows with staged spans and ownership; one unusually large span
still needs its own memory. This synchronous scan is not a reusable parse turn.

Gemini Takeout uses the same staging and writer path for original HTML activity
cells. Its known `outer-cell` delimiter frames one cell plus a fixed read buffer;
large headers are discarded while retaining only a possible delimiter suffix.
This is the existing Takeout shape, not a general HTML parser. Integer timestamp
keys retain chronological order and physical ties, with unknown dates first.
The private SQLite state holds each conversation's previous message, so links
survive interleaved activities without a whole-export map in memory. JSON
origins decode their original staged bytes as JSON; HTML records are never
wrapped in invented JSON source bytes. First-record metadata uses that same
byte boundary. The small in-memory state adapter lives with the shared export
code and serves individual captures and parser fixtures.

The existing live-before-bulk and size ordering applies to work already planned.
Preceding ordinary work drains through the parse pool and commits before the
serial export driver runs with the pool inactive. A live append arriving during
that synchronous driver waits until a subsequent discovery pass; this is not
preemption or a latency bound.

Polylogue's own export uses the same reader for `sessions`, keeping their
physical order and authored identities. Every session is checked against the
typed export schema during the scan; an invalid later session refuses the whole
document before any prefix is published. Header format and version, including
headers after sessions, apply before publication. Its typed wrapper rejects
duplicate enclosing fields; value-based source wrappers retain their existing
rules. Messages and events cite the original exported-session span, replacing
ranges that belonged to the export's former input. Memory still includes one
whole exported session and its output, rather than all sessions in the file.

The driver waits for each item's writer commit before decoding the next.
Message order is a separate logical ordinal. Repeated event facts retain their
latest logical source order, including across replay. Memory scales with the
largest conversation or chunk, its parser output and retained prompt metadata;
a large individual item or non-chunk metadata still needs that much memory.
Temporary disk scales with the export. Parser progress becomes complete only
after every item commits. An interrupted retry rescans the retained input,
resets accounting and idempotently replays committed items; it does not resume
from a durable mid-export scanner cursor. Parsers receive a first-message-owner
callback for export deduplication; the engine supplies the temporary SQLite
lookup, while source interpretation stays in sources.

## Application and transports

`daemon/service.rs` owns application state and typed request handling, including
starting the convergence loop. `daemon/server.rs` owns Unix socket framing and
listeners. HTTP and MCP adapters translate requests and present results; they
use the same application operations. Signal evaluation and embedding providers
have their own modules. This is an internal boundary in the application crate,
not another service or crate.

Production startup resolves optional backing from operator configuration and
passes it explicitly to `service::start`. In-process fixtures select
`Backing::Standalone`; changing a fixture's source home does not select its
backing. The daemon subprocess fixture uses its own configuration directory.

`service::Runtime` owns the convergence and signal workers, the optional
Sinex publisher and the archive lock. Socket and HTTP servers own their listeners and client work and stop
before the application runtime. Application shutdown first marks request
admission stopped and closes the client read pool, waking checkouts before
transport or signal-worker joins. It then stops workers before releasing the
writer lock. Independent transport shutdown still permits transport restart. Unfinished
socket input or response writes fail when peers close; already admitted store
writes finish before the writer joins. Retained writer-sender handles refuse
new work after shutdown. Shutdown closes the read pool and releases its idle
connections even when an external `Arc<Daemon>` survives. A reader already
admitted keeps its transaction until its caller drops it.

The in-process `TestDaemon` follows that order on ordinary drop and panic,
before removing its temporary archive. On Linux, test daemon subprocesses also
receive a kernel parent-death signal, with a parent-PID check closing the startup
race. Source read events do not wake discovery, and unchanged incomplete JSONL
tails wait for source growth rather than rereading retained bytes. The subprocess
tests cover forced parent death, archive-lock release and idle input reads.

`commands.rs` declares command arguments, capabilities, printers and local
output actions. CLI, MCP and HTTP build their surfaces from that table.
`daemon/measure.rs` declares measurements and their supported metrics once.

## Presentation and backing storage

`polylogue-render` owns session view projection and output selection. A view
selects typed blocks without replacing message metadata or tool fields. Prose
formats display those blocks as Pieces; structured JSON reads retain
`{session, messages}`. JSON export retains the model's importable Document,
including events and links, rather than converting it to prose Pieces. The
same renderer owns the format and view names. HTML exports and the web reader
share the content renderer; framing, navigation and transport remain with
callers. HTML output returns write errors.

The public application response boundary masks decoded JSON strings before
transport escaping, including nested metadata, error text and object keys.
Ordinary IDs, field names, types and empty values stay unchanged. If credential
keys collapse to the same masked label, numbered suffixes keep every value and
preserve ordinary keys. Blocks are also masked before model-page sizing, so a
page split cannot expose fragments of a credential. Structured export preserves
ordinary recorded fields; credential masking intentionally survives reimport.

Socket reads (and their CLI/MCP clients), the HTTP read API and the HTML session
route receive the finalized application response. Shared transport-error
construction masks validation failures too. Session headers also mask titles
in the HTML helpers. The browser-action backend has separate response paths;
this boundary does not describe those control responses.

Standalone storage keeps Polylogue's archive locally. The optional Sinex
adapter supplies primary backing; its publication and restore qualification
have separate acceptance criteria. Neither mode provides daemonless CLI
access. Shared blob storage, credential masking and development tooling live in
`polylogue-sinex-common`; projects pin immutable releases.

The optional Sinex publisher stops at an async suspension point, including
its periodic wait. Shutdown joins its thread before releasing the archive
lock. A pending publish is interrupted rather than reported successful:
frames lacking a committed acknowledgment have an unknown remote outcome.
The next pass asks Sinex which stable frame keys it holds before resending.
Synchronous archive and blob operations finish before cancellation is polled;
this does not impose a hard shutdown deadline or qualify primary mode.

Restore orders confirmations and folds later replacements in an operation-scoped
SQLite file. It checks exact repeats, conflicting positions, gaps and unit order
before opening payloads. The same scratch database holds excision's input and
machine membership sets. Its page-cache target is 1 MiB with mmap disabled;
this is not a hard process memory cap. The catalog is discarded after the
operation, and checkpoint serialization preserves the previous ordered JSON
representation.

The selected catalog enumerates distinct current source-blob references on disk.
Restore verifies each once before installing rows; later restores verify them
again. Retired references are excluded only by the selected replacements.
Row batches retain their per-unit identity, row-count and integrity checks.

Selected row batches use the blob store's sequential plain reader. Restore
copies one batch to an anonymous temporary file while checking its content hash
and length, then validates and applies one JSON row at a time. A bad later row,
read failure or unavailable scratch space rolls back the whole target
transaction. Repeated restores still verify payloads. Temporary disk scales
with catalog metadata and the largest selected batch. Memory includes the
largest descriptor, JSON row and decoded values, plus SQLite and blob-reader
buffers and frame metadata; it does not hold every unit or batch row at once.
A very large individual row or excision manifest still needs that much memory.

Source-blob publication and excision share `blobs/copy.rs` for decoding and
whole-content verification. Inline bodies use SQLite incremental BLOB reads;
file bodies use the existing sequential plain reader. Publication keeps the
verified bytes in one anonymous temporary file before placing them through the
checked blob writer. Corrupt source bytes reach no destination blob, and a
corrupt existing destination refuses placement while preserving the source.
Placement queries and checks acknowledgment groups of at most 500 blob
references before requesting the next group, releasing each group's frame names
and reply map. A refusal stops later groups; retry asks about every reference
again. The caller's unit/manifest reference list is still materialized.
Memory holds copy and codec buffers plus frame metadata. Staging disk grows
with one source blob, alongside the destination writer's temporary encoded
file. This path handles plain catalog entries; encoded and sealed
publication need their own readers.

`sinex_rows/batch.rs` writes row batches to buffered anonymous JSONL files.
It holds one row's decoded columns at a time and preserves their SQL values.
The typed row envelope and sorted column names give the same bytes with or
without transport features. Stored replacement batches retain their original
bytes and hashes. Placement returns the material ID and hash from one write,
so constructing a row event does not compress the same batch twice.

A publication pass keeps one SQLite read transaction across its inventory,
replacement and row queries. Later source commits appear in the next pass;
holding this view can retain WAL pages until the pass ends. Physical blob
retirement can still make referenced files unavailable, which verified reads
refuse. `sinex_rows/inventory.rs` keeps ordered unit identities, current
replacement descriptors and acknowledged positions in operation-scoped SQLite
with a 1 MiB page cache. The adapter discards each Have reply after at most
500 keys and visits missing units by position. Named replacement blob references
are validated before placement and converted to binary hashes in groups of 500,
using the same acknowledgment path. Historical manifests are checked
one at a time before payloads open. Temporary disk scales with the catalog and
largest row batch; one manifest, row and its blob-reference list remain memory
bounds. The canonical archive writer uses file-backed temporary tables with
a 1 MiB temporary page-cache target. Excision streams changed unit keys from
that table and uses `sinex_rows/excision/retirement.rs` to select historical
batches on disk.
A batch remains protected while any current replacement references it, including
an unaffected unit. Manifest-owner order selects the latest replacement; source
scan order does not matter. The new manifest's replacement and retirement lists
still grow with that operation. `sinex_rows/excision/digest.rs` writes their
sorted digest encoding directly into BLAKE3 through a fixed buffer, without
building another JSON tree or a serialized manifest buffer. Existing digest
bytes are unchanged.

Excision records retain exact unit identities and content-addressed replacement
batches or tombstones. Restore validates and folds all confirmed replacement
metadata before opening payloads, then replaces the selected owner rows in one
transaction. A nonempty target belonging to a different archive refuses before
payload reads or changes. Same-archive updates must include every local excision's
exact metadata; stale confirmed views refuse before reopening removed content.
Before replacing target rows, restore snapshots hashes of its existing removal
rules into the private catalog. It compares the final rules before committing,
so an incompatible view rolls back policy rows, other selected rows, the local
revision and the restore checkpoint. The hash definition is shared with index
recovery; mutable explanations and local progress do not alter the rule.
Pending batches remain in this archive's primary blob
root, and local file cleanup never follows external read roots. Cataloged source
blobs survive pending-batch retirement. Original provider files are untouched. Native Sinex material
retirement remains a separate operation; publishing a logical replacement does
not prove physical removal. Publication handles non-prefix acknowledgments;
an interrupted pending suffix has an explicit gap and restore refuses it until
the missing units arrive. Legacy excisions lacking replacement metadata refuse
primary publication and restore rather than reconstructing deleted identities by guesswork.

## Changing a boundary

Keep a rule with its owner and update connected callers together. Add a
provider through the origin and parser framework, a command through the
command table, and a measurement through its declaration. Tests exercise the
behavior through these interfaces. Follow [development.md](development.md) for
focused checks and batch regression runs.
