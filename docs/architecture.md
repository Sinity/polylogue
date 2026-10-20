# Architecture

This describes the current Rust module boundaries. The specification and Beads
also contain intended behavior that is not implemented or qualified yet.

## Inputs and stored records

`polylogue-sources` interprets provider formats. An origin declaration owns
recognition, paths, framing, parser version and provider capabilities. Opening
an input selects line or document parsing; callers do not maintain a second
framing flag. Parsers produce model values, record ranges and accounting. The
shared `Records` sink owns disposition and wire-field counts. Line parsing
supplies an independent framed total; document parsers submit the items they
discover. Finishing either path checks the same accounting invariant.

`polylogue-engine` discovers changes, retains input bytes and coordinates
parsing. Existing input directories are watched recursively. A missing location
keeps a nonrecursive watch on its nearest existing ancestor; directory creation
advances that watch before collection, so a new source appears without waiting
for the periodic fallback. Read-access events do not trigger collection. Its
worker pool feeds one bounded writer. Source interpretation stays
in sources; SQLite access stays in store. Rebuilding a derived index must not
replace the retained observations with a claim that a provider file is still
available.

Small spool revisions, hook settings and reader credentials are replaced using
`atomicwrites`, which owns temporary-file cleanup, file synchronization,
rename and directory synchronization. Each caller still owns its path and
permissions; hook installation refuses managed symlinks.

Browser captures use the provider's existing ChatGPT or Claude parser; Grok
endpoint bundles adapt native responses and attachment metadata to the shared
message model. `POST /v1/browser-captures/inspect` accepts the final capture
envelope and returns its summary and attachment descriptors without a write.
Acquired files travel by value in `session.attachments[].content_base64`;
Claude inline media stays in the raw payload and is linked to retained blobs.
`POST /v1/browser-captures` acknowledges the durable spool write. The spool
currently replaces one conversation file, so intermediate captures acknowledged
before convergence can still be overwritten.

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
record carried between jobs. Document-framed origins still read and parse a
whole document. Reimporting a retained input retries missing or outdated parser
progress without adding another revision or copy of its bytes. Restart replay
selects the same sequential reader for a chain retained as one complete file
blob, and obtains route precedence from its durable revision class.

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
before removing its temporary archive. This does not establish cleanup after
forcibly killing a parent test process, or a bound on an idle daemon's byte
reads; those remain separate work in `plnext-xni.27`.

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

## Changing a boundary

Keep a rule with its owner and update connected callers together. Add a
provider through the origin and parser framework, a command through the
command table, and a measurement through its declaration. Tests exercise the
behavior through these interfaces. Follow [development.md](development.md) for
focused checks and batch regression runs.
