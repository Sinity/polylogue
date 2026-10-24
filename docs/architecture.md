# Architecture

Polylogue retains source inputs and authored records in a local archive, derives
searchable conversations and documents, and serves them through one daemon.
This guide describes the current Rust implementation. [README.md](../README.md)
describes its commands; Beads and the target design contain intended behavior
that may still be unimplemented or unqualified.

## Module owners and interfaces

The ingestion path is `Sources → Engine → Store`; clients use application
operations, which read through Store and present through Query and Render.

| Owner | Responsibility and interface |
|---|---|
| [Model](../crates/model/src/lib.rs) | I/O-free identities, messages, blocks, documents and outcomes shared by the other crates. |
| [Sources](../crates/sources/src/lib.rs) | Recognize and interpret native formats. An [origin declaration](../crates/sources/src/origin.rs) owns discovery, framing, parser version and tool capabilities. [Parse](../crates/sources/src/parse.rs) and [Records](../crates/sources/src/records.rs) produce model values, source ranges and accounting. |
| [Engine](../crates/engine/src/lib.rs) | Discover changes, acquire retained bytes, schedule parsing and drive convergence. The [parse pool](../crates/engine/src/pool.rs) feeds Store's bounded writer; it does not own SQL or source interpretation. |
| [Store](../crates/store/src/lib.rs) | Own every SQLite connection, archive mutation, blob reference, read snapshot, derivation and excision. Its typed `Writer`, `ReadPool` and `Reader` interfaces isolate callers from database layout. |
| [Query](../crates/query/src/lib.rs) | Bind the query language and compile parameterized SQL. The application's [query operation](../crates/polylogue/src/daemon/query.rs) executes it through a Store reader and prepares answers. |
| [Render](../crates/render/src/lib.rs) | Project typed content into views and produce terminal text, Markdown, HTML and structured output. It owns content presentation; callers own transport and surrounding navigation. |
| [Application](../crates/polylogue/src/daemon/service.rs) | Own typed requests, configuration, convergence and worker lifetime. The [command table](../crates/polylogue/src/commands.rs) declares arguments, capabilities and output actions for CLI, MCP and HTTP. The [socket server](../crates/polylogue/src/daemon/server.rs) owns framing and listeners. |
| [Sinex adapter](../crates/sinex/src/lib.rs) | Optional primary backing: publish canonical archive units and retained bytes, and restore confirmed selections through Store. |
| [Testkit](../crates/testkit/src/lib.rs) | Synthetic archives built through the real ingestion loop. Store-private row fixtures remain behind its [fixture feature](../crates/store/src/fixture.rs). |

Add a source through the origin/parser framework, a command through the command
table, and a measurement through its [declaration](../crates/polylogue/src/daemon/measure.rs).
Crate dependency checks live in [xtask](../xtask/src/main.rs); development and
verification commands belong in [development.md](development.md).

## Durable state and derived state

[Archive](../crates/store/src/archive.rs) owns the directory, writer lock and
startup reconciliation. One writer serializes mutations. Databases have distinct
recovery and preservation rules:

| Store | Contents and recovery boundary |
|---|---|
| [`archive.db`](../crates/store/src/schema/archive.sql) | Canonical input identities, revisions, extents and observations; authored notes, settings, variants, kept results and deliveries; standing excision rules and extracted-blob ownership. Additive [archive migrations](../crates/store/src/schema.rs) preserve this state and reject unsupported newer formats. |
| [`index.db`](../crates/store/src/schema/index.sql) | Parsed sessions, messages, documents, source associations and retained content versions; search and derived products. It rebuilds from canonical retained inputs when its format changes or recovery requires replacement. |
| [`embeddings.db`](../crates/store/src/schema/embeddings.sql) | Purchased vectors, provider failures and archive-bound deletion receipts. Additive migrations and exact deletion policy reconciliation preserve unrelated or subsequently repurchased vectors. This database is separate from the disposable index. |
| [`blobs/`](../crates/store/src/blobs.rs) | Content-addressed retained bytes and published replacement batches. Canonical ownership determines retention; a filename's presence alone does not establish a complete admitted input. |

Index rebuilds use retained source versions, including immutable manual imports
whose provider files have disappeared. Parsed content versions are derived;
authored targets, recorded hashes and anchors remain canonical. Message ordering
uses stable source identities and positions, with opaque message IDs breaking
ties. It does not invent event times. [Version selection](../crates/store/src/writer/tests/parser_versions.rs)
and [import replay](../crates/engine/src/import/tests/replay.rs) tests cover those
boundaries.

Derived products declare dependency and rule versions in
[derive](../crates/store/src/derive.rs). Changed rules refresh their dependents;
idle derivation does not announce a new view. Search publication changes
availability without claiming that the underlying session was edited.

An unfinished index build has a durably synchronized marker and is rebuilt on
reopen. Index replacement and an operating-system boot change replace the public
generation; same-boot restarts preserve it. These mechanisms belong to
[archive generation](../crates/store/src/archive/generation.rs) and
[index retirement](../crates/store/src/archive/retired.rs), including validation
that previously applied removal rules are still present.

## Acquisition, interpretation and provenance

[Import](../crates/engine/src/import.rs) and watched collection retain the bytes
that parsing will read. The [writer's retention operation](../crates/store/src/writer/retention.rs)
checks standing refusals, durably reserves ownership before installing file
blobs, then admits verified installed bytes. Rejected first admissions discard
their private stage. An interruption leaves an exact unfinished reservation:
readers refuse its ranges, status reports `unfinished_inputs`, and recovery
requires those exact bytes. A changed provider file cannot silently replace
them. [Custody tests](../crates/engine/src/import/tests/custody.rs) cover imports,
watched inputs and parse-pool groups.

[Extracted blobs](../crates/store/src/retention/extracted.rs), such as decoded
inline media, cite a retained input/chain, exact enclosing source range and hash,
and parser version. They have independent durable owners and publication units.
Editing or excising source bytes invalidates affected associations; surviving
owners retain shared bytes. Recovery does not sweep unregistered historical CAS
files. [Extracted ownership tests](../crates/store/src/sinex_rows/tests/extracted.rs)
cover publication and restore.

[Export attachments](attachments.md) resolve through acquisition/member facts in
retained observations and the [derived asset lookup](../crates/store/src/assets.rs).
Resolution uses the admitted current message and byte basis; descriptors and
masked display labels do not rewrite source attachments. Exact byte delivery is
an interactive CLI operation with verified temporary-file installation. ZIP
import retains original and member bytes with canonical observation relationships;
unsupported coordinated removal refuses before mutation. [Public controls](../crates/polylogue/tests/e2e/cli/attachments.rs)
cover paging, transport privacy, stale delivery and directory/ZIP imports.

Parsers account for supplied records and fields, retain unknown source material,
and distinguish unknown values from zero or empty collections. Native compaction
history is context evidence, not newly authored messages. Provider-specific
interpretation and its exceptions live with the [source parsers](../crates/sources/src/lib.rs)
and [format tests](../crates/sources/tests/formats/main.rs).

A completed newer interpretation can retire its predecessor's messages, events,
links and metadata. Completion requires observed EOF and accounting through the
selected source length. Partial work survives restart and remains `reparsing`;
it cannot erase the prior interpretation prematurely. Different inputs' parser
version numbers do not compete. [Reparse tests](../crates/store/src/writer/tests/reparse.rs)
cover completion, promotion and rollback.

Document-framed imports validate the complete wrapper before publishing any
staged items. The [export driver](../crates/engine/src/import/exports.rs) keeps
ordering/deduplication state and compressed items in disposable disk-backed
scratch, then waits for each item's writer commit before decoding the next.
Retry rescans retained bytes and idempotently replays committed items; there is
no durable scanner cursor. Memory still includes the largest individual record,
conversation or decoded output. A synchronous export can delay new live work
until another discovery pass; it has no preemption or latency guarantee.

[Tool database capture](../crates/store/src/tool_databases.rs) reads one native
SQLite snapshot, stages complete logical changes and synchronizes them before
advancing its completed-prefix checkpoint. Recovery preserves altered completed
bytes and refuses ambiguous history. Capture identity binds the machine and
normalized native path, independently of clocks or host labels. Legacy unbound exports remain retained
without an inferred native profile. Cooperative
capture turns let convergence service controls and ordinary inputs between
turns; a single SQLite call or record can exceed the turn budget. The owning
[capture tests](../crates/store/src/tool_databases/tests.rs) cover completion and
recovery.

[Browser capture](../crates/polylogue/src/browser/captures.rs) acknowledges a
synchronized append to a framed spool. Exact retries are idempotent; an
interrupted matching prefix can complete, while ambiguous tails are preserved
and refused. Current acquisitions supply partial session coverage. Missing
members or omitted metadata do not prove deletion. Retained snapshot and source
ordering determine the current projection. The [spool tests](../crates/polylogue/src/browser/captures/tests.rs)
and [capture parser tests](../crates/sources/src/browser_capture/tests.rs) own
these rules.

Non-conversation text uses the model's [Document](../crates/model/src/document.rs)
and Store's [document projection](../crates/store/src/documents.rs), with source
associations and content versions separate from messages. `document:ID` supports
read and export; the `documents` relation searches current text. Supplied account
or project text remains evidenced by each immutable import. A later empty or
null collection is not an account-wide retraction. [Sidecar tests](../crates/sources/src/claude_ai/sidecars_tests.rs)
and [document convergence tests](../crates/engine/tests/convergence/excision_documents.rs)
cover identity, interpretation and standing exclusions.

## Read snapshots, queries and continuation

[ReadPool](../crates/store/src/read.rs) bounds client reader admission and gives
convergence a separate reserved reader. A reader is thread-affine and reuses
bounded SQLite connections. A checkout that would wait while its thread already
holds a slot refuses rather than deadlocking. Closing the pool wakes waiters and
refuses unused reservations; already pinned readers can finish.

On first read, shared publication exclusion pins the attached archive/index
snapshots and validates removal revisions and pending cleanup. Writers take
exclusive publication exclusion before SQLite write locks. A pinned WAL reader
then releases that exclusion, allowing later publications while retaining its
view. An idle old-index handle refreshes; a reserved replaced handle refuses.
An embeddings snapshot has its own checked removal revision and is not a claim
that all three databases were pinned simultaneously. [Admission tests](../crates/store/src/read/tests/snapshot_admission.rs)
cover those races. Cache budgets are SQLite targets with mmap disabled, not hard
process memory or result-size limits.

Ordinary query pages bind the resolved question, clock/seed, index generation
and index/archive counters. An intervening publication returns `stale_cursor`;
no transaction survives between requests. Page size may change. Semantic ranking
supports complete single-page answers, without continuation over a changing
vector set. [Query page tests](../crates/polylogue/src/daemon/query/tests/page_binding.rs)
own this contract.

Budgeted session and document reads instead bind selected content, view and
index generation. Unrelated writes can leave them usable; changed content or a
replaced index refuses continuation. Replaying the source prefix reconstructs
masking state. The budget bounds returned output, not input memory or RSS;
session reads can require one whole decoded message, and document reads one
block. Ordinary unbudgeted formats can assemble complete output. See the
[read operation](../crates/polylogue/src/daemon/service/read.rs),
[model pager](../crates/render/src/model_pages.rs) and
[read-page tests](../crates/polylogue/src/daemon/service/tests/read_pages.rs).

The `changes --since POSITION` operation prepares a finite reference batch,
validates and masks all its rows in a temporary file, then drops the reader
before socket delivery. Only its `complete` line offers a new position; the
receiver must apply or stage every prior record before persisting it. Disconnects
can leave an unknown outcome and retries can replay references. A generation
change requires replacement membership and replay of canonical notes/excisions.
An arbitrary query over `changes` retains ordinary query semantics and does not
acknowledge delivery. [Changes](../crates/polylogue/src/cli/changes.rs) and its
[tests](../crates/polylogue/src/cli/changes/tests.rs) own this protocol.

## Presentation and application lifetime

[Render](../crates/render/src/lib.rs) shares content presentation across CLI,
HTML exports and the web reader. View projection preserves message metadata and
tool fields. Structured export preserves importable model records, events and
links rather than converting them to prose.

The model's [masker](../crates/model/src/masking.rs) processes ordered source
content before view filtering, page splitting or structured encoding. Titles and
[search snippets](../crates/store/src/search/tests/masking.rs) share source-order
masking. Independently authored unordered structured fields do not establish
chronology for a secret split across them. The public application response
boundary also masks decoded strings, nested metadata, errors and object keys
before transport escaping. Recorded source bytes remain unchanged; exported
credential masking survives reimport. Browser control responses have their own
response paths. [Masking tests](../crates/model/src/masking/tests.rs) and
[CLI read tests](../crates/polylogue/tests/e2e/cli/read_masking.rs) cover the
shared content behavior.

The daemon owns archive access; CLI archive commands require it in either
backing mode. Startup receives explicit standalone or Sinex backing. The
application runtime owns convergence, signal workers, the optional publisher
and archive lock; transports own listeners and client work. Shutdown stops
admission and closes the read pool before joining transports/workers, then
releases the writer lock. Already admitted writes finish; pinned readers live
until dropped. Transport restart is separate from application shutdown.
[TestDaemon](../crates/polylogue/src/daemon/service/tests.rs) uses the same
lifetime boundary. Publisher cancellation leaves unacknowledged remote outcomes
unknown; synchronous archive/blob work has no hard shutdown deadline.

## Excision and Sinex backing

[Excision](excision.md) owns supported removals, standing refusals and known
representation limits. It preserves unrelated shared content and canonical
identities, leaves original provider files untouched, and refuses unsupported
rewrites before committing. Exact pending blob/vector deletion intents survive
interrupted paired commits. Readers refuse inconsistent removal state; reopening
validates prior rules before recovery or rebuilding the derived index.

Each new excision records an immutable exact vector-deletion policy. Embeddings
reconciliation commits deletion and its archive-bound receipt together; repeated
restore preserves later repurchases. Legacy missing policies mean unknown
coverage, not an empty deletion set, and cannot retrospectively identify vectors
to remove. [Vector cleanup](../crates/store/src/excise/vector_cleanup.rs) and
[policy restore tests](../crates/store/src/sinex_rows/tests/vector_policies.rs)
own this distinction.

[Store's Sinex row interface](../crates/store/src/sinex_rows.rs) publishes
commit-ordered units, each with checked identity, row count and content hash.
Observations, delivery events and extracted blobs are independent units, so
later facts do not mutate acknowledged parent batches. Historical embedded facts
remain compatible with verified legacy closures. Machine alias dependencies
include only aliases eligible at the unit's recorded commit position. Migration
preserves current legacy closures but cannot reconstruct acknowledged bytes
already changed by older software; those conflicts require the original verified
batches. [Machine alias tests](../crates/store/src/sinex_rows/tests/machine_alias.rs)
and [append-fact tests](../crates/store/src/sinex_rows/tests/append_facts.rs)
cover ordering and closure preservation. The
[machine publication contract](machine-publication.md) owns legacy held-frame
reporting and its cached-comparison limits.

[Restore](../crates/store/src/sinex_rows/restore.rs) validates confirmations,
gaps, replacements, source bytes and row ownership before atomically installing
the selected state. It refuses a different nonempty archive, missing local
independent facts or omitted/changed removal rules. A stale confirmed view
cannot reopen removed content. Its [catalog](../crates/store/src/sinex_rows/restore/catalog.rs)
keeps operation-wide selection on disk; payloads are verified through EOF and
applied row by row. The largest descriptor, JSON row or manifest remains an
individual memory bound. [Restore tests](../crates/store/src/sinex_rows/restore/batch/tests.rs)
cover malformed payloads and rollback.

Excision publishes exact replacement batches or tombstones. After row-frame
acknowledgment, the [Sinex adapter](../crates/sinex/src/lib.rs) requests retirement
using the latest confirmed owner and manifest digest, including on retries that
already hold every frame. Shared current references retain their backing.
Logical publication does not prove physical removal: results distinguish
completed cleanup, shared retention, pending unlink and unsupported older
servers. Legacy excisions without replacement metadata refuse primary
publication/restore. The [native backing control](../crates/sinex/src/tests.rs)
exercises real publication, restore and retirement; primary-mode qualification
remains a separate operational acceptance decision.
