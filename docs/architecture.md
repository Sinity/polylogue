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
| [Sources](../crates/sources/src/lib.rs) | Recognize and interpret native formats. An [origin declaration](../crates/sources/src/origin.rs) owns discovery, framing and tool capabilities. [Parse](../crates/sources/src/parse.rs) and [Records](../crates/sources/src/records.rs) produce model values, source ranges and accounting. |
| [Engine](../crates/engine/src/lib.rs) | Discover changes, acquire retained bytes, schedule parsing and drive convergence. The [parse pool](../crates/engine/src/pool.rs) feeds Store's bounded writer; it does not own SQL or source interpretation. |
| [Store](../crates/store/src/lib.rs) | Own every SQLite connection, archive mutation, blob reference, read snapshot, derivation and excision. Its typed `Writer`, `ReadPool` and `Reader` interfaces isolate callers from database layout. |
| [Query](../crates/query/src/lib.rs) | Bind the query language and compile parameterized SQL. The application's [query operation](../crates/polylogue/src/daemon/query.rs) executes it through a Store reader and prepares answers. |
| [Render](../crates/render/src/lib.rs) | Project typed content into views and produce terminal text, Markdown, HTML and structured output. It owns content presentation; callers own transport and surrounding navigation. |
| [Application](../crates/polylogue/src/daemon/service.rs) | Own typed requests, configuration, convergence and worker lifetime. The [command table](../crates/polylogue/src/commands.rs) declares arguments, capabilities and output actions for CLI, MCP and HTTP. The [socket server](../crates/polylogue/src/daemon/server.rs) owns framing and listeners. |
| [Sinex client](../crates/sinex/src/lib.rs) | Primary mode: send the archive's row batches and input blobs to Sinex, and restore an empty archive from them. Store lists the batches and visits their rows; this crate encodes them. |
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
| [`archive.db`](../crates/store/src/schema/archive.sql) | Canonical input identities, revisions, extents and observations; authored notes, settings, variants, kept results and deliveries; excision rules and blob rows. [`schema.rs`](../crates/store/src/schema.rs) creates it at version 1, the only version, and refuses any other. |
| [`index.db`](../crates/store/src/schema/index.sql) | Parsed sessions, messages and documents, each with one shown version and the sources that hold it; search and derived products. It is rebuilt from the stored inputs when `INDEX_VERSION` changes, which it does whenever the index schema, the row encoding or any parser's output for the same bytes changes. |
| [`embeddings.db`](../crates/store/src/schema/embeddings.sql) | Purchased vectors and provider failures, kept apart from the index so a rebuild does not buy them again. |
| [`blobs/`](../crates/store/src/blobs.rs) | Content-addressed stored bytes. A file counts once its blob row says `present`; a row still `writing` is reported as unfinished. |

Index rebuilds parse the stored inputs, including manual imports whose
provider files have disappeared. Authored targets, recorded hashes and anchors
live in `archive.db` and survive a rebuild. A message seen in several inputs is
stored once; [the writer](../crates/store/src/writer/rows.rs) keeps one row per
input chain that holds it in `message_source` and shows the version that wins:
within one input the newest chain, within one chain the earliest record (or, for
inputs whose records restate whole conversations, the latest), and across inputs
the originating session, then a complete version, then the more faithful route,
then the smaller content hash. Message ordering uses stable source identities
and positions, with opaque message IDs breaking ties. It does not invent event
times. [Import replay](../crates/engine/src/import/tests/replay.rs) tests cover
rebuilds from stored imports.

Derived products declare dependency and rule versions in
[derive](../crates/store/src/derive.rs). Changed rules refresh their dependents;
idle derivation does not announce a new view. Search publication changes
availability without claiming that the underlying session was edited.

An unfinished index build has a durably synchronized marker and is rebuilt on
reopen. Index replacement and an operating-system boot change replace the public
generation; same-boot restarts preserve it. These mechanisms belong to
[archive generation](../crates/store/src/archive/generation.rs) and
[`Archive::prepare_index`](../crates/store/src/archive.rs), which also rebuilds
an index that holds fewer excisions than the archive.

## Acquisition, interpretation and provenance

[Import](../crates/engine/src/import.rs) and watched collection store the
bytes that parsing will read. The [writer](../crates/store/src/writer/staging.rs)
checks the excision refusals, writes the blob rows as `writing` with their
revision and extents, installs the files, then marks the rows `present`. A
stop in between leaves rows still `writing`: the next pass marks those whose
file exists `present` and reports the rest in `unfinished_inputs`, readers
refuse their ranges, and only the same bytes can finish them. A changed
provider file starts a new chain instead. [Stored-byte tests](../crates/engine/src/import/tests/stored_bytes.rs)
cover imports, watched inputs and parse-pool groups. Bytes a parser decodes,
such as inline images, are stored as blobs of the input that holds them.

[Export attachments](attachments.md) resolve through ZIP and member
observations and the [asset lookup](../crates/store/src/assets.rs). A ZIP
import stores the original ZIP and each member as inputs, linked by
`observation_container`. Excising bytes read from a ZIP removes that ZIP too
and keeps its other members; a ZIP whose bytes another kept input uses
refuses. [CLI tests](../crates/polylogue/tests/e2e/cli/attachments.rs)
cover paging, stale delivery and directory and ZIP imports.

Parsers account for supplied records and fields, retain unknown source material,
and distinguish unknown values from zero or empty collections. Native compaction
history is context evidence, not newly authored messages. Provider-specific
interpretation and its exceptions live with the [source parsers](../crates/sources/src/lib.rs)
and [format tests](../crates/sources/tests/formats/main.rs).

Document-framed imports validate the complete wrapper before publishing any
staged items. The [export driver](../crates/engine/src/import/exports.rs) keeps
ordering/deduplication state and compressed items in disposable disk-backed
scratch, then waits for each item's writer commit before decoding the next.
Retry rescans stored bytes and idempotently replays committed items; there is
no durable scanner cursor. Memory still includes the largest individual record,
conversation or decoded output. A synchronous export can delay new live work
until another discovery pass; it has no preemption or latency guarantee.

[Tool database capture](../crates/store/src/tool_databases.rs) reads one native
SQLite snapshot, stages complete logical changes and synchronizes them before
advancing its completed-prefix checkpoint. Recovery preserves altered completed
bytes and refuses ambiguous history. Capture identity binds the machine and
normalized native path, independently of clocks or host labels. Cooperative
capture turns let convergence service controls and ordinary inputs between
turns; a single SQLite call or record can exceed the turn budget. The owning
[capture tests](../crates/store/src/tool_databases/tests.rs) cover completion and
recovery.

[Browser capture](../crates/polylogue/src/browser/captures.rs) acknowledges a
synchronized append to a framed spool. Exact retries are idempotent; an
interrupted matching prefix can complete, while ambiguous tails are preserved
and refused. Each capture is a framed record that restates the conversation, so
the latest capture's version of a message is shown; a message missing from a
later capture stays, since a capture does not prove deletion. Every capture
stays in the stored input. The [spool tests](../crates/polylogue/src/browser/captures/tests.rs)
and [capture parser tests](../crates/sources/src/browser_capture/tests.rs) own
these rules.

Non-conversation text uses the model's [Document](../crates/model/src/document.rs)
and Store's [document rows](../crates/store/src/documents.rs), with one shown
version chosen by source time and the sources that hold it, separate from
messages. `document:ID` supports
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

On first read, one statement pins the attached archive and index snapshots
together under the shared write lock. Writers take
exclusive publication exclusion before SQLite write locks. A pinned WAL reader
then releases that exclusion, allowing later publications while retaining its
view. An idle old-index handle refreshes; a reserved replaced handle refuses.
An embeddings snapshot is pinned separately. [Admission tests](../crates/store/src/read/tests/snapshot_admission.rs)
cover those races. Cache budgets are SQLite targets with mmap disabled, not hard
process memory or result-size limits.

Every query answer has a total order (its sort keys, then the row's key). A
page's cursor holds the last row's values of those keys with the resolved
question, clock and random seed; the next page reads the rows after them, so a
write between pages neither refuses the cursor nor repeats or skips rows whose
keys did not change. Page size may change. A query ranked by `near:` returns
one complete page. [Query page tests](../crates/polylogue/src/daemon/query/tests/pages.rs)
cover this.

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
mode. Startup receives the standalone or primary mode explicitly. The
application runtime owns convergence, signal workers, the thread that sends
to Sinex and the archive lock; transports own listeners and client work. Shutdown stops
admission and closes the read pool before joining transports/workers, then
releases the writer lock. Already admitted writes finish; pinned readers live
until dropped. Transport restart is separate from application shutdown.
[TestDaemon](../crates/polylogue/src/daemon/service/tests.rs) uses the same
lifetime boundary. Stopping the Sinex thread leaves what Sinex has not
confirmed to the next pass; synchronous archive/blob work has no hard
shutdown deadline.

## Excision

[Excision](excision.md) removes a session, document, input or text by
overwriting stored bytes in place and parsing the changed inputs again. It
commits embeddings, then the archive, then the index, and deletes the files
of blobs it marked `deleting` last; reopening the archive finishes a stopped
excision by deleting those files and rebuilding an index that holds fewer
excisions than the archive. Provider files are never edited, and what the
design cannot overwrite refuses before anything changes.

## Sinex primary mode

With the `sinex` feature and `mode = "primary"` in `sinex.toml`, Sinex holds
the archive: Polylogue's raw inputs and user state. A
[row batch](../crates/store/src/row_batches.rs) is what one archive write
added: a revision with its input, extents, blob rows and observations; a note
with its targets and relations; an excision; a setting; a delivery; or a
later observation or delivery event. Batches are ordered by
the write counter, then kind and key. Store lists them and visits their rows;
the [Sinex client](../crates/sinex/src/lib.rs) encodes each batch as JSON
lines and sends it, after the blobs its rows cite, as a `polylogue.rows`
event under the name `rows:<archive>:<position>`. Each pass asks Sinex which
names it holds and sends the rest in order, so Sinex holds a prefix of the
archive. A batch whose bytes are still being written waits for a later pass.

Once Sinex holds the archive, the archive only appends batches: excision
refuses, and Sinex never deletes what it holds. An empty archive with
`archive = "<id>"` configured is restored from every batch Sinex has
confirmed, in one transaction, after checking each batch and blob against its
content name and that positions 1 to the last are all present. The loop then
builds `index.db` from the restored rows, reading blobs from Sinex's content
store. The [rebuild test](../crates/sinex/tests/rebuild/main.rs) runs this against a
real `sinexd`.
