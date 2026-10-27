# Architecture

Polylogue stores the files AI tools write, and what people and agents add, in
a local archive; parses them into sessions, messages and documents; and serves
them through one daemon. This guide describes the current code. The
[README](../README.md) describes the commands, [vocabulary.md](vocabulary.md)
the names, and Beads the planned work.

## Crates

Data flows `Sources → Engine → Store`; clients go through the application,
which reads through Store and presents through Query and Render.

| Crate | What it owns |
|---|---|
| [Model](../crates/model/src/lib.rs) | Identities, messages, blocks, documents, events and outcomes shared by the other crates; no I/O. |
| [Sources](../crates/sources/src/lib.rs) | Reading each origin's formats. An [origin declaration](../crates/sources/src/origin.rs) holds everything known about an origin: its locations, how to recognize its files and split them into records, its parser and its tool facts. [Parse](../crates/sources/src/parse.rs) and [Records](../crates/sources/src/records.rs) turn records into model values with byte ranges and record counts. |
| [Engine](../crates/engine/src/lib.rs) | The loop: find changed files, store their bytes, schedule parsing and derivations. The [parse pool](../crates/engine/src/pool.rs) feeds Store's writer; it holds no SQL. |
| [Store](../crates/store/src/lib.rs) | Every SQLite connection and write, the blob store, derivations and excision, behind the typed `Writer`, `ReadPool` and `Reader`. |
| [Query](../crates/query/src/lib.rs) | The query language and its compiler to parameterized SQL. The application's [query operation](../crates/polylogue/src/daemon/query.rs) runs it on a Store reader. |
| [Render](../crates/render/src/lib.rs) | Views of typed content as terminal text, Markdown, HTML and JSON, and pages sized for models. |
| [Application](../crates/polylogue/src/daemon/service.rs) | Typed requests, configuration and worker lifetime. The [command table](../crates/polylogue/src/commands.rs) declares every command once for the CLI, MCP and HTTP; the [socket server](../crates/polylogue/src/daemon/server.rs) owns framing and listeners. |
| [Sinex client](../crates/sinex/src/lib.rs) | Primary mode: inputs, records and parsed sessions sent to Sinex, and the local copy rebuilt from what Sinex holds. |
| [Testkit](../crates/testkit/src/lib.rs) | Synthetic archives built through the real loop. Store's own row fixtures sit behind its [fixture feature](../crates/store/src/fixture.rs). |

The `explain` command reads the command, query-unit, field, read-view and
format declarations. Query explanation uses the same reader-backed options as execution, then
Query's pure parser and compiler. Declaration pages use a cursor naming
their subject, unit and next offset. The local `demo tour` command bundles the
main Morrow files and starts an application runtime against its own home and
archive; each narrated operation is built through the ordinary command
table. Its transcript is written under the new tour directory.

A new origin is an origin declaration and a parser, a new command is a
command-table entry, and a new measure is an entry in the
[measure table](../crates/polylogue/src/daemon/measure.rs). Crate dependency
rules are checked by [xtask](../xtask/src/main.rs).

An origin's collector receives the configured home, state and spool directories.
It runs automatically or when its declared setting is on. Antigravity's
loopback collector runs in an engine-owned worker; the loop reads its spool on
subsequent passes and joins the worker at shutdown. The optional AI Studio
Drive import runs before the pass. Both are declared in the origin table.

## Databases

[Archive](../crates/store/src/archive.rs) owns the directory, the archive lock
and the checks at open. One writer thread makes every change. Notes,
settings, deliveries and browser requests have a queue of their own that it
serves before its next batch, so during a first build they wait for at most
the transaction being written.

| File | Holds |
|---|---|
| [`archive.db`](../crates/store/src/schema/archive.sql) | Inputs, revisions, extents and observations; notes, settings, kept results and deliveries; chat actions and backfill jobs; excisions; blob rows. Opening an older one applies the numbered migrations it lacks (`ARCHIVE_MIGRATIONS` in [schema.rs](../crates/store/src/schema.rs)) in one transaction; a newer one, or one whose tables match no known version, refuses. |
| [`index.db`](../crates/store/src/schema/index.sql) | Parsed sessions, messages, documents and events, search, and derived products. Rebuilt from the archive when deleted, when its tables differ from `index.sql`, or when `INDEX_VERSION` changes, which it does whenever the index schema, the row encoding or any parser's output for the same bytes changes. |
| [`embeddings.db`](../crates/store/src/schema/embeddings.sql) | Purchased vectors and provider failures, kept apart so a rebuild does not buy them again. |
| [`blobs/`](../crates/store/src/blobs.rs) | Stored bytes by BLAKE3 hash. A blob counts once its row says `present`. |

A rebuild parses the stored inputs, including imports whose original files
are gone; notes live in `archive.db` and survive it. A
message seen in several inputs is stored once. [The writer](../crates/store/src/writer/rows.rs)
keeps one `message_source` row per input chain that holds it and shows one
version: within an input the newest chain; within a chain the earliest record,
or the latest for inputs whose records repeat whole conversations; across
inputs the session it started in, then a complete version, then the more
faithful arrival, then the smaller content hash. Messages are ordered by source
position, with message IDs breaking ties; no time is invented.
[Import rebuild tests](../crates/engine/src/import/tests/replay.rs) cover
rebuilds from stored imports.

Derived products declare their inputs and rule version in
[derive](../crates/store/src/derive.rs); a changed rule recomputes what depends
on it. An index build in progress is marked on disk and starts over on
reopen; [`Archive::prepare_index`](../crates/store/src/archive.rs) also
rebuilds an index that holds fewer excisions than the archive. An index
rebuild or a reboot starts a new index generation, which the change feed
reports.

A first build, into an empty index, writes without syncs, without copying
its write-ahead log into the database file until its rows are written, and
without the
indexes only reads use or those on natural keys (message and event IDs, text
hashes); the writer finds rows by key through hashes it keeps in memory. When
the build ends it writes the search entries, rebuilds the trigram index and
creates each dropped index once from a sorted scan, then derives, one session
at a time through every derivation. Until then a
read that needs one of those indexes scans instead.

## Storing bytes

[Import](../crates/engine/src/import.rs) and the loop store the bytes before
anything parses them. A pass feeds its plans to the parse pool a few at a
time; every two seconds it looks at Polylogue's own spools (hook events,
browser captures) and at the live files written in the last ten minutes, and
what changed there goes ahead of the work still waiting, so a capture or a
session being written shows up within seconds even during a first build. The [writer](../crates/store/src/writer/staging.rs)
checks them against excisions, commits the blob rows as `writing` with their
revision and extents, writes the files, then marks the rows `present`. After a
stop, opening the archive and every pass finish what is left: files not yet
moved under their content names are deleted (on open), a `writing` row that
stored bytes use becomes `present` when its file exists, and a `writing` row
nothing in the archive uses (decoded bytes whose parsed rows never committed)
is deleted with its file. The rest are listed in `status` as
`pending_inputs`; readers refuse their byte ranges, and only the same bytes
can finish them. A provider file that changed meanwhile starts a new chain. [Stored-byte tests](../crates/engine/src/import/tests/stored_bytes.rs)
cover imports, watched files and parse-pool groups. Bytes a parser decodes
from inside a record, such as inline images, are stored as blobs of that
input.

A ZIP import stores the ZIP itself and each file in it as inputs, linked by
`observation_zip`. Export [attachments](attachments.md) are found
through those links and the [asset lookup](../crates/store/src/assets.rs).

Parsers count every record and field they were given, keep what they could
not read as diagnostics, and keep unknown values apart from zero or empty.
Each origin's rules and exceptions live with its
[parser](../crates/sources/src/lib.rs) and its
[format tests](../crates/sources/tests/formats/main.rs).

An export that wraps many conversations in one document is checked whole
before any of it is written. The [export reader](../crates/engine/src/import/exports.rs)
keeps its ordering state and compressed items in scratch files and waits for
each item's commit before decoding the next; a retry reads the stored bytes
again and skips what was committed. Memory holds the largest single record,
conversation or decoded output.

A [tool database](../crates/store/src/tool_databases.rs) is stored as a row
dump (JSON lines) from one SQLite read of the tool's file; complete changes are written before its
checkpoint moves. Its identity is the machine and the file's path. Reading
is done in turns, so other inputs are served between them.

[Browser captures](../crates/polylogue/src/browser/captures.rs) are appended
to a capture log and synced before the extension is told they are stored. An
identical retry is a no-op; a partial matching tail is completed and a
different one refused. Each capture repeats the whole conversation, so the latest
capture's version of a message is shown, and a message missing from a later
capture stays.

The extension keeps nothing but captures the host has not yet acknowledged.
[Chat actions](../crates/store/src/chat_actions.rs) and
[backfill jobs](../crates/store/src/backfill_jobs.rs) are archive rows the
writer changes: the operator queues or starts them through the CLI, MCP or the
reader, and the extension claims them through the host, with a claim that
expires, and reports what it did.

A Gemini page capture and a Takeout export of one conversation are one
session, but they share no turn ID: the capture has the page's text with
no times, the export has times and no message IDs. Both arrivals stay
stored. The [duplicates derivation](../crates/store/src/derive/duplicates.rs)
shows a turn once, from the export, when its prompt text occurs once in
each arrival, the matched prompts keep one order, and the replies have the
same roles; the capture's messages then name the export message
(`duplicate_of`) and the reason. Other turns show from both arrivals, and
`duplicate:*` lists the hidden copies.

Text that is not a conversation is a [Document](../crates/model/src/document.rs)
in Store's [document rows](../crates/store/src/documents.rs), with one shown
version chosen by source time. `read document:ID` and `export` read it; the
`documents` unit searches it. A later export with an empty list removes
nothing.

## Reading

The [read pool](../crates/store/src/read.rs) holds a bounded number of reader
slots. A reader sees one read transaction on `index.db` with `archive.db`
attached; it starts on first use, under the shared
[write lock](../crates/store/src/write_lock.rs), so the two databases are a
committed pair. Writers hold the write lock only while the register commits,
and through schema setup, an index replacement or an excision; index writes
never take it. Outside an excision a reader waits at most for one register
commit, however long the writer keeps writing. A thread that would wait for a
slot while holding one gets an error instead of a deadlock. Closing the pool
refuses new reads; started readers finish. Cache sizes are SQLite page-cache
targets with mmap off, not a limit on process memory.

Every query answer has a total order: its sort keys, then the unit's
identity. A page's cursor holds the last row's values of those keys, with the
question, clock and random seed; the next page reads the rows after them, so a
write between pages neither refuses the cursor nor repeats or skips rows. A
query ranked by `near:` pages the same way: its similarities come from the
vectors held in memory, through `pl_similarity(text_hash)` while the query
runs, so its other conditions choose the rows before any ranking; with
words in the same query the two orders are fused (reciprocal rank fusion).
Its cursor is refused once vectors were stored or removed. Semantic setup runs
only when the answer needs ranking, including a limit that selects rows before
a count or measurement. A row answer's own semantic ranking keeps each row's similarity in a hidden column,
so one statement counts both the answer and its rows without a vector, after
stages such as `distinct`. Session `has:errors` and `has:tool_calls` inspect
its own messages: duplicates and abandoned branches are excluded. Every field
that names a session resolves an exact ID before a unique prefix and refuses
an ambiguous prefix. A query inside a
query (a combining stage, `FIELD in (…)`) is compiled on its own over the
outer query's parameters and read as a subquery of the one statement; a
union with a ranked side ranks by fusing every list's order, computed once
in the statement, so it pages like any other answer.
[Query page tests](../crates/polylogue/src/daemon/query/tests/pages.rs)
cover this.

A budgeted session or document read is bound to the selected content, view
and index generation: unrelated writes leave the cursor usable, a changed
content or new index returns `stale_cursor`. The only masking state that
crosses a block boundary is an open private key block, and a PEM header or
END line counts only whole within one block. The writer marks each message
holding either, and those holding a block that leaves a key open. A page
starts at its cursor's message, seeking to it by its key: masking starts
afresh there when no earlier message leaves a key open, and otherwise from
the state reached by masking only the marked messages from the first that
does. Cursors are sealed with a secret of the index
generation, so a changed one is refused and one from an earlier index is
stale.
Budgeted reads return both text and structured masked block slices. The web
reader uses these pages, with shared message anchors, latest selection and
metadata outlines; [the reader contract](web-reader.md) describes the API,
route ownership and authentication.
The budget limits output size, not
memory: a session page may decode one whole message, a document page one
block. See the [read operation](../crates/polylogue/src/daemon/service/read.rs),
the [model pager](../crates/render/src/model_pages.rs) and the
[read-page tests](../crates/polylogue/src/daemon/service/tests/read_pages.rs).

`changes --since POSITION` writes a finite batch of references, masked, to a
temporary file, releases its reader, then sends it. Only the final `complete`
line gives a new position; a client applies every earlier line before saving
it. A retry can send references again. A new index generation tells the
client to read everything again. A query over the `changes` unit is an
ordinary query. [The change feed](../crates/polylogue/src/cli/changes.rs) and
its [tests](../crates/polylogue/src/cli/changes/tests.rs) own this.

## Output and daemon lifetime

[Render](../crates/render/src/lib.rs) is shared by the CLI, HTML export and
the web reader. Structured export keeps model records, events and links as
data, so `import` reads them back.

The [masker](../crates/model/src/masking.rs) runs over content in source
order before views, pages or encoding, so titles and
[search snippets](../crates/store/src/search/tests/masking.rs) mask the same
way. Every application response also masks strings, metadata, errors and
object keys. Stored bytes never change.
[Masking tests](../crates/model/src/masking/tests.rs) and
[CLI read tests](../crates/polylogue/tests/e2e/cli/read_masking.rs) cover it.

The daemon owns archive writes and interactive reads. Offline schema tools
read explicit stored revisions directly; other CLI archive commands need a
running daemon in either mode. It keeps [warm state](../crates/polylogue/src/daemon/warm.rs)
between requests: the vectors `near:` ranks by (reloaded after a commit
stores or removes vectors), the provider's vectors of recent `near:` texts,
recent query answers by read position, completion data, and the archive
counts `status` shows. A write moves the read position and so retires what
was read before it. Counts that take more than 50 ms are counted again by a
worker of their own, so `status` answers from memory during a long pass and
says which read position its counts are from (`counted`). The application runtime owns the loop, signal workers,
and the archive lock; transports own listeners and client
work. Shutdown stops accepting requests and closes the read pool, joins
transports and workers, then releases the archive lock. Writes already
accepted finish. [TestDaemon](../crates/polylogue/src/daemon/service/tests.rs)
uses the same lifetime.

## Excision

[Excision](excision.md) removes whole stored inputs, with the sessions,
documents and notes only they hold, and never rewrites stored bytes; a
session or document whose inputs hold anything else refuses before anything
changes. It commits embeddings, then the archive, then the index, and
deletes the files of blobs it marked `deleting` last; reopening the archive
finishes a stopped excision. Provider files are never edited.

## Sinex primary mode

With the `sinex` feature and `mode = "primary"` in `sinex.toml`, Sinex holds
the archive and Polylogue is one of its producers. The modes differ only
where the register's rows go ([primary](../crates/store/src/primary.rs));
capture, parsing, the index, queries and every surface are the same, and
queries always read `index.db`.

| | standalone | primary |
|---|---|---|
| input bytes | `blobs/` and inline rows | Sinex's content store, read-only |
| register of inputs, notes, settings, deliveries | `archive.db`, the archive's own | `sinex.db`, a copy of what Sinex holds |
| parsed sessions, search, derived rows | `index.db` | `index.db` |
| embedding vectors | `embeddings.db` | `embeddings.db` |

- **Parts.** Each step of a captured file's appended bytes is sent as one
  part: a material of its real input (origin, path and host), a growing file
  a chain of parts, named with the parser `polylogue-<origin>` and its
  version. The material's manifest holds the
  [register rows](../crates/store/src/input_rows.rs) that place the part.
- **Register materials.** Every other change to the register is sent as a
  small register material: a look that found no new bytes (a file gone or
  back, an import of bytes already stored), each request a person or agent
  makes, and each excision. Notes, settings, kept results, deliveries and
  excisions also become their typed `polylogue.*` events.
- **Order.** The writer sends a part or register material and waits for Sinex's
  acknowledgment before the transaction that writes its rows commits, so
  `sinex.db` and `index.db` hold only what Sinex holds. Bytes wait in
  `parts/` until then. An outage fails the write; the loop tries again.
- **Sessions.** After each derive step, the loop sends the session, message,
  tool-call and turn events of every session whose rows changed, keyed by
  Polylogue's IDs and anchored at their records, in frames named by their
  content ([sessions](../crates/sinex/src/sessions.rs)). A changed parser
  re-sends the same keys, and the newer reading replaces the older one.
- **Rebuild.** When `sinex.db` is missing, opening the archive pages through
  every material Sinex holds from Polylogue and inserts the same rows again;
  the loop then rebuilds `index.db` from the stored bytes. The
  [rebuild test](../crates/sinex/tests/rebuild/main.rs) runs this on the
  Morrow example against a real `sinexd`.
- **Excision** removes the same whole inputs as in standalone mode. Before
  it commits, Sinex excises their parts and register materials, and those of
  the requests behind the notes it removes (`<root>.excise.<producer>`,
  Polylogue as the actor); Sinex refuses the whole request when it cannot
  remove one cleanly, and nothing changes. The excision itself is then sent
  as a register material, so a rebuild keeps refusing what it removed.
  Polylogue never deletes from Sinex's content store.
- **Connection.** `sinex.toml` names the NATS server, the producer
  (`polylogue`, also its NATS user), a `password_file` holding its
  password, and Sinex's store root. Credentials in the `nats` URL are
  refused, so no password sits in the configuration file. Polylogue
  writes the bytes it sends under the store root, so the account it runs
  as needs the store's group.

### Moving to primary mode

There is no tool for the move; it is a fresh collection into Sinex plus an
export of what only the standalone archive holds. The standalone archive is
left as it was.

1. **Start primary on an empty directory.** Write `sinex.toml` with
   `mode = "primary"` and start the daemon with `XDG_DATA_HOME` pointing at
   an empty directory, keeping the same state directory, so browser
   captures and hook spools are collected again. The loop collects every
   provider file and tool database into Sinex; wait until `polylogue
   status` shows `mode.state: primary`, `busy: false` and no
   `store.pending_inputs`.
2. **Export from standalone and import into primary.** Start the
   standalone daemon on its own archive (a configuration directory without
   `sinex.toml`, and a separate state directory so it does not collect
   alongside the primary one). Compare `polylogue sql "select id from
   session"` on both archives and `polylogue export` the sessions only the
   standalone one holds, together with `user-state`, into one file:
   `polylogue export <session>... user-state --out moved.json`. Then
   `polylogue import moved.json` into the primary archive. The sessions are
   mostly manual imports and captures whose files are gone; manual exports
   whose original ZIPs still exist can be imported again from those
   instead.
3. **Switch.** Point the live daemon, CLI and browser host at the primary
   archive and stop the standalone daemon.

**User state** is every request a person or agent made (notes with their
revisions, withdrawals and judgments, settings, kept results, deliveries,
chat actions and backfill jobs) and every excision, in commit order.
`archive.db` keeps each request whole, and import replays them through the
writer as the requests they were, so in primary mode they reach Sinex like
any new request. A request or excision already in the archive counts as
`present`, so importing the same file again changes nothing. An imported
excision refuses what it removed from then on. When step 1 already collected
something an excision removed (its provider file was still there), the
import stops at that excision and names it: excise it in the primary
archive and import the file again.

## Offline schema tools

[Schema tools](schema-tools.md) use a pure Sources observer and recursive diff,
called by local commands in the application's command table. Store selects
exact revisions and provides streaming byte reads in either mode. These
commands run independently of the engine, parsing jobs and the daemon.
The observer reports field names, structural paths and counts, grouping dynamic
map keys under `*`. Observed values stay out. Reads use read-only database
and blob handles, including while the daemon holds the archive lock. `xtask schema` delegates to the same command table.
