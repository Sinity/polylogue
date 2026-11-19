# Architecture

Polylogue is an archive and workbench for AI work. This page states what it is
for, the principles every area follows, the two modes it runs in, how data
flows from a file on disk to a reply, where it is stored, what a crash can
leave behind, and how configuration and time reach the code. Each area's
behaviour is on its own page (see the [index](README.md)); the crates and how
to build them are in [`AGENTS.md`](../AGENTS.md).

## Purpose and users

Polylogue captures every session its user has with AI tools (coding agents
such as Claude Code, Codex and Gemini CLI; web chats such as ChatGPT, claude.ai
and AI Studio), keeps the original bytes, parses them into one model of
sessions and their lineage, and lets people and agents search, read, measure
and annotate that history. On top of the parsed sessions it holds what its
users author about them: notes, tags, judgments, claims, saved queries,
watches and kept results.

It has two kinds of users with the same powers:
- **the operator**, the person whose sessions these are, who works mostly in
  the web workbench and the CLI;
- **agents**, who work mostly through MCP and the CLI, start every session
  with a brief from the archive, and write into it as they work.

## Principles

**The archive keeps the original inputs, and holds what cannot be
recomputed.** Every revision of every captured file, export and tool database
is stored as bytes; everything parsed from them is an index that can be built
again by reading the bytes. Exports are kept whole, and a file a tool later
deletes or rewrites stays in the archive.

**Agents are primary users.** Agents get every power the operator has: every
command, every kind of authored record, every setting. What distinguishes the
operator is authorship, which the daemon computes from the channel a write
arrives on and which no caller supplies ([authorship](authorship.md)).

**Valid input is never refused or truncated for size.** Page sizes are
defaults, transfer units continue, and memory is bounded by streaming and
read-ahead rather than by caps on what may be stored or asked for. A query
that runs past its deadline names a narrower query and the same command with
the deadline lifted ([deadlines](query.md#deadlines)).

**Invariants, not repair.** The code does not produce broken state, so there
is nothing to repair: no repair, reconcile, reindex or verify-and-fix command,
and no scheduled checks. A crash leaves only states that the open path
treats as normal ([durability](#durability)). Reads of content-addressed bytes
check their hashes as they read.

**Overlap everything.** No stage waits on another that has nothing to do with
its data. The one serial stage in a pipeline only appends and commits
([pipeline](#pipeline)).

**Configuration only where the choice is open.** A behaviour with a dominant
option is not configurable. Polylogue reads each tool's data where the tool
writes it, and recognises inputs by their declared structure; there are no
custom roots and no guessing ([capture](capture.md)).

**Polylogue's concepts are its own.** Sessions, lineage, annotations and
measures are Polylogue's. Code is shared with Sinex only where the mechanics
are identical (bytes and segments, the query grammar, the operation and failure
types, the web shell, signals), and neither product's concepts are bent to fit
the other's.

## Modes

**Standalone**: Polylogue alone, on SQLite and its own segment log. It is a
complete product: capture, archive, index, authored records, every surface.

**Primary**: Polylogue on Sinex. Polylogue captures and parses exactly as it
does standalone, but its bytes become Sinex materials and its parsed and
authored records become Sinex events; Polylogue keeps a local index that one
subscription to Sinex's confirmed stream keeps current. Everything a user can
do is the same in both modes. [Primary mode](primary-mode.md) owns what
changes.

## Data flow

1. **Capture.** The daemon watches each tool's own locations, reads the
   browser extension's captures and hook events as they arrive, and takes
   exports when a user runs `polylogue import PATH`
   ([capture](capture.md), [extension](extension.md)).
2. **Store.** Each new revision of an input is hashed, compressed and appended
   to the segment log; its extents and the input's revision and chain rows
   commit together ([archive](archive.md#stored-inputs)).
3. **Parse.** A per-origin parser reads each changed chain into sessions,
   messages, blocks, tool calls and attachments, versioned so that every
   reading is kept ([archive](archive.md#parsing), [sessions](sessions.md)).
4. **Derive.** Derived values are computed at session grain as their data
   arrives, and stored with the rows they belong to
   ([measures](measures.md#rules)).
5. **Serve.** The daemon serves the CLI, MCP, HTTP and the web workbench from
   one operation table ([interfaces](interfaces.md#operation-table)). Queries
   compile to a plan the store executes; every run is recorded
   ([saved queries](saved-queries.md#query-objects)).
6. **Author.** Users and agents write annotations (notes, tags, judgments,
   saved queries, settings) and alternates through one write path
   ([annotations](annotations.md#the-write-path)).

The daemon is the only process that opens the archive. It holds the archive
lock, runs the one writer and serves every client over its socket; a command
that needs the archive with no daemon running fails with "daemon down"
([exit codes](interfaces.md#exit-codes)).

## Stores

| Store | Holds | Rebuilt from |
|---|---|---|
| segment log | the bytes of every input revision, as content-addressed blobs in append-only segment files of seekable zstd frames | nothing: it is the source |
| `archive.db` | what cannot be recomputed: inputs, revisions, chains, extents, observations; the request log (every query run and kept result); annotations, settings among them; alternates; delivery records | nothing |
| `index.db` | everything parsed and derived: sessions, messages, blocks, tool calls, attachments, links, derived columns and tables, the search indexes; the positions in `archive.db` and the request log it has applied. Built whole in a private file and installed when the build ends ([durability](#durability)) | the segment log and `archive.db` |
| embeddings store | message vectors for semantic ranking, selected by the [paid-vector restore policy](search.md#what-is-embedded), with earlier values preserved in original source stores | the model that made them (vectors bought from a paid provider are kept as data) |

The segment container, blob hashing and member access are the common
[`hashed-blobs`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/hashed-blobs/README.md)
crate. In primary mode, Sinex holds the bytes and the records, and Polylogue's
local state is `index.db`, the embeddings store and the producer spool
([primary mode](primary-mode.md)).

## Durability

Writes reach disk in one order:

1. Bytes are appended to a segment and synced, once per write group, on a
   dedicated thread. Groups waiting together share one append and one sync.
   The next group is appended only after this group commits: appending it
   during the sync left its pages unwritten while the commit's database syncs
   ran, which then paid for them. In paired release-build runs with groups
   arriving faster than they are stored, the overlapped writer acknowledged a
   group later than the serial one (typical latency 3.2 s against 2.4 s for
   2 MiB groups, 1.2 s against 0.8 s for 256 KiB groups), also when each
   appended range's writeback was started early.
2. The extents and the rows that cite them commit in one transaction.
3. `index.db` saves the archive and request-log positions it has applied, in
   the same transaction as the rows it applied.
4. Source cursors (how far a growing file has been read) are saved only for
   committed rows.

A crash therefore leaves at most these states, and the normal start resumes from
each: a segment tail past the last committed extent, which is truncated when
the segment opens; `index.db` behind its saved position, which catches up; a
source cursor behind its committed rows, which re-reads and finds the bytes
already stored. Deduplication checks the hash index together with the hashes
appended in groups not yet committed, so a resumed write never stores a blob
twice.

Those states cover a process crash. A power loss or kernel panic also drops
writes the kernel had not yet flushed, so the stores that hold irreplaceable
data sync every commit: `embeddings.db` (paid vectors) commits with SQLite's
`synchronous=FULL`; the writer syncs `archive.db`'s log after each commit,
before the commit is acknowledged or anything is written after it; and
segments sync before their extents commit. `index.db` is rebuilt from those,
so it commits with `NORMAL`. The `archive.db` sync happens after the commit is
visible and outside the lock readers take to start a read, so a read never
waits for a disk sync; a power loss during that sync loses only that commit,
which nothing was written after. A writer also syncs `archive.db` and its log
when it starts, so what an earlier process committed and the kernel still
holds is written before it replies to anything. A failed sync stops writing for
the rest of the boot: the commit is visible, so a retry would find it stored
and report success, and the kernel may have marked the unwritten pages clean,
so no later sync in that boot, in the same process or a new one, shows that
they reached the disk. Recovery starts with a reboot
([operations.md#storage-failure](operations.md#storage-failure)).

**Building `index.db`.** A build of the index, the first one or one that
replaces an index from another schema, writes a private file,
`index-build/index.db`, in a directory with copy-on-write (and so
filesystem compression) off, with SQLite's rollback journal and temporary
sorts on disk and no transaction syncs. A failed transaction still
rolls back in the process; a lost process or boot may leave the file torn.
Before the build's first unsynced write the writer creates and syncs the
marker `index-build/unsynced`. A clean stop syncs the file and removes the
marker, and the next start resumes the build in it. A start that finds the
marker discards the file and builds again from the stored bytes, which a
build never changes, so a crash or power loss during a build costs the build
so far and nothing else. When the build ends, its writer syncs the file,
closes it, switches it to WAL while nothing else has it open, renames it
`index-build/built.db`, removes the marker and renames it over `index.db`,
syncing the directory after each step; a start that finds `built.db`
completes the rename. Pipeline reads use `built.db` between the two renames.
The active file changes with each rename under the same lock that selects
read paths and reads their file identity, including when a following sync
fails. A reader opened before a rename is refused as replaced before its
first database read; an already-started read keeps its open snapshot.
Only the build pipeline opens the private file: the
writer, its resolver and parser-context reads, the engine's planning reads
and tool-database catalog reads. Readers of that owned private file permit
SQLite's native rollback-journal recovery while refusing application writes.
The pool retains temporary tables for read hydration; serving and preserved
source databases stay genuinely read-only. Serving readers, the search thread and the
derive thread open `index.db` once it is installed, so no reader can hold up
the switch to WAL or a stop. With a rollback journal a held read blocks
the writer, so the pipeline's reads of the build file are short: a resumed
parse looks up its committed context one statement at a time, and a
tool-database comparison copies its recorded rows out in one read. While the build runs, reads are served from
the previous index when it has this build's schema identity, and otherwise
reply `index_building` with the build's progress
([installing.md](installing.md#replacing-an-index-from-another-schema)).

## Pipeline

Building the archive overlaps every stage:
- read-ahead, bounded by memory;
- workers within the process's CPU and memory allowance that hash, compress,
  parse, and resolve keys and versions against a read snapshot and bounded
  key dictionaries backed by sorted spill files; a new job
  goes to the worker that went idle last, so the ranges of one large input
  run on one worker and reuse the memory its previous range freed, and the
  pass holds about one job's memory per job in flight rather than one per
  worker; a new input starts only when its first range can be queued at
  once, so the inputs a pass has started finish before it starts more, and
  the parser state kept between an input's ranges leaves out the keyed
  context's changes, which the next range reads from the index. The same
  pool replaces stored chains' readings: one task parses a chain's ranges
  in order into a staged reading and hands it to the writer for one commit.
  Its unparsed bytes count against the pool's memory budget until the
  reading is handed to the writer; only batches carrying decoded blobs
  retain their own charge until the writer consumes them. A task larger
  than the budget parses alone, but its staged reading waiting for commit
  leaves room for live ranges. With more than
  one worker, at most all but one parse replacement chains at once, leaving
  a worker for live ranges. A chain starts after any queued or running
  range plan of its path; readings of the same path may run together;
- one writer that resolves text hashes to row identities, appends and commits;
  a bounded dictionary with sorted spill files keeps text identities for the
  writer's lifetime, including while an unreferenced text has no row.
  Builds preload hashes by row order; other writes learn identities as needed.
  The resolver reads existing messages' selected rows, routes and source claims
  in bounded groups within its existing snapshot. Shared version/layout, input
  and source-session facts are fetched once per key group, so repeated claims
  do not repeatedly read the same metadata or decode the same session facts.
  Missing build keys skip database probes. A fixed-size Bloom filter skips
  spill-file probes for unseen keys; possible matches still compare the stored
  row's complete key. Session and required foreign-key indexes remain live.
  The resolver switches to transaction upkeep after installation and releases
  its build dictionaries.
  After a failed index transaction the writer reloads row presence without
  forgetting allocated identities. Workers prepare text hashes and compress
  each step's texts together in groups ([text bodies](archive.md#evolution))
  without reserving text rows; the writer stores a group with the first of
  its texts the index lacks. Registering a version returns its row
  key and the text references it just wrote; selected blocks consume those
  references. When that call also wrote every text payload, parser continuation
  checks consume its prepared identity rather than reading and decoding its
  body again. Resolver preparation and version registration share one immutable
  message value. An uncomplicated complete selection refers to that version
  entry in the same prepared unit instead of copying its body and encoded rows;
  the reference survives a staged reading's disk roundtrip. Composition and
  sparse enrichment keep their owned selected value and time/item overlays.
  Missing or mismatched selected references refuse the transaction normally.
  Older, unchanged or reused-text versions keep full payload,
  hash and layout validation. A new message is inserted with its resolved
  selected metadata before stated sources refer to it. During a private build,
  an ordinary fresh reservation lets the insert check for the row itself. A
  split write retry can then reuse a row installed by an earlier retry. The
  first body installation consumes an actual creation. Later installations
  replace the existing blocks and routes, including multiple changes within one transaction;
  after a row application installs its selected messages and stated sources,
  one indexed operation resolves their event placements from that final state.
  Deleting a referenced message first detaches its events to preserve their
  natural placements and the foreign-key constraint;
- consumers of committed positions: derived data (computed on a thread of
  its own and written by the writer only if its inputs are unchanged), the
  request-log view and the completion catalog;
- at the end of an index build, the writer fills both full-text tables in one
  transaction in the private index. A bounded scan reads each referenced text
  group once per window and visits its texts in rowid order. Each table uses
  its installed schema's tokenizer and configuration. A stop or error rolls
  back both tables together. The writer builds the trigram and deferred
  secondary indexes before installing the build as `index.db` ([durability](#durability));
- live inputs ahead of bulk work.

The writer's transactions are sized by the bytes and rows they cover, not
by record or input boundaries: a growing file is read and committed in
chunks (16 MiB by default), queued batches share a transaction up to a row and
byte bound, and the items of an export or tool-database dump commit in
batches of about one chunk of its bytes. Each record lies whole inside one
transaction, together with the position it advances (a parser checkpoint, or
an export's delivery cursor), so readers never see part of a record and an
interrupted parse resumes after the last committed batch. An export holds
at most two batches in memory: one committing while the next is parsed. A whole-chain
replacement is the exception: it commits once, because readers must see
either the old reading or the new one. Live ranges go ahead of queued bulk chain
tasks through the pool's live queue, including while replacements run.
Between transactions the writer starts
a write-ahead log over once it passes its checkpoint size (about 1 GB),
waiting briefly for readers still reading it, so the log stays bounded while
readers follow every commit. The pipeline's database connections do not checkpoint on close.

**Build memory.** The binary captures CPU affinity, CPU quotas, physical memory
and enclosing memory limits once. The pipeline's allowance is at most 3 GiB
and at most three quarters of that process limit. At a 4 GiB limit this gives
a 384 MiB index cache, 96 MiB register cache, at most 192 MiB each for readers,
queued input and key dictionaries, and 48 MiB per search/derive connection.
Workers take a 128 MiB share each from one third of the allowance, so there
are at most eight, and never more than the CPU allowance or requested count.
These shares include both raw range buffers and prepared parser work; one
third is reserved for the worker working sets rather than added to the
in-flight allowance. The two full-text builders also have 64 MiB hash buffers
each, their page caches and one shared 64 MiB text queue. The remaining space
covers parser context, planning caches, SQLite bookkeeping and allocator
retention; the process limit leaves a further quarter outside the allowance.
Memory maps are disabled and temporary sorts spill to disk.

Build planning takes fixed pages of input IDs in input order, newest chain
first. Historical layouts, extents, comparisons and copy stops live on disk
with fixed page caches, so a large archive or one input's many revisions does
not retain their complete plan in memory. Shared prefixes stay adjacent and
are compared once. Progress reports committed parsed bytes out of the bytes
to parse, including a partially read chain. Completed parser state is restored
from its stored checkpoint when needed instead of kept for every recent file.
A record or text is still processed whole; an oversized unit runs alone and
can exceed a normal chunk or queue target. The budgets bound archive-count
growth, while its largest individual processing unit determines that extra
working space.

**Stopping.** A process has one stop request, carried by the archive it
opened; the daemon makes it on `SIGTERM`. Each request's cancellation is a
child of it. Every reader the store hands out checks it in its SQLite
progress handler and every blocking wait registers a wake on it; the loop and
its workers run under it, so the store connections opened there, their
source-byte reads and their parsing steps observe it; the writer's
reconstructible work (derivation, deferred indexes, replay of stored bytes,
copies of identical chains' readings) rolls back to its last commit. Captured
bytes in hand and writes already handed to the writer commit. A stop skips
optional checkpoint work and leaves committed WAL frames beside their
databases; normal commit syncing is unchanged, and the next open reads those
frames and resumes from committed positions
([operations](operations.md#the-daemon)).

**Targets**, on the full corpus of an active user (about 40 GB): at least
40 MB/s; searchable in about 25 minutes; at most 0.6 bytes written per input
byte; no stage idle while another holds a queue for it. Results are reported
as hardware-independent figures (bytes written per input byte, syncs per
group, stage utilisation) beside the absolute rates of the host that ran them
([benchmarks](testing.md#benchmarks)).

## Configuration and clocks

Each binary's `main` loads one typed configuration (the configuration file,
then flags, then environment variables where meaningful), validates it there
and passes it, with a `Clock`, to every component. Libraries never read the
environment or the system time zone, and no component has a default
constructor for a setting. A time zone is resolved to a named zone or the
daemon refuses to start; paths are absolute.

- **The logical day** is one `DaySetting`, a named zone and a start time
  (default 04:00), with one implementation in the common
  [`query-language`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md)
  crate. Dates, `today` and `group by day` all mean it. In primary mode
  Polylogue reads Sinex's day setting and has none of its own.
- **Two homes for settings.** Deployment configuration (the Nix modules, the
  configuration file, environment variables) holds what is set per machine:
  mode, archive path, the logical day, listen address and authentication, the
  optional notification command, the Sinex endpoint. The changeable settings
  catalog holds what any author may change at run time (quiet hours, the
  interrupt budget, enabled alternates, briefing switches), each key with a
  type, default and scope; unknown keys are refused. How they are loaded is in
  [installing](installing.md#configuration).
- **Derived outputs and cursors record the settings they were made under**, so
  a changed setting re-derives them through the usual version mechanism.
- **One instant type, one clock, one error type** whose kind names are stable
  on the wire; conversions happen only at the storage boundary. Unknown times
  stay unknown.

## Store boundary

A query resolves to a closed `Plan`, which the store executes. All SQL lives in
the `store` crate, with one row codec per table; no other crate opens SQLite. Clients of the daemon receive typed
replies, never connections or SQL.

## Deletion

Polylogue has no deletion. Re-parse is not deletion: it replaces a chain's
parsed rows with a new reading of the same stored bytes, which stay, so any
reading can be produced again; rows read from other chains and other inputs
stay as they are ([versioned rows](archive.md#versioned-rows)).
A secret that matters is edited out of its source by the operator or an agent,
as an exceptional act; both revisions stay in the archive. In primary mode,
Sinex's lifecycle (archive, restore, tombstone) applies to Polylogue's
materials as to any other source's
([Sinex lifecycle](https://github.com/Sinity/sinex/blob/master/docs/lifecycle.md)).
