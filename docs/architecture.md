# Architecture

Polylogue is an archive and workbench for AI work. This page states what it is
for, the principles every area follows, the two modes it runs in, how data
flows from a file on disk to a reply, where it is stored, what a crash can
leave behind, what the path from stored bytes to index rows guarantees, and
how configuration and time reach the code. Each area's
behaviour is on its own page (see the [index](README.md)); the crates and how
to build them are in [`AGENTS.md`](../AGENTS.md).

## Purpose and users

Polylogue captures every session its user has with AI tools (coding agents
such as Claude Code, Codex and Gemini CLI; web chats such as ChatGPT and
claude.ai; AI Studio), keeps the original bytes, parses them into one model of
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
its data. Whatever can be done for one input alone is done on every core at
once, and the serial stages only merge, load and commit
([pipeline](#pipeline)).

**One path.** Keeping the index current and building it from nothing give the
same index, and neither depends on the order in which inputs arrived
([pipeline](#one-path)).

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

**Primary**: Polylogue on Sinex. Polylogue keeps the same record and the same
index, written by the same code. Three parts are added beside them: its bytes
are handed to Sinex's store and read from there, so they exist once; its
record is shipped to Sinex in commit order and can be restored from it; and
one reader of Sinex's committed log applies what other hosts recorded and what
Sinex's lifecycle decided. Every operation is the same in both modes; what differs is listed in [primary-mode.md](primary-mode.md#the-rule).
[Primary mode](primary-mode.md) owns what is added.

## Data flow

1. **Capture.** The daemon watches each tool's own locations, reads the
   browser extension's captures and hook events as they arrive, and takes
   exports when a user runs `polylogue import PATH`
   ([capture](capture.md), [extension](extension.md)).
2. **Store.** Each new revision of an input is hashed and compressed, and
   its bytes are put: made durable and recorded as blobs. Its extents and
   the input's revision and chain rows then commit together to the record
   ([archive](archive.md#stored-inputs)); nothing parsed from those bytes
   reaches the index before that commit.
3. **Parse.** A per-origin parser reads each chain that is due into a reading:
   the sessions, messages, blocks, tool calls and attachments its records
   state ([archive](archive.md#parsing)).
4. **Compose and derive.** The rows shown for a session are composed from
   every reading that states it ([sessions](sessions.md)), and its derived
   values are computed at session grain from those rows
   ([measures](measures.md#rules)).
5. **Serve.** The daemon serves the CLI, MCP, HTTP and the web workbench from
   one operation table ([interfaces](interfaces.md#operation-table)). Queries
   compile to a plan the store executes; every run is recorded
   ([saved queries](saved-queries.md#query-objects)).
6. **Author.** Users and agents write annotations (notes, tags, judgments,
   saved queries, settings) and alternates through one write path
   ([annotations](annotations.md#the-write-path)).

The daemon is the only process that opens the archive. It holds the archive
lock, runs its two writers (the archive writer, which writes `archive.db`,
and the index writer, which writes `index.db`) and serves every client over
its socket; a command
that needs the archive with no daemon running fails with `not_running`, naming the archive and `polylogue daemon`
([exit codes](interfaces.md#exit-codes)).

## Stores

| Store | Holds | Rebuilt from |
|---|---|---|
| segment log | the bytes of every input revision, as content-addressed blobs in append-only segment files of seekable zstd frames | nothing: it is the source |
| `archive.db` | the record, which cannot be recomputed: inputs, revisions, chains, extents, observations; the request log (every query run and kept result); annotations, settings among them; alternates; delivery records | nothing |
| `index.db` and its full-text files | everything parsed and derived: each chain's reading, the shown sessions, messages, blocks, tool calls, attachments and links composed from them, each text once, derived columns and tables, the search indexes; which reading it holds of each chain and the record position it has taken in. Built whole in private files and installed when the build ends ([durability](#durability)) | the stored bytes and `archive.db` |
| embeddings store | message and annotation vectors for semantic ranking, selected by the [paid-vector restore policy](search.md#what-is-embedded), with earlier values preserved in original source stores | the model that made them (vectors bought from a paid provider are kept as data) |

The segment container, blob hashing and member access are the common
[`hashed-blobs`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/hashed-blobs/README.md)
crate. In primary mode the stores are the same except for the bytes, which
live in Sinex's store, with a holding directory on the host for bytes Sinex
has not acknowledged yet ([primary mode](primary-mode.md#local-state)).

## Durability

Writes reach disk in one order:

1. Bytes are put: appended to a segment and synced, once for all the puts
   waiting together, and their blob rows commit, with how far the segment is
   committed, in a small transaction of their own before any of those puts
   returns.
2. The record rows that cite them (extents, revisions, chains and
   observations) commit in one transaction of the archive writer. Authored
   records commit in transactions of their own on the same writer.
3. `index.db` commits a reading's rows, read from bytes the record already
   names, together with the state of the chain they were read from, the
   shown rows of the sessions they touch, and the record position the index
   has taken in.
4. Source cursors (how far a growing file has been read) are saved only for
   committed bytes.

Three invariants hold at every commit. No row of the record names bytes that
are not durable, and no put returns before the record knows its blob. Every reading in the index is the reading of its chain's
bytes up to the length its state names. The shown rows equal the composition
of the readings present ([pipeline](#pipeline)).

A crash therefore leaves at most these states, and the normal start resumes
from each:

| Left by a crash | The start |
|---|---|
| a segment tail past the last recorded blob | cuts it when the segment opens; a segment holding more than that tail past the end its record names means `archive.db` is older than its segment (put back from a copy), and the open stops, naming both lengths, without cutting anything ([operations](operations.md#restoring-archivedb-alone)) |
| a blob recorded that no extent cites yet | keeps it: the resumed write finds the blob and does not store it again |
| the record ahead of the index: a chain stored further than its reading, or record rows the index has not taken in | reads on from where each reading ends; a reading is a function of the bytes, so the rows are those an uninterrupted run would have written |
| sessions marked due for their derived values, texts waiting for their search entries | computes and writes them |
| a source cursor behind its committed bytes | re-reads and finds the bytes already stored |
| the files of an unfinished index build | discards them and builds again from the stored bytes |

A blob is stored once: a resumed write finds the bytes it was storing.

Those states cover a process crash. A power loss or kernel panic also drops
writes the kernel had not yet flushed, so the stores that hold irreplaceable
data sync every commit: `embeddings.db` (paid vectors) commits with SQLite's
`synchronous=FULL`; each `archive.db` commit's log is synced before the
commit is acknowledged or anything is written after it; and segments sync
before their blob rows commit. `index.db` is rebuilt from those,
so it commits with `NORMAL`. The `archive.db` sync happens after the commit is
visible and outside the lock readers take to start a read, so a read never
waits for a disk sync; a power loss during that sync loses only that commit,
which nothing was written after. A writer also syncs `archive.db` and its log
when it starts, so what an earlier process committed and the kernel still
holds is written before it replies to anything. A failed sync stops writing for
the rest of the boot: the commit is visible, so a retry would find it stored
and report success, and the kernel may have marked the unwritten pages clean,
so no later sync in that boot, in the same process or a new one (where the machine has a boot identifier), shows that
they reached the disk. Recovery starts with a reboot
([operations.md#storage-failure](operations.md#storage-failure)).

**The index file.** A build of the index writes private files in a directory
of its own, without syncing as it goes, and installs them by rename when it
ends; until then nothing but the build opens them. A build never changes the
record or the stored bytes, so losing one costs the build and nothing else: a
start that finds a build a crash or a stop left discards its files and builds
again from the stored bytes, a build that fails in a running daemon is
discarded the same way and the next pass builds again, and a start that finds
a finished build not yet installed installs it. A new archive's first index
is built the same way, from the files its first pass stores. While a
build runs, reads are served from the installed index when this build of
Polylogue can read it, and otherwise reply `index_building` with the build's
progress
([installing.md](installing.md#replacing-an-index-from-another-schema)). A
reader opened before an install is refused as replaced before its next read
begins; a read already started keeps the file it opened.

## Pipeline

What the path from a stored byte to an index row guarantees. Each rule below
is one a caller could see broken or an invariant the code holds, with its
reason. How the code meets them is the business of the engine's and the
store's module documentation, not of this page.

### The record and the index

- **The record is the one account of what was captured and authored.** It is
  `archive.db` with the bytes it cites: inputs, their chains and
  observations, authored records and the request log. One writer, on its own
  thread, writes it; the byte home records the blobs it puts. The same code
  writes it in both modes through four operations: put bytes (durable on this host
  before the call returns), open bytes by hash and range, commit a group of
  captured inputs or authored writes, and read what was committed after a
  position. Nothing above those operations knows where bytes are kept or
  whether Sinex is present, so the engine, the parsers, the index, queries
  and every surface are the same code in both modes
  ([primary mode](primary-mode.md)).
- **The index is a function of the record.** `index.db` is what the parsers,
  at their current parse versions, make of the stored bytes. The same
  record gives the same index whatever order its inputs arrived in, in
  however many steps their bytes were read, on whichever host, and whether
  it was kept current as the record grew or built from nothing. "The same
  index" means every reply a reader can give: the rows, their values and
  their order. Arrival order is an accident of capture; an index that
  depended on it could be made right only by repairing it.
- **The index holds nothing that cannot be computed again.** Beside its rows
  it keeps which reading it holds of each chain and the record position it
  has taken in, each committed with the rows it describes. Row keys and
  positions that are local to one index never appear in a reply, in a
  reference or in an order a reader returns: where rows tie, a natural key
  breaks the tie. Two indexes of one record may therefore differ in their
  local keys and in nothing a reader can see, and an index can be replaced
  at any time.

### Readings

- **A reading is the parsed form of one chain under one parse version.**
  It is a function of the chain's bytes up to a length and of its parser's
  parse version, and of nothing else. It is owned by its chain: its rows
  carry the input and the chain, and no other chain's reading shares a row
  with it. Only text is shared, stored once
  ([text](archive.md#text)). Because a reading owns its rows, replacing or
  removing one touches no other reading and needs no count of who else
  refers to a row.
- **One row for each thing a chain states.** A reading holds, for each
  session, message, event, link, document and alias its records state, what
  those records say about it, taken together in byte order. Reading a chain
  further changes only the things the added records state, so the work an
  append causes is in proportion to the records appended and what they
  state, never to the size of the session, the input or the archive.
- **A chain is in one of three states**: it has no reading; its reading is
  complete at a length under a parse version; or reading it failed at a
  version, with the reason
  ([parse versions](archive.md#reading-versions)). A chain is read when
  its stored length or its parser's parse version differs from what its
  state names, and for no other cause. The state is one value in one row;
  every reader judges it, with what the archive says of the chain, through
  the same function. A part of a reading commits only where the stored
  state ends; one planned from an earlier state is refused and changes
  nothing, so no plan's memory of a position can disagree with the index.
- **A chain that a newer chain of its input covers has no reading**, and a
  chain that begins like a newer one holds rows only for the records the
  newer one lacks ([covered chains](archive.md#readings)). The rows it would
  have held are the newer chain's rows for the same bytes.
- **Readers see a reading appear whole.** The first reading of a chain and
  the replacement of a reading under a new parse version each commit in
  one transaction; a chain that grows commits record by record, each record
  whole and together with the position it advances. A session is never
  shown half through one reading and half through its replacement.

### Parsers

- **Parsers are pure.** The rows a record yields depend on the bytes of its
  chain up to the end of that record, and on nothing else: no stored state,
  no other input, no lookup in the index. What a parser carries from one
  record to the next is bounded by what is still open (an unfinished
  record, the session a file's header names, the message before this one),
  not by how much it has read. Two consequences are relied on everywhere: any
  chain can be parsed alone, on any host, in any order; and a chain parsed
  in pieces gives the reading it gives parsed at once.
- **Rows name things as the source names them.** A message names its parent
  by the source's parent ID, a tool result names its call by the call ID, a
  link names its ends by their session IDs, and a record that is not a
  message but stands in a chain of parents yields an alias row saying which
  name it stands for. No row a parser yields depends on whether the thing it
  names is in the archive.
- **Names are resolved by joining rows, never by writing into them.** Three
  scopes, each resolved from rows alone: names that belong to one chain (a
  response written as several records, the number that tells equal
  content-named messages apart) when that chain's reading is formed; names
  within a session (a parent, an alias, a tool call) when the session is
  composed, from every input that carries it; names of other sessions (a
  link's ends, a branch point) when they are read. A name whose item is not
  in the archive reads as unresolved and is shown as such; when the item
  arrives, from any input and in any order, the name resolves and no row
  that held it is changed.

### Composition

- **A session's shown rows are a function of the readings that state it**,
  by the composition rules of
  [sessions.md](sessions.md#one-conversation-through-several-routes): which
  session an ID names, which reading supplies each field group of a message,
  where each message is placed.
- **Shown rows are computed, never adjusted.** When a reading is added,
  extended, replaced or removed, the rows of the sessions it states are
  computed again from the readings then present, in the same transaction.
  Nothing is corrected by the difference a step made, so there is no case
  in which the rows depend on the order of the steps. Recomputing may be
  confined to the part of a session a step can have changed, on the
  condition that the result is that of recomputing all of it.
- **Composition writes the shown rows and nothing else**, the same in
  upkeep and in a build. What follows from the shown rows (search
  membership, the trigram index on tool calls) is kept by the index schema
  as they change ([search](search.md#index-maintenance)); derived values are
  computed by their own steps.
- **Session-grain values have one mechanism.** A value of one session that
  is not a column of one of its rows (its counts, spans, values, live
  facts, active path, links, the place of each transcript entry among the
  others) is a session summary: declared, versioned, computed from the
  session's shown rows off the writer for the sessions marked due, and
  never kept by adding and subtracting what a change contributed. The code
  that composes a session marks it due with the earliest place its change
  reached; a summary that depends on the rows in order is a fold, resumed
  from the state stored before that place ([measures](measures.md#convergence)).

### One path

The daemon reaches the index two ways, which must agree.

- **Upkeep** keeps the installed index current as the record grows: for each
  chain whose state is behind, it reads on, and commits the new rows with
  the recomputed rows of the sessions they touch.
- **A build** makes an index from nothing. Every chain that is due a reading
  is prepared on its own, on as many workers as there are; the prepared rows
  are merged by key, which is where rows of one session from different
  inputs meet and are composed, equal texts become one, and row keys are
  given in key order; the tables are loaded in key order into fresh files,
  their indexes are created afterwards, and the full-text files are built
  beside them at the same time ([search](search.md#index-maintenance)). A
  build looks nothing up in what it is writing.
- **A build composes each session from its readings as a group.** Preparing
  stores each reading's content once, in content groups written in the order
  it was read ([archive](archive.md#content)), and carries each message
  reading's keys, place, items, word count and a reference to its content
  into a stream sorted by the session that states it; no body travels in the
  stream. The merge gives each reading its text keys and each content group
  its key. Each session is composed once from its readings, by the
  composition upkeep uses, with nothing stored to compare against. Groups of
  whole sessions are composed on as many threads as the build has workers,
  each into an index of its own held in memory together with what follows
  from the shown rows (the transcript lines, counts, values, live facts), reading
  the readings' content from the build file; the writer only appends their
  rows, in stream order, each key carried past the keys its table already
  holds. A session whose shown rows depend on readings stated
  under other sessions is composed from the loaded readings instead, as
  upkeep composes it: a copy named through another session, a message stated
  under several sessions, a reading holding part of a version other readings
  complete, or readings upkeep wrote while the build ran. A session's events
  that follow a message of another session are placed knowing, from what the
  merge read of every message's statements, that the message is shown
  there, so a session composed from its group looks no message up by ID
  and the build's index of message IDs is made after the sessions are
  written. An event several
  sessions state, such as a quota reading every session of an origin
  shares, is composed from the loaded readings once the sessions are. Links
  and documents, which no session holds, are composed before the sessions.
- **They give the same index.** For any set of inputs, bringing them in
  through upkeep in any order and in any division into steps gives the index
  a build gives from the same record, as read through the reader. This is
  the property the write path is tested by: one property test over generated
  inputs, orders and divisions, comparing replies and not tables
  ([testing](testing.md)).
- **Either may stand in for the other.** Because they agree, the daemon
  builds anew instead of catching up when that is less work (an index far
  behind its record, a parse version changed for most of the corpus, a
  large import into a small archive), and nothing a reader sees depends on
  which it chose.
- **A build reads the record as it stood at one position.** It writes only
  its own private files; upkeep writes only the installed index. What is
  captured while a build runs is stored and committed to the record at once,
  as always, and kept current in the installed index if there is one. When
  the build installs, it is an index behind its record like any other and
  upkeep brings it forward from the position it was built at. Preparing may
  start on a chain as soon as its bytes are stored, also while other inputs
  are still being captured; the merge takes only readings that are of their
  chain at the length the build's position gives.
- **A build never makes a request wait for it.** Its workers read chains
  without the writer or the index a request reads, so they never pause;
  an import is answered between the chains handed to the workers, and the
  writer's own build work (load, indexes, composition, full text) runs in
  steps of one transaction each, between the requests it serves. No chain
  is read twice.
- **Live inputs go before bulk work.** A file being written now is read and
  committed ahead of stored chains waiting for a reading.
- **Memory is bounded by streaming.** A build's memory does not grow with
  the number of chains, sessions or texts: prepared rows go to sorted files,
  the merge reads them forward, and the largest single record, text or
  message is the largest thing held whole. A unit larger than a working
  share runs alone. No valid input is refused or cut for its size. The
  process's allowance is the memory the system has available when it
  starts, at most its control group's limit, and three quarters of it is
  shared by the caches and the work, with no fixed ceiling: a build's
  prepared readings in flight, its workers' sort buffers, each step's run
  readers and read-ahead are given fixed shares, and the build file's cache
  takes what the step leaves. Parse workers follow the physical cores the
  process may use, each with at least a minimum share of memory; the
  memory a chain's prepared readings need is admitted per chain, so the
  number of workers does not trade against the caches. A worker holds the
  rows a chain's records look each other up by in memory while it reads
  the chain, admitted with the chain, and writes the chain's content only
  in the order it was read.

A writer's transactions are sized by the bytes and rows they cover, not by
input boundaries: a growing file is read and committed in steps, and each
record lies whole inside one transaction. A user's write waits for at most
the archive commit being written, never for an index transaction or a build
step: user writes have their own queue on the archive writer, served before
its next bulk group, and the index writer follows the archive. What follows
in the index from archive commits alone (alternate text and its search
entries, import membership) is applied by the index writer from the rows
committed after the record position it keeps, which it advances. Bytes
stored without parsed rows change nothing in the index: whether their chain
is due is judged from its state and the archive's facts.

One connection writes the index: the index writer's. Derivation and full-text
entries are computed beside it on read connections from committed rows, and
the writer applies the values in its own transactions, so no write waits on
another writer's lock.

**Stopping.** A process has one stop request, carried by the archive it
opened; the daemon makes it on `SIGTERM`. Each request's cancellation is a
child of it. Every reader checks it and every blocking wait wakes on it. Work
that can be done again from stored bytes (a build, a reading in progress,
derivation) is dropped at the stop; a build starts again from the stored
bytes, and the others resume from their last commit; captured
bytes in hand and writes already handed to the writer commit. The next open
resumes from committed positions ([operations](operations.md#the-daemon)).

### Targets

**The target is a rate.** A first build runs at 40 MB/s of input or better,
from the source files to an installed, searchable index, on eight cores and
within a few GB of memory, measured on a corpus whose index is several times
larger than that memory. At that rate a corpus of 40 GB builds in under
twenty minutes and one of 110 GB in about three quarters of an hour.

**Why it is believed reachable.**
- Parsing, decoding and preparing rows cost about 0.09 CPU-seconds per MB of
  input with the production parsers (measured on 8.59 GB of generated Claude
  Code input). At 40 MB/s that is fewer than four of eight cores, and
  preparing has no serial part.
- Two prototypes of a build that writes each page once, with the production
  parser and keeping the original bytes, message identity, citations and
  phrase search, ran that corpus at 37 and 41 MB/s, in about 440 CPU-seconds
  and writing about half a byte per input byte.

**What that evidence does not show.** The prototypes left out composition
across inputs, blocks and tool calls, events and links, derived values and
upkeep, and their corpus had one origin, one reading for each file and a
narrow vocabulary. The rate is therefore shown for a simpler index and a
simpler corpus than the real ones. A rate also means nothing without its
memory: a build whose index fits in memory runs faster than the same build
of a larger corpus, so a figure counts only when memory was scaled with the
corpus.

**What holds the design to it.** Before the build path is written, a
prototype of the whole of it (prepare, merge with composition and equal
texts made one, load in key order, index creation, full-text in several
files, derived values) is run with the production parsers on a corpus shaped
like real inputs: every origin, files that grew and files that were
rewritten, sessions carried by several inputs, many small files, and memory
scaled to the corpus. It reports the hardware-independent figures a build is
judged by: CPU-seconds per input GB; bytes read and bytes written per input
byte; the wall time of each serial stage as a share of the build; peak
memory, which must stay flat as the corpus grows; and that search over
several full-text files ranks as search over one. Where it shows a lower
rate than the target, the stage that bounds the rate is named with its
figures, and this section is restated to the rate that was shown and the
corpus it was shown on.

The index may be larger than its input. What the design excludes is a second
copy of anything in it: each text once, each message body once for each
reading, and no table that restates another. Results are reported as
hardware-independent figures beside the absolute rates of the host that ran
them ([benchmarks](testing.md#benchmarks)); no test asserts elapsed time.

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
  optional away command, the Sinex endpoint. The changeable settings
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
the `store` crate; no other crate opens SQLite. Clients of the daemon receive typed
replies, never connections or SQL.

A row the store reads or writes is a **row struct**: plain data with named
fields, declared once, whose field list is the statement's column list, the
reading of a result row, the binding of an insert and the row's byte form in
sorted files. Statements are declared beside their row struct; one test
prepares every declared statement against a fresh index and archive and
compares its columns with the struct's fields, so a statement and its struct
cannot drift apart unnoticed. Store keys are typed by their table
(`SessionPk`, `MessagePk` and the others) and bind as SQLite integers.

## Deletion

Polylogue has no deletion. Re-parse is not deletion: it replaces a chain's
reading with a new reading of the same stored bytes, which stay, so any
reading can be produced again; the readings of other chains and other inputs
stay as they are ([readings](archive.md#readings)).
A secret that matters is edited out of its source by the operator or an agent,
as an exceptional act; both revisions stay in the archive. In primary mode,
Sinex's lifecycle (archive, restore, tombstone) applies to Polylogue's
materials as to any other source's, and reaches Polylogue through its record
([primary mode](primary-mode.md#what-sinex-decides);
[Sinex lifecycle](https://github.com/Sinity/sinex/blob/master/docs/lifecycle.md)).
