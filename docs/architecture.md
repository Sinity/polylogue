# Architecture

Polylogue is an archive and workbench for AI work. This page states what it is
for, the principles every area follows, the two modes it runs in, how data
flows from a file on disk to an answer, where it is stored, what a crash can
leave behind, and how configuration and time reach the code. Each area's
behaviour is on its own page (see the [index](README.md)); the crates and how
to build them are in [`AGENTS.md`](../AGENTS.md).

## Purpose and users

Polylogue captures every session its user has with AI tools (coding agents
such as Claude Code, Codex and Gemini CLI; web chats such as ChatGPT, claude.ai
and AI Studio), keeps the original bytes, parses them into one model of
sessions and their lineage, and lets people and agents search, read, measure
and annotate that history. On top of the parsed sessions it holds what its
users author about them: notes, tags, judgments, claims, Pages, saved queries
and kept results.

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
deletes or rewrites stays in the archive. Ruling, 2026-07-22: "WE ARE
SUPPOSED TO KEEP ALL DATA"
(cc-queued:0398d7a3-0d3d-42e4-acd7-bca417e0c481@2026-07-22T12:20:09Z).

**Agents are primary users.** Agents get every power the operator has: every
command, every kind of authored record, every setting. What distinguishes the
operator is authorship, which the daemon computes from the channel a write
arrives on and which no caller supplies ([authorship](assertions.md#authorship)).
Ruling, 2026-07-03: "I want agents to have more power / capabilites rather
than less, more affordances"
(cc-queued:30f5e650-7033-4423-9f91-2b66ebd1e6b2@2026-07-03T13:20:44Z).

**Valid input is never refused or truncated for size.** Page sizes are
defaults, transfer units continue, and memory is bounded by streaming and
read-ahead rather than by caps on what may be stored or asked for. A query
that runs past its deadline names a narrower query and the same command with
the deadline lifted ([deadlines](query.md#deadlines)). Ruling, 2026-09-27:
"such arbitrary limits are themselves straight-up bugs"
(cc:62be45f2-6c40-49ce-a88e-bc566948628c:4cc20d7c-9e62-4b7a-9550-7dd97f9d48e9).

**Invariants, not repair.** The code does not produce broken state, so there
is nothing to repair: no repair, reconcile, reindex or verify-and-fix command,
and no scheduled checks. A crash leaves only states that the ordinary open path
treats as normal ([durability](#durability)). Reads of content-addressed bytes
check their hashes as they read.

**Overlap everything.** No stage waits on another that has nothing to do with
its data. The one serial stage in a pipeline only appends and commits
([pipeline](#pipeline)).

**Configuration only where the choice is open.** A behaviour with a dominant
option is not configurable. Polylogue reads each tool's data where the tool
writes it, and recognises inputs by their declared structure; there are no
custom roots and no guessing ([capture](capture.md)).

**Polylogue's concepts are its own.** Sessions, lineage, assertions, Pages and
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
   exports when a user runs `polylogue import`
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
5. **Serve.** The daemon answers the CLI, MCP, HTTP and the web workbench from
   one command table ([interfaces](interfaces.md#command-table)). Queries
   compile to a plan the store executes; every run is recorded
   ([query](query.md#query-objects)).
6. **Author.** Users and agents write assertions, Pages and settings through
   one write path ([assertions](assertions.md#write-path)).

The daemon is the only process that opens the archive. It holds the archive
lock, runs the one writer and serves every client over its socket; a command
that needs the archive with no daemon running fails with "daemon down"
([exit codes](interfaces.md#exit-codes)).

## Stores

| Store | Holds | Rebuilt from |
|---|---|---|
| segment log | the bytes of every input revision, as content-addressed blobs in append-only segment files of seekable zstd frames | nothing: it is the source |
| `archive.db` | what cannot be recomputed: inputs, revisions, chains, extents, observations; the request log (every query run and kept result); assertions, Pages and variants; settings; delivery records | nothing |
| `index.db` | everything parsed and derived: sessions, messages, blocks, tool calls, attachments, links, derived columns and tables, the search indexes; the positions in `archive.db` and the request log it has applied | the segment log and `archive.db` |
| embeddings store | message vectors for semantic ranking | the model that made them (vectors bought from a paid provider are kept as data) |

The segment container, blob hashing and member access are the common
[`hashed-blobs`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/hashed-blobs/README.md)
crate. In primary mode, Sinex holds the bytes and the records, and Polylogue's
local state is `index.db`, the embeddings store and the producer spool
([primary mode](primary-mode.md)).

## Durability

Writes reach disk in one order:

1. Bytes are appended to a segment and synced, once per write group, on a
   dedicated thread overlapped with the next group.
2. The extents and the rows that cite them commit in one transaction.
3. `index.db` saves the archive and request-log positions it has applied, in
   the same transaction as the rows it applied.
4. Source cursors (how far a growing file has been read) are saved only for
   committed rows.

The first build commits its parsed rows and positions durably in each group. A
saved `building` marker defers search and secondary indexes; reopening resumes
the committed build, including an interrupted index finish. The same boot
retains the index generation and read cursors. A changed boot rotates the
generation and its cursor secret while keeping rows and parse positions.

A crash therefore leaves at most these states, and ordinary code resumes from
each: a segment tail past the last committed extent, which is truncated when
the segment opens; `index.db` behind its saved position, which catches up; a
source cursor behind its committed rows, which re-reads and finds the bytes
already stored. Deduplication checks the hash index together with the hashes
appended in groups not yet committed, so a resumed write never stores a blob
twice.

## Pipeline

Building the archive overlaps every stage:
- read-ahead, bounded by memory;
- workers on every core that hash, compress, parse, and resolve keys and
  versions against a read snapshot and an in-memory key map;
- one writer that only appends and commits;
- consumers of committed positions: derived data, the request-log projection
  and the completion catalog;
- at the end of a first build, full-text and secondary indexes built in
  parallel, with the archive searchable before the trigram index finishes;
- live inputs ahead of bulk work.

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
  [`query-language`](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/query-language.md)
  crate. Dates, `today` and `group by day` all mean it. In primary mode
  Polylogue reads Sinex's day setting and has none of its own.
- **Two homes for settings.** Deployment configuration (the Nix modules, the
  configuration file, environment variables) holds what is set per machine:
  mode, archive path, the logical day, listen address and authentication, the
  optional notification command, the Sinex endpoint. The changeable settings
  catalog holds what any author may change at run time (quiet hours, the
  interrupt budget, enabled variants, briefing switches), each key with a
  type, default and scope; unknown keys are refused. Keys and defaults will
  be listed in the [planned settings reference](reference/README.md#planned-pages); how
  they are loaded is in
  [operations](operations.md#configuration).
- **Derived outputs and cursors record the settings they were made under**, so
  a changed setting re-derives them through the ordinary version mechanism.
- **One instant type, one clock, one error type** whose kind names are stable
  on the wire; conversions happen only at the storage boundary. Unknown times
  stay unknown.

## Store boundary

A query resolves to a closed `Plan`, which the store executes. All SQL lives in
the `store` crate, with one row codec per table; no other crate opens SQLite,
and `cargo xtask boundaries` enforces it. Clients of the daemon receive typed
answers, never connections or SQL.

## Deletion

Polylogue has no deletion. Re-parse is not deletion: it replaces a chain's
parsed rows with a new reading of the same stored bytes, and the earlier
reading stays as an earlier version ([versioned rows](archive.md#versioned-rows)).
A secret that matters is edited out of its source by the operator or an agent,
as an exceptional act; both revisions stay in the archive. In primary mode,
Sinex's lifecycle (archive, restore, tombstone) applies to Polylogue's
materials as to any other source's
([Sinex lifecycle](https://github.com/Sinity/sinex/blob/master/docs/lifecycle.md)).
Ruling, 2026-10-28: "probably just do not have deletion in the first place in
polylogue, for now"
(cc-queued:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f@2026-10-27T23:14:04Z).

## Laws

- **crash-leaves-only-resumable-states**: killing the daemon at any point
  between append, sync, commit and truncate leaves only the states listed
  under [durability](#durability), and the next open resumes each with no
  repair step; checked by fault-point kill tests.
- **index-rebuilds-equal**: an `index.db` rebuilt from the segment log and
  `archive.db` equals the one built incrementally, row for row.
- **no-blob-stored-twice**: appending a blob whose hash is in the index or in
  an uncommitted group stores nothing new.
- **settings-never-reread**: no component reads the environment, the system
  zone or a setting's default outside `main`; checked by constructing every
  component from an explicit configuration in tests.
- **same-answer-every-transport**: one operation gives the same typed answer
  over the CLI, the socket, HTTP and MCP.
- **no-deleting-operation**: no operation in the command table removes an
  input, a revision or an authored record.
