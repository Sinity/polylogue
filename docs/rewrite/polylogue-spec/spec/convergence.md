# Keeping the archive up to date

There is one loop. It runs in the daemon continuously, and the same code runs
for the first build of an empty archive, for a rebuild after the index is
deleted or its version changes, and for one new record appended to a session
file. Nothing else writes parsed data.

## The rule

> `index.db` is a function of the stored inputs, the notes in `archive.db`,
> and the versions of the code that reads them: each parser's version and
> each derived product's version, which includes the data the product uses
> (the price table's hash, shipped or overridden). At any moment some of it may be out of date; the
> loop's only job is to find what is out of date and redo it. Nothing is ever
> "broken", at worst not yet up to date.

`embeddings.db` holds paid vectors: the loop adds missing ones, but they are
kept, not recomputed (`storage/embeddings.md`). Inputs, notes and excisions
are the only facts, and inputs are unordered: merging never depends on the
order they arrived in. Anything else the index needs from outside, such as a
repository's root commit or a commit's details, is collected and stored as an
input first, so the same stored inputs always give the same index.

Reads never parse or write. The stored rows are the answer, and their
provenance (the input, its revision chain and the byte range, one per
contributing input) is how a read cites its source.

## The loop

```
loop:
  1. collect     find new or changed inputs at the known locations
  2. store       copy their bytes into the blob store, record them in archive.db
  3. parse       for every stored input whose parsed result is missing or was made
                 by an older parser version: parse it, write its sessions to index.db
  4. derive      recompute derived products the writer could not compute in the
                 session's own transaction: after a product version or price change,
                 for children of a late parent, and embeddings
  5. excise      remove whatever an excision row names that still exists
  6. publish     (Sinex-backed mode only) send Sinex what it has not confirmed
  7. wait        until a file changes, a capture arrives, or a client request writes
                 something (a note, a tag, an import, an excision) that needs deriving
```

Each step finds its work by comparing what exists with what should exist, not
by reading a to-do list:

| Step | What should exist | What exists | Work = the difference |
|---|---|---|---|
| collect | files and databases at provider locations (device, inode, size, mtime_ns, ctime_ns); git lookups for commits and repositories named in sessions | the latest `observation` per (machine, path) in `archive.db`, cached in `file_cursor`; stored git-lookup inputs | new, appended, replaced, rewritten, renamed and deleted files; quiet files due a full check; databases due a snapshot; commits with no lookup record |
| store | collected changes | `revision`, `extent` and `observation` rows | bytes not yet stored; pieces no revision covers yet (at most a minute old); quiet inputs with joinable pieces; blob rows left `writing` or `deleting` |
| parse | every revision chain of every input whose format has a parser in this build, at that parser's version, after every excision of its input | `parse_state` rows | chains never parsed, parsed only partly or still finalizing, parsed by an older version, with fewer `excised` observations seen than the input has, or parsed with an older revision of a companion than the companion now has; messages with no `first_seen` whose covering observation now exists. An input whose format has no parser yet gets status `no_parser_yet` (with the slice that adds it), is not waiting work, and is parsed when a parser appears |
| derive | each product at its current version for every session's current rows | the version and the session change each product row was computed from (`session_facts.version`, `from_changed`) | sessions whose product version differs from the code's or whose `changed` is newer than the product's (late parents, changed children); texts without a vector |
| excise | every `excision` row applied | input and observation rows, stored bytes (`applied_upto`, `excised` observations), index rows and vectors that an excision names by scope, recorded location or digest | anything an excision names that still exists, including after a union of archives; a text scan not at its end; `parse_state` and `parsed` rows whose input is gone |
| publish | when `$XDG_CONFIG_HOME/polylogue/sinex.toml` names a Sinex: every local revision's bytes, archive row and excision | what Sinex reports holding, asked at the start of each publish pass | anything Sinex does not hold (`decisions/sinex.md`); nothing local records what was sent |

Because work is always computed from state, there is nothing to repair and
nothing to lose: if the daemon is killed at any point, the next start
computes the same differences and continues. An empty `index.db` is the case
where every input is "never parsed". Rebuilding `index.db` adds no rows to
`archive.db`: which input a path belongs to is read from `archive.db`'s
observations, not from the index.

In code, the comparison is one pure function, `what_is_out_of_date(state)`,
returning the work for every step; the loop executes it. `polylogue status`
calls the same function, so "inputs waiting" in status is exactly what the
daemon will do next, and one test law covers every rebuild case.

## Collect

- Watches the provider locations from `sources/README.md` with file
  notifications, plus a rescan once a minute to catch anything notifications
  missed.
- Exports are not watched; `polylogue import PATH` sends the file's path to
  the daemon over its socket, which hands it to this step.
- Browser captures and hook events arrive as files in Polylogue's own spool
  directories, watched like any provider location.
- Google AI Studio (Drive) is listed through the Drive API by modification
  time every few minutes when credentials exist.
- Each file is classified against its latest observation by the rules in
  `storage/blobs.md` (observing a file), which also owns the quiet time and
  the full check. New lines of a growing file are stored as pieces as they
  arrive; a live session yields at most one revision and one observation per
  minute (`storage/blobs.md`, pieces as they arrive).
- A location an excision refused is not collected while the same file is
  there (`storage/archive-db.md`, never stored again).
- **Git lookups.** For each commit a session names (in `git commit` output, a
  Codex git record or a pull-request link) and each repository the archive
  has sessions in, the collector reads the repository read-only through
  `gix`, under a deadline, and stores the result as a `git-lookup` input
  record: a commit's message, files and line counts, and the commits whose
  trailers name session IDs. Derive reads only these records, so a repository
  that is rebased, pruned or deleted later changes nothing already stored,
  and a rebuild gives the same links. This is the same pattern as the
  repo-lookup record (`sources/README.md`), which derive likewise joins to
  sessions without re-parsing them.
- Provider databases are snapshotted as `storage/blobs.md` (provider
  databases) says.
- **Process sampling** (slice 2). While the daemon has live roots (harness
  processes named in hook lines of the current boot with no `root_gone` line,
  alive with the same start time; `sources/README.md`, process spool; there
  is no time window), the daemon
  samples those processes' descendants every 10 seconds and writes a line to
  its process spool only when a tree changed; with no live harness process
  nothing is sampled. At start, every root of the previous boot gets
  `root_gone`, and every root of the current boot still marked alive is
  checked once and gets `root_gone` if its process is gone, so a root that
  ended while the daemon was down does not stay alive. Format and rules: `sources/README.md`, process
  spool; cost: `behavior.md`. The spool is collected like any other.
- Regular files are collected, symlinked directories under a provider
  location are followed with cycle detection, and special files are skipped
  and counted in `status` (`behavior.md`, untrusted input).
- A file or folder that cannot be read (permission denied, too many open
  files, a locked database) is retried in the next round; a read failure is
  never recorded as permanent. `status` shows the location as unavailable
  with the error; it is never treated as absent, so its sessions are not
  marked `source:gone`.
- The periodic rescan, not file notifications, notices a stored location that
  is gone and writes an `observation` of class `deleted` (or `back` when it
  reappears), so a file deleted while the daemon was down is still noticed
  and `source:gone` has a time.

## Store

- Bytes reach the blob store before any extent refers to them, in the order
  `storage/blobs.md` gives (crash safety), so a crash at any point leaves
  work this step finishes.
- A new file becomes an input with its first revision. An append adds a piece
  to the current revision chain; a rewrite starts a new chain, stored as a
  delta when that halves it; a database gets a page patch over its base. Every
  change writes an `observation`; every revision is kept.
- Quiet append-only inputs have their small pieces joined; whole-file formats
  are stored on their own schedule (both in `storage/blobs.md`).
- Text and session excisions are applied to new bytes before they are stored
  and before the change is classified, and refused hashes are not stored
  (`storage/archive-db.md`, never stored again). The original bytes are never
  written. Each store that overwrites bytes this way writes, in the same
  transaction, an `excised` observation for the input naming the excision,
  the spans and the pre-overwrite IDs, exactly as excision's step 2 does
  (`storage/archive-db.md`, order). Classification masks excised spans
  (`storage/blobs.md`, excised inputs).

## Parse

- Parsing runs on a pool of threads, one revision chain per task. Interpretation
  is deterministic from the acquired bytes, pinned companion revisions and
  committed parser state. A provider may use a supplied lookup over that
  input/chain/parser-version state and return state changes with its row delta;
  it never reads the mutable original path, clock or network. The same writer
  owns state, output and checkpoint changes. The resume state and checkpoint
  commit with the last required row transaction of their bounded step; an
  earlier row transaction is replayable without making its state visible as a
  later completed checkpoint.
  - *Appends.* A growing chain resumes at `parse_state.parsed_upto` with its
    carried state (`carry`), so an append costs the bytes appended: only
    those bytes are read and parsed, and only the rows they touch are
    written: new messages, the session's fields, blocks that continue a
    model response begun before the cut (numbered after its earlier blocks,
    an addition to that message, never a competing version), and the
    changes the append makes to written rows (an active-path flip, a call's
    outcome), which the writer applies in the same transaction. This applies when the carry is from this
    parser version and the chain's companion revisions, and every other
    input that carries messages of the sessions the new records touch is
    unchanged since those sessions were last written (an older chain of the
    same input, another project folder's copy, an export route; rows from
    hook spools, `history.jsonl` and other event-only inputs never block an
    append). The appended records then merge against the stored messages by
    the ordinary rules: a new message ID inserts, an existing one re-merges
    that message only, so a session fed by several unchanged inputs still
    appends incrementally. A head-rewritten chain starts from the carry of
    the chain it subsumes, with offsets shifted by the first line's change in
    length (`storage/blobs.md`). Otherwise the chain is parsed whole and its
    sessions rebuilt, which gives the same rows (law 2). A parser version change discards the carry and starts over.
  - *Formats that change written messages.* Where later records change
    messages already written (Codex: a request's ID, a running command's
    outcome, an `exec` wrapper's children), the carry holds those unsettled
    messages, and an append writes the session again from the first of them,
    after checking the stored session holds exactly what the earlier runs
    wrote (`sources/codex.md`, what an incremental parse carries forward).
    A session's times are then taken over all its stored messages, never
    widened from an earlier run's value, since a run with no timed message
    yet stores the header's time as a stand-in.
  - *Rewrites.* A rewrite starts a new chain, parsed from its start; older
    chains stay parsed, and records only they hold are merged by the
    ordinary rules. Rebuilding a session reads every chain that contributes
    to it except chains a head rewrite subsumed (`revision.subsumes`), whose
    records are all in the newer chain: an older chain's records come from
    parsing its bytes again, not from rows kept per chain. A rewrite
    therefore costs parsing all of the input's contributing chains, once
    (229 real Codex rewrites: 0.92 GB of new chains, 2.32 GB parsed, 127 s;
    `evidence/codex-rewrite-deltas.md`); later appends to the newest chain
    stay incremental, because the older chains do not change again.
- A parser's path context (a folder, a file-name stem) is the input's
  first path, the path of its first observation, never its current
  location. A rename therefore parses nothing again, and a rebuilt index
  gives the same IDs after a rename as before (`sources/claude-code.md`:
  the native ID is the first observation's stem).
- Parse tasks are admitted by bytes held in memory, not by thread count
  (`behavior.md`, memory): files up to 64 MiB are read whole and count their
  size; larger ones are streamed (records for JSONL, a streaming JSON reader
  for whole-JSON documents) and count their live buffers and assembled records,
  with charges following physical payload lifetime as defined in `behavior.md`.
  A multi-gigabyte file is admitted while small files keep parsing beside it,
  and all workers together hold at most 512 MiB. Bounded output batches go to
  the writer through a channel bounded at [256 MB] of parsed batches, so
  parsing runs ahead of writing only so far. An accepted result larger than
  that capacity must advance through bounded fragments or references to
  retained input/disposable staging, using this same route. It cannot wait
  for an impossible whole-result reservation, exceed the cap, or release
  charges while its payload remains live. Fragment delivery alone does not
  select an incomplete message representation or complete its checkpoint.
  Bounded memory
  is a property of this one route; there is no separate low-memory route.
- **State is bounded work too.** A byte window does not bound an invocation
  that clones or serializes the whole preceding parse state. Growing record-ID,
  parent and open-item maps use addressable disposable state and bounded caches;
  the continuation stores only what is needed to locate its committed state.
  Lookup, mutation, serialization and cleanup are included in the step budget.
  A step cannot acknowledge a state update whose output rows remain uncommitted,
  nor discard state needed by its uncommitted successor. State work resumed
  after a crash is bound to the same acquired input/chain, parser version and
  companion basis; changed bases leave the newer work owed. A lookup, spill or
  state-write failure escapes the parse step and prevents checkpoint completion;
  it is not an unrecognized provider record or proof of EOF. Transient storage
  failures retain retryable work under the existing retry rules. This does not
  change the merge rules or create another durable task queue.
- A session's content digest, where one is needed (a Sinex revision), is
  computed by one function over the parser's output before the writer
  derives anything.
- A single writer thread writes bounded transactions within a session; neither
  a whole session nor an arbitrarily large message defines the work budget.
  It merges what the input says into the session's existing rows (below) and
  never deletes what other inputs contributed. A chain's `parse_state`
  checkpoint (its position, carry and law 120's counts) commits with the last
  row-writing transaction of that bounded parse step. A valid step that yields
  no rows commits its checkpoint/carry/accounting in a checkpoint-only writer
  transaction, bound to the same input/revision and retained extents, after
  verifying the whole acquired window and its declared acquisition anchor.
  An initial full input (including an empty one) uses its full-input/genesis
  anchor; an append continuation requires its known contiguous predecessor.
  This is not a free-standing cursor update. In particular, a supported JSONL
  separator-only append can advance verified byte progress while emitting zero sessions or records;
  it neither fabricates an empty session nor removes prior contributions.
  Zero output from this step does not mark a previously productive chain
  `no_sessions`, failed or refused.
  Keep every raw extent and its order/dependency references even when it has
  no normalized output. Reconstruction, session composition, marker and
  attachment consumers must tolerate such a chain member without requiring a
  `ParsedSession` from it. This does not advance an append past a missing
  predecessor, or any step past an unknown window or nonblank unfinished
  record; the existing partial-tail rules still apply. The acquired/inspected
  end is distinct from the last complete-record checkpoint. An unchanged
  unfinished suffix remains owed but waits for more input, rather than being
  repeatedly reparsed as runnable work or marked complete. The work comparison
  reports that waiting reason; a changed acquisition or newly available
  admission can make it runnable again. Pending row finalization is separate
  work and can continue while a tail waits. If a step yields
  several sessions (a main file and its sidechain), its checkpoint advances only after
  all rows produced by that step are committed. A crash before the checkpoint
  replays the step idempotently, including rows already committed; counts are
  changed by actual row changes, not by replaying an intent. Reaching the stored
  byte length does not set `done` while row finalization remains unfinished.
  Products waiting on a missing parent remain separately stale through
  `from_changed`; they keep the session incomplete without holding its parse
  checkpoint behind completed row work.
- **Appends update incrementally.** For an append, only the new records are
  parsed, and the writer updates in the same transaction only what they
  touch: it places new messages after the session's last `seq` (re-sorting
  only when a record lands before the end), adds their counts, tokens and
  cost to `session_facts`, adds their `tool_call` rows and search entries,
  and adds their cost to each subagent ancestor's `tree_cost_nusd` by the
  difference. A full recompute of a session's order and facts happens only on
  a rebuild of that session (below), a merge from another input that changes
  existing rows, or a session whose stored product version differs from the
  code's (then the update is a full recompute). Appending one record to a
  session with 100,000 messages costs about the same as to a session with
  10, whenever the append is incremental (the rule above).
- When a merge changes existing rows, the writer recomputes that session's
  `seq`, summary fields, `tool_call` rows, the search entries of texts it
  added or no longer uses, facts and lineage marks through bounded steps.
  Each committed message keeps its blocks, tool calls and search entries
  consistent with that message version; the row counters change in that same
  transaction. Whole-session ordering and products may take further steps:
  their version or `from_changed` remains behind until all required changes
  commit. Readers can see mixed old and new message versions during this work;
  the session remains incomplete and aggregates are degraded (`query.md`,
  results). A product it could not finish (a parent not yet present) also keeps
  its older `from_changed`, which is what makes it derive's work.
- **Oversize messages do not remove the bound.** One accepted message can
  contain many blocks, and one below-limit text can make hashing or FTS
  tokenization expensive. Preparing that representation must also yield under
  the writer's work and memory budgets. If replacement requires several steps,
  stage its parts outside the currently selected message representation; keep
  the previous complete representation readable, subject to current removal
  and privacy restrictions, until a bounded atomic selection makes the replacement's blocks, tools, text references and counts
  consistent. This is message-local publication, not an atomic switch of the
  whole session or a second task queue. A blocked prior representation is
  unavailable, never an excuse to bypass those restrictions. Retry and cleanup remain bounded and
  basis-checked. Search visibility follows `query.md`, Full-text search.
  Merely moving one huge text insert into a staging transaction does not bound
  its hashing, tokenization or SQL work; a physical representation must earn
  those bounds before claiming this contract. The existing too-large-for-SQLite
  exception (`storage/blobs.md`) stays explicit; it is not a smaller arbitrary
  record or message cap.
- A large first parse (a 1.6 GB file) consumes about 64 MB per input step,
  through the same path an append uses. Its writer work, including later
  merges and finalization, is split further when needed to preserve bounded
  occupancy (`behavior.md`, memory). The checkpoint never advances past the
  committed rows; a crash redoes only uncheckpointed work, and repeated steps
  do not duplicate rows or counts.
- **Live facts.** When a hook line or process line for a session is
  written, the writer updates that session's live columns (latest event,
  harness process and whether it is alive, open tool call, what it waits on,
  in-progress todo item, live task) in the same transaction
  (`storage/index-db.md`). The live state itself is never written: it is
  computed at read time from these facts and the current time (`model.md`,
  Live state), so no step rewrites sessions as the clock moves.
- `message.first_seen` is the smallest, over every (input, chain) whose parse
  produced the message, of the `seen_at` of the first observation of that
  chain whose revision length covers the message's last record in that chain.
  The writer keeps the minimum as chains are parsed. A message whose bytes no
  revision covers yet has none until the covering observation commits
  (within a minute); the loop fills it then. It survives joining and
  rebuilds.
- **Markers.** The writer replaces the session's rows in `marker` with the
  marker IDs the parse extracts from own, scanned messages, and adds to
  `archive.db` the notes for IDs that do not exist and are not excised
  (`storage/archive-db.md`, inline-marker notes). That is the parse step's
  one write to `archive.db`; notes in an imported Polylogue export are
  written by the import request, not by parsing (`export-format.md`).
- A chain that cannot be parsed gets a `parse_state` status (`refused`,
  `failed`, `no_sessions`) with the reason; `status` shows these. It is
  retried only when its bytes or the parser version change.
- A string inside a span an excision overwrote is parsed as excised, and a
  record touching such a span takes its content-derived IDs from the
  `excised` observation (`storage/archive-db.md`, never stored again).
- Each parse task runs under a panic guard (`catch_unwind`). A parser that
  panics gets status `failed` with the reason, and the pool goes on. Time
  never fails a task that is making progress: a large valid input on a busy
  machine finishes, however long it takes, and its committed steps are kept
  (a wall-clock bound that changes an outcome is a defect: the Python append
  route abandoned any plan that outlasted a 30 s writer hold, so a large
  session could never land, `polylogue-itddh`). A task with no
  observable byte or bounded computation progress while scheduled and able
  to proceed for [5 min] is inspected as stalled. Waiting for IO, waiting
  for admission, caller interruption or worker starvation does not establish
  a parser defect. Such an interrupted task retains its committed checkpoint
  and retryable owed work, including for unchanged bytes and parser version;
  a cancelled read is never interpreted as EOF. Only a reproducible parser
  failure gets `failed` with its reason and record offset, retried when its
  bytes or parser version change. A large single record making byte or
  computation progress is progressing work even before a record completes.
- SQLite `BUSY`, `LOCKED` and `PROTOCOL` results are transient: retried with
  backoff, never recorded as a failed input.
- **No waiting inside a write transaction.** The writer's transaction holds
  only SQLite work: parsing, downloads, Drive calls, embedding requests and
  other network or file IO happen before it opens or after it commits.
- **Fair stepping** (rule R51). Work is taken in rounds across inputs, a
  bounded step each (one session, or about 64 MB of a large file), so a large
  file or a burst of exports never starves the rest; new appends to live
  sessions come before backlog within that round, rather than repeatedly
  draining a hot input before a waiting peer gets a turn. A round takes a
  finite inventory of eligible inputs; later arrivals join the next round.
  This is a work-order guarantee, not a wall-clock latency promise under
  arbitrary load or stalled IO. Unused steps are not saved up: an input with
  nothing pending gives up its step in that round, so an input idle for an
  hour takes one step when it wakes; a single step larger than the step size
  is taken alone and leaves no debt. Writer maintenance (FTS `merge` and
  `optimize`, WAL checkpoints, `ANALYZE`, deleting an excised input's rows)
  runs as bounded steps in the same rotation, never as one long hold, and
  live appends still come first.
- **Writer hold times are reported.** Each writer transaction's duration is
  recorded with its step kind; `status` shows the longest of the last hour,
  and a transaction over [10 s] is logged with its kind and input. These are
  reports, not budgets.
- **Stalls are visible.** Only committed work counts as progress: skipping
  an input because nothing relevant changed is not progress, so a loop that
  only skips cannot look alive. If the loop has work but has committed
  nothing for 10 minutes, `status` says so, naming the step and input, and
  the daemon keeps its systemd watchdog only while it makes progress, so a
  wedged daemon is restarted.

## Sessions built from several inputs

Many sessions draw on more than one input: Claude Code tool output in
`tool-results/*.txt`, a subagent's `.meta.json`, a Codex title in
`state_5.sqlite`, hook events, a ChatGPT conversation seen in three exports
and a live capture. Two rules cover all of them.

1. **Companion files are parsed with their main file.** A provider declares
   which files belong to a main file (Claude Code: the `tool-results/` and
   `.meta.json` files next to a session or subagent file). The parser gets
   the main input and the stored companions it names, and returns one result.
   When a companion changes, its main file is parsed again: `parse_state`
   records the companion revisions each chain was parsed with
   (`storage/index-db.md`), and a companion with a newer revision makes the
   chain out of date. Companions are
   still stored as inputs of their own (`parsed` lists them against the same
   sessions).
2. **Independent inputs merge by identity, in any order.** Exports, captures,
   live files, state databases and hook spools contribute records with stable
   IDs (`model.md`). The writer merges each record into the row with the same
   ID, field by field, by the merge rules in `sources/README.md` (the same
   conversation through several routes). Their tie-breaks depend only on
   content: within one input the later revision, across inputs the higher
   BLAKE3 of the record as canonical JSON without its format's volatile
   fields (`sources/README.md`, rule 2), never the input ID or arrival. Because
   that is a fixed order, writing inputs one at a time
   gives exactly the same rows as rebuilding the session from all its inputs
   at once (the order-independence and rebuild-equivalence laws in
   `plan/testing/laws.md`). Records are only added or improved; a partial input
   never removes another input's messages.

Each message records the input its winning version came from (`input_pk`),
that input's revision chain (`source_chain`) and the byte range of its
records in the chain (`source_start`, `source_end`), plus one
`message_source` row for every other input that carries it (`index-db.md`);
events and
links record their input too, and a block records one only when a field came
from a different input. `read --view raw` reads exactly those bytes.

**Ordering key.** `seq` uses only fixed facts, so it is the same however the
inputs arrived:

1. The provider's own structure: for providers with parent links (Claude
   Code `parentUuid`, ChatGPT `mapping`), tree order along the active path.
   Siblings follow the provider's explicit order where the format has one
   (a `children` list), else record order in the file; timestamps decide
   only where the format gives neither (ChatGPT's `create_time` fallback,
   `sources/chat-exports.md`). Messages whose active-path status is unknown
   follow tree order from the root, siblings by the same rule.
2. Otherwise the **base** is the record order of the most faithful input
   holding any of the session's messages (route order from the merge rules,
   then that input's current chain; between inputs still tied, the one whose
   first record of the session has the higher BLAKE3). Every other message is
   **anchored**: it goes directly after the nearest message before it in its
   own input's record order that is already placed; with none, directly
   before the nearest message after it; with neither, by `at` before the
   first placed message whose `at` is later; with no `at`, at the end. Inputs
   are anchored in route order, then by that same content hash, so the result
   does not depend on arrival.
3. The final tie-break is the message ID.

An event read from the same input as messages is placed by its record
position, like a message. An event from another input (a hook spool line)
goes after the message with the greatest `seq` whose `at` is not after the
event's; with none, before the first message.

**Rebuilding a session.** A rebuild covers the whole session in bounded steps in these cases, and in
no others: an append whose incremental preconditions fail (Parse, above); a
newer companion revision of one of its inputs; a Codex record that would
change a settled message (`sources/codex.md`, what an incremental parse
carries forward); a parser version of one of its inputs' formats changes;
or an excision removes or changes one of its inputs. The rebuild re-merges every input listed for it
in `parsed` over its existing rows: rows the re-merge produces replace rows
with the same ID (each message, event, link, run, delivery and process row
carries the `input_pk` and `parser_version` that produced it; blocks and
tool calls go with their message),
so readers see each retained message with its consistent old or new rows;
required messages do not disappear temporarily. Once every contributing
input reaches its stored length, obsolete rows (still at the old version of
the bumped format, or from a removed input) are deleted in bounded steps,
and order, facts, search entries and the children's lineage marks are brought
up to date through the same writer. Finalization yields like ingestion: the
last transaction does not delete or recompute an entire huge session.
Until its row work finishes, the affected `parse_state` stays `parsing`, even
at the stored byte length. A crash resumes from the committed checkpoint and
remaining row/version differences. `done` is committed only after this work;
products still waiting on a missing parent remain derive's work through their
older `from_changed`. Until both parse and required products are current the
session is *incomplete* (`query.md`, `up_to_date`). `status` shows
"re-parsing (version N→N+1), x of y bytes" and says when bytes are parsed but
row finalization or derivation is still pending. Each read observes committed
transactions; no whole-family visibility switch or copy of all Index rows is
required.

## Derive

Derived products (`analysis.md`): session facts (tokens, cost, tool counts,
display title, lineage root, tree cost), lineage marks (`inherited_from` for copies recognisable only by a parent's `uuid`, `counted_as`,
pending links resolved when a parent arrives), repository identity from
stored repo-lookup records (`sources/README.md`), git correlation from stored
git-lookup records, the known-preamble set, embeddings (when enabled).
`engine` reads and writes
through `store` and holds no SQL. Each product is one variant of
`engine::derive::Product`, computed by a `match` on it, so every derived value
has exactly one computing function (rule R27).

- **Staleness is a comparison, not a list.** All per-session products share
  one version, `meta.facts_version` (it includes the hash of `prices.toml`
  and every product's code version); each `session_facts` row records the
  `version` and the session `changed` it was computed from, and the products
  of a session are stale together when either differs. When the writer
  stores a `git-lookup` record it bumps `changed` on every session naming
  one of its commits or named by one of its trailers. There is no table of
  pending work. The writer computes facts and lineage marks when it writes a
  session, so derive has work only where the writer could not: a parent
  arriving after its children (the writer bumps the children's `changed`).
  A copy's author is such a mark: writing a session sets it for the copies
  the session holds and for copies elsewhere of the messages it adds, bumping
  those sessions' `changed`, through `message_session` and `message_copy`;
  no step updates `message` as a whole (a whole-table scan took 4–34 s per
  round on 268,000 real messages, `evidence/real-converge.md`).
- **A product version change** (the facts version includes the hash of
  `prices.toml`, so editing prices recomputes every session's cost: about 50k
  rows, a second or two) makes every session whose stored version differs
  derive's work, taken in `pk` order. No cursor is kept: a restart, a second
  edit or a revert during the walk changes only which sessions differ, so
  the walk always ends with every session at the code's version. When a
  recompute changes a session's own cost, its subagent ancestors' `changed`
  is bumped in the same transaction, so a parent recomputed before its child
  is recomputed again; the result does not depend on `pk` order.
- **The known-preamble set** (`model.md`, display title) is a product of the
  whole archive, not of one session. Its current set is stored in `meta` as
  `preamble_set` (JSON); when a round computes a different set, it writes the
  new one and bumps `changed` on every session whose first human message
  starts with an added or removed preamble. Every other product is a pure
  function of one session's rows and its linked sessions' facts.
- **Embeddings** are keyed by text hash and model in `embeddings.db`, so
  their work is the texts `analysis.md` says are embedded (human prompts and
  model prose, by their `text` rows; current notes, by the hash of their
  value), never tool output or thinking, with no vector and no
  `embed_failure` row for the configured embedding key, not excised, and
  never a text flagged `looks_like_secret`. The writer connection attaches `embeddings.db` for
  this query; `up_to_date` ignores texts excluded this way.

`session.changed` rules are in `storage/index-db.md` (change counter).
Deliveries and runs (`model.md`) are not a separate step: they are parsed
from inputs like any other rows, deliveries from the hook spool's `delivery`
lines and runs from the harness state that records them (`sources/README.md`).

## Publish (Sinex-backed mode)

Only in `primary`, with Sinex configured in `$XDG_CONFIG_HOME/polylogue/sinex.toml`
(`decisions/sinex.md`). `off` never runs an automatic publisher.
The publish step asks Sinex which pieces, archive rows and excisions it holds
(by content hash and ID) and sends what is missing. Nothing local records what
was sent, so nothing can fall out of step. Publishing never blocks reads.

- `primary` (Sinex as the source of truth) is allowed only after finite fixed-basis shadow qualification and the
  real-corpus rebuild proof pass (`decisions/sinex.md`, which holds its
  rules).
- An excision is sent to Sinex as a purge and shows as pending while Sinex
  still reports holding the content.
- Sinex unreachable, rejecting or partly confirming is a visible degraded
  state, never reported as success; while it is unreachable the pending
  count is unmeasured (`decisions/sinex.md`, transport).

## The first build and rebuilds

- **Empty archive.** Collect finds every file at every location; store copies
  them; parse works through them in parallel; derive follows. Reads work
  throughout, showing what is done so far, and `status` reports progress
  (`behavior.md` has the target times).
- **`index.db` deleted.** Every stored input is "never parsed", so the loop
  rebuilds it from the blob store. Provider files are not needed, and no
  input, revision or observation is added for a path already known.
- **`INDEX_VERSION` changed** (the schema changed). The daemon builds a new
  `index.db` beside the old one with the same loop, unsynced
  (`storage/index-db.md`). While it builds, the writer writes only the new
  file, live appends included (they come first, as always), and applies
  excision deletions to the old one too. Reads use the old file, frozen at its
  last counter: every result says `up_to_date: false` with the note
  "rebuilding the index: results are as of <time>", and reads also exclude
  what excisions name. The swap (`storage/index-db.md`) happens at the first
  commit after which the new file has no never-parsed input; the old file is
  deleted then. Every
  schema change rebuilds, including purely additive ones: a full rebuild is
  minutes, and one rule is simpler than classifying changes as safe. Revisit
  only if measured rebuild time hurts.
- **A parser version changed.** Only that format's inputs are re-parsed, in
  the same `index.db`, each session rebuilt over its existing rows (above).
- **A derived product's version changed.** Only that product is recomputed.
- **`daemon --until-idle`** (a batch and test mode, `plan/slice-01.md`) runs
  rounds until no step has work. It does not wait for quiet time: pieces no
  revision covers yet are covered whatever their age, so it ends with every
  revision and `first_seen` written, also under a frozen test clock.
- **Rebuild from Sinex.** When the local archive is empty or lost and
  `$XDG_CONFIG_HOME/polylogue/sinex.toml` names a Sinex, the store step reads inputs and archive rows
  from Sinex; the loop is the same. When both exist, local bytes are used and
  Sinex fills only what is missing (`decisions/sinex.md`).

There is no "rebuild" command. Deleting `index.db`, or bumping a version, is
the rebuild.

## Failed inputs

A failed input has status `failed` and is retried when its bytes or the
parser version change; `looks_like_secret` is computed by the writer. (What
the Python version had instead: `evidence/python-not-carried.md`.)

Evidence: `evidence/architecture-review.md` (path authority, incremental
appends, one revision per minute); `evidence/deep-schema.md` (no
pending-work table); `evidence/conceptual-idempotency.md`,
`conceptual-concurrency.md`, `conceptual-lifecycle.md`, `conceptual-time.md`
(excision as a standing rule, version walks without cursors, rebuilds over
existing rows, pieces before revisions, the ordering key, git lookups as
inputs); `evidence/deep-dataflow.md` (reads never ingest, one digest
function, one bounded route); `polylogue-pt6mo`,
`polylogue-tphz4`, `polylogue-zs70u` (unreadable is not absent);
`polylogue-y2xby` (deletions noticed by rescan); `polylogue-7c5k6`,
`polylogue-wyi9p` (panic guard); `polylogue-3lfvy`, `polylogue-627m3`
(transient SQLite errors); `evidence/beads/closed-5.md`, class J (no waiting
in a transaction); `evidence/beads/closed-3.md`, class B11, and audit
`P10-r1:missed` (the Python scheduler saved up idle shares); audit `P11-A1`,
Bead `de2a` (a 14-minute FTS merge held the writer); audit `P11-A2`, `P11-A9`
(hold times); audit `P10-A6`, Bead `onbz3`, `polylogue-qlae`,
`polylogue-11cg9` (stalls); audit `P29-A5` (changes that move no count);
`polylogue-wmft` (290 GB of superseded generations); `polylogue-jc1b`
(in-place additive changes, declined); audit `P10-r1`, `P10-r2`.
