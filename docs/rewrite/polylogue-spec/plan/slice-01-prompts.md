# Slice 1 worker prompts

The dispatch prompts for workers A–H in `plan/slice-01.md`, written as each
worker would receive them. Writing them first exposed where the spec made a
worker guess (`evidence/meta-readiness.md`); those points are now settled
in the spec and folded into the prompts (what changed since the first draft:
`evidence/plan-history.md`). Nothing is left open at dispatch; a worker that
finds a new gap stops and reports it.

Every worker gets the shared preamble, then its own section.

## Shared preamble (sent to every worker)

```
You are one of nine workers building slice 1 of Polylogue in Rust: every
Claude Code session on this machine stored, parsed, searchable and readable.
The spec is in /realm/inbox/spec/polylogue-spec/ and is the source of truth; when your
code and the spec disagree, the spec wins unless it contradicts itself, in
which case stop and report the contradiction instead of picking silently.

Repository: <REPO> (crates model, hashed-blobs, sources, store, query, engine,
render, polylogue; plus xtask/). Crate boundaries and signatures already
exist with todo!() bodies (plan/slice-01.md, "Crate boundaries"); do not
change a public signature in a crate you do not own. If you need one
changed, report it. Shared files (Cargo.toml files, lib.rs, module lists,
the provider registry, SQL files, deny.toml) belong to the orchestrator.

Rules:
- Edit only the files listed under "You own". Other workers are editing the
  rest of the workspace at the same time.
- Read spec/principles.md and plan/workspace.md ("Rules that keep this
  shape") first. Pure crates (model, sources, query, render) do no IO and
  read no clock or environment. Plain names, no mechanisms beyond what the
  spec describes, no compatibility paths, no repair commands.
- Reuse what exists: the list of existing public types and functions you
  may use is attached below this preamble.
- Run `cargo check -p <your crate>` freely, using
  CARGO_TARGET_DIR=<REPO>/target-<your letter>. Do not run the whole test
  suite; the orchestrator runs it once all workers have finished.
- Test through public entry points (plan/testing.md). No mocks; use real
  SQLite in a temporary directory.
- When done, report: files changed, what passes, and every place where you
  had to choose because the spec did not say (file:line, your choice).
```

## A: data model (`polylogue-model`)

```
You own: crates/model/** .

Read: spec/model.md (all), spec/export-format.md (all), spec/glossary.md,
decisions/lineage.md, decisions/one-message-per-response.md.

The crate already has types, IDs and export-format serde from the skeleton.
Make it complete and correct for slice 1:
- Typed IDs exactly as the skeleton's model::identity defines them:
  SessionId, MessageId { session, local }, BlockId { message, local:
  BlockLocal (Index | Reasoning | Attachment) }, EventId, InputId (hex
  BLAKE3, or a UUIDv7 for a growing input), Origin (open string type with
  constants for the known origins). IDs are opaque when written: code never
  splits one to find its parts. Identity rules from model.md "Identity":
  messages without a native ID get `~<hash>-<n>`, where the hash is BLAKE3
  over the provider's own record as canonical JSON without its format's
  volatile fields and n counts within the input revision; Claude Code
  blocks use one counter over every item of the content list (thinking
  included), across the records of a split response (Codex counts
  reasoning separately, slice 2).
- Session, Message, Block (one enum variant per kind in export-format.md
  "blocks[]", including signature, encrypted data, size, url, other.data),
  Link (evidence from model.md's closed list), Event (with its id), Usage
  (with the cache-write split by lifetime, speed, tier and web search
  count), ToolOutcome { Ok, Error, Unknown { reason: UnknownReason } }
  (no stored Pending: pending is read from Unknown { NoResult } at read
  time), WrittenBy (four values; peer messages are Model with
  from_session), PromptKind, CopyEvidence.
- Serde that matches export-format.md exactly: field order as listed,
  snake_case, absent fields omitted, never `null`, times RFC 3339 UTC with
  milliseconds, content keys as `blake3:<hex>`.
- `ParsedSession` (model::types) and the skeleton's
  sources::framework::Parsed and Unrecognized (what, offset, the record's
  own time); counting into (format, what, count) rows is the writer's
  (store::writer::UnrecognizedCount, with first_at and last_at).
- Session's `models` map and `inputs` list are computed by the writer, not
  parsed; the model types carry them as plain fields.
- Tests: the model.md worked example round-trips byte for byte against its
  JSON; one test per row of the Identity table; the schemars JSON Schema
  generated and checked in at docs/export-format.v1.schema.json, with a
  test that it is current.

Do not touch: any other crate.

Done when: cargo test -p polylogue-model passes and every model.md and
export-format.md field has a Rust field.
```

## B: Claude Code parser (`polylogue-sources`)

```
You own: the sources modules framework (frame.rs), the crate root (Provider,
provider, Format, FileRole), claude_code (wire.rs, map.rs, group.rs and
their unit tests).

Read: spec/sources/README.md and spec/sources/claude-code.md (all of it,
including "what an incremental parse carries forward"), spec/model.md,
spec/convergence.md "Sessions built from several inputs",
decisions/lineage.md, decisions/one-message-per-response.md,
evidence/claude-code-real-data.md, examples/raw/claude-code/*, the Morrow
files in examples/archive/claude-code/, examples/archive/README.md.

Implement the Provider trait for Claude Code (signature in slice-01.md):
- locations(): ~/.claude/projects (never its `.git` directory).
- classify(path): session file, subagent file, .meta.json, tool-results
  side file, sessions-index.json (parsed: titles, title source `index`), a
  stored-only file (workflow records and journals, memory files:
  FileRole::StoredOnly, kept with no_parser_yet / none planned); anything
  else is Unrecognized. Each kind's input.format string is in
  spec/sources/claude-code.md.
- companions(main): the tool-results side files and the .meta.json that
  belong to a session or subagent file.
- parse(main, companions, carried) -> ParseOutcome, implemented by the
  shared framework, which frames records and calls the provider's
  `record()` once per record and `finish()` at the end (rule R88; the
  prototype's `frame` module and `claude_code::Records`):
  - Flat wire structs with optional fields, then a `match` on the record
    type (plan/workspace.md, "Parsing style"); `serde_ignored` reports
    skipped fields into Unrecognized; record type names are a strum enum.
  - The file is the session; its native ID is the file name. Records
    carrying another sessionId give a `continuation` link with `how: resume` (sources/claude-code.md).
  - A subagent file `agent-<id>.jsonl` is session `claude-code:<owning
    session id>/<agent id>`, the owning session ID taken from the `sessionId`
    its records carry. An older-layout sidechain is `claude-code:<session
    id>~side-<root uuid>`. A nested subagent's parent is `claude-code:<session
    id>/<parentAgentId>`. Sibling copies are named by `<session id>/<agentId>`.
  - Split responses: records sharing message.id + requestId are one
    message, ID = the first record's uuid, usage from the last record.
  - written_by by the rules in the `user` row, in order (typed slash
    command human, its expansion harness; isApiErrorMessage harness; peer
    model with from_session).
  - Tool call pairing by tool_use_id; outcome and exit code rules; side-file
    output. A call with no result is stored as `unknown` with reason
    `no_result`; `pending` is computed at read time from live state
    (model.md, tool outcome; law 102).
  - Order from the record tree and file order, not timestamps.
  - Subagent parent from the folder plus .meta.json (toolUseId names the
    spawning call); fork-context-ref when present.
  - progress records: `agent_progress` and `hook_progress` as the mapping
    table says; the other subtypes counted under `unrecognized` (counted
    only) with their payload not decoded, by reading `type` and `data.type`
    first. A bad record loses only itself; a trailing line without a newline
    is not parsed.
  - Incremental: with `carried`, parse only the new complete lines;
    finish() returns a ParserOutput holding the parsed rows and the new
    Carry (the prototype's RecordsOutput holds its state). An append that
    continues a stored response returns its new blocks as Continued
    (numbered after the earlier blocks; the carry holds each open
    response's next block index), and changes to written rows (an
    active-path flip, a call's outcome) as RowUpdates. After a head rewrite
    the carry comes from the subsumed chain with offsets shifted by the
    first line's change in length; re-read only the first record. The
    result must equal a full parse.
- Do NOT set inherited_from. Report each message's copy_evidence from the
  record itself: a sessionId differing from the file's (with the record's
  original uuid), an agentId differing from the file's, forkedFrom
  .messageUuid. The writer (worker C2) sets inherited_from from it.

Must pass: the parser example directories in examples/sources/claude-code/
(worker H writes them; until then, the Morrow files must produce the counts
in examples/archive/FACTS.md for #1–#4, as check_facts.py derives them).

Do not touch: other providers, the store, the engine.

Done when: every Morrow Claude Code file parses to FACTS.md's counts, tool
outcomes and links, and the crate does no IO (clippy.toml enforces it).

```

## C1: blob files and schema (`hashed-blobs`, `polylogue-store` schema)

```
You own: crates/hashed-blobs/**, crates/store/src/schema.rs (the skeleton's
store::schema: it applies the SQL files; the files in crates/store/sql/ are
the orchestrator's, copied verbatim from spec/storage/ by
crates/store/sql/copy-from-spec.py).

Read: spec/storage/blobs.md, spec/storage/archive-db.md, spec/storage/index-db.md
(slice 1 creates the whole schema of all three files, rule R63; it writes
only the tables plan/slice-01.md lists), evidence/blobstore-design.md (API
and laws).

Implement, with the signatures in slice-01.md:
- hashed-blobs: seekable zstd files named by BLAKE3 of the uncompressed
  bytes, temporary name, fsync, rename; hash checked on every read. No
  Polylogue dependency.
- schema.rs: apply the SQL files, record the schema versions in `meta`,
  refuse an archive whose schema is newer than the binary's, and report an
  index version mismatch as "rebuild index.db".

Must pass: the schema test (DDL applied to in-memory SQLite and compared
with spec/storage/: every table of all three files); the
blob-store laws in plan/testing/laws.md for the file layer (crash at any
step, corruption reported).

Do not touch: inputs.rs, archive_meta.rs, writer.rs, search_index.rs, readers.rs
(worker C2), parsing, query
compilation, rendering.

Done when: the tests above pass.
```

## C2: archive, reader and writer (`polylogue-store`)

```
You own: crates/store/src/{inputs.rs,archive_meta.rs,writer.rs,search_index.rs,readers.rs}
(the skeleton's modules of those names).

Read: spec/storage/ (all), spec/convergence.md (all),
decisions/raw-retention.md, decisions/merging.md,
decisions/storage-files.md, evidence/blobstore-design.md (API and laws),
spec/export-format.md (the store must return every field).

Implement, with the signatures of skeleton/crates/store (the authority;
slice-01.md summarizes them), seeded from prototype/crates/store/src/inputs.rs
and write.rs:
- inputs: first_batch (each item's location, format, Stat and bytes; the
  blob rows inserted as `writing` first, files written with one sync before
  and one after their renames, then every input, first revision, extent and
  observation in one transaction, as the prototype's Inputs::first_batch;
  small blobs inline in archive.db, large ones as files through
  hashed-blobs), append_piece and cover_pieces (pieces as they arrive, a
  revision and observation at most once a minute), head_rewrite (a new
  chain: a new first-line extent plus the previous chain's blob ranges,
  shifted), rewrite, and observe for renamed, deleted and back; every
  producing step writes its observation with the full Stat in the same
  transaction; read_range for a byte range of a chain; blob row states
  after a crash; revision (with local_seq), extent and observation rows. archive.db only gains rows except `meta`,
  `blob`, `extent`, excision, and the one-time set of a NULL
  `revision.blake3` (archive-db.md, Rules); `setting` is insert-only.
- readers: open_read (index.db read-only, archive.db attached) and read(Sql),
  with everything export-format.md needs.
- writer (daemon only, the WriterHandle channel): merges each record into existing rows
  by ID, field by field in the fixed order of decisions/merging.md, never
  deleting other inputs' rows; text stored once with an explicit integer
  key, text_fts rows inserted and deleted explicitly (contentless FTS5,
  spec/storage/index-db.md); update seq, summary fields and tool_call rows
  for what was appended, without rereading the session (law 57); write the
  chains' events; apply an append's Continued blocks (additions to the
  stored message, never a competing version) and RowUpdates; set
  inherited_from from each message's copy_evidence
  (decisions/lineage.md, even while the source is absent) and the facts (session_facts,
  session_model_usage, message cost), with `session_facts.version` and
  `from_changed` set only when current (spec/convergence.md, parse);
  bump the write counter in meta and session.changed, also on linked child
  sessions; write parsed, parse_state with law 120's counts (an append adds,
  a whole parse replaces; checkpointed per bounded step as defined in
  spec/convergence.md, Parse), unrecognized (per input: an append adds, a whole parse
  replaces), file_cursor (a cache of the observations), and the derive
  step's Derived marks.
- Session rebuild over existing rows (`convergence.md`, rebuilding a
  session) for parser-version bumps and excision. parse_state has one row
  per revision chain, recording the committed checkpoint and pending row
  finalization (spec/storage/index-db.md and convergence.md, Parse); "more complete" and the tie-break are in
  sources/README.md ("The same conversation through several routes").

Must pass: laws 18, 81 and the blob-store laws in plan/testing/laws.md
(proportional work, append equivalence, crash at any step, rewrite
detection, corruption reported); write -> read -> export round trip on the
model.md worked example; the Morrow Claude Code sessions written in two
input orders give identical exports.

Do not touch: hashed-blobs and schema.rs (worker C1), parsing, query
compilation, rendering.

Done when: the tests above pass and nothing in the store calls a parser.
```

## D: query language (`polylogue-query`)

```
You own: crates/query/** .

Read: spec/query.md (all, including "Results"), spec/storage/index-db.md (index.db
DDL and indexes), decisions/query-parser.md, spec/surfaces/cli.md
("Grammar" and the reference rules).

Implement:
- A pest grammar transcribed from query.md "Grammar" (the full grammar, so
  every syntax error is real); caret error messages written by hand.
- parse(text) -> Query; errors carry kind, message, byte position, a
  suggestion (nearest field name), `hint` and `retryable`.
- Field tables (plan/workspace.md, "Field tables, not macros"): one const
  table per unit with name, kind, SQL expression, description, reference
  evaluator accessor and the slice that adds it. Slice-1 fields: text
  search, origin, kind, id, cwd, branch, repo, model, since/until,
  started/last_active comparisons, messages, human_prompts, tool_calls,
  errors, input_tokens, output_tokens, cache_read_tokens,
  cache_write_tokens, cost (query.md gives their slices), has:, sort,
  limit, page_size and cursor. Any other field returns an error naming the
  slice that adds it.
- compile(q) -> Sql (text and parameters) against index.db: filters before
  ranking, bm25, full-text hits mapped to blocks through block.text_id,
  inherited and inactive rows dropped by default, snippets as query.md says;
  stable paging with opaque cursors.
- A small in-memory reference evaluator (test only) for the same subset.
- A field from a later slice gives error kind `not_yet_available`; JSON
  times are UTC with `Z`; `count` is exact except for `near:` (query.md,
  results).

Must pass: query.md examples 1, 2, 10, 11, 12, 26, 28 and 35 on the
Claude-Code-only home, with the rows in examples/archive/expected/slice-1/expected.json; reference-model agreement on generated
query trees (plan/testing.md).

Do not touch: store internals beyond reading the DDL; rendering.

Done when: the examples pass and every slice-1 field compiles.

```

## E: convergence loop (`polylogue-engine`)

```
You own: crates/engine/** .

Read: spec/convergence.md (all), spec/behavior.md (Concurrency, Failures,
targets, Sources that delete data), decisions/raw-retention.md,
decisions/lineage.md, spec/analysis.md (cost rule, session_facts),
plan/testing/laws.md.

Implement Engine::converge_once(now) -> Result<Progress> and
Engine::converge_until_idle (skeleton engine::converge), used by
`polylogue daemon --until-idle` and the law tests. The engine contains no SQL; it calls store.
- collect: walk each provider's locations (regular files only), skip files
  whose `(device, inode, size, mtime_ns, ctime_ns)` match `file_cursor`,
  classify with the provider, and decide each file's change class
  (storage/blobs.md, observing a file: the five-field key, the 50 ms racy
  guard, the sampled windows, `touched`, head_rewritten for a mutable first
  record). Collect the repository lookups of the working directories the
  index holds (Collector::repo_lookups; spec/sources/README.md, the
  repo-lookup record: the nearest `.git` walking up, read with gix, no
  subprocess; a directory that does not exist on this host has none), and
  the restored hook-spool and capture files (stored, no_parser_yet).
- inputs whose format has no parser in this build get parse_state
  `no_parser_yet` (with the slice that adds it); they are not waiting work
  for `--until-idle` and are parsed when a parser appears.
- store: by the change class: new paths through first_batch in batches of
  at most FIRST_BATCH_FILES files or FIRST_BATCH_BYTES bytes; an append
  through append_piece, covered by cover_pieces at most once a minute; a
  head rewrite through head_rewrite (a new chain reusing the previous
  chain's blob ranges); a rewrite through rewrite (deltas and joining pieces
  come in slice 2). The prototype's check_growth/store_complete are its
  older names for these steps.
- parse: inputs whose parsed/parse_state is missing or older than the
  provider's parser version: parse on a rayon pool with companions and the
  stored Carry, write on one thread in bounded session steps
  (spec/convergence.md, Parse).
- derive: only what the writer could not finish in the session's own
  transaction: sessions whose `session_facts.version` differs from the
  code's facts version (prices hash included) or whose `from_changed` is
  older than `session.changed` (a late parent, a changed child), and copies
  recognisable only by a parent's uuid or by a re-emitted response's
  message.id + requestId once that parent arrives
  (model::lineage::inherited_by_uuid), spawning calls resolved to the
  parent's block, and sessions joined to repository lookups; all sent to
  the writer as store::writer::Derived marks. The writer (worker C2) sets
  inherited_from from copy_evidence and computes the facts product (session_facts, session_model_usage,
  message cost from prices.toml) when it writes a session.
- The clock is an argument; excisions are honoured at store and parse.
- Progress counts as the --until-idle summary needs (plan/slice-01.md,
  Daemon details, Summary).
- Daemon details are in plan/slice-01.md: flock on daemon.lock; idle = no
  work from what_is_out_of_date(&reader.state()) and no file changed during
  the last pass.

Must pass: the slice-1 laws as plan/slice-01.md lists them (idempotent
ingest, append equivalence, order independence, rebuild equivalence, crash
safety, identity stability, nothing lost, nothing invented, paging, and the
lineage laws 4 and 11 (copies and late parents), 29–34 and 99 in
plan/testing/laws.md) on the Claude-Code-only home.

Do not touch: parser internals, SQL, rendering.

Done when: the laws pass and an empty archive converges on the Morrow
fixture to FACTS.md's numbers.

```

## F: rendering (`polylogue-render`)

```
You own: crates/render/** .

Read: spec/rendering.md (all), spec/surfaces/cli.md (output rules, short
IDs, and the examples: bare polylogue, search, read, summary, status),
spec/export-format.md.

Implement render_list, render_session (views transcript, summary,
dialogue), status and the overview, in terminal and Markdown formats, and
export-format JSON: message headers, tool lines (outcome first, exit code,
duration, first failing output line), collapsed thinking, subagent call
lines with child ID and message count, the resume line for inherited
prefixes, branch and compaction indicators, `[excised]` blocks, ranges
(--from, --last), short IDs, local times, unknown shown as "—", control
characters escaped. Pure: the clock and time zone are arguments. The
column rule is in surfaces/cli.md (output rules); values that differ
between runs are `[..]` in expected outputs (plan/slice-01.md, "Daemon
details").

Must pass: the slice-1 `spec/surfaces/` examples as trycmd tests (worker H wires
them, recorded against the Claude-Code-only Morrow subset); snapshot tests
of each view for Morrow #2 and #4.

Do not touch: query, store, CLI argument parsing.

Done when: the examples render byte for byte.

```

## G: socket, command line and MCP (`polylogue` binary)

```
You own: crates/polylogue/src/{main.rs,daemon.rs,cli.rs,mcp.rs} (the skeleton's
modules; the socket server is in daemon.rs).

Read: spec/surfaces/README.md (processes, results and errors),
spec/surfaces/cli.md (the authoritative command table and reference rules),
spec/surfaces/mcp-and-hooks.md (MCP, with the tool descriptions written out
in full), spec/query.md "Results", decisions/process.md, spec/behavior.md
(Concurrency).

Implement:
- The socket (daemon.rs): the daemon listens on its one owner-only Unix
  socket, newline-delimited JSON, and answers the slice-1 `Request` variants
  (plan/slice-01.md signatures) through engine, query and render; one
  request shape, no per-command operation registry. Clients send one
  request and render the response; when nothing listens they print the
  one-line "not running" message naming `polylogue daemon`, exit with code
  6, and open no database.
- With clap, every command and verb whose slice column in cli.md says 1:
  `polylogue` (overview), `status`, `import` (files and directories,
  `sources/README.md`, importing a directory), `QUERY [VERB]` with verbs list, read and export,
  `find QUERY …`, `explain`, `input cat ID`, `daemon [--until-idle]`, `mcp`.
  Argument splitting and reference resolution (bare native IDs, provider
  URLs, `@N`, `last`) exactly as cli.md says.
- --json / --jsonl on every command; exit codes as cli.md lists; the stderr
  line when the index is not up to date, and `up_to_date` in JSON.
- `polylogue mcp` over stdio with rmcp 3.x (MCP revision 2026-07-28), a
  proxy to the socket: tools `search` and `read` only in slice 1, with the
  spec's descriptions and argument schemas, `next` when truncated.
- The clock is read once per request by the daemon (POLYLOGUE_NOW in
  tests); the archive is at $XDG_DATA_HOME/polylogue or --archive.

Must pass: the CLI parts of the slice-1 trycmd examples; an MCP round trip
(search then read) against the Morrow fixture; laws 83 (socket and library
give the same rows) and 84 (daemon stopped: one-line message, no database
opened).

Do not touch: everything below the binary; call engine, query, store and
render through their public functions.

Done when: the demo commands in plan/slice-01.md run against the Morrow
fixture and against a real ~/.claude/projects copy.
```

## H: examples and test harness

```
You own: examples/sources/claude-code/**, tests/** (end-to-end tests,
trycmd files, the law harness), xtask/src/** for `xtask test`,
`xtask test --sample`, `xtask golden` and `xtask compare-old` (which runs
`xtask/compare-old/dump_old.py`, copied unchanged from `plan/scripts/`,
against the old repository and compares, as plan/compare-old.md specifies).

Read: plan/testing.md (all), examples/archive/{FACTS.md,README.md,build.py,
check_facts.py}, spec/sources/claude-code.md, the spec examples named in
plan/slice-01.md "Acceptance".

Implement:
- Parser example directories: every `examples/sources/claude-code/…`
  directory named in plan/testing/real-data-cases.md and
  plan/rewrite-rules.md, plus: basic session, split response, subagent
  with .meta.json, compaction, resume with an inherited prefix, background
  task pending then completed, side-file tool result, side file missing,
  truncated last line, a corrupt middle line, duplicate uuid, unknown record
  type, a branch (retry), a typed slash command, a peer message. Inputs are
  synthetic, or keep a real file's structure with every text, path and ID
  replaced;
  expected.json in the export format, checked by hand against
  sources/claude-code.md. `export` is in slice 1; real files are never
  copied into examples (an example may keep a real file's structure, with
  all content invented or replaced); placeholders are `[..]`.
- Fixture::claude_code_only() (plan/testing/laws.md, common parts: the
  Claude-Code-only home that build.py writes for slice 1) and the converge
  helper. Fixture::morrow() is the full archive, used from slice 2.
- trycmd cases taken from spec examples marked `<!-- test: <id> slice:1 -->`,
  run with POLYLOGUE_NOW and TZ=Europe/Warsaw.
- The slice-1 laws as test functions, comparing the archive view built from
  public output; terse output with details written under target/.

Must pass: nothing yet (the product is being written in parallel); the tests
must compile and fail only on todo!() panics.

Do not touch: product crates.

Done when: every acceptance item in plan/slice-01.md has a test that checks
it. Slice 1's expected values are
examples/archive/expected/slice-1/expected.json.

```
