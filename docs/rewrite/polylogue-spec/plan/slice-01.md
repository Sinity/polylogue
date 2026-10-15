# Slice 1: Claude Code, end to end

The first thing the operator can run on real data: every Claude Code session
on the machine, stored, parsed, searchable and readable from the terminal and
from an agent.

## What the operator will run

```console
$ polylogue daemon --until-idle          # first build; exits when caught up
collecting  ~/.claude/projects  16,047 files  19.7 GB
storing     16,047 / 16,047
parsing     16,047 / 16,047   (12 refused, see status)
done in 4m 12s

$ polylogue                              # status overview
$ polylogue chrono                       # search
$ polylogue claude-code:7f3a2c10-0002 read
$ polylogue claude-code:7f3a2c10-0002 read --view summary
$ polylogue 'origin:claude-code since:7d' list --json
$ polylogue status                       # refused files, unrecognized record types
```

and, from Claude Code with the MCP server configured by hand for now:

```
search {"query": "chrono"}
read   {"ref": "claude-code:7f3a2c10-0002", "view": "dialogue", "last": 4}
```

The numbers are illustrative; record the actual real-data result. These are
full-slice operator commands, not bootstrap instructions. Candidate demonstrations
use an explicit isolated `--archive DIR` and selected staged sources
([demonstration route](../../polylogue-sinex-joint/realization/demos/README.md)).
Restoring old spools to live directories is a separate operation under
[the migration procedure](migration.md), after its inventory and collision checks;
no copy into live state is required to build or test the candidate.

## In scope

- **Old archive data back in place** (`plan/migration.md`): the capture and
  hook spool files the reset set aside are staged for an isolated real-data
  build; live spool restoration is a separate scoped operation. Inventory retained
  revisions and compare colliding files by digest before copying; preserve differing
  bytes as distinct inputs, and never treat a skipped copy as successful restoration
  (`evidence/old-archive-detail.md` records the historical paths). Convergence
  then stores them like any spool file. Only Claude Code inputs are parsed in
  this slice. The rest (captures, hooks, Codex) get `parse_state` status
  `no_parser_yet` with the slice that adds their parser: they are not
  waiting work, so `--until-idle` finishes and queries report
  `up_to_date: true`; `status` lists them as "stored, parser arrives in
  slice N", grouped by format, status and reason with a count (about 944,000
  restored spool files are one line, not 944,000; `spec/surfaces/cli.md`,
  status --json). Worker E collects them; their format is the
  `HookSpool` variant of `sources::Format`. Codex's pre-rewrite rollouts and AI Studio follow in slices 2
  and 5.
- **Collect and store** (`spec/convergence.md`): scan `~/.claude/projects/`
  for session files, subagent files, `tool-results/*.txt` side files and
  `.meta.json`; store bytes per `spec/storage/blobs.md` (small blobs inline in
  `archive.db`, large ones as seekable zstd files; session logs as growing
  inputs whose appends are new pieces and whose rewrites are new revisions
  stored whole; change classification and observations; blob row states for
  crash safety). Delta encoding and joining quiet pieces come in slice 2.
  Periodic rescan while the daemon runs; `--until-idle` exits when nothing
  is left to do (`decisions/raw-retention.md`). For each session working
  directory the index holds, the collector stores the repository lookup of
  the nearest `.git` walking up from it, when that directory exists on this
  host (`engine::collector::Collector::repo_lookups`; `spec/sources/README.md`,
  repository identity); the derive step joins sessions to lookups by path,
  with no re-parse. Claude Code's `sessions-index.json` is parsed (title
  source `index`). A location or file that cannot be read is retried on
  the next round and shown as unavailable, never as missing; a fresh archive
  with no inputs reports `ok`. Sets of IDs reach SQL through `json_each(?)`
  or a temporary table, never an `IN (...)` list.
- **Parse** Claude Code records into sessions, messages, blocks, links and
  events exactly as `spec/sources/claude-code.md` says, including
  subagents (parent from the folder and `.meta.json`; `spec/sources/claude-code.md`), compaction, tool call pairing and
  outcomes, tool results from side files, `written_by` rules, unrecognized
  record counting, and resume/continuation links. The `inherited_from` marks
  on resumed sessions are computed by the writer, and by the derive step when a
  parent arrives later (`decisions/lineage.md`).
- **Write** `index.db`, one transaction per session, parse on a thread pool,
  one writer. Incremental parsing of appends through `parse_state` and the
  parser's carried state.
- **Storage subset.** Slice 1 creates the whole schema of all three files,
  copied verbatim from `spec/storage/` by `copy-from-spec.py` (the same DDL
  the schema test applies), and writes only the tables listed below; the
  others stay empty until the slice that first writes them. One schema for
  every slice keeps the copy step mechanical and the DDL checked as a whole;
  rule R63 then reads "writes only what its code reads", and a later slice
  starts writing a table by raising `INDEX_VERSION` (a rebuild fills it).
  `archive.db` is migrated, never rebuilt.

  | Database | Tables in slice 1 | Later |
  |---|---|---|
  | `archive.db` | all tables (later slices only start writing them) | — |
  | `index.db` | meta, input_ref, session (live columns left NULL), message, message_source, text, block, tool_call, link, event, parsed, parse_state, unrecognized, file_cursor (a cache; the observations are the authority), session_facts, session_model_usage, `text_fts` and `fts_vocab_text` | delivery, process, repo_command_last and the session's live columns (2); page_map (2); `output_fts`, `fts_vocab_output` and `tool_call_trigram` (3); run, run_session (6); marker, removed_session (7); commit_link, pr_link (11) |
  | `embeddings.db` | none | slice 9 |

  The exact columns are in `spec/storage/`. Slice 1's writer does not write
  `output_fts`, `tool_call_trigram` (slice 3) or `repo_command_last` (slice
  2) even where the prototype's writer already does.
- **Query**: the subset of `spec/query.md` needed by the demo, on the
  `sessions` unit only: text search, `origin`, `kind`, `id`, `cwd`, `branch`,
  `repo`, `model`, `since`/`until`, `started`/`last_active` comparisons,
  `messages`, `human_prompts`, `tool_calls`, `errors`, the four token sums,
  `cost`, `has:`, sort and limit. The full grammar is parsed (so errors are real), and every field
  name is known (so a misspelling gets a suggestion); other units and fields
  return "available in slice N". Every other field's slice is the `Slice`
  column of its row in `spec/query.md`.
- **The socket endpoint**: the daemon answers requests on its one Unix
  socket (`spec/surfaces/README.md`, processes), newline-delimited JSON, with
  the slice-1 subset of the request shape (`decisions/process.md`): a query
  (its text carries unit, sort and limit; the request adds page, cursor,
  fields and token budget), `read`, `export`, `input
  cat`, `explain`, `status` and `import` (the `Request` enum below). Every
  command below is a client of it; with the daemon stopped each prints the
  one-line "not running" message naming `polylogue daemon`, exits with code
  6 and opens no database (law 84). The local HTTP mapping of the same
  requests serves the web reader and arrives with it in slice 10.
- **Surfaces**: bare `polylogue`, `status`, `import`, query + `list`, `read`
  with views `transcript` (default), `summary`, `dialogue` and `raw`
  rendered per `spec/rendering.md` (terminal and Markdown); `--json`
  (`list` and search give query.md's result fields; `read --format json` and
  `export` give the export format, `spec/export-format.md`, version 1, with
  its generated JSON Schema; times in UTC); exit codes; `export` (used by
  the law tests); `input cat`; `explain`; `polylogue mcp`, a stdio proxy to
  the socket, with `search` and `read`. The commands of each slice are the slice column of
  `spec/surfaces/cli.md`.
- **Lineage marks and facts**: the writer computes `inherited_from` and the
  facts product (`session_facts`, `session_model_usage`,
  `message.cost_nusd`: counts, tokens, and cost from `prices.toml` by the
  rule in `spec/analysis.md`; unknown cost shows as "—") in the session's own
  transaction; the derive step recomputes them only for a late parent, a
  changed child or a version or price change (`spec/convergence.md`).
- **Credential masking**: the writer sets the credential flag on matching
  spans (`spec/behavior.md`, privacy); the daemon masks them once when it
  encodes a response, so CLI output, `export` and the MCP proxy are masked
  from this slice; `--reveal` works only from an interactive terminal outside
  a harness (component `bin.masking`). The web reader's masking comes with it
  in slice 10.

Sources: `polylogue-tphz4`, `polylogue-zs70u`, `polylogue-mqups`, `polylogue-1tyq5.187`.

## Not in scope (later slices, and why that is safe)

| Left out | Why it does not paint the design into a corner |
|---|---|
| File watching (notify) | The loop already finds work by comparing state; watching only wakes it sooner. |
| Other providers | `polylogue-sources` has one module per provider behind the same trait. |
| Notes, tags, saved queries | Their tables in both files exist from slice 1 (the whole schema is created then, rule R63) and stay empty until slice 7 starts writing them. |
| Delta encoding, joining pieces, `replaced` and `renamed` classes | Rewrites are stored whole and classified as rewritten; slice 2 adds the cheaper encodings and finer classes without changing what a revision means. |
| `output:`, `command:` and `path:` search | Their indexes are built from stored rows in slice 3; a rebuild fills them. |
| Full query language | Grammar is complete; only field support is partial. |
| Hooks, history.jsonl, pastes | Events are stored generically; hook data joins a session by the writer's merge by ID (`spec/convergence.md`, sessions built from several inputs). Until slice 2 provider-reported paste extraction is unevaluated; clipboard matching is a separate backed-mode capability. No typing or authorship classification follows (`model.md`, Text-origin evidence). |
| Web reader, embeddings | Separate crates/modules; no schema impact (`embeddings.db` is separate). |

## Crate boundaries for this slice

The compiling skeleton (`skeleton/crates/`) records the current component
placement and interface outline. Every type and function named below is a
skeleton path, checked by `skeleton/check_types.py` (a backticked
`crate::module::item` here must exist there). The executable bootstrap
refines placeholder signatures together with their callers, documentation
and checks; copying stubs does not satisfy bootstrap. The spec owns behavior
and `plan/workspace.md` owns dependency and write-capability boundaries.
Promote existing implementations component by component against those
contracts, rather than copying the prototype workspace or preserving its
documented model and dependency departures. This section says what each
part is for and what must hold; the signatures are not repeated.

**Model** (`model::types`, `model::identity`, `model::time`, `model::lineage`,
`model::measures`, `model::merge`).
- One time type, `model::time::Timestamp` (milliseconds since the epoch,
  UTC), with `model::time::ProviderTime` carrying model.md's quality,
  precision and bound. Wire times convert once, through
  `sources::wire::IsoTime`, `sources::wire::EpochSecs` and
  `sources::wire::EpochMillis` (rule R32); an offsetless time is unknown
  unless its format declares a zone (`spec/model.md`, Time).
- IDs: `model::identity::SessionId` is the canonical
  `<origin>:<native id>` string; an ID typed by a person is never split,
  only looked up whole or by unique prefix. `model::identity::BlockId` is
  its message plus a `model::identity::BlockLocal`: an index into the
  content list, a separate reasoning index in a format whose rewrites drop
  reasoning (Codex), or an attachment key (`spec/export-format.md` lists
  the three written forms). `model::identity::EventId` comes from the raw
  record (`spec/model.md`, Identity). `model::types::Origin` is an open set
  of short names.
- `model::types::Message` holds every field of `spec/model.md`, Message,
  that slice 1 parses: `on_active_path` is `Option<bool>` (unknown when no
  marked leaf decides it); `copy_evidence` is what the record itself says
  about its source (`model::types::CopyEvidence`), from which the writer
  sets `inherited_from` even while the source is absent; `attrs` holds the
  fields the parser did not read (rule R11).
- `model::types::Session` has no links: a parser returns links beside its
  sessions in `sources::framework::Parsed`, because a link joins two
  sessions. `model::types::ParsedSession` (a session with its events) is
  the one parsed-session type.
- Stored tool outcomes only (`model::measures::ToolOutcome`): `pending` is
  a reading of `Unknown { NoResult }` computed when asked (`spec/model.md`,
  Tool outcome).
- Money: `model::types::Usd` (nano-USD), never converted to or from
  `model::types::Credits`; `model::types::Ratio` cannot be built with a
  zero denominator. `model::measures::price` is the one cost function (rule
  R34): it reads model, time, speed, tier and cache-write lifetimes from
  the message, returns a `model::measures::Cost`, and totals are sums of
  `Cost`; a provider's own figure is a `model::measures::ProviderTotal`,
  compared with a sum, never added to one.
- `model::lineage::counted_as` applies `decisions/lineage.md`'s counting
  rule, so it takes each copy with its session's first-record time.

**Sources** (`sources::framework`, `sources::claude_code`).
- `sources::Provider` is the prototype's shape: locations, `classify` into
  a `sources::FileRole` (a stored-only file is kept with parse status
  `no_parser_yet`, detail `none planned`), companions (side files, meta
  files, tool results; a repository lookup is not a companion), and `parse`,
  which returns a `sources::framework::ParseOutcome`: parsed with the new
  carry, how far it parsed and law 120's counts, refused (not the format its
  location promised; nothing parsed), or failed at a known offset.
  `sources::provider` is the engine's one entry.
- A provider implements only `sources::framework::RecordParser` (rule
  R88); the framework frames, decodes, counts (`sources::framework::Counts`:
  used + ignored + unrecognized + undecodable = framed, law 120, stored per
  chain in `parse_state`) and builds the outcome
  (`sources::framework::outcome`). No other function returns a parsed
  session.
- The carry (`sources::framework::Carry`) is the parser's state between
  appends (`spec/sources/claude-code.md`, what an incremental parse
  carries forward): `RecordParser::finish` returns a
  `sources::framework::ParserOutput` holding what was parsed and the carry,
  as the prototype's `RecordsOutput` holds its state; it is serialized into
  `parse_state.carry`. An append's `sources::framework::Parsed` also carries
  the changes it makes to written rows: `model::types::Continued` blocks
  (numbered after the message's earlier blocks; the carry holds each open
  response's next block index) and `model::types::RowUpdate`s.
- Each `sources::Format` has its `input.format` string; the Claude Code
  side files, meta files, tool results, stored-only files and the
  repo-lookup record each have their own (`spec/sources/claude-code.md`,
  what is collected).

**Store** (`store::inputs`, `store::schema`, `store::writer`,
`store::readers`).
- `store::schema::create` makes the whole schema of all three files from
  slice 1 (rule R63); `store::schema::check_version` refuses a newer
  archive schema and asks for an index rebuild on a version mismatch.
- First observations (`spec/storage/blobs.md`, batched durability):
  `store::inputs::first_batch` takes each new path's location, format, stat
  and bytes and, as the prototype's `first_batch`, inserts the blob rows as
  `writing`, writes the files with one sync before and one after their
  renames, then inserts every input, first revision, extent and observation
  in one transaction. A round's new paths go in batches of at most
  `store::inputs::FIRST_BATCH_FILES` files or
  `store::inputs::FIRST_BATCH_BYTES` bytes.
- Known inputs: `store::inputs::append_piece` stores new complete lines as
  a piece (synced on its own); `store::inputs::cover_pieces` writes the
  revision and observation covering them at most once a minute;
  `store::inputs::head_rewrite` starts a new chain from a new first line and
  the previous chain's blob ranges; `store::inputs::rewrite` stores a new
  chain whole. Each producing step writes its observation in the same
  transaction, with the full metadata of `store::inputs::Stat`;
  `store::inputs::observe` records renamed, deleted and back.
  `store::inputs::read_range` reads a byte range of a chain, so an append
  reads only its new bytes.
- The one writer (`store::writer`): the engine sends a
  `store::writer::Batch` of unmerged chain parses
  (`store::writer::ChainParse`: where the parse started, whether it is
  whole or an append, the chain a head-rewritten chain subsumes, and an
  append's continued blocks and row updates), parse_state updates with
  law 120's counts, unrecognized counts, cursor rows and derived marks
  (`store::writer::Derived`). The writer merges each session from all its
  sources in bounded steps; checkpoint, intermediate completeness and
  bounded row finalization follow `spec/convergence.md`, Parse and Rebuilding
  a session. Events are the writer's from slice 1.
  archive.db writes go through `store::inputs`, never the writer.
- SQL is `model::types::Sql`: fixed `model::types::SqlText` fragments
  (only a `const` constructor) plus bound parameters (rule R45); query
  makes it, store runs it.

**Engine** (`engine::converge`, `engine::collector`, `engine::import`,
`engine::derive`). No SQL; it reads and writes through store.
- `engine::converge::what_is_out_of_date` reads an `engine::converge::State`
  (built by `engine::converge::State::read` from both stores) and is the
  comparison the loop and `status` share; `engine::converge::Work` lists the
  `engine::converge::Unit`s to do. Uncovered pieces (no revision yet, at most
  a minute old) and joining after the quiet time (15 minutes, slice 2) are
  separate units; `--until-idle` covers pieces at once and never waits for
  quiet time. The `engine::converge::Engine` holds an index reader.
- `engine::converge::Engine::converge_once` is one round;
  `engine::converge::Engine::converge_until_idle` runs rounds until idle.
- `engine::import::import` is `polylogue import` (slice 1).
- `engine::derive::Product` is one derived product with its version;
  `engine::derive::derive_round` reads through the index and writes its
  marks as `store::writer::Derived` in a writer batch.

**Binary** (`polylogue::daemon`). `polylogue::daemon::Request` is the one
request shape, its fields cli.md's (a query with page size, cursor and
fields; read with reference, view, format and every read option;
import with format and wait; later slices add the rest of cli.md's tables,
generated from them); every request travels in a `polylogue::daemon::Envelope`
with `deadline_ms` and, in tests, `now`.

**Query** (`query::grammar`, `query::planner`). `query::grammar::parse`
gives a position and suggestion on error; `query::planner::compile`
produces `Sql`. A result's outcome is a `model::types::ResultOutcome`
(distinct from a tool call's outcome); `Empty` is built only from a read
that succeeded and matched nothing. `query::planner::Cursor` is opaque.

The file layer of the blob store (`hashed-blobs`: seekable zstd files
named by BLAKE3, about 250 lines) has no Polylogue dependency
(`evidence/blobstore-design.md`).

## Work split (coherent outcomes, isolated workers)

Each skeleton module (`skeleton/crates/<crate>/src/lib.rs`, `mod <name>`)
becomes the file `crates/<crate>/src/<name>.rs`, so the split below names
modules, and every module slice 1 needs has one owner.

| Worker | Modules (files) | Model |
|---|---|---|
| A | `model::{types, identity, time, merge, lineage, measures, export}` (identity rewritten from `spec/model.md`, not seeded from `ids.rs`) | Opus |
| B | `sources::framework` (the per-record framework, `frame.rs` in the prototype; law 120's accounting), the `sources` lib (`Provider`, `provider`, `Format`, `FileRole`), `sources::wire` (the wire time types) and `sources::claude_code` (the parser, its carry and the copy evidence each record names), with the parser's example tests; `sources::links` (resolving pasted references) is slice 6 | Opus |
| C1 | `crates/hashed-blobs/**`, `store::schema` (applies `crates/store/sql/*.sql`, the whole schema of all three files (copied in the signature step by `copy-from-spec.py` from `spec/storage/archive-db.md`, `index-db.md` and `embeddings.md`; version checks) | Haiku, from the exact DDL and signatures |
| C2 | `store::{inputs, archive_meta}` (first_batch with its `writing` rows first, pieces and their covering revisions, head rewrites as new chains, observations in the same transaction, read_range, `cksumvfs` registration) and `store::{writer, search_index, readers}` (the writer thread merging sessions from all sources, setting `inherited_from` from copy evidence, writing events, facts and counters; read connections) | Opus |
| D | `query::{grammar, planner}` (grammar, parse, plan, SQL for the slice's fields) | Opus |
| E | `engine::{collector, converge, import, derive}` (the change classes including `head_rewritten`, the loop and `--until-idle`, import's recognition, the derive step's marks: late parents, re-emitted responses, spawning calls, repository lookups joined by path, price recomputation; collecting the lookups and the restored spool files) | Opus |
| F | `render::{terminal, formats}` (terminal, Markdown and export JSON for list/read/summary/dialogue/status) | Haiku, from the spec's output examples |
| G | `polylogue::{daemon, cli, mcp}` (the socket endpoint and its clients) | Haiku, from `spec/surfaces/` |
| H | `examples/sources/claude-code/*` expected outputs, trycmd example files (from `examples/archive/expected/slice-1/`), `xtask compare-old` | Haiku, reviewed by Opus |

The product lead allocates coherent outcomes after an executable bootstrap exists. Independent workers
use isolated worktrees; the lead/integrator owns shared files and landing. A shared checkout is an
exception with enforced prohibition of worker Git/index mutations (`plan/method.md`). Every packet
names the actual candidate/interfaces; do not dispatch A–H simultaneously from todo!() signatures.

Three concerns sit outside A–H: separately scoped restoration of preserved spools, CI/build consumers,
and shared contract/module integration. Source replacement is not permission to restore or delete
state (`plan/migration.md`). A fixed milestone client scenario has an evaluator; authors retain ordinary
correctness review and scoped verification. Bootstrap, first useful path and full slice acceptance are
separate outcomes; finish each connected path before widening.

Sources: `polylogue-ux8oj`, `polylogue-qhn6h`.

## Daemon details

- **One daemon per archive.** The daemon takes an exclusive `flock` on
  `<archive>/daemon.lock` at start; a second daemon finds it held and exits
  with a message naming the running one (pid written in the file). The lock
  is released by the kernel when the process dies, so a crash leaves nothing
  to clean up.
- **Idle.** `--until-idle` exits when `what_is_out_of_date` over a fresh
  `State::read` returns no unit and no provider file changed during the last
  round. It is a batch and test mode, so it does not wait for quiet time:
  pieces no revision covers yet are covered before it exits, whatever their
  age, so it terminates under a frozen test clock (`POLYLOGUE_NOW`) with every
  revision and `first_seen` written (`spec/convergence.md`, the first build).
- **Summary.** `daemon --until-idle --json` ends with one JSON object:
  `{"inputs_stored", "inputs_parsed", "sessions_written", "transactions",
  "per_format": {FORMAT: {"inputs_parsed", "records_read"}}, "per_input":
  [{"path", "records_read"}], "elapsed_ms"}`; the terminal form prints the
  lines shown above.
- **Test clock.** Built with the test-only `fault-injection` feature, a
  request may carry `now` (RFC 3339), and the daemon uses it instead of its
  clock for that request; `POLYLOGUE_NOW` sets the daemon's default at
  start. The same feature replaces the daemon's monotonic clock (stall and
  stuck limits, round spacing, deadlines) with a test clock that a test
  request advances, so a stall limit is reached by advancing it, never by
  waiting. Without the feature neither exists.
- **Not running.** In slice 1 the not-running message names only `polylogue
  daemon`; the systemd unit and its wording arrive in slice 2. The exit code
  is 6 (`spec/surfaces/cli.md`, exit codes).
- **Changing values in examples.** Expected outputs write values that differ
  between runs (pid, sizes on disk, durations) as `[..]`, trycmd's
  placeholder.

## Acceptance

1. **Examples.** Every spec example carries the slice that makes it pass
   (an HTML comment marker `<!-- test: <id> slice:<n> -->` above a code
   block, or the Slice column of `query.md`'s example table). Slice 1 runs its
   examples against the Claude-Code-only home,
   `examples/archive/claude-code-only/`, which `build.py` writes: Morrow
   sessions #1–#4 plus the lineage set L1–L8 (`examples/archive/FACTS.md`,
   "Lineage set"), known cost **$7.87**. The expected results on that home
   are in `examples/archive/expected/slice-1/expected.json`, recomputed from
   the raw files by `check_facts.py` (so they cannot drift). These pass as
   tests:
   - `spec/model.md` worked example;
   - `spec/query.md` examples 1, 2, 10, 11, 12, 26, 28 and 35, with the rows
     `expected.json` gives for the Claude-Code-only home (for example,
     example 1 there returns L8, L5, L1, #4, #2, #1). Example 34 needs `--why` and
     comes in slice 3; examples 13 and 14 need the `messages` unit and
     claude.ai session #8, and come in slice 4 (their markers in `query.md`);
   - `spec/surfaces/` examples: the read and summary of #2 as recorded
     for slice 1 in `examples/archive/expected/slice-1/read-2.txt` and
     `summary-2.txt` (the same session in both homes; without the `tags`
     line and note marks, since notes come in slice 7, which is why the
     spec's versions carry `slice:7`), the MCP `read` dialogue example, and
     bare `polylogue` and search as recorded in `overview.txt` and
     `search-chrono.txt` beside them, whose numbers `check_facts.py`
     checks against `expected.json` ("overview", "search chrono"); `status` is checked by
     law 46. The spec's versions of those show main-archive numbers
     (`examples/archive/expected/main/expected.json`) and carry `slice:4`. (The
     `--why` and MCP `explain` examples use Codex sessions and the `tool`
     field; they join in slice 3.)
   - the lineage set, checked against `FACTS.md`: every session and link
     (kind, `how`, evidence, spawning call), L8's pending link and L1 found
     in two project folders as one session, through `export` documents;
     own costs through `list --json`, and tree costs through the summary
     view's subagent line (the `tree_cost` field arrives in slice 6), with
     the copies in L4 and L5 never counted (the export format carries no
     computed cost).
2. **Laws** on the Morrow archive and on mutated copies of it: idempotent
   ingest, append equivalence (each Morrow file cut at every line boundary),
   order independence, rebuild equivalence, crash safety (kill during write),
   identity stability, nothing lost, nothing invented, paging, and the
   slice-1 lineage laws (`plan/testing/laws.md`: laws 4 and 11 for copies
   and late parents, 29–34 and 99; 25 and 39 are slice 6,
   `decisions/lineage.md` lists them all) on the lineage set; and every
   other law whose slice is 1 in the index of `plan/testing/laws.md`,
   including those added by the conceptual audits: cached and uncached
   answers equal (91), no branch without two message children (99),
   session times over own messages (106) and no re-store after a device
   or host rename (115), and those added on 2026-10-12: every framed record
   accounted for (120), no edit read as unchanged or absorbed (121), status
   counters equal their counts (122) and corruption found, never served
   (123), plus the untagged parts of laws 4–7, 10, 13 and 14
   (variants tagged with a later slice run from that slice).
3. **Real data.** On the operator's `~/.claude/projects` and the restored
   spool directories:
   - every file is parsed or has a `parse_state` status with a reason
     (`sessions-index.json` is parsed for titles; `companion` for
     `.meta.json` and `tool-results` files; `no_parser_yet` with `none
     planned` for workflow records, journals and memory files; `no_parser_yet`
     with its slice for the restored spool files); repository lookups are
     stored for every working directory that exists on this machine;
   - `xtask compare-old` (`plan/compare-old.md`) shows no difference that
     its expected-difference list does not explain; that comparison is how
     counts are checked against the old parsers;
   - the restored capture and hook spool files are all stored, with counts
     matching `evidence/old-archive-detail.md`;
   - the first build takes under 10 minutes; peak memory under 2 GB. The
     run records `/proc/pressure/io` beside its time, and a miss is first
     re-run with the archive on tmpfs: a slow build that is fast there is
     the disk, a slow one there is the code (`evidence/real-converge.md`,
     the evening section, where per-file syncs, not parsing, made a 120 MB
     build take 275 s);
   - `polylogue chrono`-style searches answer in under 200 ms.
4. **Masking.** A Claude Code fixture holding an `AKIA…` key and a
   `ghp_…` token: `read`, `list --json`, `export` and the MCP `read` show
   both masked; `has:secret` finds the session and `status` counts them;
   `read --reveal` unmasks them on a terminal and is refused with stdout
   piped or `CLAUDECODE` set.
5. **Operator use.** The operator runs the demo commands and uses search and
   read on their own sessions.

## Size

About 12–16k lines of Rust plus tests. The parts each carry real weight:
the Claude Code parser with its carried state (about 3k), storage with
revisions, pieces and crash-safe blobs (about 3k), the query grammar and
compiler (about 2k), convergence (about 1.5k), rendering (about 1.5k), the CLI
and MCP (about 1.5k), model types (about 1k). If one part heads well past its
share, stop and look at why before continuing. The size is design
information, not a budget.
