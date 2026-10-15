# Testing

The goal is a test suite where "green" means the product works on real data,
that stays small enough to read, and that runs in about a minute. Most of the
checking comes from a few general laws applied to many generated inputs, plus
the spec's own examples executed literally.

What the current Python suite taught (evidence in `evidence/` and this
session's audit): 786k lines of tests, mostly pinning internal wiring
(private functions, monkeypatched internals, raw table layouts), grew 2.3 test
lines per product line and made every redesign expensive, while missing real
failures such as a Codex parser that extracted under 1% of messages. The new
suite inverts that: test behavior through the real entry points, and let laws
do the work that hundreds of hand-written cases did.

Acceptance expectations and their input domain come from the current owning
spec, law or provider contract. An independent oracle can still be wrong if
it silently strengthens that contract: a fixed-input repeat test does not
establish an arbitrary-history guarantee. Keep stronger exploratory checks
separate until the semantic owner adopts their requirement. When evidence
refutes an expectation, correct the oracle with its reason; do not add
production machinery merely to pass it, or weaken a genuinely required
criterion. Review expected-output changes against the native evidence and
owning behavior, rather than treating a newly green test as its own reason.

## The layers

### 1. The spec's examples are tests

Every example in `spec/` and `examples/` that shows a command and its output
is executed by the test suite as written. The `trycmd` crate (from the clap
project) runs Markdown files containing console blocks:

````markdown
```console
$ polylogue 'repo:morrow since:2026-10-01' list
1  Oct 06 09:30  codex:01a20000-0006  Review parser (try strict mode)   6 msgs  $0.12
2  Oct 05 16:00  codex:01a20000-0005  Review parser                     9 msgs  $0.22
```
````

The examples run against a fixed synthetic archive built from
`examples/archive/` (see below). This makes the spec verifiable in the sense
the operator asked for: if the implementation's output differs from the
spec, a test fails and one of them is wrong. Changing intended behavior means
editing the example, which shows up in review.

### 2. Parser examples per provider format

For each provider, a directory of small input files covering each format
vintage and each record type the provider writes, with the exact normalized
output expected:

```
examples/sources/claude-code/
  basic-session/       input.jsonl   expected.json
  subagent/            input.jsonl   subagent-1.jsonl   expected.json
  compaction/          ...
  truncated-last-line/ ...
  unknown-record-type/ ...
```

`expected.json` is the normalized session (sessions, messages, blocks, links,
usage, tool outcomes) in Polylogue's export format. State expectations from
the native input graph and provider contract, not by blessing parser output.
For tool-bearing graphs, independently enumerate calls and answer nodes with
their native correlation evidence, expected result blocks, recorded body
presence/text and explicit or unknown outcomes. Include fallback-only,
empty and unlinked answers; message/text preservation alone cannot prove
result preservation. Tests compare with `insta` snapshots; review the native
expectations alongside a changed snapshot. A mutant that drops a result
block while keeping its message/text, and one that changes a status-less
empty answer to success, must fail the same golden comparison (law 60).
This is required provider acceptance, not a claim that an existing prototype
already runs those variants. Inputs are derived from real files with message text replaced by
neutral placeholder text and structure kept exact.

### 3. Generated inputs from inferred schemas

Polylogue infers a JSON schema for each provider record type from the
operator's real files (`spec/analysis.md`, schema inference). A generator
turns those schemas into synthetic provider files that look like the real
distribution: field presence, value types, enum values, string lengths,
record mix, session sizes, nesting depth. Text content is filler. The
generator can also produce deliberate pathologies: a truncated final line, a
duplicated record, an unknown field, a record type never seen before, a huge
single record, interleaved sessions in one file.

Generated corpora are deterministic (seeded) and are built on demand into a
cache, not committed. They feed the property tests below and the benchmarks.

The generator renders tool inputs and results per tool name from measured
key shapes (Claude Code Edit, Write, TodoWrite, Bash background tasks,
Agent; Codex function and custom tools), so file edits, todo state and
subagent calls look like real ones. Codex code-mode inputs (JavaScript
with nested tool calls) and `apply_patch` bodies come from small grammars,
and claude.ai blocks and ChatGPT mapping trees with branches have their own
generators. Coverage of parser lines by generated data is reported against
the real-data run (slice 8).

Sources: `polylogue-958ee`.

### 4. Laws (property and metamorphic tests)

The laws, each with the slice from which it applies, are in
[testing/laws.md](testing/laws.md). A slice's acceptance runs the laws of the
slices built so far.

### 5. A reference model for the query language

A deliberately simple in-memory implementation defines what queries mean;
the real implementation compiles queries to SQL, and a differential test
compares the two.

**The reference evaluator** (in `crates/query/tests/reference/`, a few
hundred lines, used only by tests):

- Input: the export documents of an archive (sessions with messages, blocks,
  links, events and notes) loaded into plain `Vec`s. Per-session facts
  (counts, tokens, cost) are computed from those documents by the same rules
  as `analysis.md`, written again as direct loops.
- It evaluates the parsed query tree (the same `polylogue-query` parser, so
  the grammar is shared and the semantics are not) by filtering the row
  vectors directly: `match` and `compare` are per-row predicates; `exists`
  scans the row's children for one child satisfying the whole inner
  expression; `seq` walks a session's tool calls in time order; scoping
  filters the second unit's rows by session membership; stages are applied in
  order on the vector (`sort by` is stable, ties broken by ID; `group by`
  collects into a `BTreeMap`; aggregates skip unknown values and count
  them).
- Out of scope: full-text relevance ranking (the reference model treats a
  text term as "the text contains every word, case-insensitively after the
  same Unicode folding as the index", and compares row sets, not order, for
  queries with text terms), `near:` (no embeddings in tests), and rendering.

**The query generator** (a `proptest` strategy) builds query trees, not
strings, so every generated query is grammatical by construction:

- It picks a unit, then builds an expression tree of bounded depth (at most
  4) from `AND`, `OR`, `NOT`, parentheses, `exists`, and leaves.
- Leaves are drawn per unit from the field table in `spec/query.md`
  ("Fields"), with values drawn from the archive being queried (an existing
  origin, tag, repo, model, tool, a time between the earliest and latest
  message, a number near an existing count), plus some values that match
  nothing, and `unknown`.
- Text leaves are words sampled from the archive's texts.
- It appends zero to three stages from the stage grammar, choosing fields
  valid for the unit.
- It always covers, with weighted probability, the shapes that broke the
  Python compiler (rule R65): alternatives (`origin:codex,claude-code`),
  negation of a group, `OR` inside a session scope without explicit
  parentheses, values differing only in Unicode case, and paths and values
  containing spaces, quotes, tabs and newlines. SQL and the reference
  evaluator compare text with one stated collation.
- The tree is printed with the canonical printer and parsed back; the test
  asserts the parse equals the tree (printer and parser agree), then runs
  the printed string in both implementations.

Compared: `rows` as JSON values in order (as sets for queries with text
terms and no `sort by`), `count`, and `outcome`. On failure, `proptest`
shrinks both the archive and the query tree and prints the smallest
query string, the archive's seed, and the two row lists. Save the actual
shrunk archive and query with the candidate revision and expected/actual
rows in the failure record. Before an external runner removes temporary
outputs, copy that small record into its existing run store; retain the
seed and build recipe as additional reproduction data.

This catches the classic query bugs (join multiplication, filters applied
after limits, conditions inside `exists` applied to different rows) without
hand-writing a case for each.

### 6. Surface snapshots

CLI output (plain and `--json`), MCP responses and rendered Markdown and HTML
are snapshot-tested on the example archive with `insta`. One example session
holds hostile text (`<script>`, quote characters, terminal escape codes,
bidirectional overrides); its HTML snapshot must be inert and its terminal
snapshot escaped (rule R56). Interactive features
(completion, the pager, fuzzy selection) are tested by driving the binary in
a pseudo-terminal and snapshotting the screen.

### 7. Checks on the operator's real data (not in the normal suite)

- **Parser comparison against the old system** (`xtask compare-old`): run the
  new parsers over the operator's stored source files and compare with the
  normalized output of the current Python parsers, recorded once
  (`plan/migration.md`). Differences are reviewed; each is either a new-code
  bug or an intended improvement.
- **Full convergence run** (`xtask full-run`): build a fresh archive from all
  real sources and report time, peak memory, and counts per provider.
- **Field scan** (`xtask field-scan`; rule R10): an informational report,
  never a pass/fail condition. Over a provider's real files it lists every
  field present in more than 0.1% of records that is not mapped, in `attrs`
  or on the ignore table, so the mapping table can be extended before the
  slice ships (the mapping-table check below is what fails a build). The
  same run reports, per origin and field, the share that is null, `unknown`
  or a trivial value (the native ID as a title, the wall time as a
  timestamp), so a degenerate column is visible before it ships.
- **Real-data sampling in tests**: the same law tests can run against a sample
  of real stored files instead of generated ones
  (`POLYLOGUE_TEST_REAL_SAMPLE=200`), or all of them (`=all`). The operator
  asked for this early on: real data at any sample size up to exhaustive.

### 8. Fuzzing and benchmarks

- `cargo fuzz` targets per parser check that no input makes a parser panic.
  Run on demand. From slice 10, the JSON API is fuzzed from its schema.
- `criterion` benchmarks for parsing and writing, plus an end-to-end
  benchmark on a generated corpus with the real archive's composition.
- Times are measured by benchmarks, never asserted in tests: a test checks
  structure and counts of work (records parsed, rows written, bytes read),
  never a wall-clock duration, because a duration depends on the machine
  (operator, 2026-06-09: "machine dependent, so no. want comprehensive
  benchmarks tho"). `behavior.md`'s targets are benchmark results to
  watch, and a missed one is a bug to investigate.
  Results are recorded for comparison; they are not a pass/fail gate.
- One end-to-end benchmark per target in `spec/behavior.md`, measured
  through the socket as a client sees it:

  | Target (`behavior.md`) | Benchmark |
  |---|---|
  | First build, all sources | converge the generated corpus with the real archive's composition from empty; and, on demand, the operator's real inputs (with IO pressure recorded) |
  | First build, Claude Code | the same, Claude Code inputs only |
  | Append appears in search | append a record to a live session of each size class; time to the first search that returns it |
  | Bare `polylogue` | cold and warm daemon, `polylogue` end to end |
  | Search, first page | the word-frequency ladder of `storage/index-db.md` (measured costs), by recency and by relevance |
  | Open a 10,000-message session | `read --from` at the start, middle and end of a generated 10,000-message session |
  | Hook recording | `polylogue hook` appending one line, process start and sync included |
  | Hook briefing | `polylogue hook --brief` with nothing to inject, and with a directory and a takeover briefing |
  | One process sample | one sampler pass over a synthetic process tree of the measured size |
  | CLI startup | `polylogue --version` and a cached-plan request, to first output |
  | Socket request | status, a cached plan and one result page through the socket, warm |
  | Reads during a rebuild | reads against the old index while a new one builds, checked for answers marked not up to date |

### 9. Agents using the shipped skill and tools

A fresh model, given only the shipped skill and the seven MCP tool
descriptions, answers the real-questions set (`evidence/real-questions.md`)
over the Morrow archive through the MCP wire, with no in-process shortcuts;
forming the query is graded separately from running it. Run on request when
the skill or a tool description changes; not a gate (slice 2).

Sources: `polylogue-z9gh.7.2`.

### 10. Real-data cases

Bugs found in real data, each becoming an example directory or a law, are
listed in [testing/real-data-cases.md](testing/real-data-cases.md).

## The example archive

`examples/archive/` is a small, fully synthetic set of provider files: a
fictional project ("Morrow", borrowed from the earlier designs) worked on
over a week with Claude Code, Codex, ChatGPT and claude.ai: a subagent, a
resume, a compaction, a Codex fork, failing and passing test runs, a ChatGPT
export with a branched conversation, a claude.ai export, and costs that add
up to known totals ($6.20). A separate Claude-Code-only home
(`examples/archive/claude-code-only/`, $7.87) adds the hard lineage cases:
nested subagents, a sibling copy, an explicit fork, a team with a message
between agents, a missing parent, the same session in two folders. Every spec
example runs against the main archive, and slice 1's against the
Claude-Code-only home. The main archive's expected results are
`examples/archive/expected/main/expected.json`, slice 1's
`examples/archive/expected/slice-1/expected.json`, both recomputed from the
raw files by `check_facts.py`. Its expected facts (counts, costs, which command
failed where) are written down in `examples/archive/FACTS.md`, so wrong
answers are obvious.

## Rules

- Test through public entry points: the CLI binary, the MCP server, the
  library's public functions. No tests of private functions, no mocks of
  internal components.
- Assert outcomes: rows, query results, rendered output, exit codes. Never
  assert identifier names, internal call order, or log wording.
- A bug fix strengthens the law or the generator that should have caught
  the whole class of bug. When the bug comes from a format variant that no
  example input shows, the fix also adds that input, scrubbed from real
  producer bytes, under `examples/sources/<format>/` with its expected
  output (rule R19). A new test function is the last resort.
- No test for a bug that the design has made impossible.
- No skipped tests; a failing test is fixed or the behavior it describes is
  changed in the spec. A test may skip only by naming an absent optional
  tool, and such a skip still fails in CI (rule R60).
- A run that selects zero tests, or a property test that generates zero
  cases, fails (nextest `--no-tests=fail`; rule R60).
- Every test has a timeout (nextest `slow-timeout` with `terminate-after`),
  so a hung test fails with its name instead of hanging the run.
- Tests assert outcomes and the order of events, never a wall-clock wait or
  an elapsed-time threshold: a test that waits for something observes the
  event that proves it (a stored row, a reply, a log line). Time-based
  limits (stall, stuck, round spacing) are reached by advancing the test
  clock, which covers both the request's `now` and the daemon's monotonic
  clock (`plan/slice-01.md`, test clock). Timing belongs
  to the benchmarks of `behavior.md`'s targets. The Python extension tests
  that waited on elapsed time failed at every worker count, so a real
  delivery defect and a slow scheduler looked the same (`polylogue-07pt`,
  `polylogue-h4shh`).
- The harness points `HOME` and the XDG directories at a temporary
  directory and fails the run if any test opens a path under the real home
  (rule R54).
- A test run tests a fixed commit: `xtask test` records the commit and fails
  if the tree changes during the run; checks never write tracked files
  (rule R64).
- Unit tests live next to the code (`foo_test.rs` style or `mod tests`);
  end-to-end tests live in `tests/`.
- The whole suite runs with `cargo nextest run` in about a minute on the
  operator's machine. Rust recompiles only changed crates, so there is no
  need for test-selection machinery; if the suite ever grows slow, fix the
  slow tests.
- A named run is the run named: when `xtask test` is given tests, files or
  packages (including the shared core's changed-package selection, joint
  decision 13), its record lists exactly what ran, and anything named that
  did not run fails the command with the missing names, never a smaller
  green run (`polylogue-gbwas`: 14 files named, 7 run, 609 passed, no
  warning).
- Size as design information, not a gate: the test code should be smaller
  than the product code.

## The spec and the code cannot drift apart

Cheap tests that make additions visible, rather than rules agents must
remember (`evidence/meta-premortem.md`):

- **Schema matches the spec.** A test extracts the `CREATE TABLE` blocks from
  `spec/storage/` and compares them with the schema the `store` crate
  creates. A new table or column needs a spec edit in the same change.
- **Commands match the spec.** The CLI's command, verb and flag list (from
  clap) is compared with the command table in `spec/surfaces/cli.md`; the MCP
  tool list with its table in `spec/surfaces/mcp-and-hooks.md`; the xtask commands with `plan/workspace.md`.
- **Grammar matches the spec.** The pest grammar is generated from, or
  compared with, the EBNF in `spec/query.md`.
- **References survive parser changes.** For each Morrow input, every
  message and block ID is recorded in a snapshot; a parser change that moves
  any ID fails, because notes would then point at the wrong content.
- **Every SQL statement prepares.** A test prepares each fixed SQL string in
  the code against a fresh schema, catching typos and stale column names
  without compile-time database access.
- **SQL inventory.** The same test keeps the list of fixed statements and
  checks, from it: each table's `INSERT`, `UPDATE` and `DELETE` statements
  sit in exactly one `store` function (rule R27); no statement uses
  `INSERT OR REPLACE`, `INSERT OR IGNORE` or `REPLACE INTO` (rules R24,
  R92) or a literal `0` as a fallback for a time column (rule R8); every
  column in `spec/storage/`'s DDL is read by at least one statement, query
  field or export field (rule R63; a release check, run once every slice is
  built, since the schema exists whole from slice 1 before its readers do); every table is read by a statement
  outside the module that writes it (rule R83; also a release check).
- **DDL shape.** The schema check fails on `ON DELETE CASCADE` (rule R92),
  on a table name from the process vocabulary or naming pending work
  ("pending", "debt", "queue", "obligation", "receipt", "census"; rules R84,
  R86), and on any table missing from the allow list, which carries one
  line per table saying why it is a table and not columns on an existing
  one (rule R87). Known-bad case: a schema with one `ON DELETE CASCADE`
  foreign key.
- **Query plans.** `EXPLAIN QUERY PLAN` runs on the compiled SQL of every
  query example in `spec/query.md` and on every fixed read statement; a full
  scan of `message`, `block`, `text` or `tool_call` fails, as does a
  temporary B-tree for an `ORDER BY` that an index should serve. A fresh
  build ends with `ANALYZE`, so the plans checked are the plans real
  archives get (rule R47).
- **Mapping tables.** Each provider's wire structs are compared with its
  mapping table in `spec/sources/`: every wire field is mapped, kept in
  `attrs`, or on the format's ignore table with a reason, and every table
  row names a real wire field (rule R10). For export formats the same check
  runs over every JSON key path of the committed examples (law 112).
- **Declared values have a producer.** Every value a closed list in
  `model.md` declares (outcome reason codes, block kinds, event kinds, link
  evidence) is produced by at least one parser on an example input, or is
  listed as reserved with the reason no format carries it yet; a declared
  value no parser can produce fails the check (the Python Claude Design
  origin declared two outcome reasons its format cannot carry,
  `polylogue-oavlb`).
- **Agent-facing text.** The shipped skill and the MCP tool descriptions may
  name only tools in `spec/surfaces/mcp-and-hooks.md` and fields in the query field
  tables; a test checks every name they mention (rule R31).
- **Parser versions.** Expected outputs under `examples/sources/<format>/`
  are changed only by `xtask golden`, which re-records a format only when its
  parser version is higher than at the last recording (rule R42). The
  re-recording is its own commit, separate from the code change, so the
  expected-output diff is reviewed on its own.
- **Known-bad cases.** The laws listed in `testing/laws.md` as easy to miss,
  and every drift check, register a known-bad case (an input or a
  deliberately broken build). A test enumerates them and fails if one has no
  known-bad case or if running it does not make the law fail (rule R61).
- **Triggers.** The schema in `spec/storage/` defines no SQL triggers; the
  schema check fails if one appears, so bulk work runs as statements, never
  per-row triggers (rule R47).

- **One-row lookups.** Part of the SQL inventory: a statement executed with
  `query_row` must bind a `UNIQUE` index (per `EXPLAIN QUERY PLAN`) or end
  its `ORDER BY` in a unique column; `LIMIT 1` without such an order fails
  (rule R76).
- **Owner stated once.** Over the DDL: a table with foreign keys to a parent
  and to a child of that parent uses a composite foreign key or drops the
  redundant column (rule R73).
- **Limits name their physical cause.** Every `const` in a product crate whose
  name contains `MAX`, `LIMIT`, `CAP` or `CEILING` carries a `// physical:`
  comment naming the limit it stands for (rule R48).
- **No compatibility names.** An identifier containing `legacy`, `compat` or
  `deprecated` in a product crate must name a provider-format vintage listed
  in `spec/sources/` (rule R74).
## Output that agents can use

- Test and comparison output is a few lines on the terminal (counts, the
  first failure) with full details written to a file under `target/`, so a
  failing run does not flood an agent's context.
- `xtask test --sample` runs a fast, seeded sample of the property tests and
  example directories for quick feedback during a step; the full run is at
  the end of the step.
- `compare-old` reports per input file, so failures can be split among
  workers by file.
- Expected outputs (`insta` snapshots, example expectations) are re-recorded
  only by `xtask golden`, never edited by hand, in a commit of their own
  after the code change, so the expected-output diff is reviewed on its own
  (for parser examples, only after raising the parser version, rule R42).
- Property tests are preferred over hand-written unit tests for agent-written
  code: published comparisons show agent unit tests are weak, while their
  property tests find real bugs.
