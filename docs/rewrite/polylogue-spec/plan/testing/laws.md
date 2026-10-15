# Laws

Property and metamorphic tests: each states something that must hold for
every input, and is checked on the Morrow archive, on altered copies of it
and on generated archives. A slice's acceptance runs the laws whose slice is
among the slices built so far, not "at or below" it, since slice 13 comes
right after slice 4 (`plan/slices.md`). The method around them (layers, rules,
output) is in [../testing.md](../testing.md).

Numbers are stable: a law merged into another keeps a one-line entry
pointing to it, so references elsewhere stay valid.

## Index

| Law | Name | Slice |
|---|---|---|
| 1 | Idempotent ingest | 1 |
| 2 | Append equivalence | 1 |
| 3 | Incomplete last line | 1 |
| 4 | Order independence | 1 |
| 5 | Rebuild equivalence | 1 |
| 6 | Parser version bump | 1 |
| 7 | Crash safety | 1 |
| 8 | Raw bytes survive the source | 1 |
| 9 | Export round trip | 1 (export documents only); 4 (re-import) |
| 10 | Identity stability | 1 |
| 11 | Lineage without duplication | 1 |
| 12 | Nothing lost | 1 |
| 13 | Nothing invented | 1 |
| 14 | Usage arithmetic | 1 |
| 15 | Paging | 1 |
| 16 | Query algebra | 3 |
| 17 | Query reference model | 3 |
| 18 | Blob store: proportional work | 1 |
| 19 | Merged into law 41 | — |
| 20 | Merged into law 7 | — |
| 21 | Change classification, per kind | 1 (the classes slice 1 implements); 2 (replaced, renamed) |
| 22 | Corruption is loud | 1 |
| 23 | Union of two archives | 13 |
| 24 | Evidence-cutoff reproduction | 3 |
| 25 | No double count in a tree | 6 |
| 26 | Merged into law 11 | — |
| 27 | Merged into law 4 | — |
| 28 | Merged into law 4 | — |
| 29 | The spawning block resolves | 1 |
| 30 | One subagent parent | 1 |
| 31 | No cycles | 1 |
| 32 | Same session, two folders | 1 |
| 33 | Every link has evidence | 1 |
| 34 | Compaction keeps the tree connected | 1 |
| 35 | Large sets of IDs work | 1 |
| 36 | Excision survives growth | 7 |
| 37 | Every stored input reaches an outcome | 1 |
| 38 | Same view from local files and from Sinex | 13 |
| 39 | A parent that loses its branch point | 6 |
| 40 | Same bytes at two paths | 1 |
| 41 | Revisions reassemble exactly | 1 |
| 42 | A delta reproduces its revision | 2 |
| 43 | Database snapshots are consistent and do not block | 2 |
| 44 | Older revisions are parsed | 1 (the Claude Code case); 2 (the Codex case) |
| 45 | A failed read is never an empty answer | 1 |
| 46 | Status is honest about what it measured | 1 |
| 47 | Rows without a time are counted, not dropped | 1 |
| 48 | Every input value reaches the archive | 1 |
| 49 | A session with messages never parses to nothing | 1 |
| 50 | Merging ignores volatile fields | 4 |
| 51 | Two sessions sharing a provider ID stay apart | 5 |
| 52 | Every surface returns the same rows | 1 (CLI and MCP); 10 (HTTP) |
| 53 | Every parameter is honoured or refused | 1 |
| 54 | A mutation over a query acts on every match | 7 |
| 55 | Every stored blob is reachable | 3 |
| 56 | Excision reaches every table | 7 |
| 57 | Twice the data, about twice the work | 1 |
| 58 | A large file stays within the memory target | 1 |
| 59 | One large input does not starve the others | 1 |
| 60 | A tool result pairs with one call | 1 |
| 61 | One writer, even through a symlink | 1 |
| 62 | Readers never write and never open a second descriptor | 1 |
| 63 | A failing parser loses only its input | 1 |
| 64 | A rebuild leaves one index | 1 |
| 65 | Unknown enum values round-trip | 1 |
| 66 | Deadlines and pool sizes | 1 |
| 67 | Which archive a command uses | 1 |
| 68 | Downloads stay on provider hosts | 4 |
| 69 | Calendar windows | 3 |
| 70 | Unpriced models and credits | 3 |
| 71 | An older binary does not write to a newer archive | 1 |
| 72 | Content never changes structure | 1 |
| 73 | Decoys stay decoys | 1 |
| 74 | Exports replace files atomically; IDs survive URLs | 1 (export); 10 (HTTP routes) |
| 75 | Destructive commands ask; agents cannot excise; claims are labelled | 7 |
| 76 | Counts match rows; one-row lookups are unique | 1 |
| 77 | The token goes only to the operator's listener | 10 |
| 78 | The WAL stays bounded | 1 |
| 79 | Ambiguity stays ambiguous | 1 (ID prefixes); 4 (sandbox links) |
| 80 | Briefings fit their budget | 2 |
| 81 | A rebuild adds no archive rows | 1 |
| 82 | Same top-k either way | 1 |
| 83 | The socket and the library give the same rows | 1 |
| 84 | Daemon stopped: one line, no database | 1 |
| 85 | Live state is a function of stored inputs and the time | 2 |
| 86 | A reused PID never inherits | 2 |
| 87 | A dead process ends liveness | 2 |
| 88 | The process sampler is proportional | 2 |
| 89 | Peers are symmetric | 2 |
| 90 | No task from prose | 6 |
| 91 | Cached and uncached answers are equal | 1 |
| 92 | Excision is immediate for reads | 7 |
| 93 | Write requests are idempotent | 7 |
| 94 | A derived product converges to its current version | 3 |
| 95 | The change feed replays to the archive | 7 |
| 96 | Codex usage sums to the reported total | 2 |
| 97 | Command normalization ignores prefixes | 2 (briefings' unresolved failures need it) |
| 98 | The same act has the same author everywhere | 2 |
| 99 | No branch without two message children | 1 |
| 100 | An edited revision replaces, it does not union | 5 |
| 101 | Live state is total | 2 |
| 102 | A clock-dependent outcome is computed at read time | 2 |
| 103 | Process times come from stored bytes | 2 |
| 104 | Activity is measured from the append, not the first sighting | 2 |
| 105 | Offsetless local times are placed in their zone | 5 |
| 106 | Session times cover own messages | 1 |
| 107 | Sequence patterns follow session order | 3 |
| 108 | An older copy stays older | 2 |
| 109 | Subscription windows anchor to reported resets | 3 |
| 110 | Every agent string gets rows or a corrected query | 3 |
| 111 | A provider database reads consistently while the provider writes | 2 |
| 112 | Every export key is mapped or excluded | 4 |
| 113 | A hook event never creates a session | 2 |
| 114 | Unknown never becomes a match or a zero | 3 |
| 115 | A device or host rename re-stores nothing | 1 |
| 116 | A stored hook line is one event, however often it is stored | 2 |
| 117 | Reading a provider database never writes it | 2 |
| 118 | A claim's check follows the archive | 7 |
| 119 | A pass writes only what its rubric allows | 7 |
| 120 | Every framed record is accounted for | 1 |
| 121 | No edit is read as unchanged or absorbed | 1 |
| 122 | Status counters equal their counts | 1 |
| 123 | Corruption in the archive's own files is found, never served | 1 |
| 124 | A fresh archive equals an upgraded one | the first release with a migration step |
| 125 | A restore from a copy reaches ok | 1 |
| 126 | Filter composition | 1 |
| 127 | A saved query selects the same rows by name and inline | 7 |
| 128 | Attaching rows never changes which owners are returned | 3 |
| 129 | A grouped result is rows | 3 |
| 130 | The order rows are read in never changes a match | 3 |
| 131 | A ranking never changes membership | 1 (relevance); 9 (near:) |
| 132 | Three-valued filters | 4 |
| 133 | Defaults are bound once | 3 |
| 134 | Every execution mode gives the same answer | 1 (pages, either search plan); 3 (`sql --jsonl`); 7 (kept results) |
| 135 | A command word is never searched for | 1 |
| 136 | A provider time is never rejected for its value | 1 |
| 137 | A value outside a field's closed set is refused | 3 |
| 138 | Hiding never changes a usage figure | 7 |
| 139 | A stale revision is refused on one machine | 7 |
| 140 | Signals behave as the joint cases say | 11 |
| 141 | The last run of each command equals a full recompute | 2 |


Each law is one test function. Until the generator exists (slice 8) the laws
run on the Morrow archive, its mutations and the synthetic builders' output
(Common parts); from slice 8, `proptest` also
runs them on generated archives. Each law below gives the statement, the
input, the operation, what is compared, and what a failure prints.

## Common parts

**Fixture.** `Fixture::morrow()` creates a temporary directory with a home
(`HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME` pointing into it), copies the
Morrow raw files to their provider locations under that home
(`.claude/projects/-home-op-morrow/…`, `.codex/sessions/…`), and sets
`POLYLOGUE_NOW=2026-10-08T12:00:00+02:00`. The ChatGPT and claude.ai exports
are placed in the fixture directory and brought in with `polylogue import`.
`Fixture::claude_code_only()` does the same for the Claude-Code-only home
(`examples/archive/claude-code-only/`: Morrow #1–#4 and the lineage set),
which is what slice 1 runs on. In slice 1, "Morrow" in a law means
`Fixture::claude_code_only()`; from slice 2, `Fixture::morrow()`.
`Fixture::generated(seed, size)` does the same with a generated archive
(layer 3); `size` is `small` (1–20 sessions, 1–60 records each) in the
normal suite.

**Variants by slice.** A variant, input or operation that needs a provider,
command or table from a later slice is tagged `(slice N)` and runs from
that slice; the untagged rest runs at the law's slice.

**Synthetic builders (slice 1).** Before the schema-driven generator exists,
laws that name a generated input use small Rust builders in `tests/` that
write Claude Code JSONL from a seed: N sessions of M records, optional
fields omitted at random, record and byte counts recorded beside the files,
and the size knobs that laws 18, 35, 57 and 58 need. From slice 8 the
generator replaces them for those laws.

**Known-bad cases.** Laws whose failure is easy to miss carry a known-bad
case (an input or a deliberately broken build) that must make them fail,
run in the normal suite: identity (10), nothing lost (12), nothing invented
(13), merging (4, 50), paging (15), counts (76), ambiguity (79) and every
drift check. For the rest, `cargo mutants` run on request shows whether the
tests can fail (rule R61).

**Converge.** "Converge" means: run the loop until nothing is left to do,
through the same library call that `polylogue daemon --until-idle` makes.
The crash law runs the binary instead, because it kills a process.

**Archive view.** Laws compare archives through public output only, never by
reading tables, so a schema change does not break them. The view of an
archive is:

1. **Sessions**: every document from
   `polylogue 'kind:*' export --format jsonl` (inherited messages, notes and
   `inputs[]` included), sorted by session ID, with `exported_at`,
   `exported_by` and every message's `first_seen` removed.
2. **Unrecognized**: the unrecognized list from `polylogue status --json`, as
   `(format, what, count)` triples, sorted.
3. **Parse problems**: from `polylogue status --json`, its `problems` as
   `(input path, status, detail)` triples and its `inputs` as `(format,
   status, detail, count)`, each sorted. (The path, not the input
   ID: an append-only input's ID is a UUIDv7 chosen when it was first seen,
   so it differs between two runs.)

Two views are equal when these three parts are equal as JSON values (object
key order ignored, array order significant, since arrays are already in a
defined order). Fields that depend on the wall clock, on when collection ran
or on how often work ran are left out: removed explicitly where the export
carries them (`first_seen`, the knowledge-time field of `model.md`, Time; the
export header), and absent by construction otherwise (last-seen times, the
index's write counter, read positions in growing files, derive bookkeeping).
Append-only inputs' UUIDv7 IDs also differ between runs, so `inputs[]` and
message `source.input` are compared by the input's path and byte range, not
its ID.
Embeddings are off in law tests, except the laws that name them (131, 92's
and 114's slice-9 parts, law 45's vector lane), which use a deterministic
local stub provider (vectors derived from the text's hash), never a network
call.

**Query results.** A fixed list of queries: every query in `spec/query.md`'s
examples, plus 50 generated ones (section 5). Results are compared as the
JSON `rows`, `count` and `outcome`, without `next_cursor` (cursors encode
archive state and are expected to differ).

**Failure output.** Every law failure prints: the law name; the fixture
(Morrow, or generated with its seed and size); the mutation with its
parameters (file, line number, byte offset, permutation, kill point); and a
unified diff of the first differing session document, or of the differing
status list. `proptest` shrinks the failing case and saves its seed in
`proptest-regressions/`, so the failure replays on the next run.

## Mutations

Each mutation is a pure function from a fixture to a modified fixture or a
sequence of steps. Line-based mutations apply to JSONL inputs (Claude Code,
Codex, hook spool); for ZIP and JSON exports only `duplicate`, `reorder` and
`delete_source` apply.

| Mutation | Exact definition |
|---|---|
| `cut_at_line(file, k)` | Replace `file` with its first `k` complete lines: the bytes up to and including the `k`-th newline. `k` ranges over `0..=n`, where `n` is the file's line count. |
| `cut_mid_line(file, offset)` | Replace `file` with bytes `[0, offset)`, where `offset` is strictly inside a line (the byte before it is not a newline, and it is not the file length). |
| `reorder(perm)` | Make the inputs appear one at a time in the order `perm` (a permutation of the fixture's files and imports), converging after each. The Morrow permutations always include "subagent file before its parent" and "#4 before #2". |
| `duplicate(input)` | Make the same bytes arrive a second time: import the same export file twice, or `polylogue import` a copy of a live session file from outside the provider locations. |
| `kill_writer(n, point)` | With the test-only cargo feature `fault-injection`, the writer aborts the process (`std::process::abort`) at the `n`-th transaction, at `point`: `before_commit`, `after_commit`, or `blob_written_not_renamed` (a blob's temporary file is written but not yet renamed into place). Without the feature the check does not exist in the binary. |
| `bump_parser(format)` | Converge with a parser version table in which `format`'s version is one higher than compiled in. The engine's configuration takes this table (defaulting to the compiled-in versions); only tests pass a different one. |
| `delete_index` | Delete `index.db` with its `-wal` and `-shm` files. |
| `reimport_export(target)` | `polylogue 'kind:*' export --with-blobs --to D`, then `polylogue import D` into `target`: either a new empty archive, or the same archive. |
| `delete_source(file)` | After converging, delete `file` from its provider location. |

## The laws

**1. Idempotent ingest.**
Slice: 1.
Input: Morrow; generated archives.
Operation: converge; converge again with no changes; also apply
`duplicate` to every input and converge.
Compared: the view after the first convergence against the view after each
later step. Also, the second convergence reports zero inputs parsed and zero
sessions written (from the `--until-idle` summary).

**2. Append equivalence.**
Slice: 1.
Input: each Morrow JSONL file `f` and each `k` in `0..=n`; generated JSONL
files with random `k`.
Operation: A: `cut_at_line(f, k)`, converge, restore the full `f` in place
(appending the rest to the same file, so the inode is kept and the change is
an append, not a replacement), converge. B: the full fixture, converge.
Compared: views of A and B equal. Also, in A the second convergence parses
only lines `k+1..n` of `f` (the summary reports `n-k` records read for `f`),
and `f` is one input with two pieces (the first `k` lines and the rest),
whose bytes read back equal the full file (`storage/blobs.md`, Inputs,
revisions and extents). At `k = 0` the empty file gives an input with no
record, and its first line arrives as an ordinary append. The pieces are
checked before joining; the fixture clock does not advance, so joining
never runs in this law.

Variant: between two content-bearing appends, append a supported JSONL
separator-only window to the same input. Its proven contiguous acquisition
advances durable byte progress while producing zero new normalized sessions
or records; every original extent, byte order and dependency still reads
back exactly. Restart and retry that window, then append the next record:
the normalized view, markers and attachment references equal the
uninterrupted full-input baseline, with no invented empty session or lost
prior contribution. A missing predecessor, unproven gap or window containing
a nonblank unfinished record must not receive separator-only completion;
use the existing unavailable/partial-tail outcome. These zero-output-chain
consumer schedules remain required, not established by ordinary append tests.

**3. Incomplete last line.**
Slice: 1.
Input: each Morrow JSONL file `f`; several `offset` values per line.
Operation: `cut_mid_line(f, offset)`, converge.
Compared: the view equals the view of `cut_at_line(f, k)`, where `k` is the
number of complete lines before `offset`; no parse problem is listed (a
partial last line waits, `behavior.md`, failures). Then restore the full `f`,
converge: the view equals the full fixture's view.

**4. Order independence.**
Slice: 1.
Input: Morrow with a set of permutations (including the two named above, plus
200 generated from a fixed seed, where each input's revisions and appends
stay in sequence and different inputs interleave freely, as real arrival
does; Sinex's law L5 run this way found two order dependencies that a few
chosen orders missed); generated archives with random
permutations; the lineage set with children stored before parents and with
parents first (all orders).
Operation: `reorder(perm)`.
Compared: the view after the last step equals the view of converging the
whole fixture at once, including identical links, `inherited_from` marks and
counts. Late-parent variant (formerly laws 27 and 28): converge without a
parent's input (#2 for #4; L2 for the sibling copy L4) and check that the
child's copies count as its own and its link is pending; then store the
parent and converge; the view equals a run that had the parent from the
start.
Variants from the conceptual audits (`evidence/conceptual-idempotency.md`,
`conceptual-identity.md`, `conceptual-time.md`):
- one message present in two location inputs with content that ties on route
  and completeness, stored in both orders: the same version wins (the
  tie-break is the higher BLAKE3 of the record as canonical JSON without
  its format's volatile fields, `spec/convergence.md`, never the input ID). Known-bad: an input-ID tie-break.
- every fallback-ID format in the generated archives (and, slice 5, a
  Grok-shaped id-less session with two identical messages, imported twice and then as an
  extended export): any order, and after `delete_index`, gives the same
  message, block and event IDs, and the message count equals the distinct
  occurrences (fallback numbering is per input revision).
- interleaved appends to several inputs of one session (live file, hook
  spool, companion) at partial lengths: the same view as storing each input
  whole.
- (slice 7) an inline marker in the copied prefix of a late-parent child, and
  `::title` markers in a parent and in its continuation: the notes in force,
  their authors and sources, and each target's title equal the parent-first
  run's, and no marker creates a conflict.
- (slice 7) several single-valued markers in one message and in different
  text blocks: shuffled extraction gives the same position-selected value.
  Moving unchanged complete marker content preserves its ID and recomputes
  its coordinates; editing only a continuation body gives a new ID and
  leaves the old note stored but out of force. A single-valued marker with
  `revises=` emits `marker_not_applied` and creates no note or revision edge.
- (slice 2) the old and the new chain of a Codex rollout that dropped a record, in
  both arrival orders: identical `seq`, with the dropped record between its
  old neighbours; and a pre-migration and a post-migration revision whose
  prompt times differ by 1 ms give the same prompt IDs and count.
- (slice 4) two captures of one ChatGPT message, one taken mid-reply, in both orders:
  the complete version wins in both.
- (slice 4) a partial capture with provider IDs (3 of a conversation's 10
  messages) and a full export of it with provisional IDs, in both orders:
  the session has all 10 messages once; the 3 captured ones carry provider
  IDs with both inputs among their sources, the other 7 stay provisional.
  Known-bad: dropping every provisional message once a provider-ID route
  exists (the session would show 3).
- (slice 4) a delayed older route: an export taken in August stored after
  September's export and a later unchanged capture of the same conversation.
  The title and current node stay September's (they follow the provider's
  update time, never arrival), and the August copy replaces nothing the
  later routes confirmed. Known-bad: ordering changing values by arrival or
  by a counter advanced only on changed content (the defect Sinex found in
  its own writer, storage.md §3.3, 2026-10-12).
- 30 sessions opening with the same pasted instruction block, in any order:
  identical display titles (the known-preamble set is a derived product).

**5. Rebuild equivalence.**
Slice: 1.
Input: Morrow after converging and (slice 7) after adding notes and tags
through the CLI; generated archives.
Operation: `delete_index`, converge.
Compared: the view and the query results equal those from before deletion.
Notes survive because they live in `archive.db`.
Also compared: every message's `first_seen` (fully determined by stored
observations, so a rebuild must reproduce it). Variants: (slice 11) with a
session that made a commit, delete the repository, then `delete_index` and converge:
the view is unchanged (commit details come from stored records, not the live
repository). With a call that has no result, rebuild an hour later: the
stored outcome is the same `unknown/no_result`. With a Claude Code file that
was appended, head-rewritten and appended again: the rebuilt rows, citations
and `message_source` rows equal the incremental ones (the subsumed chain is
never read).

**6. Parser version bump.**
Slice: 1.
Input: Morrow, for each format it contains.
Operation: converge; `bump_parser(format)`; converge.
Compared: the view is unchanged (the parser code did not change, so the
output must not either). The second convergence parses exactly the inputs of
`format` and no others (from the summary's per-format counts).
Variants: bump while appending to a file larger than one 64 MB step,
reading the session after each committed step: every retained message has
its consistent old or new version, including blocks, tool calls and search
entries, never neither. With the unchanged parser, counts never drop or
double-count replayed rows. The result stays incomplete and aggregates are
degraded while row finalization or required products remain pending, even if
all input bytes have been parsed.
Also rebuild a generated session spanning several steps with a parser version
that changes existing messages and stops extracting selected old messages.
Only those obsolete rows disappear during bounded finalization; after it,
order, facts, lineage marks and search equal a fresh build at the new version.
Include one message whose block collection exceeds a writer step and one
large searchable text below SQLite's value limit. Read the retained message
and ranked search between preparation steps: each message is complete old or
new, and unselected candidates affect neither hits nor relevance statistics.
Preparing text, publishing its representation and retiring obsolete parts each
respect the work budget; a single giant staging INSERT is not qualification.
(slice 7) With a parser that no longer extracts an inline marker, the
marker's note is out of force after the bump.

**7. Crash safety.**
Slice: 1.
Input: Morrow.
Operation: run the binary `polylogue daemon --until-idle` uninterrupted and
count its transactions `T` (from the summary). Then, for each `n` in `1..=T`
and each `point`, run a fresh fixture with `kill_writer(n, point)`, confirm
the process died, and run `--until-idle` again without the fault. The same is
done with the test-only abort at every write step of storing, joining
(slice 2) and excision (slice 7) in the blob store (formerly law 20).
Compared: the final view equals the uninterrupted view; every input reads
back byte-identical to what was observed; no blob file is missing, and every
blob's BLAKE3 matches its name; no temporary blob files are left after the
restart has converged.
Further kill points (from `evidence/conceptual-idempotency.md` 18 and
`conceptual-concurrency.md`): during a derive walk after a `prices.toml`
change; during a parser-bump rebuild; (slice 7) at every step of an
`input:`, `session:` and `text:` excision, the `index.db` side included;
(slice 2) while joining quiet pieces; during a rebuild beside the old index
and at its swap, with a reader open on the old file (the reader never errors
and never sees a corrupt page); during a first build (a restart on the same
boot continues the build; after a simulated new boot the unsynced file is
deleted and built again). Repeat on law 6's multi-step rebuild, killing
between emitted sessions before their shared parse checkpoint, during
ordering and obsolete-row removal, and immediately before and after final
completion. At every surviving commit the checkpoint's offset, carry and
record counts describe only committed row work; a checkpoint behind already
written rows is safe and replays without duplication. Reaching byte EOF does
not report completion while row finalization remains. Compare row counters
with actual rows after each restart and final order, facts, lineage and search
with the uninterrupted run. After restart an excision completes and no stored
byte or row holds the excised text.

**8. Raw bytes survive the source.**
Slice: 1.
Input: Morrow; each provider file in turn.
Operation: converge; `delete_source(file)`; converge.
Compared: the view is unchanged (sessions are not removed when their source
file disappears). `polylogue SESSION read --view raw` for a session from
`file` prints the original records, byte for byte. `polylogue input cat ID`
for `file`'s input prints the file's bytes exactly, and every blob in its
extents hashes to its name.

**9. Export round trip.**
Slice: 4 (re-import); 1 (export documents only).
Input: Morrow with notes and tags added (slice 7); generated archives. None
contains credential-shaped text: an export masks it (`spec/behavior.md`,
privacy), so a masked span does not round-trip by design; masking is
tested by the joint masking cases.
Operation: `reimport_export(new archive)`; separately,
`reimport_export(same archive)`.
Compared: new archive: the session documents equal the original's, except
`inputs[]` (the new archive's only input is the export, not the provider
files). Same archive: the view is unchanged except that `inputs[]` of each
session gains the export input; no session, message or note is added or
duplicated.
Variant (slice 7): a note revised twice, a withdrawn note, a claim with a
check and a note with `valid_until`: in the new archive every revision is
present, no relation points at a missing note, the claim's check state and
the expiry are the same as in the original.
Variant (slice 5): a generated id-less session (Grok shape) with two identical
messages, imported twice and then as an extended export: the message count
equals the distinct occurrences.

**10. Identity stability.**
Slice: 1.
Input: Morrow; generated archives.
Operation: converge and record every session, message and block ID with its
document. Then, separately: add an unrelated generated session file; append
lines to a different session's file; (slice 2) move a Claude Code session
file to another name in the same project directory; converge. (A main file's native ID is its file stem at first
observation, `spec/sources/claude-code.md`; the move is a `renamed`
observation of the same input, so the stem it was first seen under is kept.)
Compared: every recorded ID still exists, and its message and block
documents are unchanged (a session document may gain messages at the end).
After the move, the set of IDs is exactly the same. IDs a provider section
marks as movable (`model.md`, Identity, known exceptions) are excluded.
Copy variant (slice 3): copy the file under a new name instead (the old path stays).
Compared: no message is counted twice in `analyze` totals, and every
recorded ID still exists.

**11. Lineage without duplication.**
Slice: 1.
Input: Morrow (session #4 repeats #2's first 18 records); generated archives
with resumed and forked sessions.
Operation: converge.
Compared: (slice 3) `polylogue analyze --json` totals for messages, tokens and known
cost equal those computed by the test from the export: the sum over all
messages without `inherited_from`. The message count equals the number of
distinct native message IDs across all inputs. `#4 read --inherited` shows
#2's 18 messages, each with `inherited_from` naming #2's message.
Second oracle (formerly law 26): compute archive totals (messages, tool
calls, tokens, cost), then delete every message with `inherited_from` set and
compute again; the two sets of totals are equal.
Variants: two resumes of one deleted parent, and a chained resume (a
resume of a resume) of it: every copied message is counted once across the
children. Slice 4: two ChatGPT conversations sharing a prefix of node IDs
("branch in new chat") count each shared message once.

**12. Nothing lost.**
Slice: 1.
Input: Morrow (one `atis-latch` record in #2); generated archives, where the
generator records how many records of each type it wrote, including injected
unknown types and unknown fields.
Operation: converge.
Compared: for every record type the parser understands, the number of
messages, events and blocks in the export matches the generator's counts
under the provider's mapping table (`spec/sources/`), and for every type
it does not, `unrecognized` has exactly the written count (Morrow:
`record:atis-latch`, count 1). Record types the spec says are only counted
(such as Claude Code `progress`) appear in `unrecognized`'s counted list with
their exact count. The counts compared come from the generator and from the
stored inputs, never from the parser's own bookkeeping, so a parser that
drops a record cannot also hide the drop (rule R62).

**13. Nothing invented.**
Slice: 1.
Input: generated archives where the generator randomly omits optional
fields.
Operation: converge.
Compared: for each optional input field in the provider's mapping table,
when the generator omitted it, the corresponding export field is absent
(not `null`, not a default). In particular: no message time when the record
has no time (never the file's modification time); no `model` when the record
names none; a call with no result record is
`unknown` with reason `passed_without_result` when a later model response
in the session exists (never shown as pending), otherwise `no_result`
(shown as `pending` only at read time while its session is live, law 102),
and `unknown` with a reason when the result does not say.

**14. Usage arithmetic.**
Slice: 1.
Input: generated sessions with random usage; (slice 2) Codex fixtures with
shuffled and duplicated `token_count` samples; (slice 4) the same ChatGPT
conversation through an export and a capture.
Operation: converge.
Compared: for every message with usage, cache read is not greater than the
provider's total input when the provider reports cache inside input
(`analysis.md`, token rules). Codex usage is per model request: each
request-bearing message carries the last request's counts from its
`token_count` event, and repeated events do not add to it; the
sum of a session's per-request counts equals its final running total, or
`status` shows the difference (`analysis.md` §1, token rule 5). A message
seen through two routes or inherited contributes its usage once
(`analysis.md` §1, token rule 6): archive totals equal
the sum over distinct messages.
Also: for every message with both known, `reasoning ≤ output` (reasoning
tokens are part of output, never a parallel lane).

**15. Paging.**
Slice: 1.
Input: Morrow and generated archives; each query in the fixed list; page
sizes `1..=rows+1`.
Operation: run unpaged; run with the request's `page_size` set to the page size
(never `| limit`, which is a selection, `query.md`, four bounds) and follow
`next_cursor` until it is `null`.
Compared: the concatenated pages equal the unpaged `rows`, in order. A cursor
taken before `delete_index` and used after it returns an `error` saying it is
from before the rebuild.
Second variant (rules R48, R52): for every sort (time, numeric, text
relevance, semantic off, `random`) and page sizes 1, 2 and 7, new sessions
are inserted between pages. Continuation is by key over the current state
(`query.md`, Continuation): every row that existed when the first page was
read and sorts after the previous page's anchor appears exactly once, no row
whose sort key and ID did not change is skipped or repeated, rows inserted
after the anchor may appear, rows inserted before it do not, and no read
transaction stays open between pages; `random`
gives the same order on every page because its seed is in the cursor; a
per-session bound (`last:N`) is applied per session, never as one global
limit; `count` is the same whatever the page size.
Clock variant (slice 3): page a `since:7d` query and a `today` query,
sending each next page's request with `now` moved forward across midnight
(the test clock, `plan/slice-01.md`, Daemon details): the pages
concatenate to the unpaged result at the first page's `now` (the cursor
carries the resolved time).

**16. Query algebra.**
Slice: 3.
Input: generated archives; pairs of generated conditions `A`, `B`
(section 5).
Operation and comparison: the row sets (as sets of IDs) of `A B` and `B A`
are equal; `NOT NOT A` equals `A`; `A OR B` equals the union and `A B` the
intersection of `A`'s and `B`'s rows; `S | UNIT where C` returns only rows
whose session is in `S`; `Q | agg count` equals the number of rows of `Q`;
`sessions where Q | agg sum:cost` equals the sum of each matched session's
cost, once each.

**17. Query reference model.** See section 5.
Slice: 3.

**18. Blob store: proportional work.**
Slice: 1.
Input: a generated growing JSONL file of 1 GB.
Operation: append k bytes (k from 1 byte to 10 MB) and converge.
Compared: bytes written to the blob store are about k; bytes read for the
classification are at most 40 KiB plus k (counted by an IO-counting wrapper
on the store's file layer); an unchanged file costs a `stat` and no read.

**19. Merged into law 41.**

**20. Merged into law 7.**

**21. Change classification, per kind.** Slice: 1 (the classes slice 1
implements); 2 (replaced, renamed). Input: a stored JSONL file mutated each
way: unchanged; touched with its mtime restored (`touch -r`); appended;
appended with a partial last line; saved through a new inode with an
identical prefix plus more; edited in its first 4 KiB, in its last 4 KiB
before the stored end, and in the middle; truncated; truncated then regrown
past its old size; renamed; deleted. Compared: the observation class is
unchanged, unchanged, appended, appended (the partial line stored at the
next observation), replaced, rewritten (all three edits; the middle edit
within a few appends or at the next quiet-time full check), rewritten,
rewritten, renamed, deleted. Every earlier revision still reads back
byte-identically. Two further cases: a Claude Code file whose first record
changes while records are appended (Claude Code rewrites record one in
place) is classed head rewritten, not rewritten: a new chain whose extents
reuse the previous chain's blob ranges (only the new first line and the
appended bytes are stored), which subsumes the previous chain; every earlier
revision reads back byte-identically, and the parse continues from the
previous chain's carry with shifted offsets after re-reading the first
record (rule R46), giving the same rows as a whole parse; and a store step that fails (the blob write aborted by
`fault-injection`) leaves the collect cursor where it was, so the next
round stores the same bytes (rule R37).

**22. Corruption is loud.**
Slice: 1.
Input: Morrow; flip one byte in a stored blob file.
Compared: a full read fails naming the blob; a range read fails naming the
frame; nothing is silently returned.

**23. Union of two archives.**
Slice: 13.
Input: Morrow split into two archives A and B by input (and, after slice 7,
each with its own notes, including one conflicting revision of the same
note).
Operation: publish A and B to one Sinex; rebuild each from it.
Compared: identical `archive.db` rows except `local_seq` and
`applied_upto`, which a union never copies, and identical views; after slice
7, the conflicting revisions both appear, shown as a conflict.
Variants (after slice 7): A holds an excision that B predates: after the union neither
archive holds the content (rows, blobs or index). A and B each hold a
current `title` note for one session in separate families: both archives
show the same displayed title and a conflict mark (`user-data.md`, title
conflicts). Two machines with the same host name
(distinct `/etc/machine-id`): the union is plain, and neither machine
classes the other's file as a rewrite.

**24. Evidence-cutoff reproduction.**
Slice: 3.
Input: Morrow converged in the three collection stages of `FACTS.md` ("When
#4's live file was collected"), plus a Codex input rewritten between two
stages to drop a record, and the late-parent fixture of law 4 (a child whose
parent arrives in a later stage).
Operation: after each stage keep the session view and the query results;
then, on the final archive, run the same queries with `evidence_until:<stage time>`.
Compared: each `evidence_until` answer equals the view kept at that stage (minus
`first_seen`), counts, costs, lineage marks and each message's winning
version included; copies whose parent was not yet stored count as the
child's own under `evidence_until:T`, as they did at T. Known-bad: a `first_seen`
computed from revision length alone across chains, and facts read from the
current rows. Monotonicity (formerly this law) follows: what `evidence_until:T1`
shows stays visible `evidence_until:T2` unless withdrawn or excised.

Laws 4 and 11 (copies and late parents) and 25, 29–34 cover lineage
(`decisions/lineage.md`). Each runs on the main
Morrow archive, on the Morrow lineage set (`FACTS.md`, "Lineage set") and on
generated archives.


**25. No double count in a tree.**
Slice: 6.
It compares `tree_cost`, a slice-6 field.
Compared: for every subagent tree, `tree_cost(root)` equals the sum of own
`cost` over the root and all its descendants, and every session appears in
exactly one tree (lineage set: L1's tree and L5's tree as listed in FACTS).
Variants: after a `prices.toml` change and its derive walk; with
children collected before parents; with a fork subagent whose second record
repeats the parent's spawning response under a new uuid (that response's
cost counts once in the tree).

**26. Merged into law 11.**

**27. Merged into law 4.**

**28. Merged into law 4.**

**29. The spawning block resolves.**
Slice: 1.
Compared: for every subagent whose meta file has `toolUseId` and whose parent
session is present, `link.spawned_by` is the parent's block with that
`call_id` (L2, L3, L6, L7). #3's meta file has no `toolUseId`; its spawning
call is found from the parent's side (`toolUseResult.agentId` on #2's
`toolu_0203` result), so its link also resolves to that block. L4, whose meta
file has no `toolUseId` and whose parent records no matching result, has a
link with evidence `folder` and no `spawned_by` (`model.md`, the evidence
strings).

**30. One subagent parent.**
Slice: 1.
Compared: no session has two incoming `subagent` links (a nested subagent's
parent is the subagent named by `parentAgentId`, not also the folder's main
session).
Input: also a subagent file symlinked into a second session's
`subagents/` folder: one session, one parent.

**31. No cycles.**
Slice: 1.
Input: also a generated archive whose `parentUuid` chain loops.
Compared: following `subagent` and `continuation` links from any session
never returns to it; following `parent` inside a session never loops (a
cycle in the input is broken at the record that closes it and reported as a
parse problem).
Input: also a split response interleaved with its own tool result (the
result's parent is the response's first record, and the response's second
record's parent is the result): no cycle and no parse problem.

**32. Same session, two folders.**
Slice: 1.
Compared: L1 stored from both folders produces exactly the session that the
longer file alone produces.

**33. Every link has evidence.**
Slice: 1.
Compared: every link has a non-empty `evidence` naming an input field, and
removing the input that holds it (and converging) removes the link.

**34. Compaction keeps the tree connected.**
Slice: 1.
Compared: after a compaction boundary with `logicalParentUuid` (#4) or a
microcompaction (L5), every message on the active path still has a `parent`
chain to the root.
Input: also messages whose `parentUuid` names an `attachment`,
`progress` or `system` record: `parent` skips event records to the nearest
message, and the chain still reaches the root.

**35. Large sets of IDs work.** Slice: 1. Input: a generated session with
more messages than SQLite's limit on query parameters (more than 32,766).
Operation: (slice 7) excise it; separately, export it and read it. Compared: all of
its rows are gone after excision (index, search, export); the export holds
every message. Fails if any code builds an `IN (...)` list from data.

**36. Excision survives growth.** Slice: 7. Input: Morrow; excise
#2 while its transcript and its hook spool file keep growing; append records
naming it; converge. Compared: none of its content reappears in the index,
search, exports or the change feed; text excisions are applied to the
appended bytes before they are stored. The same holds for each input route
(live file, hook spool, import, browser capture) and for a file large enough
to take the streaming path (more than one 64 MB step), since the Python
excision test used a small payload that never streamed.
Variants from the conceptual audits:
- excise `input:` of a live transcript, then append to it and rescan; also
  delete `file_cursor` and converge with `index.db` deleted: no input,
  observation, revision or blob for that path or those hashes appears
  (`input list`, a blob store scan).
- re-import the original export after a `session:` excision: no blob holds
  the excised session's records; the same for a shared input
  (`history.jsonl`) and a database snapshot.
- after a `text:` excision of a live file that still holds the text, three
  quiet periods and 50 appends add no `rewritten` observation and no new
  chain.
- the excised string planted in a prompt, a Bash command, a tool result and
  a member of an export ZIP: every occurrence is gone, and a rebuild never
  parses filler as text.
- excise the parent of a resumed, a forked and a copied-subagent child: no
  child gains own messages, cost or search hits, and no child input holds
  the parent's text.
- excise one revision of a note: the other revisions' currency is unchanged
  (a withdrawn or revised note does not come back).

**37. Every stored input reaches an outcome.** Slice: 1. Input: Morrow plus
an empty session file and a file of only unknown records. Compared: after
convergence, every stored input has a terminal outcome for its parser
version (`done`, `no_sessions`, `refused`, `failed` with a reason,
`companion`, or `no_parser_yet` with the slice that adds its parser, or
`none planned`), and a
second convergence does no work for them.

**38. Same view from local files and from Sinex.** (`decisions/sinex.md`, rebuilding from Sinex) Slice: 13.
Input: Morrow converged from local files; a Sinex holding the same inputs and
archive rows.
Operation: delete `index.db` in backed mode and converge from Sinex.
Compared: identical export documents and `status --json` counts, notes and
recorded deliveries included.

This law is also the cross-project contract test: the shared example archive
and its expected results live with the contract file, and the same test runs
in the Sinex repository against Sinex's side (what Sinex stores and serves
back for that archive). Both must pass in finite fixed-basis shadow qualification before native backing is accepted. Before `primary` is allowed, the same comparison runs on the
operator's real corpus (a rebuild from Sinex compared with the local
index), and every difference is explained or fixed. The successor refuses configuration naming `mirror` as unsupported without silently changing to `off` or `primary`; status and existing reads retain their declared authority and expose the configuration error.
Input: also a session reached through two routes (an export and a
capture): the same message versions and equivalence keys from both
sides.

**39. A parent that loses its branch point.** Slice: 6. Input: L1
and its fork L5; re-parse L1 with a parser version under which L5's
branch-point message no longer exists. Compared: L5 stays linked, with its
branch point pending; nothing is stranded or dropped.

**40. Same bytes at two paths.** Slice: 1. Input: one Morrow file copied to
a second provider path. Compared: two inputs, one per location (an input is one
location, `spec/storage/blobs.md`); the view equals Morrow's, because the
session merges by message ID. Also: `source:gone` after deleting one of the
two paths is false for the session (one input remains).

**41. Revisions reassemble exactly.**
Slice: 1.
Input: generated inputs with long histories mixing appends, head rewrites
of a Claude Code file, rewrites, joins (slice 2) and snapshots (slice 2).
Compared: every revision's bytes assembled from its extents equal
the bytes observed for it, byte for byte, before and after joining (slice 2)
and after the crash points of law 7; each complete revision's bytes hash to
its recorded BLAKE3.
Each Morrow JSONL file is also stored as many appended pieces and complete
in one piece, and (slice 2) after the pieces are joined (formerly law 19):
the bytes read back are identical in every form, and the views are equal.

**42. A delta reproduces its revision.**
Slice: 2.
Input: Morrow's Codex rollout and a generated rewrite of it (records gain a
field, some records are dropped); every pair of real old and new rollouts in
the fixture set.
Compared: a revision stored as a delta reads back identical to the rewritten
file; its base is a plain blob (depth one); excising the base first
re-encodes the delta plainly and the revision still reads back identically.

**43. Database snapshots are consistent and do not block.**
Slice: 2.
Input: a test SQLite database in WAL mode written by a second process in a
loop (rows with a checksum column), snapshotted repeatedly.
Compared: every snapshot opens, passes `PRAGMA integrity_check`, and holds a
committed state (every row's checksum valid; no half-written transaction);
the writer is never blocked by a snapshot (no commit returns `SQLITE_BUSY`
or waits on a lock the snapshot holds: counted lock waits stay 0);
a later snapshot stored as a page patch reads back identical to a full copy
taken at the same moment; after `VACUUM` a new plain base is written.

**44. Older revisions are parsed.**
Slice: 1 (the Claude Code case); 2 (the Codex case).
Input: a Claude Code session file rewritten in the middle (a lossy rewrite,
`sources/README.md` rule 5), dropping two records; and (slice 2) a Codex
rollout and its rewritten revision, where the rewrite dropped
`agent_reasoning` and `user_message` records.
Compared: the session holds the dropped records (from the older chain), the
records the rewrite added (from the newer), and (Codex) per-record
timestamps from the older revision where the newer collapsed them.

**45. A failed read is never an empty answer.** (rules R1, R2, R6)
Slice: 1.
Input: Morrow; the test-only `fault-injection` feature makes one store read
fail at a chosen point: the `index.db` read behind each query unit, each
`status` component, the full-text and (slice 9) vector lanes of search, the
notes read behind a briefing (slice 7: before then briefings read no notes),
and the input a safety check reads
(the blob liveness behind excision, slice 7).
Operation: run every query example, `status --json`, a briefing (slice 2)
and an excision preview (slice 7) with each failure in turn.
Compared: the affected result's outcome is `error` or `degraded` and names
the failed part; no result says `ok` or `empty`, no count or cost shows `0`
in place of the unread value, and the safety check refuses instead of
proceeding. A failure output names the query, the injection point and the
outcome actually returned.

**46. Status is honest about what it measured.** (rule R5) Slice: 1. Input:
an empty archive; Morrow with one provider location made unreadable (`chmod
000`); Morrow with `index.db` removed mid-run. Compared: every figure in
`status --json` carries `measured`, `measured_absent` or `unmeasured`; the
empty archive reports `ok` only for what it measured as empty; the
unreadable location is an error naming the path, never zero inputs; no
unmeasured figure is shown as healthy.

**47. Rows without a time are counted, not dropped.** (rule R8)
Slice: 1.
Input: Morrow plus a generated session whose records have no times.
Operation: (slice 3) `analyze --by day`, `analyze --by month`; a `since:` filter.
Compared: the grouped totals include an `unknown` bucket holding that
session, and the totals across all buckets equal the ungrouped totals; the
filtered result excludes it and says so in `notes` with the count excluded.

**48. Every input value reaches the archive.** (rules R10, R11)
Slice: 1.
Input: generated archives (layer 3) and, with `POLYLOGUE_TEST_REAL_SAMPLE`,
real stored files. A string leaf that matches a credential pattern is
checked through the stored input (`input cat`), not the export, which masks
it.
Operation: converge; export everything.
Compared: every string leaf longer than 16 bytes in the input's records is
found in the export (as text, inside `attrs`, or reachable through the
message's raw anchor), or its JSON path is on the format's ignore table in
`spec/sources/`. The failure output lists the first ten missing values by
JSON path and record type.
Input (slice 2): also a progressive-reasoning fixture (a Codex `Reasoning` item
repeated with its `summary_text` growing): the longest version reaches the
archive.

**49. A session with messages never parses to nothing.** (rule R17)
Slice: 1.
Input: generated sessions, and every example input whose provider records
include at least one message record.
Compared: each such session has at least one message after convergence;
otherwise `status` lists it under parse problems with the input and reason,
and the law fails.

**50. Merging ignores volatile fields.** (rule R22)
Slice: 4.
Input: one ChatGPT conversation as two export vintages: messages reordered,
an attachment's bytes present in only the later one, generation-timing
fields changed, Unicode spellings unchanged.
Compared: the same message IDs in both; the merged session has every
message once; no conflict event is recorded. The attachment whose bytes are
present only in the later vintage has the same attachment block ID in both
(its fallback hash never includes acquisition state), and a claude.ai
attachment seen without bytes in an export and with inline bytes in a
capture is one attachment.

**51. Two sessions sharing a provider ID stay apart.** (rule R23)
Slice: 5.
Input: a Gemini CLI main checkpoint and a subagent checkpoint that carry the
same provider session ID in different files.
Compared: two sessions with distinct IDs, each with only its own messages.

**52. Every surface returns the same rows.** (rule R28)
Slice: 1 (CLI and MCP); 10 (HTTP).
Input: Morrow; the fixed query list.
Operation: run each query through `polylogue --json`, the MCP `search` tool
over its wire protocol, and `GET /api/v1/query`.
Compared: `rows`, `count`, `outcome` and `next_cursor` behaviour are equal
across the three (surface-only fields such as `up_to_date` aside).

**53. Every parameter is honoured or refused.** (rule R29)
Slice: 1.
Input: the parameter declarations from which the CLI, MCP and HTTP
arguments are generated.
Operation: for each declared parameter, set it to a value that should
change the result (and to an invalid value) on every surface.
Compared: the plan or result changes, or the call is refused with
`invalid_query` naming the parameter; a parameter accepted and ignored
fails. Generated from the declarations, so a new parameter is covered
without a new test.

**54. A mutation over a query acts on every match.** (rule R30)
Slice: 7.
Input: Morrow plus 300 generated sessions matching one query.
Operation: `tag` and an `excise` preview over that query with a page size of
50; then `excise --yes` after one more matching session arrives.
Compared: every match is tagged, not the first page; the excision removes
exactly the IDs its preview listed, not the newly arrived session.

**55. Every stored blob is reachable.** (rule R40)
Slice: 3.
It needs `output_fts`, which arrives in slice 3; its excision case runs from slice 7.
Input: Morrow after convergence, after excisions and after the crash points
of law 7.
Compared: every blob in the store is referenced by an input revision's
extents (attachments are inputs); every attachment row belongs
to a message; nothing references a blob that is missing.
Every `text` row is in exactly the search indexes its blocks require
(`text_fts` for prose, `output_fts` for tool output), counted from the blocks
independently of the writer (rule R62).

**56. Excision reaches every table.** (rule R30)
Slice: 7.
Input: the DDL in `spec/storage/`.
Operation: the test lists every table with a column holding a session,
message, block, text or input key, excises one Morrow session, and queries
each listed table.
Compared: no listed table still holds the excised keys. A new table with
such a column and no excision handling fails this law without anyone
writing a new test.
Canary variant (content outside key columns): plant a unique 32-byte string
in the excised session's prompt, a tool call's input, a tool result, a
title note, a judgment note and a kept result. After excision and
convergence, scan the decompressed bytes of every file under the archive
directory (databases, WAL files, blobs, snapshots) and of a full export:
the string appears nowhere. Known-bad: a `kept_result.rows_json` that keeps
the displayed value.

**57. Twice the data, about twice the work.** (rule R47)
Slice: 1.
Input: generated archives of size N and 2N with the same composition.
Operation: converge each from empty; run the fixed query list on each;
render each session.
Compared: for each stage (collect, store, parse, write, derive, query,
render), the counted work (rows read and written, bytes read, SQLite
statements stepped) on 2N is at most 2.2 times that on N. Counted, never
timed, so it is stable on any machine.
Also live appends: appending k new records to one session of a converged
archive updates `seq`, the session's facts and its tree's cost with work
proportional to k, and the result equals a full recompute from empty.

**58. A large file stays within the memory target.** (rule R49)
Slice: 1.
Input: a generated 2 GB Claude Code file, and (slice 2) a generated Codex
rollout with a single 70 MB line.
Operation: converge in a child process.
Compared: the child's peak resident memory stays within `behavior.md`'s
target, and the files parse completely.
Input: also 20 files of 250 MB arriving at once: each is streamed (it is over
the 64 MiB read-whole threshold, so each counts only its read buffer), the
bytes held across parse workers never exceed 512 MiB, and peak memory stays
in target (`behavior.md`, memory).
Repeat with many small unique records rather than a few large text records,
so UUID/parent/seen maps dominate. After its first build and again after a
restart, append one ordinary record: no step clones or serializes the complete
prefix state. Measure retained carry, decoded objects, cache and serialization
buffers as well as input bytes; any spill/lookup work is included in the step
budget. Repeat with an open split response and long parent chain. The final
rows and record accounting match reference reconstruction. A small read buffer
alone does not qualify this law.
Repeat with one accepted result larger than the parsed-batch channel's
capacity, including a single large text below SQLite's value limit. Force
writer backpressure, cancellation and restart during delivery. Bounded
fragments or retained references make progress without an impossible
whole-result reservation or an uncharged payload; readers never see a
partially selected message. Final bytes, rows and accounting match the
reference reconstruction.

**59. One large input does not starve the others.** (rule R51)
Slice: 1.
Input: the 2 GB file of law 58 plus 20 small session files arriving while
it is being parsed.
Compared: with enough large-input work remaining for multiple rounds, each
small input receives a turn in the next eligible fair round and is parsed
before the large one finishes. A small input requiring one step completes
in its own turn; twenty such inputs require twenty writer turns, not one
shared slot. Keep a live input hot and admit further inputs during the run:
priority orders a finite round but does not indefinitely postpone an already
waiting peer. Measure turn order separately from elapsed time and IO waits.
Also retain an acquired unfinished record: its unchanged suffix waits without
spinning or advancing the complete-record checkpoint, peers and any owed row
finalization still run, and extending the acquisition makes it eligible again.
`status` distinguishes waiting for input from runnable finalization and shows
the last committed unit of work throughout. Repeat during a merge changing existing rows and
law 6's rebuild, including ordering, obsolete-row deletion and finalization.
Inject a small write between large-session writer steps in each phase: it
commits before the large session finishes that phase, without waiting for a
whole-session transaction. Inspect transaction work as well as input bytes:
64 MB of parsed input cannot excuse a transaction touching the whole huge
session. Intermediate reads retain required messages and mark unfinished
sessions and aggregates as specified; the final result equals a fresh build.
Payload-lifetime controls retain a borrowed input buffer through a queued
batch, cancellation and retry: its admission charge remains until the last
physical borrower releases it. Shared storage is counted once; distinct
copies, including overlapping buffers, stay charged while live. These
controls inspect retained storage and admission directly; law 58 separately
measures peak resident memory.

**60. A tool result pairs with one call.** (rule R59)
Slice: 1.
Input: generated sessions with orphan results, results before their calls,
re-emitted calls (same call ID and input under a new message), empty
outputs and provider error texts; plus
`examples/sources/claude-code/reemitted-tool-call/`.
Compared: each result pairs with at most one call, in the same session,
with the same call ID, and never with a call that comes after it; a
re-emitted call is one `tool_call` row; an explicitly empty answer remains a
result, whose status follows provider evidence (including a documented
status default). Empty text without status is `unknown` /
`result_without_status`, not success or `no_result`; a provider error text
is kept beside the output.

Input (slice 4): ChatGPT native graphs with tool-role and `real_author`
answers to known parent calls: ordinary text, fallback-only `content.text`,
empty `parts`, explicit empty text, absent textual body, a stated success
and a stated failure; also an unlinked result. Expected calls/results are
enumerated from the native graph independently of parser output. Compare
their identities, body presence/text, pairing and outcomes, not message
counts alone. Known-bad cases drop a result block while retaining its
message/text, or turn a status-less empty answer into `ok`; both must fail.
These provider variants require slice-4 execution; the present prototype
does not establish their acceptance.

**61. One writer, even through a symlink.** (rule R35)
Slice: 1.
Input: Morrow; the archive directory reached through two paths (its real
path and a symlink to it).
Operation: start `polylogue daemon` on one path, then start a second daemon
on the other path.
Compared: the second exits at once with an error naming the first daemon's
process and the archive's resolved path; the archive is unchanged by it.

**62. Readers never write and never open a second descriptor.** (rule R38)
Slice: 1.
Input: Morrow after `daemon --until-idle` has converged; a daemon started on
the archive, then write permission removed from everything in the archive
directory but `-shm` and the lock file (collection finds nothing to store).
Operation: send every read request (every query example, `read`, `status`,
completion (slice 3), MCP `search` and `read`, the HTTP API routes (slice
10)) to the daemon, with the IO-counting wrapper of law 18 on its file
layer.
Compared: every read succeeds; no file in the archive directory is created
or modified (listing and mtimes before and after); no reader opens a
database in write mode; the wrapper records no `open` of `*.db`, `*-wal` or
`*-shm` outside SQLite itself.

**63. A failing parser loses only its input.** (rule R39)
Slice: 1.
Input: Morrow; the test-only `fault-injection` feature makes the parser of
one chosen input panic, and in a second run reproducibly stop byte and
computation progress while runnable (a hang
past the stall limit, reached by advancing the test monotonic clock,
`plan/slice-01.md`, test clock); a third run makes the writer receive `SQLITE_BUSY`
twice before succeeding; a fourth makes one input's parse slow but
progressing (each record delayed) well past the stall limit in total.
Further runs keep one valid record unfinished while exposing byte or bounded
computation progress, pause its IO, starve its worker, and cancel its input
read before EOF. Resume each transiently interrupted run without changing
the bytes or parser version.
Operation: converge.
Compared: the chosen input has a `parse_state` status of `failed` with the reason
"panic" or "stalled" and is retried only after its parser version or
bytes change; every other input's part of the view equals the uninterrupted
view; the slow but progressing input finishes and its view equals the
uninterrupted one (time never fails progressing work, `polylogue-itddh`). Known-bad: a
per-input wall-clock limit; the `SQLITE_BUSY` run records no failure and its view equals the
uninterrupted one.
The single-record progress run is not failed as stalled. IO wait, starvation
and cancellation retain committed progress and retryable owed work; after
resumption their views equal the uninterrupted run, and cancellation never
certifies a complete parse. Only the injected reproducible parser hang joins
panic in the failed-until-bytes-or-version-change expectation above.

**64. A rebuild leaves one index.** (rule R41)
Slice: 1.
Input: Morrow.
Operation: converge; raise `INDEX_VERSION` (test-only setting) and converge;
raise it again and converge; then converge twice more with nothing new.
Compared: the archive directory holds exactly one index file (plus its WAL
and shared-memory files); the last two idle rounds leave every table's row
count unchanged.

**65. Unknown enum values round-trip.** (rule R43)
Slice: 1.
Input: Morrow after converging, then rows written directly by the test with
values this binary does not know: an origin `future-tool`, a note kind
`future-kind` (slice 7), a link `how` of `future-how`.
Operation: run `read`, `status`, export, and converge with a new input.
Compared: no command errors; the export carries each unknown value exactly
as written; rows the writer rewrites for other reasons keep the unknown
values unchanged.

**66. Deadlines and pool sizes.** (rule R50)
Slice: 1.
Input: Morrow; the process started under a CPU affinity mask of two CPUs
(`taskset -c 0,1`), and in a second run inside a cgroup with a CPU quota of
200% where cgroups are available; a test-only hook makes one `git` call hang.
Operation: converge.
Compared: `status --json` reports two parse workers in both runs; the hung
call is cancelled at its deadline and recorded as failed with reason
"timed out", and convergence finishes.

**67. Which archive a command uses.** (rule R53)
Slice: 1.
Input: a default archive with Morrow and a second directory with a
generated archive; environment variables named like an archive location
(`POLYLOGUE_ARCHIVE`, `POLYLOGUE_HOME`) pointing at the second; a ZIP whose
member names include `../` and absolute paths.
Operation: run `status`, a query and `import` without `--archive`, then with
`--archive SECOND`; (slice 8) run `polylogue demo --archive EXISTING` on a
non-empty directory demo did not create; import the ZIP.
Compared: without `--archive` every command acts on the default archive and
the environment variables change nothing; every `status` and every error
names the directory used; the demo refuses with an error naming the
directory; the escaping ZIP members are refused with a reason and nothing
is written outside the archive.

**68. Downloads stay on provider hosts.** (rule R55)
Slice: 4.
Input: a local test HTTP server registered as a provider host (test-only
allow list); attachments whose download answers with a redirect to
`127.0.0.1`, to a link-local address and to an unlisted host, and one whose
URL points directly at a private address.
Operation: converge.
Compared: each of those downloads is refused and recorded with the reason;
no request reaches the redirect targets (the test servers count requests);
the allowed download is stored.

**69. Calendar windows.** (rule R66)
Slice: 3.
Input: generated sessions with messages at the instants around the start and
end of October 2026 in `Europe/Warsaw` (summer time ends on 25 October) and
around a subscription window boundary.
Operation: `group by month`, `since:1mo`, and the subscription window
report, with the time zone set to `Europe/Warsaw`.
Compared: each message falls in the half-open window `[start, end)` that
contains it in local time; October's window is 745 hours long and
September's 720; no message is counted in two windows or in none.

**70. Unpriced models and credits.** (rule R33)
Slice: 3.
Input: Morrow plus one session whose messages name a model missing from
`prices.toml`, and one subscription session reporting credits.
Operation: converge; `status --json`; `analyze --cost`.
Compared: the unpriced session's cost is unknown (`null`), never `0`, and
totals say how many sessions are unpriced; `status` lists the model under
unpriced models; credits are reported in their own column and never added
to a USD total.
Input: also a session with one unpriced message among priced ones: its
cost is shown as a known part plus an unknown count, is never compared or
sorted as if complete, and archive totals count it among partly priced
sessions.

**71. An older binary does not write to a newer archive.** (rule R68)
Slice: 1.
Input: Morrow after converging, with `archive.db`'s migration step and
`INDEX_VERSION` set one higher than this binary knows (test-only setting).
Operation: start the daemon; run `note` and `tag` (slice 7), `import`,
`status`, a query and `polylogue --version`.
Compared: the daemon refuses to start and exits non-zero, naming both
versions; every other command prints the one-line not-running message;
`polylogue --version` prints the binary's version and the schema versions it
knows; nothing in the archive changes.

**72. Content never changes structure.** (rule R18)
Slice: 1.
Input: generated sessions, each in two variants identical except that
message text blocks (not tool-result output) contain decoys: harness-like markers (`<command-name>`,
`[Request interrupted by user]`), tool-call-like JSON, `error:` and
`exit code 1` lines, and a known file-format prefix inside a text block.
Operation: converge both variants.
Compared: `written_by`, roles, tool outcomes, session boundaries and the
number of messages and blocks are identical between the variants; only the
text differs.

**73. Decoys stay decoys.** (rule R16)
Slice: 1.
Input: the decoy set of section 10 (files under provider locations that are
not sessions: settings files, empty JSONL, backups, a ZIP of unrelated
files).
Operation: converge.
Compared: every decoy has a `parse_state` status of `no_sessions` or is skipped
with a reason, and no session is created from any of them.

**74. Exports replace files atomically; IDs survive URLs.** (rule R69)
Slice: 1 (export); 10 (HTTP routes).
Input: Morrow; IDs containing `:`, `/`, `%`, `#`, spaces and non-ASCII
characters (generated sessions).
Operation: `export --to FILE` over an existing export, killed at each write
step; (slice 10) every HTTP route that takes an ID, called with each such ID.
Compared: after each kill, `FILE` is either the complete previous export or
the complete new one; each route returns the session the ID names, so every
ID is encoded once and decoded once.

**75. Destructive commands ask; agents cannot excise; claims are labelled.**
Slice: 7.
(rule R70)
Input: Morrow with one agent-written `claim` note.
Operation: run `excise` without `--yes` in a non-interactive shell; list the
MCP tools; render a briefing for Morrow's directory.
Compared: `excise` prints what it would remove, changes nothing and exits
with code 3; no MCP tool can excise or hide content permanently; the
briefing shows the claim labelled as a claim with its author.

**76. Counts match rows; one-row lookups are unique.** (rules R7, R76)
Slice: 1.
Input: Morrow and generated archives; every query example and generated
queries including `kind:`, (slice 3) `inherited` and (slice 6) `parent:`.
Operation: compile and run each query; page to the end; run every one-row
lookup against an archive where two rows share the looked-up value.
Compared: the result's `count` equals the number of rows paged; an ambiguous
lookup is an error naming the matches. Known-bad: a build where the count
compiler drops the default filter.

**77. The token goes only to the operator's listener.** (rule R72)
Slice: 10.
Input: a test daemon; the socket-owner lookup stubbed to another uid, then to
fail.
Operation: run `polylogue open` and the browser host; connect from a
stubbed foreign uid.
Compared: no client sends the token; the daemon closes the foreign
connection; with the lookup failing, both refuse. Known-bad: a client that
skips the owner check.

**78. The WAL stays bounded.** (rule R75)
Slice: 1.
Input: a generated archive large enough for a long first build.
Operation: build while a reader pages continuously through a session list.
Compared: after each page completes, the WAL returns under
`journal_size_limit` within one checkpoint step. Known-bad: a reader that
holds its transaction across pages.
Variant: with a reader running a 20 s `analyze` and an idle moment that
allows a `RESTART` checkpoint, the writer never waits on a reader (its
counted lock waits stay 0; a checkpoint never blocks it), every append made
during the `analyze` is indexed in the round after it arrives, and a read
past its time limit ends with an error naming the limit. Wait times are a
benchmark (`plan/testing.md`).

**79. Ambiguity stays ambiguous.** (rule R76)
Slice: 1 (ID prefixes); 4 (sandbox links).
Input: generated archives where a short ID prefix matches two sessions and
a note names one of them; (slice 4) a ChatGPT conversation whose sandbox
link names none of its message's files.
Compared: resolving the prefix is an error listing both sessions; the
sandbox link gets no bytes, with a reason. Known-bad: a resolver that falls
back to another stage when the first finds several.

**80. Briefings fit their budget.** (rules R77, R7)
Slice: 2.
Input: generated candidate sets and budgets from 200 to 8,000 tokens.
Compared: the rendered briefing's UTF-8 bytes / 4 is at most the budget,
and each delivery row's `bytes` equals the rendered text's length.
Known-bad: a reducer that lowers its declared cost without shrinking its
text.

**81. A rebuild adds no archive rows.**
Slice: 1.
Input: Morrow, converged. Operation: delete `index.db` and converge again.
Compared: `archive.db` row counts per table before and after are equal (no
new input, revision or observation rows), and the view equals the one
before. Catches an index-side map from path to input being treated as the
authority (`spec/convergence.md`, collect).

**82. Same top-k either way.**
Slice: 1.
Input: generated archives; text queries combined with selective and
non-selective conditions. Operation: run each query with the planner forced
to drive from the conditions, then forced to drive from the full-text hits.
Compared: the same rows in the same order, including ties
(`spec/query.md`, full-text search).

**83. The socket and the library give the same rows.**
Slice: 1.
Input: Morrow, converged; every query example and every MCP and CLI read of
the slice.
Operation: send each as a request over the daemon's socket, and, after
stopping the daemon, run the same query through the engine library
in-process on that archive (the test process is then the only opener).
Compared: identical rows, counts, outcome and cursor, byte for byte in the
JSON form (`spec/surfaces/README.md`, processes).
Known-bad: a socket handler that applies a default filter the library does
not.

**84. Daemon stopped: one line, no database.**
Slice: 1.
Input: Morrow, converged; the daemon stopped.
Operation: run every command and MCP tool of the slice (from slice 3,
live-value completion offers nothing and static completion still works;
from slice 2, the briefing hook), with file opens traced.
Compared: each command that needs the daemon prints the one-line "not
running" message naming how to start it and exits 6; `--help`, `--version`
and static completion work and print no message; the briefing hook prints
nothing and exits 0; (slice 2) the hook recording entry still appends its
spool line;
no process opens any database file.

**85. Live state is a function of stored inputs and the time.**
Slice: 2.
Input: the live set (`examples/archive/FACTS.md`, "Live set"), plus a
mutation that appends `Stop`, then `SessionEnd`, to V1's spool file.
Operation: feed the live set's lines up to 13:30 and ask at 13:30; feed the
rest and ask at 15:00, with no write between asking and the next feed; ask
after each mutation; delete `index.db`, converge again and ask at 15:00.
(Live columns hold the latest facts, so a state at an earlier time is asked
with the inputs stored up to that time.)
Compared: the states FACTS.md lists (V1 `waiting`, V2 `working`, V3 `stale`,
V3 `working` at 13:30), V1 `idle` then `ended` after the mutations, and
identical states, `until` values and evidence before and after the rebuild
(`spec/model.md`, Live state). Each state carries `until`: the end of the live window (15 minutes after
the last live evidence) or of the 24 hours after it, after which `stale`
becomes `ended`. Asking twice at different times with no write in
between gives the states for each time.
Known-bad: a state written by a timer instead of computed at read time.

**86. A reused PID never inherits.**
Slice: 2.
Input: the live set plus a mutation adding a hook spool for a new session
whose harness has V3's PID 5303, once with a different start time and once
with a different boot ID. Operation: converge.
Compared: each process line attaches only to the session whose process
identity (boot ID, PID, start) it carries; the later session is never
`working` because of the earlier one's lines.
Variant: two hook invocations have distinct hook-process PID/start pairs,
including reuse of one hook PID with a different start on the same UTC day.
Their spool files are distinct (`<hook-pid>-<hook-start>.ndjson`), while
each line carries the harness identity; no event is lost or attributed to
the other harness. File identity and harness identity are tested separately.

**87. A dead process ends liveness.**
Slice: 2.
Input: the live set's V3 (working until its `root_gone` line at 13:40),
then a mutation adding a hook line for V3 from a new process identity (a
resume). Operation: ask the live state after each.
Compared: `stale` (no `SessionEnd`) after the `root_gone` line at any later
time; live again only after the new identity's line.
Variant: V3 with the daemon stopped before its harness exits and
restarted after: `stale` after the restart, with a `root_gone` line written
at the restart for the root that died while the daemon was down.

**88. The process sampler is proportional.**
Slice: 2.
Input: a fake `/proc` tree under test control. Operation: run the sampler
with no live roots, then with roots whose trees do not change for 30
samples, then with one child started and one ended.
Compared: no `/proc` reads with no live roots; no spool lines while nothing
changes; exactly one line naming the started and ended child.

**89. Peers are symmetric.**
Slice: 2.
Input: the live set (V1 and V2 are peers, V3 is not), then generated sets
of live sessions over shared and separate repositories and worktrees. Operation: list each session's peers at one
time. Compared: B is among A's peers exactly when A is among B's.

**90. No task from prose.**
Slice: 6.
Input: a session whose prompt and messages mention Bead IDs, and which runs
`bd show` on one, plus a session that runs `bd update <id> --claim` and
later `bd close <id>`. Operation: converge and ask the live task.
Compared: no task claim in the first session; in the second, the claimed ID
until the close and none after it.

**91. Cached and uncached answers are equal.**
Slice: 1.
Input: Morrow; each query example.
Operation: run each query twice; between the runs do one of: a tag, a hide
and a saved-query revision (slice 7), a setting change (slice 3), a write that
commits while a slow read is in flight; or send the next request with `now`
moved past a `today` boundary or (slice 2) past the end of a session's live
window, with no write.
Compared: each answer with the cache on equals the answer with it off; the
in-flight read's result is never served to a request whose snapshot opened
after the write. Known-bad: a cache key of the index counter alone.

**92. Excision is immediate for reads.**
Slice: 7.
Input: Morrow; a text that a cached result, a held paging cursor and a
rebuild beside the old index all contain, and (slice 9) a stored vector and
an embedding request in flight.
Operation: excise it; when `excise` returns, query, page on, brief, and
(slice 9) run `near:` and let the embedding request complete.
Compared: none of the answers contains the text, and no vector for it is
written back.

**93. Write requests are idempotent.**
Slice: 7.
Input: Morrow; each write request kind (`note`, `tag`, `untag`, `hide`,
`unhide`, title, outcome).
Operation: send each request twice with the same request ID, and once more
after killing the daemon between commit and reply; then undo it once
(`untag`, `unhide`).
Compared: the `archive.db` rows equal one send; the undo leaves no tag or
hide in force.

**94. A derived product converges to its current version.**
Slice: 3.
Input: Morrow.
Operation: change `prices.toml`; kill derive part-way; change it again (and
separately revert it); converge. The same with a derived-product version
bump.
Compared: every session's cost, tree cost and facts equal a run that started
at the final prices; `up_to_date` is false until the walk ends.

**95. The change feed replays to the archive.**
Slice: 7.
Input: Morrow and a consumer that follows `changes --since`.
Operation: while the consumer follows, add notes, import a Polylogue export
from another archive whose notes have earlier UUIDv7 values, import an old export holding a marker,
excise a session, make a session vanish (a parser fix turns its input into
`no_sessions`), change `INDEX_VERSION`, and apply `kill_writer(n,
after_commit)`.
Compared: the consumer's replayed state equals the archive's export; every
new note is reported exactly once; the excised and vanished sessions are
reported as removed.

**96. Codex usage sums to the reported total.**
Slice: 2.
Input: Codex rollouts written by the hand-made builders (extended to Codex
in slice 2) with repeated `token_count` events at the measured rate; (slice
8) the same over generated Codex inputs; with `POLYLOGUE_TEST_REAL_SAMPLE`,
real rollouts.
Compared: for every input, the sum of stored per-request usage equals the
final `total_token_usage`, or the input is listed in `status` with the
difference. Each model request is one message.

**97. Command normalization ignores prefixes.**
Slice: 2 (briefings' unresolved failures need it).
Input: generated shell commands.
Compared: the normalized command is unchanged by prefixing `cd <path> &&`
or `NAME=value`; `a | b` normalizes its failure form to `b`; `git add -A &&
git commit -m x` normalizes to `git commit`.

**98. The same act has the same author everywhere.**
Slice: 2.
Input: a parent-to-child follow-up in Claude Code (coordinator) and in Codex
(`agent_message`); a slash command; a harness-generated instruction.
Compared: both follow-ups are `written_by: model`; the slash command is not
a human prose prompt (`prompt_kind`); the generated instruction is
`harness`; `human_prompts` counts only `prompt_kind` prose.

**99. No branch without two message children.**
Slice: 1.
Input: the Morrow split response extended with two tool calls whose results
interleave with the response's records; generated archives.
Compared: a session whose record tree has no record with two message-record
children has every message `on_active_path = true`.

**100. An edited revision replaces, it does not union.**
Slice: 5.
Input: an AI Studio prompt whose second turn is edited between two Drive
revisions.
Compared: the session has as many messages as the newest revision; the
edited turn's old text is not a message.

**101. Live state is total.**
Slice: 2.
Input: historical sessions of every origin built so far (Claude Code and
Codex from the hand-made builders in slice 2; (slice 8) a generated archive
of every origin), none with hooks or process lines, asked at a `now` one day after their last record.
Compared: every session has exactly one live state from the table in
`spec/model.md`, here `ended` (no live evidence for 24 hours); none is
`running`, none is counted in `peers_live`; asked one hour after a hooked
session's last line, it is `stale`.
Known-bad: an old session ending in a permission request shown as
`waiting`.

**102. A clock-dependent outcome is computed at read time.**
Slice: 2.
Input: the live set plus V3 with one unanswered call.
Operation: ask the call's outcome at 13:30 and at 15:00 with no write
between; delete `index.db`, converge at 15:00 and ask again.
Compared: `pending` at 13:30, `unknown` (`no_result`) at 15:00, the same
after the rebuild. Known-bad: an outcome read from a stored column that
holds `pending`.

**103. Process times come from stored bytes.**
Slice: 2.
Input: the live set.
Operation: converge; rebuild with a fake `/proc/stat` whose `btime`
differs.
Compared: identical process `started_at` values and tool-call links.

**104. Activity is measured from the append, not the first sighting.**
Slice: 2.
Input: an append-only input observed on day 0 and appended on day 9 with
day-9 times; separately, a record dated day 30 appended on day 9.
Compared: `last_activity_at` is day 9 and nothing is flagged; the day-30
record is flagged as far in the future.

**105. Offsetless local times are placed in their zone.**
Slice: 5.
Input: a Hermes JSON file (local time without an offset) and the `state.db`
rows of the same session, recorded with the collector's zone.
Compared: equal `started_at`, and no conflict event.

**106. Session times cover own messages.**
Slice: 1.
Input: Morrow.
Compared: #4's `started_at` is 2026-10-03 14:00 local, its engaged time
equals that of its own messages alone, and (slice 3) `--by day` places #4
on 3 October only (`FACTS.md`). Durations with equal or reversed times are
`unknown`, never negative, engaged-time gaps included.

**107. Sequence patterns follow session order.**
Slice: 3.
Input: a session whose provider times run backwards across a failing and a
succeeding command.
Compared: `seq(...)` and "later success" match by order in the session, and
match rows are listed in file order.

**108. An older copy stays older.**
Slice: 2.
Input: a pre-rewrite copy of a Codex rollout, migrated with a fresh mtime
after the live file is stored.
Compared: the live chain stays current, the next append is classed
`appended`, and the copy's per-record timestamps win where rule 5 says the
older copy is faithful.

**109. Subscription windows anchor to reported resets.**
Slice: 3.
Input: a fixture with a harness-reported reset; then the same fixture with
one earlier message inserted.
Compared: the windows after the reported reset are unchanged, and windows
inferred without a report are labelled inferred.

**110. Every agent string gets rows or a corrected query.**
Slice: 3.
Input: strings an agent might pass to search: bare words, a pasted session
ID, a URL, a quoted phrase, unbalanced quotes, a field name with a typo.
Compared: each returns rows, or an error that names the problem and gives a
corrected query to try (rule R76: no silent weaker search); never an
unexplained error.

**111. A provider database reads consistently while the provider writes.**
Slice: 2.
Input: Codex `state_5.sqlite` with a writer holding a write transaction
(a test thread inside the fixture).
Compared: collection neither blocks past its deadline nor stores a
half-written state; the stored snapshot equals one the writer committed.

**112. Every export key is mapped or excluded.**
Slice: 4.
Input: each committed export example (ChatGPT, claude.ai, then each format
as its slice lands).
Compared: every JSON key path in the example is on the format's mapping
table or its ignore table with a reason (`spec/sources/`); a new key fails
the law and is reported by name. This extends the mapping-table drift
check to export payloads.

**113. A hook event never creates a session.**
Slice: 2.
Input: hook spool lines for a session with no transcript, and a hook
payload that nests a transcript.
Compared: no session exists until a transcript or export provides it; the
events attach when it does; the nested transcript is not parsed as a
session (it is counted as unrecognized).

**114. Unknown never becomes a match or a zero.**
Slice: 3.
Input: Morrow plus a session with an unknown active path, a tree with a
recorded child that has no session, (slice 9) a session with no vectors,
and fields missing in some sessions.
Compared: `NOT field:x` never matches a session whose `field` is unknown
(it is listed as unknown); tree cost reports the missing child as missing,
not zero; (slice 9) `near:` with candidates lacking vectors says how many
were left out; a degraded answer exits with the code `spec/surfaces/cli.md` gives for
degraded, never 0 or the no-match code; a briefing whose section failed or
timed out says "could not read" rather than "none".

**115. A device or host rename re-stores nothing.**
Slice: 1.
Input: Morrow converged; then the same home seen with a different device
number (a remounted disk) and a renamed host (same `/etc/machine-id`).
Compared: the next convergence adds no revision and no blob, and classes
every file `unchanged`.

**116. A stored hook line is one event, however often it is stored.**
Slice: 2.
Input: one hook spool file in which the same harness event is written by
two hook invocations, and the same spool file stored twice.
Compared: event identity comes from the hook process and line position, so
re-storing the file adds nothing; two invocations are two lines and two
events, each kept, and a delivery is recorded once per sender ID.

**117. Reading a provider database never writes it.**
Slice: 2.
Input: a WAL-mode SQLite database closed cleanly (no `-wal` or `-shm`
beside it), the same database while a writer commits during the snapshot,
and the same database in a directory Polylogue cannot write.
Compared: after each snapshot the database file's size, modification time
and BLAKE3 equal their values before it (the database file only: an
existing `-shm` may change, since SQLite's readers write it); the snapshot equals the content of
the writer's last commit before the read began (a transaction rolled back
during the snapshot never appears in it, and a commit made after the read
began is acquired by the next pass; `polylogue-0ku83`); in the unwritable directory
the snapshot succeeds with `immutable=1` recorded and no file is created;
the only files ever created or changed beside the database are its `-wal` and `-shm`
(`spec/storage/blobs.md`, provider databases).

**118. A claim's check follows the archive.**
Slice: 7.
Input: Morrow with a claim on session #2, "the edit to `src/parse.rs` made
the tests pass", whose check is a `seq(...)` over its `cargo test` runs,
scoped to its own session (`id:claude-code:7f3a2c10-0002 seq(...)`)
expecting `rows`; the archive converged first without the passing run, then
with it; then with session #2's latest input not yet parsed.
Compared: the check state is `fails` (listing the rows returned) before
the passing run, `holds` after it, and `unknown` while the input waits; no
judgment note is ever written by the check; `notes where check:fails`
returns exactly the failing claims; the state is never stored.

**119. A pass writes only what its rubric allows.**
Slice: 7.
Input: Morrow plus the annotation set (`examples/archive/FACTS.md`,
"Annotation set (slice 7)"): its rubric and pass, plus generated calls
writing a note of a kind the rubric does not list and an `outcome` with a
value it does not allow.
Compared: the pass writes exactly the four notes FACTS lists and refuses
`toolu_3004`; every disallowed note is refused, each refusal names the rubric
revision and the disallowed kind, key or value; the written note has a
`derived_from` relation to the rubric's pinned revision; withdrawing
`notes where rubric:<root>@<rev>` withdraws exactly that pass's notes.

**120. Every framed record is accounted for.**
Slice: 1.
Input: a Claude Code session file with ordinary records, one line that is
not JSON, one record of a listed ignored kind, one record of a type the
provider does not know, and one assistant response split over three
records.
Compared: per input, `Used + Ignored + Unrecognized + undecodable` equals
the number of lines framed; the unknown record is counted as unrecognized
with its byte offset; the three split records give one message whose
`source_start`/`source_end` cover all three; `status` shows the counts.

**121. No edit is read as unchanged or absorbed.**
Slice: 1.
Input: an append-only file observed, then (a) edited in place with the
same length within the same timestamp tick as the observation, and (b)
edited in the middle at an offset no sampled window covers, then appended
to, then left quiet; and a file appended to continuously for 7 hours.
Compared: (a) is classed `rewritten`, not `unchanged`; in (b) the append is
stored as a piece, the stored revision hash equals the hash of the stored
bytes (not of the live file), and the quiet-time full check classes the
file `rewritten`; the continuously appended file gets a full check within
every 6 hours. A fuzz run of random edits on 4 MiB files ends with no
edit classed unchanged and every edit found by the end of the run.

**122. Status counters equal their counts.**
Slice: 1.
Input: Morrow converged; then an append, a rewrite of an input that
removes a session's messages and a parser version change, converging
after each. Variant (slice 7): an excision as a further step.
Compared: after each convergence every total in `meta` equals the `count(*)`
it stands for, and bare `polylogue` on a freshly started daemon reads those
counters plus at most one query over sessions active in the last 14
logical days, through the `session_activity` index (its statement log holds no
other `count(*)` over a session, message or tool-call table, and no scan
of sessions outside that window). How fast it answers is a benchmark
(`plan/testing.md`), not part of this law.

**123. Corruption in the archive's own files is found, never served.**
Slice: 1.
Input: Morrow converged into an archive directory without copy-on-write;
then one byte flipped in a page of `archive.db`, one in a blob file, and
`index.db` damaged so that SQLite reports it (a page header overwritten).
Compared: reading the `archive.db` page fails with a corruption error naming
the page (SQLite's checksum VFS) and `status` reports it, and no query
returns a row from it; reading the blob fails naming the blob (its BLAKE3);
the `index.db` error makes the daemon rebuild the file from `archive.db`
and the blobs. `index.db` has no checksums, so a flip SQLite does not
notice is not detected; it is rebuildable, and that is accepted
(`spec/storage/README.md`, the archive directory is not copy-on-write).

**124. A fresh archive equals an upgraded one.**
Slice: the first release with a migration step.
Input: for every earlier release's `archive.db` schema, an archive created
by that release with Morrow's notes, settings and excisions; and a fresh
archive created by this release with the same data.
Compared: after migration each old archive has the same schema as the fresh
one (tables, columns, types, indexes, `PRAGMA user_version`) and the same
rows; a migration interrupted at any statement leaves the copied-aside file
in place and the archive at its old version, and the next start completes
it. Each step that rewrites data runs in bounded batches and the step number
is written only with its last batch (from Sinex's schema-change prototype,
`sinex-spec/prototype/schema-changes/`).

**125. A restore from a copy reaches ok.**
Slice: 1.
Input: Morrow converged; a two-step copy of its archive directory
(`archive.db` first, `blobs/` after it, as `spec/storage/README.md` says a
backup should run), with one blob file deleted from the copy, as if joining
or excision had removed it during the copy; once with its provider file still
in place and once with it gone.
Operation: start a daemon on the copy and converge.
Compared: with the provider file present, the missing blob is stored again
and `status` is `ok` with every count equal to the original's; with it gone,
reading the affected input fails naming the blob, `status` counts it, and
every other session reads normally. A restore that was never drilled is a
hope (`polylogue-8jg9`).

**126. Filter composition.**
Slice: 1.
Input: Morrow and generated archives; pairs of conditions A, B on the
`sessions` unit's slice-1 fields; (slice 3) on every unit.
Operation: run `U where A B`, `U where A | where B`, and `U where A` filtered
by B row by row.
Compared: the same rows in the same order (`spec/query.md`, Composition).
Known-bad: a `| where` that re-binds defaults or runs before an earlier
stage.

**127. A saved query selects the same rows by name and inline.**
Slice: 7.
Input: Morrow with saved queries that are relations (one asks for wrappers,
`tool_calls where action:code`) and one shaped result (with `group by`).
Operation: run each relation inline, as `U where query:NAME`, and refined by
a redundant condition (`(action:shell OR NOT action:shell)`); use the shaped
result as a condition.
Compared: name and inline give equal rows; the refinement adds nothing and
the wrapper-asking query keeps its wrappers inside another query; the shaped
result is an `invalid_query` error naming its stages (`spec/query.md`, Saved
queries). Known-bad: splicing the saved text into parentheses, which lets
the outer query rebind its defaults.

**128. Attaching rows never changes which owners are returned.**
Slice: 3.
Input: Morrow and generated archives; session queries with and without
`with messages(...)`, `with tool_calls(...)[last:k]` and windows matching
nothing.
Operation: run each query with and without the attachment.
Compared: the same owner rows, in the same order, with the same `count`; an
owner with no matching children carries an empty list
(`spec/query.md`, one relation, several uses).

**129. A grouped result is rows.**
Slice: 3.
Input: Morrow; `tool_calls | group by action | count` and the same followed
by `| limit 1`, `| where count >= 6`, `| sort by count asc`.
Compared: `limit` keeps the first group of the grouped result, `where` keeps
the groups whose count passes, `sort` orders groups; none act on the
ungrouped rows (`spec/query.md`, Stages).

**130. The order rows are read in never changes a match.**
Slice: 3.
Input: Morrow with its messages and tool calls enumerated in 20 seeded
shuffles; `seq(...)`, `exists` and `| matches` queries, including
equal-time candidates.
Compared: identical matches and match rows in every shuffle (`spec/query.md`,
sequences). Known-bad: a match chosen by enumeration order.

**131. A ranking never changes membership.**
Slice: 1 (relevance); 9 (`near:`).
Input: Morrow and generated archives; queries with and without a text term's
relevance order or `near:`, with embeddings present, partly missing and
disabled.
Compared: the same candidate set and `count` in every case; only the order
differs; `near:"…" | agg count` makes no embedding call (`spec/query.md`,
membership, ranking, selection and page).

**132. Three-valued filters.**
Slice: 4.
It uses sessions #7 and #8 and `exists` over child units.
Input: Morrow (#7 and #8 have unknown cost; #8 has an unknown model) and
generated archives with unknown values in every nullable field.
Operation: for each condition C over unit U, run `U where C`, `U where NOT
C` and `U where F:unknown`; for each child condition, `exists` and `NOT
exists`.
Compared: the three results partition U with no overlap; no owner matches
both `exists` and `NOT exists` (`spec/query.md`, unknown under `NOT`).

**133. Defaults are bound once.**
Slice: 3.
It needs Codex sessions and the `messages` and `tool_calls` units.
Input: Morrow; each default of the Defaults table and a condition on its
field that every row satisfies, nested where it is not a top-level condition
(`(action:shell OR NOT action:shell)`, `(inherited:true OR NOT
inherited:true)`), and the same field as a top-level condition.
Compared: the nested condition leaves the rows unchanged (37 tool calls, not
42; 2 chrono messages, not 3); the top-level condition, or `F:*`, drops the
default as the table says (`spec/query.md`, Defaults, binding). Known-bad:
dropping a default when its field is mentioned anywhere (the Python
language's rule, found by the composability review, 2026-10-12).

**134. Every execution mode gives the same answer.**
Slice: 1 (pages, either search plan); 3 (`sql --jsonl`); 7 (kept results).
Input: generated archives; queries run interactively in one page, paged by
cursor with several page sizes, with the planner forced to each search plan
(law 82), (slice 3) exported through `sql --jsonl` in steps, and (slice 7)
kept and re-read as `result:<id>`.
Compared: the same rows, order, `count` and figures in every mode, at one
view (`spec/query.md`, four bounds).

Sources: `polylogue-ps055`, `polylogue-1tyq5.187`, `polylogue-re4a`, `polylogue-kwsb.3`, `polylogue-bt3c4`, joint #7, `polylogue-gy2yu`, `polylogue-54a31`, `polylogue-mqups`.

**135. A command word is never searched for.**
Slice: 1.
Input: queries starting with a command or verb name (`brief yesterday`,
`status`), with a near miss of a unit or command (`messges cargo`,
`sessin where repo:morrow`, `statsu`), and controls: the same words quoted
(`"brief" yesterday`), a field comparison (`cost > 2.90`), a correct unit,
and ordinary search words near a command name but under the length rule
(`date parsing`, `bug fix`, `test output`: one mistake is allowed only for
words of 4–6 letters, two for 7 or more, none under 4; `query.md`,
principle 2).
Operation: run each through every surface (CLI, socket, MCP).
Compared: the first group is refused with a suggestion and never returns an
empty search result; the controls run as queries. A silent empty search is a
wrong answer, so the refusal is the only correct one. Joint query law
"command words" (`polylogue-sinex-joint/query-laws.md`); checked by
`prototypes/query-ref/composition.py`.

**136. A provider time is never rejected for its value.**
Slice: 1.
Input: generated Claude Code sessions whose records carry a time in 1970, a
time ten years after collection, a time a week after the record's own
`first_seen`, an unparseable time text, and an offsetless time from a
format that declares none.
Operation: converge.
Compared: every record is stored and parsed; no input, record or message is
refused or dropped for its time. The far-future and far-past times are kept
as given and flagged, never used as a session's start, last activity or
freshness (`model.md`, Time); the unparseable and undeclared ones are
unknown and counted as `time_unparsed`. A session whose every time is
flagged has unknown times with reason `implausible_time` and is still
listed. Known-bad: an admission bound on times (Sinex found one in its own
admission, its law L51).

**137. A value outside a field's closed set is refused, never empty.**
Slice: 3.
Input: for each closed-set field, a value from another unit's field of the
same name (`sessions where outcome:error` (slice 7), `tool_calls where
outcome:solved`, `deliveries where outcome:ok`), a misspelt value
(`role:asistant`), and controls: every value of each set, `*`, and
`unknown` where the field allows it. Variant (slice 1): the closed-set
fields slice 1 has (`plan/slice-01.md`, query): `kind:subagent` passes,
`kind:continuation` (a value of the `links` unit's `kind`) and `kind:mian`
are refused with the set named.
Operation: run each through every surface (CLI, socket, MCP).
Compared: the first group returns outcome `error` naming the field's set
and, for a value that exists elsewhere, the unit that has it; the controls
run as queries. Known-bad: an implementation that treats an unlisted value
as a string match and returns outcome `empty` (`query.md`, match fields;
rule R2).

**138. Hiding never changes a usage figure.**
Slice: 7.
Input: an archive whose current subscription window holds several sessions;
then the same archive with one of them hidden.
Operation: read the window's usage, the usage outlook, archive totals and
`status` before and after hiding; list the sessions before and after.
Compared: every usage figure is identical, and after hiding it says one
hidden session is included; the session list drops the hidden one. Known-bad:
filtering hidden sessions before aggregation, which makes the allowance look
larger than the provider's count (`user-data.md`, hiding versus excising).

**139. A stale revision is refused on one machine.**
Slice: 7.
Input: a note N; a revision R1 of N written by the operator; then an agent
revising N (not R1), through the CLI `revise` and through MCP `note` with
`revises`; and a control revising R1.
Operation: write each through the daemon.
Compared: the revisions of N after R1 are refused, naming R1 as current,
and write nothing; the control writes R2 with `root` N and no conflict.
Conflicts arise only from a union (law 23) or from inline markers, which
are never refused as stale: a variant with a marker `::note(revises=N)`
extracted after R1 forks the family into a conflict, the same in any parse
order. Known-bad: accepting the stale
revision and showing a conflict on one machine (`user-data.md`, revisions).

**140. Signals behave as the joint cases say.**
Slice: 11.
Input: `signal` notes over the Morrow live set (`examples/archive/FACTS.md`,
live set), with the clock advanced through each case, one case per joint
case shape Polylogue facts can express: `sessions where live:waiting` with
`for: 20m` (V1 waits from 14:41, fires at 15:01 with no new data; a wait
answered before 20 minutes fires nothing); `on_new` over Morrow's failing
tool calls (each once); a rate limit of 1 per hour over the same rule (the
second deferred to the next briefing); quiet hours holding an interrupting
notice but not a passive one; three agent-written interrupting rules
exceeding the agent interrupt budget (the excess downgraded with a reason,
the operator's rule still interrupting); an agent rule withdrawn after its
first delivery.
Operation: converge and let the daemon evaluate the rules.
Compared: the `delivery` lines written (sender `signal`, recipient, rule,
rows, time, outcome, author) equal what the joint evaluator
(`polylogue-sinex-joint/prototype/signals/`) gives for the same case shape,
case by case, and nothing else is delivered; a withdrawn rule delivers
nothing. The joint semantics themselves (on_change modes, on_threshold) stay
tested by the `joint-signals` check. Known-bad: evaluating `for` only when
data arrives.

**141. The last run of each command equals a full recompute.**
Slice: 2.
Input: generated sessions of two repositories running overlapping commands,
some runs without a time; delivered in shuffled order, with an older
session imported after newer ones, one session rewritten so its last run
disappears, and one session excised (slice 7).
Operation: converge after each arrival and deletion.
Compared: `repo_command_last` equals the rows a full recompute from
`tool_call` gives (the latest run per repository and normalized command by
time, then tool-call ID, untimed runs last), after every step; no row points
at a deleted tool call. Known-bad: updating only from the session being
written, which a late older session or a deletion leaves wrong
(`storage/index-db.md`, `repo_command_last`).
