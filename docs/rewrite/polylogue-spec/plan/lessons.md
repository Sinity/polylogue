# Mistakes not to repeat

One line per lesson from the Beads that the rewrite makes obsolete (`evidence/beads/beads-switchover.md`, class *lesson*): the area is redesigned or removed, but what went wrong is worth keeping out of the new code. Each slice's design step reads the section for its area (`plan/method.md`). The grouping is rough; the Bead ID is the evidence and holds the full story.

## Ingest, parsing and storage

- Polylogue has no backup command (storage/blobs.md: backup is the filesystem's job); keep the blob write order (temp file, fsync, rename) in storage/blobs.md "Crash safety without repair" (`polylogue-027dv`)
- Aggregate in SQL, never hydrate rows to count them (workspace.md "counts, sums and joins are SQL") (`polylogue-0bn4p`)
- derived work recomputed from state each pass (`convergence.md`) (`polylogue-20d.14.9`)
- `archive.db` migrations are a short hand-reviewed list (`polylogue-377um`)
- executed command blocks instead of screenshots (`polylogue-3tl.9`)
- parser versions are numbers (`convergence.md`) (`polylogue-4e2qp`)
- one inferred schema per provider format (`polylogue-4xyn9`)
- schemas are offline only (`decisions/schema-validation.md`) (`polylogue-5qq1x`)
- one request shape over the daemon's socket, no per-command operation registry, receipts or contracts (`decisions/process.md`) (`polylogue-5vft`)
- An index rebuild is the ordinary "every input never parsed" case, never a separate cold-build or startup route (`convergence.md`, first build and rebuilds) (`polylogue-7hfu0`)
- One loop, one write path for every input (`convergence.md`); a second route is how steps get skipped (`polylogue-a0co6`)
- Commits are linked to sessions only by evidence in the session (`convergence.md` git lookups); the hook records the repository root it found, so a plain folder inside a repository is visibly that repository's, never its own history (`sources/README.md`, SessionStart `git`) (`polylogue-b5xfk`)
- Merging is a fixed per-field order independent of arrival (`convergence.md`, several inputs) and lineage keeps full lists (`decisions/lineage.md`), so these arrival-order limits don't arise (`polylogue-ddhvu`)
- Long work commits in bounded steps, so stopping waits at most one step (`convergence.md`) (`polylogue-dn9or`)
- Parse outcomes are keyed by input ID and parser version and computed from state; nothing reuses an earlier success record (`convergence.md`) (`polylogue-enmhu`)
- Recognize a file from its location and first record, then parse it once (`Provider::classify`, plan/slice-01.md) (`polylogue-g3vda`)
- Excision is a recorded fact that the loop applies; no "existence gate" may block it (storage/archive-db.md "How excision works") (`polylogue-gejxn`)
- Nothing compared across rebuilds may depend on arrival order (law 4, order independence; marker note IDs come from the message ID and the marker line) (`polylogue-gubtm`)
- Full message lists with copies marked remove tail-only storage and its failure modes (decisions/lineage.md) (`polylogue-h0a81`)
- Unrecognized content is counted per input, format and key, never stored per record (storage/index-db.md `unrecognized`) (`polylogue-hlghl`)
- One ID function over raw records, used by every path (model.md "Identity") (`polylogue-hwp8t`)
- Memory is bounded by streaming, not by size thresholds (behavior.md "Memory", "Untrusted input") (`polylogue-ipet5`)
- Unchanged files are skipped when `(device, inode, size, mtime_ns, ctime_ns)` matches the last observation (`storage/blobs.md`) (`polylogue-j690h`)
- Superseded by the operator's later rule that fixed caps are bugs: stream, and refuse only at a physical limit with the reason (behavior.md "Untrusted input") (`polylogue-k5a53`)
- Recognizing a file reads its location, a ZIP's member list and the first deciding record; size never changes how a supported shape is recognized (`sources/README.md`, import recognition) (`polylogue-9upnr`)
- No fixed ceiling on how many parts an operation may have; a large valid operation is paged or chunked, never refused for its size (`behavior.md`, untrusted input) (`polylogue-m2dfz`)
- Foreign keys stay on in `index.db`; a session is rebuilt through one path (storage/index-db.md notes; convergence.md) (`polylogue-l4azq`)
- Polylogue opens only the files in its own data directory, which it creates (storage/blobs.md) (`polylogue-muavd`)
- `excise` is one request in the shared request shape; the daemon records it in `archive.db` and applies it, with no excise-specific transport (decisions/process.md) (`polylogue-n2nhi`)
- No source fingerprint; a version number per parser and per derived product (convergence.md) (`polylogue-o21.9`)
- Declare only outcome reasons some parser derives (model.md outcome reason codes; the "declared values have a producer" drift check, `plan/testing.md`) (`polylogue-oavlb`)
- Revisions are ordered by stored structure (pieces and growth rows), never by acquisition clock (storage/blobs.md "Inputs, revisions and extents") (`polylogue-ojfkc`)
- No schema probing; the schema is fixed per `INDEX_VERSION` (storage/index-db.md) (`polylogue-ojyda`)
- one write path; the writer computes facts in the session's own transaction (convergence.md) (`polylogue-pqlya`)
- validation is the typed parse, one record at a time (decisions/schema-validation.md) (`polylogue-qk59p`)
- embeddings.db is keyed by text hash with explicit integer IDs; no tier symlink farms (storage/embeddings.md) (`polylogue-ri0hk`)
- one transaction per session, bounded steps, one writer; no separate cold-build route (`polylogue-rk0it`)
- measured parse cost ~600 MB/s per core; appends store only new bytes (evidence/parse-throughput.md, storage/) (`polylogue-sq8w2`)
- one module owns durable access (`store::archive`), enforced by `deny.toml` (plan/workspace.md) (`polylogue-t46.11`)
- one identity rule at every size: BLAKE3 of bytes; IDs from raw records (model.md) (`polylogue-vus7y`)
- there is no "accepted but unfinished" state; the next pass recomputes what is out of date (convergence.md) (`polylogue-x7u3x`)
- spool files are stored inputs (pieces never change), so a position inside one never points at different bytes; event IDs come from the raw record (model.md) (`polylogue-y53cz`)

## Reads, queries and surfaces

- Format detection reads a bounded prefix; large JSON is streamed (behavior.md memory) (`polylogue-1o8tb`)
- the result cache is keyed by the read's view and emptied on excision (`behavior.md`, warm state; law 91) (`polylogue-20d.12`)
- relative times computed at display, never stored or cached (`polylogue-20d.12.2`)
- full message lists; range reads by position (`polylogue-2go3o`)
- A read has a time limit and ends with an error naming it; long work pages instead of holding one read (`storage/blobs.md`, WAL; law 78) (`polylogue-3n7f2`, `polylogue-lv0uf`)
- parse feeds the writer as results arrive (`polylogue-4j9j`)
- saved queries keep the exact text written (`query.md`) (`polylogue-5dlv7`)
- A security-relevant setting with a missing or blank value is refused, never read as permissive; there are few settings at all (`polylogue-8k10u`)
- Watching only wakes the loop; work is computed from state, so a lost wake-up costs nothing (`convergence.md`); async stays at the edges (`plan/workspace.md`) (`polylogue-8xmva`)
- An interrupted read is an error or a `degraded` result, never an empty one (`query.md` results: outcome `empty` vs `degraded` vs `error`) (`polylogue-9eeyy`)
- One rendering module and one range rule for every surface (`rendering.md`) (`polylogue-a8khr`)
- A watched query is checked only against changed sessions its own filters can match (slice 11, watched queries) (`polylogue-bv1w.1`)
- A changed `INDEX_VERSION` simply rebuilds the index while reads continue (`convergence.md`) (`polylogue-ereoq`)
- Lineage stores full message lists with copied messages marked; nothing is composed at write or read (`decisions/lineage.md`) (`polylogue-fg2ar`)
- reads go through the daemon, whose warm state is what makes them fast (`decisions/process.md`) (`polylogue-fko9`)
- Every read pages with a cursor; nothing loads a whole result set (query.md "Results"; law 15, paging) (`polylogue-gzmd4`)
- Results say `error` or `degraded` when reading failed and `empty` only when nothing matched (query.md "Results" outcome field) (`polylogue-hy1ys`)
- A continued page continues by key over current state, with no transaction held between pages (R52; law 15) (`polylogue-hzswk`)
- No wall-clock budget may change an outcome; long work commits in steps, and only a stalled parse (no record for [5 min]) fails (convergence.md "Parse"; law 63) (`polylogue-itddh`)
- One query engine and one ordering key behind every surface; empty or degraded pages are never reported as success (query.md "Principles", "Results") (`polylogue-k1jem`)
- Readers open SQLite read-only and never create anything (spec/surfaces/README.md) (`polylogue-lcakh`)
- No configuration is read from the current directory; the CLI reaches the daemon through an owner-only socket and needs no credential (`spec/surfaces/README.md`, processes) (`polylogue-lsm04`)
- There is no source narrowing; locations are fixed and status lists every source (spec/surfaces/cli.md) (`polylogue-mxzt2`)
- Budgeted output trims each item to the space left and lists what was cut, with the command to get it (user-data.md briefing budgets) (`polylogue-p4uv0`)
- reads go through the daemon in the one request shape; the old read protocol's size came from its operation registry, which is gone (decisions/process.md) (`polylogue-r3cuz`)
- surfaces render the result of the one query plan; nothing re-lowers a query by hand (`polylogue-v1tha`)

## Tests

- Extension tests should assert the order of events, not wall-clock waits (`plan/testing.md`, test rules) (`polylogue-07pt`)
- No permanent scoring service: a bug fix adds its input as an example; mutation testing runs on request (`plan/testing.md`) (`polylogue-88jp`)
- No production branch exists for tests; tests use real SQLite and no mocks (`plan/testing.md`) (`polylogue-bjeu1`)
- Rules live in `principles.md` and in tests that compare spec and code; docs explain, they are not a parallel truth (`polylogue-cpf`)
- One list of rules (`principles.md`, `plan/rewrite-rules.md`) with anchors the drift checks resolve; no second doctrine index beside them (`plan/testing.md`, the spec and the code cannot drift apart) (`polylogue-cpf.7`)
- Tests share the small Morrow fixture; the default suite stays under about 90 s (`plan/testing.md`) (`polylogue-csex1`)
- `archive.db` gains rows only, keyed by UUIDv7; saved queries are notes; a migration copies the file aside first and is tested on a copy of real data (`storage/archive-db.md`) (`polylogue-f35tn`)
- Lists of capabilities are checked against the code by a test, never maintained by hand (testing.md, "spec and code cannot drift apart") (`polylogue-fxtbj`)
- Tests assert outcomes, not wall-clock times; timing belongs in benchmarks (plan/testing.md) (`polylogue-h4shh`)
- Work must be proportional to new bytes, tested by counting IO (law 18) (`polylogue-j2w78`)
- plan/testing.md: no test-selection machinery; `--sample` for quick feedback (`polylogue-reindex-ws-a`)

## Process and code structure

- Keep command handlers thin and one concept per file (plan/method.md, plan/workspace.md); xtask surface makes growth visible (`polylogue-1r9c`)
- No audit-lane machinery: a one-off analysis is a query plus notes or a kept result (`polylogue-9e5`)
- One owner per table and one writer (`plan/workspace.md` crate rules, `storage/`) (`polylogue-a7xr`)
- Write access is a type only the right process can get (`plan/slice-01.md`: `Writer`, `Reader`, `ArchiveWriter`) (`polylogue-a7xr.29`)
- No declared metadata without a consumer; field tables drive the code they describe (`plan/workspace.md`) (`polylogue-ew7ti`)
- The rewrite has no maintenance or repair commands (`principles.md`); nothing to preview (`polylogue-f57q`)
- The devshell holds only what every build needs; advisory scanners run in CI (plan/workspace.md "Nix") (`polylogue-gpdsa`)
- Derived results are written in the same transaction as their data; no receipts (convergence.md "Parse") (`polylogue-oaq1k`)
- no format guessing from content; the location decides the format (principles.md 8) (`polylogue-sba6s`)
- no debt ledger or ownership: work is recomputed from state (convergence.md) (`polylogue-sowzm`)
- no receipts; `xtask full-run` reports time, memory and counts (plan/workspace.md) (`polylogue-uh71q`)
- no seal; plain SQLite transactions (`polylogue-w5zpc`)
- there is no promotion or readiness certification: the index is current or catching up (`up_to_date`) (`polylogue-zfe3j`)

## From other sources

- Findings answered one by one as code grew the codebase; "out of scope" and "the design is wrong here" are valid answers (`evidence/ideas-ai-history.md`; the `polylogue-1tyq5` review-debt family).
- No reader may mean a missing consumer, not an unwanted feature: measure before deleting kept data (2026-09-25 specification, delivery B7).
- An implementation must not generate its own oracle; declarations agreeing with tests generated from them prove nothing (deliveries I1, I5).
- A fast ingest that leaves derived work waiting is not a finished build (delivery I4).

## From the Python version's query and briefing code

Moved here from the spec's rule text, where they interrupted the rules.

- A limit was clamped silently to 1–1000 and bad input became 10: refuse invalid limits instead of guessing.
- Attached rows were capped at 200 per session and 5,000 per page, text at 2,000 characters, and pages over 2,500 sessions refused: no arbitrary caps; page by key instead.
- A cursor refused a larger page size as "widening", a side effect of its snapshot design: continue by key and let page size change.
- `--why` checked at most six conditions and skipped boolean groups: report every condition.
- The briefing budget estimated up to 1.3 tokens per word, and a reduced item reported a cost of one token while keeping its full content: charge the rendered text, once.
- Recall had its own daemon-only read route: every query, recall included, takes the one path through the daemon.
- Two figures were badly wrong yet computed exactly, a 7.69× Codex cost inflation and a 376.6-billion-token total, both from wrong definitions: every figure names its definition.

## From the Python test suite

From `evidence/deep-tests.md` (793k lines in 2,045 test files).

- Hundreds of tests carried a hand-written "Anti-vacuity: …" docstring describing the mutation that would make them fail: sound idea, done by hand per test, so it became boilerplate. The rewrite registers one known-bad case per law and drift check, and a test checks each one fails (rule R61).
- Session and archive builders were rewritten per file (`_seed_session` in 13 files, `_seed_archive` in 12) because the shared builders were awkward: the rewrite has one fixture (the Morrow archive), example directories and, until the slice-8 generator, shared synthetic builders in `tests/` (`plan/testing/laws.md`, common parts), never per-file builders.
- 84 files asserted on mock calls and 51 read the product's own source to pin its shape: both pass on wrong results and fail on refactors. The rewrite tests public output only (laws, examples) and lets the compiler, `deny.toml` and drift checks hold code shape.

From writing this spec (2026-10-12).

- A finding that changes a rule reached one file and left the old rule in others: the parse interface stayed whole-file in three plan files after the per-record change, and Sinex found the same with nine copies of a superseded window. After any change that alters a rule, search the whole Spec for the old wording and fix every copy in the same change.
- Prototypes and real-data runs found what reviews did not: a wrong delta mechanism (dictionary instead of patch-from), a scope rule that cost 50x, text missing from one of two search indexes, a nonce reused across blobs. Every rule with a performance or correctness consequence should get a small executable check before it is trusted.
