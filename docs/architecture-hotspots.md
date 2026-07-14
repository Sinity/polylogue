# Execution Control Center Hotspot Map

> Source-grounded map for polylogue-1r9c ("Decompose Polylogue execution
> control centers"). Line counts and function extents below are measured
> against this document's own landing commit; re-measure before trusting a
> stale number (`wc -l <path>` / the `ast`-walk one-liner in the AC-3 slice
> below reproduces the function-extent numbers).

## Why this exists

Production Python grew from ~255k to ~281k lines while the largest
execution hubs kept expanding faster than the codebase average. Bigger
modules are not automatically wrong — cohesive, well-tested code earns its
size — but these eight control centers are where a single file changing
touches disproportionately many call paths, so their maintenance and
change-risk gravity needs explicit tracking rather than growing invisibly.
This map exists so a future change to any of these eight starts from a
decision ("does this belong here, in a new module, or in an existing
neighbor?") instead of "the file is already 5,000 lines, one more function
won't hurt."

## The eight control centers

| # | Control center | Path | Size | Public contract | Layer |
| - | --- | --- | --- | --- | --- |
| 1 | Storage read tier | `storage/sqlite/archive_tiers/archive.py` | 11,382 lines | `ArchiveStore` — every SELECT-shaped query surface (sessions, messages, blocks, insights reads, search) | `storage/` |
| 2 | API facade | `api/archive.py` | 5,881 lines | `Polylogue.repository`/`.backend` verb surface consumed by CLI/MCP/daemon | `api/` (surface) |
| 3 | Storage repair | `storage/repair.py` | 5,558 lines | `repair_*`/`preview_*`/`run_safe_repairs`/`collect_archive_debt_statuses_sync` — every integrity-repair entrypoint the CLI `check`/`repair` commands and daemon convergence call | `storage/` (maintenance-adjacent, see note below) |
| 4 | Daemon HTTP | `daemon/http.py` | 4,609 lines | the daemon's REST/web-shell route table | `daemon/` (surface) |
| 5 | Storage write tier | `storage/sqlite/archive_tiers/write.py` | 4,595 → **4,210** lines (this bead's slice 1 landed) | `write_parsed_session_to_archive` + session/message/block/tag/work-event/phase writers | `storage/` |
| 6 | CLI query dispatch | `cli/archive_query.py`, fn `_execute_archive_query_stdout` | 2,488 lines file / 632-line function (174-805) | the `find`/`read`/`analyze` query-mode stdout path | `cli/` (surface) |
| 7 | Daemon service loop | `daemon/cli.py`, fn `run_daemon_services` | 1,936 lines file / 475-line function (1007-1481) | `polylogued run` — daemon startup and the ~10 concurrent maintenance loops (see `polylogue-9e5.7`'s lock/starvation map) | `daemon/` |
| 8a | MCP read tools | `mcp/server_tools.py`, fn `register_read_tools` | 1,286 lines file / 490-line function (781-1270) | every read-only MCP tool registration | `mcp/` (surface) |
| 8b | MCP mutation tools | `mcp/server_mutation_tools.py`, fn `register_mutation_tools` | 497 lines file / 370-line function (42-411) | every write MCP tool registration | `mcp/` (surface) |

Note on #3: `storage/repair.py`'s placement is itself a case study for the
polylogue-c9y placement doctrine — under the new rule-5 test ("integrity
repair — detecting and fixing rows that violate an invariant the write path
should have prevented") this is squarely `maintenance/` territory by
function, but it lives under `storage/` today because it also owns
low-level SQL the `maintenance/` package doesn't otherwise touch (receipt
files, WAL journal-mode manipulation, quarantine census staging). A future
slice should decide: either `maintenance/` absorbs the orchestration layer
and calls into `storage/` for the SQL primitives (matches the doctrine), or
the doctrine gets a documented exception for repair modules that are
SQL-heavy enough to need `storage/`'s proximity. Not decided in this pass —
flagged as an open question for whichever child bead executes #3's slice.

## Call boundaries and ownership seams

- **#1 (read tier) ← everything reads through it.** `#2` (API facade), the
  six insight-reader mixins in `storage/repository/`, and most of `#6`/`#7`/
  `#8a` ultimately call into `archive_tiers/archive.py`'s `ArchiveStore`
  methods. It has the widest fan-in of the eight and the highest blast
  radius per line changed — any extraction here needs the broadest
  regression net (full `devtools test tests/unit/storage/` plus the
  `api`/`mcp` contract suites) before it can be trusted.
- **#5 (write tier) → #1 boundary.** The write tier and read tier are
  siblings under `archive_tiers/`, not a layered pair — `write.py` does not
  import from `archive.py` for its own writes (it owns its own INSERT/
  UPDATE SQL), but downstream repair (#3) and insights materialization read
  through #1 the rows #5 wrote. This is why #5's extraction (this bead's
  landed slice) is lower-risk than #1's would be: #5 has narrower fan-out
  than #1 has fan-in.
- **#3 (repair) has three genuinely tangled internal blocks, not one.**
  Investigated in detail during this pass: the file's "quarantined
  accepted raw" repair (~81-1436, ~1,350 lines, its own witness/receipt
  dataclasses), "browser capture origin" repair (~168-3527, ~2,090 lines,
  its own witness/receipt dataclasses, and it reuses `_SemanticCanonicalWitness`
  declared up near the quarantined-raw dataclasses despite being
  browser-origin-only — the two blocks' declarations are interleaved even
  though their logic is not), and "raw materialization replay"
  (~3541-4267, ~730 lines) all cross-reference a shared pool of generic
  helpers (`_open_archive_index_connection`, `_resolve_convergence_debt`,
  `_session_insight_*`) that don't belong to any one block. A naive
  line-range extraction of any one block is unsafe without first mapping
  which of these shared helpers each block actually needs — that mapping
  is real remaining work, not a copy-paste job, which is why #3 is **not**
  the slice this bead executes (see "Why slice 1 targeted #5, not #3 or
  #1" below).
- **#2 (API facade) → #1 + insights.** `api/archive.py` composes `#1`'s
  `ArchiveStore` reads with `insights/` accessors into the async `Polylogue`
  facade's verb surface. It has moderate fan-in (CLI, MCP, daemon HTTP all
  construct a `Polylogue` and call through it) but its own internals are
  more independently addressable than #1's — it's mostly a long list of
  verb methods, not a single interdependent state machine.
- **#4 (daemon HTTP) and #7 (daemon service loop)** are siblings under
  `daemon/`; #4 already delegates to a family of `web_shell_*.py` satellite
  modules (11 of them today) for the web-reader concern, which is exactly
  the extraction pattern the rest of this map recommends — #4's remaining
  bulk is route registration and request/response shaping that hasn't yet
  been split the same way.
- **#6 (CLI query dispatch)** sits below `cli/click_app.py`'s root dispatch
  and above `archive/query/` (the DSL lowering) and `#1`/insights readers;
  `_execute_archive_query_stdout`'s 632 lines are mostly per-output-format
  branching (plaintext/JSON/table/transcript) that could plausibly become a
  registry keyed by output format, structurally similar to this bead's
  landed write-effects registry (`polylogue-0aj`) and to `insights/registry.py`'s
  existing descriptor pattern.
- **#8a/#8b (MCP tool registration)** are already reasonably organized —
  `register_mutation_tools` delegates to `register_personal_state_tools` and
  a separate `register_assertion_review_tools`; `register_read_tools` is the
  largest un-delegated block of the two. Investigated as a candidate for
  this bead's slice and set aside (see below) because the win from further
  splitting is smaller than #5's: MCP tool registration functions are
  already mostly independent `@mcp.tool()` closures with minimal
  cross-tool coupling, so the "materially smaller" bar is harder to clear
  without either an arbitrary split (bad — no real seam) or a much larger
  investigation than this pass's budget allowed.

## Import/layer constraints

- Surfaces (`cli/`, `mcp/`, `api/`, `daemon/`) may not import substrate
  (`archive/`, `storage/`) internals directly — enforced by
  `docs/plans/layering.yaml` / `devtools verify layering`. Every extraction
  candidate above that crosses a surface/substrate boundary must keep that
  boundary intact; none of the slices considered in this pass required
  crossing it (all four "storage tier" hotspots, #1/#3/#5, are
  substrate-internal splits, and the MCP/CLI/daemon candidates, #4/#6/#7/#8,
  were split-within-surface candidates).
- `storage/sqlite/archive_tiers/write.py` and `.../archive.py` are
  siblings inside the `archive_tiers` package — extractions within this
  package (like this bead's slice 1) stay inside `storage/`'s layer and
  don't need a `layering.yaml` update. A future extraction that moved
  something from `archive_tiers/` into `archive/` (crossing from
  low-level-SQL to domain-meaning per the polylogue-c9y placement doctrine)
  would need to check `layering.yaml` for a new permitted edge.
- Every new module under `polylogue/` requires `devtools render
  topology-projection && devtools render topology-status` — done for this
  bead's slice 1 (`session_annotations_write.py`).

## Mutation/read contracts

- **#5 (write tier)**: single-writer discipline — the daemon is the sole
  SQLite writer; `write_parsed_session_to_archive` and the session
  tag/work-event/phase upserts extracted in this bead's slice 1 all take an
  already-open `sqlite3.Connection` and never manage their own transaction
  boundary beyond the `with conn:` scoped to their own statement (matching
  `archive/write_effects.py`'s choke-point contract for what happens after
  the row write returns).
- **#3 (storage repair)**: repair proof/receipt semantics — every repair
  action that mutates rows on the strength of an inferred authority
  decision (quarantined-raw acceptance, browser-origin copy-forward) writes
  an append-only receipt file with a proof digest *before* the mutation
  commits, and the receipt-lock/finish helpers are themselves part of the
  "cohesive contract" that makes #3 higher-risk to split than #5: an
  extraction that separated the receipt-writing helpers from the mutation
  logic they attest to would need its own proof that the receipt still
  covers exactly the mutation it claims to.
- **#1/#2 (read tier / API facade)**: read-only, no mutation contract, but
  wide fan-in — the risk on these two is regression breadth, not
  transactional correctness.

## Why slice 1 targeted #5, not #3 or #1

AC-2 asks for "at least the first coherent slice" to make a named control
center "materially smaller ... with no duplicate execution path," and AC-1
asks this map to carry a prioritized sequence with explicit non-goals. The
three biggest hotspots by line count are #1 (11.3k), #3 (5.6k), #5 (4.6k).
This pass investigated extraction candidates in #3 and #1 before choosing
#5:

- **#3 was investigated first** (it is the second-largest and the bead
  description names "storage repair" explicitly). Dependency tracing (see
  "Call boundaries" above) found the three obvious block boundaries
  (quarantined-raw / browser-origin / raw-materialization-replay) share a
  pool of generic helpers with each other and with the ~1,250 lines of
  primitive `RepairResult`/orphan-repair machinery at the bottom of the
  file, and one dataclass (`_SemanticCanonicalWitness`) is declared inside
  the quarantined-raw block's textual region but is actually
  browser-origin-only. A same-day extraction without first building the
  full helper-dependency graph risks either an import cycle or, worse, a
  receipt/proof-semantics bug in code whose entire job is to be a trusted
  attestation of a data-integrity decision. That graph-building is real,
  separate work — tracked as a child bead (see below) rather than rushed.
- **#1 was investigated second.** It is the single biggest number, but it
  is also the widest-fan-in module in the whole codebase (see "Call
  boundaries"): every other hotspot either calls into it directly or calls
  something that does. A first extraction slice here needs the broadest
  regression net of any of the eight, which does not fit a single pass's
  verification budget alongside the other seven beads in this cluster; it
  is the natural next slice once #5's extraction pattern (self-contained
  CRUD triple, module-level dataclasses, lazily-imported cross-module
  helper to avoid a cycle) is validated in production and the child bead
  below can reference it as precedent.
- **#5 was chosen and executed** because dependency tracing found a
  genuinely self-contained sub-contract: the session tag/work-event/phase
  CRUD block (dataclasses `ArchiveSessionTag`/`ArchiveSessionWorkEvent`/
  `ArchiveSessionPhase` plus their six upsert/read functions, ~530 lines
  total) has no callers from and no callers into `write_parsed_session_to_archive`
  or any other write.py function — its only shared dependency is the
  generic `_json_dumps` helper (used by 8 other call sites in `write.py`,
  so it correctly stays in `write.py` and is imported back lazily, inside
  the moved functions' bodies, to avoid a module-init-time circular
  import). This is the "at least the first coherent slice" AC-2 asks for.

## What landed (this bead, slice 1 of N)

- **New module**: `polylogue/storage/sqlite/archive_tiers/session_annotations_write.py`
  — the `ArchiveSessionTag`/`ArchiveSessionWorkEvent`/`ArchiveSessionPhase`
  dataclasses and their `upsert_session_tag`/`read_session_tags`/
  `upsert_session_work_event`/`read_session_work_events`/
  `upsert_session_phase`/`read_session_phases` functions, plus the four
  small helpers only they use (`_json_loads`, `_json_tuple`, `_json_int`,
  `_refresh_session_profile_count`, `_table_exists`).
- **`write.py`** re-exports all nine public names unchanged (same import
  path for every external caller — `archive_tiers/archive.py` and the test
  suite import `ArchiveSessionPhase`/`ArchiveSessionWorkEvent` from
  `polylogue.storage.sqlite.archive_tiers.write` exactly as before) and no
  longer contains the moved code: **4,595 → 4,210 lines (−385, −8.4%)**.
- **No duplicate execution path**: the moved functions are the same
  functions, same bodies, at a new import path; `write.py` does not keep a
  parallel copy.
- **Verification**: `mypy --strict` clean on both files;
  `tests/unit/storage/test_archive_tiers_write.py` (the file's dedicated
  suite) passes 63/63 unchanged; `ruff check` clean;
  `devtools render topology-projection && devtools render topology-status`
  regenerated for the new module; `devtools render all --check` green.

## Remaining slices — durable child beads

Filed as of this bead's landing, each scoped to one control center with an
explicit non-goal so it doesn't balloon into a second "decompose
everything" mega-bead:

| Bead | Control center | Scope | Non-goal |
| --- | --- | --- | --- |
| `polylogue-redt` | #1, storage read tier | Build the helper-dependency graph for `archive_tiers/archive.py` (which methods/private helpers form independently-callable groups) before proposing any extraction; land the first extraction only once that graph identifies a genuinely self-contained group with the same "no shared-helper coupling" property #5's slice had. | Splitting `ArchiveStore`'s public class shape or changing any read query's SQL/semantics. |
| `polylogue-u5dw` | #3, storage repair | Build the helper-dependency graph across the quarantined-raw / browser-origin / raw-materialization-replay blocks (which of `_open_archive_index_connection`, `_resolve_convergence_debt`, `_session_insight_*`, etc. each block actually needs) and resolve the `_SemanticCanonicalWitness` declaration-vs-usage mismatch before any line-range move; land one block's extraction once the graph is built. Also carries the c9y placement-doctrine question (storage/ vs maintenance/ for repair orchestration) as a decision this bead's design should make explicitly. | Changing any repair receipt/proof format or the WAL journal-mode handling. |
| `polylogue-1vzf` | #6, CLI query dispatch | Registry-ize `_execute_archive_query_stdout`'s per-output-format branches (plaintext/JSON/table/transcript), following the write-effects-registry (`polylogue-0aj`) / insights-registry (`insights/registry.py`) pattern already proven in this codebase. | Changing any output format's actual rendering content. |
| `polylogue-gikp` | #2, API facade | Investigate whether `api/archive.py`'s verb methods group into cohesive sub-facades (e.g. tag/mark mutation verbs vs. read/query verbs vs. context-pack verbs) that could become mixins on the async `Polylogue` facade, mirroring `SessionRepository`'s existing 10-mixin composition. | Changing the public `Polylogue` class's method signatures or return types. |
| `polylogue-kchb` | #4, daemon HTTP | Extend the existing `web_shell_*.py` satellite-module pattern (11 modules already split out) to the remaining route-registration bulk in `daemon/http.py`. | Changing any route's request/response contract. |
| `polylogue-avmq` | #7, daemon service loop | Coordinate with `polylogue-yp0` (daemon internal event bus) — `run_daemon_services`'s 475 lines are mostly loop-startup wiring for the ~10 concurrent maintenance loops; the event-bus work is likely to reshape this function's body directly, so a standalone extraction before that lands risks being redone. Filed blocked-by `polylogue-yp0`. | Executing ahead of `polylogue-yp0` landing at least a first bus consumer. |
| `polylogue-w9di` | #8a/#8b, MCP tool registration | Lower priority than the other six — investigated in this pass and found to already have the register_personal_state_tools/register_assertion_review_tools delegation pattern; re-investigate only if `register_read_tools`'s 490 lines grow materially further, and prefer grouping by MCP tool *domain* (search, insights, corrections) over an arbitrary line-count split. | Splitting tools that are already independently testable `@mcp.tool()` closures purely to hit a line-count target. |

## Architecture budget (AC-6)

This map itself is the reviewable audit artifact for now — there is no
automated size/complexity gate in `devtools verify` yet, and adding one
that would "block only unjustified future growth, not legitimate cohesive
code" is a judgment call this pass did not attempt to encode as a
mechanical rule (a naive line-count ceiling would immediately false-flag
#1's `ArchiveStore`, which is legitimately the single read surface for a
five-tier schema). Recommended follow-up, not filed as a separate bead
because it depends on the six child beads above landing first: once two or
three of the child extractions land, `devtools status` or a new `devtools
lab` check could diff each of these eight paths' current line count
against the number recorded in this table and warn (not fail) when one
grows past its post-extraction baseline by some threshold — giving a
renewed-growth signal without pretending a fixed ceiling is the right
model for inherently-central modules like #1.

Ref polylogue-1r9c.
