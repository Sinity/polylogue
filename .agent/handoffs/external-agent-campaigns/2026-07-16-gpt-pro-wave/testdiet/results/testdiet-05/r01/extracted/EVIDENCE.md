# testdiet-05 evidence record

## Repository and snapshot evidence

The attached project-state archive identifies:

- generated snapshot time: `2026-07-17T043202Z`;
- repository branch: `master`;
- commit: `f654480c` (full hash `f654480cadb7cc4c194704e24dfd483199547b35`);
- merge base against `origin/master`: the same full hash;
- branch delta patch: 0 bytes;
- branch delta file list: 0 bytes;
- branch delta log: 0 bytes;
- branch diff stat: empty.

The overview also says `dirty=true` and reports 1,123.3 MiB of ignored local runtime state. Because every tracked-delta artifact is empty and the reconstructed tracked worktree is clean at the named commit, the patch is based on the clean tracked snapshot. Ignored local state was neither inspected as code authority nor copied into the result.

## Repository instructions

Root `AGENTS.md` establishes the relevant architectural constraints:

- semantics belong in the substrate/product layer, while surfaces remain thin adapters;
- public query behavior is shared across CLI, MCP, Python API, and daemon HTTP;
- `actions` is a derived view and exact-selection behavior depends on bounded relation composition;
- query-unit expressions lower through the current Lark DSL and SQLite archive relations;
- adding a new production module would require topology regeneration.

This patch adds no production module. It places aggregation in the SQLite archive substrate and only threads context through surface adapters.

## Architecture decision inspected

`architecture/06-query-cancellation-and-bounds.md` recommends:

- one immutable outer `QueryExecutionContext`;
- one owned read lifecycle;
- dedicated read-only SQLite connection in its worker thread;
- progress-handler and `interrupt()` cancellation;
- common terminal states including work-budget exhaustion;
- logical selected/scanned/emitted progress;
- no hard semantic row caps or renderer substitution;
- lossless page/spool result lifecycle;
- parser token/depth/length preflight;
- deterministic cancellation and cleanup proof.

The current snapshot already contains the first execution-control slice. This patch composes with that implementation, adds the missing read transaction/work receipt/budget pieces, and fixes the named aggregate materialization defect. It does not claim the absent parser or durable spool portions.

## Beads evidence

### `polylogue-z9gh.1`

Title: `Make archive queries interruptible and resource-bounded`.

The design names `QueryExecutionContext`, `QueryAdmissionController`, and `InterruptibleSQLiteRead`, requires a dedicated read-only worker connection, SQLite progress-handler cancellation, exact-connection interrupt, common cleanup ownership, and no permanent semantic refusal by size.

Its 2026-07-16 investigation comment identifies the exact mechanism fixed here:

- `_all_aggregate_rows` in `archive/query/unit_results.py` is the executor behind multi-field `group by ... | count` pipelines;
- it manually pages every matching row with `LIMIT/OFFSET`;
- it accumulates all rows as live Python objects before grouping;
- a progress-handler primitive alone does not fix the Python accumulation loop;
- the accepted implementation direction is SQL `GROUP BY`/`COUNT(*)`, not a hard row cap.

This is the primary authority for replacing all-row materialization with the SQLite aggregate page.

### `polylogue-z9gh.9.1`

Title: `Land the shared query transaction across every read surface`.

It requires one canonical request/executor/page boundary, exact totals, stable page boundaries, snapshot/ref identity, lossless advancing continuation, result refs/spools for aggregate/unstable plans, and parity across CLI/MCP/HTTP/Python. It explicitly says no path should serialize or materialize a full result merely to discard it.

The current patch is deliberately a coherent subset: it removes one full-result materialization path and passes one context through API/MCP/HTTP query units. Durable result refs/spools and migration of every read surface remain open.

### `polylogue-1xc.14.1`

This workload-profile Bead names C-03 as the first production query canary: a mixed archive plus exact-session action query must keep the selective bound in both ranking legs, and a mutation restoring global-first composition must fail. It also requires shared receipts with work, cancellation/progress, and cleanup evidence.

The snapshot already contains a hand-authored C-03 storage canary. This patch reuses its source-backed 50,000-VM-step discriminator through `InterruptibleSQLiteRead` and the production receipt, without claiming the full workload-profile generator.

## Relevant source findings

### Existing execution-control survivor

Commit `fd7b3549292927fbd69e0cb07dff9a1205d8e6c8` added the production execution-control layer and tests across API/MCP/HTTP. The baseline source already has:

- dedicated `ArchiveStore.open_existing(...)` per controlled call;
- worker-thread offload for async routes;
- shared cancellation event;
- SQLite progress handler;
- exact-connection `interrupt()`;
- weighted admission with interactive reservation;
- cancellation, timeout, disconnect, failure receipts;
- API/MCP/HTTP `query_units` integration.

Therefore this patch extends those types and call paths instead of creating `transaction.py`, a second controller, or surface-local timeout logic.

### Concrete aggregate defect

Baseline `archive/query/unit_results.py` contains:

- `_all_aggregate_rows`, which repeatedly calls the terminal row query method with a page size of 1,000;
- an unbounded Python `rows` list;
- Python `Counter` grouping after all matching rows have been retained;
- page slicing only after the complete group set has been created.

The ordinary rows-terminal executor is already bounded to one `limit + 1` page. The defect is specific to multi-field aggregation. Single-field aggregate queries already use `ArchiveStore.query_unit_counts` and remain on that route.

### Current descriptor authority

`archive/query/metadata.py` advertises aggregate fields beyond the baseline Python row map, including `session.repo`, assertions, and observed events. The new closed SQL lowerer follows the current descriptors and current relation implementations. Tests cover messages, actions, blocks, files, assertions, observed events, and delegations.

### Existing C-03 evidence

`tests/unit/storage/test_archive_tiers_archive.py::test_exact_session_action_count_bounds_pairing_before_global_ranking` builds 512 sessions, measures real SQLite progress callbacks, and verifies:

- bounded and mutant queries return the same exact-session result;
- the bounded relation performs fewer than 50,000 VM steps;
- a global-first mutation performs at least 50,000 VM steps.

The new controlled-runner canary uses the same production relation mutation and threshold while exercising receipt accounting and cleanup.

## Tests and public routes inspected

- `tests/unit/archive/query/test_execution_control.py`
- `tests/unit/archive/test_query_multi_aggregate.py`
- `tests/unit/storage/test_archive_tiers_archive.py`
- `tests/unit/api/test_facade_contracts.py`
- `tests/unit/mcp/test_server_surfaces.py`
- `tests/unit/daemon/test_web_reader.py`
- `tests/unit/cli/test_query_expression.py`
- `polylogue/api/archive.py`
- `polylogue/mcp/archive_support.py`
- `polylogue/mcp/server_tools.py`
- `polylogue/daemon/http.py`
- current query expression, metadata, relation, and payload code followed from those call sites.

## History inspected

- `fd7b3549292927fbd69e0cb07dff9a1205d8e6c8` — first execution-control implementation;
- `6610f3b0726a5fda353109d119eaefa8b31ef993` — Beads closure note for that slice;
- `f654480cadb7cc4c194704e24dfd483199547b35` — attached baseline.

The attached Beads export still reports `polylogue-z9gh.1` as open, while the repository history contains a later commit whose subject says it closes `z9gh.1`. This is a source-of-record contradiction. The current code and later open shared-transaction Bead show that the first execution-control slice landed but the broader bounded-query contract remains incomplete. The patch treats current source as authoritative for what exists and the complete Bead text as authority for remaining intent.

## Dependency contradiction

The mission says to integrate after the query algebra survivor in `testdiet-01`. The attached testsuite archive contains no `testdiet-01` result package or patch. Current source already has the terminal query algebra, descriptors, aggregate pipeline, and exact-session action-bound canary. The implementation therefore uses those current production contracts and records the missing dependency artifact rather than inventing names or blocking the coherent fix.

## Why this scope is coherent

The source, Bead comment, architecture decision, and existing tests converge on one end-to-end defect and one extension point:

1. multi-field aggregation is the known Python memory/work amplification path;
2. SQLite already owns structural predicates and selective action/event/file relations;
3. the existing runner already owns cancellation/admission and the exact connection;
4. surface adapters already create one outer context but did not expose result work to it;
5. SQL grouping plus context-threaded receipts fixes the defect through real storage and public query-unit routes without defining a parallel framework.

The unimplemented parser and durable spool contracts require broader product decisions and cross-surface migration. They are explicitly separated in `HANDOFF.md` rather than represented by scaffolding.
