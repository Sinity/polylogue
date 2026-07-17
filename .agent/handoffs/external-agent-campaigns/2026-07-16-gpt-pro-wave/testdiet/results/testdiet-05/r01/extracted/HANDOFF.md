# testdiet-05 handoff: bounded query work, cancellation, progress, and cleanup

## Mission and outcome

This package implements the strongest coherent slice of the bounded-query contract that is supported by the attached Polylogue snapshot. It fixes the concrete unbounded mechanism named by `polylogue-z9gh.1`: multi-field `... | group by a, b | count` queries no longer page every selected terminal row into a growing Python list before aggregation. The selected relation, grouping, exact denominator and quality statistics, stable ordering, and requested aggregate page now execute in one SQLite statement.

The patch also extends the snapshot's existing execution-control layer rather than creating a parallel framework. One controlled query call now owns an explicit SQLite read transaction, can enforce an optional deterministic VM-work budget through the production progress handler, records actual progress callbacks and result-page facts, distinguishes `work_budget_exceeded`, and records cleanup completion. The Python API, MCP tool, and daemon HTTP query-unit adapters pass the same outer `QueryExecutionContext` into the shared executor. The CLI receives the SQL aggregation fix but is not yet migrated under the execution-control lifecycle.

This is a rebuilt, substantially improved first valid delivery. The earlier chat artifact was not a usable package: the standalone patch contained only a `FileNotFoundError`, and the linked ZIP did not exist. No part of that artifact was treated as implementation authority.

## Snapshot identity

- Repository branch: `master`
- Baseline commit: `f654480cadb7cc4c194704e24dfd483199547b35`
- Baseline commit subject: `chore(beads): file archive-insight benchmark findings wofr/vhjs/r7p6/1wtm`
- Relevant existing execution-control commit: `fd7b3549292927fbd69e0cb07dff9a1205d8e6c8` (`feat(query): interruptible, admission-controlled archive read execution (#2964)`)
- Relevant Beads closure commit: `6610f3b0726a5fda353109d119eaefa8b31ef993`
- Patch base: tracked files at the exact baseline commit above
- Initial tracked delta: clean after reconstructing from the attached repository snapshot

The archive's generated overview says `dirty=true`, but its branch-delta patch, branch-delta file list, branch-delta log, and branch diff stat are all empty, and the merge base is the baseline commit. I therefore treat the clean tracked tree at `f654480...` as the code authority. The overview's dirty flag is consistent with the archive also reporting more than 1 GiB of ignored local runtime state; no ignored state is included in this package.

## Inspected authority and evidence

The implementation was based on, and cross-checked against:

- root `AGENTS.md` architecture and substrate-first rules;
- the attached mission, `[testdiet 05] Bounded query work, cancellation, and cleanup`;
- `architecture/06-query-cancellation-and-bounds.md`;
- complete attached Beads records for `polylogue-z9gh.1`, `polylogue-z9gh.9.1`, and `polylogue-1xc.14.1`;
- current source in `archive/query/execution_control.py`, `archive/query/unit_results.py`, query expression/metadata definitions, archive-tier SQLite relations, and API/MCP/HTTP/CLI adapters;
- current tests for execution control, aggregate query units, the C-03 action-bound canary, Python API, MCP, daemon HTTP, and CLI aggregation;
- git history around `fd7b35492`, `6610f3b07`, and the baseline commit.

The attached testsuite archive does not contain the `testdiet-01` query-algebra survivor artifact named as a dependency. Current descriptors, source behavior, tests, Beads, and history were therefore used as authority; this patch does not invent a dependency API or helper.

See `EVIDENCE.md` for the source-level findings and contradictions.

## Production mechanism

### 1. SQL-backed bounded multi-field aggregation

`ArchiveStore.query_unit_multi_counts(...)` is a new substrate operation for multi-field aggregate pages. It uses a closed field-expression lowerer tied to current query-unit descriptors and supports:

- messages;
- actions, including the existing selective/bounded action relation and follow-up relation when required;
- blocks;
- de-duplicated file paths at `(session_id, normalized path)` grain;
- durable assertions through the attached read-only user tier and current assertion defaults;
- observed events through the existing source/materialized union CTE;
- delegations through the current delegation view;
- `session.origin` and `session.repo` dimensions where descriptors advertise them.

The statement composes these CTE stages:

1. `selected`: applies the real structural predicate and session filters;
2. `grouped`: counts group tuples and per-field missing/explicit-unknown observations;
3. `ranked`: computes stable ordinal order plus exact full-result statistics with window functions;
4. `page`: selects only the requested ordinal window;
5. `stats`: preserves denominator and quality facts even when the requested page is empty or beyond the end.

Python retains only the returned aggregate page. No semantic row cap was added. The requested limit is still a page size, and `limit + 1` remains the existing `has_next` probe.

The lowerer preserves the prior observable distinctions among SQL `NULL` (rendered as `[missing]`), the empty string, and the literal token `unknown`. Assertion status/visibility/author defaults match current row semantics. Aggregate order matches the former Python tuple ordering, including deterministic tie-breaking by all group keys.

### 2. One owned read snapshot

`InterruptibleSQLiteRead.run(...)` now begins an explicit read transaction before invoking the query work and rolls it back in `finally`. This makes multiple index-tier statements in one controlled call observe the same snapshot after the first read establishes it. Cleanup order is progress-handler removal, transaction release, store close, then receipt completion.

This snapshot currently governs the main index connection and its attached user tier. A lazily opened separate source-tier connection is not yet folded into the same lifecycle; that is a stated continuation item rather than an unsupported claim.

### 3. Deterministic SQLite work containment and truthful progress

`QueryExecutionContext` accepts an optional `sqlite_vm_step_budget`. The runner selects a progress cadence no larger than that budget, counts each actual SQLite progress callback, and aborts when the lower-bound opcode count reaches the declared budget. A budget of zero aborts before query work begins. The terminal state and exception are distinct from cancellation and deadline expiry:

- receipt state: `work_budget_exceeded`;
- exception: `QueryWorkBudgetExceededError`.

The receipt now records:

- actual progress callback count;
- conservative SQLite VM-step lower bound;
- declared VM-step budget;
- exact selected-row denominator where the SQL aggregate executor knows it;
- rows placed in the logical result page;
- logical pages produced;
- cleanup completion.

These counters are recorded by production execution and result-building code. The tests do not duplicate the aggregation algorithm to manufacture a work count.

`rows_emitted` means rows built into the logical envelope, not transport acknowledgement. That distinction is intentional and documented as a risk until the shared result-ref/page transaction exists.

### 4. Shared context propagation

The existing outer contexts are now passed into `query_unit_envelope(...)` and `query_unit_rows(...)` through:

- Python API `Polylogue.query_units` route;
- MCP `query_units` tool and archive support;
- daemon HTTP `/api/query-units` route.

This lets the shared result executor update the same receipt owned by admission/cancellation/deadline control. No adapter-specific counter, timeout, or aggregation implementation was added.

## Decisions

1. **Push grouping into SQLite rather than add a hard row cap.** A cap would change exhaustive aggregate semantics and violate the architecture decision. SQL grouping removes result-cardinality-sized Python memory while preserving exact totals.
2. **Retain current public response shapes.** Existing aggregate envelopes, pipeline metadata, offset behavior, ordering, and `has_next` semantics remain intact.
3. **Use actual progress-handler observations.** The receipt reports a lower bound based on callback cadence, not a guessed row scan count.
4. **Do not choose a public work budget without evidence.** The production primitive exists and is deterministically tested, but API/MCP/HTTP do not impose an arbitrary default. Workload-profile evidence is the named authority for policy.
5. **Do not fabricate `QueryResultRef` or spool semantics.** Aggregate pages are lossless on a stable archive, but durable snapshot-bound continuation remains `z9gh.9.1` work.
6. **Keep this patch substrate-first.** The SQL relation and work lifecycle live in storage/query execution; surfaces only pass the shared context.
7. **Do not delete existing tests or helpers.** No dominated deletion is proposed in this package.

## Changed files

- `polylogue/storage/sqlite/archive_tiers/archive.py`
  - multi-aggregate page/row types;
  - closed SQL field lowerer and CTE composition;
  - one-statement bounded multi-field aggregation;
  - progress-handler cleanup and read-snapshot lifecycle seams.
- `polylogue/archive/query/unit_results.py`
  - removes `_all_aggregate_rows` and Python all-row grouping;
  - routes multi-field counts to the SQLite aggregate page;
  - records result page and exact denominator facts in the shared context.
- `polylogue/archive/query/execution_control.py`
  - optional deterministic VM-work budget;
  - typed work-budget terminal state/error;
  - production work/page/cleanup receipt fields;
  - explicit read transaction ownership and cleanup.
- `polylogue/api/archive.py`
  - passes the outer execution context to the query-unit executor.
- `polylogue/mcp/archive_support.py`
  - accepts and passes the shared context.
- `polylogue/mcp/server_tools.py`
  - passes the MCP route's context to archive support.
- `polylogue/daemon/http.py`
  - passes the daemon route's context to the query-unit executor.
- `tests/unit/archive/query/test_execution_control.py`
  - deterministic work-budget, snapshot, receipt, cleanup, surface-context, and irrelevant-growth tests.
- `tests/unit/archive/test_query_multi_aggregate.py`
  - lossless page, exact stats, SQL-only execution, relation coverage, de-duplication, and data-state tests.

`PATCH.diff` is the complete apply-ready unified diff. `FILES/` is omitted because no complete replacement is necessary to disambiguate the patch.

## Acceptance matrix

| Mission property | Status | Evidence in this patch |
| --- | --- | --- |
| Remove result-cardinality-sized Python materialization | PASS | `_all_aggregate_rows` removed; multi-field grouping/count/page/stats are one SQLite statement; mutation-sensitive SQL-only test |
| Irrelevant-growth semantic equivalence | PASS for exact-session action aggregate | Existing C-03 result equivalence retained; new controlled-runner canary compares bounded and global-first mutant |
| Selected-work scaling | PASS for the implemented exact-session action route | Production progress receipt remains below 50,000 VM-step lower bound; global-first mutation reaches at least 50,000 with identical result |
| Lossless aggregate pagination/addressability | PASS for static snapshot page concatenation; PARTIAL globally | Stable page concatenation, exact denominator/quality facts on every page and beyond-end page; no durable opaque cursor/result ref/spool yet |
| Cancellation/deadline propagation | PASS for existing API/MCP/HTTP controlled route | Existing cancellation, deadline, disconnect, heartbeat, and admission survivors remain green |
| Deterministic work-budget termination | PASS at shared execution primitive; PARTIAL policy | Typed budget state/error and exact callback cadence tested; no evidence-derived public default configured |
| Truthful progress | PARTIAL | Actual SQLite callbacks, VM lower bound, exact aggregate denominator, logical emitted rows/pages; no exact scanned-row/relation-expansion/spool-byte counters |
| Cleanup | PASS for owned index reader/snapshot/progress/admission; live envelope unverified | Snapshot consistency, connection closure, cleanup receipt, and admission release tested; no incident-scale RSS/PSS/temp/FD run |
| API/MCP/HTTP shared context | PASS for `query_units` | Production route tests and receipt assertions |
| CLI shared execution context | PARTIAL | CLI receives SQL bounded aggregation and behavior tests pass, but it is not under `InterruptibleSQLiteRead` in this patch |
| All read surfaces and result kinds | NOT IMPLEMENTED | list/search/session/messages/tree/topology and insight paths remain outside one shared transaction |
| Parser token/depth/length budget | NOT IMPLEMENTED | No safe source-backed parser-budget contract was present in this slice |
| Durable result refs/spool/resume | NOT IMPLEMENTED | Remains the central `z9gh.9.1` continuation |
| No semantic hard cap or renderer substitution | PASS | No selected-row cap or response truncation was introduced |

## Apply order

Apply `PATCH.diff` once against `master` at `f654480cadb7cc4c194704e24dfd483199547b35`:

```bash
git checkout master
git reset --hard f654480cadb7cc4c194704e24dfd483199547b35
git apply --check PATCH.diff
git apply PATCH.diff
```

The conceptual dependency order inside the patch is:

1. archive-store page types, SQL lowerer, and read lifecycle;
2. execution-control receipt/budget/snapshot behavior;
3. shared unit result executor;
4. API/MCP/HTTP context adapters;
5. focused tests.

## Verification performed

All verification used the reconstructed attached snapshot, not an operator worktree.

- 31 execution-control and multi-aggregate tests passed.
- 1 existing storage C-03 canary passed.
- 11 Python API `query_units` tests passed.
- 8 MCP `query_units` tool tests passed.
- 7 daemon HTTP query-unit endpoint tests passed.
- 7 selected real CLI aggregate tests passed.
- Total focused passing tests: **75**.
- Ruff check: passed on all 9 changed files.
- Ruff format check: all 9 changed files already formatted.
- Focused mypy: passed on all 7 changed production files.
- Python bytecode compilation: passed on all 7 changed production files.
- `git diff --check`: passed.
- Fresh-clone `git apply --check`: passed.
- Fresh-clone apply followed by `git diff --check`: passed.

The exact commands and non-acceptance invocations are in `TESTS.md`.

## Important limitations and risks

1. **No durable result spool or query result ref.** Each later aggregate page reruns the grouped statement. Static page concatenation is exact, but a changing archive can move page boundaries without a snapshot-bound cursor.
2. **SQLite temp work is not yet certified.** Python memory is bounded by page size, but a high-cardinality grouping can use SQLite sorter/temp storage. The optional VM budget can terminate work; it does not make a partial aggregate complete or resumable.
3. **Only the main index connection is interrupted and counted.** Separate lazy source-tier reads are outside the progress handler, interrupt, and snapshot seam.
4. **No public work-budget policy.** The primitive is production-ready, but a default requires workload-profile evidence and a typed surface contract. Arbitrary values would turn host protection into accidental semantic refusal.
5. **Progress is conservative and scoped.** VM steps are a lower bound at callback cadence. Selected rows are exact only for the new multi-aggregate executor. Logical page emission is not network delivery acknowledgement.
6. **CLI lifecycle remains separate.** It benefits from the SQL fix but does not yet share admission/cancellation/receipt ownership.
7. **Read snapshots can retain WAL history.** The deadline/cancellation path bounds intended lifetime, but incident-scale WAL effects were not measured.
8. **No full-suite or live-system certification.** The operator daemon, browser, live archive, secrets, NixOS deployment, incident corpus, RSS/PSS/swap, temp directory, and file-descriptor steady state were unavailable and are explicitly unverified.
9. **No parser depth-limit survivor.** Deeply nested input remains outside this coherent patch.
10. **Current terminal vocabulary is retained.** Existing `cancelled`, `timed_out`, and `disconnected` names were not renamed wholesale to the architecture document's longer spellings; only the missing `work_budget_exceeded` state was added.

## Exact continuation map

A substantial second pass still has high value. It should not be a small patch around this executor; it should implement the shared query transaction already specified by `polylogue-z9gh.9.1`:

1. introduce a canonical request/page/result-ref boundary with snapshot identity and stable opaque continuation;
2. use keyset continuation for stable indexed rows and an owned bounded spool for aggregates, recursive graphs, and unstable plans;
3. migrate CLI, list/search, session/message/block/action/file, topology/tree, and insight reads under the same outer context;
4. make source-tier reads read-only, snapshot-owned, progress-counted, and interruptible;
5. add parser token/depth/length preflight with a typed `syntax_budget_exceeded` result;
6. derive public VM/temp/admission budgets from workload profiles rather than constants chosen in tests;
7. certify repeated incident-scale cancellation and completion against RSS/PSS/swap/temp/reader/FD steady-state envelopes;
8. distinguish computation completion, spool completion, page construction, transport delivery, disconnect abandonment, and explicit resumable ownership.

A small repair could add exact totals to the existing single-field aggregate receipt or map the new budget exception at individual adapters, but that would add limited value without a public policy and result lifecycle. The next high-return iteration is therefore **substantial**, not cosmetic.
