# testdiet-05 test design and execution record

## Test strategy

The tests use real `ArchiveStore` SQLite files, current query parsing/lowering, current repository/facade adapters, and current public API/MCP/HTTP/CLI routes. They avoid a test-only aggregate implementation or a wall-clock-only work proxy.

The strongest new laws are:

1. multi-field aggregates never invoke the terminal row method and therefore cannot restore `_all_aggregate_rows` unnoticed;
2. aggregate pages concatenate to the exact stable group sequence and retain exact full-set facts on every page, including an out-of-range page;
3. exact-session action aggregation has identical semantics under irrelevant archive growth while production VM-work counters remain bounded;
4. a global-first action-relation mutation preserves output but crosses the declared work threshold;
5. a VM-work budget terminates at a deterministic progress-callback boundary with a typed state and complete cleanup;
6. two statements in one controlled call observe one read snapshot while a concurrent writer commits;
7. the shared API route records actual SQLite progress, exact selected rows, one logical page, emitted rows, and cleanup;
8. lowerer tests exercise all supported SQL relation families and the distinct states `NULL`, empty string, and literal `unknown`.

## Production dependencies and anti-vacuity mutations

### Work amplification

**Production dependency:** `ArchiveStore.query_unit_multi_counts`, `_action_relation_for_query`, and the shared `unit_results` count executor.

**Mutation A:** restore `_all_aggregate_rows` and page all selected terminal rows into Python before grouping.

**Expected failure:** `test_multi_field_group_does_not_materialize_terminal_rows` replaces the descriptor's row query method with a function that raises. The SQL aggregate must still succeed; the restored loop raises immediately.

**Mutation B:** change the exact-session action relation back to a global-first action ranking/pairing relation.

**Expected failure:** `test_exact_session_multi_aggregate_work_is_not_amplified_by_irrelevant_growth` still sees the same one-row aggregate but the production receipt's VM-step lower bound rises from below 50,000 to at least 50,000. The pre-existing C-03 storage canary fails on the same mutation.

### Lossless page semantics

**Production dependency:** grouped/ranked/page/stats CTEs, stable aggregate ordering, existing `limit + 1` lookahead, and pipeline result metadata.

**Mutation:** apply `LIMIT/OFFSET` before grouping, omit the full-ranked stats CTE, sort ties without all group keys, or return page-local denominator/quality counts.

**Expected failure:** `test_multi_field_group_pages_concatenate_losslessly_with_exact_facts` fails because page concatenation differs, `has_next` is wrong, or denominator/missing/unknown facts change across pages or disappear beyond the final page.

### Data-state conservation

**Production dependency:** closed SQL field expressions in `_query_unit_multi_group_field_sql`.

**Mutation:** replace the expressions with `COALESCE(NULLIF(value, ''), 'unknown')`.

**Expected failure:** `test_multi_field_group_distinguishes_missing_empty_and_explicit_unknown` merges three semantic states and reports false missing/unknown counts.

### Relation-specific semantics

**Production dependencies:** file de-duplication CTE, selective action CTE, observed-event source/materialized union, delegation view, assertion defaults, and session join.

**Mutations and failures:**

- remove file `(session_id, path)` grouping: duplicate edits count as two in `test_file_multi_field_group_uses_lossless_file_relation`;
- bypass the selective action relation: exact-session work canary crosses its threshold;
- omit observed-event CTE composition: observed-event aggregate query fails or returns no source-derived events;
- map delegation `basis` directly from a nullable block id: action/edge grouping differs;
- omit assertion defaults or session join: `test_multi_field_sql_lowerer_covers_assertion_defaults_and_session_join` returns missing/default mismatch or no repository group.

### Ignored cancellation or work budget

**Production dependencies:** SQLite progress handler, `QueryExecutionContext.record_sqlite_progress`, `abort_reason`, `_abort_error`, and exact connection interruption.

**Mutation:** make the progress guard always return zero, stop recording callbacks, remove the work-budget branch, or interrupt a different/shared connection.

**Expected failure:** `test_sqlite_vm_work_budget_interrupts_deterministically_and_cleans_up` no longer raises `QueryWorkBudgetExceededError` at exactly five 2,000-opcode callbacks, reports the wrong terminal state, or misses cleanup. Existing cancellation/deadline/heartbeat/disconnect tests exceed their SLO or report a wrong state when cancellation/interrupt wiring is removed.

### Snapshot and leaked state

**Production dependencies:** `begin_read_snapshot`, `end_read_snapshot`, progress-handler removal, `ArchiveStore.close`, and admission release.

**Mutation A:** remove `begin_read_snapshot` while retaining the same connection.

**Expected failure:** `test_runner_holds_one_read_snapshot_until_owned_cleanup` observes `(1, 2)` instead of `(1, 1)` after the concurrent writer commits.

**Mutation B:** skip rollback, progress-handler removal, connection close, cleanup receipt, or admission release.

**Expected failure:** the retained store remains queryable instead of raising `sqlite3.ProgrammingError`; `cleanup_complete` remains false; or `in_flight_weight` remains nonzero. Existing exact-once release tests also catch permit underflow/double release.

### Context propagation and double-counted progress

**Production dependencies:** API/MCP/HTTP adapters pass the same outer context to `query_unit_envelope`, and the shared executor calls `_record_result_page` once.

**Mutation A:** omit the context argument at the facade or adapter.

**Expected failure:** `test_api_multi_aggregate_receipt_reports_actual_work_and_delivery` returns correct rows but has zero progress/pages/emitted rows and no exact selected count.

**Mutation B:** call `_record_result_page` twice or record both fetch-limit rows and response rows.

**Expected failure:** the same test reports two pages or four emitted rows instead of one page and two rows. This is the explicit double-counted-progress survivor.

### Depth-limit mutation

The architecture's parser token/depth/length budget is not implemented in this patch, so there is no honest passing depth-limit survivor to claim. The required next test should construct a deeply nested but lexically valid expression and prove a streaming pre-parser rejects it with `syntax_budget_exceeded` before Lark transformation or recursive allocation. Removing or moving that preflight after parsing must make the test recurse/allocate or return the wrong terminal state. This remains a decision-complete continuation item, not hidden coverage.

## Exact successful commands

Environment used for every `uv` command:

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=/tmp/testdiet05/repo
```

### Core execution and aggregate suite

```bash
uv run --frozen pytest -q \
  tests/unit/archive/query/test_execution_control.py \
  tests/unit/archive/test_query_multi_aggregate.py
```

Result:

```text
31 passed in 11.32s
```

### Existing C-03 storage canary

```bash
uv run --frozen pytest -q \
  tests/unit/storage/test_archive_tiers_archive.py::test_exact_session_action_count_bounds_pairing_before_global_ranking
```

Result:

```text
1 passed in 1.43s
```

### Python API route

```bash
uv run --frozen pytest -q tests/unit/api/test_facade_contracts.py -k query_units
```

Result:

```text
11 passed, 264 deselected in 3.87s
```

### MCP route

```bash
uv run --frozen pytest -q tests/unit/mcp/test_server_surfaces.py -k query_units_tool
```

Result:

```text
8 passed, 74 deselected in 7.32s
```

### Daemon HTTP route

```bash
uv run --frozen pytest -q tests/unit/daemon/test_web_reader.py -k query_units_endpoint
```

Result:

```text
7 passed, 141 deselected in 5.75s
```

### CLI aggregate routes

```bash
uv run --frozen pytest -q tests/unit/cli/test_query_expression.py -k \
  'session_to_message_pipeline_group_by_count_executes_exact_counts or terminal_aggregate_sort_count_asc_executes_before_limit or cli_json_reports_terminal_aggregate_counts or observed_event_tool_outcomes_are_terminal_aggregate_query_units or file_unit_aggregate_counts_distinct_paths or terminal_action_source_exposes_followup_class_rows_and_aggregates or terminal_observed_event_tool_finished_aggregate_reads_blocks_without_materialization'
```

Result:

```text
7 passed, 418 deselected in 2.26s
```

### Static, type, and syntax checks

```bash
uv run --frozen ruff check \
  polylogue/api/archive.py \
  polylogue/archive/query/execution_control.py \
  polylogue/archive/query/unit_results.py \
  polylogue/daemon/http.py \
  polylogue/mcp/archive_support.py \
  polylogue/mcp/server_tools.py \
  polylogue/storage/sqlite/archive_tiers/archive.py \
  tests/unit/archive/query/test_execution_control.py \
  tests/unit/archive/test_query_multi_aggregate.py
```

Result: `All checks passed!`

```bash
uv run --frozen ruff format --check <the same 9 changed files>
```

Result: `9 files already formatted`

```bash
uv run --frozen mypy \
  polylogue/api/archive.py \
  polylogue/archive/query/execution_control.py \
  polylogue/archive/query/unit_results.py \
  polylogue/daemon/http.py \
  polylogue/mcp/archive_support.py \
  polylogue/mcp/server_tools.py \
  polylogue/storage/sqlite/archive_tiers/archive.py
```

Result: `Success: no issues found in 7 source files`

```bash
uv run --frozen python -m compileall -q <the same 7 production files>
git diff --check
```

Result: both passed with no output.

### Patch application

`PATCH.diff` was generated with:

```bash
git diff --binary --full-index --no-ext-diff > PATCH.diff
```

It was then checked and applied to a fresh clone of the baseline repository:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

Result: all commands passed. The fresh clone showed exactly the 9 expected modified files and the same diff stat: 1,278 insertions and 156 deletions.

## Non-acceptance invocations encountered

These commands are not counted as verification passes:

1. An initial combined surface command referenced stale/nonexistent paths such as `tests/unit/api/test_api.py`; pytest exited with status 4 and ran no tests. The repository paths were resolved with `rg`, then the corrected commands above passed.
2. A second combined command used the stale C-03 node name `test_query_unit_rows_exact_session_action_count_bounds_pairing_before_global_ranking`; pytest exited with status 4 and ran no tests. The current node is `test_exact_session_action_count_bounds_pairing_before_global_ranking`, which passed.
3. A long combined shell reached 100% for the seven daemon HTTP tests but hit the outer command timeout before the CLI leg and before printing the daemon summary. The daemon and CLI commands were rerun independently; both passed with the results above.

These were command-selection/aggregation issues, not product test failures.

## Unverified work

The following were not run and are not claimed:

- full repository pytest suite;
- `devtools verify --quick` or full generated-topology/render checks (no production files were added, so topology regeneration was not triggered);
- live daemon, MCP client, browser, or operator archive;
- incident-scale memory/RSS/PSS/swap/temp/FD/reader steady-state certification;
- NixOS/deployment checks;
- query-result spool/resume tests;
- parser depth/token budget tests;
- concurrent page resume against a mutating archive.

## Proposed slower certification survivors

These are retained for a later certification lane, not substituted for the focused tests:

1. Generate the C-03 workload through the promoted workload-profile mechanism, then compare semantic output and production receipt thresholds against the global-first mutation at small/medium/live tiers.
2. Repeatedly run high-cardinality aggregates to completion, budget abort, deadline abort, explicit cancellation, and disconnect while sampling RSS, PSS, swap, SQLite temp bytes, WAL size, open readers, file descriptors, and admission weight until steady state returns.
3. Add owned aggregate spooling, disconnect/resume, expiry, abandonment, and cleanup tests that concatenate every page exactly once under a captured snapshot.
4. Add a parser-depth mutation survivor before Lark transformation.
5. Exercise source-tier-backed reads after source connections join the same read-only progress/interrupt/snapshot lifecycle.

No existing tests or helpers are proposed for deletion.
