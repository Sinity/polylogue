# TESTS — continuity replay revision r02

## Test strategy

The test design separates four authorities:

1. **Fixture construction authority** — schema-v2 corpus constants are compiled into real archive tiers through existing test-infra builders and storage primitives.
2. **Independent oracle authority** — expected facts, refs, and grades are read from a separate manifest section and are never computed by the production route.
3. **Production route authority** — the primary replay uses the official MCP SDK over stdio JSON-RPC and reaches the production FastMCP server, query parser/lowering, archive APIs, and SQLite tiers.
4. **Fault authority** — controlled mutations alter discovery, call arguments, or returned responses without rewriting the independent oracle.

A scenario passes only when declaration validation, discovery, accepted plan signature, budgets, pagination invariants, projected facts, evidence refs, and incident grades all pass.

## Production dependencies exercised

The all-scenario integration path exercises:

```text
MCP ClientSession / stdio_client
  -> polylogue.mcp.cli main --role read
  -> production FastMCP registration
  -> query_units / provider_usage / explain_query_expression / list_read_view_profiles
  -> RuntimeServices and archive facade
  -> query expression parser, terminal-unit selection, aggregation, lowering, transaction
  -> archive SQLite tiers and user assertion/usage data
  -> production JSON response budgeting
```

The mutation suite uses the same registered production handlers in-process to make narrow fault injection deterministic and fast.

## Added/changed tests

### `tests/unit/product/test_continuity_scenarios.py`

Checks:

- exactly eight declarations and all required jobs;
- inheritance through the existing `NamedScenarioSource` seam;
- valid existing workflow references;
- allowed tools, discovery arguments, plan signatures, coverage/result semantics, budgets, stop conditions, and exact item identity paths;
- aggregate count-probe requirements for supported terminal units;
- schema-v2 corpus/oracle separation;
- raw incident curriculum contains no planted grade field;
- all six independent grade outcomes;
- fact/evidence projection behavior and declaration payloads.

Representative removal/mutation that should fail: remove a required declaration field, use an unknown workflow, remove an exact identity path from a paginated step, or embed an expected grade in raw corpus input.

### `tests/unit/mcp/test_prompt_query_parity.py`

Renders all shipped prompts containing `query_units`, parses every advertised expression with `parse_unit_source_expression`, and checks every advertised argument is present in the production-discovered tool schema.

Representative mutation that should fail: restore either parser-invalid recipe (`since` embedded in an action predicate or file predicate `repo:`).

### `tests/integration/test_continuity_replay.py`

Checks:

- direct SQLite census of 129 coordinator children, 91 target members, 38 other children, four invocations, six curriculum cases, and 1,950 tokens;
- all eight scenarios through official MCP stdio JSON-RPC;
- discovery protocol/tool receipt;
- incident facts and grades equal the independent oracle;
- 91 members enumerated exactly once over six pages `[17, 17, 17, 17, 17, 6]`;
- separate aggregate count selects 91;
- stable query/result refs and exact-enumeration flags;
- cost route preserves aggregate counters in a response-budget envelope;
- all six named mutation families fail with the intended diagnostic;
- dropping the Workflow filter broadens 91 to 129 and fails;
- changing only expected 91 to 92 leaves observed 91 and reports the skew.

Representative production dependency: `query_units` filtering, continuation, aggregation, MCP framing, and archive-backed item identity.

### `tests/infra/test_archive_scenarios.py`

Existing regression coverage confirms the extended `ScenarioContentBlock.tool_result()` remains compatible with the established fixture compiler and archive projections.

### `tests/unit/mcp/test_server_surfaces.py`

The full existing file validates that prompt corrections and continuity use of registered read tools do not change the documented MCP surface or surrounding query/prompt behavior.

### `tests/unit/archive/query/test_execution_control.py`

Existing tests confirm cancellation/deadline/disconnect behavior in the underlying query execution substrate. They do not prove that the new stdio replay sends or observes MCP cancellation.

## Named mutation matrix

| Mutation | Injected defect | Expected result | Executed result |
|---|---|---|---|
| `lost-request-state-continuation` | Drops continuation state and retries offset zero | `pagination_offset_mismatch`, execution | Failed as expected |
| `capped-pseudo-total` | Ends after first 17-row page while exact count remains 91 | `pagination_count_mismatch`, execution | Failed as expected |
| `identical-call-topology-replay` | Replays first-page item identities under second-page metadata | `duplicate_pagination_identity`, execution | Failed as expected |
| `hidden-fact-or-grammar-discovery` | Removes `continuation` from discovered `query_units` schema | `missing_discovered_arguments`, discovery | Failed as expected |
| `missing-source-coverage` | Coherently removes prior-art source row and count | `non_single_projection`, source coverage | Failed as expected |
| `unreasonable-query-classification` | Removes the shipped-instruction fact from sessions-only curriculum input | `attempt_grade_mismatch`, reasoning | Failed as expected |

The independently executed mutation summary produced one failure report per mutation and no accidental pass.

## Oracle-independence anti-vacuity tests

### Route mutation

The test removes only this production expression filter:

```text
AND text:"workflow_run:wf_synthetic_841"
```

The route then observes all 129 coordinator children rather than the expected 91 target-run attempts. The diagnostic is:

```json
{
  "kind": "fact_mismatch",
  "failure_class": "source_coverage",
  "fact": "attempt_transcripts",
  "expected": 91,
  "observed": 129
}
```

### Oracle mutation

The archive and production route are unchanged. Only the independent expected fact is changed from 91 to 92. The route still observes 91 and reports expected 92 versus observed 91. This proves the oracle is not regenerated from the route result.

## Commands and final results

All commands were run from the reconstructed repository at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` with the patch present.

### Static quality

```bash
.venv/bin/ruff check \
  devtools/continuity_replay.py \
  polylogue/mcp/server_prompts.py \
  polylogue/product/continuity_scenarios.py \
  tests/infra/archive_scenarios.py \
  tests/infra/continuity.py \
  tests/infra/continuity_mutations.py \
  tests/integration/test_continuity_replay.py \
  tests/unit/mcp/test_prompt_query_parity.py \
  tests/unit/product/test_continuity_scenarios.py

.venv/bin/ruff format --check <same files>
.venv/bin/mypy --strict <same files>
.venv/bin/python -m py_compile <same files>
git diff --check 536a53efac0cbe4a2473ad379e4db49ef3fce74d
```

Result:

```text
All checks passed!
9 files already formatted
Success: no issues found in 9 source files
```

Python compilation and `git diff --check` exited successfully with no output.

### Focused continuity suite

```bash
.venv/bin/python -m pytest -q -p no:randomly \
  tests/unit/product/test_continuity_scenarios.py \
  tests/unit/mcp/test_prompt_query_parity.py \
  tests/integration/test_continuity_replay.py
```

Result:

```text
20 passed in 9.51s
```

The same 20-test suite was run after applying `PATCH.diff` to a fresh detached worktree at the exact base commit:

```text
20 passed in 10.39s
```

### Existing MCP surface regression suite

```bash
.venv/bin/python -m pytest -q -p no:randomly \
  tests/unit/mcp/test_server_surfaces.py
```

Result:

```text
82 passed in 18.77s
```

### Existing archive-scenario regression suite

```bash
.venv/bin/python -m pytest -q -p no:randomly \
  tests/infra/test_archive_scenarios.py
```

Result:

```text
4 passed in 1.17s
```

### Existing query execution-control tests

```bash
.venv/bin/python -m pytest -q -p no:randomly \
  tests/unit/archive/query/test_execution_control.py::test_cancel_interrupts_expensive_statement_within_slo \
  tests/unit/archive/query/test_execution_control.py::test_deadline_aborts_expensive_statement \
  tests/unit/archive/query/test_execution_control.py::test_client_disconnect_cancels_and_releases
```

Result:

```text
3 passed in 2.13s
```

These three tests establish the production substrate's interruptibility. They are not counted as a transport-cancellation proof for the replay runner.

### Fresh synthetic corpus and CLI replay

A new archive root was seeded through `seed_continuity_archive()`. The direct SQLite census printed:

```json
{
  "coordinator_children": 129,
  "incident_members": 91,
  "other_children": 38,
  "workflow_invocations": 4,
  "final_result_count": 1,
  "incident_curriculum_cases": 6,
  "usage_input_tokens": 1200,
  "usage_output_tokens": 300,
  "usage_cached_input_tokens": 400,
  "usage_cache_write_tokens": 50,
  "usage_total_tokens": 1950
}
```

Replay command:

```bash
.venv/bin/python devtools/continuity_replay.py \
  --archive-root /mnt/data/work/final-cli-final/archive \
  --oracle tests/data/continuity/catalog.json \
  --transport stdio \
  --output /mnt/data/work/final-cli-final/report.json
```

Result summary:

```text
schema_version: 2
fixture_id: continuity-synthetic-v2
transport: mcp-stdio-json-rpc
protocol_version: 2025-11-25
server: polylogue 1.28.1
scenario_count: 8
passed: 8
failed: 0
status: pass
elapsed_ms: 3607.369
report_bytes: 79582
report_sha256: d1871519f9b6bda7003c1fa7a50ca1dd33c3af439be435a4de3376b51d90453e
```

Incident receipt summary:

```text
observed_calls: 25
observed_response_bytes: 276340
observed_scenario_elapsed_ms: 468.446
member pages: 6
page totals: 17, 17, 17, 17, 17, 6
enumerated items: 91
unique identities: 91
exact count probe: 91
population_count_verified: true
exact_enumeration_verified: true
```

### Patch application

```bash
git worktree add --detach /mnt/data/work/r02-applycheck \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d
cd /mnt/data/work/r02-applycheck
git apply --check /mnt/data/r02-package/PATCH.diff
git apply /mnt/data/r02-package/PATCH.diff
git diff --check 536a53efac0cbe4a2473ad379e4db49ef3fce74d
```

Result: all commands exited successfully. The continuity suite then passed in the applied worktree as recorded above.

## Uncounted exploratory command

An exploratory command that combined the continuity tests, archive-scenario tests, and the entire MCP server-surface file in one pytest process exceeded a 300-second command ceiling after partial progress. It was not counted as evidence. Each component was rerun separately and completed cleanly: 20 continuity tests, 82 MCP surface tests, and 4 archive-scenario tests.

## Checks not performed

- full repository pytest matrix;
- Nix flake/check/build/deployment verification;
- live daemon or SSE/HTTP transport;
- authorized operator archive or real incident replay;
- cold external model transcript;
- runner-issued MCP cancellation and child-kill grace measurement;
- live latency/memory SLO measurement;
- git/PR/Beads effect verification;
- package-manager lock refresh.

No claim in the handoff relies on those unperformed checks.
