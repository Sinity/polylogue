# Test and Verification Record

## Environment

Tests were run from the reconstructed authoritative checkout at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` with `/opt/pyvenv/bin/python`. The installed `pytest-randomly` plugin produced an environment-specific seed failure during early work, so focused commands explicitly used `-p no:randomly`. This does not alter production code or test assertions.

## Executed focused suites

### Storage race, lifecycle, bulk, identity, evidence, and queue health

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q \
  tests/unit/storage/test_archive_tiers_assertions.py \
  tests/unit/storage/test_archive_tiers_assertion_write_through.py \
  tests/unit/api/test_assertion_candidate_evidence_disclosure.py \
  tests/unit/api/test_assertion_candidate_queue_health.py
```

Result: `63 passed in 3.93s`.

Production dependencies exercised:

- real SQLite `assertions` DDL and two independent connections;
- `upsert_assertion`, `judge_assertion_candidate`, and `judge_assertion_candidates`;
- the actual busy-timeout/write-reservation helper;
- annotation, recall-pack, and overlay write-through identities;
- public evidence resolution via `Polylogue.resolve_ref`;
- queue-health reads across user and operations tiers.

Anti-vacuity mutations:

- remove `BEGIN IMMEDIATE`, the nested no-op write reservation, or terminal-state preservation: the forced two-connection interleaving can overwrite an accepted row and fails;
- swallow/commit the trigger-induced write error: rollback assertions fail;
- restore payload-hash recall-pack identity: the stable-name update test duplicates instead of updating;
- restore random annotation identity: exact retry creates another row;
- remove the five-preview bound or per-reference exception handling: evidence tests fail;
- classify an unobserved empty queue as healthy or delete old rows: queue-health tests fail;
- remove per-item SAVEPOINT isolation, retry checks, or injection separation: existing bulk lifecycle assertions fail.

### Modified facade contracts and cross-tier debt reads

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q \
  tests/unit/api/test_facade_contracts.py::test_no_undiscovered_async_methods \
  tests/unit/api/test_facade_contracts.py::test_archive_debt_returns_shared_payload_on_empty_archive \
  tests/unit/api/test_facade_contracts.py::test_archive_tiers_api_archive_debt_reads_archive_consistency \
  tests/unit/api/test_facade_contracts.py::test_facade_judges_candidate_assertion_in_user_tier
```

Result: `4 passed in 2.42s`.

This verifies that the new queue-health method is declared on the public facade, candidate judgment still uses the same facade lifecycle, and the dedicated canonical read-only connection fixes attached-database detach locking for archive-debt reads.

### Canonical CLI, action contracts, completion, and deterministic JSON

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q --maxfail=1 \
  tests/unit/cli/test_assertion_candidates.py \
  tests/unit/cli/test_judge_command.py \
  tests/unit/cli/test_note.py \
  tests/unit/cli/test_cli_action_contracts.py \
  tests/unit/cli/test_completion_matrix.py \
  tests/unit/cli/test_completions_contract.py \
  tests/unit/cli/test_deterministic_output.py
```

Result: `307 passed in 6.91s`.

Production dependencies exercised:

- live Click registration and query-first parser;
- root `judge` list/review/status/bulk paths;
- absence of the duplicate nested candidate command;
- explicit candidate-ref and injection guards;
- `--json` alias and deterministic review JSON;
- production terminal `note` capture, root judgment, replay, conflict, facade visibility, and durable SQLite receipts.

Anti-vacuity mutations:

- re-register `mark candidates`: retirement/registry and completion expectations drift;
- bypass root `judge`: the production canary route test fails;
- default `inject=True`: guard and durable context-policy assertions fail;
- generate a random id despite an idempotency key: replay produces a second candidate;
- accept changed content under the same key: conflict assertion fails;
- pass the transform wrapper kind during edit/supersede: the lifecycle-kind test fails.

### MCP review/capture, product workflows, and shared search affordances

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q --maxfail=1 \
  tests/unit/mcp/test_assertion_judgment_tools.py \
  tests/unit/mcp/test_candidate_capture_tool.py \
  tests/unit/product/test_query_action_workflows.py \
  tests/unit/surfaces/test_search_envelope_contract.py
```

Result: `29 passed in 21.36s` with 72 Python 3.13 `fork()` deprecation warnings from the existing product-test multiprocessing path.

This verifies capability declaration, replacement-field preservation, non-boolean injection rejection, evidence-field serialization, MCP idempotency-key forwarding, root-judge workflow ownership, and shared affordance compatibility. It is unit/registration coverage, not a live MCP transport session.

### Direct status command

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q --maxfail=1 \
  tests/unit/cli/commands/test_status.py
```

Result: `92 passed in 2.75s`.

### Status routing and diagnostics

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q --maxfail=1 \
  tests/unit/cli/test_status.py \
  tests/unit/cli/test_status_diagnostics.py
```

An initial run found that `assertion_candidate_queue_health` was missing from the archive-facade route catalog. After adding the user-tier/operations-telemetry route, the rerun result was `77 passed in 21.02s`.

Anti-vacuity: remove the route entry and `test_archive_facade_route_catalog_covers_public_async_facade` fails with the new public method as an uncovered route.

### Daemon status

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q --maxfail=1 \
  tests/unit/daemon/test_daemon_status.py \
  tests/unit/daemon/test_status_maintenance_failures.py
```

Result: `59 passed in 5.19s`.

### Post-type-fix affected rerun

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q \
  tests/unit/api/test_assertion_candidate_evidence_disclosure.py \
  tests/unit/api/test_assertion_candidate_queue_health.py \
  tests/unit/cli/test_judge_command.py \
  tests/unit/cli/test_assertion_candidates.py \
  tests/unit/cli/test_note.py
```

Result: `25 passed in 3.12s`. This overlaps suites above and is recorded only as a final affected-code rerun, not added to a unique-test total.

## Snapshot verification

Snapshot regeneration completed with:

```bash
/opt/pyvenv/bin/python -m pytest -p no:randomly -q \
  tests/unit/cli/test_help_snapshots.py \
  tests/unit/cli/test_terminal_snapshots.py \
  --snapshot-update
```

Result: `10 passed`; five snapshots were initially updated, then two polluted snapshots were cleaned after a terminated test process had inserted unrelated Tokio panic text.

A clean isolated help-snapshot rerun passed: `1 passed in 1.30s`.

The latest isolated terminal PTY rerun produced `6 passed, 3 failed`. Each mismatch consisted of an unrelated Tokio worker panic (`unexpected error when polling the I/O driver: Bad file descriptor`) injected before otherwise valid CLI help. The committed snapshot files were checked and contain no Tokio panic, stack backtrace, or bad-file-descriptor text. This PTY-only environment fault remains unverified in the normal project devshell.

## Static and generated checks

### Ruff

All changed Python files were formatted, then linted:

```bash
python -m ruff format <all changed Python files>
python -m ruff check <all changed Python files>
```

Result: `All checks passed!`.

### Mypy

```bash
/opt/pyvenv/bin/python -m mypy \
  polylogue/storage/sqlite/archive_tiers/user_write.py \
  polylogue/storage/sqlite/archive_tiers/archive.py \
  polylogue/api/archive.py \
  polylogue/surfaces/payloads.py \
  polylogue/cli/commands/judge.py \
  polylogue/cli/commands/note.py \
  polylogue/cli/query_group.py \
  polylogue/cli/query_verbs.py \
  polylogue/cli/commands/status.py \
  polylogue/daemon/status.py \
  polylogue/mcp/server_mutation_tools.py \
  polylogue/product/workflows.py \
  polylogue/operations/action_contracts.py \
  polylogue/cli/command_inventory.py
```

Result: `Success: no issues found in 14 source files`.

### Compile and generated docs

```bash
/opt/pyvenv/bin/python -m compileall -q polylogue devtools
/opt/pyvenv/bin/python -m devtools.render_cli_reference --check
/opt/pyvenv/bin/python -m devtools.render_product_workflows --check
git diff --check
```

Results: compilation succeeded; both generated documents reported sync OK; `git diff --check` succeeded.

### Apply readiness

`PATCH.diff` was generated with full object ids and binary-safe diff output, including the two new test files. A detached worktree was created at the exact base commit and checked with:

```bash
git -C /mnt/data/ann04-applycheck apply --check /mnt/data/ann04-package/PATCH.diff
```

Result: `APPLY_CHECK_OK`.

## Incomplete or unverified checks

Two broad aggregate pytest invocations exceeded the container's per-command ceiling, one at approximately 31% and one at approximately 96%, without a reported failure before termination. Neither is counted as a pass.

The complete repository suite, live daemon, live MCP transport, browser UI, deployment, live archive, and genuine operator canary were not available. The exact live canary is in `HANDOFF.md` and must be run by the operator.
