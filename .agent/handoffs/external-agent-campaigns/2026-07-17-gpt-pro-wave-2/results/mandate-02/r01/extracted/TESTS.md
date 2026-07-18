# TESTS — design, anti-vacuity, and execution record

## Test design

The tests are organized around production dependencies, not DTO-only construction.

| Test | Production dependency exercised | Representative mutation/removal that must fail |
|---|---|---|
| `test_seeded_bead_history_query_round_trips_direct_refs_and_candidate_overlap` | Beads adapter -> reconciliation -> index v40 DDL -> `replace_work_evidence_graph` -> `SQLiteQueryStore` -> `SessionRepository.find_bead_work_sessions` and traversal | remove `subject_key` persistence; omit repository filter; recover sessions only from focal-node evidence; map time overlap to `observed_effect`; collapse evaluation into claim/effect |
| reconciliation three-fact test | `WorkEvidenceGraph` semantic endpoint validation and `reconcile_work_effects` | create claim->effect edge; store evaluation as claim state; accept effect without repository snapshot |
| candidate forcing test | `EffectAssociation` basis conversion | treat `time_overlap` or `file_overlap` as causal/direct |
| Beads baseline/branch/correction test | `effects_from_beads_records` stable identity and relations | remove branch from identity; synthesize missing baseline; drop correction/supersession relation |
| git/GitHub/artifact/receipt relation test | effect adapters and one-to-many PR relation projection | lose squash commit ref; restrict PR to one Bead; treat unverified receipt as passed |
| observed-event/correlation adapter test | exact identity priority and legacy projection | let incidental shared source refs override exact commit identity; map file/time overlap direct |
| Claude missing-evidence tests | `project_claude_workflow_evidence` | borrow coordinator evidence for a missing artifact; mark artifact-derived calls resolved; erase admitted source path |
| ObjectRef tests | public parser/formatter registry | remove any new kind or make colon-bearing IDs lossy |
| lifecycle declaration checks | current index version and semantic delta registry | bump v40 without a declaration or classify the semantic graph change as clone-forward SQL |

The seeded route uses only committed synthetic fixture data:

- repository `repo:sinity/polylogue`
- repository snapshot `repository-snapshot:sinity/polylogue:master:bf8191b3f56aa40da8f271df7f3385c712825497`
- branch `branch:master`
- Bead `polylogue-7fj`
- interactions `evt-create`, `evt-edit`, `evt-claim`, `evt-close`, `evt-correct`
- four direct archived session identities `codex-session:evt-*`
- one candidate-only session `codex-session:time-overlap-only`
- squash PR `github-pr:sinity/polylogue#3051`
- committed artifact identity `artifact:PATCH.diff`
- synthetic receipt `verification-receipt:pytest:work-effects-route`
- acceptance assertion `assertion:work-effects-route:acceptance`

The claimed result, observed effects, and acceptance assertion are created independently and traversed as separate nodes.

## Environment and harness

The repository's normal `uv run --frozen` environment could not be resolved because the sandbox had no usable network/cache for missing wheels. The available `/opt/pyvenv` had Python 3.13, Pydantic 2, pytest, and pytest-asyncio, but lacked `aiosqlite`, `ijson`, `tenacity`, `sqlite_vec`, Hypothesis, ruff, and mypy.

Focused execution therefore used temporary, non-delivered compatibility modules under `/tmp/polylogue-task/test-stubs` and an import shim `/tmp/polylogue-task/run_pytest_shim.py`. The shim:

- avoids importing unrelated heavy package initializers;
- supplies an async sqlite3-compatible `aiosqlite` interface and minimal `ijson` import compatibility;
- initializes the real source/index/user/ops tier DDL used by `SessionRepository`;
- omits only the unavailable sqlite-vec embeddings tier;
- does not replace the production work-evidence models, adapters, SQL, repository mixins, query store, graph persistence, or traversal/query functions.

These temporary files are not part of `PATCH.diff` or the ZIP. Results below are therefore strong self-contained route evidence but are not a substitute for the repository's managed full dependency environment.

## Commands and results

### Focused work/effect/Claude route

```text
/opt/pyvenv/bin/python /tmp/polylogue-task/run_pytest_shim.py -q \
  -o addopts='' -p no:randomly -p no:random-order \
  --confcutdir=tests/unit/insights \
  tests/unit/insights/test_work_reconciliation.py \
  tests/unit/insights/test_claude_workflow_evidence.py \
  tests/unit/insights/test_work_evidence.py \
  tests/unit/insights/test_work_effects_route.py
```

Result: `13 passed, 2 warnings in 0.55s`.

The warnings are only unknown pytest configuration options `timeout` and `timeout_method`, because the sandbox environment lacks the repository's timeout plugin.

### ObjectRef parser/formatter

```text
/opt/pyvenv/bin/python /tmp/polylogue-task/run_pytest_shim.py -q \
  -o addopts='' -p no:randomly -p no:random-order \
  --confcutdir=tests/unit/core \
  tests/unit/core/test_refs.py
```

Result: `82 passed, 2 warnings in 0.14s`.

### Current schema declaration policy

```text
/opt/pyvenv/bin/python /tmp/polylogue-task/run_pytest_shim.py -q \
  -o addopts='' -p no:randomly -p no:random-order \
  --confcutdir=tests/unit/storage \
  tests/unit/storage/test_index_fast_forward_lifecycle.py::test_current_index_schema_has_a_complete_delta_declaration \
  tests/unit/storage/test_index_fast_forward_lifecycle.py::test_schema_policy_rejects_an_index_bump_without_a_delta_declaration
```

Result: `2 passed, 2 warnings in 0.44s`.

### Python syntax/import compilation

```text
/opt/pyvenv/bin/python -m compileall -q \
  polylogue/core/refs.py \
  polylogue/insights/claude_workflow_evidence.py \
  polylogue/insights/work_evidence.py \
  polylogue/insights/work_reconciliation.py \
  polylogue/storage/query_models.py \
  polylogue/storage/repository/insight/work_evidence.py \
  polylogue/storage/sqlite/archive_tiers/index.py \
  polylogue/storage/sqlite/lifecycle.py \
  polylogue/storage/sqlite/queries/work_evidence.py \
  polylogue/storage/sqlite/query_store_work_evidence.py \
  tests/unit/core/test_refs.py \
  tests/unit/insights/test_claude_workflow_evidence.py \
  tests/unit/insights/test_work_reconciliation.py \
  tests/unit/insights/test_work_effects_route.py
```

Result: passed with no output.

### Patch hygiene

```text
git diff --cached --check
```

Result: passed with no output.

Package validation additionally runs `git apply --check` against clean snapshot commit `bf8191b3f56aa40da8f271df7f3385c712825497`.

## Known unrelated lifecycle-test failures

Running the entire lifecycle unit file produced `7 passed, 2 failed`. The two failures are:

- `test_nonsemantic_delta_without_operations_is_rejected`
- `test_delta_without_a_declared_class_is_rejected`

Both monkeypatch a version-37 declaration and call `index_delta_declaration_report(37)` while retaining all declarations newer than 37. The report intentionally classifies declarations above the requested current version as invalid. On the authoritative unmodified snapshot, the same setup returns `(38, 39, 37)`, already contradicting the tests' expected `(37,)`. With this correct v40 declaration, it returns `(38, 39, 40, 37)`. The current-version completeness and missing-declaration policy tests pass. This package does not alter unrelated historical test semantics to hide the baseline inconsistency.

## Remaining verification

Unverified in this sandbox:

- `devtools test` / managed testmon affected selection;
- `devtools verify --quick` and default affected verification;
- ruff format/check;
- mypy;
- generated-surface/schema rendering policy beyond the targeted lifecycle checks;
- full storage bootstrap including sqlite-vec embeddings;
- full suite/property/integration tests;
- live git/GitHub/Beads/Dolt state;
- operator daemon, MCP server, browser, secrets, deployment, and private archive;
- the real `wf_54d4fb2e-841` incident replay and its 25-open-P1 before/after census.
