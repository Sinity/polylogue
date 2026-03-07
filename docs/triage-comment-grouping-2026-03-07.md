# Triage + Comment Grouping (2026-03-07)

## Snapshot
- Branch: `master`
- HEAD: `a912566`
- Tracked modified files: `56`
- Untracked artifacts: QA/docs/local tooling files

## Group A: Core Runtime Code (high-priority, commit as coherent units)

### A1. Schema generation/validation runtime
- `polylogue/schemas/schema_inference.py`
- `polylogue/schemas/validator.py`
- `polylogue/schemas/synthetic.py`
- `polylogue/schemas/providers/*.schema.json.gz`
- `polylogue/storage/backends/schema.py`
- `polylogue/storage/store.py`
- `polylogue/storage/backends/async_sqlite.py`

Comment:
- These are primary behavioral changes and should be committed with direct tests proving registry/latest schema loading, strict validation semantics, and drift/privacy behavior.

### A2. Pipeline stage wiring + CLI exposure
- `polylogue/pipeline/prepare.py`
- `polylogue/pipeline/services/__init__.py`
- `polylogue/cli/commands/check.py`
- `polylogue/cli/commands/run.py`
- `polylogue/cli/commands/demo.py`
- `polylogue/showcase/runner.py`

Comment:
- Keep stage ordering and validation defaults explicit; avoid optional dead branches that preserve old behavior unless required.

### A3. Source iteration/refactor surface
- `polylogue/sources/source.py`

Comment:
- Refactor completed: single shared path now drives path discovery, cursor initialization, mtime skip selection, and summary logging for both iterators.

### A4. Stage option canonicalization
- `polylogue/pipeline/runner.py`
- `polylogue/cli/commands/run.py`

Comment:
- Refactor completed: stage choices now come from a single exported constant (`RUN_STAGE_CHOICES`) instead of duplicated literals.

### A5. Registry-only schema identity/loading path
- `polylogue/schemas/registry.py`
- `polylogue/schemas/validator.py`
- `polylogue/schemas/synthetic.py`

Comment:
- Refactor completed: canonical schema provider mapping is centralized in `SchemaRegistry`; validator/synthetic now read through registry instead of separate file-loading paths.

## Group B: Tests (must travel with runtime groups)

### B1. Schema/validation tests
- `tests/unit/core/test_provider_schema_meta.py`
- `tests/unit/core/test_schema_verification.py`
- `tests/unit/core/test_synthetic_corpus.py`
- `tests/unit/cli/test_check.py`
- `tests/unit/core/test_schema.py`
- `tests/integration/test_contracts.py`

### B2. Pipeline/storage tests
- `tests/unit/pipeline/test_services.py`
- `tests/unit/storage/test_backend.py`
- `tests/unit/storage/test_parse_tracking.py`
- plus associated storage/pipeline/unit updates already touched

### B3. Demo/showcase tests
- `tests/unit/showcase/test_runner.py`
- `tests/unit/cli/test_demo.py`

Comment:
- Treat tests as first-class deliverable; no runtime change should be staged without corresponding assertions.

## Group C: Documentation + Session Reports (non-runtime)
- Modified docs: `docs/architecture.md`, `docs/cli-reference.md`, `docs/demo.md`
- New report docs:
  - `docs/session-recovery-2026-03-05.md`
  - `docs/workload-schema-qa-2026-03-05.md`
  - `docs/demo-parse-validate-audit-2026-03-05.md`
  - `docs/task22-test-audit-2026-03-05.md`
  - `docs/remaining-workload-tracker-2026-03-05.md`
  - `docs/schema-composition-and-quarantine-report-2026-03-06.md`
  - `docs/tasklist-master-2026-03-06.md`
  - `docs/workload-closure-2026-03-06.md`

Comment:
- Keep these split from runtime commits; they are valuable continuity artifacts but not core executable changes.

## Group D: QA Artifacts (archive/curation)
- `qa_outputs/*`
- `qa_archive/*`
- `QA_SESSION.md`

Comment:
- Keep a single curated index and archive heavy logs by date; avoid spraying large artifacts into runtime commits.

## Group E: Local Tooling Noise / Policy-risk files
- `.cclsp.json`
- `.mcp.json`
- `.gitignore` (currently includes AGENTS-related entries)
- `AGENTS.md`, `CLAUDE.md` (global policy docs; handle intentionally)

Comment:
- Local machine configs should stay untracked unless intentionally standardized.
- `.gitignore` edits touching tracked governance files are high-risk and should be explicitly justified or removed.

## Immediate Action Order
1. Stage runtime+tests as coherent commit(s).
2. Stage docs/report/QA groups separately only after runtime stability is confirmed.
3. Normalize or drop policy-risk edits (`.gitignore`, local tool files) intentionally.

## Additional Unification Candidates
1. Consolidate provider identity handling for non-schema flows (roles/filtering/reporting) into a shared provider identity module, mirroring the new registry canonicalization for schema flows.
2. Extract ingest-stage status transitions (`acquired` → `validated` → `parsed`) into a typed state helper to reduce ad-hoc status branching across services.
3. Add a first-class QA artifact command/workflow to produce indexed run evidence without manual path wrangling.
