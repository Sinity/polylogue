# TESTS — Public claims rendered view

## Test design and anti-vacuity

The new tests call production projection, SQLite writer/judgment, storage adapter, renderer, and generated-surface code. They do not duplicate the production status algorithm in fixtures.

| Test / group | Production dependency exercised | Representative mutation that must fail it |
| --- | --- | --- |
| `test_integrity_verdicts_project_to_distinct_public_renderings` | `project_public_claims`, 37t.14 status map, badges | Remove the provider call, collapse cycle/frame/closed-loop to one badge, or map stale to supported. |
| `test_broken_reference_is_distinct_from_explicitly_unsupported_evidence` | `UNRESOLVED` versus `NOT_SUPPORTED` policy | Treat broken refs as supported or collapse them to the same public/integrity rendering. |
| `test_missing_integrity_verdict_fails_closed_as_unresolved` | Missing-receipt fallback and remediation ref | Default a missing receipt to fresh/supported or omit the 37t.14 dependency marker. |
| `test_review_and_privacy_block_supported_verdicts` | Lifecycle and publication/privacy gates | Let a candidate/rejected/private finding inherit a supported verdict unqualified. |
| `test_integrity_or_declaration_privacy_hold_dominates_lifecycle_and_redacts` | Effective privacy hold and redaction | Only inspect declaration privacy, or leave copy/statistic/evidence visible for a 37t.14 held-private verdict. |
| `test_supersession_selects_one_live_finding_and_conflicting_actives_fail_closed` | Existing supersession facts and one-current-row selection | Ignore `supersedes` or choose one of two live active rows as supported silently. |
| `test_evidence_epoch_advance_degrades_same_finding_without_rewriting_it` | Verdict-driven stale transition | Cache the old supported result or mutate/duplicate the finding to express staleness. |
| `test_all_presets_reuse_one_projection_status_and_include_qualifiers` | Shared projection and generic renderers | Recompute status per surface or omit epoch/frame/definition qualifiers. |
| `test_public_boundary_rejects_private_paths_in_copy_and_receipts` | Typed sanitization of prose, refs, and nested statistics | Remove recursive statistic validation or accept a private absolute path. |
| `test_real_storage_lifecycle_controls_public_support` | `upsert_findings_as_assertions`, `judge_assertion_candidate`, `list_public_finding_inputs`, projection | Bypass canonical judgment, fail to resolve the active successor, or let reseeding duplicate rows. |
| `test_seed_population_uses_snapshot_values_and_sanitized_demo_refs` | Production `claim_vs_evidence_findings()` | Change a numerator, denominator, handler split, origin frame, epoch, or evidence ref. |
| `test_rejected_candidate_cannot_render_supported` | Canonical rejection state | Allow a supported receipt to override a rejected candidate. |
| `test_held_private_declaration_survives_storage_and_blocks_publication` | FINDING payload persistence plus projection | Drop disclosure from the stored payload or ignore it after acceptance. |
| devtools drift/coverage tests | `rendered_artifacts`, `build_report`, marker coverage, parity | Hand-edit generated output, remove a public marker, add an unknown marker, or split statuses across presets. |
| generated-surface and docs tests | Registry and docs topology | Remove the public-claims surface registration or docs-map entries. |

## Exact verification performed

Environment observed during tests: Python 3.13.5, pytest 9.1.1.

### Final affected regression suite

```bash
uv run --frozen --extra dev pytest -q \
  tests/unit/insights/measurement/test_public_claims.py \
  tests/unit/storage/test_public_claims_projection.py \
  tests/unit/devtools/test_public_claims.py \
  tests/unit/scenarios/test_demo_archive_convergence.py \
  tests/unit/storage/test_archive_tiers_assertions.py \
  tests/unit/devtools/test_generated_surfaces.py \
  tests/unit/devtools/test_release_readiness.py \
  tests/unit/devtools/test_render_docs_surface.py \
  tests/unit/devtools/test_project_motd.py \
  tests/unit/daemon/test_standing_queries.py \
  tests/unit/cli/test_import.py \
  tests/unit/api/test_facade_contracts.py::test_compile_context_message_view_can_opt_out_of_assertion_injection \
  tests/unit/api/test_facade_contracts.py::test_list_assertion_claims_filters_lifecycle_claims \
  tests/unit/api/test_facade_contracts.py::test_query_units_returns_assertion_rows \
  tests/unit/api/test_facade_contracts.py::test_resolve_ref_returns_assertion_payload \
  tests/unit/api/test_facade_contracts.py::test_resolve_ref_returns_finding_provenance_payload \
  tests/unit/api/test_facade_contracts.py::test_facade_judges_candidate_assertion_in_user_tier
```

Result: **118 passed in 10.89s**.

### Focused projection/storage/render suite

```bash
uv run --frozen --extra dev pytest -q \
  tests/unit/insights/measurement/test_public_claims.py \
  tests/unit/storage/test_public_claims_projection.py \
  tests/unit/devtools/test_public_claims.py
```

Result after final privacy/sanitization corrections: **31 passed in 1.74s**.

### Static checks

```bash
uv run --frozen --extra dev ruff format --check <all 13 touched Python files>
uv run --frozen --extra dev ruff check <all 13 touched Python files>
```

Result: **13 files formatted; all checks passed**.

```bash
uv run --frozen --extra dev mypy \
  polylogue/insights/measurement/public_claims.py \
  polylogue/storage/sqlite/archive_tiers/user_write.py \
  polylogue/storage/sqlite/finding_provenance.py \
  polylogue/scenarios/corpus.py \
  devtools/public_claims.py devtools/render_public_claims.py \
  tests/unit/insights/measurement/test_public_claims.py \
  tests/unit/storage/test_public_claims_projection.py \
  tests/unit/devtools/test_public_claims.py
```

Result: **Success: no issues found in 9 source files**.

### Generated surfaces and repository contracts

```bash
uv run --frozen --extra dev python -m devtools render all --check
```

Result: **passed**. CLI reference, output schemas, OpenAPI, devtools reference, demo datasheet, quality reference, workflows, public claims, docs surface, MCP surfaces, topology, and pages were synchronized; generated local links resolved.

```bash
uv run --frozen --extra dev python -m devtools verify public-claims --json
```

Result: **passed** with 4 claims, 9 artifacts, 3 monitored public surfaces, zero problems. Default counts: 1 `capability-only`, 3 `unknown`; three integrity statuses `unresolved`.

```bash
uv run --frozen --extra dev python -m devtools verify layering --json
```

Result: **zero violations**.

```bash
uv run --frozen --extra dev python -m devtools verify topology --json
```

Result: **non-blocking**, zero orphan/missing/conflict/kernel findings. Nine existing storage files remain classified as topology `tbd`; none is introduced by this patch.

```bash
git diff --check
```

Result: **passed**.

### Live production-route smoke

A temporary real `user.db` was initialized. The three production seeds were written as candidates, accepted through `judge_assertion_candidate`, and paired with a JSON receipt using schema `polylogue.evidence-integrity-verdicts.v1`. Then:

```bash
uv run --frozen --extra dev python -m devtools render public-claims \
  --archive-root /mnt/data/ann05_e2e/archive \
  --verdicts /mnt/data/ann05_e2e/verdicts.json \
  --output-dir /mnt/data/ann05_e2e/out \
  --compatibility-path /mnt/data/ann05_e2e/public-claims.yaml \
  --json
```

Result: **9 artifacts written, 4 claims rendered**. Verified-export statuses were:

```text
category.local-evidence-system             capability-only
finding.handler-class-split                supported
finding.per-origin-inspection-counts       supported
finding.silent-proceed-lower-bound         supported
```

This proves the command uses the real candidate-to-active lifecycle, storage adapter, receipt loader, projection, and renderer. The smoke used a temporary archive only; no operator live archive was accessed.

### Fresh-clone patch proof

Against a fresh clone of the supplied all-refs bundle at exact commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```bash
git apply --check PATCH.diff
git apply --index PATCH.diff
git diff --cached --name-status
```

Result: **passed; 29 paths staged**, including all eight ignored generated artifacts.

The patch was also applied to another fresh clone and the focused suite was executed there:

```bash
python -m pytest -q \
  tests/unit/insights/measurement/test_public_claims.py \
  tests/unit/storage/test_public_claims_projection.py \
  tests/unit/devtools/test_public_claims.py
```

Result: **31 passed in 2.49s**.

## Known baseline failure

The broader API contract test below fails in the patched tree:

```bash
uv run --frozen --extra dev pytest -q \
  tests/unit/api/test_facade_contracts.py::test_archive_tiers_api_archive_debt_reads_archive_consistency
```

Failure:

```text
sqlite3.OperationalError: database source_debt is locked
polylogue/storage/sqlite/archive_tiers/archive.py:11830
```

The identical isolated test was then run against a pristine fresh clone at the snapshot commit and failed identically. It is therefore classified as a pre-existing baseline defect, not a patch regression.

## Incomplete broad verification

```bash
uv run --frozen --extra dev python -m devtools verify --quick
```

The command completed repository-wide ruff format and ruff check successfully, then exceeded the five-minute command cap during full-repository mypy. No full quick-gate result is claimed. Focused mypy for all touched production and central tests passed as recorded above.

A broad run of the full 275-test API facade contract file was also not used as a completion claim: it exceeded the command cap. The six assertion/finding/judgment facade routes affected by this patch were run explicitly and all passed.

## Verification not performed

- Operator live daemon, browser, secrets, NixOS deployment, current worktree, or private live archive: **unverified and not accessed**.
- Final 37t.14 evaluator integration: **unverified because the evaluator is absent from the snapshot**.
- External publication/cold-reader test of local ignored `.agent` packet refs: **unverified**.
