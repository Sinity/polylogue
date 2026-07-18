# TESTS — Seed ontologies and governed bootstrap

## Test strategy

The new acceptance file exercises production storage and annotation routes, not a parallel fixture framework. It initializes the real USER-tier DDL, persists real immutable schema/batch rows, writes real assertion envelopes, uses the generic judgment lifecycle, calls the asynchronous batch importer, and queries through the typed structural join.

### New real-route tests

| Test | Production dependency exercised | Representative mutation/removal that must fail the test |
| --- | --- | --- |
| `test_seed_catalog_is_registered_and_replayed_without_user_schema_bump` | Global schema registration, exact seed definitions, `persist_builtin_annotation_schemas`, fresh USER bootstrap, same-version replay, immutable `persist_annotation_schema` identity | Remove a seed from `BUILTIN_ANNOTATION_SCHEMAS`; remove the USER same-version replay hook; change an enum/authority field; bump `USER_SCHEMA_VERSION`; or allow an existing `(schema_id, version)` definition to drift |
| `test_goal_events_exclude_derived_inactivity_and_remain_distinct_from_outcomes` | Goal/outcome schema definitions, row validation, schema-qualified annotation identity | Add `unresolved_inactive` or generic inferred abandonment to the goal enum; collapse goal and outcome schemas; remove structural/rule/judged authority; remove historical-backfill provenance; or omit schema id/version from assertion identity |
| `test_rejected_nomination_preserves_tag_and_full_bootstrap_evidence` | Real tag assertion, ontology candidate write, `BEGIN IMMEDIATE`, candidate coercion, full nomination provenance, reject judgment, governance receipt, terminal retry | Register a schema during nomination; convert affinity into confidence; drop view disagreement/residue/rare/epoch/privacy/crosswalk fields; mutate/delete the source tag; remove the immediate transaction; or let an exact retry resurrect candidate status |
| `test_rename_and_split_register_only_operator_output_schemas[rename]` | Generic supersede lifecycle, one operator output schema, immutable registration, receipt idempotency | Persist the draft identity, accept without an actual rename, omit the governance receipt, delete the source tag, or make retry non-idempotent |
| `test_rename_and_split_register_only_operator_output_schemas[split]` | Generic supersede lifecycle, multiple operator output schemas, immutable registration, receipt idempotency | Permit fewer than two split outputs, register the draft instead of outputs, omit one output row, or lose source evidence |
| `test_governance_rolls_back_judgment_when_schema_registration_conflicts` | One `BEGIN IMMEDIATE` transaction spanning generic judgment, schema registration, and receipt | Move schema persistence outside the transaction; commit the generic judgment before registration; swallow immutable-definition conflict; or leave a governance receipt after rollback |
| `test_accept_then_durable_batch_import_requires_label_judgment_for_active_query` | Accept governance, archive-local durable schema resolution, `AnnotationBatchImportRequest`, real batch persistence, candidate label write, typed join, operator judgment, exact batch replay | Add custom schema to global registry; make schema acceptance create annotation membership; let agent batch rows land active; remove durable fingerprint verification; remove importer `BEGIN IMMEDIATE`; or let replay downgrade the operator-accepted row |

The catalog test pins all five seed families, their exact construct enums, common evidence/abstention contract, activity grain, knowledge authority set, and reusability purpose/authority set. This prevents a vacuous “five rows exist” implementation.

## Existing regression and generated-contract coverage

- `tests/unit/annotations/test_schema.py` exercises canonical schema parsing, normalization, registration, validation, and immutable definitions.
- `tests/unit/annotations/test_write.py` exercises the shared single-row annotation writer and its candidate/active authority behavior.
- `tests/unit/storage/test_archive_tiers_assertions.py` exercises the unified assertion substrate, candidate preservation, judgment, and query behavior.
- `tests/unit/storage/test_archive_tiers_user_audit.py` has an exhaustive invariant requiring every `AssertionKind` to have an audit surface; omitting either new kind fails it.
- `tests/unit/storage/test_archive_tiers_ddl.py` exercises fresh tier construction, user version, parent-directory creation, and same-version no-op/preservation behavior.
- `tests/unit/devtools/test_render_openapi.py` exercises current-model OpenAPI generation and published synchronization.
- `tests/unit/cli/test_cli_output_schemas.py` exercises published Pydantic/CLI output-schema synchronization.
- `tests/unit/cli/__snapshots__/test_plain_cli_snapshots.ambr` pins the fresh USER-tier built-in schema count at six.

## Verification environment

The snapshot was tested in a bare Python 3.13 container that lacks several repository dependencies. A focused runner inserted import-only module/package shims for unavailable eager imports (`aiosqlite`, `dateparser`, `ijson`, and package initializers that transitively require them). It then invoked pytest against the exact production leaf modules. No production storage, annotation, schema, importer, join, or judgment function was replaced or mocked.

The renderer runner similarly bypassed unrelated eager package initializers and invoked the exact `main()` functions from `devtools.render_openapi` and `devtools.render_cli_output_schemas`.

The shim is verification infrastructure only and is not part of `PATCH.diff` or the result ZIP.

## Commands and results

### Syntax, patch, and generated artifacts

```text
python -m compileall -q polylogue/annotations polylogue/core/enums.py \
  polylogue/storage/sqlite/archive_tiers tests/unit/annotations/test_seed_ontology.py
```

Result: passed in the implementation checkout and again after applying `PATCH.diff` to a detached clean worktree.

```text
git diff --check
```

Result: passed.

```text
git apply --check PATCH.diff
```

Result: passed against detached base `536a53efac0cbe4a2473ad379e4db49ef3fce74d`; the patch was then applied and compiled successfully.

```text
python /mnt/data/work_ann01/run_renderer.py devtools.render_openapi --check
python /mnt/data/work_ann01/run_renderer.py devtools.render_cli_output_schemas --check
```

Result:

```text
render openapi: sync OK: docs/openapi/search.yaml
render cli-output-schemas: sync OK: docs/schemas/cli-output
```

### New acceptance tests

```text
python /mnt/data/work_ann01/run_pytest_shim.py \
  -o addopts= --confcutdir=tests/unit/annotations \
  tests/unit/annotations/test_seed_ontology.py
```

Result: **7 passed**, 2 warnings. The warnings are only unknown `timeout`/`timeout_method` pytest configuration because the timeout plugin is absent.

### Schema and storage regressions

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/annotations \
  tests/unit/annotations/test_schema.py
```

Result: **69 passed**, 2 timeout-plugin warnings.

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/storage \
  tests/unit/storage/test_archive_tiers_assertions.py
```

Result: **32 passed**, 2 timeout-plugin warnings.

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/storage \
  tests/unit/storage/test_archive_tiers_user_audit.py
```

Result: **2 passed**, 2 timeout-plugin warnings.

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/storage \
  tests/unit/storage/test_archive_tiers_ddl.py \
  -k 'tier_ddl_builds_fresh_database and user or database_bootstrap_creates_parent_directory or database_bootstrap_leaves_current_tier_unchanged or bootstrap_sets_user_version and user'
```

Result: **7 passed, 1 skipped, 21 deselected**, 2 timeout-plugin warnings. The skip is an existing environment/parameter skip, not a failure introduced by the patch.

### Annotation writer regressions

Full focused file:

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/annotations \
  tests/unit/annotations/test_write.py
```

Result: **16 passed, 2 failed**. Both failures occurred before the tested annotation behavior because active-archive initialization requires the unavailable `sqlite_vec` module/extension:

- `TestAnnotationRoundtripWithQueryAndJudge.test_import_query_judge_roundtrip`
- `TestAnnotationRoundtripWithQueryAndJudge.test_value_predicates_preserve_json_types_and_reject_mixed_numeric_scalars`

Observed exception: `RuntimeError: archive embeddings initialization requires sqlite-vec`, caused by `ModuleNotFoundError: No module named 'sqlite_vec'`.

The unaffected subset was run explicitly:

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/annotations \
  tests/unit/annotations/test_write.py \
  -k 'not import_query_judge_roundtrip and not value_predicates_preserve_json_types_and_reject_mixed_numeric_scalars'
```

Result: **16 passed, 2 deselected**, 2 timeout-plugin warnings.

The new end-to-end acceptance test independently covers custom schema registration, batch import, typed query, operator judgment, and replay without initializing the embeddings tier.

### Generated contract tests

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/devtools \
  tests/unit/devtools/test_render_openapi.py
```

Result: **4 passed**, 2 timeout-plugin warnings.

```text
python /mnt/data/work_ann01/run_pytest_shim.py -q \
  -o addopts= --confcutdir=tests/unit/cli \
  tests/unit/cli/test_cli_output_schemas.py \
  -k published_schema_matches_current_model
```

Result: **18 passed, 14 deselected**, 2 timeout-plugin warnings.

## Native entry-point failures in this container

These are environment failures and are recorded rather than presented as passing verification.

```text
python -m pytest -q tests/unit/annotations/test_seed_ontology.py
```

Exit 4 before collection: `ModuleNotFoundError: No module named 'hypothesis'` from `tests/conftest.py`.

```text
python -m devtools test tests/unit/annotations/test_seed_ontology.py
```

Exit 1 before collection: `ModuleNotFoundError: No module named 'ijson'` through the managed test command's eager source-decoder imports.

```text
python -m devtools.render_openapi --check
```

Exit 1 before rendering: `ModuleNotFoundError: No module named 'dateparser'` through eager daemon/API imports.

A direct ordinary import of the broader SQLite/API package also encounters missing `aiosqlite`. An attempted frozen managed environment could not be materialized because the container has no dependency network/cache coverage.

## Verification still required in the owner environment

Run these in Polylogue's fully provisioned managed environment:

```text
devtools test tests/unit/annotations/test_seed_ontology.py
python -m devtools.render_openapi --check
python -m devtools.render_cli_output_schemas --check
devtools test tests/unit/annotations/test_schema.py \
  tests/unit/annotations/test_write.py \
  tests/unit/storage/test_archive_tiers_assertions.py \
  tests/unit/storage/test_archive_tiers_user_audit.py \
  tests/unit/devtools/test_render_openapi.py \
  tests/unit/cli/test_cli_output_schemas.py
```

Then run the repository's normal broad pre-merge verification gate. A disposable archive CLI/MCP smoke test should import one built-in seed batch and one governed archive-local batch, judge one label, and confirm exact replay preserves the judgment.
