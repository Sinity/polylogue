# Test design and verification record

## Execution status

**Repository tests were not run by the package author.** No `pytest`, `devtools test`, or `devtools verify` test command was invoked. This follows the task requirement not to claim test execution. The test changes below are implementation artifacts for the integrating lane.

Static checks and standalone production-code probes were run; their exact status appears below.

## Authored production-route tests

### `tests/unit/api/test_analysis_evidence_kernel.py`

Test: `test_analysis_definition_run_and_finding_survive_real_route_rerun`

This is the privacy-safe synthetic end-to-end proof. It contains no operator data, private claims, or copied archive content. It generates generic Codex-provider sessions with text such as `synthetic archive item 1`.

Production dependencies exercised:

1. `ArchiveStore.write_parsed()` writes synthetic sessions through source/index storage.
2. `make_default_convergence_stages()` supplies the production standing-query stage.
3. `DaemonConverger.converge_sessions()` invokes that stage.
4. `ArchiveCanonicalPlanEvaluator` materializes the watched-query result set and query evaluation receipt; no fake evaluator is injected.
5. `put_analysis_definition()` and `put_analysis_run()` persist DB-native objects in `user.db`.
6. `upsert_findings_as_assertions()` uses the real assertion lifecycle.
7. `Polylogue.resolve_ref()` reads the definition, both run receipts, and the finding through public typed payloads.

Behavior asserted:

- canonically equivalent key ordering and composed/decomposed `é` produce one definition row and one `analysis:<hash>`;
- the first run binds the exact query, result set, evaluation receipt, generations, evaluator/world, privacy, and retention state;
- an analysis finding without `analysis_run_ref` is rejected;
- a finding that cites the first run while claiming the second result set is rejected;
- ingesting a second matching synthetic session creates a different production result set and evaluation receipt;
- the changed corpus and evaluator produce a different analysis-run receipt;
- both run rows remain readable and the old receipt is not overwritten;
- public definition/run payloads expose the exact structured receipt;
- the finding payload contains no `body_text` and exposes three resolvable evidence refs: query, result set, and analysis run.

Representative anti-vacuity mutations that must fail this test:

| Mutation/removal | Expected failure |
|---|---|
| Replace canonical definition hashing with random IDs or raw insertion order | Equivalent-definition identity/row-count assertions fail |
| Revert default standing-query evaluator injection or replace production stage with no-op | No baseline/evaluation receipt exists |
| Remove `analysis_run_ref` requirement from the writer | Missing-run rejection fails |
| Skip query/result membership verification | Mismatched-run rejection fails |
| Exclude input/evaluator/world data from receipt digest | First and second receipt distinction can collapse; storage component-isolation test catches individual omissions |
| Change run insertion to overwrite/upsert | Two-row count or old-receipt equality fails |
| Return the old pending stub for `analysis` | Definition resolution fails |
| Expose prose-only finding payload or omit run provenance | `body_text`/evidence assertions fail |

### `tests/unit/storage/test_query_objects.py`

Test: `test_analysis_definitions_and_runs_are_content_addressed_immutable_history`

Production dependency: `polylogue.storage.sqlite.query_objects` against the production fresh `USER_DDL` connection fixture.

Coverage:

- Unicode/key-order-stable definition identity and idempotent row storage;
- exact query/result/evaluation binding;
- exact idempotency for an identical run;
- independent digest sensitivity with a fixed timestamp for archive generation, evaluator ref, evaluation-world parameters, privacy classification, retention class, and exact input;
- changed-corpus/evaluator rerun creates a distinct receipt;
- old and new receipts coexist and remain readable;
- raw SQL update of an analysis run is rejected by the production immutability trigger.

Anti-vacuity mutations:

- remove any one component from `_analysis_receipt_payload()` and its corresponding component-isolation uniqueness assertion fails;
- return only the newest row from `get_analysis_run()` and old-receipt equality fails;
- remove input validation and mismatched result/evaluation relationships can be persisted;
- remove update trigger and raw SQL overwrite succeeds.

### `tests/unit/storage/test_durable_migrations.py`

Modified tests:

- `test_user_tier_v3_migrates_to_current_with_verified_backup_receipt`
- `test_user_tier_v5_annotation_migration_requires_verified_backup_and_matches_fresh_ddl`

Production dependencies:

- real durable migration discovery and sequencing;
- verified backup manifest/receipt policy;
- `010_analysis_evidence_kernel.sql`;
- canonical fresh `USER_DDL`;
- SQLite foreign keys, checks, indexes, and triggers.

Coverage:

- user schema reaches v10 through a contiguous migration sequence;
- v10 is backup-gated rather than silently classified no-backup;
- migrated and fresh databases have semantically equivalent durable analysis schema objects;
- raw SQL cannot bind a result set/evaluation receipt from a different query;
- raw SQL cannot update definitions, runs, or run inputs.

Anti-vacuity mutations:

- omit `010` from migration discovery or leave `USER_SCHEMA_VERSION` at 9 and version assertions fail;
- add fresh DDL but omit migration DDL, or vice versa, and schema parity fails;
- remove the evidence-match trigger and the mismatched raw insert succeeds;
- add an additive-no-backup marker and the backup-policy expectation changes/fails.

### Other modified tests

- `tests/unit/storage/test_archive_tiers_assertions.py` verifies fresh table/trigger inventory and absence of legacy parallel overlay tables.
- `tests/unit/api/test_facade_contracts.py` confirms only still-unimplemented ref kinds return pending payloads; `analysis` is no longer accepted as pending.
- `tests/unit/cli/__snapshots__/test_plain_cli_snapshots.ambr` expects current user schema v10.

## Focused commands for the integrating lane

Run these after the durable slot is admitted or renumbered:

```bash
.venv/bin/python -m devtools test \
  tests/unit/storage/test_query_objects.py \
  -k analysis

.venv/bin/python -m devtools test \
  tests/unit/storage/test_durable_migrations.py \
  -k user_tier

.venv/bin/python -m devtools test \
  tests/unit/storage/test_archive_tiers_assertions.py

.venv/bin/python -m devtools test \
  tests/unit/api/test_analysis_evidence_kernel.py

.venv/bin/python -m devtools test \
  tests/unit/api/test_facade_contracts.py \
  -k resolve_ref

.venv/bin/python -m devtools test \
  tests/unit/cli/test_plain_cli_snapshots.py \
  -k json_status_snapshot
```

Then run the repository gates appropriate to the local lane:

```bash
.venv/bin/python -m devtools verify --quick
# Follow with the full local/CI gate required by AGENTS.md and the active change train.
```

If the repository's `devtools` entry point is intended to be invoked directly in that environment, use the equivalent `devtools test …` / `devtools verify --quick` commands.

## Static checks actually executed

All commands below ran in `/mnt/data/polylogue_impl` with the repository virtual environment where applicable.

| Check | Result |
|---|---|
| `.venv/bin/ruff format --check` over 12 changed Python files | Passed: `12 files already formatted` |
| `.venv/bin/ruff check` over 12 changed Python files | Passed: `All checks passed!` |
| `.venv/bin/mypy` over seven changed production modules | Passed: `Success: no issues found in 7 source files` |
| `.venv/bin/python -m compileall -q` over changed Python files | Passed |
| `.venv/bin/python -m devtools render topology-projection --check` | Passed; wrote/check-matched 1,034 rows, nine pre-existing TBD entries |
| `.venv/bin/python -m devtools render topology-status --check` | Passed |
| `git diff --check chisel-authoritative-baseline` | Passed |
| Import `AnalysisDefinitionPayload` after payload repair | Passed |
| `uv.lock` and `docs/topology-status.md` drift check | No change |

## Standalone production-code probes actually executed

These were direct scripts over production modules, not pytest/devtools tests.

### Fresh schema and storage/public acceptance probe

The final probe created temporary source/index/user/ops databases with `initialize_archive_database`, then used production query-object, assertion, and facade APIs. It asserted:

- user schema version 10;
- required tables/triggers;
- `PRAGMA integrity_check = ok`;
- zero rows from `PRAGMA foreign_key_check`;
- canonical-equivalent definitions share one ref/row;
- missing run citation is rejected;
- run/result mismatch is rejected;
- a changed result/evaluation/evaluator creates a second run;
- both run receipts remain readable;
- public definition, both runs, and finding resolve with typed payload kinds;
- finding body prose is absent from the provenance payload;
- all three declared finding evidence refs resolve.

Final result:

```json
{"definition_identity_stable": true, "finding_body_text_exposed": false, "finding_evidence_count": 3, "historical_run_count": 2, "mismatched_run_rejected": true, "missing_run_rejected": true, "old_run_still_readable": true, "payload_kinds": ["analysis-definition", "analysis-run-receipt", "analysis-run-receipt", "finding-provenance"], "probe": "analysis-kernel-acceptance", "result": "pass", "run_refs_distinct": true, "user_version": 10}
```

### Verification defects discovered and repaired

- The first fresh-schema launch accidentally used the system Python, which did not contain `aiosqlite`; it stopped before opening a database. The identical initializer passed under `.venv/bin/python`.
- A public resolver probe then found a real runtime `RecursionError` in Pydantic schema generation because the recursive `JSONDocument` type alias had been used directly as four model field annotations. The implementation was changed to bounded mapping fields with explicit `require_json_document` validators. Formatting, lint, strict typing, import, and runtime resolution then passed.

### Patch application proof

`PATCH.diff` was applied to a separate detached worktree at synthetic baseline `291c57effbb8483d39e08cc5e215fa9f35819fdf`.

- `git apply --check`: passed.
- `git apply`: passed.
- `git diff --check`: passed.
- SHA-256 comparison of all 15 resulting paths against the implementation worktree: passed.

The package author did not run tests in that worktree.

## Remaining verification

The following remain unverified:

- every authored repository test listed above;
- full quick/full project gates;
- migration from the operator's actual live user tier;
- authenticated backup receipt and stopped-daemon/single-writer rollout;
- restart and schema/runtime convergence;
- deployed daemon ingest-loop activation;
- composition with the public-claims branch if it lands;
- support/grounding verdict semantics from `polylogue-37t.14`;
- privacy retention/excision enforcement from `polylogue-rxdo.2`.
