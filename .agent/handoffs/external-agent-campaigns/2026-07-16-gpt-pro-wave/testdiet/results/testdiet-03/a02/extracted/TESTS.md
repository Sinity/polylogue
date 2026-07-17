# testdiet-03 test design and execution

## Test design

### `test_singleton_drive_document_sequence_preserves_source_fallback_identity`

Production dependency: `parse_payload` lowering and `parse_drive_payload` in `polylogue/sources/dispatch.py`.

A raw one-document JSON file reaches replay as a one-item parsed sequence. Both entry points must retain the source fallback ID exactly. Multiple-document sequences continue to receive deterministic indexed suffixes.

Representative mutation: restore unconditional `f"{fallback_id}-{index}"` in both singleton loops. Expected failure: `stable-source-id-0` replaces `stable-source-id`.

### `test_incremental_and_rebuild_public_facts_match`

Production dependencies exercised directly:

- `seed_demo_archive` for the existing deterministic provider-native corpus;
- `parse_sources_archive` for incremental update and idempotent reprocess;
- `repair_message_fts_index_sync` for scoped FTS repair;
- `repair_session_insights` for scoped materialization repair after reconnect;
- the Click `maintenance rebuild-index` command for real durable raw replay, inactive generation construction, insight materialization, readiness, and promotion;
- `ArchiveIdentity`, `source_revision_snapshot`, and existing workload receipt types for durable/generation identity.

Route:

1. Seed the production-shaped demo archive with overlays.
2. Append one Codex response item to a real JSONL source.
3. Ingest it incrementally and reprocess the identical source again.
4. Prove the second pass is a no-op at the durable raw-row level.
5. Insert a user assertion and prove source, content, and workload identity do not change.
6. Delete the selected session's FTS row and insight/thread materializations.
7. Close the writer connection.
8. Repair FTS and insights through freshly opened production connections.
9. Project the active incremental generation under exact source/recipe/output keys.
10. Check that projection against planted source facts and transform invariants.
11. Invoke the real full rebuild command and require a distinct promoted generation.
12. Project the rebuilt generation independently and check the same planted facts.
13. Require exact canonical fact equality and exact overlay preservation.
14. Emit two existing `WorkloadReceipt` values with one spec ID but different runtime, generation, and receipt IDs.

Compared facts:

- sessions, messages, blocks, active path, and content hashes;
- attachment and reference facts;
- lineage links;
- contentless FTS result joined to block identity;
- session profiles;
- nine insight-materialization types;
- thread facts;
- deliberate absent and unknown semantics.

Each fact key includes exact source identity, a recipe digest over the real production files and version constants, and a column-level output contract. Generation and receipt identity remain separate.

The expected set is not copied from either route. It consists of hard-coded demo invariants, exact expected IDs/topology/materialization counts, and SHA-256/size computed directly from the planted provider-native files.

Anti-vacuity mutations/removals named by the test:

- remove the rebuild command's insight-materialization stage;
- skip scoped FTS repair;
- retain stale terminal-session materialization rows;
- restore the singleton Gemini `-0` identity suffix;
- fold user assertions into raw content or workload identity.

The test physically plants stale FTS and insight rows. The old Gemini suffix mutation was separately executed and failed both the focused source law and the full survivor.

### `test_overlay_state_does_not_change_rebuild_identity`

Production dependency: durable user-tier assertion writer plus source/index identity projections and `WorkloadEnvelopeSpec`.

The test inserts a user assertion after capturing source snapshot, content identity, and workload spec. Every computational identity remains stable, while the assertion is present in `user.db`.

Representative mutation: include assertion rows in the content or workload hash. Expected failure: pre/post identities differ.

## Verification environment

- baseline: `b9052e09103502017c0f510ecc699aac395de23c`
- Python: 3.13.5
- pytest: 9.0.2
- Ruff: 0.15.22
- isolated applied worktree: fresh detached worktree at the baseline, then `PATCH.diff`
- `PYTHONPATH`: applied worktree plus the supplied environment's installed packages
- pytest plugin choice: `-p no:randomly`, matching the repository's own `pytest_add_cli_args` policy. Running the plugin directly produced an environment-level invalid-seed error before test setup; the repository's configured lane disables that plugin.

## Commands and results

### Patch construction and application

```bash
git diff --binary --no-ext-diff HEAD -- \
  polylogue/sources/dispatch.py \
  tests/unit/sources/test_source_laws.py \
  tests/unit/storage/test_incremental_rebuild_equivalence.py > PATCH.diff

git worktree add --detach /tmp/testdiet03-apply-check \
  b9052e09103502017c0f510ecc699aac395de23c
git -C /tmp/testdiet03-apply-check apply --check PATCH.diff
git -C /tmp/testdiet03-apply-check apply PATCH.diff
git -C /tmp/testdiet03-apply-check diff --check
```

Result: passed. All three changed files in the applied worktree were byte-identical to the implementation worktree.

Patch SHA-256 before ZIP creation: `43144cc9cc27c16533f60776c1d70a727212158f2afe5f2c9db3eb26837280cc`.

### Static checks on the freshly applied tree

```bash
ruff format --check \
  polylogue/sources/dispatch.py \
  tests/unit/sources/test_source_laws.py \
  tests/unit/storage/test_incremental_rebuild_equivalence.py

ruff check \
  polylogue/sources/dispatch.py \
  tests/unit/sources/test_source_laws.py \
  tests/unit/storage/test_incremental_rebuild_equivalence.py

python -m compileall -q \
  polylogue/sources/dispatch.py \
  tests/unit/sources/test_source_laws.py \
  tests/unit/storage/test_incremental_rebuild_equivalence.py

git diff --check
```

Result:

```text
3 files already formatted
All checks passed!
compileall: exit 0
git diff --check: exit 0
```

### Focused survivor, repeated

```bash
python -m pytest -q -p no:randomly \
  tests/unit/sources/test_source_laws.py::test_singleton_drive_document_sequence_preserves_source_fallback_identity \
  tests/unit/storage/test_incremental_rebuild_equivalence.py::test_incremental_and_rebuild_public_facts_match \
  tests/unit/storage/test_incremental_rebuild_equivalence.py::test_overlay_state_does_not_change_rebuild_identity
```

Run 1:

```text
3 passed in 5.21s
```

Run 2, from a new pytest basetemp and newly seeded archives:

```text
3 passed in 4.82s
```

This repeat exercises deterministic source seeding and creates independent incremental/rebuild generations each time.

### Adjacent existing route tests

```bash
python -m pytest -q -p no:randomly \
  tests/unit/sources/test_source_laws.py::test_parse_drive_payload_contract \
  tests/unit/sources/test_source_laws.py::test_parse_drive_payload_recurses_lists_and_detected_payloads \
  tests/unit/cli/test_archive_maintenance_cli.py::test_rebuild_index_source_replay_expands_every_execution_selection_to_authority_cohorts \
  tests/unit/cli/test_archive_maintenance_cli.py::test_rebuild_index_force_write_option_is_retired \
  tests/unit/cli/test_archive_maintenance_cli.py::test_partial_rebuild_requires_no_promote_before_archive_mutation \
  tests/unit/cli/test_archive_maintenance_cli.py::test_rebuild_index_helper_returns_typed_empty_replay_receipt \
  tests/unit/storage/test_repair.py::test_repair_session_insights_dry_run_reports_scoped_rebuild \
  tests/unit/storage/test_index.py::test_rebuild_index_rebuilds_message_fts_only
```

Result: 13 parametrized cases passed in 3.13s.

### Executed representative mutation

The freshly applied tree was copied to a disposable worktree and both new singleton guards in `dispatch.py` were replaced with the previous unconditional indexed fallback. Then these tests ran:

```bash
python -m pytest -q -p no:randomly \
  tests/unit/sources/test_source_laws.py::test_singleton_drive_document_sequence_preserves_source_fallback_identity \
  tests/unit/storage/test_incremental_rebuild_equivalence.py::test_incremental_and_rebuild_public_facts_match
```

Expected result: exit 1, two failures in 4.23s.

Observed first failure:

```text
['stable-source-id-0'] != ['stable-source-id']
```

Observed survivor failure: the rebuilt selected-session set lacked `aistudio-drive:demo-00`. This demonstrates that the production correction is observed by the full equivalence oracle rather than only by a unit helper assertion.

## Checks attempted but not completed

Two broader combinations—one including the complete source-law file and one wider related selection—did not complete inside a 120-second outer harness and emitted no final pytest summary. They are recorded as unverified, not as passing or failing product evidence. The complete repository suite, Nix gate, and archive-scale/nightly lanes were not run.

## Remaining verification

- complete repository CI/devtools gate in the operator's normal environment;
- full auto-census differential from `polylogue-hjwr`;
- async targeted insight equivalence after `polylogue-lyv4` is repaired;
- live daemon/browser/NixOS/archive-scale proof;
- additional mutation executions for omitted insight stage and overlay-hash contamination.
