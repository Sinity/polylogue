# Test design and execution record

## Test strategy

The tests target the production dependency that carries freshness rather than introducing a parallel model. The main new module, `tests/unit/storage/test_derived_freshness_survivor.py`, computes expected relations independently from the production ledgers and names the implementation mutation that each witness is intended to kill.

The production-route tests cover varied arrival orders, equal-count replacement, content/recipe drift, superseded workers, current terminal failures, startup/restart proof, targeted repair, and insight rebuild. Existing tests were extended where the production seam already had the appropriate fixtures.

## Survivor witnesses and anti-vacuity mutations

1. **Real FTS trigger relation across arrival orders.** Inserts empty and searchable blocks, changes empty to searchable and searchable to empty, updates source hashes, and deletes rows. The expected relation is derived directly from `blocks`; it is compared with `messages_fts_identity` and `messages_fts_docsize`. Removing any identity insert/update/delete trigger arm fails.

2. **Equal-count FTS rowid reuse and recipe drift.** Drops the production insert/delete trigger arms, deletes a block, reuses its rowid for a different block, and proves source/index counts still agree while identity does not. Repair must converge. Changing the stored recipe afterward must become stale. Removing the block-ID or recipe comparison fails.

3. **Same-named legacy trigger programs cannot authorize ready state.** Replaces current trigger bodies with old same-named programs, records a superficially ready freshness row, and proves readiness rejects it. Reverting exact SQL-program comparison to name-only detection fails.

4. **Every embedding recipe field is computational.** Mutates canonicalization, selector, chunking, provider, model revision, dimensions, task input type, normalization, tool implementation, and input schema one at a time. Every mutation must change the key. Deleting any declared field fails. Authorization policy is explicitly absent from the recipe.

5. **Equal-count embedding message replacement is selected.** Seeds one source message but metadata for a different message, with matching aggregate counts and `needs_reindex=0`; calls the production pending selector with the old bypass request. Replacing the metadata `LEFT JOIN` with an inner join, trusting counts, or honoring the bypass fails.

6. **Superseded success and terminal failure cannot clear newer debt.** Starts an old key, then a new key/generation. Both old terminal outcomes are rejected and the newer attempt remains pending; the newer success becomes current. Removing key/generation/attempt matching fails.

7. **Superseded production success cannot reopen completed newer debt.** Completes the newer attempt through `_record_archive_embedding_success`, then delivers the old success. The compatibility `needs_reindex` projection remains clear and the newer key remains current. Restoring the old unconditional fallback status write fails.

8. **Superseded production terminal error cannot degrade newer debt.** Sends a late non-retryable provider error through `record_embedding_failure` with the old attempt. The call returns no failure row, does not alter status, and leaves the new active attempt pending. Dropping the `attempt` argument or its conditional failure transition fails.

9. **Provider usage source and pricing recipe changes are pending until rebuild.** Exercises the real source triggers and exact receipt comparison, then mutates a pricing catalog hash field. Retaining only `materializer_version`, omitting the catalog, or stamping current before rebuilding fails.

Additional route tests verify daemon startup sequencing, convergence selection, archive write behavior, schema rejection, repair, and provider-usage rebuild stamping.

## Commands and final results

All commands ran from the implementation worktree at base `b9052e09103502017c0f510ecc699aac395de23c` with Python 3.13.5. The four primary groups are disjoint and total 291 passing tests.

### Modified daemon routes

```bash
python -m pytest -q \
  tests/unit/daemon/test_convergence_stages.py \
  tests/unit/daemon/test_daemon_cli.py \
  tests/unit/daemon/test_embedding_convergence_progress.py
```

Result: **136 passed in 16.01 seconds**.

### Modified embedding routes

```bash
python -m pytest -q \
  tests/unit/storage/test_archive_tiers_embedding_write.py \
  tests/unit/storage/test_embedding_contracts.py \
  tests/unit/storage/test_embedding_needs_reindex_race_evidence.py
```

Result: **49 passed in 2.97 seconds**.

### Modified FTS, repair, and schema routes

```bash
python -m pytest -q \
  tests/unit/storage/test_fts_bloat_invariants.py \
  tests/unit/storage/test_repair.py \
  tests/unit/storage/test_schema_policy_contracts.py
```

Result: **69 passed in 22.46 seconds**.

### Modified insight and independent survivor routes

```bash
POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-testdiet04-final-insight \
python -m pytest -q \
  tests/unit/storage/test_session_insight_refresh.py \
  tests/unit/storage/test_derived_freshness_survivor.py
```

Result: **37 passed in 8.02 seconds**.

### Expanded adjacent-route reruns

Before the final anti-vacuity test was added, three broader adjacent families also completed successfully in the same final production worktree:

- FTS/search/repair/startup adjacent family: **149 passed in 11.45 seconds**;
- embedding storage/API/MCP/orphan-convergence adjacent family: **115 passed in 8.02 seconds**;
- insight/provider endpoint/convergence adjacent family: **85 passed in 9.23 seconds**.

These counts are supplemental and overlap the disjoint 291-test result; they are not added to it.

### Patch-applied clean-worktree test

`PATCH.diff` was applied to a detached clean worktree at the named base. The resulting 36 changed paths were byte-compared with the implementation worktree.

```bash
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
python -m pytest -q tests/unit/storage/test_derived_freshness_survivor.py
```

Results:

- patch apply check: passed;
- patch apply: passed;
- patched content comparison: **36 of 36 paths identical**;
- survivor module: **9 passed in 1.80 seconds**.

## Static and generated-surface checks

The changed Python file set was formed from tracked modifications plus untracked new Python files, producing 32 paths.

```bash
mapfile -t pyfiles < <(
  { git diff --name-only -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u
)
ruff format --check "${pyfiles[@]}"
ruff check "${pyfiles[@]}"
git diff --check
```

Results:

- Ruff format: **32 files already formatted**;
- Ruff lint: **all checks passed**;
- Git whitespace check: passed.

The same Ruff checks passed in the clean patch-applied worktree.

The three new production modules were checked independently under strict typing:

```bash
mypy --strict --follow-imports=skip \
  polylogue/storage/derivation_identity.py \
  polylogue/storage/embeddings/freshness.py \
  polylogue/storage/insights/session/provider_usage_freshness.py
```

Result: **success, no issues in 3 source files**.

The available environment had mypy 2.3.0, outside the repository’s declared `<1.21` range. A transitive changed-production run reported four errors in unchanged modules; a base comparison produced the same four errors. That out-of-range run is recorded as baseline noise, not a green repository-wide type gate.

Task-relevant generated surfaces were regenerated twice:

```bash
python -m devtools render topology-projection
python -m devtools render topology-status
```

The before/after hashes were identical:

- `docs/plans/topology-target.yaml`: `8f5189f790e4b19117097c226c67ff06f9d570564d10a77b9b96778401e655cc`;
- `docs/topology-status.md`: `9945da238834e2028e13364e6991e7b76ff5ae30cf6d776230587fcef15a22fe`.

A broader `python -m devtools render all --check` invocation passed the CLI reference, CLI output schemas, OpenAPI, and devtools reference checks, then exceeded the external 180-second command budget during later rendering. It did not report generated drift before termination. The full command is therefore not claimed as complete.

## Incomplete or unverified checks

The following were not executed and are not claimed:

- complete repository unit/property/integration suite;
- repository-pinned `devtools verify`, testmon seed, or full mypy environment;
- complete `render all --check` run;
- Nix package, service, browser, or deployed daemon validation;
- live operator archive migration/rebuild;
- large-archive FTS exact-audit resource envelope;
- live metrics or retained `ops.db` drift history;
- integrated testdiet-02/testdiet-03 worktree.

One monolithic all-modified pytest invocation exceeded its external command budget after progressing through most of the selection. It produced no test failure. The same modules were rerun to completion in the four disjoint groups above, which are the authoritative results.
