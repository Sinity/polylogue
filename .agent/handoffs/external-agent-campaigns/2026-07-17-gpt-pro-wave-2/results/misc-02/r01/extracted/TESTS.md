# TESTS — `polylogue-wmsc`

## Test intent

The tests exercise the real archive selector, daemon/manual/preflight callers, canonical embeddings DDL, sqlite-vec writer, failure lifecycle, status projections, and sibling-tier convergence route. They do not substitute a parallel freshness implementation. Every new behavioral test names the production dependency and the representative regression or removal that must make it fail.

## New invariant suite

`tests/unit/storage/test_embedding_freshness_invariant.py` contains the direct acceptance proof.

| Test | Production dependency exercised | Anti-vacuity mutation/removal |
|---|---|---|
| `test_per_source_convergence_mutation_removing_shared_predicate_misses_changed_content` | `_archive_pending_embedding_session_ids()` → `select_pending_archive_session_window()` → shared exact predicate | Remove the shared derivation predicate or restore status-only clean-row trust; the same-ID changed-content fixture is omitted. |
| `test_daemon_backlog_mutation_restoring_stale_check_bypass_misses_changed_content` | `_drain_archive_embedding_backlog_once()` and configured `EmbeddingRecipe` | Restore `include_stale_checks=False` or stop passing the recipe; daemon backlog returns zero instead of one. |
| `test_manual_backfill_mutation_restoring_stale_check_bypass_misses_changed_content` | Click manual backfill route `_run_archive_backfill()` | Restore the manual bypass or select outside the canonical function; provider/store work is not invoked for the stale fixture. |
| `test_preflight_mutation_restoring_stale_check_bypass_misses_changed_content` | `build_preflight_report()` → `_select_archive_pending_window()` | Restore the preflight bypass; pending conversation/message and cost estimates become zero. |
| `test_recipe_model_swap_makes_every_materialized_session_stale` | v3 desired-key SQL and `EmbeddingRecipe.current()` | Ignore recipe identity/model in the desired key; already-materialized sessions remain falsely fresh. |
| `test_config_change_then_old_terminal_error_cannot_clear_new_generation` | `_reconcile_embedding_config_change()`, `record_embedding_failure()`, exact generation/key predicates | Remove generation/key conditions from the terminal failure update; the old non-retryable attempt clears the newer pending mark. The interleaving is explicit database operations, not sleeps. |
| `test_archive_check_reconciles_recipe_on_sibling_embeddings_tier` | `_archive_embed_check()` and `_reconcile_archive_embedding_config_change()` | Reconcile `index.db` rather than sibling `embeddings.db`; the derivation generation/key does not advance. |
| `test_unscoped_or_legacy_failure_receipt_cannot_project_over_keyed_generation` | unscoped failure recording and generation-zero failure resolution | Allow an unscoped or v2 receipt to project status after keyed state exists; current pending debt is cleared or blocked incorrectly. |
| `test_same_id_full_replace_reselects_and_atomically_replaces_vector_and_meta` | canonical selector, `complete_embedding_attempt_success()`, vec/meta tables | Skip content identity, omit full-session deletion, or publish without the terminal generation guard; selection or exact one-row replacement fails. |
| `test_derivation_key_value_shape_is_storage_neutral_and_generation_free` | `DerivationSubject`, `DerivationIdentity`, `DerivationKey`, `DerivationKeyLike` | Add generation/lifecycle/eligibility/result-integrity fields or embedding storage coupling to the shared value shape. |
| `test_recipe_mutation_removing_any_declared_computational_field_preserves_wrong_reuse` | all 12 `EmbeddingRecipe.identity()` fields | Remove any field from canonical recipe identity; mutating that field no longer changes the digest. The test is parameterized over canonicalization, selector, chunking, provider, model, revision, dimensions, task, input type, normalization, tool implementation, and input/schema version. |

The last parameterized test contributes twelve cases. Together with the other ten functions, the new file collects twenty-two tests.

## Existing tests updated or used as compatibility witnesses

- `tests/unit/storage/test_archive_tiers_embedding_write.py`: direct vector/meta/status writes, batched writes, retryable/terminal errors, audit retention, supersession, requeue, and acknowledgement. This is the canonical writer compatibility suite.
- `tests/unit/storage/test_embedding_contracts.py`: stale hash selection, bounded/unbounded selector parity, status counting, provider failure lifecycle, and existing archive embedding contracts. The tests that previously asserted a caller bypass now assert that restoring that bypass is a defect.
- `tests/unit/storage/test_embedding_needs_reindex_race_evidence.py`: prior success-write/config-change race evidence remains green under the stronger key/generation implementation.
- `tests/unit/storage/test_embedding_orphan_reconcile.py`: split-tier orphan behavior remains green; one obsolete monkeypatch argument tied to the removed selector switch was deleted.
- `tests/unit/daemon/test_embedding_convergence_progress.py`: convergence reads the embeddings tier and rejects a clean compatibility status row without derivation/materialization proof.
- `tests/unit/cli/test_embed_status_fast.py`: default status now attempts the bounded exact shared predicate rather than categorically avoiding exact state.
- CLI/API/MCP/readiness/metrics/runtime/source-selection suites listed below exercise all downstream projections and invocation routes.

No existing test or helper was deleted. No dominated deletion is proposed in this package.

## Commands and results

### Consolidated production-route compatibility run

The container could not install the locked development extras, so this run used the snapshot's project runtime packages plus `/opt/pyvenv`'s pytest. Plugin autoload was disabled because Hypothesis, pytest-xdist, and pytest-timeout were absent. A test-only import shim supplied the minimal Hypothesis configuration imports from `tests/conftest.py`, and a test-only plugin supplied xdist's `worker_id = "master"` fixture. Neither shim replaced production code, sqlite-vec behavior, selector logic, or writer behavior.

Equivalent command used:

```bash
cd /tmp/polylogue-repo
export PYTHONPATH=/tmp/polylogue-test-stubs:$PWD/.venv/lib/python3.13/site-packages:$PWD
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/opt/pyvenv/bin/python -m pytest \
  -p pytest_asyncio.plugin \
  -p pytest_worker_stub \
  -o addopts='' \
  tests/unit/storage/test_embedding_contracts.py \
  tests/unit/storage/test_embedding_freshness_invariant.py \
  tests/unit/storage/test_archive_tiers_embedding_write.py \
  tests/unit/storage/test_embedding_needs_reindex_race_evidence.py \
  tests/unit/storage/test_embedding_orphan_reconcile.py \
  tests/unit/daemon/test_embedding_convergence_progress.py \
  tests/unit/cli/test_embed_status_fast.py \
  tests/unit/api/test_embedding_readiness_api.py \
  tests/unit/daemon/test_embedding_readiness.py \
  tests/unit/daemon/test_metrics_endpoint.py \
  tests/unit/mcp/test_embedding_status_tool.py \
  -q
```

Result:

```text
174 passed, 2 warnings in 9.51s
```

The two warnings were unknown timeout configuration options because pytest-timeout was unavailable with plugin autoload disabled. They were not test failures or production warnings.

### Frozen-patch clean-worktree run

`PATCH.diff` was applied to a detached clean worktree at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`, then the new invariant suite and canonical writer suite were run through the same existing runtime:

```bash
cd /tmp/wmsc-apply-check
export PYTHONPATH=/tmp/polylogue-test-stubs:/tmp/polylogue-repo/.venv/lib/python3.13/site-packages:$PWD
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/opt/pyvenv/bin/python -m pytest \
  -p pytest_asyncio.plugin \
  -p pytest_worker_stub \
  -o addopts='' \
  tests/unit/storage/test_embedding_freshness_invariant.py \
  tests/unit/storage/test_archive_tiers_embedding_write.py \
  -q
```

Result:

```text
29 passed, 2 warnings
```

This run proves the delivered patch applies independently of the implementation worktree and that all twenty-two new cases plus the seven canonical writer cases execute after application.

### Syntax, patch, topology, and audit checks

```bash
cd /tmp/polylogue-repo
python -m compileall -q \
  polylogue/storage/derivation_identity.py \
  polylogue/storage/embeddings/identity.py \
  polylogue/storage/embeddings/materialization.py \
  polylogue/storage/embeddings/preflight.py \
  polylogue/storage/embeddings/status_payload.py \
  polylogue/storage/sqlite/archive_tiers/embedding_write.py \
  polylogue/storage/sqlite/archive_tiers/embeddings.py \
  polylogue/daemon/convergence_stages.py \
  polylogue/daemon/embedding_backlog.py \
  polylogue/cli/commands/embed.py
git diff --check
```

Result: clean.

Generated topology checks:

```bash
python devtools/build_topology_projection.py
python -m devtools render topology-status --check
python -m devtools verify topology --json
```

Observed topology result: `blocking=false`, zero orphan files, zero missing declared files, zero owner conflicts, and zero kernel-rule findings. Nine pre-existing storage-root `TBD` classifications remain; this patch adds no new `TBD` entry. The projection recognizes both new Python files and classifies `derivation_identity.py` as a storage-root primitive.

The read-only audit harness from `HANDOFF.md` was run against a synthetic same-ID/same-count changed-content archive. Result:

```json
{
  "eligible_sessions": 1,
  "exact_pending_sessions": 1,
  "old_bypass_pending_sessions": 0,
  "missed_by_old_bypass": 1,
  "source_identity_drift": 1,
  "partition_check": true
}
```

Unshown audit fields were zero. This proves the query numerically separates the exact predicate from the historical bypass on the target defect fixture.

## Development-environment limitation

The normal dependency hydration command was attempted:

```bash
cd /tmp/polylogue-repo
uv sync --extra dev --frozen
```

It failed while downloading `virtualenv==21.2.0` after three retries because DNS resolution for `files.pythonhosted.org` was unavailable. The package therefore does not claim execution of:

- full Ruff format/lint;
- strict mypy;
- `devtools verify --quick` or `devtools verify --all`;
- the complete repository test suite;
- tests requiring a real Hypothesis/xdist/timeout plugin installation.

The integrator should run the normal project gate in the repository devshell:

```bash
nix develop --command devtools verify --quick
nix develop --command devtools test \
  tests/unit/storage/test_embedding_freshness_invariant.py \
  tests/unit/storage/test_archive_tiers_embedding_write.py \
  tests/unit/storage/test_embedding_contracts.py \
  tests/unit/daemon/test_embedding_convergence_progress.py \
  tests/unit/cli/test_embed_status_fast.py
```

## Live checks not performed

No live daemon, archive, provider credentials, Voyage call, production file-set, or deployment was accessed. The following remain operator/integrator verification:

1. Run the read-only census from `HANDOFF.md` against a retained pre-rebuild archive copy and the rebuilt v3 tier.
2. Measure the exact status predicate's plan/runtime on the real archive; default status has a five-second SQLite progress budget and a legacy aggregate fallback.
3. Confirm provider budget before rebuilding the expensive embeddings tier.
4. Run postflight status/detail and daemon convergence, then retain or retire the v2 backup according to observed parity.
