# testdiet-02 — convergence restart and debt liveness handoff

## Mission and result

This package implements a survivor law for daemon convergence across bounded deferral, durable debt, process restart, stage-scoped retry, repeated convergence, and eventual quiescence.

The investigation found a production composition defect in `polylogue.daemon.cli._drain_convergence_debt_once`: each persisted row already carried a stage identity, but the retry loop discarded that identity, ran every default convergence stage for the subject, cleared every debt row for that subject, and recreated only the stages still pending. That behavior could:

- replay an unrelated derived stage during a retry;
- consume a future-dated debt row that was not selected for retry;
- replace a durable debt identity instead of advancing its attempt history;
- reset retry history after a still-pending result.

`PATCH.diff` changes the production drain to run the recorded stage for the recorded subject, update that exact row after another defer/failure, and clear only that exact row after success. The historical generic `stage="convergence"` row remains an explicit all-stage migration fallback. No universal lifecycle table, alternate convergence state machine, or test-only production trace was added.

A real-SQLite survivor now composes the typed archive writer, source-to-session resolution, `DaemonConverger`, the production insights and FTS stages, the `CursorStore` `ops.db` ledger, public debt status, the current partial-convergence workload declaration, and fresh Python interpreter processes. It proves that a hot insights retry preserves row identity and increments attempts without touching unrelated FTS debt, a later quiet retry materializes only the intended session, the independent FTS retry drains only its own row, repeated retry is a no-op, cleanup is complete, and the stable terminal facts equal an uninterrupted run.

## Snapshot identity

Patch base:

- Source archive manifest: `polylogue-manifest.json`
- Manifest source: `/realm/project/polylogue`
- Manifest generated: `2026-07-17T043202Z`
- Manifest branch: `master`
- Commit: `f654480cadb7cc4c194704e24dfd483199547b35`
- Commit subject: `chore(beads): file archive-insight benchmark findings wofr/vhjs/r7p6/1wtm`
- Bundle refs: `master`, `origin/master`, and `origin/HEAD` all resolve to that commit
- Manifest dirty flag: `true`
- Recoverable tracked dirty patch: none

The archive's branch-delta patch, changed-file list, and branch-only log are empty. Overlaying the supplied `polylogue-working-tree.tar.gz` onto a clean worktree at the named commit leaves `git status` clean and no tracked diff. A direct tree comparison found no differing common tracked files. The tar intentionally omits some repository-local files and contains ignored packaging/runtime material, so the manifest's `dirty=true` cannot be converted into a source patch. `PATCH.diff` therefore targets the named commit, not an invented dirty state.

## Inspected evidence

Repository instructions and architecture:

- `AGENTS.md`: substrate-first semantics, daemon sole-writer rule, five-tier durability model, `ops.db` convergence ledger, `false_means_pending`, hot-file quiet deferral, and session-scoped retry methods.
- Test Suite Diet `architecture/03-concurrent-writes-publication-and-resume.md`: domain-specific receipt-backed sagas, observable intermediate states, exact effect identity, idempotent resume, and deterministic failpoints rather than sleeps.
- Test Suite Diet `architecture/09-capture-delivery-and-deployed-status.md`: production-route composition, durable/idempotent retry, explicit residual debt, bounded cleanup, and no parallel test-only service registry.
- Test Suite Diet `dossiers/convergence-restart-retry.md` and `areas/daemon-convergence.md`: the exact survivor obligations, planned paths, real-SQLite/frozen-barrier guidance, final-state equality, scoped retry, status agreement, and mutation requirement.

Production source followed beyond the obvious files:

- `polylogue/daemon/convergence.py`
- `polylogue/daemon/convergence_stages.py`
- `polylogue/daemon/convergence_standing_queries.py`
- `polylogue/daemon/cli.py`
- `polylogue/daemon/convergence_debt_status.py`
- `polylogue/sources/live/convergence_debt.py`
- `polylogue/sources/live/convergence_outcome.py`
- `polylogue/sources/live/convergence_debt_retry.py`
- `polylogue/sources/live/cursor.py`
- `polylogue/sources/live/batch_observability.py`
- `polylogue/storage/sqlite/archive_tiers/ops.py`
- `polylogue/storage/sqlite/archive_tiers/ops_write.py`
- `polylogue/storage/sqlite/archive_tiers/write.py`
- `polylogue/scenarios/workload.py`

Tests and test infrastructure inspected:

- `tests/unit/daemon/test_daemon_convergence.py`
- `tests/unit/daemon/test_convergence_stages.py`
- `tests/unit/daemon/test_convergence_final_state.py`
- `tests/unit/daemon/test_convergence_debt_alert.py`
- `tests/unit/daemon/test_daemon_cli.py`
- `tests/unit/daemon/test_daemon_status.py`
- `tests/unit/daemon/test_catch_up_observability.py`
- `tests/unit/sources/test_live_batch_support.py`
- `tests/unit/sources/test_live_catchup_planning.py`
- `tests/infra/frozen_clock.py`

Beads inspected from the supplied export:

- `polylogue-1xc.14`, “Declare workload envelopes and resource receipts once” (`open`)
- `polylogue-1xc.14.1`, “Derive archive-scale workload profiles from provider schemas” (`in_progress`)

The later `polylogue-1xc.14.1` notes say the partial-convergence canary and workload-receipt substrate landed as a first slice, while production-route canaries remained open. The implementation reuses those current types and does not claim the parent Beads are closed.

History inspected:

- Function lineage for `_drain_convergence_debt_once`, including its origin in `a73d69b758` and later session/FTS additions.
- `c20286459` — workload profiles and receipt-backed production-route artifacts.
- `f0c1b489b` — archive contract verification repair.
- `36001d023` — durable publication convergence and receipt barriers.
- `a952221cd` — session-scoped standing-query convergence stage.
- Recent `polylogue/daemon/cli.py` and convergence-path history through the named snapshot.

## Mechanism

### Production retry selection

The drain now builds the default stage tuple once, indexes it by `stage.name`, and groups due `source_path`/`session_id` debt by the persisted stage. For each group it creates a real `DaemonConverger` containing only that stage and executes the existing path- or session-scoped production method. Results are keyed by `(stage, subject_type, subject_id)` so two rows for the same session cannot share or overwrite each other's outcome.

The generic legacy `convergence` stage is the only all-stage fallback. If it remains pending, only the generic row is removed and replaced with exact pending-stage rows; unrelated exact rows survive.

### Durable row lifecycle

For exact stage rows:

- success clears only `(stage, subject_type, subject_id)`;
- defer/failure calls `CursorStore.record_convergence_debt` on that same identity;
- the existing SQL upsert preserves `debt_id` and `created_at_ms` while incrementing attempts when the retry is due;
- `materializer_version` is retained;
- an unavailable stage is recorded honestly on the same row instead of silently disappearing.

FTS-surface debt uses the same exact-row clear/update rule.

### Survivor and oracle

`tests/infra/convergence_harness.py` is a narrow adapter around existing typed writers and real SQLite tables. It does not implement convergence decisions. It supplies:

- two typed Codex archive sessions, one target and one unrelated;
- a sparse active-growing source variant using the production hot-source threshold;
- direct, deterministic retry-deadline barriers in the real `ops.db` row;
- real message-FTS staleness by deleting only the target session's FTS rows;
- raw-authority facts from `source.db`;
- a stable materialization oracle over profile, materialization receipts, work events, phases, threads, thread membership, and table cardinalities, excluding attempt-time timestamps.

The survivor creates debt in the parent pytest process and invokes the real production drain in fresh Python interpreters. This is a real persistence/process boundary without a sleep or a fake daemon registry.

## Decisions

- Keep the domain-specific `convergence_debt` ledger; do not introduce a universal lifecycle table.
- Reuse `false_means_pending`, existing session-scoped stage methods, and the existing SQL upsert.
- Treat stage identity as part of retry identity, not metadata.
- Preserve generic `convergence` rows as a compatibility fallback rather than deleting support for them.
- Use `next_retry_at` updates and source size/mtime boundaries as deterministic barriers; do not sleep.
- Use a fresh interpreter for restart proof; do not claim a systemd/service-supervisor kill test.
- Use durable facts and an uninterrupted baseline as the oracle; do not add a production trace seam that mirrors the algorithm.
- Emit the existing `WorkloadReceipt` for the declared `partial_convergence_canary_spec`; do not create another receipt vocabulary.
- Leave all existing tests and helpers in place.

## Changed files

- `polylogue/daemon/cli.py`
  - stage-scoped debt selection;
  - exact-row success/failure/defer handling;
  - preservation of materializer version;
  - generic-row migration fallback.
- `tests/infra/convergence_harness.py`
  - real-SQLite typed archive fixture;
  - durable debt-row and retry-deadline probes;
  - independent raw/materialized fact oracles;
  - real FTS staleness and query probes.
- `tests/unit/daemon/test_convergence_restart_law.py`
  - one cohesive restart/retry/quiescence survivor with process boundaries, public status, cleanup, baseline equivalence, and workload receipt.

`FILES/` is intentionally omitted because the unified patch fully disambiguates all changes.

## Acceptance matrix

| Mission obligation | Evidence in patch | Result |
| --- | --- | --- |
| Bounded work may defer honestly | Real insights stage returns pending for a hot source; ops row is `deferred`; public failed count remains zero | Satisfied |
| Debt survives restart | Debt is written before a fresh Python process invokes the production drain | Satisfied |
| Retry addresses intended session/stage | Insights-only converger selection; future-dated FTS debt and stale FTS rows remain unchanged | Satisfied |
| No unrelated acquisition replay | Session-scoped retry uses the persisted session id; raw source facts remain unchanged; unrelated session remains unmaterialized | Satisfied |
| Retry count and debt identity survive | Same `debt_id`/`created_at_ms`, attempts advance from 1 to 2 after a hot retry | Satisfied |
| Consume debt only after success | Hot retry retains exact row; quiet retry clears it after materialization | Satisfied |
| Repeated convergence is idempotent | Final fresh-process drain returns zero and stable facts do not change | Satisfied |
| Same terminal facts as uninterrupted | Stable insight/profile/event/thread oracle equals a separately seeded uninterrupted archive | Satisfied |
| No duplication | Stable facts and table counts remain identical after FTS retry and repeated drain | Satisfied |
| Public status and cleanup agree | Final ledger empty, failed/retry-due counts zero, quiescent receipt has cleanup complete | Satisfied |
| Deterministic test control | Direct retry deadlines plus sparse-size/quiet boundary; no sleep/polling | Satisfied |
| Real production route | Typed writer, real stage factories, real `DaemonConverger`, real `CursorStore`, real CLI drain, real SQLite | Satisfied |
| Mutation sensitivity | Unmodified snapshot fails on changed debt UUID after first hot retry | Satisfied |
| No parallel lifecycle framework | Existing ledger and workload receipt reused | Satisfied |

## Proposed dominated tests — not deleted

These are review candidates only, pending independent local certification and coverage evidence:

- `tests/unit/daemon/test_daemon_cli.py::test_drain_convergence_debt_retries_due_items_without_source_failure`
- `tests/unit/daemon/test_daemon_cli.py::test_drain_convergence_debt_retries_session_subjects_without_source_lookup`
- `tests/unit/daemon/test_convergence_stages.py::test_profile_canary_defers_hot_session_then_converges_when_quiet`

The survivor dominates their broad happy-path temporal story, but the first still isolates source-path handling, the second is a fast session-method diagnostic, and the third is a direct stage/receipt diagnostic. No deletion is included.

`tests/unit/daemon/test_convergence_final_state.py` is not proposed for deletion: it exercises live-ingest composition, although its legacy debt-table assertion deserves a separate cleanup review.

## Apply order

From a clean checkout at `f654480cadb7cc4c194704e24dfd483199547b35`:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
python -m pytest -q tests/unit/daemon/test_convergence_restart_law.py
```

Then run the broader local gate appropriate to the operator's environment.

## Exact verification performed

Successful checks:

```text
ruff format/check on all three changed files
  result: clean; one file was formatted before final patch generation

python -m py_compile on all three changed files
  result: pass

mypy polylogue/daemon/cli.py tests/infra/convergence_harness.py tests/unit/daemon/test_convergence_restart_law.py
  result: Success: no issues found in 3 source files

pytest -q tests/unit/daemon/test_convergence_restart_law.py -vv
  result: 1 passed in 8.29s

pytest -q tests/unit/daemon/test_daemon_cli.py tests/unit/daemon/test_convergence_stages.py tests/unit/daemon/test_convergence_restart_law.py
  result: 126 passed in 15.65s

pytest survivor with --randomly-seed=101, 202, 303
  result: 1 passed in 7.75s; 1 passed in 8.42s; 1 passed in 7.98s

pytest -q test_catch_up_observability.py test_convergence_debt_alert.py test_live_batch_support.py test_live_catchup_planning.py
  result: 106 passed in 22.65s

git diff --check
  result: pass

git apply --check PATCH.diff in a separate clean worktree at the named commit
  result: pass

apply PATCH.diff in that clean worktree, then run the survivor through the shared venv with that worktree first on PYTHONPATH
  result: 1 passed in 8.16s

anti-vacuity run: copy the survivor/harness onto the unmodified snapshot and run it
  result: expected failure, exit 1; 1 failed in 3.18s
  first failure: debt_id changed after the first still-hot retry
```

Supplemental check with known baseline failures:

```text
pytest over catch-up/status/debt/source planning set
  result: 158 passed, 2 failed

The same two failures reproduce on the unmodified clean snapshot:
- test_build_daemon_status_detects_broken_append_head_blocks_converged
- test_daemon_and_direct_claim_guard_share_mixed_frontier_summary
```

The repository's `.venv/bin/python -m devtools test tests/unit/daemon/test_convergence_restart_law.py` wrapper was attempted but refused to run because the container exposes only 64 MiB at `/dev/shm`; it exited 125 before pytest. Direct pytest execution is the verified result.

## Risks and remaining verification

- The process-restart proof uses fresh Python interpreters, not the operator's live daemon, systemd unit, NixOS deployment, watcher process, or signal/kill lifecycle. Those checks are unverified.
- The complete repository suite, Nix build, packaging gate, pre-push gate, and `devtools verify --quick` were not run. The focused and adjacent suites above are verified.
- The manifest advertises a dirty source, but no tracked dirty patch is recoverable. Any operator-only change excluded from the archive is necessarily unverified.
- Legacy generic `stage="convergence"` rows intentionally retain all-stage behavior because they carry no narrower stage identity. Exact rows are scoped.
- The survivor covers insights and FTS stage coexistence. It does not independently repeat the law for embeddings, standing queries, or Sinex publication barriers.
- `ops.db` is explicitly disposable in current architecture. This patch proves restart survival when that file survives process restart; it does not redefine tier durability across operator deletion/reset.
- The two unrelated daemon-status failures are baseline defects, not fixed or masked here.

A further iteration is likely a **small repair** if CI exposes formatting, platform, or test-isolation differences. A **substantial second pass** would be justified only to add a real service-supervisor kill/restart witness, extend the same law across embedding/standing-query/Sinex stages, or reconcile the two existing status-frontier failures. Those extensions would add meaningful coverage but are not required to make this patch coherent or apply-ready.
