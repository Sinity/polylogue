# Test design and execution

## Survivor

Primary test:

```text
tests/unit/daemon/test_convergence_restart_law.py::test_convergence_debt_survives_restart_and_reaches_one_terminal_fact_set
```

The test is one state law rather than a set of disconnected mock examples.

### Sequence

1. Seed two Codex sessions through `write_parsed_session_to_archive`, including real `source.db` raw authority and real `index.db` parsed rows.
2. Make only the target source active-growing with the production hot-source size threshold and a deterministic future mtime.
3. Run the real insights stage through `DaemonConverger.converge_batch`.
4. Observe `StageState.PENDING` and the production deferred error.
5. Map the source path to its exact session through `record_convergence_outcome`; verify a durable `ops.db` insights row with attempts 1 and status `deferred`.
6. Delete only the target session's real FTS rows and record a separate FTS session debt row with a future retry deadline.
7. Force only insights due and launch a fresh Python process that calls `_drain_convergence_debt_once`.
8. While the source remains hot, verify the same insights `debt_id` and `created_at_ms`, attempts 2, deferred status, untouched FTS row, still-stale FTS materialization, unchanged raw authority, and no profile for target or unrelated session.
9. Truncate the target source below the production hot-source threshold, force insights due, and retry in another fresh interpreter.
10. Verify only insights debt is cleared, only the target profile is materialized, unrelated session remains untouched, FTS debt remains exact and future-dated, and source authority is unchanged.
11. Force FTS due and retry in a third fresh interpreter.
12. Verify only FTS debt is cleared, FTS rows are repaired, and all insight facts remain byte/value-equivalent to their pre-FTS snapshot.
13. Run another fresh-process drain; verify zero work, empty ledger, zero public failed/due counts, unchanged facts, and unchanged raw authority.
14. Seed a separate quiet archive and run uninterrupted insights convergence.
15. Compare stable profile/materialization/work-event/phase/thread/table facts between interrupted/restarted and uninterrupted archives.
16. Build the current `partial_convergence_canary_spec` `WorkloadReceipt` with all declared phases in order and cleanup/quiescence complete.

## Production dependencies exercised

- `polylogue.storage.sqlite.archive_tiers.write.write_parsed_session_to_archive`
- `polylogue.sources.live.batch_observability.session_ids_for_source_path` through `record_convergence_outcome`
- `polylogue.sources.live.convergence_debt.convergence_debt_from_states`
- `polylogue.sources.live.convergence_outcome.record_convergence_outcome`
- `polylogue.sources.live.cursor.CursorStore`
- `polylogue.storage.sqlite.archive_tiers.ops_write.add_convergence_debt`
- `polylogue.daemon.convergence.DaemonConverger`
- `polylogue.daemon.convergence_stages.make_insights_stage`
- `polylogue.daemon.convergence_stages.make_fts_stage`
- `polylogue.daemon.cli._drain_convergence_debt_once`
- `polylogue.daemon.convergence_debt_status.convergence_debt_summary_info`
- `polylogue.scenarios.partial_convergence_canary_spec`
- `polylogue.scenarios.WorkloadReceipt`

The harness reads SQLite only for independent facts and deterministic deadline control. It never decides whether a stage needs work or how to converge it.

## Independent oracle

Debt oracle:

- exact `debt_id`;
- exact `(stage, subject_type, subject_id)`;
- `created_at_ms` preservation;
- attempt count;
- status and retry deadline;
- cleanup/absence.

Materialized oracle:

- `session_profiles` stable columns;
- `insight_materialization` receipts excluding attempt-time materialization timestamps;
- session work events;
- session phases;
- threads and thread membership;
- archive table cardinalities;
- FTS match count and stage check;
- raw authority identity/hash/size/acquisition facts.

Public oracle:

- deferred insights are not reported as failed;
- unrelated failed FTS debt is visible;
- final failed and retry-due counts are zero.

The terminal comparison is against a separately seeded uninterrupted archive. It does not infer success from the production retry loop's own trace.

## Determinism

No sleeps or polling loops are used.

- Retry readiness is controlled by writing `next_retry_at` on the actual `ops.db` row.
- Hotness is held deterministically with a sparse file above the production threshold and a future mtime.
- Quietness is reached deterministically by truncating below the size threshold.
- Restart is a fresh interpreter process, not an object recreation in the same process.
- The unrelated stage is blocked by a future retry deadline and checked through real FTS state.

## Anti-vacuity mutation

Representative mutation: restore the snapshot's old retry implementation, which ignores `debt.stage`, executes all default stages for the selected subject, clears all subject debt, and recreates pending rows.

Actual sensitivity run:

```text
PYTHONPATH=<clean-unmodified-snapshot> <shared-venv>/bin/pytest -q \
  tests/unit/daemon/test_convergence_restart_law.py -x
```

Result:

```text
exit status 1
1 failed in 3.18s
first failure: deferred_again.debt_id != due_insights_debt.debt_id
```

Why it fails: the old loop deletes the due insights row and recreates it after the still-hot retry, so durable effect identity and accumulated attempt history are lost. The same old all-stage execution also repairs/consumes the deliberately unrelated FTS obligation; later assertions protect that stage-scope dimension.

Other named mutations guarded by the survivor:

- Drop `record_convergence_debt` after a bounded false result: no durable row exists before restart.
- Clear debt before stage success: the still-hot retry loses the insights row.
- Skip `_drain_convergence_debt_once`: the quiet target never gains a profile and debt never reaches quiescence.
- Resolve a different source/session: the unrelated session gains facts or target facts remain absent.
- Run all stages for exact debt: FTS rows and FTS debt change during the insights retry.
- Recreate instead of upsert exact debt: UUID/creation time changes and attempts reset.
- Clear by subject instead of exact stage: future-dated FTS debt disappears during insights completion.

## Commands and results

### Formatting, lint, syntax, typing

```text
.venv/bin/ruff format polylogue/daemon/cli.py \
  tests/infra/convergence_harness.py \
  tests/unit/daemon/test_convergence_restart_law.py
result: one file formatted; final files stable

.venv/bin/ruff check <same files>
result: All checks passed

.venv/bin/python -m py_compile <same files>
result: pass

.venv/bin/mypy <same files>
result: Success: no issues found in 3 source files
```

### Primary and adjacent tests

```text
.venv/bin/pytest -q tests/unit/daemon/test_convergence_restart_law.py -vv
result: 1 passed in 8.29s

.venv/bin/pytest -q \
  tests/unit/daemon/test_daemon_cli.py \
  tests/unit/daemon/test_convergence_stages.py \
  tests/unit/daemon/test_convergence_restart_law.py
result: 126 passed in 15.65s

for seed in 101 202 303; do
  .venv/bin/pytest -q --randomly-seed="$seed" \
    tests/unit/daemon/test_convergence_restart_law.py
done
result: 1 passed in 7.75s; 1 passed in 8.42s; 1 passed in 7.98s

.venv/bin/pytest -q \
  tests/unit/daemon/test_catch_up_observability.py \
  tests/unit/daemon/test_convergence_debt_alert.py \
  tests/unit/sources/test_live_batch_support.py \
  tests/unit/sources/test_live_catchup_planning.py
result: 106 passed in 22.65s
```

### Patch integrity

```text
git diff --check
result: pass

git apply --check PATCH.diff in a separate clean worktree at
f654480cadb7cc4c194704e24dfd483199547b35
result: pass

apply patch in that worktree and run survivor
result: 1 passed in 8.16s
```

### Supplemental baseline failures

A larger adjacent run produced `158 passed, 2 failed`. Both failures reproduce unchanged on the unmodified snapshot:

```text
tests/unit/daemon/test_daemon_status.py::test_build_daemon_status_detects_broken_append_head_blocks_converged
tests/unit/daemon/test_daemon_status.py::test_daemon_and_direct_claim_guard_share_mixed_frontier_summary
```

They concern append-head/frontier status and are outside this patch.

### Repository wrapper limitation

```text
.venv/bin/python -m devtools test tests/unit/daemon/test_convergence_restart_law.py
result: exit 125 before pytest
reason: wrapper requires a non-disk-backed pytest temp area and refused the
container's 64 MiB /dev/shm
```

Direct pytest execution above is verified. The complete repository suite, Nix/package build, live daemon, service supervisor, and deployment checks remain unverified.
