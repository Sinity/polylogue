# Evidence and design findings

## Authority and snapshot reconstruction

The supplied project-state archive is the code authority. Its manifest identifies:

```text
source: /realm/project/polylogue
generated_at: 2026-07-17T043202Z
branch: master
commit: f654480cadb7cc4c194704e24dfd483199547b35
dirty: true
```

The all-refs bundle resolves `master`, `origin/master`, and `origin/HEAD` to that commit. The branch-delta patch, file list, and log are empty.

A clean detached worktree was created from the bundle and the supplied working-tree tar was overlaid. `git status --short --untracked-files=all` and `git diff --stat` were empty. A direct tree comparison found no differing common files; differences were packaging exclusions/ignored material (`.agent`, `.beads`, `.claude`, lockfiles omitted from the tar; `.benchmarks` and a browser-extension package lock present only in the tar). Therefore the manifest's dirty flag does not identify a recoverable tracked source patch. The apply base is the named commit.

## Repository constraints

`AGENTS.md` establishes the constraints used here:

- semantics belong in the substrate/product route, not duplicated surface logic;
- the daemon owns writes and remains the sole SQLite writer;
- `ops.db` holds `convergence_debt` and other daemon operational state;
- `DaemonConverger` stages have path, batch, and session-scoped variants;
- session-scoped methods exist specifically to retry debt without resolving back to source paths;
- `false_means_pending` means bounded successful work can remain pending rather than becoming a failure;
- hot sources intentionally defer insights until quiet.

The patch changes the daemon's existing production retry composition and uses the existing ledger/upsert. It does not add a second scheduler, repository, state machine, or lifecycle table.

## Test Suite Diet decisions applied

### Concurrent writes, publication, and resume

`architecture/03-concurrent-writes-publication-and-resume.md` says cross-tier work is an observable idempotent saga; checkpoints mean effects are durably receipted; resume skips only proven effects and retries pending/unknown effects idempotently; convergence debt retains its own domain state; deterministic transaction/publication barriers are preferred over sleeps.

Applied here:

- debt identity is treated as the effect identity;
- the existing upsert advances attempts on the same row;
- success clears only the proven stage effect;
- retry deadlines are deterministic barriers;
- raw/source authority is checked not to change during derived retry;
- generic legacy debt is migrated to exact pending stages without deleting unrelated exact rows.

### Capture delivery and deployed status

`architecture/09-capture-delivery-and-deployed-status.md` requires production-route composition, durable/idempotent queue/receipt behavior, explicit residual debt, bounded cleanup, and no parallel harness registry.

Applied here:

- the fresh process imports the production daemon drain;
- the harness only seeds and observes existing production types;
- final public debt status is checked against the durable ledger;
- residual debt is explicit throughout and empty only at quiescence;
- the existing `WorkloadReceipt` records the declared phase/cleanup result;
- no claim is made about a service-supervisor termination receipt or deployed service graph.

## Dossier and area packet

The supplied `dossiers/convergence-restart-retry.md` names these obligations:

- honest `false_means_pending` handling;
- debt survival across process restart;
- session-scoped retry that does not widen to irrelevant backlog;
- idempotence/no duplication;
- equality with uninterrupted final facts;
- status/alert agreement with debt and quiescence.

It proposes exactly:

- `tests/infra/convergence_harness.py`
- `tests/unit/daemon/test_convergence_restart_law.py`

The area packet asks for real SQLite, explicit barriers, the sequence `source append -> writer commit -> bounded stage -> debt -> restart -> retry -> quiescence`, and mutation of debt persistence/session scope/retry. The delivered test realizes that first slice with the current partial-convergence workload type.

## Beads

### `polylogue-1xc.14`

The shared `WorkloadEnvelopeSpec`/`WorkloadReceipt` contract is still open. Its design prohibits turning resource limits into semantic caps and requires explicit workload/build/archive/phase/cleanup evidence.

The survivor uses the existing receipt model and does not claim complete adoption across all performance paths.

### `polylogue-1xc.14.1`

The workload-profile child is `in_progress`. Its current notes say a first implementation slice landed deterministic workload profiles, archive composition, typed workload artifacts, and partial-convergence canary substrate; production-route canaries and further profile work remain.

The patch reuses `partial_convergence_canary_spec` and the typed writer. It does not invent a corpus generator or claim the Bead is complete.

## Production route findings

### `polylogue/daemon/convergence.py`

- `ConvergenceStage` already defines `check_sessions` and `execute_sessions`.
- `false_means_pending=True` maps a false result to `StageState.PENDING` and preserves an honest error.
- `DaemonConverger.converge_sessions` can target exact session subjects.

No new convergence state machine was needed.

### `polylogue/daemon/convergence_stages.py`

- The real insights stage defers active-growing source sessions until quiet.
- The real FTS stage supports exact session check/execute methods.
- Default stages carry stable names used in persisted debt rows.

These facts make stage-scoped retry implementable by selecting the existing stage object.

### `polylogue/sources/live/convergence_outcome.py`

- Path convergence output is resolved to session ids once when debt is created.
- Session debt is then durable and can be retried without rereading/re-resolving the source.

The retry patch preserves that intended boundary.

### `polylogue/sources/live/cursor.py` and ops DDL/write route

- `convergence_debt` has `UNIQUE(stage, target_type, target_id)` and a stable `debt_id`.
- `record_convergence_debt` uses `BEGIN IMMEDIATE` around read/modify/write.
- The SQL upsert explicitly preserves the existing `debt_id`, increments attempts, preserves `created_at_ms`, and updates the retry/error/status fields.
- Insights quiet deferral is stored as `status='deferred'`; other errors are `failed`.

The old CLI drain bypassed the value of this upsert by deleting the row first.

### `polylogue/daemon/convergence_debt_status.py`

- Public status reads `ops.db`.
- Deferred rows do not count as failures.
- Failed counts and retry-due counts are projected from the durable rows.

The survivor checks both the deferred and final/quiescent projections.

### Defect in `polylogue/daemon/cli.py`

At the named snapshot, `_drain_convergence_debt_once`:

1. selected due rows;
2. collapsed all session ids and paths without grouping by `debt.stage`;
3. instantiated one `DaemonConverger` with every default stage;
4. ran all stages for every selected subject;
5. on success, cleared all debt for the subject;
6. on pending/failure, cleared all debt for the subject and recreated rows from the whole subject state.

The persisted stage identity therefore did not control execution or cleanup. A due insights row could repair a non-due FTS stage, clear its future obligation, and replace its own durable identity.

## History

`git blame` shows the retry loop originated in `a73d69b758` (`perf(daemon): observe and retry live catch-up convergence (#1033)`) and accumulated session and FTS support later. The stage-agnostic grouping/clear behavior survived those additions.

Relevant later architecture/history inspected:

- `a952221cdcc4813ffcc4c9c18c4fd8981d5bbb2a`: session-scoped standing-query convergence and durable receipts.
- `36001d023b2cfe793cb19fdd7c42a87597356f48`: durable Sinex publication obligations and convergence barriers.
- `f0c1b489b84cd04aac840315e7e55fa23eb97e39`: trustworthy archive contract verification.
- `c20286459cf2c3d1e4c968a8584f13e7cd382ff2`: workload profiles and receipt-backed production-route artifacts.

These commits reinforce exact identity, durable receipt, session scoping, and real-route anti-vacuity rather than a new generic lifecycle abstraction.

## Mutation evidence

The survivor and helper were copied onto a clean unmodified worktree at the named snapshot. The test failed after the first fresh-process hot retry:

```text
assert deferred_again.debt_id == due_insights_debt.debt_id
```

The UUIDs differed. This is direct evidence that the old clear/recreate implementation loses durable debt identity. The delivered patch passes the same test and advances attempts from 1 to 2 on the same row.

The survivor also contains independent assertions that the future FTS row and stale FTS materialization remain untouched during the insights retry. They guard the second half of the defect: stage widening and subject-wide cleanup.

## Contradictions and boundaries

- Manifest `dirty=true` versus clean reconstructed tracked tree: reported explicitly; no dirty source patch is fabricated.
- Dossier generated against older `git_head=21f78b4db2ba62ff44b5f16dfab96067bc249b4c` versus current snapshot `f654480cadb7cc4c194704e24dfd483199547b35`: current source and later Beads notes win; all named paths/types were re-resolved.
- Dossier readiness says `prepared-not-execution-grade`, while the current snapshot now contains the workload-profile/receipt first slice. The delivered implementation uses that current substrate but does not claim full program completion.
- `ops.db` is disposable, yet debt survives ordinary process restart when the file remains. The patch does not claim survival after operator reset/deletion.
- The production service graph/termination receipt work in architecture 09 is not present as a complete deployed proof here. Fresh interpreter restart is verified; live supervisor/systemd behavior is not.
