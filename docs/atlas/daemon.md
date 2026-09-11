# Daemon

## Area boundary

The daemon is the intended live write owner. It holds process-lifetime rebuild exclusion, serializes admitted SQLite mutations, drives source watching and derived convergence, and exposes HTTP/UDS readers (`polylogue/daemon/cli.py:2542-2600`; `polylogue/daemon/write_coordinator.py:1-6`).

## Runtime ownership

- `run_daemon_services` holds archive writer/rebuild exclusion for the process lifetime because startup recovery, acquisition, and periodic convergence all mutate storage (`polylogue/daemon/cli.py:2542-2585`).
- `DaemonWriteCoordinator` serializes write effects inside the daemon process; admitted work retains authority until it actually finishes, including after caller cancellation (`polylogue/daemon/write_coordinator.py:280-340`; `polylogue/daemon/write_coordinator.py:241-279`).
- HTTP and UDS mutation routes receive a bridge to that coordinator rather than opening an independent writer path (`polylogue/daemon/cli.py:3042-3085`; `polylogue/daemon/write_coordinator.py:665-729`).
- Startup recovers embedding lifecycle before publishing API sockets, then performs FTS and lineage readiness and blob-publication reconciliation before catch-up (`polylogue/daemon/cli.py:3042-3122`).
- Periodic owners cover raw materialization, insights, convergence debt, FTS, WAL, embeddings, status, judgments, blob GC/publication reconciliation, secret scans, and acquisition (`polylogue/daemon/cli.py:3138-3174`).
- Archive-bound FTS, derived-readiness, and checkpoint writers receive the selected archive root even when the active index generation is stored elsewhere, so a generation path cannot borrow another archive's admission (`polylogue/daemon/convergence_stages.py:99-110`; `polylogue/daemon/cli.py:316-325`; `polylogue/storage/sqlite/wal_checkpoint.py:153-187`).

## Current converger

`DaemonConverger` is an ordered generic stage runner. A stage supplies check/execute functions, optional batch/session functions, barriers, and `false_means_pending` semantics (`polylogue/daemon/convergence.py:59-87`; `polylogue/daemon/convergence.py:162-176`).

Current default order:

1. Optional Sinex publication when configured (`polylogue/daemon/convergence_stages.py:1227-1250`).
2. Raw parse recovery (`polylogue/daemon/convergence_stages.py:1250-1260`).
3. Raw-authority verdict cache (`polylogue/daemon/convergence_stages.py:1250-1260`).
4. FTS (`polylogue/daemon/convergence_stages.py:1250-1260`).
5. Embeddings (`polylogue/daemon/convergence_stages.py:1250-1260`).
6. Claude workflow derivation (`polylogue/daemon/convergence_stages.py:1250-1260`).
7. Delegation work evidence (`polylogue/daemon/convergence_stages.py:1250-1260`).
8. Derived tables (`polylogue/daemon/convergence_stages.py:1250-1260`).
9. FTS readiness and standing queries (`polylogue/daemon/convergence_stages.py:1250-1260`).

The watcher is constructed with this converger and the shared write coordinator (`polylogue/daemon/cli.py:3210-3232`; `polylogue/daemon/cli.py:3238-3245`).

## Execution semantics

- Stages execute in order and barrier failures prevent downstream execution for the affected subject (`polylogue/daemon/convergence.py:300-380`; `polylogue/daemon/convergence.py:390-500`).
- Batch execution rechecks subjects after a false result from a `false_means_pending` stage, so completed siblings can become done while remaining work stays pending (`polylogue/daemon/convergence.py:470-530`).
- Session-scoped convergence uses the same ordered stage and barrier model rather than a second convergence engine (`polylogue/daemon/convergence.py:530-620`).

## Quiet-window deferral

- Derived-table rebuilds can rehydrate an entire large session, so actively changing source sessions are removed from the current batch (`polylogue/daemon/convergence_stages.py:533-560`; `polylogue/daemon/convergence_stages.py:607-634`).
- If every selected session is hot, the stage returns false; if some are cool, those are rebuilt and the stage still returns false to preserve the remaining obligation (`polylogue/daemon/convergence_stages.py:533-560`; `polylogue/daemon/convergence_stages.py:607-634`).
- Deferred derived-table work uses ordinary bounded retry backoff (`polylogue/sources/live/convergence_debt_retry.py:12-30`).

## `convergence_debt`

- A pending stage is classified as deliberate deferral rather than failure; failed and deferred rows remain distinguishable (`polylogue/sources/live/convergence_debt.py:10-20`; `polylogue/sources/live/convergence_debt.py:52-80`).
- Debt is persisted in disposable `ops.db`, keyed by stage, target type, and target ID, with retry time and attempt state (`polylogue/storage/sqlite/archive_tiers/ops.py:160-190`).
- Successful passes clear stale stage debt; unresolved work is recorded against session IDs when available, otherwise source paths (`polylogue/sources/live/convergence_outcome.py:13-49`).
- The periodic retry owner executes only the recorded stage and subject. Legacy rows named `convergence` retain all-stage fallback behavior (`polylogue/daemon/cli.py:977-1050`; `polylogue/daemon/cli.py:2247-2304`).

## Current runtime state

Runtime evidence outside Git, observed 2026-08-25:

- `systemctl --user is-enabled polylogued.service` returned `masked-runtime`.
- `systemctl --user is-active polylogued.service` returned `inactive`.
- Unit properties reported `LoadState=masked`, `UnitFileState=masked-runtime`, and runtime control fragment `/run/user/1000/systemd/user.control/polylogued.service`.

This means the ownership machinery described above is present but no live daemon currently holds it. The executable owner remains `run_daemon_services` (`polylogue/daemon/cli.py:2663-2700`).

## SELECT-HYBRID direction

Decision record outside Git: `polylogue-04r9f` is closed with SELECT-HYBRID selected. The target is one small recurring registry, per-key `VALID/MISSING/STALE/EXCESS`, `DONE/PENDING/FAILED`, domain-owned required/inspect/compute/publish adapters, process-local scheduling, compute outside the writer lease, and publish-time binding revalidation under `BEGIN IMMEDIATE`.

The registry itself has not landed in this HEAD. Current code still exposes the generic `ConvergenceStage` abstraction and the nine-stage default list (`polylogue/daemon/convergence.py:59-87`; `polylogue/daemon/convergence_stages.py:1227-1260`). The 04r9f experiment implementations were explicitly disposable, so their absence is intentional task state rather than a missing merge.

Two of its properties do hold for embeddings (polylogue-c0l7n). `embed_archive_session_sync` is split into admitted reservation, lease-free provider computation, and admitted publication that re-acquires a fresh generation binding and revalidates the reserved attempt (`polylogue/storage/embeddings/materialization.py`). The `embed` stage refuses to call a provider while the writer gate is held and defers to convergence debt instead (`polylogue/daemon/convergence_stages.py`), and the three production owners, live-batch ingest, convergence-debt retry, and the embedding backlog, run the pass through `polylogue/daemon/embedding_owner.py` on the process's shared bounded compute capacity. No other derivation computes outside the lease yet.

## Workload-probe honesty

- `_scalar_int` returns `Evidence[int]`, so a failed read is `Unavailable` and cannot be spelled as zero (`polylogue/operations/daemon_workload_probe.py:169-182`).
- A cheap table count that did not answer is labelled `"unavailable"` with `UNKNOWN_TABLE_COUNT`, never `"exact"` (`polylogue/operations/daemon_workload_probe.py:215-232`).
- A readiness count that did not answer refuses the whole derived-readiness block, which reports `checked: false` plus the sqlite reason instead of comparing zero to zero and claiming ready (`polylogue/operations/daemon_workload_probe.py:1471-1500`).
- Diagnostic attempt counts report `null` for an unanswered read, distinct from a measured `0` (`polylogue/operations/daemon_workload_probe.py:588-640`).

## Gotchas

- `ops.db` debt is retry bookkeeping, not durable semantic authority; losing it may repeat convergence work (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:70-75`; `polylogue/storage/sqlite/archive_tiers/ops.py:153-170`).
- Returning false can mean pending bounded work, not failure; callers must inspect typed stage state (`polylogue/daemon/convergence.py:120-142`; `polylogue/sources/live/convergence_debt.py:52-80`).
- Schema-blocked startup may keep health/lifecycle reporting alive while withholding archive mutation and convergence (`polylogue/daemon/cli.py:3055-3074`).
- Process-local serialization is not proof that arbitrary external code cannot open SQLite directly; exclusion and surface routing are separate controls (`polylogue/daemon/write_coordinator.py:1-6`; `polylogue/daemon/cli.py:2683-2699`).

## DISCREPANCIES

- The daemon contract says it owns all writes and the main process is the sole SQLite writer. Code enforces that policy inside daemon routes, but writable `ArchiveStore` entry points remain directly callable outside the coordinator (`polylogue/storage/sqlite/archive_tiers/archive.py:1034-1083`; `polylogue/daemon/write_coordinator.py:1-6`).
- The repository contract compresses convergence to FTS, embeddings, and insights. The current default converger also includes raw parse recovery, raw-authority caching, Claude workflow, delegation evidence, FTS readiness, standing queries, and optional Sinex publication (`polylogue/daemon/convergence_stages.py:1227-1260`).
- Operationally, the daemon is runtime-masked and inactive, so the documented live-owner posture is not the machine’s current state. This discrepancy is external runtime state, not represented in repository files.

verified: 83ffd21c3 2026-09-11
