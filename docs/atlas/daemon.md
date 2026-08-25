# Daemon

## Area boundary

The daemon is the intended live write owner. It holds process-lifetime rebuild exclusion, serializes admitted SQLite mutations, drives source watching and derived convergence, and exposes HTTP/UDS readers (`polylogue/daemon/cli.py:2663-2700`; `polylogue/daemon/write_coordinator.py:1-6`).

## Runtime ownership

- `run_daemon_services` holds archive writer/rebuild exclusion for the process lifetime because startup recovery, acquisition, and periodic convergence all mutate storage (`polylogue/daemon/cli.py:2683-2699`).
- `DaemonWriteCoordinator` serializes write effects inside the daemon process; admitted work retains authority until it actually finishes, including after caller cancellation (`polylogue/daemon/write_coordinator.py:178-196`; `polylogue/daemon/write_coordinator.py:223-340`).
- HTTP and UDS mutation routes receive a bridge to that coordinator rather than opening an independent writer path (`polylogue/daemon/cli.py:3133-3157`; `polylogue/daemon/write_coordinator.py:621-632`).
- Startup recovers embedding lifecycle before publishing API sockets, then performs FTS and lineage readiness and blob-publication reconciliation before catch-up (`polylogue/daemon/cli.py:3120-3157`; `polylogue/daemon/cli.py:3166-3205`).
- Periodic owners cover raw materialization, insights, convergence debt, FTS, WAL, embeddings, status, judgments, blob GC/publication reconciliation, secret scans, and acquisition (`polylogue/daemon/cli.py:3209-3231`).

## Current converger

`DaemonConverger` is an ordered generic stage runner. A stage supplies check/execute functions, optional batch/session functions, barriers, and `false_means_pending` semantics (`polylogue/daemon/convergence.py:59-87`; `polylogue/daemon/convergence.py:162-176`).

Current default order:

1. Optional Sinex publication when configured (`polylogue/daemon/convergence_stages.py:1111-1124`).
2. Raw parse recovery (`polylogue/daemon/convergence_stages.py:1125-1128`).
3. Raw-authority verdict cache (`polylogue/daemon/convergence_stages.py:1127-1129`).
4. FTS (`polylogue/daemon/convergence_stages.py:1129-1130`).
5. Embeddings (`polylogue/daemon/convergence_stages.py:1130-1131`).
6. Claude workflow derivation (`polylogue/daemon/convergence_stages.py:1131-1132`).
7. Insights (`polylogue/daemon/convergence_stages.py:1132-1133`).
8. Standing queries (`polylogue/daemon/convergence_stages.py:1133-1136`).

The watcher is constructed with this converger and the shared write coordinator (`polylogue/daemon/cli.py:3238-3243`; `polylogue/daemon/cli.py:3263-3272`).

## Execution semantics

- Stages execute in order and barrier failures prevent downstream execution for the affected subject (`polylogue/daemon/convergence.py:266-329`; `polylogue/daemon/convergence.py:351-495`).
- Batch execution rechecks subjects after a false result from a `false_means_pending` stage, so completed siblings can become done while remaining work stays pending (`polylogue/daemon/convergence.py:438-489`).
- Session-scoped convergence uses the same ordered stage and barrier model rather than a second convergence engine (`polylogue/daemon/convergence.py:497-600`).

## Quiet-window deferral

- Insight rebuilds can rehydrate an entire large session, so actively changing source sessions are removed from the current batch (`polylogue/daemon/convergence_stages.py:1450-1462`; `polylogue/daemon/convergence_stages.py:2363-2375`).
- If every selected session is hot, the stage returns false; if some are cool, those are rebuilt and the stage still returns false to preserve the remaining obligation (`polylogue/daemon/convergence_stages.py:538-557`; `polylogue/daemon/convergence_stages.py:619-653`).
- Hot-insight debt retries no earlier than 60 seconds and, when the source is available, no earlier than source mtime plus that delay (`polylogue/sources/live/convergence_debt_retry.py:12-18`; `polylogue/sources/live/convergence_debt_retry.py:21-50`).

## `convergence_debt`

- A pending stage is classified as deliberate deferral rather than failure; failed and deferred rows remain distinguishable (`polylogue/sources/live/convergence_debt.py:10-20`; `polylogue/sources/live/convergence_debt.py:52-80`).
- Debt is persisted in disposable `ops.db`, keyed by stage, target type, and target ID, with retry time and attempt state (`polylogue/storage/sqlite/archive_tiers/ops.py:153-170`).
- Successful passes clear stale stage debt; unresolved work is recorded against session IDs when available, otherwise source paths (`polylogue/sources/live/convergence_outcome.py:13-49`).
- The periodic retry owner executes only the recorded stage and subject. Legacy rows named `convergence` retain all-stage fallback behavior (`polylogue/daemon/cli.py:939-970`; `polylogue/daemon/cli.py:2247-2304`).

## Current runtime state

Runtime evidence outside Git, observed 2026-08-25:

- `systemctl --user is-enabled polylogued.service` returned `masked-runtime`.
- `systemctl --user is-active polylogued.service` returned `inactive`.
- Unit properties reported `LoadState=masked`, `UnitFileState=masked-runtime`, and runtime control fragment `/run/user/1000/systemd/user.control/polylogued.service`.

This means the ownership machinery described above is present but no live daemon currently holds it. The executable owner remains `run_daemon_services` (`polylogue/daemon/cli.py:2663-2700`).

## SELECT-HYBRID direction

Decision record outside Git: `polylogue-04r9f` is closed with SELECT-HYBRID selected. The target is one small recurring registry, per-key `VALID/MISSING/STALE/EXCESS`, `DONE/PENDING/FAILED`, domain-owned required/inspect/compute/publish adapters, process-local scheduling, compute outside the writer lease, and publish-time binding revalidation under `BEGIN IMMEDIATE`.

This direction has not landed in this HEAD. Current code still exposes the generic `ConvergenceStage` abstraction and the eight-stage default list (`polylogue/daemon/convergence.py:59-87`; `polylogue/daemon/convergence_stages.py:1097-1136`). The 04r9f experiment implementations were explicitly disposable, so their absence is intentional task state rather than a missing merge.

## Workload-probe honesty

Open pointer: `polylogue-uuf2g`.

- `_scalar_int` catches any `sqlite3.Error` and returns ordinary integer zero (`polylogue/operations/daemon_workload_probe.py:167-172`).
- Exact table counts label that result `"exact"`, making an SQL failure indistinguishable from a genuinely empty table (`polylogue/operations/daemon_workload_probe.py:203-212`).
- Exact readiness counts use the same helper directly (`polylogue/operations/daemon_workload_probe.py:291-294`).
- Consequently, workload and readiness reports can claim exact zero and derive false readiness after lock, I/O, or corruption errors. Treat exact zero as untrustworthy until uuf2g closes (`polylogue/operations/daemon_workload_probe.py:1378-1405`).

## Gotchas

- `ops.db` debt is retry bookkeeping, not durable semantic authority; losing it may repeat convergence work (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:70-75`; `polylogue/storage/sqlite/archive_tiers/ops.py:153-170`).
- Returning false can mean pending bounded work, not failure; callers must inspect typed stage state (`polylogue/daemon/convergence.py:120-142`; `polylogue/sources/live/convergence_debt.py:52-80`).
- Schema-blocked startup may keep health/lifecycle reporting alive while withholding archive mutation and convergence (`polylogue/daemon/cli.py:3055-3074`).
- Process-local serialization is not proof that arbitrary external code cannot open SQLite directly; exclusion and surface routing are separate controls (`polylogue/daemon/write_coordinator.py:1-6`; `polylogue/daemon/cli.py:2683-2699`).

## DISCREPANCIES

- `CLAUDE.md` says the daemon owns all writes and the main process is the sole SQLite writer. Code enforces that policy inside daemon routes, but writable `ArchiveStore` entry points remain directly callable outside the coordinator (`CLAUDE.md:101-109`; `polylogue/storage/sqlite/archive_tiers/archive.py:1034-1083`; `polylogue/daemon/write_coordinator.py:1-6`).
- `CLAUDE.md` compresses convergence to FTS, embeddings, and insights. The current default converger also includes raw parse recovery, raw-authority caching, Claude workflow, standing queries, and optional Sinex publication (`CLAUDE.md:103-107`; `polylogue/daemon/convergence_stages.py:1097-1136`).
- Operationally, the daemon is runtime-masked and inactive, so the documented live-owner posture is not the machine’s current state. This discrepancy is external runtime state, not represented in repository files.

verified: 4abb7a80bca2160d27fdc799891305cf02b680ff 2026-08-25
