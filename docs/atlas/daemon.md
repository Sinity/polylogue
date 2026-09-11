# Daemon

## Area boundary

The daemon is the intended live write owner. It holds process-lifetime rebuild exclusion, serializes admitted SQLite mutations, drives source watching and derived convergence, and exposes HTTP/UDS readers (`polylogue/daemon/cli.py:2542-2600`; `polylogue/daemon/write_coordinator.py:1-6`).

## Runtime ownership

- `run_daemon_services` holds archive writer/rebuild exclusion for the process lifetime because startup recovery, acquisition, and periodic convergence all mutate storage (`polylogue/daemon/cli.py:2542-2585`).
- `DaemonWriteCoordinator` serializes write effects inside the daemon process; admitted work retains authority until it actually finishes, including after caller cancellation (`polylogue/daemon/write_coordinator.py:280-340`; `polylogue/daemon/write_coordinator.py:241-279`).
- HTTP and UDS mutation routes receive a bridge to that coordinator rather than opening an independent writer path (`polylogue/daemon/cli.py:3042-3085`; `polylogue/daemon/write_coordinator.py:665-729`).
- Startup recovers embedding lifecycle before publishing API sockets, then performs FTS and lineage readiness and blob-publication reconciliation before catch-up (`polylogue/daemon/cli.py:3042-3122`).
- Periodic owners cover raw materialization, insights, convergence debt, FTS, WAL, embeddings, status, judgments, blob GC/publication reconciliation, secret scans, and acquisition (`polylogue/daemon/cli.py:3138-3174`).
- Archive-bound FTS, embeddings, and readiness writers receive the selected archive root even when the active index generation is stored elsewhere, so a generation path cannot borrow another archive's admission (`polylogue/daemon/convergence_stages.py:94-111`; `polylogue/daemon/cli.py:316-325`; `polylogue/storage/sqlite/wal_checkpoint.py:153-187`). The typed session-profile factory likewise resolves the active generation at each read/write boundary (`polylogue/daemon/convergence.py:145-223`), but the daemon composition has not yet registered that owner with the watcher.

## Current converger

`DaemonConverger` remains the ordered generic stage runner. A stage supplies check/execute functions, optional batch/session functions, barriers, and `false_means_pending` semantics (`polylogue/daemon/convergence.py:280-313`; `polylogue/daemon/convergence.py:352-374`). It also owns a typed derivation registry: `converge_derivations` recomputes domain obligations from each adapter's output inspection rather than from stage state (`polylogue/daemon/convergence.py:388-446`).

Current default order:

1. Optional Sinex publication when configured.
2. Raw parse recovery.
3. Raw-authority verdict cache.
4. FTS.
5. Embeddings.
6. Claude workflow derivation.
7. Delegation work evidence.
8. FTS readiness and standing queries (`polylogue/daemon/convergence_stages.py:1016-1059`).

There is no generic derived-tables stage in this list. Session profiles are a separate typed domain, driven by `SessionProfileConvergenceOwner` when an integration supplies the adapter, compute capacity, and write bridge (`polylogue/daemon/convergence.py:62-143`). The current daemon CLI constructs `LiveWatcher` without its optional `session_profile_callback`, so this owner is a tested composition seam rather than a claim of live CLI adoption (`polylogue/daemon/cli.py:3269-3283`; `polylogue/sources/live/watcher.py:406-438`).

## Execution semantics

- Stages execute in order and barrier failures prevent downstream execution for the affected subject (`polylogue/daemon/convergence.py:475-538`; `polylogue/daemon/convergence.py:540-603`).
- Batch execution rechecks subjects after a false result from a `false_means_pending` stage, so completed siblings can become done while remaining work stays pending (`polylogue/daemon/convergence.py:470-530`).
- Session-scoped stage convergence uses the same ordered stage and barrier model rather than a second stage engine (`polylogue/daemon/convergence.py:796-900`). Typed session-profile convergence instead pages required keys, inspects output validity, computes outside the writer lease, and admits each publication through the bridge (`polylogue/daemon/convergence.py:62-143`; `polylogue/daemon/derivation.py:1-180`).

## Quiet-window deferral

- Session-profile quietness is now a domain-owner policy: the factory injects a clock and checks each candidate source key, while `SessionProfileDerivation.quiet` lets the kernel retain that key as `PENDING` (`polylogue/daemon/convergence.py:145-223`; `polylogue/storage/derived/session/derivation.py:494-583`; `polylogue/daemon/derivation.py:181-360`).
- The retired generic derived-table stage no longer returns `false` or owns a retry row. The optional embeddings stage retains its own hot-source deferral and ordinary bounded retry behavior (`polylogue/daemon/convergence_stages.py:249-252`; `polylogue/sources/live/convergence_debt_retry.py:12-30`).

## `convergence_debt`

- A pending stage is classified as deliberate deferral rather than failure; failed and deferred rows remain distinguishable (`polylogue/sources/live/convergence_debt.py:10-20`; `polylogue/sources/live/convergence_debt.py:52-80`).
- Debt is persisted in disposable `ops.db`, keyed by stage, target type, and target ID, with retry time and attempt state (`polylogue/storage/sqlite/archive_tiers/ops.py:160-190`).
- Successful passes clear stale stage debt; unresolved work is recorded against session IDs when available, otherwise source paths (`polylogue/sources/live/convergence_outcome.py:13-49`).
- The periodic retry owner executes only the recorded stage and subject. Legacy rows named `convergence` retain all-stage fallback behavior (`polylogue/daemon/cli.py:977-1050`; `polylogue/daemon/cli.py:2247-2304`).
- Typed session-profile derivation reports are not yet part of this stage-debt retry route: the retry owner rebuilds its stage map from `make_default_convergence_stages`, which intentionally has no profile stage (`polylogue/daemon/cli.py:2190-2243`; `polylogue/daemon/convergence_stages.py:1016-1059`). Retry/composition integration remains pending.

## Current runtime state

Runtime evidence outside Git, observed 2026-08-25:

- `systemctl --user is-enabled polylogued.service` returned `masked-runtime`.
- `systemctl --user is-active polylogued.service` returned `inactive`.
- Unit properties reported `LoadState=masked`, `UnitFileState=masked-runtime`, and runtime control fragment `/run/user/1000/systemd/user.control/polylogued.service`.

This means the ownership machinery described above is present but no live daemon currently holds it. The executable owner remains `run_daemon_services` (`polylogue/daemon/cli.py:2663-2700`).

## SELECT-HYBRID direction

Decision record outside Git: `polylogue-04r9f` is closed with SELECT-HYBRID selected. The target is one small recurring registry, per-key `VALID/MISSING/STALE/EXCESS`, `DONE/PENDING/FAILED`, domain-owned required/inspect/compute/publish adapters, process-local scheduling, compute outside the writer lease, and publish-time binding revalidation under `BEGIN IMMEDIATE`.

The typed registry seam has landed for session profiles: `DaemonConverger` accepts registered derivations, and `SessionProfileDerivation` is the single domain adapter whose inspection delegates to the shared classifier (`polylogue/daemon/convergence.py:388-446`; `polylogue/storage/derived/session/derivation.py:180-252`; `polylogue/storage/derived/session/derivation.py:494-505`). The generic default stage list remains for non-profile work and no longer includes a derived-table stage (`polylogue/daemon/convergence_stages.py:1016-1059`).

The profile owner is not yet wired into `run_daemon_services` or `run_live_watcher`: the watcher callback exists and invokes an explicitly supplied owner after ingest, but the current CLI passes only the embedding owner (`polylogue/sources/live/watcher.py:359-362`; `polylogue/sources/live/watcher.py:2081-2103`; `polylogue/daemon/cli.py:3274-3283`). This is a composition gap, not evidence that the typed domain is absent; it also means this atlas does not claim live adoption.

These properties hold for embeddings and the explicitly composed session-profile owner. `embed_archive_session_sync` is split into admitted reservation, lease-free provider computation, and admitted publication that re-acquires a fresh generation binding and revalidates the reserved attempt (`polylogue/storage/embeddings/materialization.py`). The `embed` stage refuses to call a provider while the writer gate is held and defers to convergence debt instead (`polylogue/daemon/convergence_stages.py`), and its production owners run through `polylogue/daemon/embedding_owner.py` on shared bounded compute capacity. Session profiles likewise compute outside the lease and publish through `DaemonWriteThreadBridge` when the owner is composed (`polylogue/daemon/convergence.py:62-143`). Profile callback registration and retry integration remain pending, so no live-adoption claim follows from the seam.

## Workload-probe honesty

- `_scalar_int` returns `Evidence[int]`, so a failed read is `Unavailable` and cannot be spelled as zero (`polylogue/operations/daemon_workload_probe.py:169-182`).
- A cheap table count that did not answer is labelled `"unavailable"` with `UNKNOWN_TABLE_COUNT`, never `"exact"` (`polylogue/operations/daemon_workload_probe.py:215-232`).
- A readiness count that did not answer refuses the whole derived-readiness block, which reports `checked: false` plus the sqlite reason instead of comparing zero to zero and claiming ready (`polylogue/operations/daemon_workload_probe.py:1471-1500`).
- Diagnostic attempt counts report `null` for an unanswered read, distinct from a measured `0` (`polylogue/operations/daemon_workload_probe.py:588-640`).

## Gotchas

- `ops.db` debt is retry bookkeeping, not durable semantic authority; losing it may repeat convergence work (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:70-75`; `polylogue/storage/sqlite/archive_tiers/ops.py:153-170`).
- Returning false can mean pending bounded stage work, not failure; stage callers inspect `StageState`, while typed derivations expose `DerivationReport.pending` and `.failed` (`polylogue/daemon/convergence.py:352-374`; `polylogue/daemon/derivation.py:300-360`; `polylogue/sources/live/convergence_debt.py:52-80`). A profile is not certified by generic stage state.
- Schema-blocked startup may keep health/lifecycle reporting alive while withholding archive mutation and convergence (`polylogue/daemon/cli.py:3055-3074`).
- Process-local serialization is not proof that arbitrary external code cannot open SQLite directly; exclusion and surface routing are separate controls (`polylogue/daemon/write_coordinator.py:1-6`; `polylogue/daemon/cli.py:2683-2699`).

## DISCREPANCIES

- The daemon contract says it owns all writes and the main process is the sole SQLite writer. Code enforces that policy inside daemon routes, but writable `ArchiveStore` entry points remain directly callable outside the coordinator (`polylogue/storage/sqlite/archive_tiers/archive.py:1034-1083`; `polylogue/daemon/write_coordinator.py:1-6`).
- The repository contract compresses convergence to FTS, embeddings, and insights. The current default converger also includes raw parse recovery, raw-authority caching, Claude workflow, delegation evidence, FTS readiness, standing queries, and optional Sinex publication; session-profile publication is a separate typed derivation and the generic list has no derived stage (`polylogue/daemon/convergence_stages.py:1016-1059`; `polylogue/daemon/convergence.py:388-446`).
- The typed session-profile owner and watcher callback are available, but the daemon CLI does not currently pass that callback and convergence-debt retry remains stage-only (`polylogue/daemon/cli.py:2190-2243`; `polylogue/daemon/cli.py:3274-3283`; `polylogue/sources/live/watcher.py:406-438`). Production composition is therefore pending; this sheet records the seam without claiming live adoption.
- Operationally, the daemon is runtime-masked and inactive, so the documented live-owner posture is not the machine’s current state. This discrepancy is external runtime state, not represented in repository files.

verified: f6df6366a 2026-09-11
