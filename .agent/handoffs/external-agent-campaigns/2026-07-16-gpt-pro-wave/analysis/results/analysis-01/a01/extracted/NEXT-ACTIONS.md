# Next actions

These actions are proposals only. This audit did not run `bd update`, `bd close`, `bd dep`, `bd export`, an archive apply, or any production mutation.

## Phase 0 — Repair tracker truth before implementation

Use one bookkeeping branch, created from the newest merged default branch, and merge it before any unrelated checkout.

1. Restore `z9gh.1` to the later per-ID state: `in_progress`, assignee `Sinity`, updated record content from `6610f3b0`. Do not close it from the misleading commit subject.
2. Restore `xnws` closed with the later close reason from `6610f3b0`/PR #2963.
3. Record `20d.17` as the sole status owner and close `703` as superseded after copying its unique evidence/acceptance.
4. Record `kwsb.2` as the destructive-mutation transaction owner and close `t46.9` as superseded after copying its unique inventory/discovery/guard requirements.
5. Replace hard `avmq → yp0` with a soft relation.
6. Add the raw chain edges, declaration-kernel edges, `yyvg.7 → 3v1` soft relation, and reparent `b054.1.1` to `88jp`.
7. Normalize the schema-workload root/child claim notes so one branch and explicit order are visible.
8. Add `z9gh.1` and `xnws` as `gxjh.1` regression cases.

After every checkout/merge/worktree operation, re-read every changed row from the live tracker. Parse every JSONL line, inspect per-ID `updated_at`, run repository graph lint, and confirm there is exactly one canonical parent per issue. Do not use a temporary-worktree `bd export` to resolve conflicts.

**Phase-0 acceptance:** the effective active count is 554; `703`, `t46.9`, and `xnws` are not executable leaves; `z9gh.1` is claimed; no active child has a closed parent; no P0/P1 hard dependency points to P3/P4 solely for adjacent transport; graph remains acyclic.

## Phase 1 — Land shared kernels and remove false parallelism

### Status (`20d.17`)

Inventory every shared fact in daemon status, CLI direct fallback, workload diagnostics, MCP/HTTP/web, and coordination. Define component identity, dependencies, detail/cost class, deadline, source fingerprint/event invalidation, privacy, freshness/staleness, last-good evidence, and projection fields. Convert request paths to snapshot reads before deleting old probes.

Verify a deliberately stalled raw/debt/embedding/Beads/archive/handoff component cannot delay healthy components. Record cold/warm component timing, p50/p95, output bytes, omissions, fingerprints, cache decisions, and exact detail refs. Keep 703's parity and citation-drift fixtures.

### Daemon supervision (`avmq`, `09rn`, `enj7`)

Generate an inventory test from every `asyncio.create_task`/server task in `run_daemon_services`. Introduce one service registry and supervisor. Prove deterministic ordering, no duplicate starts, declared failure policy, partial-start rollback, bounded cancel/await, orphan diagnostics, missing-tier behavior, and production-derived named test profiles. Do not wait for EventBus wiring.

### Declaration kernel (`o21.1` first)

Land the smallest storage-free identity/ownership/completeness/introspection kernel. Then adapt query declarations (`z9gh.3`), MCP projection (`t46.8.1`), definition closure (`9e5.31.1`), maintenance (`71ey`), and marker/work-event vocabulary (`rii.1`/`37t.2.1`) in separate domain commits/branches. A completeness test must fail any unowned declaration or any second registry for the same identity.

### Destructive transactions (`kwsb.2`)

Choose at least two real mutation domains before generalizing. For each, define a domain PlanSpec/target resolver/actuator and route all surfaces through a shared transaction that binds actor, archive identity, operation/version, target-set/preview digest, expiry, conflict policy, receipt, and postflight. Verify direct adapter calls fail the inventory/parity test. Keep storage guards behind the transaction.

## Phase 2 — Close code/proof clusters in dependency order

### Raw authority

Run the generated/sanitized scale proof first. It must exercise real planner/executor cardinality, finite retry injection, no starvation, monotonic executable backlog decrease, complete per-plan outcomes, bounded RSS/PSS/swap/temp/CPU/I/O, responsive health, and two quiescent fixed-point census identities. Only then close `hjpx`; only after code review and a current verified backup may an operator consider `yla8` live execution. `lkrc` closes after postflight authority/reconciler evidence.

A local implementer must verify no cursor/head regression, no non-executable plan crosses apply, preview membership is revalidated, crash-left applications recover from typed evidence, and complete frontier counts include `proven_current` while blocking derives from residual state.

### Query execution and surfaces

For `z9gh.1`, prove cross-class aging/fairness and exact cleanup under cancel-before-admission, during SQLite, during Python-side production, deadline, disconnect, and worker failure. `4s3c` must supply live steady-state RSS/PSS/swap/temp evidence.

For `z9gh.9.1`/`t46.8.2`, classify every remaining direct MCP/HTTP/archive open. Migrate true query transactions to shared execution/paging/resume. Prove HTTP disconnect, advancing continuation, resumed receipt, no raw expression leakage, generated MCP schema/help/role parity, and duplicate-alias removal. Do not convert writer/mutation routes into read transactions.

### Schema workload

Use one branch and the sequence recorded in `EXECUTION-CLUSTERS.csv`. `.1` must finish bounded journal/replay, cancellation/process-death cleanup, live receipt, and memory/I/O proof. `.2` must block unsafe property names/credentials while separately reporting review-only values and keeping local provenance out of public artifacts. `.3` must retain all evidence families while separating latest/recommended/default/promoted semantics with machine-readable rationale. Parent closure requires regenerated artifacts, runtime-resolution parity, promotion receipt, and canaries.

### Extension/capture/UX

Finish transport/provider/project/failure matrix under `ptx`, capture gap/status truth under `3v1`, and exception-only UX convergence under `yyvg.7`. Verify expired submit intent becomes `outcome_unknown`, no implicit retry occurs, reconciliation requires exact provider evidence, and deployed receiver/extension status agrees. Keep `yyvg.6` external and prove no campaign identity appears in extension/receiver storage or APIs.

## Phase 3 — Wire downstream consumers

Wire `yp0` only after the daemon service registry is stable. Register committed-domain publishers and subscribers through the lifecycle metadata, but retain slow reconciliation. Measure reduced idle polling/SQLite reads and prove missed event/subscriber failure does not lose correctness.

Wire definition closure, maintenance target actuation, and marker/work-event adapters only after `o21.1` identity/completeness is stable. For maintenance, eliminate the catalog/private-dispatch mismatch or make the declaration own both advertised capability and actuator binding. For markers/work events, lower into existing typed services/session-event channels; do not create marker-specific durable tables.

## Phase 4 — Re-audit after state changes

Regenerate the package from a fresh archive after graph repair and merged implementation. Compare:

- active/effective counts and per-ID latest history;
- hard-edge cycles, priority inversions, active supersessions, and closed parents;
- source symbols versus stale descriptions;
- branch/write-hotspot ownership;
- accepted live receipts for raw, query, status, schema, and extension;
- whether any consumer created a parallel registry/protocol despite the new edges.

Another iteration is low-value against the unchanged snapshot. It becomes high-value after Phase 0 or when fresh live receipts/current merged source are available, because it can test whether these decisions survived implementation rather than repeat the same static evidence.
