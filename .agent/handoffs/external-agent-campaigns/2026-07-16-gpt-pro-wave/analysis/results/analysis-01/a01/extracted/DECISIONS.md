# Decisions

Each record separates observed fact, source-supported inference, unresolved uncertainty, recommendation, preserved information, and falsification evidence. “Supersede” means the target record closes with a durable pointer and all unique acceptance/evidence is moved to the owner; it never means silently deleting context.

## D-01 — Status authority

**Observed fact.** `20d.17` explicitly supersedes active `703`; source has multiple rich status assemblers and only a whole-payload cache.

**Inference.** `703` is a structural symptom/earlier formulation of the stronger component-snapshot contract, not an independent product authority.

**Uncertainty.** The final home of the storage-free component protocol and exact component inventory are not implemented in source.

**Decision.** Retain `20d.17`; supersede/close `703`; execute one serialized status branch.

**Preserve.** 703's three-source inventory, one-assembly rule, fact parity across CLI/daemon/MCP/Python/web, and citation-drift fixture. Add 20d.17 freshness, deadline, last-good, source-fingerprint, component isolation, and resumable-detail semantics.

**Falsification.** A current source/acceptance requirement owned by 703 that cannot be represented as a component or projection of 20d.17.

## D-02 — Destructive mutation product model

**Observed fact.** `t46.9` mandates a universal executor; `kwsb.2` explicitly rejects one. Current source removed a previous speculative generic operation framework because it had only one consumer.

**Inference.** Keeping both open would create two authorization/dispatch authorities and invite direct-adapter bypasses.

**Uncertainty.** The first two concrete domain pilots and exact shared transaction API are not yet in current source.

**Decision.** Retain `kwsb.2`; supersede/close `t46.9`; use a shared `MutationTransaction` with domain-owned PlanSpec/actuator.

**Preserve.** Operation inventory, declaration-derived discovery/help/schemas, bound preview/confirmation token, actor/archive/version/target digest, conflict/idempotency policy, receipts/postflight, and storage defense in depth.

**Falsification.** Two production mutation domains demonstrate a single concrete executor that preserves domain planning/actuation without speculative unused kinds.

## D-03 — Daemon supervision versus event transport

**Observed fact.** P1 `avmq` hard-depends on P3 `yp0`; `avmq` calls EventBus adjacent transport. EventBus core has no production consumer; `run_daemon_services` owns tasks directly.

**Inference.** The hard edge is a false blocker and priority inversion. Supervision can be implemented and tested without event-driven wakeups.

**Uncertainty.** The final service registry may expose event subscriptions as optional metadata, but that does not make EventBus a prerequisite.

**Decision.** Replace hard edge with soft relation. Implement `avmq`, `09rn`, and `enj7` together; wire `yp0` afterward.

**Preserve.** Event vocabulary and failure isolation; slow reconciliation; service ownership/prerequisites/readiness/retry/shutdown/profiles/status identity.

**Falsification.** A production supervisor cannot account for or start any required service without a wired EventBus.

## D-04 — Raw-authority safety chain

**Observed fact.** Scale proof, fixed-point execution, live operator authorization, and final closure have different acceptance semantics and code/live boundaries. Source contains substantive typed planner/apply/recovery/proof machinery.

**Inference.** Merging the Beads would erase independent safety gates; missing dependency edges, not duplicate records, are the defect.

**Uncertainty.** Tracker notes' live archive cardinalities and backup state were not reproduced.

**Decision.** Add `hjpx → hjpx.2` and `yla8 → hjpx`; retain `lkrc → yla8`. Code cluster `hjpx.2/hjpx/lkrc`; separate live phase `yla8`.

**Preserve.** Immutable plan/evidence identity, complete census, finite retry/fairness, bounded resources, stopped daemon, current backup, quiescence, exact operator authorization, postflight receipt, judgment path.

**Falsification.** Equivalent accepted evidence already closes one prerequisite, or product scope is formally narrowed to remove that gate.

## D-05 — Query execution residual ownership

**Observed fact.** Shared execution control is implemented and production-routed at three surfaces, but many direct MCP/HTTP reads remain. The export regressed `z9gh.1` from a later in-progress record.

**Inference.** Reimplementing execution control would duplicate landed authority; remaining work divides into core fairness/scale and transaction/surface migration.

**Uncertainty.** The complete set of direct opens that are true query transactions versus harmless bounded metadata reads needs route-by-route confirmation during implementation.

**Decision.** Restore `z9gh.1` in progress; keep fairness/core cleanup there; hard-depend on `4s3c`; move direct surface migration, HTTP disconnect, and resumed receipt to `z9gh.9.1`.

**Preserve.** Dedicated read-only worker, progress interrupt, exact cleanup, shared admission, safe receipts, semantic no-refusal, advancing continuation, and generated adapter parity.

**Falsification.** A direct site is not a query transaction, or accepted live evidence already covers the delegated scale/surface requirement.

## D-06 — Declaration kernel and consumers

**Observed fact.** No declaration package exists. Multiple Beads claim a shared kernel relationship, while maintenance already has a catalog/dispatch split and marker/work-event APIs are absent.

**Inference.** Soft relations are insufficient; parallel registries will form unless consumers block on kernel identity/completeness.

**Uncertainty.** The exact package name and minimum declaration schema are not fixed by current source.

**Decision.** Add hard dependencies from `z9gh.3`, `9e5.31.1`, `71ey`, and `rii.1` to `o21.1`; relate `rii.1` to `37t.2.1`.

**Preserve.** Domain semantic ownership and separate lowering/actuation adapters. Share only declaration identity, ownership, completeness, schema/introspection, and event-kind vocabulary.

**Falsification.** `o21.1` scope explicitly excludes a consumer, or a newer accepted product decision authorizes a separate registry.

## D-07 — Schema workload execution

**Observed fact.** Root and three P1 children are all in progress on the same generated-schema hotspot; some journal work landed, while privacy and default-role questions remain.

**Inference.** Parallel branches risk artifact/provenance drift; merging children would lose distinct acceptance semantics.

**Uncertainty.** The feature branch versus current origin merge status may change after the snapshot.

**Decision.** Keep all children; one owner/branch; order `.1` replay/memory/cancellation/live receipt, `.2` privacy/promotion, `.3` catalog roles/default, then parent closure.

**Preserve.** All positive evidence families, privacy blocker/review distinction, deterministic output, cleanup, runtime resolution, default rationale, and promotion receipt.

**Falsification.** Current merged source proves the children touch disjoint artifacts and can pass independently without generated-output conflict.

## D-08 — Browser extension and campaign authority

**Observed fact.** Provider-neutral transport/receipt and extension worker exist; capture/UX residuals remain; no campaign IDs leak into extension/receiver.

**Inference.** Transport, capture truth, presentation policy, and multi-step orchestration are layered, not duplicates.

**Uncertainty.** Live provider/project/failure matrix and deployed parity were not available.

**Decision.** Keep `ptx`, `3v1`, `yyvg.7`, `yyvg.6`; soft-relate `yyvg.7` to `3v1`; serialize extension/receiver writes; keep campaign branch separate.

**Preserve.** Durable `outcome_unknown` handling, exact reconciliation receipt, gap evidence, exception-only attention, external private campaign ledger, and no extension-owned campaign state.

**Falsification.** A new accepted product decision moves campaign identity/ledger into the receiver, with corresponding source and migration plan.

## D-09 — Beads monotonic synchronization

**Observed fact.** Repository instructions describe stale branch reimport; two current rows are older than Git history; guard coverage is hook-local.

**Inference.** These are real regression fixtures for `gxjh.1`, not isolated bookkeeping mistakes.

**Uncertainty.** The live Dolt DB was not available, so the exact checkout sequence was reconstructed from Git/history and repository documentation.

**Decision.** Restore `z9gh.1` and `xnws`; add both to `gxjh.1`; keep `8jg9.1` blocked by it; apply all tracker changes in one immediately merged branch.

**Preserve.** Per-row latest status/assignee/parent/dependencies/close reason, transactionality, conflict refusal, complete receipt, and post-checkout re-verification.

**Falsification.** A later tracker revision or a different proven corruption mechanism explains either row.

## D-10 — Verification parentage

**Observed fact.** Active `b054.1.1` is parented to closed audit `b054.1`; `88jp` is the active verification-risk program.

**Inference.** The closed parent is historical provenance, not an executable owner.

**Uncertainty.** Another active verification program may be preferred by maintainers, but none is better supported in the snapshot.

**Decision.** Reparent to `88jp`; retain historical relation to `b054.1`.

**Preserve.** Bounded seed/resume, deterministic cleanup, zero unowned baseline quarantine, repeated witness SLO, and child proof lanes.

**Falsification.** `b054.1` is intentionally reopened as the implementation program or a better active parent is adopted.

## D-11 — Layered lookalikes retained

**Observed fact.** Several clusters share vocabulary or files but have different semantic authority: query algebra/transaction/adapters, rebuild exclusivity/raw authority/freshness, scenario catalog/live gate, inbound/outbound assets, and parent/implementation/proof families.

**Inference.** Collapsing them would reduce count while discarding safety or acceptance meaning.

**Decision.** Retain them as distinct. Use dependencies and branch serialization where they write the same hotspot.

**Falsification.** A source-validated owner demonstrably covers every distinct acceptance criterion and preserves all evidence/authority boundaries.
