# Implementation order

`ARCH-04` denotes `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/testdiet/context/testsuite_diet/architecture/04-destructive-and-authentication-boundaries.md`.

This sequence combines the six-tool cutover lane with bead `polylogue-t46.9`. It follows the architecture's migration order—bind executable specs, prove one destructive vertical slice, route reset/raw authority, then reversible/judgment writes, and remove direct calls only after bypass tests (`ARCH-04:116-126`). Safety hotfixes precede structural work.

## 0. Freeze the evidence baseline

**Owners:** `polylogue/mcp/declarations/`, `polylogue/operations/`, devtools policy/inventory.

**Work:** Check in a generated semantic mutation inventory keyed by stable operation id, current entry path, handler, affected tiers, and proposed six-tool home. Seed it with all 104 MCP declarations and every bypass row in `EVIDENCE.md`. Mark read-only handlers presently hidden behind write (`get_metadata`, seven `list_*` families) for immediate projection correction. Record the current source-role contradiction (`review`) rather than erasing it.

**Acceptance:** Every live MCP declaration maps exactly once; every known CLI/API/MCP/daemon/internal semantic mutation maps exactly once or has a typed exclusion with rationale. Inventory generation is deterministic at HEAD.

**Falsification:** Adding a new registered mutating handler or direct semantic storage call without an operation id fails CI and names the file/symbol. A hand-maintained allowlist that can authorize merely by containing a name does not count (`ARCH-04:38-40,104-110`).

## 1. Land immediate safety/correctness fixes

**Owners:** MCP mutation/personal-state/maintenance registrars; `storage/sqlite/archive_tiers/user_write.py`.

**Work:** Implement bead `polylogue-jn40` on the ten named MCP handlers as a temporary fail-closed gate. Independently fix bead `polylogue-41ow`: begin the terminal-status preserve/write decision under one immediate user-tier transaction using canonical busy-timeout/profile; preserve explicit competing-judgment conflict and exact-retry behavior.

**Acceptance:** All ten handlers reject omitted/false confirm before effect. The forced two-connection interleaving cannot revert an accepted/rejected candidate to candidate. Exact judgment retry remains idempotent; changed judgment conflicts.

**Falsification:** Removing any interim gate makes its focused MCP test perform/attempt the effect. Removing the immediate transaction or terminal-status preservation makes the real forced-interleaving test fail.

**Do not:** Represent these changes as completion of t46.9. Boolean confirm and storage atomicity are necessary but not the cross-surface authority boundary.

## 2. Make `OperationSpec` executable

**Owners:** `polylogue/operations/specs.py`, new executor/receipt modules, operation declaration consumers.

**Work:** Extend each spec with stable id and semantic version; input/result schemas; capability; reversibility/destructive class; resolver/disclosure; preview and confirmation strength/expiry; handler binding; affected tiers; idempotency/conflict policy; receipt/public projection; postflight/recovery/audit obligations. Build `OperationRequest`, immutable `PreviewReceipt`, `OperationReceipt`, typed failures, and one `OperationExecutor`. Derive action contracts, CLI help, MCP discovery, HTTP policy, examples, and inventory from the registry.

**Acceptance:** Executor enforces the full sequence `resolve → authorize → preview → confirm → apply → receipt → postflight`. No handler is callable through the public registry without all required declaration fields. Storage guards and write-effects hooks are invoked behind handlers.

**Falsification:** A spec missing capability/resolver/handler/receipt policy cannot register. Mutating a spec version changes confirmation validity. Directly calling a registered handler outside executor fails an architectural test or is inaccessible outside the executor module.

## 3. Implement confirmation, idempotency, and recovery ledgers

**Owners:** operations executor; ops-tier storage; auth/receiver context.

**Work:** Implement canonical preview hashing, opaque 256-bit token issuance, token-hash storage, binding to principal/capability fingerprint/archive/fileset/receiver/op/spec/request/target/effect/expiry/nonce/strength, single use, and apply-time re-resolution. Add durable request/idempotency and operation/checkpoint receipts. Define restart behavior: unused preview tokens may expire/invalidate; completed operation receipts survive and answer retries.

**Acceptance:** Altering any bound field fails before mutation. Target or effect drift returns `preview_stale`. Same idempotency key + same request returns the same receipt; changed request returns `idempotency_conflict`. A consumed token cannot start a second operation, but an exact network retry returns the prior receipt.

**Falsification:** Tests independently mutate actor, capability set, archive identity, receiver/fileset, op version, request input, target rows, effect plan, expiry, and nonce. Any mutation that still applies is a blocker.

## 4. Prove the first vertical slice: suppress, prune, excise

**Owners:** operations; `security/excision.py`; lifecycle worker; API; CLI reset/excise; MCP; daemon HTTP; storage/effects.

**Work:** Define three separate specs:

- `session.suppress` — R2 write, preserves evidence, before-image/inverse receipt.
- `session.prune` — D1 operate, deletes rebuildable projections only, CB.
- `session.excise` — D2 operate, cross-tier/blob/lineage destruction, CB, durable checkpoint/audit/non-resurrection postflight; classify archive-wide/broad scope as D3/CB+.

Route every current CLI/API/MCP/daemon/lifecycle entry through them. Make legacy `delete_session` a deprecated adapter to `session.prune` (or an explicitly documented composite suppress+prune if compatibility evidence requires it), never excision. Put `ArchiveWriteGateway`/effects and excision non-resurrection guards behind the handlers.

**Acceptance:** Every surface produces the same canonical target/effect digest and receipt for an equivalent request. Suppression leaves governed evidence. Prune leaves source/user evidence and carries rebuild obligations. Excision covers source/index/user/embedding/ops/blobs/lineage and proves non-resurrection. Crash/failure injection at every cross-tier checkpoint reconciles to completed or explicitly incomplete state, never false success.

**Falsification:** A direct call to `ArchiveStore.delete_sessions` or `apply_session_excision` from a surface makes the bypass test fail. Reintroducing the second excision write chokepoint bypass from bead `polylogue-layg` fails non-resurrection. Changing targets between preview/apply returns `preview_stale`.

## 5. Route reset, raw retention, and raw-authority repair

**Owners:** CLI reset; maintenance raw identity; live-ingest/raw-retention; daemon startup/repair; tier storage/migration.

**Work:** Split broad reset into `archive.reset.derived` D1 and `archive.reset.durable` D3. Put filesystem paths/tier identities/bytes/backup/rebuild readiness into PB previews and durable checkpoints. Route raw materialization as R1 and live raw-authority repair as D3 CB+. Route `repair_superseded_raw_snapshots`, `_compact_superseded_raw_snapshots`, and `cleanup_superseded_raw_snapshots` through one `maintenance.cleanup.durable_or_blob` D2 spec with human-bound CB; disable automatic live compaction until that route exists. Reclassification to R1/D1 requires proof that every deleted raw row/blob is byte-exactly reconstructable, with the reconstruction source/version/digest recorded in preview and receipt. Ensure daemon/system execution consumes an operator-authorized preview for D3.

**Acceptance:** No `unlink`/`rmtree`, durable raw-row/blob deletion, durable tier rewrite, or raw authority frontier actuation occurs outside executor. Live writer/quiescence, exact-reconstruction classification evidence, and backup requirements are declared and tested. Partial failures resume/reconcile from the operation journal.

**Falsification:** Change a path, tier generation, raw-retention authority set, source/version/digest, authority frontier, or archive identity after preview; apply must fail without touching storage. An automatic live cleanup that deletes durable raw state without a human-bound CB or proven reclassification is a blocker. SIGKILL at each checkpoint must not produce an unreceipted “success.”

## 6. Route reversible write families and remove over-confirmation safely

**Owners:** API user-state methods; MCP mutation/personal-state; daemon user-state HTTP; query CLI; user-tier storage.

**Work:** Route candidate capture, annotation batch, notes, tags, marks, annotations, views, recall packs, workspaces, metadata, corrections, suppression, and excision-request submission through specs. Add expected generation to overwrite/delete requests, exact before-images, inverse/restore operations, no-op receipts, and stable idempotency keys. Move the eight read-only current write tools to query/read projection.

**Acceptance:** Equivalent writes across CLI/API/MCP/daemon produce equivalent receipts. Duplicate adds/removes are explicit no-ops. Changed-generation overwrite/delete conflicts. Every R2 delete/clear has a tested inverse or restore handle.

**Falsification:** Removing before-image capture makes restore tests fail. Any direct user-tier mutation from a surface fails the bypass check. Only after these tests pass may jn40 booleans be removed from tag/mark/soft-delete handlers.

## 7. Route and consolidate judgment

**Owners:** operations; judgment API/storage; root `polylogue judge`; query verb compatibility; MCP review registrar.

**Work:** Define one `assertion.candidate.decide` spec requiring `assertion:judge`, PJ preview, explicit intent, expected generation, and RJ receipt. Preserve accept/reject/defer/supersede, explicit injection authorization, immutable lifecycle history, per-item SAVEPOINT outcomes, exact retry, and changed conflict. Consolidate root judge as canonical; retire the duplicate query mutation after parity.

**Acceptance:** Write profile cannot discover/apply judgment; review/admin can. Caller `actor_ref`, recipe content, prompt/resource text, and archive content cannot grant `assertion:judge` or injection authorization. Two-process competing decisions yield one valid serial result plus an explicit conflict/idempotent retry.

**Falsification:** A write-only MCP instance can invoke judge; an edited retry returns success; or a non-user candidate becomes injectable without explicit accepted judgment.

## 8. Implement `run` without delegated authority

**Owners:** operation registry/executor; saved-query/recipe storage; MCP run adapter.

**Work:** Make saved query execution a typed ref/version expansion. For recipes, freeze the expanded DAG, compute union capabilities and aggregate target/effect digest before any step, and issue one whole-plan preview/confirmation if any step is destructive. Persist step receipts/checkpoints and compensation policy.

**Acceptance:** Saved content cannot add a capability. No step executes before full authorization. A recipe change after preview is stale. Resume does not repeat completed destructive steps or widen target scope.

**Falsification:** Insert an unauthorized operation into a saved recipe or mutate its version after preview; execution must fail before the first effect.

## 9. Route operate families and internal system actors

**Owners:** maintenance planner/registry/replay; ingest pipeline; derived convergence; embeddings; FTS/lineage startup; blob GC; migration; backup/vacuum.

**Work:** Register and route all operate rows from `REPORT.md`. Replace direct maintenance preview/execute split with executor preview/apply. Give system actors narrow standing grants for bounded R1/selected D1 operations, with target/byte/time/tier limits and durable receipts. R1 uses C0; each selected D1 run still consumes a system-bound CB. D2 remains human CB; D3 remains human CB+.

**Acceptance:** MCP, daemon HTTP, CLI, startup, convergence, lifecycle, and ingest all create operation receipts. `ArchiveWriteGateway` is used as handler effect infrastructure rather than public authorization. Every direct internal mutation in the census is removed or typed as a storage-internal call reachable only from a registered handler.

**Falsification:** Adding a new startup/convergence direct writer fails CI. A system grant exceeding its declared target/count/byte/tier budget fails closed and emits an operator-preview requirement.

## 10. Cut MCP to exactly six tools

**Owners:** MCP declarations/adapter/server, compatibility telemetry, documentation.

**Work:** Expose only `query`, `read`, `write`, `judge`, `run`, `operate`. Fold current query/get/graph/explain/status forms into typed `query` inputs and current read/get/context/status projections into `read`; expose operation lifecycle through `operate`. Role-filter declarations/capabilities at discovery. Keep old names as compatibility aliases only outside the default surface until semantic/cold-model parity and telemetry gates pass.

**Acceptance:** Default read profile exposes only query/read. Write adds write/run. Review adds judge. Admin adds operate. No old per-operation tool appears in the default discovery response. Every old tool has a parity fixture for selection, ordering, totals/coverage, refs, errors, bounds, lifecycle, and authorization.

**Falsification:** A removed tool provides a capability not discoverable/executable through the six tools; a read role sees privileged schema; or a prompt/resource can trigger an effect.

## 11. Delete compatibility paths and close beads

**Owners:** all surface owners; devtools; release notes.

**Work:** Remove legacy handler registration, boolean confirmation adapters, duplicate root/query judgment paths, direct surface API/storage calls, and misleading `delete_session` terminology only after all previous acceptance gates. Update generated MCP equivalence, operation inventory, docs, and Beads.

**Acceptance:** Source review and generated inventory find no unclassified semantic mutation. Real-route bypass tests cover each adapter family. `polylogue-t46.9`, `polylogue-jn40`, `polylogue-41ow`, and the relevant t46.8 migration beads close with linked proof. `polylogue-a7xr.18` is reconciled by either full effect-gateway coverage behind executor or typed exemptions.

**Falsification:** Reverting any executor call or adding one direct storage/API mutation makes a test fail with the exact route. A green declaration lint without a real mutation-route failure is insufficient.

## Local implementer verification checklist

Before merging each operation family, verify the canonical spec id/version, resolver and disclosure, principal/capability source, destructive class, preview shape, confirmation level, affected tiers, idempotency/conflict behavior, receipt durability, postflight/recovery, all surface adapters, all internal actors, direct-call bans, stale-preview cases, retry/crash behavior, and negative discovery/injection tests. Run focused tests plus the repository's quick verification command; this report does not claim those tests have run.
