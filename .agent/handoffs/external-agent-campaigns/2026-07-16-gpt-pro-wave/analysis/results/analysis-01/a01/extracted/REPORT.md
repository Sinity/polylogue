# Bead overlap and contradiction audit

**Job:** `analysis-01`  
**Snapshot time:** `2026-07-17T04:32:02Z`  
**Archived source:** local `master` `f654480cad`; every archived tracked file that was present matched that commit, with one untracked `browser-extension/package-lock.json`.  
**Later bundled committed source:** `origin/master` `0d081e5bc0`, five commits ahead, used where its raw-authority/schema changes supersede the archived source.  
**Mutation policy:** no Bead, source, daemon, archive, or deployment state was mutated.

## Executive conclusion

The active portfolio is large but not structurally chaotic. The snapshot contains **555 open/in-progress rows** across **976 total Beads**; history reconstruction removes one false active (`polylogue-xnws`) and restores one regressed claim (`polylogue-z9gh.1`), leaving **554 effective active rows**. The hard dependency graph has **235 active-to-active hard edges and no cycle**. The highest-value repair is therefore targeted adjudication, not a wholesale re-model.

Five changes materially improve execution safety:

1. Close `polylogue-703` into `polylogue-20d.17`; it is the only explicit active-to-active supersession and its parity evidence is already claimed by the stronger component-snapshot owner.
2. Close `polylogue-t46.9` into `polylogue-kwsb.2`; the two encode incompatible destructive-operation products, and current source explicitly warns against reviving a speculative universal operation framework.
3. Remove the hard `polylogue-avmq → polylogue-yp0` blocker. Supervision can land without production event-bus wiring; the bus should consume the service registry downstream.
4. Make the raw-authority completion order explicit: `hjpx.2` scale proof before `hjpx`, then `yla8` authorized live gate, then `lkrc` closure. These are distinct safety authorities and should not be merged.
5. Make `o21.1` a real prerequisite for query declarations, definition closure, maintenance declarations, and work-event/marker vocabulary. They keep domain ownership but must not create parallel declaration registries.

The report proposes **19 graph/record changes** and **16 execution clusters**. The full 555-row snapshot-active census is in `ACTIVE-CENSUS.csv`; every row records source area, intended authority class/claim, hard and soft dependencies, acceptance semantics, explicit source paths, staleness signal, hotspot, and disposition.

## What was audited

The review parsed every current Bead row and every dependency edge, reconstructed per-ID tracker history from all bundled Git refs, inspected repository workflow instructions, compared the archived source with local and later bundled master refs, and read current implementations at the active P0/P1 write hotspots. It also used the separate test-suite package only as a stale cross-check.

The authority order applied was:

1. archived source at `f654480cad` for what the snapshot actually contained;
2. later committed `origin/master` `0d081e5bc0` where those five commits supersede local source;
3. repository instructions, especially the documented Beads branch-reimport hazard;
4. latest per-ID Bead record reconstructed from Git history;
5. older notes/plans and the slightly stale test-suite package.

This distinction matters. The snapshot overview calls local `origin/master` `f654480cad`, but the all-refs bundle also contains a newer `refs/snapshot/origin/master` `0d081e5bc0`. The latter adds five raw-authority/schema commits (`#2965`, `#2966`, `#2967`, `#2968`, `#2970`; 22 files, +1058/−117). The report never treats live values in notes as independently reproduced evidence.

## Portfolio and graph diagnostics

| Signal | Observed |
| --- | --- |
| Total Beads | 976 |
| Snapshot open/in-progress | 555 |
| Effective open/in-progress after history repair | 554 |
| Snapshot P0/P1 active | 114 |
| Active hard edges | 235 |
| Active hard cycles | none |
| Active→active supersedes | `20d.17 → 703` only |
| Active children with closed parent | `b054.1.1 → b054.1` only |
| Exact duplicate active titles | none |
| P0/P1 hard dependency on P3/P4 | `avmq → yp0` only |

The absence of cycles and exact duplicate titles is not proof that the portfolio is clean. The real defects are semantic: incompatible authority claims, stale descriptions after code landed, false hard blockers, and missing prerequisite edges around shared registries and live gates.

## Decision-ready graph changes

| ID | Priority | Change | Why |
| --- | --- | --- | --- |
| G01 | P0 | polylogue-z9gh.1 status in_progress | Export row updated 2026-07-16T12:56:52Z is older than per-ID Git history row updated 2026-07-17T00:25:42Z at 6610f3b0; later row is in_progress and assigned Sinity. |
| G02 | P1 | polylogue-xnws status closed | Export row updated 2026-07-16T23:43:40Z is older than Git history row updated/closed 2026-07-16T23:59:23Z at 6610f3b0; close reason cites merged PR #2963. |
| G03 | P1 | polylogue-20d.17 supersedes polylogue-703 | 20d.17 has the only active→active supersedes edge and its note explicitly says it absorbs 703; 20d.17 defines the stronger component-snapshot protocol. |
| G04 | P1 | polylogue-kwsb.2 supersedes polylogue-t46.9 | kwsb.2 requires a shared transaction protocol and explicitly rejects a universal mutation executor; t46.9 requires a universal OperationExecutor. Current operation_contract.py documents removal of the previous speculati… |
| G05 | P1 | polylogue-avmq blocks→relates-to polylogue-yp0 | avmq calls EventBus adjacent transport, not lifecycle authority. yp0 core exists but has no production references outside its module. The edge makes P1 supervision wait on P3 transport wiring. |
| G06 | P0 | polylogue-hjpx blocks polylogue-hjpx.2 | hjpx acceptance requires the July-15-shaped bounded convergence proof; hjpx.2 is that proof and says hjpx AC6 remains unproven. |
| G07 | P0 | polylogue-yla8 blocks polylogue-hjpx | yla8 notes retain the live authorization gate only after the successor fixed-point proof; hjpx says it is the executable successor to the failed live gate. |
| G08 | P0 | polylogue-lkrc blocks polylogue-yla8 | Existing edge correctly keeps final authority/reconciler closure behind the authorized live gate. |
| G09 | P0 | polylogue-z9gh.1 blocks polylogue-4s3c | z9gh.1 residual AC requires live RSS/PSS/swap steady-state proof; the later z9gh.1 row delegates this envelope to 4s3c. |
| G10 | P0 | polylogue-z9gh.1 consumer-ownership polylogue-z9gh.9.1 | Shared execution_control.py is implemented and used by one API, MCP, and HTTP query path, while many direct MCP/HTTP reads remain. z9gh.9.1 owns transaction/paging/disconnect/resume semantics. |
| G11 | P1 | polylogue-z9gh.3 blocks polylogue-o21.1 | z9gh.3 needs a query declaration registry; o21.1 is the shared storage-free declaration kernel and no declarations package exists yet. |
| G12 | P1 | polylogue-9e5.31.1 blocks polylogue-o21.1 | 9e5.31.1 notes say DefinitionClosureGraph waits for and consumes o21.1, but the graph has no hard edge. |
| G13 | P1 | polylogue-71ey blocks polylogue-o21.1 | Maintenance has both MaintenanceTargetSpec and a separate private _REPLAY_DISPATCH; 71ey is the pilot to unify them through the shared kernel. |
| G14 | P2 | polylogue-rii.1 blocks+relates-to polylogue-o21.1;polylogue-37t.2.1 | rii.1 says marker kinds and work-event kinds are one channel with two encodings. 37t.2.1 already depends on o21.1; source has session_events but no record_work_event/emit_decision API. |
| G15 | P0 | polylogue-yyvg.7 relates-to polylogue-3v1 | UX exception/attention policy consumes capture truth and gap evidence from 3v1 but does not own capture authority. |
| G16 | P1 | polylogue-b054.1.1 parent-child polylogue-88jp | b054.1 is closed audit work, while b054.1.1 is an active verification harness/proof owner. It is the only active child with a closed parent; 88jp is the active verification-risk program. |
| G17 | P1 | polylogue-1xc.14.1 execution-order 1xc.14.1.1→1xc.14.1.2→1xc.14.1.3→parent closure | Root and all three children are simultaneously in_progress on the same source hotspot. Canonical origin includes journal replay fixes, but privacy/promotion and catalog-default semantics remain distinct. |
| G18 | P1 | polylogue-gxjh.1 validates polylogue-z9gh.1;polylogue-xnws | Both rows demonstrate ordinary branch reimport restoring an older valid row. CLAUDE.md documents this exact hazard; current guard compares updated_at only around hooks. |
| G19 | P1 | polylogue-ptx;polylogue-3v1;polylogue-yyvg.7 description current-source-residuals | BrowserActionIntent/Receipt, receiver authority, outcome_unknown, and extension worker are present; remaining live-provider/project/failure and convergence evidence is not fully proved. |

`GRAPH-CHANGES.csv` carries the exact evidence, information to preserve, falsification condition, and application phase for each row.

## High-priority adjudications

### Status: one owner, one absorbed structural task

**Observed fact.** `polylogue-20d.17` has an explicit `supersedes` edge to still-open `polylogue-703`, and its note says it absorbs 703. Source still contains a whole-payload `StatusSnapshot`, a large `build_daemon_status`, independent CLI direct fallback assembly, and a separate coordination envelope. There is no `StatusComponentSpec`.

**Decision.** Keep `20d.17` as the canonical component-snapshot owner and close `703` as superseded. Move, do not discard, 703's inventory of shared facts, one-assembly invariant, cross-surface parity suite, and citation-drift fixture. Implement in one serialized status branch because the files are large and overlapping.

**Acceptance boundary.** A stalled component cannot delay healthy components; every component reports freshness, age, deadline, last-good evidence, state, and detail ref; compact projections stay within byte/latency budgets; exact diagnostics are explicit, bounded, cancellable, and resumable. A local implementer must prove CLI, daemon HTTP, MCP, Python, web, and coordination consume the same snapshot semantics without rebuilding rich status inline.

### Destructive mutation: adjudicate the product contradiction

**Observed fact.** `t46.9` says `OperationSpec` becomes the single executable declaration and every mutation route uses a universal `OperationExecutor`. `kwsb.2` says the missing abstraction is a shared transaction protocol, **not** a universal mutation executor, with domain-owned PlanSpec/actuator semantics. Current `OperationSpec` is metadata, while `operation_contract.py` records that an earlier generic request/ack framework was collapsed because only IMPORT used it.

**Decision.** Keep `kwsb.2`; supersede `t46.9`. The shared authority is `MutationTransaction`: resolve target, authorize, bind preview/token/digest, invoke a domain-owned actuator, receipt, and postflight. `OperationSpec` remains an inventory/discovery source and maps to transaction policy, but it is not the universal authorization/dispatch object.

**Preserved from `t46.9`.** Complete operation inventory; declaration-derived schemas/help/role discovery; removal of direct adapter mutation calls after parity; retirement of legacy confirm booleans after bound-token coverage; storage guards and `ArchiveWriteGateway` as defense in depth.

### Daemon lifecycle: remove a priority inversion, retain separate authorities

**Observed fact.** `avmq` is P1 and hard-blocked by P3 `yp0`, even though `avmq` explicitly calls EventBus adjacent transport. `run_daemon_services` remains a monolithic composition root spawning HTTP, API, watcher, and periodic tasks. `event_bus.py` contains a tested typed core but no production references outside that module.

**Decision.** Remove the hard edge and add a soft relation. Implement `avmq`, `09rn`, and `enj7` together: one `DaemonServiceSpec` registry, one supervisor, production-derived named profiles, bounded shutdown, and orphan diagnostics. Wire `yp0` afterward as event transport consuming the lifecycle registry; retain slow reconciliation as correctness truth.

### Raw authority: add the missing safety chain, do not merge it

**Observed fact.** Current origin source has a typed frontier census, executable actuator classification, preview-bound apply with revalidation and non-executable refusal, crash recovery, and a real scale proof runner that calls `repair_raw_materialization` and requires a census receipt. The active Beads still divide four different responsibilities: scale evidence (`hjpx.2`), fixed-point executor (`hjpx`), operator-authorized live gate (`yla8`), and final authority/reconciler closure (`lkrc`).

**Decision.** Keep all four. Add `hjpx → hjpx.2` and `yla8 → hjpx`; retain `lkrc → yla8`. Execute as:

`hjpx.2 scale proof → hjpx fixed-point acceptance → yla8 stopped-daemon authorized live gate → lkrc closure`.

The code work for `hjpx.2`, `hjpx`, and `lkrc` belongs on one serialized raw-authority branch because they touch the same planner/census/replay files. `yla8` remains a separate no-code live execution phase after review and explicit authorization. No live apply, cursor reset, evidence deletion, or manual SQL repair is authorized by this report.

### Query execution: shipped core, residual ownership split

**Observed fact.** PR #2964's shared `QueryExecutionContext`, weighted `QueryAdmissionController`, dedicated `InterruptibleSQLiteRead`, async cancellation/connection interrupt, and sync HTTP runner exist and are production-used by one API, MCP, and HTTP query path. Many direct MCP/HTTP `ArchiveStore.open_existing` routes remain. The exported `z9gh.1` row is also older than a later in-progress/assigned record.

**Decision.** Restore `z9gh.1` to `in_progress`. Keep cross-class fairness and execution-core cleanup in `z9gh.1`; add hard dependency on `4s3c` for live RSS/PSS/swap proof. Move remaining surface migration, HTTP disconnect, and resumed-delivery receipt scope to `z9gh.9.1`, which already owns transaction/paging/resume semantics. Keep `t46.8.2` as the generated MCP adapter migration and `t46.8.2.1` as its duplicate-alias regression, in the same serialized surface branch after declarations.

### Declaration family: shared kernel, domain-owned semantics

**Observed fact.** There is no `polylogue/declarations` package. Maintenance already has a public `MaintenanceTargetSpec` catalog plus a separate private `_REPLAY_DISPATCH`; `superseded_raw_snapshots` is advertised but absent from that dispatch. Existing `session_events` storage/read paths exist, while `record_work_event` and `emit_decision` do not. Several Beads say they consume or wait for `o21.1` but lack hard edges.

**Decision.** Add hard prerequisites from `z9gh.3`, `9e5.31.1`, `71ey`, and `rii.1` to `o21.1`; relate `rii.1` to `37t.2.1`. `o21.1` owns identity, declaration completeness, ownership, and storage-free introspection. Query declarations own query semantics; definition closure owns graph semantics; maintenance owns target planning/actuation; marker and work-event adapters share one event-kind vocabulary and lower into existing typed storage. None creates a second generic registry.

### Schema workload hotspot: serialize, do not collapse

**Observed fact.** `1xc.14.1` and all three P1 children are simultaneously `in_progress` with the same assignee and overlapping generation files. Later origin source includes bounded journal replay transaction work, but the children retain distinct acceptance: replay/memory/cancellation and live receipt; privacy/promotion; catalog-role/default semantics.

**Decision.** Keep the children, but normalize execution to one branch/owner and one visible order: `.1` then `.2` then `.3`, then parent canaries/receipts/promotion closure. Rewrite `.1` to the residual proof only so landed journal work is not reimplemented.

### Extension/campaign: four authorities, serialized shared writes

**Observed fact.** Source has `BrowserActionIntent`, exact `BrowserActionReceipt`, receiver-owned durable state, lease expiry to `outcome_unknown`, explicit reconciliation, and a real extension action worker. No campaign ID vocabulary appears in extension/receiver source.

**Decision.** Retain `ptx` for action transport/receipts, `3v1` for capture truth/status, `yyvg.7` for exception-driven UX policy, and `yyvg.6` for external campaign orchestration. Add a soft `yyvg.7 → 3v1` relation. Serialize branches touching `browser-extension/src/**` and `polylogue/browser_capture/**`; keep campaign orchestration on a separate local-tooling branch with a private ledger.

### Beads synchronization: current portfolio supplies regression fixtures

**Observed fact.** The repository instructions warn that checkout hooks re-import branch-local JSONL by blind upsert, and the guard only snapshots around hooks and compares `updated_at`. The current export contains two strictly older rows than Git history: `z9gh.1` and `xnws`.

**Decision.** Keep `gxjh.1` as the sole monotonic synchronization owner and `8jg9.1` as its policy consumer. Add both regressions as acceptance fixtures that must preserve status, assignee, parent/dependency identity, close reason, and newer per-row revision under ordinary import/checkout/merge. Apply tracker repair in one immediately merged bookkeeping branch, never sibling bookkeeping branches.

### Verification hierarchy: repair the only active-child/closed-parent edge

**Observed fact.** `b054.1` is a completed audit, while active `b054.1.1` is a bounded verification harness/proof owner. It is the only active child whose parent is closed.

**Decision.** Reparent `b054.1.1` to active verification-risk program `88jp`; retain a historical relates/discovered link to `b054.1`. Keep its children as distinct proof lanes under one harness root.

## Clusters that share a branch versus remain separate

The executable cluster map is in `EXECUTION-CLUSTERS.csv`. The governing rule is semantic ownership plus write overlap:

- Same branch when Beads share a source-of-truth file set and acceptance cannot be reviewed independently: status, daemon supervision, raw-authority code, query transaction surfaces, schema workload generation, and extension/receiver work.
- Separate branch or phase when authority must remain independent: EventBus after supervision; live raw apply after code review; external campaign after transport; domain declaration consumers after the shared kernel.
- Same bookkeeping branch for all graph/status corrections, merged before any unrelated checkout, because repository instructions document stale-row reimport.

## Other clusters explicitly retained as distinct

The census records no merge for layered pairs that look similar but own different semantics: `4p1` read algebra versus `t46` surface consolidation versus `z9gh` query transaction; `b5l.1` rebuild exclusivity versus raw-authority repair; `1xc.13` source freshness projection versus raw authority; `t8t` scenario catalog versus `z9gh.7` terminal live gate; `5k5l` inbound assistant-output asset acquisition versus `83u.3` uploaded input bytes and `ptx` outbound browser action transport; and the parent/implementation/proof families under `1vpm`, `ovme`, `60i5`, `303r.2`, and `cuxz`.

## Limitations and unresolved uncertainty

This is a static, source-validated graph audit. It did not run the live daemon, browser extension, archive repair/apply, deployment, or full test suite. Live archive sizes, memory figures, cursor counts, backup state, and provider behavior reported in Bead notes remain tracker evidence. The working-tree tar omits many tracked hidden/agent files by packaging design, although the XML slices and Git bundle supplied those contexts; every archived tracked file that was present matched local head. The manifest validates all ordinary artifacts; its self-entry has no SHA-256 and a pre-serialization byte count, which is a packaging/self-reference limitation rather than evidence corruption.

The strongest residual uncertainty is not another duplicate hunt. It is whether the proposed graph repairs survive the next merged code/Beads state and whether live acceptance receipts close the raw, query, status, schema, and extension gates. Another unchanged static pass has low marginal value. A second iteration becomes high-value after the bookkeeping repair or when it includes: a fresh Bead export, merged current master, live daemon/query/status receipts, the schema promotion receipt, and extension provider/project/failure matrix evidence.

## Package map

- `REPORT.md` — conclusions, portfolio diagnosis, and cluster decisions.
- `EVIDENCE.md` — exact source/Bead/history evidence, negative evidence, and limitations.
- `DECISIONS.md` — decision records separating facts, inference, uncertainty, recommendation, preservation, and falsification.
- `NEXT-ACTIONS.md` — ordered non-mutating execution plan and local verification gates.
- `ACTIVE-CENSUS.csv` — all 555 snapshot-active Beads with effective status repair.
- `GRAPH-CHANGES.csv` — proposed record/edge changes with evidence and preservation requirements.
- `EXECUTION-CLUSTERS.csv` — branch/phase grouping and acceptance gates.
- `AUDIT-STATS.json` — compact machine-readable scope and graph statistics.
