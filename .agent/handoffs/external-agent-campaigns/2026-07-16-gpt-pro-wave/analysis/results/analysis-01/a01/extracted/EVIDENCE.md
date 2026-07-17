# Evidence ledger

## Evidence authority and snapshot integrity

The primary archive was inspected directly, including its working-tree tar, all-ref Git bundle, rendered Bead export, manifest, snapshot audit, repository instructions, and source slices. The outer and nested tar member names were checked for traversal. The working-tree tar has 2,779 members and one safe symlink, `AGENTS.md → CLAUDE.md`.

The manifest contains 42 artifact entries. Every non-self artifact with a digest matched its SHA-256. `polylogue-manifest.json` intentionally has `sha256: null`; its recorded 18,941 bytes differ from the final 18,945-byte file because a manifest cannot stably hash/count its own post-serialization content without an external convention. No ordinary artifact digest failed.

The archived source declares local `master` `f654480cadb7cc4c194704e24dfd483199547b35` and `dirty=true`. Hash comparison found no content change in any archived tracked file relative to that commit and one untracked file, `browser-extension/package-lock.json`; many tracked hidden/agent files were intentionally absent from the working-tree tar but present through other snapshot slices/Git. The all-ref bundle contains later `origin/master` `0d081e5bc0b5e5bedcb3256d1779da8c0f091c65`, five commits and 22 files ahead (+1058/−117). Those later commits are:

- `0dc5773a9` — preserve complete raw frontier counts (#2965)
- `805d49286` — stream safe raw authority replay (#2966)
- `9ddfc8d81` — raw authority scale proof runner (#2967)
- `067c87e49` — bound schema journal replay transactions (#2968)
- `0d081e5bc` — guard and meter raw authority proofs (#2970)

The source basis is therefore explicit in every conclusion: archived local source for snapshot reality, later bundled origin where it contains a superseding committed implementation.

## Repository workflow evidence

`CLAUDE.md:263-276` defines Beads as durable tracker/devloop and requires graph lint before shipping Bead deltas. `CLAUDE.md:278-289` documents that checkout/merge hooks re-import the checked-out branch's `.beads/issues.jsonl`, and that an older branch can make a closed Bead appear open. `CLAUDE.md:291-304` forbids sibling bookkeeping branches and recommends folding tracker state into the code branch or merging bookkeeping immediately. `CLAUDE.md:305-308` requires re-reading a Bead after checkout/merge/worktree creation. `CLAUDE.md:309-317` requires hand-merging JSONL by ID and preferring later `updated_at`, rather than using `bd export` during conflict resolution.

`.agent/scripts/bd-reimport-guard.py:2-19` confirms plain-upsert behavior and wraps hooks with before/after snapshots. Lines 103-139 restore rows only when the pre-hook `updated_at` is newer or the row vanished, then re-export. This is useful but not the full monotonic/import/merge/receipt contract described by `gxjh.1`.

## Tracker-history reconstruction

All historical `.beads/issues.jsonl` versions reachable from bundled refs were parsed by issue ID. There were 11,162 row versions for 1,067 IDs. Exactly two current export rows had a strictly later per-ID record in Git history:

| Bead | Export | Later history | Decision |
| --- | --- | --- | --- |
| `z9gh.1` | open, unassigned; updated 2026-07-16 12:56:52Z | in_progress, Sinity; updated 2026-07-17 00:25:42Z at `6610f3b0` | restore in_progress/assignee; do not infer closed from the commit subject |
| `xnws` | open; updated 2026-07-16 23:43:40Z | closed; updated 2026-07-16 23:59:23Z at `6610f3b0`; close reason cites PR #2963 | restore closed |

The `z9gh.1` commit subject says “close,” but the actual later row is `in_progress`; the row, not the subject, controls. This distinction prevents inventing a closure that the tracker record does not support.

## Complete active-graph evidence

The current export contains 976 issues: 537 open, 18 in progress, and 421 closed. Snapshot-active count is 555; effective active count is 554 after restoring `xnws` closed. Priorities among snapshot-active rows are P0=15, P1=99, P2=138, P3=195, P4=108. The active hard graph has 235 edges and no cycle. There are no exact duplicate active titles and no active row has multiple parent-child edges. The only active-to-active explicit supersession is `20d.17 → 703`; the only active child with a closed parent is `b054.1.1 → b054.1`; the only P0/P1 hard dependency on a P3/P4 row is `avmq → yp0`.

`ACTIVE-CENSUS.csv` is the complete row-level evidence for this section.

## High-value Bead records

| Bead | State | Authority evidence | Recorded dependencies |
| --- | --- | --- | --- |
| polylogue-20d.17 | open P1 | Define one StatusComponentSpec and StatusSnapshot protocol reused by daemon/archive and agent-coordination status. | parent-child→polylogue-20d, relates-to→polylogue-20d.14, supersedes→polylogue-703, relates-to→polylogue-cuxz, relates-to→polylogue-s7ae.8 |
| polylogue-703 | open P2 | The convergence-snapshot bead (4bu) defines the shared payload for converging-state; this bead is the structural follow-through: ONE status assembly module in the substrate comput… | parent-child→polylogue-t46 |
| polylogue-kwsb.2 | open P1 | The missing abstraction is a shared transaction protocol, not a universal mutation executor. | parent-child→polylogue-kwsb |
| polylogue-t46.9 | open P1 | Polylogue already declares operation safety in operations/specs.py and action_contracts.py, but CLI, API, MCP, daemon, maintenance, and repair adapters still enforce role, dry-run… | relates-to→polylogue-a7xr.18, relates-to→polylogue-jn40, relates-to→polylogue-jnj.5, parent-child→polylogue-t46, relates-to→polylogue-t46.8 |
| polylogue-avmq | open P1 | Supervise every daemon background service through one lifecycle contract | blocks→polylogue-yp0 |
| polylogue-yp0 | open P3 | Adding an event/subscriber requires one declared registry entry and completeness checks, not bespoke loop wiring. | blocks→polylogue-9e5.7, parent-child→polylogue-a7xr |
| polylogue-hjpx | in_progress P0 | 2026-07-17 static follow-up from yla8 read-only preflight: current live execution is blocked by one non-stream-safe authority component over the 1 GiB bounded replay envelope. | parent-child→polylogue-lkrc, discovered-from→polylogue-yla8 |
| polylogue-hjpx.2 | in_progress P1 | hjpx AC6 remains unproven after the execution-foundation phase. Small unit fixtures exercise 25 singleton raws and skewed cohorts, but they do not match the 2026-07-15 preflight s… | parent-child→polylogue-hjpx, blocks→polylogue-hjpx.1 |
| polylogue-yla8 | in_progress P0 | Chunk 1 rejected two full replays while preserving the accepted index: (1) a 25,898,236-byte ChatGPT browser capture for session 69d5383e-69d0-8327-a899-94a89ff35ea4 hit "conflict… | parent-child→polylogue-1xc, relates-to→polylogue-1xc.13, relates-to→polylogue-b5l.2, relates-to→polylogue-n2wy |
| polylogue-lkrc | in_progress P0 | Polylogue has accumulated separate repair actuators and incident Beads for origin-mismatched browser raws, competing canonical heads, duplicate raw identities, replaced snapshots… | parent-child→polylogue-1xc, relates-to→polylogue-1xc.13, related→polylogue-2qx, relates-to→polylogue-b5l.1, blocks→polylogue-yla8, discovered-from→polylogue-yla8.10 |
| polylogue-z9gh.1 | open P0 | Cancellation, deadlines, off-event-loop execution, and admission control belong to the shared query transaction rather than MCP-specific wrappers. | relates-to→polylogue-1xc.14, related→polylogue-20d.14, related→polylogue-oxz, parent-child→polylogue-z9gh.9 |
| polylogue-z9gh.9.1 | open P0 | Land the shared query transaction across every read surface | supersedes→polylogue-20d.5, relates-to→polylogue-7q16, relates-to→polylogue-9l5.6, relates-to→polylogue-rxdo.3, blocks→polylogue-z9gh.1, blocks→polylogue-z9gh.2, relates-to→polylogue-z9gh.3, parent-child→polylogue-z9gh.9 |
| polylogue-o21.1 | open P1 | Missing producer, handler, role gate, schema, example, generated output, or consumer edge produces one source-locatable Diagnostic containing declaration id, owner path, and exact… | relates-to→polylogue-1xc.14, parent-child→polylogue-o21 |
| polylogue-z9gh.3 | open P0 | One query declaration registry generates MCP schemas/descriptions, a searchable capability/coverage catalog resource, typed structured-plan input, DSL completions/help, OpenAPI/do… | related→polylogue-o21, relates-to→polylogue-o21.1, related→polylogue-t46.8, parent-child→polylogue-z9gh, relates-to→polylogue-z9gh.9.1 |
| polylogue-9e5.31.1 | open P1 | Seed representative policies: one durable data family, one event/write-effect family, one registry/declaration family, one query parse-to-render path, and one CLI/MCP/HTTP/Python… | parent-child→polylogue-9e5.31 |
| polylogue-71ey | open P1 | The canonical maintenance target catalog advertises seven targets, but resumable replay has a private six-target _REPLAY_DISPATCH that omits superseded_raw_snapshots. The generate… | discovered-from→polylogue-9e5.31, parent-child→polylogue-o21, relates-to→polylogue-o21.1, relates-to→polylogue-sl1 |
| polylogue-rii.1 | open P2 | record_work_event/emit_decision write surface routed through the existing idempotent ingest seam (no parallel writer); flows into the run-projection read models. Today agents can… | parent-child→polylogue-rii |
| polylogue-37t.2.1 | open P1 | Implement the provider-neutral author-declared structure channel: collision-tested line/inline syntax parsed at block enrichment, exact message/block provenance, malformed evidenc… | parent-child→polylogue-37t.2, relates-to→polylogue-o21, blocks→polylogue-o21.1 |
| polylogue-ptx | in_progress P0 | One versioned BrowserActionIntent contract can create a new Chat conversation or reply to an exact existing conversation with text plus multiple hash-pinned attachments; successfu… | blocks→polylogue-3v1.1, blocks→polylogue-83u.3, blocks→polylogue-83u.4, parent-child→polylogue-bby, blocks→polylogue-kwsb.1 |
| polylogue-3v1 | in_progress P1 | The extension (MV3, popup + badge, content bridges for chatgpt.com/claude.ai/grok.com/x.com) works end-to-end but its trust surface is thin: the operator cannot tell at a glance w… | parent-child→polylogue-jlme |
| polylogue-yyvg.7 | in_progress P0 | F4/ys30/yyvg.4 owns capture dot and Save action in provider-native per-message action rows, resolved by one ProviderAdapter identity contract; no ordinal/text-only durable authori… | parent-child→polylogue-yyvg |
| polylogue-yyvg.6 | in_progress P0 | The orchestrator owns MissionId, RunId, IterationId, DeliverableId, PackageRevisionId, readable prompt/attachment/result filenames, retry/alternative/supersedes lineage, prompt pr… | parent-child→polylogue-yyvg |
| polylogue-gxjh.1 | open P1 | Import compares project/database/branch identity and per-row revision/updated_at in one transaction, creates missing rows, accepts demonstrably newer rows, preserves equal rows, a… | parent-child→polylogue-8jg9 |
| polylogue-8jg9.1 | open P1 | Standalone .agent/tools/bead-lint.py removed (superseded); allowlist at .agent/tools/bead-lint-allow.txt unchanged. | parent-child→polylogue-b054, blocks→polylogue-gxjh.1 |
| polylogue-b054.1 | closed P1 | Audit the full open set for cases where one executable abstraction, normalized relation, declarative registry, transaction boundary, or authority rule can make several special cas… | parent-child→polylogue-b054 |
| polylogue-b054.1.1 | in_progress P1 | Fresh-worktree and warm-cache runs emit machine-readable receipts consumed by devtools verify and agent guidance, with one tested configuration contract across focused, xdist, see… | relates-to→polylogue-09rn, parent-child→polylogue-b054.1, discovered-from→polylogue-hjpx, relates-to→polylogue-vyxq, relates-to→polylogue-wple, relates-to→polylogue-y6tb |
| polylogue-88jp | open P2 | Absorbs remaining metric-only scope of c52g, znwj, and n4hb; their landed PR #2787 behavior tests become evidence inputs. | relates-to→polylogue-09rn |
| polylogue-xnws | open P3 | WHY: the evidence-honesty doctrine says degraded modes are loud and unknown never renders as zero/blank, but no census exists of except-and-continue sites in production code. Dogf… | — |

Short, decision-bearing record text:

- `20d.17` design defines one `StatusComponentSpec`/`StatusSnapshot` protocol; its note says it “absorbs polylogue-703,” and the dependency list has `supersedes→polylogue-703`.
- `703` requires one shared status assembly and a CLI/daemon/MCP/Python/web parity suite with a citation-drift fixture. That evidence must move, not disappear.
- `kwsb.2` says: “The missing abstraction is a shared transaction protocol, not a universal mutation executor.”
- `t46.9` says: “Make OperationSpec the single executable declaration and add an OperationExecutor used by every external and internal mutation route.”
- `avmq` says event delivery is adjacent to service supervision and that EventBus remains transport; `yp0` says the core is landed and production loop conversion is follow-up work.
- `hjpx.2` says `hjpx` AC6 remains unproven and owns the July-15-shape scale proof; `hjpx` calls itself the executable successor to the failed live gate; `yla8` retains operator authorization and fail-closed protections; `lkrc` owns final authority closure.
- `rii.1` says marker kinds and work-event kinds are one channel with two encodings; `37t.2.1` already hard-depends on `o21.1`.
- `gxjh.1` requires per-row revision/`updated_at` monotonicity, conflict refusal, transactionality, and receipts; `8jg9.1` already hard-depends on it.

## Source evidence by hotspot

### Status

- `polylogue/daemon/status_snapshot.py:50-75` defines a single whole-payload `StatusSnapshot` with captured time, age, fresh/stale state, and refresh error.
- `status_snapshot.py:292-336` returns that whole cached payload and refreshes it from `daemon_status_payload`; refresh is process-global and lock-guarded, not independently componentized.
- `polylogue/daemon/status.py:1986-2033` begins the large `build_daemon_status` and synchronously assembles DB/storage/FTS/freshness/raw frontier/replay/cursor/debt/failure/embedding facts.
- `polylogue/cli/commands/status.py:1376-1404` first calls `/api/status`, then falls back to independent direct JSON/text status assembly.
- `polylogue/coordination/envelope.py:249-291` has a separate coordination projection path.
- Repository-wide search found no `StatusComponentSpec` symbol.

This source supports absorption of 703 into 20d.17 but does not prove the component protocol has landed.

### Destructive mutations

- `polylogue/operations/specs.py:47-62` defines `OperationSpec` as metadata: consumes/produces/path/code/surfaces, mutability, previewability, idempotency, effects, and safety guards.
- `polylogue/operations/operation_contract.py:3-12` records that the previous generic `OperationRequest`/`OperationAck` framework was removed because only IMPORT had a production consumer, and says to reintroduce a shared base only when a second operation lands.
- `operation_contract.py:14-26` retains only genuinely reused scheduling status/follow-up shapes and says the wire envelope is unchanged.

This directly contradicts using `t46.9` to preemptively make a universal executor the product authority. It does not reject shared transaction primitives; that is why `kwsb.2` survives.

### Daemon lifecycle and event transport

- `polylogue/daemon/cli.py:1089-1108` exposes one very large `run_daemon_services` composition function.
- `daemon/cli.py:1252,1281,1311,1321,1387,1417,1424` creates maintenance, browser receiver, API, UDS, periodic, catch-up, and watcher tasks directly; `1434-1449` gathers them.
- `polylogue/daemon/event_bus.py:1-48` explicitly says the core lands typed pub/sub and production loop conversion is follow-up work.
- `event_bus.py:120-129` says one bus is meant to be constructed in `run_daemon_services`, but repository-wide production search found no reference to `EventBus` or its event types outside that module.
- No `DaemonServiceSpec` or `ServiceHarness` symbol exists.

This supports separate lifecycle and transport authorities and falsifies the current hard blocker.

### Raw authority

Later bundled origin source supplies the current committed implementation:

- `polylogue/storage/raw_reconciler.py:51-80` defines mutually exclusive frontier states and admitted actuators; only safely rekeyable and duplicate-alias states are executable.
- `raw_reconciler.py:99-147` binds every item to state, actuator, evidence digest, preconditions, witness, inputs, and stable plan ID; the census records complete state counts and plan inventories.
- `raw_reconciler.py:949-985` persists a complete census under the offline/daemon exclusion boundary, computes plans, and records a dry-run receipt with complete frontier counts.
- `raw_reconciler.py:1197-1243` requires a unique selected plan set, verifies membership in the preview, re-derives current plans, refuses non-executable plans, and records the apply census.
- `raw_reconciler.py:1316-1349` recovers interrupted planned applications from typed durable evidence.
- `devtools/raw_authority_scale_proof.py:40-68` records pass cardinality, fixed point, wall time, RSS/PSS/swap, CPU, and I/O.
- `raw_authority_scale_proof.py:109-124` samples Linux process and pressure evidence.
- `raw_authority_scale_proof.py:164-219` invokes the real `repair_raw_materialization`, rejects incomplete metrics, requires a census receipt, and emits resource/convergence evidence.

The source proves meaningful implementation has landed but not the Bead-level live closure. It supports the ordered proof/executor/live-gate/closure chain.

### Query execution and residual surface migration

- `polylogue/archive/query/execution_control.py:126-183` defines immutable query identity, workload class, deadline, cancellation event, receipt, and abort checks.
- `execution_control.py:186-223` defines weighted admission, FIFO within class, and reserved interactive capacity.
- `execution_control.py:296-337` opens one dedicated read-only `ArchiveStore` and installs a cancellation/deadline progress guard.
- `execution_control.py:417-474` runs reads off the event loop, translates disconnect cancellation into exact connection interrupt, and drains/receipts cleanup.
- `execution_control.py:477-503` supplies the synchronous HTTP variant sharing the process-wide admission controller.
- Production consumers exist at `polylogue/api/archive.py:2907-2958`, `polylogue/mcp/server_tools.py:303-333`, and `polylogue/daemon/http.py:3507-3533`.
- Direct MCP opens remain at `server_tools.py:240,394,699,966,1079` and in `server_resources.py`/`server_prompts.py`; direct daemon HTTP opens remain at `http.py:2202,2570,2816,2911,3096,3947`.

This is source evidence for a shipped core plus a residual transaction/surface migration, not a reason to reopen a parallel execution architecture.

### Declaration, maintenance, and event-kind registry

- `polylogue/declarations/` is absent in both archived local and later origin trees.
- `polylogue/maintenance/targets.py:21-68` defines `MaintenanceTargetSpec` and its canonical catalog metadata.
- `targets.py:158-234` advertises `superseded_raw_snapshots` among maintenance targets.
- `polylogue/maintenance/replay.py:101-128` maintains a separate private `_REPLAY_DISPATCH`; that dispatch lacks `superseded_raw_snapshots`.
- `replay.py:594-624` turns an advertised-but-unwired target into a typed unsupported failure, explicitly telling maintainers to edit the private dispatch.
- `polylogue/storage/sqlite/queries/session_events.py:48-118` provides typed session-event reads.
- Repository-wide source search found no `record_work_event` or `emit_decision` API.

This supports one shared declaration kernel and separate domain adapters. It also gives 71ey a concrete current-source pilot rather than an abstract architecture exercise.

### Schema workload generation

- `polylogue/schemas/generation/observation_journal.py:159` defines the journal; `:210` creates a permission-restricted local SQLite journal; later iterators replay profile/unit/membership/terminal evidence; `:1044` owns close/cleanup.
- `workload_profiles.py:370-404` emits aggregate structural profiles with explicit privacy policy/classification.
- `archive_workload_profile.py:715-795` builds privacy review metadata and writes deterministic gzip artifacts for staging/promotion review.
- `provider_bundle_packages.py:74-97,269-316` selects and publishes latest/default/recommended catalog roles.
- Feature branch `feature/feat/schema-workload-profiles` contains additional cleanup, order invariance, privacy, provenance, and receipt commits; later origin #2968 bounds journal replay transactions.

These files overlap heavily, but the three child acceptance semantics remain different. The evidence supports serialized execution, not merging the children.

### Browser action, capture, UX, and campaign boundary

- `polylogue/browser_capture/models.py:347-375` defines provider-neutral operations, terminal statuses, and outcomes including `outcome_unknown`.
- `models.py:450-464` defines the exact provider-observation receipt; `:477-495` defines durable receiver-authoritative `BrowserActionIntent` state.
- `polylogue/browser_capture/actions.py:343-370` converts an expired durable submit intent to `outcome_unknown`, rather than retrying blindly.
- `actions.py:430-508` enforces terminal idempotency/conflict and lease-owner updates; `:511-533` only reconciles an `outcome_unknown` action with an exact receipt.
- `browser-extension/src/background.js:2048-2217` contains the action transport/worker and ambiguous-submit handling.
- Search found no `MissionId`, `CampaignId`, `mission_id`, or `campaign_id` in extension/receiver source.

This supports four distinct authorities and a hard boundary against extension-owned campaign orchestration.

## Stale test-suite package cross-check

The separate package labels itself “slightly stale,” so it was not used to override source or Beads. It independently names `lkrc`/`yla8` as evidence-authority and authorization owners, `z9gh.1` as query cancellation/bounds owner, and `20d.17` as the status component-snapshot contract. That agreement increases confidence in the retained ownership boundaries but adds no authority beyond current source/tracker history.

## Negative evidence and limits

Negative source searches were used only where the snapshot was broad enough to make them meaningful: no `StatusComponentSpec`, `DaemonServiceSpec`, `ServiceHarness`, declaration package, production EventBus consumer, `record_work_event`, `emit_decision`, or campaign-ID leakage. Absence in source does not prove absence in a live uncommitted worktree outside the snapshot.

No live browser, daemon, deployment, archive apply, or full test evidence was available or claimed. No Bead mutation was performed. The proposed changes remain recommendations until a local implementer applies them, re-reads the changed rows after checkout, runs graph lint, and verifies current source/tests.
