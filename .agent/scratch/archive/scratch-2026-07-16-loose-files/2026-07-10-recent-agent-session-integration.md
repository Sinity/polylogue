---
created: "2026-07-10"
purpose: "Evidence-graded integration of the closed Fable Polylogue session, the July 10 Codex strategy/control session, and the immediately following runtime-recovery context."
status: "active-integration-note"
project: "polylogue"
sessions:
  - "claude-code-session:3347cf34-ca12-45ae-918f-781c7f96a704"
  - "codex-session:019f49d8-0185-7c43-8793-db6e57db13e1"
  - "codex-session:019f4aab-a008-7073-9142-c13acc96bdfd"
  - "codex-session:019f4aab-ca7b-7691-aedd-55b4364d1225"
scope: "Synthesis only; no Beads, product-code, branch, daemon, or browser mutation performed by this task."
---

# Recent Agent Session Integration

## Evidence

### Primary session evidence

- Fable raw transcript:
  `/home/sinity/.claude/projects/-realm-project-polylogue/3347cf34-ca12-45ae-918f-781c7f96a704.jsonl`.
  This is the primary source for Fable's investigation, corrections, dialogue,
  benchmark census, cloud-bootstrap work, and unfinished state. The session was
  compacted and closed; it was read, not resumed.
- Codex strategy/control transcript:
  `/home/sinity/.codex/sessions/2026/07/10/rollout-2026-07-10T04-25-20-019f49d8-0185-7c43-8793-db6e57db13e1.jsonl`.
  This is the primary source for the repo/backlog survey, strategy synthesis,
  cloud execution, browser/control experiments, recall defect, and runtime
  failure discovery.
- Immediate parent/control context:
  `/home/sinity/.codex/sessions/2026/07/10/rollout-2026-07-10T08-16-29-019f4aab-a008-7073-9142-c13acc96bdfd.jsonl`.
  It records the runtime-mask/rebuild plan, the local Sol-lane decision, and the
  request for durable session integration.
- This synthesis lane:
  `/home/sinity/.codex/sessions/2026/07/10/rollout-2026-07-10T08-16-40-019f4aab-ca7b-7691-aedd-55b4364d1225.jsonl`.
  It is evidence only for the scope and live-state observations made while
  writing this note.

### Distilled local evidence

- `.agent/scratch/2026-07-10-broad-project-strategy-and-verifiability.md`:
  Codex's verified system/strategy analysis, revised one-week campaign, and
  later cloud/browser/runtime updates.
- `.agent/scratch/2026-07-10-agent-dialogue.md`: append-only Fable/Codex
  dialogue with 16 consensus items and execution updates through entry [11].
- `.agent/scratch/2026-07-10-fable-notes.md`: Fable's corrections, ownership
  record, census results, and delegation matrix.
- `.agent/archive/devloop-2026-07/README.md` and `PACKET-README.md`: the
  conductor packet's retired state and its former state-machine semantics.
- `.agent/archive/conductor-history/2026-07-01/README.md`: conductor history is
  archaeology only and is not a startup path.
- `.beads/issues.jsonl`: read-only current records for `polylogue-s7ae*`,
  `polylogue-1hj`, and `polylogue-ptx`. No Beads command or mutation was used
  by this synthesis task.

### External/candidate evidence

- `/realm/inbox/gpt-pro-sol/00-README.md` and prompt files `01` through `06`:
  design-review requests, not accepted specifications.
- `/realm/inbox/gpt-pro-sol/10-cloud-implementation-lanes.md` and
  `/realm/inbox/gpt-pro-sol/polylogue-cloud/{00-README.md,LAUNCH.md,01-04*.md}`:
  cloud executor/runbook material.
- `/realm/inbox/gpt-pro-sol/results/06-strategy-falsification.txt`: GPT-5.6 Pro
  candidate memo, verdict **NARROW with a hard 30-day stop gate**. Its first
  pass lacked both July 10 strategy/dialogue notes and is not project authority.

## Verified Facts

1. The technically credible core is the local, cross-provider evidence ledger:
   durable raw acquisition, normalized sessions/messages/blocks/actions,
   structural outcomes, lineage, exact/structured search, stable evidence refs,
   and rebuildable read models. Broader adoption/value claims remain less well
   evidenced than the core.
2. The backlog survey observed 612 Beads records: 171 closed, 440 open, one in
   progress, 104 blocked, and 336 dependency-ready. Raw `ready` is therefore not
   an execution queue; delivery gate, readiness, ownership, overlap, and proof
   quality are required selectors.
3. Fable corrected three material errors during the dialogue:
   - `cpf.5` was already closed; remembered Bead state was stale.
   - canonical `index.db` schema v29 was real, while production was still v24
     at the time of the strategy discussion.
   - the affordance review did **not** recommend retiring 59 MCP and 34 CLI
     surfaces. Reviewed decisions were keep 56/59 MCP, keep all 34 CLI, retire
     exact duplicate `v6vy`, and hold `moyt` consolidation for parity proof.
4. The live source tier was migrated v2 to v3 through the supported backup and
   migration path. The recorded verified backup manifest is
   `/realm/data/captures/polylogue/backups/2026-07-10-source-v2-pre-v3/polylogue-archive-20260710T043717Z/manifest.json`.
5. Fable's read-only eligibility census found a large Receipts candidate pool:
   Claude Code 4,450 L2-eligible claims (1,912 after the two-per-session cap)
   and Codex 2,713 (223 after cap). Both clear the 60-per-origin target. The cap
   discards 92% of Codex candidates versus 57% of Claude candidates, confirming
   strong session clustering.
6. Only 3 of 1,735 claim sessions (0.17%) in the probe contained duplicate
   `(session_id, tool_id)` tool-use groups. For the v24 benchmark frame,
   exclude-and-disclose is simpler than backporting ranked pairing, provided the
   excluded refs/counts remain auditable.
7. Cloud bootstrap hardening landed through PR #2631 (`c68585b8b`). It made the
   render check real/visible, added bounded pytest workers and `/tmp` base-temp,
   updated cloud docs, and left testmon seeding behind a measurement gate.
8. Codex Cloud canary `k6fm` completed and merged through PR #2632
   (`4dabc85dd`). The reliable control surface was `codex cloud exec/list/status/
   diff/apply`, not browser submission. A plain cloud `devtools verify --quick`
   was an invalid invocation outside the synced environment; packets were
   corrected to `uv run devtools ...`.
9. `kj22` made `fuzz_*.py` discoverable and thereby exposed real failures in
   the formerly skipped suite. It was correctly held from merge pending failure
   classification. `v6vy` returned a bounded MCP-contract diff and passed 294
   focused tests before its broader/static integration work.
10. The browser extension already has a double-gated reverse channel. A
    no-submit ChatGPT canary was queued, claimed, filled the addressed composer,
    and acknowledged `dry_run_filled_not_sent`. The command contract currently
    carries text/target/submit only. Attachments and model verification still
    require CDP, and duplicate tabs with the same conversation id select
    nondeterministically.
11. The control/recall investigation found a high-value perception defect: a
    333-message Codex session produced an effectively empty resume brief even
    though its raw tail contained a concrete next action. The announced repair
    is an on-read degraded-profile fallback to the latest archived assistant
    state. At integration time it exists only as uncommitted edits in
    `/realm/tmp/worktrees/polylogue-resume-fallback`.
12. Browser capture produced seven provider-native GPT Pro envelopes, but the
    normal directory import rejected the staged live-inbox path as
    `invalid_path`. Capture and ingestion therefore do not yet compose into a
    reliable operator workflow.
13. The deployed `polylogued.service` was active-looking but stale: it expected
    source v2 after the archive had reached source v3 and retried failed ingest.
    It was runtime-masked and stopped. A current-source daemon then exposed the
    expected index v24 versus canonical v29 incompatibility. Live observation
    during this synthesis: `polylogued.service` is `inactive/dead` and
    `masked-runtime`; `polylogue-index-rebuild.service` is transient and
    `activating`; source reports v3 and the replacement index file reports v29.
    Convergence/completeness was not yet verified.

## Decisions

- Preserve Polylogue as a local cross-provider evidence graph. Do not turn it
  into an execution runtime or a second work tracker.
- Beads remains sole work authority. Markdown packets, prompt files, task
  snapshots, and coordination projections are evidence/transport artifacts,
  never a parallel state machine.
- Reconcile execution planning into `s7ae` as a compact read-only
  `planned_work` projection. Ship it internal/experimental first; public MCP
  registration waits for a real `s7ae.5` consumption proof.
- `planned_work` freshness uses content epochs, not mtimes: git HEAD plus
  normalized dirty-state hash, canonical selected-Bead/dependency hash, policy
  hash/version, and short-TTL resource observation. Unknown ownership, proof,
  avoidance, or exclusivity cannot authorize dispatch.
- Keep `s7ae.1` closed. Its row-count AC was delivered. Create a successor for
  semantic projection and byte budgets, relate it to `s7ae.1`, and make it a
  hard prerequisite of corrected `s7ae.5`.
- Receipts aligns by transcript position, not wall-clock time. V1 remains
  balanced at 60/60, capped at two sampled claims per session, with labeler
  family balanced across claim origin. Duplicate-tool-id sessions are excluded
  and disclosed. Any powering pilot must use a predeclared disjoint frame.
- Name the repeated verification pathology **fixture/matrix vacuity**. A
  behavioral/evidence checker must turn red when its claimed real proof is
  removed or corrupted. Cheap static policy lints need ordinary negative
  fixtures, not a mutation framework.
- No broad surface-retirement campaign. Lifecycle retirement requires a
  reviewed decision and replacement/proof, not a generated zero-use label.
- Cloud agents use synthetic data only and never mutate `.beads/**`. Real
  archive census/extraction, real Hermes fingerprinting, live canaries, archive
  migration/rebuild, and `s7ae.5` remain local.
- Do not resume the closed Fable session. Its durable output is already in the
  dialogue, Fable notes, prompt packets, PR #2631, and this integration note.
- New ChatGPT Pro conversations are globally rate-limited to at most one per
  five minutes; widen immediately to one per fifteen minutes after any
  rate-limit response. Prefer continuing an appropriate existing conversation.

## Critiques

- Test quantity is not the constraint. The main weakness is proof authority:
  closure matrices can validate row shape without executing the named proof;
  synthetic provider fixtures can validate an invented schema rather than the
  deployed artifact; hand-written contract shadows can drift from handlers;
  browserless DOM/source tests do not prove operator journeys.
- The strategy and tech tree are coherent but oversubscribed. The original
  one-week plan was estimated at two to three times realistic integration
  capacity. Six to eight simultaneous cloud branches would optimize quota use
  while overwhelming review/merge throughput.
- A large backlog and sophisticated mechanisms do not establish recurring user
  value. The external falsification memo correctly demands comparative evidence
  and stop conditions. It overreaches when it treats absent adoption telemetry
  as direct evidence of no value, especially because its package omitted the
  same-day strategy and dialogue evidence.
- Coordination view names currently overpromise semantic projection. Measured
  `status`/other views were 51-64 KiB and retained unrelated historical arrays,
  systemd daemons were classified as build resources, retired conductor refs
  remained in handoffs, and harness launcher metadata leaked into agent rows.
- Control remains fragmented: Polylogue can recall sessions, the extension can
  post text, CDP can upload/check models, Codex CLI can launch cloud tasks, and
  raw JSONL can recover a failed resume brief, but no honest bounded operator
  projection composes those handles with degradation and provenance.
- The active-looking stale daemon is a particularly serious observability
  failure. Service liveness did not mean ingest freshness, and the retry loop
  consumed substantial IO/memory while perception silently aged.

## Plans And Proposals

### Bounded value campaign

The local strategy proposes a one-week campaign; the external reviewer proposes
a 30-day narrower comparison. They can be reconciled by treating the week as a
pre-registered first tranche of the 30-day falsification, not as permission for
the broad roadmap.

The load-bearing outcomes are:

1. Current Hermes observed-schema compatibility and fidelity proof.
2. Current-UI Playwright journeys with real browser evidence.
3. WorkflowProofSpec/gap compiler that detects removed Web/Hermes authority.
4. Receipts benchmark or an honest methodological negative result.
5. One compact `s7ae` planned-work dispatch whose proof refs return to the
   coordination envelope.
6. Export, clean restore, answer reproduction, and stop drill for the narrow
   archive/evidence core.

Use the external comparative gate as the continuation test: Polylogue versus
direct provider files/ripgrep/a small SQLite baseline, with raw-evidence
resolution, correctness, median time, and maintenance share disclosed. Exact
thresholds from the first GPT memo remain proposals until locally adjudicated.

### Proof and benchmark details

- Freeze claim patterns, verifier families, validity windows, and ambiguity
  exclusions before any outcome pilot.
- Run a coarse powering pilot only on a temporally disjoint frame excluded from
  the final sample; allow it to change symmetric sample size/family inclusion,
  not labels or case selection.
- WorkflowProofSpec should enumerate the reviewed supported registry as-is,
  require one executable positive or an open gap per declared surface, and cap
  mandatory/default generated cases rather than create a Cartesian matrix.
- Preserve the invented Hermes superset fixture as an
  `unsupported-future-schema` case, and add a separate observed-v11 supported
  fixture.

## Execution And Cloud Results

- Completed: source durable migration v2 to v3 with verified backup.
- Completed: cloud-bootstrap PR #2631 and `k6fm` PR #2632, including cleanup.
- Completed evidence: Receipts eligibility census v0 and pairing-ambiguity
  probe; neither is the final frozen benchmark.
- Completed control proof: extension reverse-channel no-submit canary.
- In progress at integration time:
  - production index v29 rebuild under transient
    `polylogue-index-rebuild.service`;
  - resume-brief fallback edits, uncommitted;
  - `kj22` failure classification, with `pyproject.toml` modified;
  - `v6vy` integration/static verification, with its bounded diff present;
  - local Sol `d1y` hook install/liveness lane, worktree created but no edits or
    commits yet.
- Failed/blocked evidence that must survive:
  - ordinary cloud `devtools` invocation outside `uv run` produced inherited
    mypy errors;
  - browser task submission did not create the Codex task;
  - Claude Code Web reported its session limit;
  - capture-to-live-import rejected the staged directory with `invalid_path`;
  - stale deployed daemon could not ingest source v3;
  - current source could not serve the old derived index v24 without rebuild.

## Unfinished Actions And Ownership

- **Coordinator/current Codex parent:** finish and verify index rebuild; restore
  one current daemon under the single-writer contract; prove ingest freshness;
  classify/fix the capture inbox `invalid_path`; review/integrate child lanes.
- **Resume fallback worktree**
  (`/realm/tmp/worktrees/polylogue-resume-fallback`, branch
  `feature/fix/resume-brief-fallback`): uncommitted changes to
  `polylogue/insights/resume.py` and `tests/unit/core/test_resume.py`. Must prove
  degraded-profile signaling and latest-assistant fallback before publication.
- **Local Sol lane `d1y`**
  (`/realm/tmp/worktrees/polylogue-d1y`, branch
  `feature/feat/hooks-install-liveness`): owns one-command Claude/Codex hook
  install/status/uninstall and liveness evidence. It must not add advisory
  injection or a scheduler and must wait for rebuild IO ownership before tests.
- **`kj22` worktree:** owns fuzz collection plus classification/repair of the
  newly visible failures. A collection-count-only merge is not acceptable.
- **`v6vy` worktree:** owns exact duplicate MCP retirement and all contract,
  discovery, generated-reference, and direct-doc updates. Do not mix with other
  MCP registration work.
- **Existing `f2qv.2` owner:** retain ownership of the completed-looking dirty
  canonical-checkout work. Do not reassign or duplicate it.
- **Fable:** no implementation ownership remains. Do not resume/reopen it.
- **GPT strategy conversation:** a follow-up with the missing July 10 notes was
  sent, asking for explicit `s7ae`/`1hj`/graveyard reconciliation and a strict
  drop order. No follow-up result file was present when this note was written.
- **Tracker coordinator, after runtime stabilizes:** amend stale premises and
  encode only adjudicated successor/gap work in Beads. This synthesis did not
  perform that surgery.

## Contradictions

### `s7ae` versus the conductor graveyard

There is no contradiction if `s7ae` remains an evidence projection. The retired
conductor was a second operational state system: active-loop markdown,
operating logs, focus/mode transitions, generated sidecars, demo radar, helper
scripts, handoff packets, and its own resume protocol. Its archive README says
Beads now owns the loop and explicitly forbids resurrection.

`s7ae` may survive only with these boundaries:

- read authoritative Beads/git/session/hook/resource evidence;
- emit bounded, expiring, provenance-bearing projections;
- never own work status, transitions, claims, or a packet directory;
- remove `.agent/conductor-devloop/*.md` from `s7ae.5` AC;
- replace the stale handoff requirement with queryable envelope/message/proof
  refs;
- add the `s7ae.1` semantic/byte-budget successor before the live proof;
- keep `planned_work` internal until a real adapter consumes it.

Any proposal to revive `ACTIVE-LOOP.md`, packet sync scripts, mode transitions,
or a markdown mission database is conductor resurrection and should be dropped.

### `1hj` versus the conductor graveyard

`1hj` can remain as the bounded message-delivery leg under `s7ae`, not as a
group chat or task tracker. Its legitimate scope is scoped user-tier messages,
TTL/expiry, delivery receipts, session-start/direct delivery through registered
ContextSources, and queryable evidence. The context scheduler owns budget,
deduplication, trust, cooldown, and the inclusion ledger. Messages may carry
Bead/query/evidence refs but may not replace Beads work state.

The external NARROW verdict means `1hj` should not expand broadly on faith.
Subject it to the deliberately thin two-runtime coordination falsification; if
it does not improve accuracy/time or produce auditable handoff value, freeze it.

### `ptx` versus the conductor graveyard

`ptx` is an actuator hardening bead for an already-existing browser posting
channel. It is not a scheduler, work queue, or session state machine and should
not be folded into a revived conductor. It owns the receiver/extension posting
path, attachments, capture of what was sent, and the safety/targeting contract.

Its current tracker text contains a real internal contradiction: the original
AC says flip posting on and prove live attachment posting, while the later
design note says every reverse-channel command should become a visible dry-run
draft requiring explicit operator send. The live canary supports the safer
draft-first model. Before implementation, adjudicate and rewrite the AC around:

- draft-first/no automatic submit by default;
- explicit operator send authority;
- deterministic selection when duplicate conversation tabs exist;
- explicit model evidence/selection;
- attachment upload plus acquisition of the exact sent blobs;
- provider/conversation addressing and auditable acknowledgments.

Do not create a parallel CDP/browser-control ontology. CDP remains a temporary
capability adapter until `ptx` covers these gaps.

### Resume/control defects versus the graveyard

An empty resume brief, fragmented browser/cloud handles, and stale-daemon
freshness are evidence that the current control plane is incomplete. They do
not justify restoring the old packet. Repair the read model, degradation
semantics, liveness/freshness health, and bounded `s7ae`/`ptx` projections at
their owning substrates.

## Recommended Sequence And Drop Order

1. **Runtime truth first.** Complete the v29 index rebuild, verify source/index
   integrity and convergence, start exactly one current daemon, and prove new
   Claude/Codex/GPT material ingests. Fix or durably track the inbox
   `invalid_path`. No heavy competing tests while rebuild IO owns the host.
2. **Finish perception repairs.** Verify/publish the resume-brief degraded
   fallback. Make daemon health distinguish process liveness, schema
   compatibility, ingest freshness, backlog, and retry failure.
3. **Integrate already-returned bounded lanes.** Classify `kj22` failures before
   merging collection; finish `v6vy` contracts/static gate; let `d1y` proceed
   only after rebuild pressure clears. Do not open additional cloud branches
   until these are reviewed.
4. **Tracker truth before new coordination implementation.** Preserve
   `s7ae.1` closed, create its successor, amend `s7ae.5`, constrain `1hj`, and
   adjudicate `ptx` draft-first AC. Resolve other known stale premises
   (`fs1.1`, `1ilk`, `212.2`) before dispatch.
5. **Run the bounded falsification tranche.** Freeze Receipts protocol, produce
   current Hermes/browser/proof-gap red evidence, and test one internal compact
   `planned_work` dispatch plus a thin two-runtime message/handoff path.
6. **Continue only on proof.** Promote successful narrow artifacts into the
   30-day direct-files/ripgrep/small-SQLite comparison and export/restore drill.
   Surface expansion waits for the predeclared continuation gate.

Drop in this order when time, integration capacity, or evidence fails:

1. Preact/Vite or other Web rewrite/scaffold work and visual polish.
2. Broad coordination telemetry/write legs, public `planned_work` MCP exposure,
   and any conductor-like packet/state machinery.
3. Pairwise proof generation beyond mandatory cells; `x35k`/`e5b5`; broad
   usage/cost expansion beyond the already-owned `f2qv.2` work.
4. Session-to-commit/D1 specimen and `hjwr`/`g9f2` implementation; retain as
   execution-grade future Beads, not current campaign requirements.
5. Full Receipts labeling if the pilot/protocol/privacy gate fails; publish the
   census/methodological negative instead.
6. Broad `1hj`/context/assertion/web/evaluation expansion if the thin
   coordination/value experiment does not beat simpler baselines.

Never drop raw evidence integrity, source compatibility, lineage/fidelity,
exact/structured retrieval, stable evidence refs, export/restore, or honest
degradation. Those are the narrow core under both strategies.

## Uncertainties

- The index file reports v29 while the rebuild service is still activating;
  completeness, convergence debt, FTS integrity, and ingestion freshness remain
  unverified.
- The resume fallback, `kj22`, `v6vy`, and `d1y` states may change after this
  note. The worktree observations above are snapshots, not completion claims.
- The GPT falsification follow-up may amend the NARROW verdict after reading the
  missing notes. No returned amendment was available locally.
- Census v0 recognizes only a proxy family of English test-success claims and
  does not yet verify claim/verifier family correspondence or verifier outcome.
- The `planned_work` content-cursor design is strong but unimplemented and has
  unresolved normalization/canonical-JSON stability risks.
- No comparative evidence yet proves that Polylogue's coordination/assertion/
  context/web layers outperform direct files, ripgrep, small SQLite, or native
  provider history.
- The exact safety/product decision for `ptx` auto-submit versus draft-only has
  not been ratified in the tracker.
- Live resource figures for the stale daemon came from the controlling session's
  runtime observation; this synthesis did not independently reproduce the
  historical IO/RAM totals.
