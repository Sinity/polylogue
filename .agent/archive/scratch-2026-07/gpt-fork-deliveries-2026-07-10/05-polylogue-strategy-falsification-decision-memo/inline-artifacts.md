## Turn 22 (role=assistant) — original decision memo answering the initial ask, before new July-10 evidence was supplied: verdict NARROW with a hard 30-day stop gate

Completed against the attached falsification brief. fileciteturn0file0

## Verdict: **NARROW, with a hard 30-day stop gate**

Polylogue should not continue its current broad roadmap. The repository demonstrates a technically credible **local, cross-provider, lineage-aware evidence ledger**: it preserves raw source evidence, normalizes heterogeneous sessions into queryable messages and actions, reconciles divergent histories, and can produce durable references back to the originating material. That foundation is supported by `CLAUDE.md:3-7`, `polylogue/storage/source.py:1-33`, `polylogue/storage/index.py:106-210`, `polylogue/storage/index.py:423-552`, and `.agent/demos/lineage-validation/current/README.md:5-15`.

The attachments do **not** demonstrate that the broader system—assertion memory, context injection, embeddings, coordination, evaluation hosting, execution-adjacent planning, or the evidence cockpit—delivers enough recurring user value to justify its opportunity cost. The honesty audit reports an empty assertion store and no durable MCP/context-use telemetry, while the embedding demo produced only two embedded records out of 43 and no semantic retrieval results. Those are evidence of unproven adoption and value, not merely unfinished code.

The essential product boundary is:

- Keep bounded source acquisition, immutable/raw evidence, normalized sessions/messages/blocks/actions, lineage, exact and structured search, stable evidence references, fidelity manifests, and two or three portable evidence exports.
- Keep a compact CLI and MCP/API read surface. The affordance review was verified: it recommends **3 MCP retire/retire-candidates, 56 MCP keep, and all 34 CLI surfaces keep**, not mass deletion (`.agent/scratch/2026-07-09-affordance-usage-review.md:178-190`; Beads `polylogue-v6vy`, `polylogue-moyt`).
- Retain only the minimal daemon needed to serve the core.
- Freeze active expansion under `polylogue-37t`, `polylogue-s7ae`, `polylogue-rii`, `polylogue-bby`, `polylogue-mhx`, `polylogue-9l5`, `polylogue-rxdo`, broad `polylogue-3tl` work, and nonessential `polylogue-fs1` breadth.
- Close or re-scope evaluation hosting (`polylogue-stc`, `polylogue-wnse`), the dependency-heavy web reporting loop (`polylogue-bby.15`), coordination messaging/scheduling (`polylogue-s7ae.5`), assertion-production work presented as adoption proof (`polylogue-mrxt`), and execution or remote-control programs.

The smallest dependency closure capable of testing the thesis is **`polylogue-xyel` alone**: it has no declared open dependencies and consumes the already-closed `polylogue-212.7` contract. It should be re-scoped from a one-PR demonstration into a comparative receipts benchmark rather than pulling in `polylogue-212.2` or another six-dependency proof stack.

The 30-day portfolio is limited to five outcomes:

1. A preregistered 24-task comparison of Polylogue against direct provider files, ripgrep, and a small SQLite implementation.
2. A 20-claim receipts benchmark through `polylogue-xyel`, spanning at least two origins.
3. A raw-evidence, lineage, idempotence, and per-origin fidelity gate.
4. A deliberately thin two-runtime coordination falsification using only already-shipped read-only projections.
5. An export, clean restore, answer-reproduction, and stop drill.

Continue the narrow core only if it reaches at least 90% correctness on cross-provider/provenance tasks and produces either a 15-percentage-point accuracy gain or a 2× median-time advantage over the baseline. At least 95% of supported claims must resolve to raw evidence, every unsupported claim must be caveated, and maintenance work must remain under 30% of the portfolio.

Stop new feature development immediately for correctness below 80%, raw-evidence resolution below 90%, any known uncaveated unsupported structural claim, or an export/restore failure. Also stop when a small direct-source SQLite implementation comes within 10 percentage points of accuracy and 25% of completion time, or parser and infrastructure repair consumes more than half of the month.

The strongest argument against narrowing is that the archive, assertions, context compiler, coordination envelope, and evaluation artifacts may form a compounding system whose value cannot be measured component by component. That remains plausible. The package, however, contains neither live usage telemetry nor repeated independent use capable of distinguishing an emerging flywheel from mechanisms built ahead of demand.

Material unknowns include the two requested July 10 strategy/dialogue scratch files, which were not present in the supplied package; any definition of `WorkflowProofSpec`; a current live Hermes database; the live production archive itself; independent-user behavior; and current external competitor capabilities. All claims about provider-native systems, generic observability, Hermes as a runtime, and evaluation frameworks therefore remain `UNVERIFIED_EXTERNAL`.

[Download the complete decision memo](sandbox:/mnt/data/polylogue-strategy-falsification-decision-memo.md)

---

## Turn 39 (role=assistant) — bash heredoc that wrote the full amendment memo (~19KB) to /mnt/data/polylogue-strategy-falsification-amendment-2026-07-10.md -- the amendment document's full text as authored, before the model's turn-41 restatement

bash -lc cat > /mnt/data/polylogue-strategy-falsification-amendment-2026-07-10.md <<'EOF'
# Amendment to the Polylogue strategy falsification decision memo

**Date:** 2026-07-10  
**Supersedes:** only the specific findings identified below; the original memo otherwise remains in force.  
**New evidence:** `.agent/scratch/2026-07-10-agent-dialogue.md` and `.agent/scratch/2026-07-10-broad-project-strategy-and-verifiability.md`, both newer than the Chisel package.

## Amended verdict

**NARROW remains the verdict, but the coordination freeze is narrowed.** Polylogue still has not earned the broad archive→memory→coordination→evaluation roadmap, and it still must not become an execution runtime. The July 10 evidence does, however, make one deliberately thin `s7ae`/`1hj` delivered-agent-room experiment worth funding for seven days. This is not a pivot decision and not continuation of `37t.11` or the full `s7ae.3` design. It is a bounded test of whether the existing evidence graph, blackboard storage, compact work projections, and an external runtime can prevent a duplicate/conflicting action or repair a handoff. The external runtime launches and executes; Beads remains the sole work authority; Polylogue supplies expiring evidence and records message/proof receipts. The experiment is load-bearing for the **control-plane option**, not for the archive-core thesis.

The new notes also remove two former unknowns: `WorkflowProofSpec` now has a concrete anti-vacuity design, and a real Hermes v11 artifact was probed. They do not establish adoption or product value. The Receipts eligibility census establishes a large usable sample frame—1,912 capped Claude claims and 223 capped Codex claims—but verifier outcomes and comparative user benefit remain unmeasured.

## Evidence interpretation: absence versus negative evidence

| Capability | Current evidence | Decision |
|---|---|---|
| Assertions and context memory | An empty assertion store and absent durable resume-use telemetry show **no adoption/value evidence yet**. They do not show that a bounded experiment cannot work. The n=5 context-pack result remains too weak and includes a stale-pack false claim. | Preserve compatibility; keep `37t.11`, auto-injection, candidate generation, and the broader memory flywheel frozen. Do not use the empty store as proof of worthlessness. |
| Receipts | The v0 census found 4,450 Claude and 2,713 Codex proxy-eligible claims before the two-per-session cap; the cap leaves 1,912 and 223. Duplicate `(session_id, tool_id)` ambiguity affects only 3 of 1,735 claim sessions. | Fund the bounded benchmark. This is positive feasibility evidence, not yet value evidence. Exclude and disclose the three ambiguous sessions rather than backporting a new pairing query. Beads: `polylogue-xyel`, `polylogue-212.2`, `polylogue-212.7`, `polylogue-3tl.16`. |
| Coordination | Durable blackboard notes and the typed envelope already exist (`polylogue/archive/blackboard.py:1-8,33-57`; `polylogue/storage/sqlite/archive_tiers/user_write.py:666-697`; `polylogue/coordination/payloads.py:215-234`). External execution is available. | This is enough substrate for one bounded behavior experiment. It is not evidence that the full program is valuable. |
| Existing coordination implementation | Current named views are 51–64 KiB. `build_coordination_envelope` constructs all evidence arrays before projection, and `project_coordination_envelope` removes only a few fields (`polylogue/coordination/envelope.py:61-118,121-163`). `s7ae.5` still requires a retired `.agent/conductor-devloop/*` packet. | Affirmative evidence **against the current implementation path** as an agent dispatch/control surface. Repair projection semantics first; do not complete the generalized scheduler/advisory stack. |
| Conductor-style control | Repo doctrine says Beads is the devloop and explicitly forbids resurrecting conductor packets or `devloop-*` scripts (`.agent/CONVENTIONS.md:9-13,36-42`). | Hard stop boundary: no second task state, queue, active-loop file, packet directory, mode machine, or Polylogue-owned dispatch lifecycle. |
| Verification-gap compiler | The current closure checker validates path/list shape, not collection, execution, coverage, or behavioral authority (`devtools/verify_closure_matrix.py:1-17,57-105`). Workflows declare multiple surfaces, while executable golden paths are CLI commands (`polylogue/product/workflows.py:14,25-57,320-484`). | A small `WorkflowProofSpec` compiler is justified as a **trust instrument**, not a product-value proof. It must prove anti-vacuity with real authority witnesses. |
| Hermes/Web witnesses | The Hermes fixture creates a synthetic v17 superset (`tests/unit/sources/test_parsers_local_agent.py:21-90`) while the parser requires the complete column superset and assumes `active` (`polylogue/sources/parsers/hermes_state.py:28-76,129-140,167-176`). The current Web tests are browserless while the deployed reader has measured oversized responses. | Use one Hermes v11 structural fixture and one current-UI browser journey to test proof authority. Do not turn either into a product expansion lane this week. |

## Changes to the first memo

### Withdraw

- Withdraw “`WorkflowProofSpec` is unavailable/unknown.” Its design is now sufficiently concrete to test, though no implementation is attached.
- Withdraw “Hermes current state is unavailable.” The July 10 audit records a real v11 probe and a reproducible incompatibility. Raw private content is still unavailable.
- Withdraw “`polylogue-xyel` alone, as written, is the smallest proof.” Its PR→authoring-session premise inherits the broken `session_commits` path described under `polylogue-212.2`; it must be re-scoped to position-aligned claims/actions/outcomes.
- Withdraw “the shipped `s7ae.1`/`.4` views are sufficient for the thin trial.” Their measured payloads are not agent-grade projections.
- Withdraw the recommendation to execute `polylogue-moyt` now. Only `polylogue-v6vy` is a proven exact duplicate; `moyt` remains pending parity proof.

### Narrow

- Narrow “freeze coordination” to: freeze the broad `s7ae.2`, `s7ae.3`, `37t.11`, `d1y`, `rii`, and public-MCP expansion path, while funding one internal/experimental `s7ae`/`1hj` delivered-room falsification.
- Narrow the assertion-store finding from “directional failure” to “no adoption evidence.” Expansion still lacks justification because telemetry and repeat use are absent.
- Narrow the execution-adjacent planning prohibition: permit one generated, read-only, expiring `planned_work` projection consumed by an external adapter. It may not assign, schedule, launch, merge, or become public MCP before the live proof.
- Narrow Hermes and browser work to anti-vacuity/authority witnesses. They are engineering trust proofs, not evidence of market or user value.

### Retain unchanged

- Retain **NARROW**, the archive/search/provenance core, the direct-source/ripgrep/small-SQLite comparison, the 30-day stop thresholds, and the export/restore stop drill.
- Retain the hard prohibition on an execution runtime or conductor replacement.
- Retain freezes on embeddings, evaluation hosting, Web cockpit/v2 expansion, automatic context injection, broad analytics/object models, and new source breadth unrelated to a scored task.
- Retain the manual affordance result: 56 MCP tools and all 34 CLI commands remain; no mass-retirement campaign.

## Mandatory Bead surgery before dispatch

The coordinator is the sole Beads writer. No experiment worktree opens until these edits are reviewable and `.agent/scripts/bd-graph-lint` is clean.

1. **`polylogue-s7ae.1`: keep closed.** Create one successor child under `polylogue-s7ae`, related to `s7ae.1`, for explicit per-view allowlists, ref-only/detail policy, composite freshness cursor, stale-cursor rejection, and an 8 KiB serialized budget. Make it the hard prerequisite of the revised live proof. This is a successor requirement, not a rewrite of honestly completed row-count acceptance.

2. **`polylogue-1hj`: amend in place to a v1 delivered-room slice.** Scope to repo/work-item/direct recipients, TTL, a cap of three messages, and one `coordination_ack`/delivery receipt per recipient session. Delivery occurs through an external launch/start adapter or explicit experimental CLI pull; no Web board, no mid-session advisory system, no generalized context scheduler, and no task lifecycle. Agent-authored messages remain quoted peer data, never operator authority.

3. **`polylogue-s7ae.5`: replace the stale acceptance criteria.** Remove `.agent/conductor-devloop/*`, `37t.11` injection, `pj8`, `d1y`, and the full `s7ae.3` as hard prerequisites. Require only the compact-projection successor, the narrowed `1hj` delivery slice, existing archive evidence from `s7ae.4`, two external agents in separate worktrees, one message/ack, and returned proof refs. The handoff is a bounded ref set in the existing envelope, not a packet directory.

4. **`polylogue-s7ae.3`: leave open but frozen.** Reverse the sequencing assumption: the broad scheduler-mediated advisory program may be reconsidered only after `s7ae.5` succeeds; it is not built in order to make the proof pass. `polylogue-37t.11` remains frozen and is not a seven-day dependency.

5. **Receipts:** amend `polylogue-xyel` to implement position-aligned Receipts v1 under the closed `polylogue-212.7` packet contract. Amend `polylogue-212.2` so PR-authoring archaeology is a later typed-edge experiment, not a premise of the first receipt. Relate the result to `polylogue-3tl.16`; ledger it as `capability` until protocol, label, privacy, and audit gates pass.

6. **Proof authority:** create one successor child under `polylogue-9e5`, related to `polylogue-t8t`, for the mandatory-only `WorkflowProofSpec`/`verify gaps` compiler. Do not create a new root program. Create a successor under `polylogue-fs1`, related to closed `fs1.1`, for observed-v11 recognition plus the retained unsupported-future-schema fixture. Amend `polylogue-1ilk` to test the current UI first; defer Preact/Vite and visual-regression breadth.

7. **Existing ownership:** land or resolve `polylogue-f2qv.2` with its current owner; do not duplicate it. Record the final `polylogue-k6fm` cloud canary refs in its Bead before treating it as closed. `polylogue-kj22` and `polylogue-v6vy` are optional hygiene only; `polylogue-moyt` is deferred.

## Strict seven-day portfolio

There are exactly three experiments. Day 1 surgery and Day 7 adjudication are mandatory campaign work, not additional product lanes.

### Experiment A — Receipts plus a direct-source value test

Freeze claim patterns, verifier-family mapping, validity windows, session cap, and duplicate-tool-id exclusion before reading outcomes. Produce 60 Claude Code and 60 Codex packets, with no more than two claims per session. Two different labeler families each cover both origins; the operator adjudicates disagreements and audits 24 agreement cases. In parallel, run 12 blinded paired audit tasks using Polylogue packets versus provider files/ripgrep/a small SQLite query.

**Success:** at least 95% of supported labels resolve to raw structural evidence; every unsupported/unknown case is caveated; no privacy leak; no more than two of the 24 agreement-audit cases are wrong and no systematic rubric failure occurs. Product value is positive only if the paired task shows either a 15-point correctness advantage or a 2× median-time advantage with no more than a five-point correctness loss.

**Stop:** any protocol change after outcome inspection, any private-data leak, more than 10% unresolved because Polylogue lacks the needed structural evidence, or audit invalidation. If the full 120-case publication becomes invalid, publish the census and methodological negative; do not silently shrink, unbalance, or relabel the estimand.

### Experiment B — Delivered agent room through `s7ae`/`1hj`

Run three paired scenarios with two external agents in separate worktrees: overlapping edit scope, duplicate work intent, and handoff with one hidden constraint. Controls use Beads plus ordinary launch prompts without Polylogue room evidence. Treatment uses the compact experimental projection and one scoped message delivered at launch/start, with a durable ack and returned proof refs. Codex/Claude/Hermes remain the executors.

**Success:** serialized treatment projection is at most 8 KiB; any changed repo/Beads epoch invalidates it; unknown owner/proof/exclusivity fields never authorize action; message delivery and ack are queryable; treatment changes the correct behavior before damage in at least two of three scenarios; there are zero material false blocks; added operator overhead is at most ten minutes per scenario.

**Stop and freeze coordination:** any second task authority or durable mission state is required; any stale packet is accepted; any unknown field authorizes work; a message is presented as operator authority; the treatment fails to change behavior in two scenarios; or implementation exceeds two worker-days before the first paired run. On failure, freeze `1hj`, `s7ae.3`, the revised `s7ae.5`, `37t.11` integration, and public `planned_work` registration.

### Experiment C — Anti-vacuity proof-gap compiler

Implement mandatory obligations only, capped at 40 generated cases. It reads the existing workflow/surface registry and requires either an executable proof or an open gap Bead. Use four seeded failures: remove the Hermes observed-v11 proof, remove the current-Web browser journey, name a nodeid that does not collect/run in its lane, and declare a supported surface with neither proof nor gap. Pairwise generation, learned prioritization, and telemetry write legs are out of scope.

**Success:** all four seeded failures turn the report red; restoring them turns it green; the report never calls the existing path-only closure matrix sufficient; default execution completes within 60 seconds on the campaign fixture.

**Stop:** the implementation reduces to another path/manifest lint, cannot distinguish a collecting test from an executed proof, or consumes more than 1.5 worker-days. Failure of this experiment drops the compiler; it does not invalidate Receipts evidence.

## Schedule, concurrency, and drop order

**Day 1:** Bead surgery, protocol freeze, exact file ownership, resource classes, and graph lint. No implementation before this is complete.  
**Day 2:** produce 12 leak-free Receipts packets; an ≤8 KiB projection with stale-cursor rejection; one synthetic message/ack; and a red `verify gaps` result for missing Hermes/Web authority. Each experiment must pass its own prerequisite; at least three of the four campaign checks must exist to continue parallel execution.  
**Days 3–5:** run the full Receipts extraction/labels, the three coordination pairs, and the minimum compiler.  
**Day 6:** operator audit, direct-source paired comparison, anti-vacuity mutations, and reruns.  
**Day 7:** cold-reader packet, archived token/cost-equivalence per landed verified outcome, Bead closure/freeze notes, and the narrow/pivot/stop decision.

Resource assumptions: one coordinator/integrator and sole Beads writer; one local private-data lane; at most two cloud/synthetic implementation lanes after the proven canary; no more than three open branches; no replacement task until a returned branch is reviewed; one heavy DB/full-suite/live-browser operation at a time; deployed v24 is read-only; new-schema work uses scratch v29; no live rebuild. The host assumptions remain 24 hardware threads, 32 GiB RAM, and observed test peaks around 10.4 GiB RSS.

Drop order, first to last:

1. Drop `v6vy`, `kj22`, all `moyt` work, testmon experiments, Web/Hermes repairs beyond the two authority witnesses, compiler telemetry, and pairwise generation.
2. Drop compiler polish and any second browser journey; retain only the four anti-vacuity checks.
3. If the projection or message/ack prerequisites miss Day 2, drop the entire coordination experiment and record a freeze decision—do not build scheduler or conductor scaffolding to rescue it.
4. If Receipts validity fails, drop full publication and emit the methodological negative; retain the 12-case paired core-value test only if it remains blinded and valid.
5. Never drop the direct-source comparison, raw-evidence/privacy gate, or Day 7 scope decision. If those cannot run, the campaign result is **no earned continuation**, not permission for more implementation.

## Seven-day decision thresholds

- **Continue the narrow core:** Experiment A clears its trust gates and its paired baseline value threshold. The original 30-day comparison and export/restore gates still govern longer funding.
- **Earn a 30-day evidence/control-plane option:** the narrow-core condition passes **and** Experiment B passes. This authorizes only another bounded external-runtime trial; it does not unfreeze the broad roadmap or public MCP planning surface.
- **Narrow further:** Receipts/provenance is valid but the room experiment fails. Freeze coordination/context expansion and keep archive, lineage, refs, search, fidelity, and evidence exports.
- **Stop new feature development:** any hard trust/privacy failure; raw-evidence resolution below 90%; a known unsupported claim emitted without caveat; the direct-source baseline comes within ten points of accuracy and 25% of completion time without repeated preference for Polylogue; or more than half of campaign effort is consumed by parser/infrastructure repair. Preserve/export the archive as specified in the original memo.

## Unsupported or still unknown

- The operator reports that the Codex Cloud canary now completed launch/recover/verify/merge. The attached notes prove bootstrap PR #2631 at `c68585b8b`, environment setup, CLI launch/recovery commands, and creation of the `polylogue-k6fm` task; they do **not** include the canary's final diff, verification run, PR, or merge artifact. Treat completion as operator-supplied current evidence until those refs are recorded in `polylogue-k6fm`.
- The Receipts v0 census is a proxy over English test-success phrases. Claim-family correspondence, verifier outcomes, contradiction prevalence, final classifier accuracy, and product-value comparison remain unknown.
- No independent-user adoption, retention, task frequency, or willingness-to-pay evidence is attached.
- Whether the delivered room changes behavior is unknown; that is the experiment's question.
- `WorkflowProofSpec` is a design, not an implemented or validated compiler.
- Availability of a second independent labeler family and sufficient operator time for the 24-case audit is unknown.
- Current repository/Bead/PR state after the July 10 notes is unknown except where the operator has supplied a newer status.
- Claims about current provider-native history, observability platforms, Hermes runtime breadth, or evaluation frameworks remain `UNVERIFIED_EXTERNAL`.
EOF
wc -l /mnt/data/polylogue-strategy-falsification-amendment-2026-07-10.md


---

