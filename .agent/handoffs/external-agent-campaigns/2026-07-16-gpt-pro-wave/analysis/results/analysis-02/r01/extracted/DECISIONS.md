# Decisions for the next external-agent campaign

These decisions extend the repository’s existing external campaign workspace. They do not add campaign concepts to browser-extension, receiver, capture, or product-domain models.

## D1. Preserve distinct funnel entities

**Decision:** Reports and ledgers must keep action/run, provider conversation, provider turn, artifact, package revision, integration track, PR, merge, and Bead outcome as separate identities. Never force all stages onto a single package denominator.

**Owner:** `polylogue-yyvg.6`; `.agent/handoffs/external-agent-campaigns/`.

**Acceptance:** A report can represent 27 conversations, 26 package-bearing conversations, 28 package revisions, and 45 downloads without overwriting or silently reconciling any count. Every rate names its entity and denominator.

**Falsification:** A fixture that adds a same-chat package revision or a non-package proof conversation changes only the correct entity counts. If all stage counts change together, the model is still wrong.

## D2. Replace prose-only stage state with immutable evidence events

**Decision:** Advance `schemas/result.schema.json` to version 2 and add timestamped, source-attributed events. Keep per-attempt `result.json` immutable and `results/index.json` rebuildable, as the current workspace already specifies.

**Owner:** `polylogue-yyvg.6`; exact target `.agent/handoffs/external-agent-campaigns/schemas/result.schema.json`.

**Acceptance:** Submission, conversation binding, first/terminal turn, artifact offer/acquisition/hash validation, package validation, triage, worktree, apply check, repair, verification, PR, merge, and Bead-state transitions each carry `occurred_at`, `recorded_at`, `source_kind`, and `evidence_ref`. Unknown historical stages remain absent or explicitly unknown; import code does not synthesize timestamps.

**Falsification:** Rebuilding `results/index.json` from immutable receipts must reproduce campaign counts. Deleting any one fixture event must make the corresponding stage count fail.

`RESULT-TELEMETRY-V2.schema.json` is the exact proposed replacement shape.

## D3. Gate scale on canonical terminal-output custody

**Decision:** Do not expand a campaign wave while any prior attempt has a terminal assistant output with no canonical artifact outcome. Manual download is allowed only as an explicit degraded acquisition source, never silently treated as canonical capture success.

**Owner:** `polylogue-3v1` for capture completeness; `polylogue-ptx` for generic actions; `polylogue-yyvg.6` for orchestration.

**Acceptance:** For each submitted attempt, the ledger ends in one of: terminal with no artifact expected; artifact acquired and hash-valid; explicit provider/output failure; explicit irrecoverable historical gap. Offered artifact count equals acquired refs plus explicit failed/declined outcomes. A closed-tab live fixture and a replay fixture both pass through ordinary canonical capture.

**Falsification:** Seed a terminal output attachment, remove its captured ref or blob, and require dispatch gating and campaign completeness to fail.

## D4. Route by mission readiness, not by a universal “implementation package” template

**Decision:** Use three explicit mission shapes:

1. `implementation`: current source, one owning Bead, dependency-ready, named production route, bounded write/avoid set, apply-ready patch, real-route tests.
2. `repair`: same conversation or successor attempt, concrete failed receipt/review finding, supersedes lineage, cohesive replacement package.
3. `analysis_or_research`: cross-cutting decisions, dependency-premature work, architecture reconciliation, or external evidence; no pressure to manufacture a product patch.

**Owner:** the shared ChatGPT Pro contracts and workload `campaign.json` files.

**Acceptance:** `prompt_shape` is recorded. Implementation dispatch fails when the owner is closed/superseded without an explicit continuation, a dependency is unmet, the snapshot is stale, the allowed footprint is absent, or the production route is not named. Analysis/research output is evaluated on decision closure, not patch presence.

**Falsification:** Reclassify a known dependency-wait package as implementation; `check-dispatch.py` must block it. A broad analysis job must pass without a placeholder patch.

## D5. Measure package composition; do not impose a naive byte cap

**Decision:** Triage records compressed/uncompressed size, member count, patch bytes, test/verification bytes, design/metadata bytes, fixture/snapshot/archive bytes, copied-input detection, and largest members. Large evidence is permitted when declared and necessary. Undeclared bulk, copied project-state archives, padding, and synthetic fixtures presented as runtime proof fail triage.

**Owner:** `triage-package.py` and result schema.

**Acceptance:** The validator explains why each member above a configurable threshold exists. The exact supplied project-state archive or equivalent copied input is rejected. A 20 MiB necessary fixture can pass with a declaration and owning test; an 8 MiB irrelevant replacement/snapshot copy fails.

**Falsification:** Add a giant inert member without changing patch or tests; package quality metrics must not improve, and triage must flag it.

## D6. Make same-chat repair a first-class attempt and package revision

**Decision:** A retry or repair never overwrites an earlier result. It receives a new `attempt_id`, optional `package_revision`, `supersedes_attempt_id`, reason code, and explicit evidence of what failed.

**Owner:** `polylogue-yyvg.6`, prompt renderer, results importer.

**Acceptance:** The two prior same-chat package pairs backfill as distinct revisions linked to one conversation. A repair can be compared to its predecessor by artifact hash, touched paths, patch retention, review findings resolved, and eventual delivery outcome.

**Falsification:** Importing two artifacts from the same conversation with different hashes must not deduplicate them by conversation ID or overwrite the first receipt.

## D7. Keep local integration as an independent authority stage

**Decision:** Package self-verification can support triage but cannot publish work. A local implementer must reconcile current source and Beads, create a clean worktree, inspect the patch, record `git apply --check`, repair against current architecture, run production-route and anti-vacuity tests, and record a PR/merge/Bead receipt.

**Owner:** integration coordinator; `results/<job>/<attempt>/integration/`.

**Acceptance:** Every merged track has a clean-base identity, applied/retained/replaced patch record, local repair reason, exact test command/outcome, PR URL/number, merge commit, and current Bead state. A package with a competing seam or stale API is rejected or reimplemented rather than massaged into a nominal merge.

**Falsification:** Remove the production dependency or the claimed behavior; the anti-vacuity test must fail. Apply the package to a fresh current worktree; stale patches must be visible before review.

## D8. Pace generation by reconciliation and integration capacity

**Decision:** Use a small capture-and-ledger pilot before broad fan-out. After prerequisites freeze, external drafting may fan out, but active integration worktrees remain bounded. Do not launch a new wave while prior terminal/acquisition gaps or an untriaged queue exceed the available integration lanes.

**Owner:** `check-dispatch.py`, workload coordinator, `polylogue-yyvg.6`.

**Acceptance:** First launch 3–5 mixed attempts to prove action, terminal, output, and ledger events. Then expand to 8–12 dependency-disjoint implementation attempts only if every pilot attempt is terminally reconciled and triaged. Cap simultaneously active integration worktrees at the available 3–4 local lanes; the next wave waits until the triage queue is at or below that cap.

**Falsification:** Seed one unresolved terminal-output gap or an over-cap integration queue; dispatch must report the exact blocked gate.

## D9. Rebuild current delivery projection from immutable receipts

**Decision:** `INTEGRATION-LEDGER.json`-style package adjudication remains immutable evidence, but current delivery state must be rebuilt from result receipts, Git/PR history, and Beads. A package marked pending can later project as delivered without rewriting its original adjudication.

**Owner:** `polylogue-yyvg.6`; results index builder.

**Acceptance:** Backfilling PR #2953 changes the current CaptureJobs projection to delivered/closed while preserving package 24’s original “preserved draft awaiting reconciliation” decision. Contradictions are rendered, not overwritten.

**Falsification:** Rebuild after changing a Bead from open to closed; only the projection changes. The immutable package receipt hash and original decision remain byte-identical.

## D10. Preserve the generic product boundary

**Decision:** External campaign cadence, prompt fragments, package naming, mission state, triage, and integration stay under `.agent/handoffs/external-agent-campaigns/`. Browser product code exposes only provider-neutral BrowserAction and canonical capture behavior.

**Owner:** `polylogue-ptx`, `polylogue-3v1`, `polylogue-yyvg.6`.

**Acceptance:** No campaign/job/package-specific field, route, queue, or UI is added to browser-extension or receiver product contracts. External orchestration can submit a generic action and discover its conversation, turns, and files through canonical capture.

**Falsification:** Search product schemas/routes/UI for campaign/job/package-revision concepts. Any result outside generic evidence references fails the boundary.

## Explicitly rejected shortcuts

- Do not report “28 conversations → 28 packages → 4 merges”. The entity mapping is false.
- Do not treat `processing_state: processed` as delivered code.
- Do not infer completion or acquisition time from browser filenames.
- Do not use package size, patch count, or self-reported tests as a merge score.
- Do not revive the Sol-specific launch/work-package model removed by PR #2928.
- Do not close an owning Bead merely because one package slice merged.
- Do not claim historical output recovery until live replay produces evidence.
