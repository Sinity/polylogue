# Decisions

## Decision summary

The campaign program will maximize construct validity and report value per operator hour, not raw label count. It will use Polylogue’s immutable schema/batch/assertion substrate, keep structural observations authoritative, label only irreducible semantic questions, and publish no accepted aggregate until exact-context calibration and join diagnostics pass.

## Ranked construct order

### D1 — Launch failure acknowledgment first

**Decision:** Adopt `failure.acknowledgment@v1` as campaign 1, targeting structurally failed tool-result blocks.

**Reasons:**

- It directly supports the outreach question: whether agent prose acknowledges or proceeds past structurally observed failure.
- Failure membership is already anchored in normalized structural fields, so judges do not decide whether the tool failed.
- The judgment window is bounded and cheap.
- The current 5,000-row report leaves 3,375 immediate follow-ups ambiguous under marker rules; annotation addresses the dominant uncertainty.
- Fifty existing human labels provide rubric anchors and known counterexamples, though not an unbiased prevalence sample.
- The construct can be falsified and calibrated with per-class confusion, unlike a broad “quality” score.

**Report unlocked:** Weighted three-class failure-follow-up prevalence and uncertainty by origin/model/tool/era/length, plus next-three-turn sensitivity and marker precision/recall.

### D2 — Run task-completion-versus-claimed second

**Decision:** Adopt as campaign 2 after campaign 1 proves the pipeline.

**Reasons:**

- It expands the credibility result from local action failure to assistant claims of task completion.
- Existing completion receipts already identify lexical claims and structural support/contradiction classes.
- Semantic scope and sufficiency need judgment, but structural outcomes must remain higher-authority evidence.
- It requires longer context and a more complex rubric, so it is a poor first operational test.

**Report unlocked:** Claimed-complete statements classified as structurally supported, contradicted/repaired, contradicted/unrepaired, or unresolved, split by claim scope and provider/model.

### D3 — Repair terminal-state materialization before auditing it

**Decision:** Do not mass-annotate current terminal states. Fix `polylogue-vhjs` and `polylogue-wofr`, rebuild the derived index, then label a stratified validation sample.

**Reasons:**

- Current live evidence cited by `vhjs` says all 8,507 non-null labels lack `terminal_state_method`.
- Current live evidence cited by `wofr` says all 1,575 bounded-large profiles are terminal-state unknown.
- The DDL contains method provenance, while current profile/write code drops it; the bounded path deliberately writes unknown.
- Structural outcome is the production authority. Human/agent labels should estimate residual error after deterministic repair, not become a shadow terminal-state store.
- Process termination and objective posture are orthogonal; “clean finish” cannot imply objective completion.

**Report unlocked after repair:** Outcome-conditioned cost/duration/retry/tool analytics and a method/evidence coverage audit including marathon sessions.

### D4 — Validate pathology detectors rather than replace them

**Decision:** Sample detector positives, hard negatives, and random negatives; do not label every session for pathology.

**Reasons:**

- Current pathology rules are deterministic, versioned, and intentionally LLM-free.
- Current write code already emits candidate pathology assertions, despite a stale comment implying otherwise.
- Annotation value lies in estimating precision/recall and discovering systematic rule errors.

**Report unlocked:** Detector-version validity, prevalence, false-positive clusters, missed-pattern bounds, and provider/model coverage.

### D5 — Split session quality into comparative dimensions

**Decision:** Defer generic session-quality/derailment labels. Define dimensions and use blinded pairwise judgments with three independent judges.

**Reasons:**

- “Quality” combines several incompatible constructs.
- Absolute scales invite construct drift and judge sycophancy.
- The repository already has comparative judgment, blinding, calibration, and cascade mechanisms suited to this problem.
- Pairwise dimensions can expose ties and incomparability rather than force a false scalar.

**Report unlocked:** Dimension-specific rankings with uncertainty, disagreement, tie, and incomparability rates.

### D6 — Fix title authority before title QA

**Decision:** Treat `polylogue-ih67` as an ingest/authority defect. Perform only a small post-fix canary annotation.

**Reasons:**

- The reported 3,101 Codex UUID-like titles arise because the canonical daemon route bypasses title assembly.
- Labels cannot repair a route that never produced the intended title.
- A canary is useful only after source authority and backfill are corrected.

**Report unlocked after repair:** Title-source coverage, UUID residuals, generated-title acceptance, and retrieval quality/lift.

## Operational decisions

### D7 — Distinguish scheduling shards from durable batches

**Decision:** Schedule work in 100-item manifests, but import one target/judge label row per durable batch.

**Basis:** Current `AnnotationBatchImportRequest` has a batch-wide target/actor/model/prompt; JSONL rows have no target. All row validation and assertion identities use the request target.

**Consequence:** Campaign 1’s 600 items × two base contexts produce 1,200 base `annotation_batches`, not 12. The six 100-item shards are external campaign receipts referenced from batch metadata.

**Rejected alternative:** One durable batch targeting a result-set and embedding per-item targets inside label JSON. This would break exact target joins and misstate assertion grain.

### D8 — Use one complete schema row per target/judge

**Decision:** Each current durable batch carries exactly one valid annotation row under campaign convention, even though the importer permits multiple rows sharing the same target.

**Reason:** It makes retries, disagreement, lineage, counts, and one-target joins unambiguous. Distinct judges and adjudicators use distinct batches.

### D9 — Keep agent outputs candidate-only until adjudication

**Decision:** Base and third-judge imports remain candidate assertions. Consensus/adjudication creates a separate durable decision; publication queries accepted/active labels only.

**Rejected alternative:** Majority-vote overwrite or deletion of disagreeing candidates. That destroys provenance and hides correlated errors.

### D10 — Use two genuinely independent base judges

**Decision:** Two different model families or immutable checkpoints judge every low-context categorical item; a third handles escalation.

**Reason:** Same-checkpoint prompt or temperature variants are robustness probes, not independent votes.

### D11 — Use 72 hidden gold items in the 600-item pilot

**Decision:** Twelve gold items per 100-item shard, approximately 30 Claude Code, 30 Codex, and 12 Claude.ai.

**Gold source:** The 50 committed human labels are anchor candidates. Add at least 22 fresh, double-reviewed cases in missing cells. Existing labels count as hidden only in a no-tool sandbox with remapped IDs and no access to committed labels; otherwise produce 72 fresh hidden cases.

**Reason:** This gives each exact base judge context more than the recommended 30-gold minimum and lets drift be measured per shard.

### D12 — Sample prevalence independently of the marker classifier

**Decision:** Select the 600-item prevalence cohort from the structural-failure frame by origin/model/era/length, not by predicted acknowledgment class.

**Reason:** The marker is under evaluation. Predicted-class stratification without weights creates selection bias. Marker-balanced rows may be used in a separate calibration cohort.

### D13 — Use balanced-head plus thin-origin census

**Decision:** Census origins with fewer than 60 eligible items; split the residual quota evenly across the two largest origins; then stratify era, length, and model head/tail.

**Illustrative current allocation:** 49 Claude.ai, 276 Claude Code, 275 Codex.

**Reporting rule:** Use inverse-probability weights for overall prevalence. Show unweighted balanced-origin comparisons only as comparisons.

### D14 — Cap within-session clustering

**Decision:** Sample no more than three failures per session and publish a one-failure-per-session sensitivity result.

**Reason:** Marathon sessions can dominate event-level estimates and violate independent-item assumptions.

### D15 — Escalate uncertainty rather than forcing labels

**Decision:** Third-judge escalation occurs on disagreement, abstention, confidence below 0.70, truncation/ref failure, canary hit, unknown calibration, or a predetermined high-consequence slice. Operator review follows unresolved third-judge cases, gold misses, rubric contradictions, and random audits.

**Reason:** Ambiguity is a measurable property of the evidence, not a formatting failure.

## Calibration decisions

### D16 — Calibrate exact execution contexts

**Decision:** Never pool calibration across different `(actor_ref, execution_context_id, construct/dimension)` contexts.

**Basis:** Current judgment calibration explicitly stratifies by exact actor and execution context. Prompt, model, runtime, parameters, tool policy, rubric, and evidence template all contribute to context identity.

### D17 — Apply campaign-specific multiclass gates

**Decision:** Use the following release thresholds for campaign 1:

- at least 30 gold overlaps per exact base context;
- accuracy ≥ 0.85;
- macro-F1 ≥ 0.80;
- consequential `silent_proceed` recall ≥ 0.90;
- base-judge raw agreement ≥ 0.80;
- Cohen’s kappa ≥ 0.70;
- per-origin accuracy gap ≤ 0.10 where each origin has at least 20 gold;
- abstention ≤ 10% overall and ≤ 20% in each reported stratum;
- escalation ≤ 30%;
- zero schema-drift, invalid-row, missing-target, or ambiguous-target diagnostics.

These are ann-03 recommendations. The current comparative calibration module does not calculate multiclass annotation confusion, macro-F1, or kappa.

### D18 — Treat calibration failure as a stop, not a weighting trick

**Decision:** Do not publish an aggregate by merely down-weighting a judge context that fails calibration. Pause the affected cell/context, repair the rubric/evidence or change the judge, create a new execution context, and re-run the affected items.

## Provenance decisions

### D19 — Encode exact model and prompt identity in refs and metadata

**Decision:** Use immutable-digest-bearing `model_ref` and `prompt_ref`, and duplicate all exact identities in metadata.

**Reason:** Current durable columns include `actor_ref`, `model_ref`, and `prompt_ref`, but do not separately pin model version/digest, runtime version, prompt SHA, rubric SHA, or execution-context ID.

Required conventions:

- `actor_ref = agent:<stable-harness>`;
- `model_ref = agent:<runtime>/<model>@sha256:<digest>`;
- `prompt_ref = analysis:sha256:<prompt_sha>`;
- `source_result_ref = result-set:sha256:<packet_sha>`;
- metadata contains context, runtime/model/prompt/rubric/evidence/output hashes and sampling lineage.

### D20 — Make the source frame and every evidence packet content-addressed

**Decision:** Freeze a frame manifest with archive as-of/cursor, query hash, code commit, sampler version, seed, frame cell counts, selected IDs, and inclusion probabilities. Each item packet receives its own SHA.

**Reason:** A batch can be immutable while still being irreproducible if the evidence bytes and sampling frame are not pinned.

### D21 — Exact retries reuse identity; changed runs do not

**Decision:** Use the same batch ID only for byte/semantic-identical normalized provenance and output. Any changed stochastic output, model, prompt, parameters, runtime, evidence, or schema gets a new context/attempt and batch ID.

**Basis:** Current durable storage compares canonical batch provenance and rejects incompatible reuse.

### D22 — Evolve schema by new version only

**Decision:** Freeze v1. Register v2 for any label/rubric shape change. Keep qualified schema IDs in reports and use an explicit versioned crosswalk/bridge sample.

**Current limitation:** Schema status is part of the canonical definition, so in-place active→deprecated mutation is also drift. Deprecation must remain policy or move to a future separate mechanism.

## Cost decisions

### D23 — Use local judges for the first campaign

**Decision:** Prefer two local Ollama contexts with tools/network/repository access disabled. Use API/subscription lanes only for a deliberately different-model robustness check or when local calibration fails.

**Reason:** Local execution protects private transcripts, makes hidden-gold isolation easier, and has negligible marginal API cost. The campaign’s main cost is operator adjudication, not token billing.

### D24 — Record actual cost telemetry

**Decision:** Every call logs input/output token estimates or measured counts, elapsed time, retries, throttle/rejection, model digest, and energy/runtime lane where available.

**Reason:** Planning costs are scenarios. The first campaign should replace them with observed medians and tail latency before campaign 2.

## ann-02 interface assumptions

ann-03 depends on ann-02 for the following artifacts. If an item is absent, the first run remains a candidate-only pilot and does not produce accepted population claims.

1. **Canonical construct package:** exact `failure.acknowledgment@v1` schema definition, rubric text, field semantics, examples/counterexamples, schema SHA, prompt SHA, and evidence-template version.
2. **Execution-context contract:** canonical JSON fields and hashing algorithm over prompt, rubric, runtime, model digest, parameters, tool policy, and evidence template.
3. **Trusted gold protocol:** who may create/adjudicate gold, blinding rules, disagreement handling, versioning, and leakage controls.
4. **Multiclass metrics:** confusion matrix, accuracy, macro-F1, per-class precision/recall, kappa/agreement, abstention, and per-stratum diagnostics by exact context.
5. **Promotion contract:** how independent candidate assertions and operator adjudication become accepted/active labels without deleting candidates.
6. **Calibration reference identity:** a stable result/analysis ref stored in each batch’s `ann02_calibration_ref` metadata.

The current judgment modules provide useful comparative-calibration, blinding, and cascade primitives, but ann-03 does not assume they already implement this multiclass annotation workflow.

## Adjudicated contradictions

| Apparent claim | Current evidence | Decision |
|---|---|---|
| `annotation_batches` pins exact judge model/version/prompt hash | Columns store actor/model/prompt refs and metadata, but no first-class version/digest/SHA/context fields and no required resolution of those refs | Encode immutable identities in refs and required metadata; audit completeness |
| A “batch” can be a 100-item campaign shard | Import rows have no target; request target is batch-wide | Operational shard and durable batch are different units |
| Current marker calibration is prevalence evidence | The 50 cases were selected by predicted class and are provider-skewed | Use as anchors/calibration, not an unbiased prevalence sample |
| Pathology assertion emission is future work | Current `user_write.py` already mirrors detector findings into pathology assertions | Validate current detector/write behavior; treat comment as stale |
| Terminal-state labels can be calibrated now | Method provenance is absent and bounded-large sessions are all unknown in cited live state | Repair/rebuild first, then annotate audit sample |
| Clean terminal state means task completed | `polylogue-37t.23` documents clean-but-unfinished sessions and authority ordering | Keep terminal process state and objective posture orthogonal |
| All-ref actor/execution-context commit is current | `672786a07` is not an ancestor of HEAD | Do not depend on its APIs or ref kinds |
| The snapshot contains an actionable dirty patch | Chisel reports dirty, but branch delta is empty and replayed tracked source showed no ordinary diff | Treat named HEAD/current source as authority; record dirty cause as unresolved |

## Non-decisions

This runbook does not select the actual local model checkpoints, because their availability and immutable digests were not present in the snapshot. It does not fix `vhjs`, `wofr`, or `ih67`; those are separate implementation jobs. It does not claim the July 4 failure frame is current on July 17. It does not promote any candidate label, publish a live rate, or assert live daemon/archive verification.
