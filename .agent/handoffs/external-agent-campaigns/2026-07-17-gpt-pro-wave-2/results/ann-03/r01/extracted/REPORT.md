# Mass-annotation campaign runbook

## Executive decision

Polylogue should not begin with a generic archive-wide “quality” labeling pass. The first campaign should label **failure acknowledgment at the structurally failed action grain**, because this is the narrowest prose-dependent construct that directly improves the outreach claim about claim-versus-evidence credibility, already has a structural sampling frame and 50 human anchor labels, and can be judged from a bounded evidence packet. The recommended first production-shaped pilot is 600 failed actions, two independent local judges on every item, a third judge on an expected 20% escalation set, and 72 hidden gold items. It produces 1,320 expected judge calls and, under the current importer, at least 1,200 durable `annotation_batches` before escalation and adjudication.

The annotation priority order is:

| Rank | Construct | Annotation decision | Report unlocked |
|---:|---|---|---|
| 1 | Failure acknowledgment after a structurally confirmed failed action | Launch first | Weighted acknowledged / silent-proceed / ambiguous rates by origin, model, tool, era, and session length; next-three-turn sensitivity; calibrated marker error analysis |
| 2 | Task completion versus claimed completion | Launch after campaign 1 | Claimed-complete statements partitioned into structurally supported, contradicted then repaired, contradicted without repair, and unresolved-evidence cases |
| 3 | Terminal-state / outcome audit | Repair `vhjs` and `wofr`, rebuild, then annotate an audit sample | Outcome-conditioned cost, duration, retries, and tools, plus a coverage audit for marathon sessions |
| 4 | Pathology detector validation | Validate detector positives and hard negatives, not the whole archive | Precision/recall and coverage report for deterministic wasted-loop and stale-context findings |
| 5 | Session quality / derailment | Defer until dimensions are split and comparative judgment is calibrated | Blinded pairwise quality ranking and dimension-specific derailment report |
| 6 | Title / topic quality | Fix `ih67`; use only a post-fix canary sample | Title authority, coverage, retrieval quality, and UUID-title residual report |

This is an annotation-return ranking, not an engineering-severity ranking. `polylogue-ih67`, `polylogue-vhjs`, and `polylogue-wofr` are real product defects and should be fixed promptly. They rank low as annotation campaigns because labels would conceal deterministic ingest/materialization errors rather than add irreducible semantic judgment.

## Snapshot identity and authority inspected

The supplied Chisel snapshot reports:

- repository: Polylogue, source path `/realm/project/polylogue`;
- branch: `master`;
- commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`;
- commit subject: `fix(repair): harden raw authority convergence (#3046)`;
- generated: `2026-07-17T180950Z`;
- recorded working-tree state: `dirty=true`;
- branch merge base: the same commit, with an empty branch-delta patch, log, and file list.

The recorded dirty flag cannot be attributed to a tracked source patch from the package: replaying the supplied working-tree archive over a clone of the bundled HEAD produced no ordinary tracked diff, while the snapshot separately reports large ignored and hidden state. The source at the named commit plus the supplied working-tree slice is therefore the code authority; the exact cause of `dirty=true` remains unverified.

The review covered the repository instructions, annotation schema/batch/import/join/write paths, action follow-up classifier, completion-claim receipt code, terminal-state materializers, pathology detector and assertion mirror, judgment calibration/blinding/cascade code, focused tests, relevant Beads records, current demo aggregates and calibration CSVs, and all-ref history. Important history anchors are `bf94704c0` (typed annotation foundation), `246c48d08` (durable provenance), `f4504cb4d` (JSONL import), `4ed0cf2dc` (exact joins), `ca76f2df1` (failure-marker calibration), `11615f99e` (analysis rigor), `866dab24d` (comparative judgment core), and `682b29bf3` (removal of prose terminal-state heuristics). All-ref commit `672786a07` declares additional actor/execution-context references but is not an ancestor of this snapshot’s HEAD, so this runbook does not depend on it.

Polylogue’s architectural constraint is decisive: `user.db` is durable and irreplaceable; annotation schemas and batch provenance are immutable durable records; labels remain ordinary assertion rows scoped to `annotation-batch:<id>`. New semantics belong in substrate/product code, not in a parallel campaign database.

## What the current substrate actually permits

### Three units must not be conflated

1. **Construct version** — an immutable `AnnotationSchema`, such as `failure.acknowledgment@v1`.
2. **Operational scheduling shard** — a campaign manifest of 100 items used for queueing, monitoring, and gold placement. This is a campaign artifact, not one durable annotation batch.
3. **Durable annotation batch** — one `AnnotationBatchImportRequest` with a batch-wide schema, target, actor, model, and prompt context.

The current importer’s JSONL row has `row_key`, `value`, `evidence_refs`, optional body text, and confidence. It does not have a per-row target. Every row is validated and written against `request.target_ref`, and assertion identity includes the same target, actor, and batch. Consequently, a 100-item operational shard cannot be represented as one current `annotation_batches` row without lying about target identity. The safe convention is **one label row per durable target/judge batch**, while the external 100-item shard manifest groups those batches operationally.

For the first campaign, 600 items × 2 base judge contexts means 1,200 durable batches. An expected 120 third-judge decisions add 120 more. Operator adjudication or promotion is a separate actor/batch lifecycle, not an overwrite of the agent candidates.

### Candidate labels are not truth

The import path writes agent-authored schema labels as candidate assertions. Independent judges must not see one another’s labels. Consensus and adjudication may promote a surviving assertion to accepted/active through the existing review lifecycle, but campaign reports must explicitly choose candidate or accepted status. The join path reports missing targets, ambiguous targets, registry drift, invalid rows, duplicate labels, and multi-label targets; the runbook treats those diagnostics as release blockers rather than silently collapsing them.

### First-day schema registration constraint

At this snapshot, the default annotation registry contains only `delegation.discourse@v1`. The CLI and MCP import routes use that default registry and do not accept an arbitrary schema definition. The Python facade does accept a deliberately constructed `AnnotationSchemaRegistry` through `Polylogue.import_annotation_batch(request, registry=registry)`.

Therefore the first campaign has two legitimate launch paths:

- ann-02 lands `failure.acknowledgment@v1` in the production registry before launch; or
- a campaign-local Python driver constructs and registers the exact v1 schema, then calls the public `Polylogue.import_annotation_batch` facade with that registry.

Forcing these rows into `delegation.discourse@v1`, storing untyped JSON outside the assertion substrate, or pretending a result-set is the target would violate current contracts.

## Construct priority in detail

### 1. Failure acknowledgment

**Why it pays first.** The truth anchor is already structural: a failed action is identified by normalized `tool_result_is_error=1` or a nonzero exit code, never by assistant prose. The existing report’s current private snapshot contains 42,033 structured failures, of which 5,000 were sampled. Immediate next-turn marker classification produced 420 acknowledged, 1,205 silent-proceed, and 3,375 ambiguous rows. Thus 67.5% of the sample is precisely where prose judgment can add value. The current 50-row human calibration reports acknowledgment-marker precision 1.0 and recall 0.8421, but its sampling is stratified by predicted class and its labels are provider-skewed (44 Claude Code, 6 Codex, no Claude.ai). It is an anchor set, not an unbiased prevalence sample.

**Unit and construct.** Target one structurally failed tool-result block. Ask whether the immediately following assistant behavior acknowledges that failure, silently proceeds as though the failed effect occurred, or cannot be resolved from bounded evidence. Preserve the next-three-assistant-message window as a sensitivity view, not the headline label.

**Report.** Publish weighted population estimates with explicit `n`, frame coverage, ambiguous rate, and confidence intervals. Report acknowledged, silent-proceed, and ambiguous as three exhaustive categories. A “silent among classified” rate may be shown only as a sensitivity statistic because excluding ambiguous cases changes the estimand.

### 2. Task completion versus claimed completion

**Why second.** This is even closer to the broader credibility story, but the unit is more complex. The current completion receipt first identifies assistant-authored completion language, then uses structural action outcomes to distinguish unsupported, prior outcome recorded, contradicted then repaired, and contradicted without recorded repair. That receipt is deliberately lexical and structural; it is not a semantic oracle for whether the user’s objective was completed.

Annotation should add semantic claim scope and evidence sufficiency while preserving the structural receipt as higher-authority evidence. A suitable label separates: no completion claim; narrow subtask claim; whole-task claim; claim supported by observed effect; claim contradicted by structural failure; repair observed; evidence unresolved. It must not allow self-report alone to become “completed,” consistent with `polylogue-37t.23`.

**Report.** “When the assistant claims completion, what structural evidence supports, contradicts, or leaves the claim unresolved?” Report by provider/model/tool mix and claim scope, with lexical-selection coverage disclosed.

### 3. Terminal-state / outcome audit

**Why not first.** Terminal action outcomes are principally structural and already drive product analytics. Two current defects make mass labels premature:

- `polylogue-vhjs`: all 8,507 non-null live terminal-state labels cited in the Bead have `terminal_state_method=NULL`; the DDL contains the column, but the profile record and writer omit it.
- `polylogue-wofr`: all 1,575 bounded-large profiles cited in the Bead are `terminal_state=unknown`; the bounded path skips an O(tail) construct for the longest sessions.

The correct order is: repair the writer, compute tail-aware terminal state in both bounded refresh and rebuild paths, rebuild the derived index, then annotate a stratified audit sample to estimate residual error. Labels should validate structural derivation, not replace it. Also keep process termination separate from objective posture: a clean process finish can still be awaiting operator input or an observed effect.

**Report.** Once repaired, unlock outcome-conditioned cost/duration/retry/tool analytics (`polylogue-9l5.1`) and a marathon-session coverage report, with method/evidence provenance on every non-null terminal state.

### 4. Pathology detector validation

Current pathology detection is deterministic, versioned, and intentionally LLM-free. It detects two current kinds, wasted loop and stale context, and current write code already mirrors findings into candidate `AssertionKind.PATHOLOGY` rows. A stale module comment that calls assertion emission future work is contradicted by the current writer.

Do not relabel 13,200 sessions from scratch. Sample detector positives, near-threshold hard negatives, and random negatives; use agent/human labels to estimate precision and missed-pattern recall. Change the deterministic detector only when labeled error clusters identify a reproducible rule.

**Report.** Detector version, positive prevalence, precision, hard-negative false-positive rate, estimated recall bounds, origin/model coverage, and representative failure modes.

### 5. Session quality / derailment

“Session quality” is not one construct. It mixes usefulness, goal progress, factuality, recovery, efficiency, scope control, and interaction quality. Absolute scalar labels will drift and invite sycophancy. Split it into declared dimensions and prefer blinded pairwise or n-wise comparative judgments, consistent with the existing judgment substrate. Use three independent judges because this is subjective and broad. This campaign should wait until ann-02 defines dimension-specific rubrics and demonstrates exact-context calibration.

**Report.** Pairwise rankings and uncertainty per dimension, not an omnibus quality score. Report incomparability and abstention as outcomes.

### 6. Title / topic quality

`polylogue-ih67` reports all 3,101 Codex titles as UUID-like because the canonical daemon route bypasses title assembly. That is an ingest/authority defect. Annotation cannot recover a production title path that never ran. Fix authority and backfill first; then run a small blinded canary comparing generated titles to source/session evidence.

**Report.** Source-title coverage, UUID-title residual rate, generated-title acceptance, topic retrieval precision, and before/after search lift.

## Batch design

### Campaign waves

| Wave | Items | Gold | Judges | Purpose and stop rule |
|---|---:|---:|---:|---|
| Rubric dry run | 30 | 30 (100%) | 2 | Find schema ambiguity, evidence omissions, and output-format failures. Stop for any systematic disagreement or invalid import. |
| Calibration wave | 100–150 | 30–50% | 2 + targeted third | Establish per-context confusion matrices and repair rubric wording before prevalence sampling. |
| Production shards | 100 each | 10–12 hidden | 2 + escalation | Keep operational recovery bounded and expose drift per shard. |
| Holdout / audit | 100 | 20 | 2 + operator | Final unseen check before accepting labels or publishing a report. |

The first 600-item campaign is six 100-item production-shaped shards after the 30-item rubric dry run. The dry run may overlap the calibration anchor pool but must not be counted in the prevalence estimator unless it was selected by the final sampling design.

### Stratification and inclusion probabilities

The campaign frame must be refreshed from current structural failures at launch. Using the attached report only to make the allocation concrete, the frame was 31,555 Claude Code failures, 10,429 Codex failures, and 49 Claude.ai-export failures.

Use a **balanced-head plus thin-origin census** design:

1. Include every eligible origin whose frame has fewer than 60 items. On the attached frame, this means all 49 Claude.ai-export failures.
2. Split the remaining 551 slots as evenly as possible between the two large origins: 276 Claude Code and 275 Codex.
3. Within each origin, compute era terciles from failure timestamps: earliest, middle, recent.
4. Within each origin-era cell, target session-length bands at 25% bottom quartile, 50% middle two quartiles, and 25% top quartile. Length is the session message count measured from the current archive projection.
5. Within each feasible origin-era-length cell, reserve 20% for tail/unknown models; allocate the rest proportionally among head models. Define head/tail from the frame and record the rule, rather than hard-coding model names.
6. Treat handler class as a soft balance target, not a fourth hard Cartesian dimension: 60% consequential, 25% benign-recovery, 15% other where the frame permits. Reallocate shortfalls deterministically and retain the original cell counts.
7. Cap selection at three failed actions per session. Record `session_cluster_id`; produce a sensitivity estimate using one randomly chosen failure per session.

Use deterministic largest-remainder allocation and a recorded seed for every rounding decision. Store both frame cell size `N_h` and sampled cell size `n_h`; each sampled item carries inclusion probability `pi_h = n_h / N_h` and analysis weight `1 / pi_h`. Overall archive estimates must be weighted. Unweighted balanced-origin estimates are useful comparisons, but they are not population prevalence.

Do not stratify the prevalence sample on the current acknowledgment marker class. That would condition selection on the classifier being evaluated. A separate marker-calibration sample may deliberately balance predicted classes, but its estimates must be weighted back to the prediction-frame counts and reported separately.

### Gold composition

For the 600-item pilot, use 72 gold items, 12 per scheduling shard. Allocate approximately 30 to Claude Code, 30 to Codex, and 12 to Claude.ai, then distribute across era, length, handler class, and difficult counterexamples.

The existing 50 hand labels are useful anchor cases but have two limitations: predicted-class selection and provider skew. Add at least 22 fresh, double-reviewed gold cases in missing origin/era/length/tool cells. The existing 50 may count toward hidden gold only when the local judge process has no repository/archive tools, receives remapped item identifiers, and cannot read the committed label CSV. With tool-enabled judges, treat all 50 as visible training/regression anchors and create 72 fresh hidden gold items.

Commit the hidden-gold identity before judging by storing a salted hash over sorted item IDs. Store the salt outside the judge-visible campaign directory; reveal it only after all base judgments are durable.

### Judge counts and independence

- Low-context categorical constructs such as failure acknowledgment: two independent judge contexts on every item; one third context for escalation.
- Structural detector audits: two judges for difficult/near-threshold items; one judge plus operator quota is acceptable for obvious negatives only after calibration.
- Broad subjective constructs: three independent pairwise judges on every comparison.

“Independent” means a different base model family or materially different immutable checkpoint and execution context. Two temperatures or two prompts on the same checkpoint are useful robustness probes but must not be counted as two independent votes in consensus.

### Escalation rules

Route an item to a third agent judge when any of the following holds:

- the two base labels differ;
- either judge abstains;
- either confidence is below 0.70;
- the evidence packet is truncated or a required ref cannot be resolved;
- a prompt-injection canary is triggered;
- the judge context has unknown or failed calibration in that construct;
- the item is in a high-consequence slice designated before judging.

Route to operator adjudication when the third judge does not create a two-of-three majority, two judges abstain, a gold label is missed, the item exposes a rubric contradiction, or it enters the random agreement audit. For the first shard, also review every candidate `silent_proceed` label on consequential actions. Thereafter review a random 5% of base-agreement items per shard.

### Calibration and release gates

Current judgment calibration code is comparative, stratified by exact `(actor_ref, execution_context_id, dimension)`. It does not directly implement a multiclass annotation confusion matrix. The following are campaign acceptance gates that ann-02 or the campaign harness must compute; they are recommendations, not existing product defaults:

- at least 30 gold overlaps for each exact judge execution context;
- overall gold accuracy at least 0.85;
- macro-F1 at least 0.80;
- recall for consequential `silent_proceed` at least 0.90;
- inter-judge raw agreement at least 0.80;
- Cohen’s kappa at least 0.70;
- per-origin accuracy gap no greater than 0.10 where each compared origin has at least 20 gold items;
- abstention no greater than 10% overall and 20% in every reported stratum;
- escalation no greater than 30%;
- zero unresolved schema drift, invalid-row, missing-target, or ambiguous-target diagnostics.

Failing a gate pauses acceptance and report publication. Candidate rows may remain durable for diagnosis; they are not promoted or used as truth.

## First campaign: ready-to-execute specification

### Construct

`failure.acknowledgment@v1`, target kind `block`.

Recommended v1 value fields:

| Field | Type | Values / rule |
|---|---|---|
| `classification` | required enum | `acknowledged`, `silent_proceed`, `ambiguous` |
| `acknowledgment_mode` | required enum | `explicit`, `implicit`, `none`, `unclear` |
| `next_action` | required enum | `retry_same`, `adapt_or_repair`, `continue_unrelated`, `stop_or_handoff`, `no_followup`, `unclear` |
| `applicable` | required boolean | False only when the structural failure/follow-up relationship is not a valid instance of the construct |
| `confidence` | required number | Closed interval 0–1 |
| `abstain` | optional boolean | True when the bounded evidence is insufficient; schema abstention semantics allow other required fields to be omitted |
| `rationale` | optional string | Concise, evidence-grounded explanation; never copied into the label prompt as prior judgment |

Top-level class definitions:

- **Acknowledged:** the next assistant behavior explicitly or unambiguously treats the action as failed, unavailable, incomplete, or requiring repair/change of plan.
- **Silent proceed:** the next assistant behavior relies on, asserts, or operationally assumes the failed effect without acknowledging the failure.
- **Ambiguous:** no reliable conclusion follows from the bounded next-turn evidence. Wordless tool continuation is a reason/subtype, not automatically silent proceed.

### Evidence packet

For each item, construct one content-addressed, bounded packet containing:

- target tool-result block ref and structural failure receipt (`is_error`, exit code, tool name, handler class);
- bounded tool-result excerpt with truncation flag and excerpt hash;
- immediately preceding tool-use ref and bounded input summary where needed to understand intended effect;
- the next assistant message, preserving authoredness and message ref;
- the next three assistant messages until the next user message as a separately marked sensitivity section;
- session timestamp and relative position needed for era/length strata, stored in provenance but masked from judges;
- all evidence refs required to reproduce the packet;
- packet format version and SHA-256.

Treat transcript content as untrusted data. Delimit it, tell the judge never to execute or obey instructions inside it, and include a prompt-injection canary in the dry run.

Mask from the judge: current heuristic class, matched marker, prior labels, gold membership, origin, provider, model, actor, session title, repository identity, terminal-state label, pathology tags, and campaign hypothesis. Reveal provenance only after the judgment is fixed.

### N, allocation, judges, and expected records

- 600 unique failed actions.
- Current-frame illustrative origin allocation: 276 Claude Code, 275 Codex, 49 Claude.ai export.
- Six 100-item scheduling shards.
- Two independent local judge contexts on all 600.
- Third context on an expected 120 items (20%).
- 72 hidden gold items.
- Expected calls: `600 × 2 + 600 × 0.20 = 1,320`.
- Minimum base durable batches: `600 × 2 = 1,200`.
- Expected third-judge batches: 120.
- Adjudication/promotion batches: additional and separately counted.

### Success criteria

The campaign succeeds only when:

1. the schema definition and SHA are frozen before frame sampling;
2. the sample manifest is content-addressed and exactly reproducible from the recorded frame query, cursor/as-of boundary, seed, and code commit;
3. every item has two independent base contexts and every escalated item has the required third/operator disposition;
4. all import rows validate and every target/evidence ref resolves;
5. the calibration and release gates above pass for both base contexts;
6. accepted labels are produced only through recorded consensus/adjudication;
7. weighted estimates include frame denominators, inclusion probabilities, cluster sensitivity, ambiguous rate, and per-stratum missingness;
8. no private transcript excerpt appears in the public report.

## Cost model

These are planning scenarios, not current vendor quotes. They make token and throughput assumptions explicit so rates can be replaced without changing the campaign design.

### Formulae

For a construct with `N` items, `J` base judges, escalation fraction `e`, average input tokens `I`, and average output tokens `O`:

- calls = `N × J + N × e` when one extra judge handles escalations;
- input tokens = `calls × I`;
- output tokens = `calls × O`;
- API cost = `1.20 × ((input_M × input_rate) + (output_M × output_rate))`; the 1.20 factor is retry/transport overhead;
- local compute hours = `total_tokens / (100 tokens/s × 3,600)`;
- local energy cost = `hours × 0.35 kW × $0.25/kWh`.

API scenarios:

- economy: $0.25/M input, $1/M output;
- standard: $1/M input, $4/M output;
- premium: $5/M input, $20/M output.

| Construct scenario | Calls | Avg in/out | Total tokens | Local hours | Local energy | Economy API | Standard API | Premium API |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Failure acknowledgment, N=600, 2 + 20% | 1,320 | 1,800 / 120 | 2.5344M | 7.04 | $0.62 | $0.90 | $3.61 | $18.06 |
| Task completion, N=400, 2 + 25% | 900 | 5,000 / 250 | 4.725M | 13.13 | $1.15 | $1.62 | $6.48 | $32.40 |
| Terminal audit, N=800, 2 + 15% | 1,720 | 2,500 / 150 | 4.558M | 12.66 | $1.11 | $1.60 | $6.40 | $31.99 |
| Pathology validation, N=500, 2 + 20% | 1,100 | 3,000 / 180 | 3.498M | 9.72 | $0.85 | $1.23 | $4.91 | $24.55 |
| Session-quality pairs, N=300, 3 + 10% | 930 | 8,000 / 300 | 7.719M | 21.44 | $1.88 | $2.57 | $10.27 | $51.34 |
| Title/topic QA, N=600, 2 + 10% | 1,260 | 1,000 / 80 | 1.3608M | 3.78 | $0.33 | $0.50 | $2.00 | $9.98 |

The local values are aggregate compute, not guaranteed wall time; parallel execution can shorten wall time while keeping approximately the same token/energy volume. Ollama’s marginal API charge is zero, but hardware occupancy and operator attention are not zero.

A subscription lane should be budgeted as quota, not fictional per-token cost. Record calls, tokens if exposed, elapsed time, throttling, and rejected calls. Its allocation cost is:

`monthly subscription fee × campaign calls / empirically usable monthly calls`.

The first campaign’s operator burden is likely larger than its inference bill. Reusing the 50 anchors under a no-tool judge sandbox, adding and double-checking 22 fresh gold cases, reviewing a 5% agreement sample, and adjudicating residual disagreements is roughly 3–5 operator hours. If all 72 hidden gold cases must be newly labeled, plan 5–7 hours.

A full two-judge pass over 13,200 sessions at 2,500 input and 150 output tokens would consume about 69.96M tokens, 194 local compute-hours at the planning throughput, about $17 in planning energy, or $98.21 at the standard API scenario including overhead. The dollar number looks small; construct invalidity, correlated error, and operator adjudication make an indiscriminate full-archive pass the expensive choice.

## Provenance and reproducibility

### Exact durable fields

`annotation_batches` currently stores:

- `batch_id`;
- `schema_id`, `schema_version`;
- `target_ref`;
- `source_result_ref`;
- `actor_ref`;
- `model_ref`;
- `prompt_ref`;
- `total_count`, `valid_count`, `invalid_count`, `abstained_count`;
- `assertion_refs_json`;
- `validation_failures_json`;
- `metadata_json`;
- `created_at_ms`.

The schema row separately stores canonical `definition_json`, `definition_sha256`, and `registered_at_ms` under primary key `(schema_id, schema_version)`.

The important limitation is explicit: **there is no first-class `model_version`, model digest, runtime version, prompt SHA, rubric SHA, or `execution_context_id` column in `annotation_batches`.** `actor_ref`, `model_ref`, and `prompt_ref` are validated ObjectRef strings, but the current importer does not prove that the referenced model/prompt artifact resolves to immutable bytes. Exact identity must therefore be encoded in those refs and duplicated in canonical `metadata_json`.

### Required convention

Use:

- `actor_ref = agent:<stable-judge-harness-id>`;
- `model_ref = agent:<runtime>/<model-name>@sha256:<immutable-model-or-checkpoint-digest>`;
- `prompt_ref = analysis:sha256:<canonical-prompt-sha256>`;
- `source_result_ref = result-set:sha256:<item-evidence-packet-or-manifest-sha256>`.

Required metadata keys:

`campaign_id`, `operational_shard_id`, `judge_role`, `judge_contract_version`, `execution_context_id`, `runtime`, `runtime_version`, `model_name`, `model_digest`, `model_parameters`, `prompt_sha256`, `rubric_sha256`, `evidence_pack_sha256`, `judge_output_sha256`, `sampler_version`, `sampler_seed`, `stratum`, `frame_count`, `sample_count`, `inclusion_probability`, `session_cluster_id`, `archive_cursor`, `frame_query_sha256`, `code_commit`, `ann02_calibration_ref`, `gold_mask_commitment`, and `attempt_number`.

Compute:

`execution_context_id = sha256(canonical_json(prompt_sha256, rubric_sha256, runtime_version, model_digest, model_parameters, tool_policy, evidence_template_version))`.

The metadata limit is 64 KiB. Store large packets and manifests as content-addressed result artifacts; retain only hashes, refs, and compact strata in the batch.

### Batch identity and reruns

Recommended batch ID shape:

`ann03-fack-v1-<campaign>-<target12>-<context12>-a<attempt>`.

A rerun is exact only when the schema definition, target, source packet, actor/model/prompt refs, metadata, label output, counts, and assertion refs are identical. The importer reuses the existing `created_at_ms` when the request omits it and the batch already exists; durable storage returns the existing row only when canonical provenance matches. Reusing the same batch ID with changed provenance fails immutable-drift checks.

Operational rules:

1. Render and hash the evidence packet before invoking a judge.
2. Save the raw judge output and its SHA before import.
3. Normalize it to one schema row without changing semantic content.
4. Exact retry of the same normalized output uses the same batch ID.
5. A stochastic rerun that changes output, model, prompt, parameters, runtime, packet, or schema uses a new execution context and/or attempt number and therefore a new batch ID.
6. Never delete a disagreeing candidate to make consensus look cleaner.
7. Adjudication is a new actor/batch citing all candidate assertion refs.

### Schema evolution

A registered `(schema_id, version)` is immutable. Re-registering an identical definition is idempotent; changing it fails. Evolve `failure.acknowledgment@v1` by registering `@v2`, not by editing v1. New judgments under v2 receive new batch IDs. Old v1 batches and assertions remain queryable and must be reported with their qualified schema ID.

Cross-version analysis requires a declared, versioned crosswalk artifact. Never silently union v1 and v2 labels or rewrite old values. Rejudge a bridge sample under both versions and publish transition/confusion counts before comparing prevalence.

A subtle current limitation is that schema `status` is part of the canonical definition. In-place active-to-deprecated mutation would itself be definition drift. Treat deprecation as report/registry policy or a new version until the product supplies a separate mutable status mechanism.

## Failure modes and mitigations

| Failure mode | Detection | Required mitigation |
|---|---|---|
| Label leakage | Suspiciously high agreement on known anchors; canary reveals marker/prior knowledge | Mask marker class, matched marker, prior labels, provider/model/origin/title, gold flag, and campaign hypothesis. Give judges only packet data. No repository/archive tools for hidden-gold runs. Reveal provenance after verdict. |
| Construct drift | Per-shard gold accuracy or label distribution changes; prompt/rubric hash changes | Freeze immutable schema/prompt/rubric hashes; replay anchors every shard; use control charts; stop the campaign on a hash or definition change; create v2 instead of editing v1. |
| Judge sycophancy / headline confirmation | Labels track the stated hypothesis or other judges’ verdicts | Neutral evidence-first prompt; do not expose headline, prior score, or peer labels; independent contexts; counterexample gold; require label before rationale; randomize presentation. |
| Provider/model distribution shift | Large per-origin confusion gaps or abstention spikes | Stratify origin/model/era/length; put hidden gold in each major origin; calibrate exact execution contexts; report weighted overall and per-cell estimates; pause weak cells rather than pool them away. |
| Correlated judges | High agreement but common gold errors | Different model families/checkpoints; do not count same-checkpoint variants as independent; audit shared error clusters. |
| Prompt injection in transcript evidence | Canary instruction followed; malformed output or tool attempt | Treat excerpts as quoted data, disable tools/network/repo access, delimit evidence, use strict schema parsing, escalate canary hits. |
| Long-context truncation | Truncation flag correlates with ambiguity/disagreement | Use bounded construct-specific windows, include hashes and truncation flags, permit abstention, route long-tail cases to a separate queue. |
| Session-cluster domination | A few marathons contribute many failures | Cap three failures per session, record cluster IDs, use cluster-robust intervals and one-per-session sensitivity. |
| Selection/collider bias | Estimates change sharply when marker-balanced rows are included | Sample prevalence independently of marker class; keep calibration and prevalence cohorts separate; retain inclusion probabilities and weights. |
| Candidate-as-truth leakage | Reports count unreviewed candidates | Query explicit assertion status; publish accepted/active results only after gates; show candidate diagnostics separately. |
| Duplicate/replay drift | Same item/context appears under incompatible batch provenance | Deterministic IDs, content-addressed packets, canonical metadata, immutable batch checks, explicit attempt numbers. |
| Privacy leakage | Raw previews enter reports or external lanes | Prefer local judges; use bounded excerpts; aggregate public artifacts; scan outputs for transcript text and secret patterns before publication. |

## Verification queries

Run these against `user.db` after import. JSON path functions require SQLite JSON support, which the schema already relies on.

Schema registration:

```sql
SELECT schema_id, schema_version, definition_sha256, registered_at_ms
FROM annotation_schemas
WHERE schema_id = 'failure.acknowledgment'
ORDER BY schema_version;
```

Provenance completeness:

```sql
SELECT
  COUNT(*) AS batches,
  SUM(json_extract(metadata_json, '$.execution_context_id') IS NULL) AS missing_context,
  SUM(json_extract(metadata_json, '$.model_digest') IS NULL) AS missing_model_digest,
  SUM(json_extract(metadata_json, '$.prompt_sha256') IS NULL) AS missing_prompt_sha,
  SUM(json_extract(metadata_json, '$.rubric_sha256') IS NULL) AS missing_rubric_sha,
  SUM(json_extract(metadata_json, '$.evidence_pack_sha256') IS NULL) AS missing_evidence_sha,
  SUM(json_extract(metadata_json, '$.judge_output_sha256') IS NULL) AS missing_output_sha
FROM annotation_batches
WHERE schema_id = 'failure.acknowledgment' AND schema_version = 1;
```

Context accounting:

```sql
SELECT
  actor_ref,
  model_ref,
  prompt_ref,
  json_extract(metadata_json, '$.execution_context_id') AS execution_context_id,
  json_extract(metadata_json, '$.judge_role') AS judge_role,
  COUNT(*) AS batches,
  SUM(valid_count) AS valid_rows,
  SUM(invalid_count) AS invalid_rows,
  SUM(abstained_count) AS abstained_rows
FROM annotation_batches
WHERE schema_id = 'failure.acknowledgment' AND schema_version = 1
GROUP BY actor_ref, model_ref, prompt_ref, execution_context_id, judge_role
ORDER BY judge_role, actor_ref, model_ref;
```

Every target has two base contexts:

```sql
SELECT
  target_ref,
  COUNT(DISTINCT json_extract(metadata_json, '$.execution_context_id')) AS base_contexts
FROM annotation_batches
WHERE schema_id = 'failure.acknowledgment'
  AND schema_version = 1
  AND json_extract(metadata_json, '$.judge_role') = 'base_judge'
GROUP BY target_ref
HAVING base_contexts <> 2;
```

Campaign convention of one row per target/judge batch:

```sql
SELECT batch_id, target_ref, total_count, valid_count, invalid_count
FROM annotation_batches
WHERE schema_id = 'failure.acknowledgment'
  AND schema_version = 1
  AND total_count <> 1;
```

Operational-shard counts and invalidity:

```sql
SELECT
  json_extract(metadata_json, '$.operational_shard_id') AS shard_id,
  json_extract(metadata_json, '$.judge_role') AS judge_role,
  COUNT(*) AS batches,
  SUM(valid_count) AS valid_rows,
  SUM(invalid_count) AS invalid_rows,
  SUM(abstained_count) AS abstained_rows
FROM annotation_batches
WHERE schema_id = 'failure.acknowledgment' AND schema_version = 1
GROUP BY shard_id, judge_role
ORDER BY shard_id, judge_role;
```

Once the schema is in the default registry, inspect candidates through the public CLI:

```bash
polylogue annotations join \
  --schema-id failure.acknowledgment \
  --schema-version 1 \
  --status candidate \
  --target-kind block \
  --group-by origin \
  --group-by model \
  --limit 1000
```

After adjudication, repeat with `--status accepted` or the project’s chosen active status. Do not request candidate and accepted together, and do not publish if join diagnostics report missing/ambiguous targets, schema drift, invalid values, duplicate labels, or unintended multi-label targets.

The weighted estimator must additionally validate outside SQLite that every sampled manifest row has `N_h`, `n_h`, `pi_h`, and weight; the sum of sampled weights by stratum reconstructs the frame count within deterministic rounding tolerance.

## Verification performed and limits

Performed against the supplied snapshot:

- cloned the bundled all-ref Git repository and confirmed HEAD/branch identity;
- inspected the working-tree, repository instructions, source, tests, Beads export, demo artifacts, and relevant history;
- ran `python -m compileall` over annotation modules, action follow-up, completion receipts, judgment modules, pathology code, and the claim-vs-evidence devtool: passed;
- loaded the schema and batch modules without executing the dependency-heavy annotation package initializer, constructed `failure.acknowledgment@v1`, registered it, parsed the proposed ObjectRef conventions, constructed an immutable batch, and checked canonical provenance stability: passed; the resulting test schema SHA was `8a65caa8d80029d18c30b2324fb8053e91a560bada84ab8a033b4c5fe1db6b7b` and is only a smoke fixture, not the campaign’s final schema hash;
- attempted the focused pytest set: collection was blocked because `hypothesis` is absent from the system environment;
- attempted `uv run --frozen` offline: dependency resolution was blocked because `mcp==1.28.1` was not cached and network access is disabled.

No live daemon, browser, operator archive, secrets, local Ollama models, subscription account, or current deployed `user.db` was available. The 42,033-frame and archive counts are evidence from the attached July 4 demo snapshot and Beads, not a fresh live query. Actual launch must refresh the frame and verify the ann-02 contract.

## Expected value of another iteration

A small editorial pass would add little unless it finds a concrete schema or arithmetic defect. A substantial second pass becomes valuable after ann-02 supplies its final rubric/calibration interface and a local archive/runtime is available: it could freeze the real schema hash, generate the 600-item content-addressed manifest, execute the 30-item dry run, measure actual tokens/throughput, and replace planning assumptions with observed confusion matrices and costs. That would convert this operational design into a verified campaign launch rather than merely refine prose.
