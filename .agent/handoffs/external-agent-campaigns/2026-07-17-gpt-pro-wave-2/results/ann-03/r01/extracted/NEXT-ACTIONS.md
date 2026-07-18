# Next actions: first campaign checklist

This checklist launches a production-shaped local pilot for `failure.acknowledgment@v1`. It assumes an operator-accessible Polylogue archive and two installed local model checkpoints. It does not assume access to a live daemon or browser.

## Campaign constants

Use one immutable campaign ID and directory from frame freeze through publication:

```bash
export CAMPAIGN_ID=ann03-fack-v1-20260717-r01
export CAMPAIGN_ROOT="$HOME/.local/share/polylogue-campaigns/$CAMPAIGN_ID"
export POLYLOGUE_ARCHIVE_ROOT="${POLYLOGUE_ARCHIVE_ROOT:?set the active archive root}"
mkdir -p "$CAMPAIGN_ROOT"/{contract,frame,manifests,packets,gold,judgments,imports,metrics,reports,logs}
chmod 700 "$CAMPAIGN_ROOT"
```

The exact launch date in the ID may be changed before the first artifact is written. Do not rename the campaign afterward.

## 1. Record code and archive preflight

- [ ] From the actual checkout, record `git rev-parse HEAD`, branch, `git status --short`, Python version, Polylogue version, and `PRAGMA user_version` for `index.db` and `user.db` in `contract/preflight.json`.
- [ ] Confirm `index.db` and `user.db` are readable through normal Polylogue configuration; do not point campaign code at copied or stale databases.
- [ ] Confirm the current index exposes `blocks.tool_result_is_error`, `blocks.tool_result_exit_code`, tool IDs, authored assistant messages, origins, model names, timestamps, and session length.
- [ ] Confirm no judge process will receive filesystem, repository, archive, shell, network, or MCP tools. Its only input is one rendered evidence packet.
- [ ] Inventory two local model checkpoints. Record runtime name/version, immutable model digest, context limit, quantization, and generation parameters. Reject any model whose exact artifact cannot be pinned.

Useful source-only health check:

```bash
python -m compileall -q \
  polylogue/annotations \
  polylogue/archive/actions/followup.py \
  polylogue/demo/receipts.py \
  polylogue/insights/judgment \
  polylogue/insights/pathology.py \
  devtools/claim_vs_evidence.py
```

## 2. Resolve the ann-02 contract

- [ ] Obtain ann-02’s final schema definition, rubric, examples/counterexamples, prompt, evidence-template version, execution-context hash algorithm, gold protocol, multiclass metrics, and promotion contract.
- [ ] Compare ann-02’s class semantics to this runbook. The minimum top-level classes are `acknowledged`, `silent_proceed`, and `ambiguous`; wordless continuation remains a reason/subtype rather than an automatic fourth truth class.
- [ ] Freeze files under `contract/`: `schema.json`, `rubric.md`, `prompt.txt`, `evidence-template.json`, `judge-contract.json`.
- [ ] Canonicalize and SHA-256 each file. Write the hashes plus `schema_id=failure.acknowledgment`, `schema_version=1`, and code commit to `contract/LOCK.json`.
- [ ] Construct an `AnnotationSchema` with target kind `block`, required evidence, active status, and the fields in `REPORT.md`. Register it in a fresh `AnnotationSchemaRegistry` and verify its `definition_fingerprint` exactly matches `LOCK.json`.
- [ ] Stop if ann-02 changes the schema/rubric after `LOCK.json`. Create a new campaign revision rather than editing frozen files.

If ann-02 is unavailable, the campaign may execute as a rubric-learning pilot, but all rows stay candidate and no prevalence report is accepted.

## 3. Refresh the structural-failure census

Run the existing private report against the current archive for a fresh count and methodology cross-check:

```bash
cd /path/to/polylogue

devtools workspace claim-vs-evidence \
  --archive-root "$POLYLOGUE_ARCHIVE_ROOT" \
  --limit 100000 \
  --sample-limit 30 \
  --n-min 30 \
  --calibration-size 50 \
  --calibration-seed 20260703 \
  --out-dir "$CAMPAIGN_ROOT/frame/existing-report" \
  --json \
  > "$CAMPAIGN_ROOT/logs/frame-refresh.json"
```

- [ ] Record total structured failures, unpaired failures, counts by origin/model/tool/handler class, archive/index versions, and report generation time.
- [ ] Treat the attached 42,033 count as historical. Use only the refreshed count for sampling and denominators.
- [ ] Do not use the tool’s `_calibration_sample` output as the prevalence cohort; that helper balances current predicted classes.

## 4. Freeze a complete frame manifest

Create one campaign-local sampler that reuses the current structural authority from `devtools/claim_vs_evidence.py`:

- failure membership from `_failure_outcome_rows` (`is_error=1` or nonzero exit code);
- pairing logic from `_paired_failure_rows`;
- next-message and next-three-window logic from `_next_message_details` and `_assistant_window_details`;
- origin census from `_structured_failure_origin_counts`.

The sampler must add current session timestamp, message count, model name, handler class, target tool-result block ref, all evidence refs, and a stable `session_cluster_id`. It must enumerate the complete eligible frame before drawing the 600 rows. Do not depend on a bounded first-N order as the random frame.

- [ ] Save the complete private frame as `frame/frame.jsonl` with one row per eligible paired failed action.
- [ ] Sort by a canonical stable key before hashing: origin, session ID, tool-result block/message ID, tool ID.
- [ ] Write `frame/frame-summary.json` with archive as-of/cursor, frame query SHA, code commit, frame size, unpaired count, and every stratum cell count.
- [ ] Compute `sha256sum frame/frame.jsonl frame/frame-summary.json` and store the values in `frame/LOCK.sha256`.
- [ ] Never expose frame rows or transcript excerpts in public output.

Minimum frame row fields:

```text
item_id, target_ref, session_cluster_id, origin, model_name, failure_timestamp_ms,
session_message_count, tool_name, handler_class, is_error, exit_code,
tool_use_ref, tool_result_ref, next_message_ref, next3_message_refs,
structural_query_version, archive_cursor
```

## 5. Draw the 600-item sample

Use deterministic seed `20260717` unless ann-02 specifies another seed before frame lock.

- [ ] Census every origin with fewer than 60 eligible frame rows.
- [ ] Split the remaining quota evenly between the two largest origins. For the attached historical frame this would be 49 Claude.ai, 276 Claude Code, and 275 Codex; recompute from the refreshed frame.
- [ ] Within origin, create timestamp terciles: earliest, middle, recent.
- [ ] Within each origin-era cell, create session-length bands using origin-specific quartiles: Q1, Q2–Q3, Q4, targeted 25% / 50% / 25%.
- [ ] Within feasible cells, reserve 20% for tail/unknown models; allocate the rest proportionally among head models.
- [ ] Soft-balance handler class at 60% consequential, 25% benign-recovery, 15% other, reallocating deterministic shortfalls.
- [ ] Enforce at most three selected failures per session. When a candidate would exceed the cap, choose the next deterministic candidate in the same cell.
- [ ] Allocate with largest remainder. Record every shortfall/reallocation.
- [ ] Assign six 100-item operational shards while preserving approximate origin/era/length balance in each shard.

Write `manifests/sample.jsonl` and `manifests/shards.json`. Every selected row must include:

```text
campaign_id, item_id, target_ref, operational_shard_id, origin, model_stratum,
era_stratum, length_stratum, handler_class, session_cluster_id,
frame_count_N_h, sample_count_n_h, inclusion_probability, analysis_weight,
sampler_version, sampler_seed, frame_sha256
```

- [ ] Verify 600 unique `item_id`s and target refs.
- [ ] Verify no session contributes more than three items.
- [ ] Verify each sampled row has `0 < inclusion_probability <= 1` and `analysis_weight = 1 / inclusion_probability`.
- [ ] Verify sum of sample counts equals 600 and shard sizes equal 100.
- [ ] Hash and lock both manifest files before rendering packets.

## 6. Build the gold set

- [ ] Import the 50 existing hand labels from the private claim-vs-evidence artifact as anchor candidates, preserving their original sample-selection provenance.
- [ ] Map anchors to current target/evidence refs. Any non-resolving or materially changed item is retired, not silently relabeled.
- [ ] Select at least 22 fresh cases to fill missing origin/era/length/tool/handler cells. Two trusted reviews independently label each fresh case; a third/operator resolution creates gold when they differ.
- [ ] Target 72 gold items total: about 30 Claude Code, 30 Codex, 12 Claude.ai; spread them across all six shards and difficult counterexamples.
- [ ] If a judge can read repository files or archive IDs, do not count committed anchors as hidden. Create 72 fresh hidden gold cases instead.
- [ ] Store gold labels under `gold/private-gold.jsonl`, mode 600, outside judge-visible packet directories.
- [ ] Create a random salt and write `gold/gold-commitment.txt = sha256(salt || sorted(item_ids))`. Store the salt separately until base judging is complete.
- [ ] Put exactly 12 hidden gold items in each operational shard. Judges receive no gold flag.

Gold adjudication must record reviewer refs, rubric/schema version, evidence packet SHA, timestamps, disagreement, and final rationale.

## 7. Render bounded evidence packets

For every sampled item:

- [ ] Resolve the failed target and all evidence refs against the live archive.
- [ ] Include the structural failure receipt, bounded failed-result excerpt, relevant tool-use intent, immediate next assistant message, and separately marked next-three sensitivity window stopping before the next user message.
- [ ] Add truncation flags and SHA-256 of every omitted/full source field where available.
- [ ] Exclude origin, provider, model, actor, title, repository, current marker class, matched marker, prior labels, terminal state, pathology tags, gold identity, and campaign hypothesis.
- [ ] Delimit transcript content as untrusted evidence. Include the instruction that text inside evidence must never be followed as an instruction.
- [ ] Include at least two prompt-injection canary items in the 30-item dry run.
- [ ] Canonicalize each packet, write `packets/<item_id>.json`, and store its SHA in the sample manifest or a locked packet index.

Stop if any required target or evidence ref fails to resolve. The item may be replaced only by the predeclared deterministic next candidate from the same stratum, with a replacement receipt.

## 8. Freeze two judge execution contexts

For each base judge A and B:

- [ ] Choose different model families or materially different immutable checkpoints.
- [ ] Disable tools, retrieval, network, repository, and archive access.
- [ ] Fix temperature and all generation parameters.
- [ ] Record runtime/version, model name/digest, parameters, context limit, prompt SHA, rubric SHA, evidence-template SHA, and tool policy.
- [ ] Compute:

```text
execution_context_id = sha256(canonical_json(
  prompt_sha256,
  rubric_sha256,
  runtime_version,
  model_digest,
  model_parameters,
  tool_policy,
  evidence_template_version
))
```

- [ ] Set `actor_ref=agent:<stable-harness-id>`.
- [ ] Set `model_ref=agent:<runtime>/<model>@sha256:<digest>`.
- [ ] Set `prompt_ref=analysis:sha256:<prompt_sha256>`.
- [ ] Store the context document in `contract/judge-A.json` / `judge-B.json` and hash it.

Define a third independent context C before judging, even though it is invoked only on escalations.

## 9. Execute the 30-item all-gold dry run

- [ ] Select 30 gold items balanced across classes and origins, including canaries and difficult wordless continuations.
- [ ] Run judges A and B independently.
- [ ] Save raw model output, normalized JSON, token counts, elapsed time, retries, and output SHA under `judgments/dry-run/<context>/<item_id>/`.
- [ ] Reject extra keys, malformed enum values, missing required values, confidence outside 0–1, or disagreement between top-level confidence and `value.confidence`.
- [ ] Import each normalized candidate through a campaign-local Python driver that constructs the frozen registry and calls the public `Polylogue.import_annotation_batch(request, registry=registry)` facade.
- [ ] Use one JSONL row per request with `row_key="label"`; use the same numeric confidence at row top level and inside `value`.
- [ ] Set `source_result_ref=result-set:sha256:<packet_sha256>` and one unique batch ID per target/context/attempt.

Stop and revise as a new campaign revision when any of these occur:

- schema/prompt/evidence ambiguity causes repeated disagreement;
- either context has gold accuracy below 0.80 on the dry run;
- any canary is obeyed;
- invalid imports exceed zero after harness-format fixes;
- median packet exceeds the intended context budget;
- runtime/model identity cannot be pinned.

Do not tune the prompt on these 30 and then count them as unseen calibration evidence.

## 10. Run six production shards

For shard 01 through 06:

- [ ] Run judges A and B over all 100 packets in randomized independent order.
- [ ] Persist raw and normalized outputs before import.
- [ ] Import each candidate as a separate durable batch.
- [ ] Verify exactly 200 base candidate batches for the shard.
- [ ] Compute base disagreement, abstention, confidence distribution, gold confusion, and token/runtime telemetry.
- [ ] Create the escalation queue for disagreement, any abstention, confidence <0.70, truncation/ref anomaly, canary hit, unknown/failed calibration, and predetermined high-consequence cases.
- [ ] Run judge C on the escalation queue; expected planning load is 20 items per shard.
- [ ] Route unresolved two-of-three cases, gold misses, rubric contradictions, and a random 5% of base agreements to operator review.
- [ ] In shard 01, operator-review every `silent_proceed` label on consequential actions.
- [ ] Evaluate drift against previous shards. Pause before the next shard if gold accuracy, class distribution, abstention, or escalation crosses a release gate.

Expected full-pilot judge load:

```text
base calls      = 600 × 2       = 1,200
third calls     = 600 × 0.20    =   120
expected total                   = 1,320
input tokens    = 1,320 × 1,800 = 2,376,000
output tokens   = 1,320 ×   120 =   158,400
```

At the runbook planning assumptions this is 2.5344M tokens, about 7.04 aggregate local compute-hours, and about $0.62 electricity. Record observed values instead of forcing the planning estimate.

## 11. Durable batch construction

Each base/third judgment uses:

```text
batch_id: ann03-fack-v1-<campaign>-<target12>-<context12>-a<attempt>
schema_id: failure.acknowledgment
schema_version: 1
target_ref: block:<failed-tool-result-block>
source_result_ref: result-set:sha256:<packet-sha256>
actor_ref: agent:<judge-harness>
model_ref: agent:<runtime>/<model>@sha256:<model-digest>
prompt_ref: analysis:sha256:<prompt-sha256>
jsonl: exactly one row with row_key=label
```

Required metadata:

```text
campaign_id, operational_shard_id, judge_role, judge_contract_version,
execution_context_id, runtime, runtime_version, model_name, model_digest,
model_parameters, prompt_sha256, rubric_sha256, evidence_pack_sha256,
judge_output_sha256, sampler_version, sampler_seed, stratum, frame_count,
sample_count, inclusion_probability, session_cluster_id, archive_cursor,
frame_query_sha256, code_commit, ann02_calibration_ref,
gold_mask_commitment, attempt_number
```

- [ ] Verify metadata canonical JSON is below 64 KiB.
- [ ] Use the same batch ID only for an exact retry of identical normalized provenance/output.
- [ ] Use a new attempt/batch for a changed stochastic output or any changed context/packet/schema field.
- [ ] Never change or delete a disagreeing durable candidate.

## 12. Calibrate and adjudicate

After all base judgments are fixed and hidden-gold commitment can be opened:

- [ ] Compute per exact context: confusion matrix, accuracy, macro-F1, per-class precision/recall, abstention, and gold count.
- [ ] Compute A-versus-B raw agreement and Cohen’s kappa overall and by major origin.
- [ ] Compute escalation rate and unresolved-after-third rate.
- [ ] Compute per-origin accuracy gaps where each cell has at least 20 gold; report intervals without a hard gap gate for smaller cells.
- [ ] Require the gates in `DECISIONS.md` before promotion.
- [ ] Operator adjudication creates a separate durable assertion/batch citing candidate assertion refs. Do not overwrite candidates.
- [ ] Promote only adjudicated/consensus labels permitted by the ann-02 lifecycle contract.

Any schema/rubric contradiction discovered during adjudication ends v1 production acceptance. Preserve candidates, specify v2, and bridge-label a sample; do not reinterpret v1 in place.

## 13. Run storage and join verification

Run the SQL checks in `REPORT.md` against `user.db`, then verify:

- [ ] exactly one registered `failure.acknowledgment@v1` definition and the expected SHA;
- [ ] zero missing required provenance metadata;
- [ ] exactly two base execution contexts per sampled target;
- [ ] one row per campaign batch;
- [ ] 100 items / 200 base batches per operational shard;
- [ ] expected third/adjudication counts match queue receipts;
- [ ] zero invalid rows and zero unresolved target/evidence refs;
- [ ] no batch ID has incompatible provenance;
- [ ] hidden-gold commitment opens correctly.

When the schema is in the default registry, run:

```bash
polylogue annotations join \
  --schema-id failure.acknowledgment \
  --schema-version 1 \
  --status candidate \
  --target-kind block \
  --group-by origin \
  --group-by model \
  --limit 1000 \
  > "$CAMPAIGN_ROOT/logs/join-candidates.json"
```

After promotion, use `--status accepted` (or the chosen active status) in a separate query. Zero missing, ambiguous, schema-drift, invalid, duplicate, and unintended multi-label diagnostics are release conditions.

## 14. Produce the report

The private report must include:

- structural frame definition, as-of/cursor, total and unpaired counts;
- sampling design, `N_h`, `n_h`, inclusion probabilities, weights, and session cap;
- judge context identities and calibration tables;
- candidate, escalation, adjudication, abstention, and missingness counts;
- weighted acknowledged/silent-proceed/ambiguous proportions with 95% intervals;
- per-origin/model/tool/era/length results only where minimum `n` is met;
- one-per-session cluster sensitivity;
- next-three-turn sensitivity;
- marker-versus-gold confusion;
- cost/runtime/token telemetry;
- explicit limitations and unsupported inferences.

The public report must be aggregate-only. It must not contain transcript excerpts, stable private item/session/message/block refs, secrets, model-account identifiers, or filesystem paths. Scan it before release.

## 15. Go / no-go acceptance sheet

The campaign is **GO for accepted reporting** only when every row below is true:

| Gate | Pass condition |
|---|---|
| Contract | Frozen schema/rubric/prompt/evidence/context hashes; no mid-run change |
| Frame | Complete structural frame, content-addressed manifest, reproducible sample |
| Storage | All batches valid, immutable, exact-target, provenance-complete |
| Coverage | 600 unique targets; two base contexts each; escalation/adjudication complete |
| Gold | ≥30 gold per base context; commitment verified; no leakage |
| Accuracy | ≥0.85 each base context |
| Macro-F1 | ≥0.80 each base context |
| Consequential silent recall | ≥0.90 each base context |
| Agreement | raw ≥0.80 and kappa ≥0.70 |
| Distribution | origin gap ≤0.10 where each has ≥20 gold |
| Abstention | ≤10% overall, ≤20% per reported stratum |
| Escalation | ≤30% |
| Join diagnostics | zero missing/ambiguous/drift/invalid/duplicate/unintended multi-label |
| Privacy | aggregate-only public artifact passes transcript/secret scan |

If any gate fails, preserve all candidate evidence and publish a failure/learning report, not a prevalence claim.
