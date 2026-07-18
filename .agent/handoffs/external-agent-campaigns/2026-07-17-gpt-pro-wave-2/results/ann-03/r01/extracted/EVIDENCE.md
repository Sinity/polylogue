# Evidence ledger

## Evidence method

This ledger separates:

- **Observed fact:** directly present in the supplied snapshot source, Beads export, demo artifact, or Git history.
- **Source-supported inference:** follows from combining observed facts but is not itself a stored product claim.
- **Recommendation:** ann-03 campaign policy proposed by this runbook.
- **Unresolved:** requires the operator’s live archive/runtime or ann-02 output.

No private transcript preview from the attached calibration/sample files is reproduced here.

## Snapshot and repository authority

| Status | Evidence | Finding |
|---|---|---|
| Observed | `polylogue-overview.md`; `polylogue-overview.json`; `polylogue-manifest.json` | Snapshot generated `2026-07-17T180950Z`, source `/realm/project/polylogue`, branch `master`, commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`, `dirty=true`. Snapshot counts include 42 artifacts, 986 Beads, 28 open issues, 7 open PRs, and 8,423 all-ref commits. |
| Observed | `polylogue-branch-delta.md`, `polylogue-branch-delta.patch`, `polylogue-branch-delta-log.txt`, `polylogue-branch-delta-files.txt` | Merge base is the same HEAD commit and branch delta is empty. |
| Observed | bundled `polylogue-all-refs.bundle`; Git log | HEAD subject is `fix(repair): harden raw authority convergence (#3046)`. The bundle was cloneable and resolves `master`/`origin/master` to the named commit. |
| Observed | `polylogue-ignore-audit.md`, `polylogue-snapshot-audit.md` | Snapshot reports large ignored local state and tracked hidden surface, while the branch patch is empty. |
| Source-supported inference | Working-tree archive replay over bundled HEAD plus ordinary `git status` | The package does not expose an ordinary tracked source diff explaining `dirty=true`; the dirty cause may be ignored/local state or omitted hidden material. Exact cause is unresolved. |
| Observed | `CLAUDE.md:17-39` | Substrate owns meaning; CLI/MCP/API/daemon are leaf surfaces. |
| Observed | `CLAUDE.md:89-110` | `user.db` v6 is durable/irreplaceable; immutable annotation schemas and batch provenance live there; labels remain assertion rows scoped to annotation-batch refs. |
| Observed | `CLAUDE.md:162-178` | Durable-tier evolution is additive numbered SQL migrations; derived tiers rebuild. |

## Annotation schema and durable provenance

| Status | Evidence | Finding |
|---|---|---|
| Observed | `polylogue/storage/sqlite/migrations/user/006_annotation_schemas_batches.sql:4-22` | `annotation_schemas` is keyed by `(schema_id, schema_version)` and stores canonical definition JSON plus a 64-hex SHA-256. |
| Observed | `polylogue/storage/sqlite/migrations/user/006_annotation_schemas_batches.sql:34-68` | `annotation_batches` stores batch/schema IDs, target/source/actor/model/prompt refs, counts, assertion refs, validation failures, metadata JSON, and creation time. It does not define first-class model-version, model-digest, runtime-version, prompt-hash, rubric-hash, or execution-context columns. |
| Observed | `polylogue/annotations/schema.py:136-237` | `AnnotationField` is typed and closed; enum/numeric constraints are validated. Labels are not an unconstrained JSON blob. |
| Observed | `polylogue/annotations/schema.py:266-433` | `AnnotationSchema` includes target kinds, evidence policy, abstention convention, status, canonical definition JSON, and definition fingerprint. Status participates in the definition document/fingerprint. |
| Observed | `polylogue/annotations/schema.py:529-547` | Row validation checks schema target kind, evidence policy, required/extra fields, and typed values. |
| Observed | `polylogue/annotations/schema.py:550-618` | Registry reuse of the same ID/version is idempotent only for an identical definition; incompatible reuse fails. |
| Observed | `polylogue/annotations/schema.py:626-746`; migration seed at SQL lines 24-32 | The production default registry currently registers only `delegation.discourse@v1`. |
| Observed | `polylogue/annotations/batch.py:114-134` | `AnnotationBatch` explicitly contains the durable fields named above and snapshots canonical provenance. |
| Observed | `polylogue/annotations/batch.py:173-181` | Target, actor, model, and prompt are syntactically normalized ObjectRefs; `source_result_ref` must use kind `result-set`. No semantic model/prompt artifact resolution is enforced here. |
| Observed | `polylogue/annotations/batch.py:200-259` | Canonical provenance includes refs, metadata, counts, failures, and assertion refs; its bytes are the exact-retry identity. |
| Observed | `polylogue/storage/sqlite/archive_tiers/user_annotations.py:101-131` | Durable schema registration returns identical existing definitions and rejects incompatible reuse. |
| Observed | `polylogue/storage/sqlite/archive_tiers/user_annotations.py:222-308` | Batch persistence compares canonical provenance, returns an exact existing row, and rejects incompatible batch-ID reuse. |
| Source-supported inference | DDL plus batch/schema code | Current durable provenance can pin an exact model/prompt only by campaign convention: immutable identifiers in refs plus hashes/version fields in metadata. The database does not independently prove those refs resolve to immutable bytes. |
| Source-supported inference | Schema status is fingerprinted; durable definition reuse fails on change | In-place active→deprecated status change is currently definition drift. A separate status mechanism or new schema version is needed. |

## Import, row grain, lifecycle, and joins

| Status | Evidence | Finding |
|---|---|---|
| Observed | `polylogue/annotations/importer.py:35-39` | Import limits are 1 MiB payload, 10,000 rows, 64 KiB per line, 4 KiB per ref, and 64 KiB metadata. |
| Observed | `polylogue/annotations/importer.py:46-73` | JSONL rows contain `row_key`, `value`, evidence refs, optional body, and confidence. The request—not the row—contains target, schema, actor, model, and prompt. |
| Observed | `polylogue/annotations/importer.py:187-220` | The request target must resolve; every row validates under that same target. |
| Observed | `polylogue/annotations/importer.py:221-249` | Evidence refs are live-resolved; assertion identity uses the request target, actor, row key, and batch ref. |
| Observed | `polylogue/annotations/importer.py:263-314` | Exact retry reuses an existing batch timestamp when omitted; schema, batch, and candidate assertions are persisted atomically. Agent-authored assertions are written through the shared write path. |
| Source-supported inference | Request-wide target plus per-row schema/value only | A current durable batch is single-target and single judge context. A 100-item scheduling shard cannot be one current batch unless target identity is falsified. |
| Recommendation | Runbook D7/D8 | Use 100-item operational manifests externally and one complete row per target/judge durable batch. |
| Observed | `polylogue/annotations/write.py:1-20` | Agent writes are candidate declarations; live-ref resolution belongs at import. |
| Observed | `polylogue/annotations/write.py:116-170` | Annotation assertion IDs are deterministic and rows are validated against the active registry. |
| Observed | `polylogue/annotations/write.py:197-268` | Durable batch, schema, target, actor, and declared assertion linkage are enforced. |
| Observed | `polylogue/annotations/write.py:276-327` | Existing assertion identity/content drift fails rather than overwriting durable evidence. |
| Observed | `polylogue/annotations/join.py:48-67` | Join requests require exact schema version and explicit assertion statuses; active and accepted cannot be requested together. |
| Observed | `polylogue/annotations/join.py:114-136` | Result diagnostics include missing and ambiguous targets, schema drift, invalid values, multiple labels, and duplicate label counts. |
| Observed | `polylogue/annotations/join.py:239-295` | Durable schema is compared to the active registry, and status filtering is explicit. |
| Observed | `polylogue/annotations/join.py:339-430` | Exact target resolution and typed row revalidation occur before grouped output; no silent collapse of duplicates/multi-labels. |
| Observed | `polylogue/cli/commands/annotations.py:37-90` | `polylogue annotations import` takes exact batch/schema/target/source/actor/model/prompt options but no schema-definition/registry option. |
| Observed | `polylogue/cli/commands/annotations.py:93-136` | Join surface exposes exact schema version/status, target kind, and structural groups. |
| Observed | `polylogue/api/archive.py:1872-1901` | The public Python facade accepts an explicitly constructed `AnnotationSchemaRegistry`; this is the valid first-day path for a campaign schema not yet in the default registry. |
| Observed | `polylogue/core/refs.py:8-72, 122-159` | Valid ObjectRef kinds include `block`, `agent`, `result-set`, `analysis`, and `annotation-batch`; object IDs are opaque and may contain colons. |

## Failure-acknowledgment evidence

| Status | Evidence | Finding |
|---|---|---|
| Observed | `polylogue/archive/actions/followup.py:1-25` | Failure truth is structural; acknowledgment classification is a bounded prose heuristic with explicit recall caveats. |
| Observed | `polylogue/archive/actions/followup.py:32-112` | Existing classes are acknowledged, silent proceed, wordless continuation, and ambiguous; short/missing text is ambiguous, marker hits acknowledged, and otherwise prose may be silent proceed. |
| Observed | `devtools/claim_vs_evidence.py:73-107` | Current defaults: limit 5,000, sample limit 30, `n_min=30`, calibration size 50, calibration seed 20260703. |
| Observed | `devtools/claim_vs_evidence.py:293-352` | Structural frame is tool-result `is_error=1` or nonzero exit code, including null/false `is_error` cases with nonzero exit. |
| Observed | `devtools/claim_vs_evidence.py:545-558` | Paired rows receive immediate-next and next-three assistant context. |
| Observed | `devtools/claim_vs_evidence.py:597-670` | Existing report counts failures by origin and allocates a bounded origin-stratified sample. |
| Observed | `devtools/claim_vs_evidence.py:681-720` | The 50-row marker calibration sample is deliberately selected across the current predicted classes, not as an unbiased structural-frame sample. |
| Observed | `docs/findings/claim-vs-evidence.md:5-7, 24-44` | Published finding frames the 24.1% value as a bounded lower bound, grounds failure structurally, and reports marker calibration rather than an intent oracle. |
| Observed | logical snapshot demo `.agent/demos/claim-vs-evidence/public-summary.json`, `summary.json`, `claim-vs-evidence.report.json` extracted from `polylogue-agent-demos-and-prompts.xml` | Captured July 4, index schema v24: 42,033 total structured failures, 101 unpaired, 5,000 inspected. Immediate next turn: 420 acknowledged, 1,205 silent, 3,375 ambiguous. Next-three: 722 acknowledged, 1,851 silent, 2,427 ambiguous. |
| Observed | same demo report | Sampled origins: 3,752 Claude Code from 31,555 failures; 1,241 Codex from 10,429; 7 Claude.ai from 49. |
| Observed | `.agent/demos/claim-vs-evidence/ack-marker-calibration.labels.csv` | 50 human labels: 19 acknowledged, 10 silent-proceed, 21 ambiguous; 44 Claude Code and 6 Codex; no Claude.ai; handler classes 43 consequential, 4 benign-recovery, 3 other. |
| Observed | `summary.json` calibration object | Acknowledgment-marker precision 1.0, recall 0.8421052631578947 on the committed 50 labels. |
| Source-supported inference | 3,375 / 5,000 | 67.5% of the bounded sample remains ambiguous under the existing immediate marker classifier, making semantic judgment the highest-value addition. |
| Source-supported inference | Predicted-class sampling plus provider distribution | The 50 labels are good anchors and classifier calibration, but cannot directly estimate population prevalence and need provider/era/length supplementation. |
| Recommendation | Runbook sample design | Draw prevalence from the structural frame independent of marker class; retain a separate marker-calibration cohort. |

## Completion-claim evidence

| Status | Evidence | Finding |
|---|---|---|
| Observed | `polylogue/demo/receipts.py:18-37` | Completion-claim population is lexically declared, with a bounded deterministic sample. |
| Observed | `polylogue/demo/receipts.py:313-435` | Assistant-authored claim text defines the population; structural action outcomes classify support/contradiction/repair rather than trusting prose. Current classes include `unsupported_by_structural_tool_evidence`, `prior_outcome_recorded`, `contradicted_then_repaired`, and `contradicted_without_recorded_repair`. |
| Observed | `tests/unit/demo/test_demo_completion_claims.py:23-375` | Tests pin stable manifests and structural receipts and go red when failure/repair evidence, `is_error` cases, protocol exclusion, outcome ordering, command/tool identity, or evidence fingerprints are removed/drifted. |
| Source-supported inference | Receipt mechanism plus its tests | Existing code is a strong lexical/structural cohort, not a semantic proof that the user’s objective was completed. Annotation should label claim scope/evidence sufficiency and retain structural authority. |
| Observed | Bead `polylogue-37t.23` in `polylogue-beads-export.jsonl` | Process termination and objective posture are orthogonal; a self-reported claim without observed/evaluated effect cannot become completed. |

## Terminal-state and outcome evidence

| Status | Evidence | Finding |
|---|---|---|
| Observed | `polylogue/archive/session/runtime.py:281-398` | Terminal state is derived from structural tool results and typed events. After removal of prose/clean-finish heuristics, absence of evidence becomes unknown; later structural success can clear earlier failure. |
| Observed | history commit `682b29bf3` | Current HEAD includes deletion of prose-keyword terminal-state/activity heuristics. |
| Observed | `polylogue/storage/insights/session/rebuild.py:754-889` | The bounded-large path emits terminal state unknown with zero confidence and `large_session_bounded` evidence rather than computing tail state. |
| Observed | `polylogue/storage/insights/session/profiles.py:315-390` | Profile construction carries terminal state/confidence/evidence but omits `terminal_state_method`. |
| Observed | `polylogue/storage/insights/session/storage.py:50-100, 278-337` | Profile column/value write lists omit `terminal_state_method`. |
| Observed | `polylogue/storage/sqlite/archive_tiers/index.py:933-972` | Index DDL contains `terminal_state_method`, establishing a writer/DDL mismatch. |
| Observed | Bead `polylogue-vhjs` | Live finding: `terminal_state_method` is NULL for all 8,507 non-null labels cited (4,371 error-left, 4,136 clean-finish). |
| Observed | Bead `polylogue-wofr` | Live finding: all 1,575 bounded-large profiles cited are terminal-state unknown; lengths span 350–96,748 messages and occupy the longest decile. |
| Observed | Bead `polylogue-61zb` | Refresh/rebuild heavy-session degraded-shape parity was fixed, but that does not add terminal-state computation to the bounded bundle. |
| Observed | Bead `polylogue-t0p.1` | Claude background completion outcomes are parsed structurally, including typed exit status, supporting structural rather than prose outcome authority. |
| Source-supported inference | DDL/writer mismatch plus Beads | Annotation before repair would mix true semantic residual error with known deterministic missing provenance/coverage defects. |
| Recommendation | D3 | Repair/rebuild first; annotate a post-repair audit sample only. |
| Observed | Bead `polylogue-9l5.1` | Planned outcome-conditioned cost/duration/retry/tool analytics explicitly require structural action outcome fields, not prose. |
| Observed | Bead `polylogue-9l5.2` | Cross-provider comparison requires coverage caveats on every row and refusal of bare numbers when provenance is absent. |

## Pathology evidence

| Status | Evidence | Finding |
|---|---|---|
| Observed | `polylogue/insights/pathology.py:1-56` | Detector is deterministic and LLM-free; current version is 4; current kinds are wasted loop and stale context; diagnostic exclusions/cost rules are explicit. |
| Observed | `polylogue/insights/pathology.py:104-188` | Wasted-loop and stale-context rules are concrete, reproducible structural rules. |
| Observed | `polylogue/storage/sqlite/archive_tiers/user_write.py:1210-1269` | Current storage code already mirrors detector findings into deterministic candidate `AssertionKind.PATHOLOGY` rows and preserves promoted status. |
| Observed | `tests/unit/insights/test_pathology.py:81-204` | Tests cover repeated consecutive failures, success breaks, cancellations, diagnostic/mixed failures, severity, lossy resume, clean resume, subagent exclusion, version/counts, determinism, and clean projections. |
| Adjudicated contradiction | `pathology.py` module commentary versus current writer | Any comment implying assertion emission is future work is stale; current source wins. |
| Recommendation | D4 | Use annotations to estimate detector validity on positives/hard negatives/random negatives, not to create a parallel pathology truth store. |

## Judgment, blinding, calibration, and cascade evidence

| Status | Evidence | Finding |
|---|---|---|
| Observed | `docs/design/analysis-rigor.md:18-44` | Archive counts define population frames; validity and measurement failures must be resolved before quantitative claims. |
| Observed | `docs/design/analysis-rigor.md:94-101` | Blinding is part of elicitation design, not an optional display choice. |
| Observed | `docs/design/analysis-rigor.md:179-210` | Judgments are noisy samples; pairwise judgments are preferred for broad subjective questions; judge identity includes model plus prompt hash and should be calibrated against gold. |
| Observed | `docs/design/analysis-rigor.md:222-249` | Elicitation, cascades, operator routing, and a three-agent/blinded/operator-spot-check example are explicitly described. |
| Observed | `polylogue/insights/judgment/types.py:37-55` | `actor_ref` identifies the actor/family and `execution_context_id` fingerprints exact prompt/tools/runtime/config; context changes must not be folded into actor identity. |
| Observed | `polylogue/insights/judgment/calibration.py:1-10, 85-137` | Calibration is stratified by exact `(actor_ref, execution_context_id, dimension)` and computes agreement with trusted gold on overlap; no cross-context pooling. |
| Observed | `polylogue/insights/judgment/cascades.py:25-104` | Current defaults use minimum agreement 0.8 and minimum gold overlap 5; nondecisive, high-stakes, disagreement, quota, unknown, insufficient, or low-calibration cases route to operator. |
| Observed | `polylogue/insights/judgment/blinding.py:21-38, 61-118` | Default masked fields include provider/model/actor/prior provenance and execution context; a receipt binds masked order/rubric; reveal is allowed only after a verdict; leak checks fail on surviving fields. |
| Observed | `tests/unit/insights/judgment/test_calibration.py:29-172` | Tests prove two contexts for the same actor remain separate, missing gold is unknown, and no cross-context pooling function is exposed. |
| Observed | `tests/unit/insights/judgment/test_blinding.py:57-140` | Tests freeze the mask list, prove visible projection cannot recover fields, require verdict before reveal, and use mutation tests to detect a removed production mask. |
| Observed | `tests/unit/insights/judgment/test_cascades.py:39-105` | Tests route nondecisive/unseen/low-calibration/insufficient-gold/disagreement/quota cases to operator and allow only well-calibrated covered decisive screens to stop. |
| Observed | Beads `polylogue-rxdo.9.11`, `.9.12`, `.9.14`, `.9.15`, `.9.16` | Comparative judgment, exact-context calibration, exploration/blinding, cascade routing, and judgment UX intent are tracked. The pure mechanism substrate landed; some UX work was consolidated/deferred. |
| Source-supported inference | Current calibration source and types | The existing calibration implementation is comparative-judgment agreement, not a ready-made multiclass annotation confusion/macro-F1 pipeline. ann-02 must supply or specify that interface. |
| Recommendation | D16-D18 | Calibrate each exact multiclass judge context with ≥30 gold and use stronger campaign-specific release gates. |

## Beads and campaign-state evidence

All records below are from `polylogue-beads-export.jsonl`; current source wins where notes describe an older implementation state.

| Bead | Status | Evidence used |
|---|---|---|
| `polylogue-rxdo.7.1` | closed P1 | Durable immutable annotation schemas and independent batch provenance; merged PR #2765. |
| `polylogue-rxdo.7.2` | closed P1 | Candidate-only bounded JSONL import, live target/evidence validation, CLI/MCP shared operation; merged PR #2767. |
| `polylogue-kmts` | closed P1 | Generic exact-target join, explicit status, no silent fanout, and diagnostic counts; merged PR #2768. |
| `polylogue-212.9.1` | closed P1 | Fable packet requires versioned schema, independent batches, adjudication/disagreement, exact joins, denominators, missingness, and evidence resolution. |
| `polylogue-sru` | closed epic | Claim-vs-evidence campaign reached finding-grade methodology and productized action follow-up classification. Its historical count was later refreshed in the attached demo. |
| `polylogue-vhjs` | open P2 | Terminal-state method/evidence provenance absent for all cited non-null live labels. |
| `polylogue-wofr` | open P2 | Bounded-large sessions are terminal-state blind despite terminal detection being tail-bounded. |
| `polylogue-ih67` | open P1 | All 3,101 indexed Codex titles cited are UUIDs because canonical raw-record daemon ingest bypasses provider assembly/title authority. |
| `polylogue-37t.23` | open P1 | Objective posture is distinct from terminal process state; self-report without observed effect cannot establish completion. |
| `polylogue-9l5.1` | open P4 | Outcome-conditioned analytics depend on structural outcomes. |
| `polylogue-9l5.2` | open P4 | Cross-provider analytics require explicit coverage tiers and composition refusal when evidence is absent. |
| `polylogue-61zb` | closed P1 | Heavy-session refresh/rebuild parity landed, but terminal-state bounded-path blindness remains separate. |
| `polylogue-t0p.1` | closed P1 | Background completion outcomes now preserve structural status/exit-code evidence. |

## History evidence

| Commit | In HEAD? | Relevance |
|---|---:|---|
| `bf94704c0` | yes | Typed annotation schema foundation. |
| `246c48d08` | yes | Durable schema/batch provenance. |
| `f4504cb4d` | yes | Provenance-stamped JSONL import. |
| `4ed0cf2dc` | yes | Exact structural target joins. |
| `52b77aa38` | yes | Public Python annotation import facade. |
| `ca76f2df1` | yes | Failure acknowledgment marker calibration. |
| `c68c278ef` | yes | Action follow-up classification exposed to query. |
| `11615f99e` | yes | Analysis-rigor program adopted. |
| `866dab24d` | yes | Comparative judgment/calibration/blinding/cascade substrate. |
| `682b29bf3` | yes | Prose terminal-state heuristics deleted. |
| `9163d0134` | yes | Agent-facing archive reads bounded. |
| `672786a07` | no | Declares actor/execution-context refs on another remote branch; runbook deliberately does not assume it. |

## Test evidence and verification performed

### Focused tests inspected

- Annotation durability: `tests/unit/annotations/test_durable_storage.py` tests incompatible schema reuse, independent same-target batches, exact canonical retry, opaque refs, NFC collision rejection, detached provenance, and insert-once replay.
- Import: `tests/unit/annotations/test_importer.py` tests candidate/failure roundtrip, independent batches, concrete schema, and exact retry.
- Join: `tests/unit/annotations/test_join.py` tests grouping/nonjoins, generic non-delegation targets, active+accepted rejection, lifecycle provenance, and registry drift diagnostics.
- Completion receipts: `tests/unit/demo/test_demo_completion_claims.py` contains explicit anti-vacuity tests for removing structural failure/repair, losing `is_error`, including protocol material, wrong time/order, or missing command/tool identity.
- Claim-vs-evidence: `tests/unit/devtools/test_claim_vs_evidence.py` tests bounded artifacts, origin stratification, minimum-`n` refusal, deterministic reproduction, and stable tool-result identity.
- Pathology and judgment tests are listed in the sections above.

### Commands and outcomes

| Command/check | Outcome |
|---|---|
| Clone bundled all-ref Git repository; inspect branch/HEAD/history | Passed. |
| Replay supplied working-tree source over HEAD and inspect ordinary tracked status | No ordinary tracked diff found; exact dirty cause unresolved. |
| `python -m compileall -q` over annotation, failure-followup, completion, judgment, pathology, and devtool modules | Passed. |
| Direct isolated load of `schema.py` and `batch.py`; construct/register proposed smoke `failure.acknowledgment@v1`; parse proposed refs; construct immutable batch; compare canonical provenance bytes | Passed. Smoke schema SHA `8a65caa8d80029d18c30b2324fb8053e91a560bada84ab8a033b4c5fe1db6b7b`; not the final ann-02 schema. |
| Focused system `pytest` set | Not executed: collection failed in `tests/conftest.py` because `hypothesis` is not installed. |
| `UV_OFFLINE=1 uv run --frozen pytest -q tests/unit/annotations/test_schema.py` | Not executed: offline resolver could not fetch uncached `mcp==1.28.1`. |

The two pytest failures are environment/dependency failures, not test failures. No claim is made that the focused tests passed in this container.

## Source-supported inferences that drive the runbook

1. **Failure acknowledgment has the highest annotation marginal value.** Structural failure already fixes the frame; prose classification leaves 67.5% ambiguous in the current sample; a bounded three-class judgment directly reduces that uncertainty.
2. **Mass batch scheduling cannot be mapped one-to-one to current durable batch rows.** The importer’s batch-wide target makes a separate scheduling-manifest layer necessary, but that layer remains an artifact/reference rather than a new truth database.
3. **Current provenance is sufficient only with a strict convention.** Canonical metadata is durable, but exact model/prompt/runtime identity is not enforced by separate columns or artifact resolution.
4. **Terminal/title defects should be fixed, not labeled around.** Their observed error mechanism is deterministic and upstream of judgment.
5. **Pathology should be validated, not replaced.** A deterministic versioned detector and candidate assertion mirror already exist.
6. **Broad quality needs pairwise dimensions.** Existing rigor/judgment mechanisms and the construct’s multidimensionality make absolute omnibus labels invalid.
7. **Current 50 gold labels cannot alone support provider-generalized prevalence.** Their selection and provider distribution require a fresh stratified prevalence sample and supplemental gold.

## Recommendations not claimed as existing product behavior

The following are ann-03 design choices, not current hard-coded defaults:

- 600-item first campaign and six 100-item scheduling shards;
- 276/275/49 illustrative origin allocation under the attached frame;
- 72 hidden gold items and exact gold allocation;
- two independent base judges plus expected 20% third-judge escalation;
- confidence threshold 0.70;
- ≥30 gold per exact context, accuracy 0.85, macro-F1 0.80, consequential silent recall 0.90, kappa 0.70, and the remaining release gates;
- local throughput/energy and API scenario rates;
- exact metadata key set and execution-context hash recipe;
- one row per target/judge batch as campaign convention;
- content-addressed frame/item packet formats and batch-ID naming.

## Unresolved evidence before live launch

- Current live structural-failure frame and exact origin/model/era/length distributions.
- Available local Ollama model checkpoints, immutable digests, actual throughput, context limits, and energy draw.
- ann-02’s final schema/rubric/prompt/gold/calibration/promotion interface.
- Current deployed `user.db` schema rows and assertion lifecycle state.
- Whether existing 50 anchors can be hidden from the chosen judge runtime; tool-enabled judges require entirely fresh hidden gold.
- Actual operator adjudication time and post-third unresolved rate.
- Post-fix/rebuild terminal-state and title quality, because `vhjs`, `wofr`, and `ih67` are open.
