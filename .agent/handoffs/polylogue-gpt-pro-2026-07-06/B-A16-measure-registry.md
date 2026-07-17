## [assistant 350]
bash -lc cd /mnt/data/poly_pkg/polylogue && python3 - <<'PY'
import json, textwrap
issues=[]
for l in open('polylogue-beads-export.jsonl'):
    d=json.loads(l)
    if d.get('_type')=='issue': issues.append(d)
ids=['polylogue-9l5','polylogue-9l5.1','polylogue-9l5.2','polylogue-9l5.3','polylogue-9l5.4','polylogue-9l5.5','polylogue-9l5.6','polylogue-9l5.7','polylogue-9l5.8','polylogue-9l5.9']
by={d['id']:d for d in issues}
for id in ids:
    d=by.get(id)
    if d:
        print('\n###',id,d.get('priority'),d.get('status'),d.get('issue_type'),d.get('title'))
        print('labels',d.get('labels'))
        for f in ['description','design','acceptance_criteria','notes']:
            v=d.get(f)
            if v:
                print(f.upper()+':')
                print(textwrap.shorten(v.replace('\n',' | '), width=1200, placeholder=' ...'))
PY

## [assistant 368]
According to the 2026-07-05 A16 note, the right target is not “more `analyze` commands,” but a measure registry whose rows compose over `Query × Projection × Render` and whose variants are produced by grouping/window/comparison swaps, not bespoke code paths. fileciteturn2file0

Below I’m using the code snapshot in the uploaded bundle, so line references like `polylogue/insights/archive_models.py:45-92` refer to that snapshot and should be verified locally before implementation.

## Recommendation

**[proposal] Defended recommendation:** add a first-class `MeasureSpec` registry beside the existing insight registry, then make current insights and future analytics consume it. Concretely: `insights/registry.py` remains the surface/presentation registry, while a new `polylogue/analytics/measures.py` or `polylogue/insights/measures.py` declares analytic measures as composable specs. `InsightType` currently describes display fields, query model, operations method, CLI command, readiness exemption, and rendering metadata; it explicitly says semantics stay in archive/storage layers and the registry owns only transport/presentation (`polylogue/insights/registry.py:1-14`, `69-87`). That is too shallow for construct-valid analytics. Keep it, but do not overload it into doing measure semantics.

**[proposal] Runner-up rejected:** one materialized insight or one CLI mode per analytic, such as `analyze cache-amplification`, `analyze thinking-tax`, `analyze thread-gini`, etc. This is tempting because `insights/registry.py` already auto-wires CLI/MCP/API-style surfaces. I reject it because it recreates exactly the hand-written-mode explosion A16 warns against: cross-model, longitudinal, per-repo, per-week, and post-compaction variants would become new code instead of grouping/window/comparison parameters. The 9l5 epic already says every delivered analytic should land as a registered measure over the query algebra, with evidence tier, sample frame, confounds, coverage preconditions, and automatic tier footnotes; 9l5.7 is explicitly the keystone for that. Bead `polylogue-9l5`, `polylogue-9l5.7`.

## Layer 1 — today’s substrate

**[evidence] Today’s `insights/registry.py` is a registry of insight products, not a measure registry.** It registers product rows like `session_profiles`, `session_work_events`, `session_phases`, `threads`, `archive_coverage`, `tool_usage`, `session_costs`, `cost_rollups`, `usage_timeline`, and `archive_debt`, each with an operations method and display fields (`polylogue/insights/registry.py:365-708`). That is a useful surface registry, but it does not declare formulas, denominators, uncertainty, coverage gates, or confounds.

**[evidence] The raw ingredients for many measures already exist.** `SessionEvidencePayload` carries message/substantive/tool/thinking counts, cost, wall/tool/thinking/output durations, compaction count, tool categories, logical-session id, token lanes, credit cost, timing provenance, and cost provenance (`polylogue/insights/archive_models.py:45-92`). `SessionLatencyProfilePayload` carries median/p90/max tool latency, stuck tool count, agent/user response medians, and a construct-boundary warning that agent response includes model delay plus intervening tool execution (`archive_models.py:129-140`). `ThreadPayload` carries thread session count, depth, branch count, total messages, cost, wall duration, origin/work-event breakdown, confidence, and member evidence (`archive_models.py:255-271`). The `actions` view exposes tool result error and exit code (`polylogue/storage/sqlite/archive_tiers/index.py:324-343`). `session_profiles` materializes the central denominator fields, including logical session id, message/substantive counts, tool/thinking counts, durations, workflow/terminal labels, timing provenance, token lanes, and cost provenance (`index.py:799-852`). `session_runs`, `session_observed_events`, and `session_context_snapshots` provide the run/subagent/event/context graph substrate (`index.py:913-1015`).

**[evidence] The code already understands provenance distinctions that the registry must enforce.** Provider usage coverage distinguishes exact Claude Code and Codex usage from estimate-only ChatGPT/Claude.ai exports, partial AI Studio/Gemini CLI/Hermes rows, and unsupported origins (`polylogue/storage/usage.py:62-145`). Usage counters preserve input, output, cached input, cache write, reasoning output, and total tokens as separate lanes (`storage/usage.py:161-214`). Cost basis payloads explicitly say the cost axes do not sum to one number because the same usage can be expressed as provider-reported, API-equivalent, subscription-equivalent, catalog-priced, or surcharge bases (`polylogue/archive/semantic/pricing.py:101-108`). Codex token handling already has the critical disjoint-lane warning: Codex input includes cached input, Codex output includes reasoning, and summing naïvely produced roughly an 8x inflation class (`polylogue/storage/sqlite/archive_tiers/write.py:2608-2636`).

**[evidence] The rigor audit is close to the right meta-layer.** `insights/rigor.py` declares which products have evidence payloads, inference payloads, fallback markers, confidence fields, readiness semantics, and stable consumer fields (`polylogue/insights/rigor.py:45-82`). `insights/audit.py` already computes bounded evidence/inference/fallback/stale-version/confidence rollups over insight products (`polylogue/insights/audit.py:53-201`). This should be generalized from “product rigor” to “measure validity.”

**[evidence] Current docs already warn against several misleading denominators.** `docs/insights.md` says `engaged_duration_ms` is message-clustered wall clock and does not measure human attention, keyboard focus, or operator presence; `tool_active_duration_ms` only sums paired timestamped tool events and does not invent duration for unpaired/untimestamped events (`docs/insights.md:69-83`). It also says workflow shape is a threshold classifier over observable features, not task importance, agent quality, correctness, or productivity (`docs/insights.md:92-102`), and terminal state is a read-only boundary signal, not a task-success judgment (`docs/insights.md:104-113`). Session latency docs say agent response latency includes tool execution, provider tool latency needs timestamped pairs, and user response latency caps long idle gaps (`docs/insights.md:119-132`).

## Layer 2 — near-term substrate change

**[proposal] Add `MeasureSpec`, not new `analyze` modes.** The minimum useful shape:

```python
class MeasureSpec(BaseModel):
    id: str
    construct: str
    operationalization: str
    reducer: Literal["count","sum","ratio","mean","median","pXX","entropy","Gini","zstd-ratio"]
    column_expr: str
    unit_frame: Literal["sessions","actions","messages","observed-events","work-events","phases","threads"]
    denominator_expr: str | None
    evidence_tier: Literal["structural","provider-reported","derived","heuristic","mixed"]
    required_coverage: CoverageGate
    confounds: tuple[str, ...]
    provenance_mixing_flags: tuple[str, ...]
    uncertainty: UncertaintySpec
    null_policy: Literal["suppress","zero","exclude","separate-unknown"]
    formula_version: int
    output_schema: str
    footnote_template: str
```

**[proposal] Put the measure registry below the insight registry.** `insights/registry.py` should eventually register insight surfaces that are projections of measures. For example, `usage_timeline` can remain a surface, but `cache_amplification_ratio` and `thinking_tax` should be reusable measure specs over `usage_timeline`/`session_model_usage`/provider usage events, not hardwired fields of that one surface.

**[proposal] Add a `measure` pipeline stage after aggregate support lands.** It should be algebraic:

```text
sessions where repo:polylogue
| measure stuck_tool_density by origin, model window iso_week compare previous_4_weeks
```

No separate “longitudinal stuck tool density” code. `window iso_week` plus `compare previous_4_weeks` is the longitudinal variant. `by origin, model` is the cross-model variant.

**[proposal] Make coverage preconditions suppress invalid measures, not merely annotate them.** A measure whose denominator mixes tiers should fail closed unless the registry declares an allowed mixed-tier formula. Example: `credit_vs_api_divergence` may intentionally compare `api_equivalent_usd` to `subscription_credits`; `outcome_conditioned_cost` must not silently mix provider-reported costs with catalog estimates and then emit a single “cost.”

**[proposal] Extend `insights/audit` to `measure audit`.** It should report, per measure: eligible rows, suppressed rows, evidence tier distribution, coverage gate failures, stale materializer versions, and whether uncertainty output is appropriate. Bead `polylogue-9l5.7`.

## Layer 3 — full direction

**[proposal] The full design is a cartesian analytics algebra.** A user or agent chooses:

`unit-frame × selection × measure × grouping × window × comparison × uncertainty × projection/render`

That means `cache_amplification_ratio by model per iso_week`, `cache_amplification_ratio by repo`, and `cache_amplification_ratio compare pre/post compaction policy` are all the same measure. The measure registry owns the construct and denominator; the query algebra owns the slice; the grouping/window/comparison operators own variants; renderers own tables, charts, JSON, MCP, and footnotes.

**[proposal] Query objects and analysis runs should record measure invocations.** A measure output should be addressable: query run id, measure id, formula version, unit frame, group/window/comparison, coverage gate result, row count, and footnote. This connects directly to the earlier query-as-object/analysis-run idea: an external agent can annotate a result, save a cohort, or cite a specific measure run.

**[proposal] Agents should be able to discover valid measures before designing an analysis.** The MCP/CLI surface should answer: “what measures are valid over `actions` with `origin=codex-session` and `window=iso_week`?” This prevents agents from inventing invalid “productivity” numbers from whatever columns are visible.

## Registry rows for the 16 requested measures

In the table below, “suppress” means the registry should refuse to render the measure as a bare number. It can still return a diagnostic explaining which coverage precondition failed.

| Measure | Construct operationalized | Exact formula | Unit-frame / denominator | Evidence tier | Top confounds | Suppress when |
|---|---|---|---|---|---|---|
| **cache-amplification ratio** | Prompt-cache reuse intensity, not model intelligence or efficiency. | `Σ cache_read_tokens / max(Σ fresh_input_tokens, 1)`, where fresh input is the disjoint non-cache input lane after provider normalization. | Sessions or observed provider-usage events; denominator is fresh input tokens for rows with cache lane coverage. | Provider-reported where usage coverage is exact; mixed if estimates enter. | Provider cache semantics differ; Codex input includes cached input unless disjointed. | No provider-reported cache lane; origin coverage is estimate-only/unsupported; fresh/cached lanes are not separable. **Mix flag: cached vs fresh tokens.** |
| **reasoning “thinking tax”** | Share of output budget spent on hidden/explicit reasoning. | `Σ reasoning_output_tokens / max(Σ reasoning_output_tokens + Σ completion_output_tokens, 1)`, with `completion_output_tokens = max(output_tokens - reasoning_output_tokens, 0)` when output includes reasoning. | Observed-events or sessions; denominator is decomposed output lane, not raw output if raw output includes reasoning. | Provider-reported only where reasoning lane exists. | Reasoning may be absent for providers that still reason; output/reasoning overlap semantics differ. | Reasoning lane unavailable, or output semantics unknown. **Mix flag: reasoning subset vs additive output.** |
| **latency three-lane share** | Time allocation among thinking, tool activity, and output within a session. | Vector: `(Σ thinking_duration_ms / Σ wall_duration_ms, Σ tool_duration_ms / Σ wall_duration_ms, Σ output_duration_ms / Σ wall_duration_ms)`. | Sessions; denominator is wall duration with structural or timestamped timing provenance. | Derived from timestamped/structured timing. | Lanes may overlap; wall duration can include human absence. | `wall_duration_ms=0`; timing provenance is only `sort_key_estimated`; lane coverage below threshold. **Mix flag: source timing vs ingest/sort-key timing.** |
| **interaction-latency asymmetry** | Relative delay of agent-side vs user-side response loops. | Prefer log ratio: `median(log1p(agent_response_ms)) - median(log1p(user_response_ms))`; plain ratio render optional. | Sessions; denominator is paired response-gap observations in latency profiles. | Derived structural timing. | Agent response includes tool execution; user latency caps idle gaps. | Missing latency profile; too few paired gaps; unpaired/untimestamped sessions dominate. **Mix flag: tool time inside agent response.** |
| **engaged-vs-wall efficiency** | Fraction of wall time that falls inside message-clustered engaged intervals. | `Σ engaged_duration_ms / max(Σ wall_duration_ms, 1)`. | Sessions; denominator is wall duration. | Derived deterministic, not attention evidence. | Five-minute idle threshold is arbitrary; not keyboard focus or human presence. | Missing wall duration; timestamp coverage insufficient; user asks for “attention/productivity” without accepting the construct boundary. **Mix flag: derived engagement vs real attention.** |
| **stuck-tool density** | Rate of provider tool starts left unresolved beyond stuck threshold. | `Σ stuck_tool_count / max(Σ timestamped_tool_start_count, 1)`, rendered per 100 starts. If only `tool_use_count` exists, use it only with coverage caveat. | Sessions or actions; denominator is timestamped provider tool starts. | Structural when paired start/output evidence exists. | Providers differ in tool event fidelity; long-running valid tools can look stuck. | No latency profile; action/tool coverage gap for origin; timestamped start count unavailable. |
| **compaction pressure** | Context-window pressure as observable compaction frequency. | `1000 * Σ compaction_count / max(Σ message_count, 1)`. Optional token-denominator variant: `Σ compaction_count / Σ total_tokens`. | Sessions; denominator is messages by default. | Structural if `session_events.event_type='compaction'`; derived if inferred from text. | Provider/harness compaction policy differs; resumed logical sessions can spread compactions over physical sessions. | Compaction events unavailable for origin; physical/logical lineage unresolved for requested frame. **Mix flag: physical vs logical session grain.** |
| **substantive density** | Share of messages counted as substantive content rather than metadata/tool/system noise. | `Σ substantive_count / max(Σ message_count, 1)`. | Sessions or messages; denominator is all messages in eligible sessions. | Derived deterministic classification. | “Substantive” is an archive rule, not human value; provider block/message boundaries differ. | Message parser coverage incomplete; provider format does not expose comparable message roles/blocks. |
| **outcome-conditioned cost** | Cost conditioned on structural result boundary. | `Σ selected_cost_basis_usd / count(distinct logical_session_id)` grouped by `terminal_state × exit_code_class`, where `exit_code_class ∈ {0, nonzero, missing}` from actions. | Sessions with attached actions; denominator is distinct logical sessions per outcome cell. | Mixed: exit code structural, terminal_state derived/inference, cost provider-reported/priced. | Terminal state is not task success; cost basis differs. | Cost provenance not selected; terminal_state confidence below gate; no action outcome coverage. **Mix flag: structural exit_code + heuristic terminal_state + cost basis.** |
| **tool-mix entropy** | Diversity of tool affordance use. | `H = -Σ p(tool_category) log2 p(tool_category)`, optionally normalized by `log2(K)` for comparable 0–1 scale. | Actions; denominator is total action count in group. | Structural/derived depending on action_kind mapping. | Tool taxonomy version changes; origins without action rows would falsely look low-entropy. | Action coverage gaps; fewer than minimum actions; mixed taxonomy versions. |
| **workflow-shape transition matrix** | Empirical transition dynamics between workflow/action states. | `P(j|i)=count(i→j)/Σ_j count(i→j)` over a versioned alphabet. | Actions for structural alphabet; work-events for heuristic workflow labels. Denominator is outgoing transitions from state `i`. | Structural if action alphabet; heuristic if work-event labels. | Alphabet choice dominates interpretation; session boundaries/resumes break transitions. | Fewer than minimum transitions; mixed alphabet versions; heuristic labels requested without tier footnote. |
| **session-redundancy zstd ratio** | Textual repetitiveness / context churn, not quality. | `uncompressed_session_text_bytes / max(zstd_compressed_bytes, 1)` over normalized selected text. Higher means more compressible/redundant. | Sessions or messages; denominator is compressed bytes of normalized text projection. | Derived deterministic. | Boilerplate/system prompts inflate redundancy; repeated inherited prefixes across physical lineage overcount. | Selected projection unavailable; normalized text empty; logical lineage dedup not applied where requested. **Mix flag: physical vs logical lineage grain.** |
| **thread-cost Gini** | Inequality/concentration of cost across work threads. | `Gini({thread_cost_basis_usd}) = Σ_iΣ_j |x_i-x_j| / (2 n² mean(x))`. | Threads; denominator is thread count and mean selected cost basis. | Derived over cost basis; depends on thread confidence. | Thread construction confidence; cost basis/provenance heterogeneity. | Thread confidence below gate; cost basis mixed or unavailable; too few threads. **Mix flag: provider/API/subscription cost basis.** |
| **subagent fan-out** | Branching intensity of orchestrated multi-agent work. | `count(child runs where role='subagent') / count(parent main runs)`; distribution variant reports p50/p90 child count per parent. | Runs or threads; denominator is main runs or root threads. | Structural/derived from run projection and topology. | Hidden subagents not captured; child links inferred from report text may be low confidence. | `session_runs` not materialized; subagent role/link confidence missing; physical/logical duplicate children unresolved. **Mix flag: physical vs logical run lineage.** |
| **pathology epidemiology** | Rate of detector-defined pathologies over time/model, not general badness. | `pathology_session_count / eligible_session_count`, grouped by `iso_week × model`, with detector version pinned. | Sessions; denominator is sessions eligible for the detector in that week/model cell. | Derived structural if detector uses structured events; heuristic if detector uses text/labels. | Detector version drift; capture coverage changes over time/model. | Detector not run/current; eligible denominator unavailable; model coverage too sparse. **Mix flag: detector tier + model/origin coverage.** |
| **credit-vs-API divergence** | Difference between subscription/credit accounting and API-list-equivalent accounting. | `(api_equivalent_usd - subscription_equivalent_usd_or_credits) / max(api_equivalent_usd, ε)`. Also render absolute delta. | Sessions or usage timeline buckets; denominator is API-equivalent selected cost basis. | Mixed but intentionally declared: cost-basis comparison. | Credit formula/provider plan changes; cache reads are free/discounted differently by plan. | Either basis absent; pricing catalog stale; subscription-credit view not configured. **Mix flag: API vs subscription/credit basis.** |

## Denominator hazards that must be flagged automatically

**[evidence] Cached vs fresh tokens are dangerous.** Codex input includes cached tokens, and the code has a dedicated disjoint-lane function because billing fresh input and cache read additively would double-count (`polylogue/storage/sqlite/archive_tiers/write.py:2608-2636`). Any token measure using cache or reasoning lanes must declare whether raw provider fields are inclusive or already decomposed.

**[evidence] Provider vs origin is already a transition seam.** `insights/registry.py` maps legacy `provider`/`source_name` to public `origin` fields (`registry.py:99-143`). Measures should use `origin` as the public grouping dimension but retain provider/source provenance internally.

**[evidence] Cost basis must not collapse to `total_usd`.** `CostBasisPayload` says bases are independent and do not sum because the same usage may be expressed in multiple accounting views (`polylogue/archive/semantic/pricing.py:101-108`). Any denominator using cost must name the basis.

**[evidence] Timing provenance is not uniform.** `SessionEvidencePayload` carries `timing_provenance` (`archive_models.py:86`), and docs explain that engaged/tool/wall durations have different construct boundaries (`docs/insights.md:69-83`). Latency measures must suppress or split rows by timing provenance.

**[evidence] Physical vs logical lineage matters.** `session_profiles` carries both `session_id` and `logical_session_id` (`polylogue/storage/sqlite/archive_tiers/index.py:799-802`), and threads/topology carry derived grouping. Measures over compaction, redundancy, thread cost, and subagent fan-out should default to logical grain when the question is about work, not physical archive rows.

## Uncertainty layer

**[proposal] Every measure gets an `UncertaintySpec`.**

For proportions and rates where the denominator is a count, use Wilson intervals:

```text
rate = k / n
ci = wilson(k, n, confidence=0.95)
render: 12.4% [9.8, 15.6], n=742
```

Use this for stuck-tool density, pathology epidemiology, substantive density, compaction-pressure-as-rate if normalized to count, and tool/error rates.

For skewed medians and pXX values, use bootstrap CI plus `n`. Cost, latency, thread cost, fan-out, and redundancy are skewed; do not report only mean±SE.

```text
median = median(x)
ci = bootstrap_percentile(x, statistic=median, B=2000, confidence=0.95)
render: p90=18.2s [12.1, 29.7], n=388
```

For two-sample comparisons, render an effect size, not just a p-value. Use:

- proportions: risk difference and risk ratio; optionally Cohen’s h.
- skewed numeric distributions: Cliff’s delta plus median difference.
- transition matrices/tool distributions: Jensen-Shannon divergence, with permutation test only if needed.
- Gini: bootstrap CI and absolute delta between groups.

For census counts over the entire selected archive, do not invent sampling uncertainty. The registry should render `coverage=census` or `coverage=full-selected-frame`, not a Wilson interval. Bead 9l5.7 explicitly warns not to attach CIs to full-population counts.

**[proposal] Coverage-tier gate:**

```text
coverage=structural       direct structured event/action/session evidence
coverage=provider         provider-reported usage/cost/timing
coverage=derived          deterministic derived field from archive evidence
coverage=heuristic        keyword/rule/model judgment
coverage=mixed-declared   intentionally mixes tiers and names each tier
coverage=suppressed       required tier missing or denominator invalid
```

Composition rule: if a comparison spans different coverage tiers and the measure is not declared `mixed-declared`, the renderer refuses the bare number and emits an actionable error. This is the 9l5.2/9l5.7 doctrine generalized.

## One footnote rendered three ways

Take `stuck_tool_density`:

```python
MeasureSpec(
    id="stuck_tool_density",
    construct="Provider tool starts unresolved beyond stuck threshold per timestamped tool start",
    reducer="ratio",
    column_expr="latency.stuck_tool_count",
    unit_frame="sessions",
    denominator_expr="timestamped_tool_start_count",
    evidence_tier="structural",
    required_coverage={"session_latency_profiles": "present", "timing_provenance": "timestamped_or_structural"},
    uncertainty={"kind": "proportion", "interval": "wilson"},
    footnote_template="n={n}, coverage={coverage}, timing_provenance={timing_provenance}"
)
```

Plaintext:

```text
stuck_tool_density = 3.8% [2.9, 5.1]; n=9,901 tool starts, coverage=structural, timing_provenance=timestamped_pairs
```

JSON:

```json
{
  "measure": "stuck_tool_density",
  "value": 0.038,
  "interval": {"method": "wilson", "level": 0.95, "lower": 0.029, "upper": 0.051},
  "n": {"unit": "tool_starts", "value": 9901},
  "coverage": "structural",
  "timing_provenance": "timestamped_pairs",
  "suppressed": false,
  "formula_version": 1
}
```

MCP content payload:

```json
{
  "type": "measure_result",
  "measure_id": "stuck_tool_density",
  "text": "stuck_tool_density = 3.8% [2.9, 5.1]",
  "metadata": {
    "n": 9901,
    "n_unit": "tool_starts",
    "coverage": "structural",
    "timing_provenance": "timestamped_pairs",
    "uncertainty": "wilson_95",
    "formula_version": 1
  }
}
```

## Implementation sketch

**[proposal] Phase 1: registry-only, no new materialization.** Add `MeasureSpec`, `CoverageGate`, `UncertaintySpec`, and `MeasureResult`. Register five measures first: `cache_amplification_ratio`, `thinking_tax`, `stuck_tool_density`, `tool_mix_entropy`, and `credit_vs_api_divergence`. These stress the provenance system immediately.

**[proposal] Phase 2: evaluator over existing insight payloads.** Build measure evaluators against existing `session_profiles`, `session_latency_profiles`, `usage_timeline`, `tool_usage`, `actions`, `threads`, and run-projection tables. The evaluator receives a selected relation from the query algebra and returns grouped measure results with coverage/uncertainty metadata.

**[proposal] Phase 3: DSL/composer integration.** Add `| measure <id> by <dims> window <window> compare <comparison>` as the general analytics stage. Composer completion should discover measure ids, valid unit frames, required coverage, and warnings before execution.

**[proposal] Phase 4: materialize only when useful.** Expensive measures like zstd redundancy, transition matrices, Gini over large thread sets, and pathology epidemiology can materialize as derived read models. But their materialized tables are cache/projection artifacts of registered measures, not the canonical semantics.

## Open questions for the operator

1. Should the measure registry live under `polylogue/analytics/` as a new subsystem, or under `polylogue/insights/` to stay close to current registry/read-model code? My lean: `polylogue/analytics/` for stats/measure algebra, with `insights` consuming it.

2. What is the default public cost basis for reports: `api_equivalent_usd`, `provider_reported_usd`, or a dual view every time? My lean: no default for comparisons; require explicit basis.

3. Should logical-session folding be the default for all session-level measures? My lean: yes for “work” questions, no for “archive row/capture” questions, and the registry must expose the grain.

4. Should heuristic measures be allowed in public reports by default? My lean: yes only with visible tier footnotes and no “quality/productivity” naming.

5. Should external annotation/assertion measures join this registry immediately, or wait until assertion-batch import exists? My lean: register the unit-frame now, but suppress until batch/schema coverage exists.

## What’s missing

I did not run the live daemon, query the live 38GB archive, or validate any formulas against real rows. I inspected the uploaded snapshot and code package only; live code may have moved since the bundle.

I did not read every 9l5 child fully beyond the directly relevant ones, though I did inspect `9l5`, `.2`, `.4`, `.7`, `.12`, and the usage/cost epic.

I did not verify whether `zstandard` is already a dependency in the current branch; the zstd measure may need either a dependency gate or a fallback implementation.

I did not inspect all storage queries for `session_latency_profiles` and `session_model_usage`; the exact evaluator should verify whether `timestamped_tool_start_count` is available directly or must be derived.

I did not check whether topology/run-projection materialization is complete enough for subagent fan-out over the live archive; the schema supports it, but coverage needs a live audit.
