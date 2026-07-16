## [assistant 360]
I take A11 as the brief: design `find <query> | compact` as a token-budgeted, decision-dense corpus digest for external LLM handoff, with block scoring, lineage dedup, budget fitting, and an honest drop manifest. The snapshot is dated 2026-07-05, so code-line claims below should be verified locally before turning this into Beads or implementation work. fileciteturn1file0

## Defended recommendation

[proposal] Implement `compact` as a **new projection/render preset over the existing read algebra**, not as a new context subsystem and not as a mutation/action. Concretely:

```text
find <query> | compact --budget 60000 --audience external-llm --manifest full
```

should lower to:

```text
SelectionSpec(query)
→ CompactProjectionSpec(scoring_policy, lineage_policy, budget_policy, manifest_policy)
→ RenderSpec(format=markdown|json, layout=corpus-compaction-pack)
```

The output should be a `CorpusCompactionPack` envelope with two renderable parts: a human/LLM-facing digest and a machine/auditor-facing manifest. The digest is what you paste into GPT-pro/Gemini; the manifest is what lets a later Polylogue run, or a skeptical auditor, reconstruct what was included, summarized, deduplicated, clipped, or dropped.

[evidence] This fits the current architectural direction better than a new subsystem because the swarm brief already defines the read surface as `Query × Projection × Render`, with Query selecting sessions/units, Projection choosing evidence families/body policy/exclusions, and Render deciding format/destination/layout. It also explicitly says named views are currently a closed preset vocabulary and should move toward composable projection algebra and user-defined views/macros. fileciteturn1file4

Runner-up rejected: extend `build_context_image` / `compile_context` with a `detail_level="corpus_compact"` mode. That is tempting because those already have token-budgeted context images and omission accounting. I reject it because `compile_context` is a seed/ref handoff compiler; `compact` is a **cohort optimizer**. It needs cross-session ranking, lineage-family dedup, block-level material-origin filtering, fairness constraints, external-LLM manifests, and decision-density scoring. If we shove this into `ContextImage`, `ContextImage` becomes a second read algebra.

## Layer 1 — today’s substrate

[evidence] Polylogue is already surprisingly close on the primitives. `messages.material_origin` exists precisely because role is not enough: Claude Code can carry command wrappers, provider context bundles, and tool-result envelopes through `role=user`, so `material_origin` is the authoredness/material axis for projections and accounting (`docs/data-model.md:72-85`; `polylogue/core/enums.py:176-192`). That directly supports A11’s instruction to drop `runtime_protocol`, `runtime_context`, and most `tool_result` material. fileciteturn1file0

[evidence] The tool-result side is also not merely prose. `blocks` has `tool_result_is_error` and `tool_result_exit_code`, and the `actions` view exposes paired tool-use/tool-result rows with `is_error` and `exit_code` (`polylogue/storage/sqlite/archive_tiers/index.py:182-224`, `324-343`; `docs/internals.md:175-185`). This matters because `compact` should keep error→fix pairs and terminal outcomes based on structure, not regex vibes.

[evidence] Lineage dedup is also partially there. The lineage model says fork/resume/subagent/auto-compaction copies physically replay parent context, and index schema v12 stores only the divergent tail plus `session_links.branch_point_message_id` and `inheritance`, while reads compose parent prefix + child tail (`docs/internals.md:214-224`). The design doc’s decision is “default reporting unit: logical,” with physical artifacts still recoverable (`docs/design/session-lineage-model.md:171-176`). The earlier session also confirmed prefix-dedup and logical composition are load-bearing parts of the live direction. fileciteturn1file2

[evidence] `compile_context` / `build_context_image` already prove the product has the right “bounded context image with omissions” instinct. `compile_context` is documented as reusing query/read primitives and recording unsupported or missing inputs as omissions, not creating a parallel memory store (`polylogue/api/archive.py:2271-2277`). `ContextImage` already carries `segments`, `object_refs`, `evidence_refs`, `assertion_refs`, `omitted`, `caveats`, and `token_estimate` (`polylogue/context/compiler.py:103-117`). The MCP `build_context_image` tool is explicitly a thin lens over the same shared compiler and mentions token-budgeted accumulation and omission accounting (`polylogue/mcp/server_context_tools.py:1-8`, `48-67`).

[evidence] But today’s context image is not enough. It picks seed sessions, compiles `messages`, `temporal`, `chronicle`, query-unit rows, and injectable assertions; it uses bounded message windows and a tail-biased token fit (`polylogue/api/archive.py:301-365`, `368-415`, `2356-2495`). `context_image_payload` clamps `max_sessions` to 20 and defaults to message windows (`polylogue/api/archive.py:2518-2575`). That is useful for handoff, but it is not a corpus compactor.

[evidence] The single-client contract is also not ready yet. B8 says the projection algebra exists as `QueryProjectionSpec = SelectionSpec × ProjectionSpec × RenderSpec`, but live `read` still executes named views and `projection_spec.py` is “a contract builder, not an executor.” B8 also says there is no single wire request DTO and `preview` does not yet exist. fileciteturn1file12

So: today, Polylogue can fake this by `find <query>`, `read --view context-image`, and maybe external summarization. It cannot yet do the thing A11 wants honestly and repeatably.

## Layer 2 — near-term substrate change

[proposal] Add `compact` as a **projection preset first**, not a whole analysis framework. The near-term target is a deterministic, non-LLM compactor that produces an evidence pack good enough for a GPT/Gemini handoff.

The syntax should be something like:

```text
polylogue find 'repo:polylogue semantic:"hot daemon composer"' \
  | compact --budget 60000 --layout external-llm --manifest full
```

and later, in the composer:

```text
find repo:polylogue semantic:"delegation rhetoric"
  | compact budget:60k audience:gemini
```

The output envelope:

```json
{
  "kind": "CorpusCompactionPack",
  "pack_ref": "compact:...",
  "query_ref": "query-run:...",
  "result_relation_ref": "result:...",
  "budget": {"target_tokens": 60000, "estimated_tokens": 58740},
  "digest_markdown": "...",
  "manifest": {...},
  "evidence_refs": [...]
}
```

### Block-selection scoring

[proposal] The scorer should operate on candidate blocks/windows, not whole sessions. The selected query gives a cohort of logical sessions; each logical session yields candidate units:

`message`, `block`, `action`, `assertion`, `session_event`, `session_summary`, `terminal_outcome`, and eventually `artifact`.

The first pass is hard filtering:

```text
drop by default:
  material_origin in {runtime_protocol, runtime_context}
  material_origin == generated_context_pack unless explicitly requested
  block_type == tool_result if success, long, and unreferenced
  low-signal context boilerplate, provider wrappers, repeated tool stdout

keep by default:
  material_origin in {human_authored, operator_command}
  assistant_authored messages with decisions, plans, results, contradictions
  tool_result blocks with is_error=1 or nonzero exit_code
  tool_use/tool_result pairs that form error→fix→verify chains
  final summaries, terminal outcomes, merged PR/commit/verification summaries
  assertions/judgments attached to selected sessions
  compaction summaries and continuation boundaries
```

[evidence] This filter is grounded in the data model: `material_origin` exists to distinguish human-authored prose from provider/runtime material, and structured tool-result outcomes exist specifically to avoid regex-guessing failures (`docs/data-model.md:76-85`; `docs/internals.md:175-185`). The situation brief also identifies structured tool outcomes, error flags, and exit codes as Polylogue’s sharpest defensible wedge. fileciteturn1file13

[proposal] After filtering, score each candidate with an additive score vector:

```text
score =
  + query_relevance                 # lexical/semantic hit score
  + authoredness_weight             # human/operator > assistant > generated
  + decision_signal                 # "decided", "we will", "root cause", "recommendation"
  + outcome_signal                  # "merged", "verified", "tests pass", terminal status
  + error_fix_signal                # failed action + subsequent fix + later success
  + contradiction_signal            # claim vs tool outcome mismatch, if present
  + novelty_within_lineage_family   # not already represented by ancestor/sibling
  + artifact_link_signal            # commit/report/file/link attached
  + recency_or_terminal_position    # late-session summaries/outcomes
  + diversity_bonus                 # covers another session/model/task
  - spam_penalty                    # runtime/tool/protocol bulk
  - redundancy_penalty              # duplicate text/prefix/repeated stdout
  - length_penalty                  # huge block with low density
```

Use named reasons in the manifest, not just a number. A block should say:

```json
"score_reasons": ["human_authored", "query_hit", "decision", "terminal_outcome"]
```

This matters because an external LLM should see why something is present, and an auditor should be able to tell whether the scorer quietly overfit to recency or query relevance.

### Error→fix pair detection

[proposal] Treat a failed action as a small narrative unit, not an isolated block.

A retained error window should include:

1. the command/tool-use summary,
2. the structured failure outcome,
3. the assistant/user interpretation immediately after,
4. the next material fix attempt,
5. the verifying success or terminal unresolved status.

For example:

```text
[ERROR-FIX PAIR ref=action:...]
- failed: pytest tests/foo.py, exit=1
- diagnosis: assistant says fixture path was wrong
- fix: edited tests/foo.py
- verify: pytest tests/foo.py, exit=0
```

This is better than including long stdout or dropping the failure entirely. It directly supports external analysis of agent behavior: “what happened, what did the agent claim, what did the tools say?”

### Lineage dedup

[proposal] `compact` should deduplicate at **logical lineage-family grain**, not by raw text.

Algorithm:

First, group selected physical sessions by logical root / lineage family. Use the existing `session_links` and logical session identity. Within each family, sort by lineage topology and time.

Second, build a composed logical transcript for each selected leaf: inherited parent prefix + divergent tail. But for the compaction pack, emit inherited material **once** per lineage family.

Third, when both parent and child sessions are in the cohort, include:

```text
<<<LINEAGE-FAMILY root=... logical=... selected_physical=N>>>
shared prefix: included once / summarized / dropped
child A tail: included selected blocks
child B tail: included selected blocks
```

Fourth, if the branch point is unresolved or dangling, do not dedup silently. Mark:

```json
"lineage_status": "unresolved",
"dedup_policy": "physical_fallback",
"degraded_reason": "missing branch_point_message_id"
```

[evidence] This follows the existing lineage doctrine: dedup is by lineage, not content hash; default reporting unit is logical; raw physical artifacts remain recoverable (`docs/design/session-lineage-model.md:158-176`). It also avoids the known silent-failure class from earlier lineage bugs, where incomplete composition was dangerous precisely because the user could not tell what was missing. fileciteturn1file6

### Budget fitting

[proposal] Use a **stratified greedy water-fill**, not pure knapsack and not LLM summarization as the first pass.

Pure knapsack maximizes score/token but tends to starve smaller sessions, late-stage outcomes, and minority providers. LLM summarization first is worse: it destroys auditability before the deterministic manifest exists. The compactor should be deterministic first; LLM summarization can be a later optional transform with its own assertion/provenance record.

Budget plan for `B = target_tokens`:

```text
Reserve:
  3%  pack header, methodology, query spec, caveats
  7%  corpus map and per-session skeletal summaries
  10% manifest-visible drop summary / citation table
  80% evidence blocks and dense extracted windows
```

For a 60k pack, that leaves roughly 48k tokens for evidence. For a 200k pack, roughly 160k.

Packing phases:

Phase 0: build candidate inventory with token estimates and score reasons.

Phase 1: include mandatory skeletons for every selected logical session: title, origin/model, date range, query match reason, terminal status, lineage role, and counts by material origin. These are tiny, but prevent the external LLM from thinking omitted sessions do not exist.

Phase 2: include mandatory high-value anchors: human-authored turns, explicit decisions, error→fix pairs, terminal outcomes, and active assertions/judgments.

Phase 3: fill remaining budget using stratified score/token density. The strata should be lineage family, session, task type, provider/model family, and evidence kind. This prevents one massive session from eating the pack.

Phase 4: degrade in this order:
1. clip long retained blocks;
2. collapse contiguous low-value runs into deterministic summaries like “17 successful tool results omitted”;
3. replace low-score sessions with skeleton only;
4. drop whole low-score sessions only if the user asked for hard budget and skeletons exceed budget;
5. if even skeletons exceed budget, emit an index-only pack with explicit failure state.

Never silently trim the tail. The current context compiler is tail-biased because it is for handoff continuity (`polylogue/api/archive.py:368-415`). Corpus compaction should be **decision-density-biased**.

### Fidelity drop-manifest

[proposal] The drop manifest is not optional. It is the feature.

Minimum manifest:

```json
{
  "source": {
    "query_text": "...",
    "lowered_selection": "...",
    "archive_epoch": "...",
    "created_at": "...",
    "projection": "compact:v1"
  },
  "budget": {
    "target_tokens": 60000,
    "estimated_tokens": 58740,
    "estimator": "polylogue_words_v1",
    "hard_budget": true
  },
  "corpus": {
    "logical_sessions_total": 37,
    "physical_sessions_total": 54,
    "lineage_families_total": 12,
    "blocks_considered": 18492,
    "blocks_included": 412,
    "blocks_summarized": 260,
    "blocks_dropped": 17820
  },
  "drop_by_reason": {
    "runtime_protocol": 5021,
    "runtime_context": 817,
    "tool_result_success_spam": 9102,
    "lineage_duplicate_prefix": 2110,
    "budget_low_score": 770,
    "redacted": 0
  },
  "sessions": [
    {
      "logical_session_ref": "...",
      "physical_session_refs": ["..."],
      "included_tokens": 1800,
      "dropped_tokens_estimate": 9200,
      "included_blocks": 21,
      "summarized_blocks": 4,
      "dropped_blocks": 310,
      "lineage_policy": "shared_prefix_once",
      "boundary_markers": ["S001"]
    }
  ],
  "anchors": {
    "S001.M004.B002": "session:.../message:.../block:2"
  }
}
```

Digest markers should be terse but machine-stable:

```text
<<<PACK query_run=qrun:... result=result:... budget=60000>>>

<<<SESSION S001 logical=logical:... physical=session:... origin=claude-code-session tokens=1840>>>
[ANCHOR S001.M012.B000 score=91 reasons=human_authored,query_hit,decision]
User: ...

[ANCHOR S001.A003 score=88 reasons=tool_error,fix_pair]
Tool failed: pytest ..., exit=1
Fix/verify: ...

<<<OMITTED S001 reason=tool_result_success_spam count=87 est_tokens=9400>>>
<<<END SESSION S001>>>
```

The receiving LLM can reason over the digest, but every factual claim can be traced back by anchor. That is what makes this an analysis substrate, not a prompt hack.

## Layer 3 — full direction

[proposal] In the full design, `compact` should become one member of a larger family of **external-analysis projections**:

```text
find <query> | compact
find <query> | evidence-pack
find <query> | specimen-gallery
find <query> | report-pack
find <query> | annotate-with <schema>
```

But `compact` comes first because it unlocks the parallel-R&D flywheel: package the archive’s own prior work, send it to another frontier model, import the returned analysis as assertions, then query those assertions and run the next loop.

[evidence] This aligns with the strategic brief’s binding constraint: the missing gate is demonstrated use that an outsider can see, and the finished demos so far prove honesty/machinery more than agent/operator uplift. fileciteturn1file13 It also matches the raw-log idea of composing long, impactful prompts from filtered AI conversations, raw logs, and dynamic queries instead of dumping everything possibly relevant into context. fileciteturn1file7

[proposal] The full object model should be:

```text
query_definition
query_run
result_relation
compact_pack
external_llm_run
annotation_batch
assertions/judgments
followup_query_run
report_artifact
```

This is where your earlier query-object idea matters. `find <query> | compact` should not just print Markdown. It should optionally record:

```text
query_run_ref
result_relation_ref
compact_pack_ref
render_artifact_ref
```

Then an external LLM’s output can attach to `compact_pack_ref` or `result_relation_ref` as an assertion batch. Later:

```text
assertions where scope:compact_pack:<id> and key:directive_intensity
```

or:

```text
from result:<id> | compact --budget 200k
```

[evidence] C10 already proposed that committed composer runs write recall entries to `user.db` with query text, resolved spec, result fingerprint, and timestamp; named recall entries become macros. fileciteturn0file5 My improvement is to split that into query definition / query run / result relation / render artifact, because a compact pack is not just “history”; it is a reusable analysis object.

## Why this is distinct from `build_context_image` / `compile_context`

[evidence] `build_context_image` is an MCP-facing context assembly tool: select sessions through query algebra, then compile bounded context image payloads with token accumulation and omission accounting (`polylogue/mcp/server_context_tools.py:1-8`, `48-67`). `compile_context` is the shared engine, deliberately not a new memory store or handoff ontology (`polylogue/context/compiler.py:1-6`; `polylogue/api/archive.py:2271-2277`).

[proposal] `compact` should reuse that discipline but not that shape.

`compile_context` answers: “What context should I hand to an agent so it can continue/review/debug?”

`compact` answers: “Given a cohort of sessions, what is the highest-value, lowest-spam, lineage-deduplicated evidence digest for an external model to analyze?”

That difference changes everything: scoring, lineage grouping, retained evidence kinds, budget allocation, and the manifest.

So the implementation should probably share lower-level helpers: token estimation, `ContextSegment`-like sections, evidence refs, omission objects. But the top-level DTO should be separate:

```text
ContextImage              # handoff / continuation
CorpusCompactionPack      # external analysis / R&D flywheel
```

Both are projections over read algebra. Neither should become a silo.

## Browser-auto-capture interaction

[proposal] The browser-auto-capture flywheel should treat external LLM work as first-class evidence.

Flow:

```text
1. Polylogue captures prior local AI sessions and browser R&D.
2. Operator/agent runs: find <query> | compact --budget 200k.
3. Pack is fed to GPT-pro/Gemini/DeepResearch/Fable/whatever.
4. Browser auto-capture imports the external run/result.
5. External output is linked back to compact_pack_ref.
6. Agent imports structured annotations/assertions from the external output.
7. Polylogue queries those annotations for the next compact pack.
```

This is not “browser capture as another ingestion source.” It is “browser capture closes the loop on outsourced cognition.” The digest packages Polylogue’s own past R&D for the next lane, and the captured external lane becomes new archive evidence.

[evidence] The broader project framing is already “local-first archive and analysis layer for AI/agent sessions” across providers, with daemon, browser capture, cost, lineage, and analysis ambitions. fileciteturn1file13 The raw log also repeatedly points at using LLMs to process large prior chat/log corpora into more useful higher-order forms rather than hand-carrying everything. fileciteturn1file15

## Concrete Beads I would add

[proposal] I would add one P1/P2 bead under the query/read-algebra + external-legibility axis:

**`find <query> | compact`: token-budgeted corpus compaction projection**

Description: Build a deterministic corpus-compaction projection over selected AI-session cohorts for external LLM handoff. It scores blocks by authoredness/material origin, query relevance, decision/outcome/error-fix value, and redundancy; deduplicates lineage prefixes; fits a hard token budget; emits an explicit drop manifest with source anchors.

Design:
- Add `CompactProjectionSpec` and `CorpusCompactionPack` payload.
- Add block/window candidate builder over logical sessions.
- Implement material-origin filtering and structured tool-outcome retention.
- Implement lineage-family grouping and shared-prefix dedup.
- Implement stratified greedy budget fitting.
- Render Markdown + JSON manifest.
- Add CLI terminal/projection `| compact`.
- Reuse existing query/read/ref payloads; do not fork `compile_context`.

Acceptance:
- A fixture with runtime protocol/context/tool-result spam compacts to a digest that excludes those blocks by default and reports counts by `material_origin`.
- A fixture with failed tool result → fix → verify keeps the pair and cites all source refs.
- A fork/resume fixture selected with both parent and child includes inherited prefix once and reports duplicate-prefix omissions.
- A 60k-token budget test proves deterministic degradation order: full blocks → clipped blocks → summarized runs → skeleton-only sessions → explicit budget failure.
- Manifest round-trip test: every digest anchor resolves to source session/message/block/action.
- Comparison test proves `read --view context-image` and `| compact` remain separate projections sharing helpers but not payload shape.

Related beads: `4p1`/read algebra, `fnm` pipeline/projection roadmap, `fnm.13` set algebra, `t46` contract-thin surfaces, `20d` interactive performance, `37t` context/memory loop, `4ts` lineage truth, `3tl` external legibility.

## Open questions for the operator

Should `compact` produce only deterministic extraction, or may it optionally call a local/external LLM to summarize low-score runs? My recommendation: deterministic v1 only; LLM summaries become generated analysis packs/assertions in v2.

Should the manifest be included inside the pasted digest, or emitted as a sidecar JSON artifact? My recommendation: both. The pasted digest needs a compact drop summary; the full manifest should be sidecar JSON.

Should `compact` default to logical-session grain even if the query selected physical sessions? My recommendation: yes, with `--physical` as an explicit opt-out.

Should browser-auto-captured external outputs automatically link back to the compact pack that prompted them? My recommendation: yes if the pack ref is present in the prompt header; otherwise offer a manual `polylogue link artifact --derived-from compact:<id>` affordance.

## What’s missing

I did not run the live daemon or execute code against the real archive. The code-line references are from the uploaded snapshot and should be verified locally.

I did not inspect a live `build_context_image` output, only the code and uploaded design docs.

I did not verify whether a newer post-snapshot bead already exists for A11. The A11 file itself is the fresh prompt I used. fileciteturn1file0

I did not design the full annotation-import loop here; I only connected `compact` to the earlier assertion/annotation/query-object direction.

The exact block-scoring weights should be calibrated on a real corpus slice. The important design claim is the score vector + manifest, not the first constants.
