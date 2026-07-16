According to the 2026-07-06 bundle/chatlog state, the prior pass did a lot of real integration: it turned the GPT-Pro corpus into three new epics, filed a Tier-0 bug cluster, enriched about 25 existing beads, and recorded the main sequencing doctrine. The resulting graph reportedly reached 471 issues, with new epics for analysis provenance, work-graph units, and missing units; the same closing note says the deep-research lanes D01/D02/D07 were off-brief and still needed fresh reruns. 

My read: the previous pass captured the convergent substrate well. What it missed is mostly second-order: residual work wasn’t always made executable, a few filed bead diagnoses are stale or too broad, and some strategic lanes were left as notes instead of becoming visible tech-tree nodes.

## 1. The biggest miss: `ejm3` should not stay as “done but in-progress”

The chatlog says the integration ended with “~45 new beads, 3 new epics, ~25 enrichments,” and the living note lists the final integration map, residuals, and sequencing doctrine.  The pasted scratch transcript also says the whole corpus was “digested” and pushed, while simultaneously listing the three off-brief deep-research lanes as unresolved. 

That is the first thing I would fix. `ejm3` should either close with explicit successor beads for the residuals, or remain open with a visible checklist. Right now it risks becoming a fuzzy “integration happened” bead whose unresolved pieces are only in notes.

Concrete change: split the residuals into three execution-grade research beads:

`3tl.competitive-landscape-bakeoff`: rerun D01 against LangSmith, Langfuse, Helicone, Braintrust, OpenTelemetry GenAI, Rewind/QS-style personal capture, and local log/SQLite tools. Output is not “market research”; it is a positioning matrix plus one README-ready category claim.

`mhx.local-embedding-stack-shootout`: rerun D02 against the real retrieval problem: fork/resume positives, Polish recall probes, long tool-heavy sessions, and local hardware constraints.

`fs1.eval-target-matrix`: rerun D07 as an adapter decision: Atropos, Terminal-Bench, SWE-bench, Inspect/Verifiers-style evals, and internal spec-card trajectory schema.

Then close `ejm3` only after those child beads exist, or retitle it as the active integration epic.

## 2. The off-brief “deep research” residual is more important than the note made it sound

The prior pass correctly identified that D01/D02/D07 went off-brief and re-answered the analysis-substrate question instead of doing competitive landscape, local model stack, and RL/eval landscape.  But it treated that as a residual, not a front-line gap. I would promote it.

For D01, the competitive context changed what I’d recommend. LangSmith is positioned around LLM observability, tracing, monitoring, evals, and agent debugging; Langfuse is an open-source LLM engineering platform with tracing, evals, prompt management, and self-hosting; Helicone is an AI gateway/observability system focused on routing, logging, caching/rate limits, costs, and prompt management. ([LangChain][1])

That means Polylogue’s white space should be phrased very tightly:

Polylogue is not primarily an online production observability platform. It is a **local post-hoc system of record / flight recorder for AI work**: cross-provider, export/browser-capture based, evidence-linked, reconstructive, and designed to answer “what actually happened across my AI work?” after the fact. That positioning should become a testable product claim, not just copy.

For D02, current local retrieval candidates now include Qwen3 Embedding/Reranker models, BGE-M3, and Nomic embedding models. Qwen3 Embedding has embedding and reranking model families with multilingual/code retrieval claims; BGE-M3 supports dense retrieval, lexical matching, multi-vector interaction, >100 languages, and long inputs; Nomic’s text embedding line has Matryoshka-style variable dimensionality and local deployment routes. ([arXiv][2])

So the local-embedding bead should not pick a model in prose. It should create a benchmark harness over Polylogue’s own labeled positives: lineage fork/resume pairs, same-commit episodes, Polish `ł/Ł` query fixtures, and long tool-output sessions. The output should be a default-lane decision with recall@k/MRR/latency/index-size, not a subjective “best model” memo.

For D07, Atropos is a credible target because it is an async RL environment framework for LLM agents with trajectory/environment structure; Terminal-Bench and SWE-bench are also relevant because they encode terminal-agent and real GitHub issue repair tasks. ([GitHub][3]) Polylogue should not hard-code “Atropos export” as the identity of the eval lane. It should define an internal `SpecCard + Trajectory + EvidenceRefs` schema first, then adapters.

## 3. `cpf.6` needs a wording correction before coding agents waste time

The corpus itself later narrowed the temporal bug: the problem in the inspected code was uncontrolled wall-clock use inside relative-date parsing, not necessarily a module-level `RELATIVE_BASE` frozen at import/process start. It also narrowed the `sort_key_ms=COALESCE(...,0)` issue: some read paths already handle NULLs explicitly, so this needs a targeted ordering/window audit rather than a blanket claim. 

So update `cpf.6`:

Old framing: “RELATIVE_BASE frozen at import; sort_key 1970 everywhere.”

Better framing: “relative-date parsing lacks a clock seam; some window/sort paths synthesize or mishandle timeless sessions.”

Acceptance criteria should be:

`parse_date` and query lowering accept an injected/frozen clock.

`since:7d` under a frozen clock is deterministic and shifts only when the injected clock shifts.

No direct `datetime.now()` remains in query-time parsing paths except behind the clock seam.

A targeted audit enumerates every `sort_key_ms` / `COALESCE(...,0)` ordering/window path and classifies it as fixed, safe, or intentionally synthetic.

Timeless sessions are excluded from timed lower-bounded windows by default, but visible with `include_timeless` and explicit `time_confidence`.

The temporal spec’s own test strategy already supports this: hermetic clock, backfill invariant, comparator properties, timeless-window regression, half-open tiling, and lint fixtures. 

## 4. `l4kf.2` appears to carry a stale cross-machine-sync hazard

The living note records a cross-machine-sync concern from the corpus: `raw_id` allegedly contains machine-local `source_path`, causing the same session to get different raw IDs across machines.  In the current packed code I inspected, acquisition records use the raw blob hash as `raw_id` when available, not `source_path`. That changes the risk.

The real sync risk is subtler:

Content hash identity is good for deduping bytes.

Observation provenance still needs to preserve machine/source path/native identity as a multimap.

Session identity can still collide through `origin:native_id`, especially around non-injective origin mapping.

Two machines may ingest the same logical session through different acquisition routes; the byte identity may match while path/native metadata differs.

So update `l4kf.2` from “raw_id embeds source_path” to “content-addressed raw identity is okay; acquisition provenance and session identity still need explicit union semantics.”

Acceptance criteria should include two fixtures:

Same bytes, different `source_path` on two machines: one raw blob identity, two acquisition observations.

Different bytes, same origin/native ID across machines: collision/quarantine or explicit conflict state, never silent overwrite.

## 5. `at44` should not merely “wire user_settings”; it needs a scoped settings model

The previous pass live-verified that `user_settings` exists but is dead, then decided to keep the table and wire it rather than fold runtime settings into assertions.  I agree with keeping a settings table for state: settings are not epistemic claims.

But the risk is that `user_settings` becomes a flat global key-value table. That would recreate the dead-table problem in a different form.

The bead should require:

A registry of allowed runtime setting keys.

A partition between deployment secrets and runtime preferences.

Scope support: global, repo, origin, surface, maybe actor/harness.

A resolver that reports the winning layer and shadowed candidates.

Typed validation and migration behavior.

Subscription tier can be the first consumer, but the design should not hard-code a one-setting table around subscription tier. This matters because cost correctness wants subscription tier, context scheduler wants runtime policy, MCP wants saved prompt/resource preferences, and install/doctor wants configured-repo state.

## 6. `37t.15` should be promoted above most frontier work

The prior pass filed `37t.15` as the upsert assertion chokepoint for the blackboard ACTIVE hole, and the living note lists it among Tier-0 bugs.  I would treat this as P1 at minimum, possibly P0 if any coordination or recall work is about to land.

Reason: almost every exciting thing in the tech tree assumes agent-authored content does not become operator-grade memory by accident. Query findings, recall packs, annotation imports, distillery candidates, notifications, goals/decisions, and context scheduler all rely on this gate.

The invariant should be inside `upsert_assertion`, not at individual call sites:

`author_kind != user ⇒ status=CANDIDATE + context_policy.inject=false`

and terminal judged rows must not be resurrected by later agent writes. The master synthesis explicitly says recursive safety is one chokepoint, not per-path, and names the `blackboard_post` active-hole class. 

## 7. There is likely Beads graph hygiene debt from the integration

My mechanical check of the packed Beads export found three missing dependency refs around the embedding lane: `emb-targets` / `emb-eval` references from `mhx` children do not resolve to extant bead IDs. That should be fixed because missing deps are poisonous for coding-agent planning: agents either ignore blockers or chase ghosts.

I would add a one-off `beads-graph-integrity-after-tech-tree` task, or make it part of `ejm3` closeout:

No missing dependency refs.

No tech-tree bead without either real AC or an explicit `horizon:vision` non-executable marker.

No in-progress integration epic whose residuals only live in notes.

No stale path refs to `/tmp` or scratch-only files unless the content is copied into a stable artifact.

This is unglamorous, but it is exactly what prevents a tech tree from becoming a mood board.

## 8. The scratch/corpus durability story needs cleanup

The chatlog says the corpus was treated as ephemeral and Beads should be self-contained; later scratch text says the preserved bundles under `.agent/scratch/corpus-gpt-pro-2026-07-06/bundles/` are the only durable copy.   Those two statements are in tension.

My recommendation: file a small “R&D corpus provenance escrow” bead. It does not need to preserve everything forever, but it should make the state unambiguous.

Acceptance criteria:

A manifest lists each corpus input, content hash, source filename, and whether it is copied into repo/scratch/artifact storage.

Every Bead created from the corpus has either self-contained design/AC or a manifest ref that can be resolved later.

`ejm3` notes stop saying “only durable copy” unless that copy is actually in a durable project artifact.

Future coding agents should not need the original 2.8MB corpus bundle to execute a bead; they should only need it to audit provenance.

## 9. The MCP collapse should include usage telemetry before deletion

The previous pass correctly aligned on `t46` thin-contract first and MCP collapse to verbs/resources/prompts. The corpus notes that the current MCP surface already has resources, prompts, and many tools, and the real gaps are static/unsubscribed resources and hardcoded prompts rather than user-derived prompts. 

I would add one acceptance criterion to `t46.8`: shadow-mode telemetry before tool deletion.

For each old tool, record:

called count by client/model/harness,

replacement verb/resource/prompt,

golden parity status,

last-seen timestamp,

whether removal breaks known clients.

Then delete by observed compatibility, not only by design purity. This matters because MCP agents may have prompts or learned behavior around old tool names. The target shape is right, but the migration needs anti-breakage instrumentation.

## 10. The SDK bead should be reworded away from “async-only / 130 methods”

The prior reviews already corrected two stale claims: MCP tool count in the inspected snapshot was 96, not ~130, and the SDK/facade issue is not simply “async-only.” The real SDK gap is a small, stable, versioned public boundary with stable models and query objects. 

So `4822` should say:

Problem: downstream consumers reach into internals or raw SQLite because the public boundary is broad/unstable.

Not: “there is no sync API” or “130-method async facade.”

Acceptance criteria should include public `__all__`, stable DTO namespace, capability/schema version checks, and examples that do not import internal modules.

## 11. The launch story still needs a claims ledger

The corpus repeatedly converges on “one robust, surprising, clickable number” and “finding as first-class object.”  What I think is still missing is a stricter **claims ledger for Polylogue’s own public claims**.

Not just findings about agent behavior, but every README/landing-page/launch-post claim should be one of:

`proven` — backed by a finding/proof artifact,

`capability` — code exists, no measured result claim,

`aspirational` — roadmap only,

`retired` — no longer true.

This is very Polylogue-native. It turns radical honesty into a product surface and prevents “flight recorder” positioning from becoming marketing fog.

Suggested bead under `3tl`/`212`:

`public-claims-ledger`: maintain `docs/claims.yml` or a user-tier finding-backed ledger; README claims must link to one entry; CI fails if a quantitative claim lacks a finding ref.

## 12. Add a “thinking-vs-doing drift” measure if the fields exist

The brainstorm had “thinking-vs-doing drift” as an early signal that a model got worse for your work: compare reasoning/thinking time or tokens against tool-active time, trended by model/month/task-shape.  I did not see it clearly promoted in the final integration map; the closest filed items are activity spans and efficiency measure packs. 

I would add it as a suppressed/experimental measure, not a public quality score.

Possible definitions:

`thinking_token_share = reasoning_tokens / total_output_tokens`, where available.

`thinking_wall_share = model_thinking_duration_ms / session_wall_ms`, where available.

`tool_active_share = tool_duration_ms / session_wall_ms`.

Trend by model family, repo, workflow shape, and month.

Coverage gate: emit only when provider fields support it; otherwise `insufficient_evidence`.

This would be interesting to you because it catches “the model feels smarter but does less” or “upgrade inflated hidden reasoning cost” from your own corpus, before it becomes a vague preference.

## 13. The “find | compact” lane is important enough to protect from context-compiler confusion

The corpus digest makes the distinction well: `compile_context` answers “what should I hand to an agent to continue/review/debug,” while `find <q> | compact` answers “what is the best evidence digest for an external model to analyze?” 

I would ensure the `fnm.14` bead keeps separate DTOs:

`ContextImage` for continuation/handoff.

`CorpusCompactionPack` for external analysis/R&D.

Shared helpers are fine: token estimation, refs, omission accounting, segment rendering. But the top-level object should stay separate because the scoring is different. A continuation context wants current decisions/open loops; a corpus compaction pack wants representative evidence, contradictions, outcome/error-fix paths, and drop manifest.

This distinction is central to the R&D flywheel: compact pack → external GPT-Pro/DeepResearch → browser auto-capture → imported annotations → next pack. 

## 14. My suggested immediate backlog deltas

I would encode these as the next cleanup batch:

`ejm3.closeout-residuals`
Make integration closure truthful: split D01/D02/D07 into child beads, fix missing deps, close or explicitly keep `ejm3` open.

`3tl.competitive-landscape-bakeoff`
Real D01 rerun. Include LangSmith/Langfuse/Helicone/Braintrust/OpenTelemetry GenAI/Rewind-QS/local logs. Output: positioning matrix plus one public claim.

`mhx.local-retrieval-shootout`
Real D02 rerun. Benchmark Qwen3 Embedding/Reranker, BGE-M3, Nomic, current Voyage baseline, maybe one local reranker, on Polylogue’s own positives.

`fs1.eval-adapter-matrix`
Real D07 rerun. Decide internal trajectory/spec-card schema first, then adapters to Atropos / Terminal-Bench / SWE-bench / Inspect-style tools.

`cpf.6.correct-diagnosis`
Retitle and rewrite AC around clock seam and targeted sort-key audit.

`l4kf.2.correct-sync-hazard`
Replace stale raw-id/source-path claim with content-hash raw identity plus acquisition/session identity merge hazards.

`37t.15.priority-bump`
Make assertion write chokepoint a prerequisite for context scheduler, recall, distillery, standing query findings, and coordination messages.

`at44.scoped-settings-registry`
Keep `user_settings`, but require typed registry, scope/actor/layer semantics, resolver explain output, and first consumers.

`4822.public-sdk-boundary`
Reword from async-only/huge API to stable public API/DTO boundary and schema capability checks.

`3tl.public-claims-ledger`
Every public claim gets status and evidence/finding ref.

`9l5.thinking-vs-doing-drift`
Experimental measure with coverage gates; no composite productivity score.

`beads.graph-integrity-tech-tree`
Fix missing deps, no-AC vision markers, scratch provenance, and stale path refs.

## My sequencing take

Do not start with the shiny frontier. Start with the credibility path that makes every frontier safe:

1. `37t.15`, `kwsb.1`, `cpf.6`, `8jg9.4`, `9e5.28-.30`: safety and honesty.
2. `t46` thin contract and API boundary: avoid building objects twice.
3. `rxdo` query/finding identity and `fnm.14 find|compact`: make the flywheel real.
4. Rerun D01/D02/D07 as explicit research beads.
5. `b5l` before queued index bumps, then batch durable migrations through `60i5`.
6. Only then push episode, graph, SDK, and public interop outward.

The previous integration got the general tech tree right. The most valuable next move is to make the residuals executable, correct stale diagnoses before agents implement them literally, and turn product claims into evidence-linked objects rather than prose.

[1]: https://www.langchain.com/langsmith/observability?utm_source=chatgpt.com "LangSmith: Agent & LLM Observability Platform"
[2]: https://arxiv.org/abs/2506.05176?utm_source=chatgpt.com "[2506.05176] Qwen3 Embedding: Advancing Text ..."
[3]: https://github.com/nousresearch/atropos?utm_source=chatgpt.com "Atropos is a Language Model Reinforcement Learning ..."

