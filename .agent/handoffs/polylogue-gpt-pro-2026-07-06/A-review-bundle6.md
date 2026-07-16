## [user 307]
You are a senior engineer + technical strategist reviewing R&D for Polylogue, a local, single-writer, cross-provider archive for AI coding/chat sessions (split-tier SQLite: durable source/user, rebuildable index/embeddings, disposable ops; query-first CLI + MCP + daemon). The attached file has two parts: Part 1 is a master synthesis (shared context, identical across 6 sibling files — do not base deliverables on it, use it only to understand the whole); Part 2 is this file's unique slice of ~15 subagent design specs. Five sibling files hold the rest; your outputs must stay disjoint from theirs, so anchor everything you produce to Part 2.

Do three things.

1. Vet, hard. Adversarially verify the Part-2 specs — especially the conclusions: bug diagnoses (are the file:line claims and failure modes
real and correctly characterized?), the tier/regime/schsments, and any strategic or prioritization verdicts.Separate what you confirm, refute, and extend/correct, with reasoning. Then research around it: use your own knowledge and web search for
prior art, competing approaches, relevant standards/libure patterns, and anything the specs assume but don'tjustify. Bring in what's missing; challenge what's overconfident. Note where you'd need to read live source to be sure.
2. Write a self-contained analysis. Produce one coherenontains all the value in this file's material plus yourvetting and research — written so a reader who never se Restate and own the substance; never say "the specclaims" or reference the attachment. Organize it well (themes, confirmed problems, design decisions and their tradeoffs, your corrections, external context, open questions, a prioritized take). Be opinionated and concrete; preserve the sharp specifics (numbers, file paths,       mechanisms, risks) rather than abstracting them away.

3. Propose the bead changes that would enact everything your analysis covers — nothing substantive should be analysis-only. Output them as a single fenced ```json block with this shape:

{
  "file": <the N from the file's "file N of 6" header>,
  "proposed_beads": [
    {
      "proposed_id": "p<N>-01",                 // namespaced by file N so 6 sets never collide
      "change_kind": "new",                     // new
      "target_bead": null,                      // for update/close/supersede: the extant bead id, e.g. "polylogue-f2qv"
      "title": "...",                           // ≤72
      "issue_type": "feature|bug|task|epic|chore",
      "priority": 0-3,
      "labels": ["area:storage", ...],
      "description": "problem + evidence (what's true t
      "design": "approach, tier/regime, key mechanisms, alternatives rejected",
      "acceptance_criteria": ["observable, checkable ...", ...],
      "dependencies": ["polylogue-<extant>", "p<N>-0X"],   // extant bead ids and/or sibling proposed ids
      "parent": "polylogue-<epic>|p<N>-0X|null",
      "risks": ["top 1-3"],
      "confidence": "high|med|low",             // after your vetting
      "provenance": "one line: which finding/recommendaacts"
    }
  ]
}

Rules for the bead set: use real extant Polylogue bead es them (for target_bead, dependencies, parent); everyproposed bead must trace to something in your analysis (provenance); prefer update/supersede on an existing bead over a duplicate new one; size them as landable units with genuine acceptance criand keep them scoped to this file's payload so the sixJSON blocks concatenate into one non-overlapping backlog.

## [assistant 331]
bash -lc cd /mnt/data/poly_tree2/polylogue && python3 - <<'PY'
from pathlib import Path
checks = {
 'index': 'polylogue/storage/sqlite/archive_tiers/index.py',
 'embeddings': 'polylogue/storage/sqlite/archive_tiers/embeddings.py',
 'user': 'polylogue/storage/sqlite/archive_tiers/user.py',
 'enums': 'polylogue/core/enums.py',
 'refs': 'polylogue/core/refs.py',
 'session_summaries': 'polylogue/archive/session_summaries.py',
 'pricing': 'polylogue/cost/pricing.py',
 'subscription_pricing': 'polylogue/cost/subscription_pricing.py',
 'cost_compute': 'polylogue/cost/cost_compute.py',
 'cost_enrichment': 'polylogue/pipeline/cost_enrichment.py',
 'hybrid': 'polylogue/storage/search_providers/hybrid.py',
 'hybrid_sessions': 'polylogue/storage/search_providers/hybrid_sessions.py',
 'search_hits': 'polylogue/archive/query/search_hits.py',
 'search_models': 'polylogue/storage/search/models.py',
 'archive_execution': 'polylogue/archive/query/archive_execution.py',
 'miss': 'polylogue/archive/query/miss_diagnostics.py',
 'http': 'polylogue/daemon/http.py',
 'browser_server': 'polylogue/browser_capture/server.py',
 'browser_receiver': 'polylogue/browser_capture/receiver.py',
 'protocols': 'polylogue/protocols.py',
 'mcp_insight': 'polylogue/mcp/server_insight_tools.py',
 'mcp_prompts': 'polylogue/mcp/server_prompts.py',
 'mcp_resources': 'polylogue/mcp/server_resources.py',
 'config': 'polylogue/config.py',
}
for name,p in checks.items():
    path=Path(p)
    print(name, path.exists(), path.stat().st_size if path.exists() else '')
PY

rg -n "CREATE TABLE IF NOT EXISTS (session_links|threads|repos|session_repos|session_commits|session_profiles|user_settings|assertions)|CREATE VIEW IF NOT EXISTS actions|message_embeddings|VectorProvider|def reciprocal_rank_fusion|def search_scored|_resolve_ranked_session_hits|_hybrid_score_components|SessionSearchMatchPayload|def _dispatch_get|def _check_cross_origin|def _check_auth|query_completions|resources|@mcp.prompt|@mcp.tool" polylogue | sed -n '1,300p'

## [assistant 359]
bash -lc cd /mnt/data/poly_pkg2/polylogue && python3 - <<'PY'
import json, textwrap
issues={}
for l in open('polylogue-beads-export.jsonl'):
    d=json.loads(l)
    if d.get('_type')=='issue': issues[d['id']]=d
for id in ['polylogue-7xv','polylogue-f2qv','polylogue-w8db','polylogue-mhx','polylogue-mhx.4','polylogue-kwsb','polylogue-37t','polylogue-37t.11','polylogue-3tl','polylogue-3tl.4','polylogue-jnj','polylogue-cpf','polylogue-4ts','polylogue-83u','polylogue-fnm','polylogue-t46','polylogue-9l5','polylogue-gjg','polylogue-4p1','polylogue-bby','polylogue-20d','polylogue-s7ae']:
    b=issues.get(id)
    print('\n',id, 'exists', bool(b))
    if b:
        print('P',b.get('priority'),'type',b.get('issue_type'),'status',b.get('status'),'title',b.get('title'))
        print('desc', textwrap.shorten((b.get('description') or '').replace('\n',' | '), 250))
        print('design', textwrap.shorten((b.get('design') or '').replace('\n',' | '), 300))
        print('acc', textwrap.shorten((b.get('acceptance_criteria') or '').replace('\n',' | '), 250))
PY

## [assistant 389]
bash -lc cd /mnt/data/poly_pkg2/polylogue 2>/dev/null || cd /mnt/data/poly_pkg/polylogue && python3 - <<'PY'
import json, re, textwrap
issues=[]
for l in open('polylogue-beads-export.jsonl'):
 d=json.loads(l)
 if d.get('_type')=='issue': issues.append(d)
terms=['episode','compaction','cost','provider usage','config','runtime','search relevance','ranking','security','recall','MCP','docs onboarding','temporal','time','SDK','python api','launch','entity mention','verification','world effect','setting','f2qv','gjg','w8db','y4c','9l5','mcp','3tl','cpf','mhx','37t','bby','a5e']
for term in terms:
    matches=[i for i in issues if term.lower() in (' '.join(str(i.get(k,'')) for k in ['id','title','description','design','labels'])).lower()]
    if matches:
        print('\nTERM',term, len(matches))
        for i in matches[:15]: print(i['id'],i.get('priority'),i.get('status'),i.get('issue_type'),i.get('title')[:100])
PY

## [assistant 392]
bash -lc cd /mnt/data/poly_pkg2/polylogue 2>/dev/null || cd /mnt/data/poly_pkg/polylogue && python3 - <<'PY'
import json, textwrap
ids=['polylogue-7xv','polylogue-gjg','polylogue-f2qv','polylogue-w8db','polylogue-y4c','polylogue-mhx.4','polylogue-37t.11','polylogue-3tl','polylogue-3tl.4','polylogue-3tl.7','polylogue-3tl.13','polylogue-cpf','polylogue-kwsb','polylogue-jnj.5','polylogue-9l5.6','polylogue-9l5.7','polylogue-mhx.3','polylogue-fnm.12','polylogue-fnm.13','polylogue-pj8','polylogue-t46','polylogue-20d.1','polylogue-20d.14','polylogue-s7ae','polylogue-bby.11','polylogue-4ts.5']
issues={}
for l in open('polylogue-beads-export.jsonl'):
 d=json.loads(l)
 if d.get('_type')=='issue': issues[d['id']]=d
for id in ids:
 b=issues.get(id)
 if b:
  print(id, '|', b.get('priority'), b.get('status'), b.get('issue_type'), '|', b.get('title'))
 else:
  print(id,'NOT FOUND')
PY

## [assistant 425]
bash -lc python3 - <<'PY'
import json
beads = [
  {
    "proposed_id":"p6-01",
    "change_kind":"new",
    "target_bead":None,
    "title":"Episode unit: logical tasks across sessions",
    "issue_type":"epic",
    "priority":2,
    "labels":["area:analytics","area:lineage","area:embeddings","from:rnd-file-6"],
    "description":"File 6's episode design identifies a missing unit above threads/session_links: a logical task hypothesis that groups distinct sessions across providers, repos, time, and artifacts. I verified that repos/session_repos/session_commits and session_profiles exist, and that embeddings are message-level only; the episode unit is therefore plausible but needs explicit identity, false-merge, and rebuild invariants.",
    "design":"Create an episode epic over derived index + embeddings and user-tier stitch judgments. Index tier owns episodes/episode_members/episode_edges/episode_produced_refs and materializes from sessions, repo/commit attribution, time, session vectors, and artifact/error corroborators. User tier owns edge-keyed EPISODE_CONFIRM/SPLIT/REJECT assertions. Correct the proposed id design: ep:<anchor_session_id> alone is not split-safe; use anchor plus member_set_hash prefix or a deterministic split-aware component key. Candidate-only thematic similarity must never merge without a hard corroborator.",
    "acceptance_criteria":["A written implementation plan records the corrected episode identity key and why anchor-only ids are unsafe under SPLIT constraints.","Episode tables are derived-tier only; user-tier stitch judgments are assertion rows and survive ops reset --index.","False-merge fixtures cover high-embedding/same-title/different-repo pairs and assert candidate-only, never linked/corroborated.","Shared commit SHA or shared normalized error fingerprint can promote cross-provider sessions to linked/corroborated with evidence_json.","Read/query surfaces expose episode refs without consumers re-deriving session grouping."],
    "dependencies":["polylogue-7xv","polylogue-4ts.5","polylogue-mhx.3","polylogue-mhx.2"],
    "parent":"polylogue-9l5",
    "risks":["Embedding similarity can false-merge thematic neighbors.","Split constraints can destabilize deterministic ids if anchor-only.","Episode cost rollup can repeat lineage double-count bugs if 4ts is incomplete."],
    "confidence":"med",
    "provenance":"Vets and corrects the Part-2 Episode Unit spec: scope boundary, DDL, embeddings need, false-merge floor, and cost rollup."
  },
  {
    "proposed_id":"p6-02",
    "change_kind":"new",
    "target_bead":None,
    "title":"Session embeddings v2 for episode scoring",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:embeddings","area:analytics","from:rnd-file-6"],
    "description":"The episode scorer needs session-level vectors but the repo snapshot only has message_embeddings and message_embeddings_meta in embeddings.db v1. File 6 proposes session_embeddings with message-centroid default and summary_text fallback.",
    "design":"Add embeddings.db v2 session_embeddings + session_embeddings_meta as a derived tier. Prefer message_centroid over eligible human/assistant message vectors to avoid new Voyage calls; use summary_text only when no message vectors exist. Verify sqlite-vec vec0 metadata/aux syntax in implementation before adopting the proposed '+origin TEXT' shape. Wire as a daemon convergence stage after message embeddings are current, with stale/needs_reindex metadata and no blocking ingest.",
    "acceptance_criteria":["EMBEDDINGS_SCHEMA_VERSION bumps to v2 via canonical DDL, not an in-place migration helper.","Eligible sessions with message vectors receive a centroid vector with zero new embedding-provider calls.","Sessions without usable message vectors are marked pending or embedded by summary_text fallback with content_hash provenance.","Converger catch-up is bounded and false_means_pending-style; ingest remains unblocked.","A test verifies sqlite-vec schema creation on the installed extension, including metadata/partition column syntax."],
    "dependencies":["polylogue-mhx.2","polylogue-mhx.3"],
    "parent":"p6-01",
    "risks":["sqlite-vec virtual table syntax may differ from the draft.","Centroid vectors may underperform summary embeddings for heterogeneous long sessions.","Embedding rebuild cost can be large without blue-green/batched schema discipline."],
    "confidence":"high",
    "provenance":"Derived from Episode §1b and verified against current embeddings.db message-only schema."
  },
  {
    "proposed_id":"p6-03",
    "change_kind":"new",
    "target_bead":None,
    "title":"Episode scorer/materializer and stitch judgments",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:analytics","area:query","area:user-db","from:rnd-file-6"],
    "description":"The episode feature requires a pure four-signal scorer, derived materializer, and durable human stitch decisions. This is separable from session-vector creation and should be implemented after vectors exist.",
    "design":"Implement candidate blocking over repo/time/session-ANN; pure score_edge(a,b,judgments) over repo, time, embedding, and artifact/error corroboration; connected components only over corroborated+ edges; candidate neighbors attach for review but do not merge. Add AssertionKind/ObjectRefKind registrations and user_audit surface for EPISODE_CONFIRM/SPLIT/REJECT and episode refs. Persist episode_edges for explainability and audit.",
    "acceptance_criteria":["Scorer is pure and unit-tested independently of SQLite.","episode_edges records per-signal scores, floor_passed, corroborators_json, and negative_json.","EPISODE_REJECT, CONFIRM, and SPLIT assertions survive index reset and are replayed during materialization.","Component materialization is idempotent: unchanged member_set_hash skips writes.","render openapi, cli-output-schemas, topology projection, and user_audit checks are updated for new kinds."],
    "dependencies":["p6-02","polylogue-7xv","polylogue-4ts.5"],
    "parent":"p6-01",
    "risks":["Human judgments keyed only by pairs may not express larger-cluster semantics cleanly.","Candidate blocking can drop true positives if ANN/repo/time blocks miss shared artifacts.","Component recomputation could be expensive without incremental debt tracking."],
    "confidence":"med",
    "provenance":"Turns the Episode algorithm/test plan into an execution bead and adds the identity correction from vetting."
  },
  {
    "proposed_id":"p6-04",
    "change_kind":"new",
    "target_bead":None,
    "title":"Entity-mention query unit with trust axis",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:query-units","area:index","area:recursive-safety","from:rnd-file-6"],
    "description":"File 6's missing-units slice correctly identifies entity-mention as a TABLE, not a VIEW: prose/regex/entity extraction is too expensive and trust-sensitive to rerun per query. It also needs a structural vs candidate provenance axis to avoid laundering mined prose into fact.",
    "design":"Add index.db entity_mentions table and materializer. Emit structural mentions from typed tool inputs/results, web constructs, and session_commits; emit candidate mentions only from authored material_origin text with low confidence and recursive-safety gating. Register entity-mention as a query unit with kind/value/provenance fields. Batch this derived schema bump with p6-05.",
    "acceptance_criteria":["entity_mentions table and indexes exist in canonical index DDL with INDEX_SCHEMA_VERSION bump.","Structural rows prove byte/field presence in cited typed source; property test fails if fabricated.","Candidate rows are never emitted from generated_context_pack, runtime_context, runtime_protocol, or tool_result prose.","Query DSL can filter and group entity-mentions by entity_kind, entity_value/entity_key, provenance, and session fields.","Generated schemas, topology projection, and query-unit registry are updated."],
    "dependencies":["polylogue-37t.11","polylogue-9l5.7"],
    "parent":"polylogue-9l5",
    "risks":["Regex false positives, especially commit-like hashes and PR numbers.","Table growth can be high at block granularity.","Name collisions with existing raw_artifacts/files vocabulary if unit naming is loose."],
    "confidence":"high",
    "provenance":"Implements the Part-2 three-missing-units verdict for entity-mention as a trusted/candidate table."
  },
  {
    "proposed_id":"p6-05",
    "change_kind":"new",
    "target_bead":None,
    "title":"World-effect and verification-run query views",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:query-units","area:index","area:honesty","from:rnd-file-6"],
    "description":"File 6's missing-units slice correctly classifies world-effect and verification-run as derived VIEW/query projections over existing actions, observed_events, and run data. They should ride the same derived schema bump as entity_mentions rather than require separate migrations.",
    "design":"Add world_effects VIEW joining session_observed_events.evidence_refs_json to producing actions, preserving null-cause degradation; add verification_runs VIEW over actions and verification observed-events, with pass/fail resolved only from exit_code then is_error then NULL. Register both query units. Keep temporal-fallback causality off by default or confidence-tagged to avoid fabricated action→effect links.",
    "acceptance_criteria":["world_effects and verification_runs are canonical index DDL views batched with p6-04.","Every non-null world_effect cause_action_block_id resolves to a real action/tool_use in the same session.","verification_runs.passed is TRUE iff exit_code=0 or is_error=0 when exit_code is null; both unknown yields NULL.","Zero-row pass-rate/verification insights fail coverage gates rather than rendering 0%.","No persistent view references an attached user.db; cross-tier joins, if needed, are runtime/TEMP only."],
    "dependencies":["p6-04","polylogue-9l5.7","polylogue-9l5.6"],
    "parent":"polylogue-9l5",
    "risks":["json_each/evidence_refs_json joins may be slower than expected.","Verification command LIKE patterns can overmatch without token anchoring.","Downstream measures may double-count action-grain verification views and event-grain observed_events."],
    "confidence":"high",
    "provenance":"Implements and corrects the Part-2 world-effect and verification-run missing-unit specs."
  },
  {
    "proposed_id":"p6-06",
    "change_kind":"new",
    "target_bead":None,
    "title":"Compaction snapshots: source-tier capture and fallback",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:source-db","area:compaction","area:blob-store","from:rnd-file-6"],
    "description":"The compaction lifecycle design is right that pre-compact context is irreplaceable raw evidence. Current code has compaction markers/parsing but no durable compaction_snapshots table. The spec's own risk is load-bearing: PreCompact may not carry the assembled context, so exactness must be verified and encoded as capture_tier.",
    "design":"Add source.db v3 compaction_snapshots and snapshot blob_refs via numbered migration with backup manifest. Capture hook payload context when present; otherwise serialize a reconstructed JSONL-boundary prefix using lineage/effective-context logic and mark capture_tier=jsonl-boundary. Snapshot ids are natural keys over origin/native session/boundary ordinal; context_blob_hash is the content hash. Add retention/purge policy hooks because snapshots are sensitive durable blobs.",
    "acceptance_criteria":["A live or fixture PreCompact payload inspection test records whether assembled context bytes are actually available.","source migration 003 adds compaction_snapshots and snapshot blob_refs with backup manifest and one PRAGMA step.","Hook and jsonl-boundary paths each produce idempotent rows on re-ingest of the same boundary.","Blob GC preserves referenced snapshot blobs and can reclaim after explicit snapshot deletion.","All surfaces disclose capture_tier and never pool hook-exact and reconstructed-prefix results silently."],
    "dependencies":["polylogue-gjg","polylogue-d1y","polylogue-4ts.5"],
    "parent":"polylogue-gjg",
    "risks":["PreCompact payload may lack assembled context.","Snapshot blobs may capture secrets and very large context windows.","Durable source migration must be batched with other source v3 changes."],
    "confidence":"high",
    "provenance":"Turns Part-2 gjg snapshot tier decision and Risk 1 into a concrete child of polylogue-gjg."
  },
  {
    "proposed_id":"p6-07",
    "change_kind":"new",
    "target_bead":None,
    "title":"Compaction loss forensics and re-grounding source",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:compaction","area:context","area:assertions","from:rnd-file-6"],
    "description":"The compaction spec's strongest product move is not just measuring loss; it writes high-score lost items as durable refs-only assertions and lets the context scheduler re-ground future compacted sessions. This closes the compaction loop instead of producing a report no agent uses.",
    "design":"Add index-tier compaction_loss diff over snapshot vs effective post-boundary context; extract file-path/tool-outcome/marked-decision/cited-ref item classes; emit structural loss counts with a construct caveat that structural absence is not semantic forgetting. Write high-score lost items to user assertions as refs-only SYSTEM context sources, never verbatim prose. Register a 37t.11 ContextSource for pre-compact-resume/SessionStart(source=compact) and an experiment arm to measure uplift.",
    "acceptance_criteria":["compaction_loss retained/lost/transformed partition is exhaustive and disjoint by property test.","No-snapshot path returns explicit degraded/no_snapshot status, never zero loss.","High-score durable loss records survive ops reset --index and are proposed by the re-grounding ContextSource in a simulated second compaction.","MCP/CLI view returns drillable refs and construct-validity caveat for every headline count.","Assembled re-grounding output is refs-only or quoted with trust-class fencing and budget-capped."],
    "dependencies":["p6-06","polylogue-37t.11","polylogue-9l5.7","polylogue-4ts.5"],
    "parent":"polylogue-gjg",
    "risks":["Structural absence can overstate forgetting.","Re-grounding can inject stale or low-trust content if not gated by 37t.11.","Loss ranking weights could look scientific before labeled calibration exists."],
    "confidence":"med",
    "provenance":"Implements Part-2 gjg loss-forensics, durable recursion, and re-grounding ContextSource design."
  },
  {
    "proposed_id":"p6-08",
    "change_kind":"update",
    "target_bead":"polylogue-f2qv",
    "title":"Cost correctness: exact path, tier, interval union",
    "issue_type":"bug",
    "priority":1,
    "labels":["area:cost","area:honesty","area:lineage","from:rnd-file-6"],
    "description":"The cost-correctness slice is substantially confirmed: physical-session summary rollups sum replayed/forked sessions; session_reported_costs is written but the exact provider-cost reader path is effectively dead; subscription dollars hardcode a Pro-style denominator; and exact/priced provider evidence can be weakened by enrichment. These are credibility bugs, not analytics polish.",
    "design":"Update f2qv scope/AC to require: logical-session rollups; a live reader for session_reported_costs; provider_reported/priced/heuristic precedence lattice that never demotes stronger evidence; closed-vocabulary subscription tier as a setting assertion with explicit unconfigured state; per-model credit rates loaded from one source of truth; Codex disjoint-lane property tests; and wall_duration_ms interval union for parallel subagents. Do not hardcode 2026 model rates without checking the live provider catalog in the implementation PR.",
    "acceptance_criteria":["Day/week/archive summaries sum cost/duration on logical-session grain and expose both session_count and logical_session_count.","A fixture with session_reported_costs rows returns provider_reported cost in every cost surface that claims exactness.","Cost enrichment cannot replace provider_reported with catalog_priced or heuristic.","Subscription cost views fail loudly with tier_unconfigured instead of silently assuming Pro; configured tier is surfaced in payloads.","Parallel child sessions with overlapping wall intervals roll up by interval union, not sum of durations."],
    "dependencies":["polylogue-4ts.5","polylogue-f2qv.1","polylogue-f2qv.2","polylogue-f2qv.3","polylogue-f2qv.4","polylogue-f2qv.5","polylogue-w8db"],
    "parent":"polylogue-f2qv",
    "risks":["Provider model catalogs change; stale rates silently reintroduce false zeros.","Logical representative selection can hide tail costs if 4ts lineage is incomplete.","Interval union needs reliable start/end timestamps by origin."],
    "confidence":"high",
    "provenance":"Vets Part-2 f2qv cost map against code and converts corrections into epic-level AC."
  },
  {
    "proposed_id":"p6-09",
    "change_kind":"update",
    "target_bead":"polylogue-w8db",
    "title":"Runtime preferences as safe setting assertions",
    "issue_type":"epic",
    "priority":2,
    "labels":["area:config","area:user-db","area:recursive-safety","from:rnd-file-6"],
    "description":"The config/runtime-preferences slice is confirmed: config.py already has a deployment resolver, while user_settings exists in DDL/migration but has no read/write helpers. Assertions already have the columns needed for scoped settings. The correct move is a setting-assertion resolver, not reviving the dead table.",
    "design":"Update w8db to implement runtime settings as AssertionKind.SETTING rows with scope precedence global→repo→origin→surface and actor precedence operator→agent→harness. Keep deployment secrets out of user.db. Copy-forward any future user_settings rows into assertions via durable migration but do not destructively drop the table in the first PR. Add batched resolver, config explain payload, macro/preset validation, and learned-default candidates gated through judgment flow.",
    "acceptance_criteria":["A registry partitions deployment-only keys from runtime setting keys; secret keys cannot be written as setting assertions.","Resolver returns effective value plus shadowed candidates, winning scope/actor/layer, and provenance.","Existing user_settings rows, if any, are copied forward into assertions in a manifest-gated migration; old table is not required by new readers.","fnm.12 macros validate at definition time, carry expansion provenance, and are cycle/depth guarded.","Learned defaults are candidate assertions only, self-exclude config/judge telemetry, and never auto-apply."],
    "dependencies":["polylogue-fnm.12","polylogue-37t.12","polylogue-y4c"],
    "parent":"polylogue-w8db",
    "risks":["Runtime/deployment key-space bleed can leak secrets.","Settings stored as assertions can blur observation vs decision without a distinct kind and explain surface.","Per-key DB lookups can regress CLI latency unless resolved in one batch."],
    "confidence":"high",
    "provenance":"Part-2 config engine vet: user_settings dead, assertions viable, but destructive drop should be delayed."
  },
  {
    "proposed_id":"p6-10",
    "change_kind":"new",
    "target_bead":None,
    "title":"Assertion safety chokepoint for agent authors",
    "issue_type":"bug",
    "priority":1,
    "labels":["area:assertions","area:recursive-safety","area:security","from:rnd-file-6"],
    "description":"Multiple Part-2 designs rely on external/agent annotations, learned defaults, recall evidence, and settings. The current assertion write path can allow non-user authors to land as active if call sites pass or default active status. This is a recursive-safety hole: agent-authored claims can become injectable authority before judgment.",
    "design":"Add one coerce_agent_authored chokepoint inside upsert_assertion: author_kind != user forces status=CANDIDATE and context_policy.inject=false for new/nonterminal rows, and never resurrects terminal judged rows. Every writer — blackboard, recall, recipe, transform/pathology, setting suggestion, prompt distillery, goal/decision — inherits the gate. Expose explicit operator promotion through judge_assertion_candidate/bulk judgment only.",
    "acceptance_criteria":["Unit tests prove non-user author_kind cannot create or update an ACTIVE injectable assertion through upsert_assertion.","Terminal accepted/rejected/superseded/deleted rows are not resurrected by later agent writes.","blackboard_post and annotation import both route through the chokepoint.","Context compiler/scheduler refuses inject:true unless author_kind=user or an explicit operator judgment promoted the row.","Generated schemas and user_audit surfaces document the candidate default."],
    "dependencies":["polylogue-37t.11","polylogue-37t.12","polylogue-w8db"],
    "parent":"polylogue-37t.11",
    "risks":["Over-tight gating can slow legitimate agent-maintained state.","Existing tests may assume active defaults for helper-created assertions.","Bulk migration of existing agent-authored active rows needs careful audit."],
    "confidence":"high",
    "provenance":"Extends Part-2 config/recall/security findings and the blackboard_post recursive-safety bug into one write-path invariant."
  },
  {
    "proposed_id":"p6-11",
    "change_kind":"new",
    "target_bead":None,
    "title":"Search relevance labels, eval, and explainability",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:search","area:retrieval","area:evaluation","from:rnd-file-6"],
    "description":"The search-relevance slice is directionally correct: RRF is a sound unsupervised fusion baseline, but Polylogue needs evidence-backed relevance judgments, logical-session de-duplication, and explainable score components to keep search honest at scale. Existing search paths and payload score components are close but not a quality loop.",
    "design":"Add AssertionKind.RELEVANCE candidate/accepted labels over query/ref pairs; nDCG/MRR/recall@k evaluation lanes for lexical/vector/hybrid; logical-session collapse in one shared helper so siblings/forks do not duplicate; field boosts and MMR/diversity as optional rerank stages with explain payloads. Labels from agents are candidates until judged. Do not make typo-hints/bare-query recovery central; the operator rejected bare-query as a product path.",
    "acceptance_criteria":["Relevance labels can be imported/reviewed as assertions with query/ref/split metadata and candidate status for agent authors.","A seeded eval fixture computes nDCG@k for lexical, vector, and hybrid lanes and fails on regression.","Hybrid search returns score_components sufficient to explain lexical/vector/RRF contributions.","One shared session-resolution helper collapses logical-session siblings across CLI/API/daemon paths.","Explicit-query miss diagnostics remain, but no bead depends on bare-query typo fallback."],
    "dependencies":["polylogue-mhx.3","polylogue-mhx.4","polylogue-fnm.13","polylogue-t46"],
    "parent":"polylogue-mhx.3",
    "risks":["Small or biased label sets can Goodhart search quality.","MMR/diversity can hide best hits unless opt-in/explainable.","Multiple search paths can drift unless t46 unifies them."],
    "confidence":"med",
    "provenance":"Converts Part-2 search relevance/ranking quality spec plus external RRF/MMR/nDCG research into a single eval loop."
  },
  {
    "proposed_id":"p6-12",
    "change_kind":"new",
    "target_bead":None,
    "title":"Loopback daemon and capture receiver hardening",
    "issue_type":"bug",
    "priority":1,
    "labels":["area:security","area:daemon","area:browser-capture","from:rnd-file-6"],
    "description":"The daemon/capture security spec is confirmed against the snapshot: GET routes lack Host validation, Origin checks are POST-only and allow missing Origin, access_token query auth is broad, and browser-capture token auth is optional and direct string compare. Loopback is not a security boundary against browser-origin/confused-deputy attacks.",
    "design":"Implement central Host+Origin allowlist middleware for GET and POST before dispatch; scope query-string access_token to the route that truly needs it; auto-mint a 0600 browser-capture receiver token, compare with hmac.compare_digest, and plumb token to the extension; add spool quota/governor. Update threat docs to state same-user residual plainly: this defends browser/network trust classes, not hostile same-user processes reading the token/archive.",
    "acceptance_criteria":["Rebinding fixture with Host: attacker.com is rejected for GET and POST while localhost/127.0.0.1/[::1] web shell still works.","Browser capture without valid token is rejected in dogfood/e2e tests; valid token succeeds through the real extension handshake or explicit override.","Only the intended SSE/browser route accepts access_token in the query string; other routes require header/UDS-equivalent auth.","Spool quota rejects unbounded local POST floods with an auditable error.","Threat model docs list each mitigation and same-user residual without overclaiming."],
    "dependencies":["polylogue-kwsb","polylogue-jnj.5"],
    "parent":"polylogue-kwsb",
    "risks":["Too-strict Host allowlist can break reverse proxy or web shell flows.","Token distribution to browser extension is a fragile seam.","Same-user residual remains and must not be framed as solved."],
    "confidence":"high",
    "provenance":"Vets Part-2 daemon/capture hardening and threat-model red-team against code and DNS rebinding prior art."
  },
  {
    "proposed_id":"p6-13",
    "change_kind":"new",
    "target_bead":None,
    "title":"Cross-project recall(task_hint) MCP/context source",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:mcp","area:memory","area:retrieval","from:rnd-file-6"],
    "description":"The recall(task_hint) spec is well-grounded: VectorProvider.query(text, limit) already provides free-text embedding search, while current similarity tools are session-seeded. The missing capability is agent-facing text-hint recall that joins relevant prior sessions with judged assertions/lessons/caveats and degrades honestly when embeddings are unavailable.",
    "design":"Implement recall_report behind the Polylogue facade and expose it as MCP read tool or, if MCP-collapse lands first, as a continuity/context verb. It embeds task_hint, retrieves top messages/sessions with repo include/exclude filters, joins assertion evidence by trust class, budgets output, and returns explicit retrieval_lane=vector|fts plus embeddings_status. Register as a 37t.11 ContextSource so later scheduler paths can use the same logic.",
    "acceptance_criteria":["recall(task_hint) returns no padded hits below min_similarity and marks vector disabled/unavailable with FTS fallback reason.","Returned evidence distinguishes OPERATOR judged assertions from QUOTED unjudged agent text and includes resolve_ref for every item.","Tool registration updates EXPECTED_TOOL_NAMES/TOOL_CONTRACT or the MCP-collapse equivalent compatibility map.","Token budget is enforced with a drop manifest or omitted count.","A fixture shows a cross-repo hint retrieving a prior session and attached judged lesson without querying the current repo only."],
    "dependencies":["polylogue-mhx.4","polylogue-mhx.3","polylogue-37t.11","polylogue-pj8"],
    "parent":"polylogue-mhx.4",
    "risks":["Voyage/API availability can make recall look flaky without honest FTS degradation.","Unjudged agent-authored lessons can launder into instructions if trust classes are weak.","A one-off MCP tool can become obsolete if the MCP surface collapses first."],
    "confidence":"high",
    "provenance":"Implements Part-2 cross-project recall MCP design while aligning it with context scheduler and MCP-collapse constraints."
  },
  {
    "proposed_id":"p6-14",
    "change_kind":"new",
    "target_bead":None,
    "title":"MCP resource-first continuity surface",
    "issue_type":"epic",
    "priority":2,
    "labels":["area:mcp","area:api-contract","area:agent-ux","from:rnd-file-6"],
    "description":"The MCP surface slice argues that Polylogue currently overuses tools and underuses MCP resources/prompts: static resources are not subscribed, prompts are hardcoded, and many read tools duplicate the DSL. The MCP spec supports resource subscribe/listChanged and prompt/tool listChanged, so the product should make live archive state referenceable instead of forcing agents through 130 flat tools.",
    "design":"Design a compatibility-first MCP surface: keep existing tools until goldens prove equivalence, add a single query(expression) read-algebra tool, promote session/origin/recent/read-heavy objects to polylogue:// resources with subscribe/list_changed, publish polylogue://tools/map taxonomy, and mint dynamic prompts from saved recall-packs/views. Role-gating remains read/write/admin; destructive writes stay tools/actions. Deletion of aliases happens only after per-tool equivalence tests.",
    "acceptance_criteria":["Server declares resources.subscribe/listChanged capabilities and emits notifications when subscribed session/origin resources change.","polylogue://tools/map exists and groups current tools with preferred replacements.","Saved recall-pack/view can register a dynamic prompt or prompt-equivalent with listChanged notification.","query(expression) covers search/list/facets/basic aggregates in golden tests before any legacy tool deletion.","No read-heavy resource fetch burns a write/admin tool path; role gating remains explicit in tests."],
    "dependencies":["polylogue-t46","polylogue-pj8","polylogue-fnm.12","polylogue-37t.11","p6-13"],
    "parent":"polylogue-pj8",
    "risks":["Silent capability loss during tool collapse.","Unbounded resource subscriptions can become a notification firehose.","Clients may not yet implement resource subscriptions reliably; retain polling fallback."],
    "confidence":"med",
    "provenance":"Converts Part-2 MCP surface product ideas and MCP spec research into an incremental compatibility plan."
  },
  {
    "proposed_id":"p6-15",
    "change_kind":"update",
    "target_bead":"polylogue-cpf",
    "title":"Temporal doctrine and clock seam enforcement",
    "issue_type":"epic",
    "priority":2,
    "labels":["area:temporal","area:doctrine","area:testing","from:rnd-file-6"],
    "description":"The temporal doctrine slice is correct: date/time semantics are an honesty boundary. The repo snapshot confirms parse_date currently calls datetime.now inside the parser; relative-date tests need a clock seam, and event/source/observed/sort times need explicit provenance rather than ad hoc ISO/TEXT handling.",
    "design":"Update cpf/time doctrine to require core/clock.py/frozen_clock injection for all relative date parsing, four time kinds (source event time, observed/acquired time, derived sort key, processing/materialized time), UTC epoch-ms storage, half-open intervals, timezone-unknown defaults, skew bands, and explicit sort_key provenance/confidence. Add lint/tripwire for new durable TEXT timestamp columns and naive datetime.now use outside allowed clock seams.",
    "acceptance_criteria":["parse_date and query relative dates accept/test through an injectable clock; frozen_clock tests cannot observe process start time.","docs/doctrine/time.md defines the four time kinds and maps each major timestamp column to one kind.","New durable DDL with TEXT timestamp columns or unprovenanced sort keys fails a devtools policy check.","since/until filters use half-open intervals and explicitly preserve/label timeless sessions rather than pinning them to 1970.","OpenTelemetry-style source vs observed timestamp terminology is reflected in docs/payloads where relevant."],
    "dependencies":["polylogue-cpf.1","polylogue-cpf.4"],
    "parent":"polylogue-cpf",
    "risks":["Changing sort semantics can break cursor stability and user expectations.","Backfilled sessions need stable historical ordering without hiding synthetic confidence.","Linting datetime.now too broadly can create noise in harmless rendering code."],
    "confidence":"high",
    "provenance":"Vets Part-2 temporal doctrine ideas and confirms the relative-date clock-seam issue in code."
  },
  {
    "proposed_id":"p6-16",
    "change_kind":"new",
    "target_bead":None,
    "title":"Public Python SDK and query objects",
    "issue_type":"epic",
    "priority":2,
    "labels":["area:sdk","area:api-contract","area:analytics","from:rnd-file-6"],
    "description":"The Python API/library-consumer slice identifies a real product gap: the async Polylogue facade is large and unstable as a public surface, while downstream analytical consumers want a small sync-friendly API, stable models, schema capability checks, and columnar outputs. Claims about Lynchpin internals are plausible but need verification in the private repo; the SDK design itself is independently warranted.",
    "design":"Create polylogue.sdk as a narrow versioned public API: sync open(), capabilities(), query objects with count/stream/to_arrow/to_polars/save, get/read over ObjectRefs, stable polylogue.models return types, typed errors, schema pin-and-warn, and optional Arrow/Polars extras. Keep raw SQLite views as a documented escape hatch with schema version binding; avoid persistent cross-attached views and generate TEMP views or materialized public views where needed.",
    "acceptance_criteria":["polylogue.sdk exposes a documented __all__ of roughly 20 methods/classes and does not import substrate internals from consumer examples.","A sync notebook-style example can query sessions/messages and stream rows without asyncio boilerplate.","Query object supports .count(), .stream(), .save(), and .to_arrow() behind optional dependencies with graceful ImportError messaging.","capabilities() reports archive schema versions and SDK compatibility warnings.","docs/sdk-cookbook.md examples execute in CI against a demo archive."],
    "dependencies":["polylogue-t46","polylogue-fnm.12","p6-14"],
    "parent":"polylogue-t46",
    "risks":["Freezing the public model namespace too early can ossify internal payload mistakes.","Optional Arrow/Polars dependencies can complicate install story.","Raw SQL escape hatches can recreate the unsupported-table coupling the SDK is meant to end."],
    "confidence":"med",
    "provenance":"Implements Part-2 Python API/library-consumer surface while noting private Lynchpin claims need live verification."
  },
  {
    "proposed_id":"p6-17",
    "change_kind":"update",
    "target_bead":"polylogue-3tl",
    "title":"First-value docs and executable onboarding",
    "issue_type":"epic",
    "priority":1,
    "labels":["area:docs","area:demo","area:onboarding","from:rnd-file-6"],
    "description":"The docs/onboarding slice is right that a stranger should reach a real result and citable proof on the demo corpus without importing private history or installing a full dev shell. It also correctly says docs lint that only resolves command names does not prove examples run.",
    "design":"Update external-legibility scope to require docs/first-value.md, executable golden-path fenced blocks against a throwaway demo archive, worked query examples above the EBNF grammar, a top-level polylogue explain query teaching verb, and a narrated demo tour. Fold launch-lane README de-meta and install-matrix proof into this route rather than adding a separate docs site. Use generated snippets where possible so docs/help/howtos drift together.",
    "acceptance_criteria":["docs/first-value.md contains only demo-corpus commands and reaches a real search/result/proof artifact in one copy-paste sequence.","verify_doc_commands or a sibling lane executes tagged example blocks against a throwaway demo archive and diffs stable stdout/JSON.","search.md starts with a 10-query worked ladder before the formal grammar.","polylogue explain '<query>' is available in CLI and uses existing parser/explain machinery.","README retained claims are demonstrable via demo tour or cited finding; no unsupported packaged-install claim appears before the release artifact is smoke-tested."],
    "dependencies":["polylogue-3tl.4","polylogue-3tl.7","polylogue-3tl.12","polylogue-3tl.13","polylogue-pj8"],
    "parent":"polylogue-3tl",
    "risks":["Executable docs can become flaky unless demo corpus is deterministic.","Adding teaching surfaces can displace Tier-0 correctness if misprioritized.","Over-narrated docs can dilute the forensic/data-story positioning."],
    "confidence":"high",
    "provenance":"Converts Part-2 docs/onboarding and launch staircase into concrete 3tl updates without creating a separate docs-site gate."
  },
  {
    "proposed_id":"p6-18",
    "change_kind":"new",
    "target_bead":None,
    "title":"Outcome/time pathology measures from actions",
    "issue_type":"feature",
    "priority":2,
    "labels":["area:analytics","area:measures","area:agent-forensics","from:rnd-file-6"],
    "description":"The Part-2 Ideas slice adds a valuable analytical layer: gap taxonomy, retry storms, tool-error recovery latency, context-gluttony, cost-per-outcome, delegation yield, wasted-test-run, and per-session scorecards. These should not become bespoke analyze modes; they should be registered measures over existing action/session/profile units.",
    "design":"Create a small measure bundle over actions/session_profiles/session_gap_profile: retry_storm, tool_error_recovery_latency, context_gluttony_index, wasted_test_run_minutes, and cost_per_successful_outcome. Each measure declares unit frame, denominator, coverage precondition, evidence tier, confounds, and uncertainty. First materializer is session_gap_profile span ledger, then derived pathologies/measures. Exclude low-confidence metrics from launch claims until fixture/eval coverage exists.",
    "acceptance_criteria":["session_gap_profile ledger partitions wallclock into typed spans with coverage/confidence fields and clock-seam tests.","retry_storm and tool_error_recovery fixtures use structural action/error signatures, not prose.","context_gluttony_index counts whole-file reads vs grep/glob-like targeted actions with command/tool provenance.","Each measure has construct-validity metadata and refuses on insufficient coverage instead of rendering false zeros.","A per-session scorecard panel renders only registered measures and footnotes evidence tier."],
    "dependencies":["polylogue-9l5.7","polylogue-9l5.1","polylogue-9l5.8","p6-05","p6-15"],
    "parent":"polylogue-9l5",
    "risks":["Classification of gaps and wasted runs can be subjective without structural signals.","Per-agent/model comparisons can Goodhart if used as sole score.","Sparse timestamps by provider can make measures unavailable for large corpus slices."],
    "confidence":"med",
    "provenance":"Enacts the Part-2 Ideas section while keeping the measure-algebra discipline instead of bespoke analytics."
  }
]
obj={"file":6,"proposed_beads":beads}
print(json.dumps(obj, indent=2))
PY

## [assistant 431]
According to a document from 2026-07-05, I treated `rnd-bundle-6-of-6.md` as **file 6 of 6** and anchored this review to its Part 2 payload, not to the shared master synthesis. The file itself says the unique payload begins after the divider and contains this file’s subagent outputs; I used the preamble only for orientation and collision avoidance. fileciteturn2file10

## Executive verdict

File 6 is one of the stronger slices. Its center of gravity is not “more analytics.” It is the infrastructure that makes Polylogue useful as a real substrate: episode stitching, compaction forensics, cost truth, runtime preferences, search relevance, security hardening, recall, MCP ergonomics, docs/onboarding, temporal doctrine, and public SDK shape.

The strongest conclusions survive adversarial review:

The **cost-accounting defects** are real and urgent. Physical-session rollups, dead provider-reported cost reads, missing current-model credit rates, and hardcoded Pro-tier subscription math are all concrete correctness failures, not speculative design debt. fileciteturn3file3

The **daemon/browser-capture security concerns** are real. DNS rebinding and forged local browser captures are credible attack classes for a loopback daemon that trusts missing `Origin` or lacks `Host` gating; the receiver token and Host/Origin middleware direction is correct, but must not overclaim protection against hostile same-user processes. fileciteturn3file17turn3file18

The **episode unit** is conceptually right but must be more conservative than the initial design. Cross-session “same task” inference is useful only if false merges are prevented by hard corroborators, durable reject/confirm/split judgments, and a visible candidate tier. The file’s false-merge tests are exactly the kind of construct-validity guard this feature needs. fileciteturn2file7turn2file8

The **MCP surface redesign** is directionally correct. Polylogue should not present dozens of near-duplicate tools to agents when the read DSL, resources, and prompts can carry most of the surface. The MCP spec distinguishes resources as URI-addressed contextual data, tools as model-invoked actions, and prompts as reusable interaction templates; it also has capabilities for resource subscription and list-change notifications, which directly support the “continuity API” direction. fileciteturn3file12 citeturn129470search1turn129470search5turn129470search19

The most important correction is prioritization: **do not ship all of this as one frontier wave.** Cost truth, security, config-state cleanup, temporal correctness, and MCP/recall ergonomics are high-leverage. Episodes, ranking-learning, broad entity graphs, SDK/Arrow polish, and launch/docs are valuable, but should be gated behind the credibility fixes and the first external artifact.

## What I confirmed

### Cost accounting is the clearest Tier-0 file-6 item

The cost slice is unusually well grounded. It identifies two rollup paths that sum physical sessions instead of logical sessions, the dead `session_reported_costs` read path, missing current-model credit rates, Pro-tier hardcoding, and the absence of a strict strength lattice for cost provenance. The strongest file-local evidence is the defect table: day/week summaries sum physical profiles; cost rollups sum `total_usd` over physical session insights; `_session_level_estimate` returns `None`; `session_reported_costs` is written but not read; current Opus/Sonnet model rates are absent; and subscription USD uses `credit_cost / 21_700_000 * 20.0`. fileciteturn3file3

My verification against the code snapshot confirmed the core defects. The only nuance: the phrase “every subscription-cost figure is silently $0” is too broad if read literally. Catalog-priced/token-derived paths can still produce numbers. The narrower and correct statement is: **provider-reported session-level exact costs are dead, and current-model subscription-credit costs can collapse to zero because the current model credit rates are missing.** That is enough to block external cost claims.

The design should land as one `f2qv` correctness lane, not scattered fixes. It needs one shared cost provenance lattice:

`provider_reported > subscription_credit > catalog_api_equivalent > heuristic > unavailable`

No path should demote within the lattice unless explicitly marked as a fallback. Every output should carry `cost_view = api_equivalent | subscription_credit | provider_reported`, subscription tier assumptions, and an “unconfigured tier” warning rather than silently defaulting to Pro.

### Security hardening is not optional

The security slice correctly distinguishes real risks from residuals. Host/Origin validation before dispatch is the right fix for browser-mediated attacks against loopback services. The browser-capture receiver needs an always-on token or HMAC-like authorization, constant-time comparison, and spool quotas. Query-string bearer tokens should be scoped to routes that actually need EventSource compatibility, not accepted globally. The threat model must keep saying that a hostile same-user process can still read local token files and SQLite databases unless OS/disk protections prevent it. fileciteturn3file17turn3file18

External context strengthens this. GitHub’s DNS-rebinding writeup shows local web apps remain attackable when they trust loopback without robust host/origin validation, and Chrome’s Private Network Access work adds browser preflights but does not remove the server’s obligation to reject bad requests itself. citeturn129470search3turn129470search4

My correction: **do not frame this as “authenticated browser capture” without a threat-class qualifier.** It is “authenticated against confused-deputy browser pages and unauthenticated local POSTs.” It is not “secure against malicious same-user local code.”

### Episodes are useful, but only with candidate-first semantics

The episode unit is a good abstraction: a logical task spanning sessions, providers, cwd roots, time, artifacts, and commits. It should sit above provider lineage/threads, not replace them. The proposed placement is mostly right: derived episode tables in `index.db`, session-level embeddings in `embeddings.db`, and durable human/operator stitch judgments in `user.db.assertions`. The file correctly notes that `session_embeddings` do not exist yet and proposes `message_centroid` as the cheap default over existing message vectors. fileciteturn2file7

The hard false-merge floor is the most important part. Same title, same day, high embedding similarity, or same theme must never be enough to merge sessions. Shared SHA, shared error fingerprint, strong same-repo+time evidence, or explicit user confirmation should be required for corroborated+ edges. The test suite proposed in the file is the right bar: high-embed/different-repo pairs must remain candidates; shared-SHA cross-provider pairs should link; reject/split/confirm judgments must survive rebuild. fileciteturn2file8

My correction: connected components over corroborated edges can still create **transitive over-merge**. If A↔B and B↔C are strong but A and C have negative evidence, naïve connected components merge all three. Episode materialization needs cut constraints and maybe component-level consistency checks, not only pairwise scoring. The file includes split constraints, but acceptance should explicitly test this chain case.

### Compaction lifecycle belongs in durable source/provenance, not only index summaries

The compaction lifecycle design is right in spirit: Polylogue should record compaction boundaries, what was retained/lost/transformed, and how later sessions re-ground lost facts. The file correctly reuses existing concepts: `context-snapshot` refs already exist, content-addressed blobs can deduplicate large snapshots, `session_events(event_type='compaction')` marks boundaries, and `AssertionKind` can grow schema-free. fileciteturn3file0

The risk analysis is also good. PreCompact hooks may not actually carry the assembled context, so capture tier must be honest. “Loss” cannot mean “model forgot,” because summaries may semantically preserve a fact in transformed form. Durable records should store refs, not giant injected prose in `user.db`, and snapshot blobs need retention/purge policy. fileciteturn3file0

My correction: this subsystem should align with a general provenance model. W3C PROV describes provenance as entities, activities, and agents involved in producing data, and OpenTelemetry traces model operations with spans, attributes, events, links, and parent-child relationships. Polylogue does not need to implement either standard wholesale, but compaction events, query runs, analysis runs, and replay plans should all look like provenance DAG nodes, not bespoke one-off tables. citeturn844460search5turn844460search4

### The missing-units slice is mostly correct

The three missing units are valuable because they turn implicit evidence into queryable relations.

`entity-mention` should start as a table, but only structural mentions should be trusted by default. Prose-regex candidates should be candidate-tier, recursive-safety-gated, and capped for confidence. The file itself flags the high false-positive risk for hex IDs, code symbols, and broad regex extraction. fileciteturn3file10

`world-effect` is a good view if the cause comes from explicit observed-event evidence refs to producing actions. It should not infer causality from time adjacency except as a lower-trust optional branch. The file’s proposed `session_observed_events → evidence_refs_json[0] → blocks → actions` path is the right structural spine. fileciteturn3file10

`verification-run` is useful, but the first version should not rely on broad string matching like `LIKE '%test%'` unless it is clearly confidence-tiered. It should reuse or introduce a central command classifier so the same classification feeds claim-vs-evidence, replay, and analytics.

### Config/runtime preferences should not revive `user_settings`

The config slice is correct that deployment config and runtime preferences are different things. `config.py` already handles deployment-layer config; the dead `user_settings` table should not become a second, half-wired state store. The file notes that `user_settings` exists but has no read/write helpers, while assertions already have scope, target, key, value, author, staleness, context policy, status, and supersession fields. fileciteturn2file11

The proposed settings-as-assertions direction is pragmatic, but semantically delicate. A preference is not always an epistemic assertion. I would still use `AssertionKind.SETTING` or an assertion-compatible row shape, but enforce a registry that separates deployment secrets from runtime preferences. The file’s risk list correctly calls out deployment/runtime bleed and secret leakage as the top issue. fileciteturn1file5

The design should require:

A closed registry for runtime keys.

A deployment-only partition for secrets.

Batched scope resolution, not one DB query per setting.

`author_kind=agent` settings must default to candidate and never auto-apply.

A migration that either proves `user_settings` empty or copy-forwards it into assertions with a backup manifest.

### Search relevance needs eval-first discipline

The search relevance slice points in the right direction: collapse duplicate search paths, add `why` explanations, collect relevance judgments, and use `AssertionKind.RELEVANCE` / labels for evaluation. The broader bundle says search resolution currently lives in two paths and should collapse into one shared helper; it also calls out `variant_count`/lineage sibling duplication as a real class of bug. fileciteturn1file13

The external context matters here. Reciprocal Rank Fusion is a well-known IR method for combining multiple result rankings and is scale-insensitive across different scoring systems, which fits Polylogue’s lexical/vector/hybrid setup. citeturn844460search3

My correction: do not add MMR, learning-to-rank, model-weight tuning, or ranker knobs before there is an evaluation corpus. Phase one should be “one resolution path + explainable score decomposition + relevance labels + nDCG/MRR gate.” Only after enough labeled queries exist should Polylogue tune RRF constants, recency weights, or diversity reranking. Also, session-level diversity reranking depends on session embeddings, so it should either depend on the episode/session-embeddings work or stay message-level.

### Recall is a strong near-term agent affordance

The `recall(task_hint)` MCP tool is one of the best file-6 ideas. The load-bearing primitive exists: `VectorProvider.query(text, limit)` embeds arbitrary text and KNN-searches stored message embeddings, whereas existing similar-session flows are session-seeded rather than free-text-seeded. The file’s proposed MCP tool contract makes recall cross-repo by default, attaches lessons/blockers/corrections/pathologies, carries token budget, and uses `min_similarity` instead of padding weak hits. fileciteturn2file2turn2file9

The risk analysis is exactly right: query embedding needs a live provider key, so FTS fallback must be explicit; and trust-class laundering is the main danger. Recall can become a prompt-injection amplifier if an agent-authored lesson is surfaced as operator doctrine. fileciteturn2file9

My correction: the first recall payload should be structured JSON plus a rendered Markdown view, not only a string. Agents need stable refs, trust class, retrieval lane, score, evidence refs, and “why included.” The human-readable report can be a renderer over the payload.

### MCP should become resource/prompt-first, not tool-heavy

The MCP slice’s thesis is correct: Polylogue should not expose a flat wall of tools when it has a DSL, resources, and saved views. The file says Polylogue already has static resources, hardcoded prompts, many tools, and read/write/admin gating; the gaps are resource subscriptions/list-changed and user-authored prompts from saved recall packs/views. fileciteturn2file1

I refute the precise tool count in the file for the code snapshot I inspected. I found fewer than “~130” concrete `@mcp.tool()` definitions and a lower `EXPECTED_TOOL_NAMES` count than that. But the strategic conclusion survives: there are too many overlapping tools, and read operations should move toward `query(expression)` and resource fetches.

The MCP spec supports this directly: resources are URI-addressed context, tools are invocable actions, prompts are reusable templates, and the schema includes resource `subscribe` and `listChanged` capabilities. citeturn129470search1turn129470search5turn129470search19

### Python SDK / library consumer surface is real, but the “async-only” diagnosis is overstated

The file’s SDK slice identifies the right consumer problem: external code reaches into internals, reimplements models, or reads SQLite because there is no small, stable, versioned public surface. That is a real issue. The proposed `polylogue.sdk` / `polylogue.models`, query objects, and columnar bridges are appropriate.

But I would not state “the facade is async-only” as an absolute for the current package snapshot. I saw a sync API namespace in the code tree. The stronger, safer conclusion is: **the public SDK boundary is not stable or attractive enough, so consumers still bypass it.** Build the narrow SDK and model namespace; do not over-index on sync-wrapper as if nothing exists.

### Docs and launch are execution, not more invention

The docs/onboarding slice is practical and should land. A demo-corpus golden path, doc commands that actually execute, “How do I…?” recipes, a query ladder above the grammar, `polylogue explain "<query>"`, and a narrated tour are exactly the missing bridge between powerful substrate and first value. fileciteturn3file13

The product/launch and red-team slices are also correct in the uncomfortable way. The 80/20 minimal core is not enough to be Polylogue, but it is a useful adversarial baseline. The file’s own counter is the right one: keep the cheap differentiators that grep cannot reproduce — `material_origin`, real continuity tools, content-hash idempotency, and split-tier durability — and defer museum-like analytics until live use demands them. fileciteturn3file5turn2file15

## What I refute or downgrade

The tool count in the MCP slice is probably stale or rounded up. The exact number should be regenerated from the current `EXPECTED_TOOL_NAMES` and decorators before any deletion plan. The conclusion “collapse the surface” is right; the claim “~130” should not be used as a cited fact without live regeneration.

The cost slice should not say all costs are zero or every cost path is dead. Provider-reported exact session totals are dead; current-model credit rates are missing; subscription-tier math is hardcoded; rollups double-count lineage. These are severe enough without overstatement.

The SDK slice should not say “async-only” without qualification. The gap is stable public SDK/DTO/columnar ergonomics, not necessarily total absence of sync access.

The security slice should not imply loopback auth solves same-user compromise. It solves browser/network confused deputy and forged local receiver traffic. Same-user residual remains.

The episode slice should not trust connected components alone. Add component consistency tests for transitive false merges.

The entity-mention slice should not ship broad prose/code-symbol mining as fact. Start structural-first, candidate-second.

The launch slice should not create another docs/product iceberg. It should cut to: release tag, cold install proof, one demo-corpus first-value path, one reproducible finding, one short screencast.

## External prior art and design pressure

There are four external anchors worth importing into the design.

First, provenance should be a first-class mental model. W3C PROV’s entity/activity/agent framing and OpenTelemetry’s span/event/link model both support a general “what produced this?” graph. This argues for representing compaction events, recall runs, query runs, evidence packs, replay plans, and reports as linked provenance objects rather than isolated feature tables. citeturn844460search5turn844460search4

Second, replay/reproduction should be conservative. ReproZip is strong prior art for tracing command executions and dependencies into reproducible packages; it shows the right ambition, but also why Polylogue should start with **dry reconstruction and replay plans**, not promises of exact deterministic replay. citeturn844460search2turn844460search18

Third, MCP should use its three primitives properly. Official MCP resources, tools, and prompts map neatly onto Polylogue’s needs: resources for session/query/result state, tools for side effects, prompts for saved workflows/recall packs. Resource subscription/list-change support is especially relevant to live continuity. citeturn129470search1turn129470search5turn129470search19

Fourth, hybrid search should respect IR practice. RRF is a defensible default for fusing rankings, but learning/reranking should be evaluation-gated. The RRF paper’s value is that it fuses rankings simply; it is not a license to add a pile of ranker knobs without labels. citeturn844460search3

## Prioritized take

Tier 0: fix credibility and safety. Land cost correctness, browser/daemon security hardening, config-state cleanup, and temporal clock/doctrine. These directly protect the “honest system of record” claim.

Tier 1: make agents happier immediately. Land `recall(task_hint)` with trust classes and FTS fallback, collapse MCP reads toward `query`/resources, make saved views/recall packs discoverable prompts, and add docs/onboarding that executes against demo data.

Tier 2: enrich the relational substrate. Add missing units where they are structurally grounded: `entity-mention`, `world-effect`, `verification-run`. Add compaction snapshots and loss records, but keep them provenance-linked and retention-aware.

Tier 3: build frontier analytics only after evidence exists. Episode stitching is worthwhile but should be candidate-first and false-merge-gated. Search ranking improvements should wait on relevance labels and eval. SDK/Arrow outputs should follow the service contract, not precede it.

Tier 4: launch discipline. Publish one finding and one working install/demo route before adding more surfaces.

## Proposed bead changes

```json
{
  "file": 6,
  "proposed_beads": [
    {
      "proposed_id": "p6-01",
      "change_kind": "update",
      "target_bead": "polylogue-f2qv",
      "title": "Make cost provenance lattice enforceable",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:cost", "area:honesty", "tier:index", "launch-gate"],
      "description": "Provider-reported session costs are written but not read, current-model subscription credit rates are missing, physical-session rollups double-count lineage, and subscription USD assumes Pro tier. These defects directly block external cost findings.",
      "design": "Implement one cost provenance lattice provider_reported > subscription_credit > catalog_api_equivalent > heuristic > unavailable. Wire session_reported_costs into pricing, add current Anthropic credit-rate coverage, make subscription tier a validated runtime setting, and make every cost payload expose cost_view, tier assumption, provenance, and unconfigured warnings. Do not re-derive lineage dedup; consume 4ts logical grain.",
      "acceptance_criteria": [
        "A fixture with session_reported_costs yields provider_reported status and never falls back to catalog.",
        "Current curated Anthropic model keys have non-null credit rates or an explicit unsupported-model caveat.",
        "Day/week and cost-rollup summaries use logical-session grain and preserve physical/logical counts separately.",
        "Subscription USD refuses or warns when tier is unset; no silent Pro default.",
        "Cost payload schema exposes provenance and cost_view; regression tests cover exact-to-priced demotion."
      ],
      "dependencies": ["polylogue-4ts"],
      "parent": "polylogue-f2qv",
      "risks": ["Tier setting typo silently mis-scales numbers", "Logical representative selection can reintroduce lineage bugs", "Existing reports may change materially"],
      "confidence": "high",
      "provenance": "Cost-accounting correctness vet: confirmed D1-D5; corrected overbroad zero-cost wording."
    },
    {
      "proposed_id": "p6-02",
      "change_kind": "update",
      "target_bead": "polylogue-kwsb",
      "title": "Central Host/Origin gate for daemon HTTP",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:security", "area:daemon", "threat:dns-rebinding"],
      "description": "Loopback HTTP routes rely on partial Origin checks and lack a central Host allowlist. GET routes and web-shell bootstrap remain vulnerable to browser-mediated local attacks if a hostile page can reach the daemon.",
      "design": "Add middleware before dispatch that validates Host and Origin against bind-derived loopback/operator allowlists, with explicit route policy for health/metrics/SSE. Treat Private Network Access as defense-in-depth only; server-side validation remains mandatory. Include same-origin shell allowlist tests and rebinding fixtures.",
      "acceptance_criteria": [
        "GET and POST requests with a hostile Host are rejected before route dispatch.",
        "Legitimate 127.0.0.1, localhost, [::1], and configured api_host variants still load the web shell and SSE.",
        "Origin-missing browser-like requests do not bypass policy except for documented non-browser CLI cases.",
        "Threat-model docs list the exact enforcing middleware and residual same-user risk."
      ],
      "dependencies": [],
      "parent": "polylogue-kwsb",
      "risks": ["Over-strict host matching can break the web shell", "Reverse-proxy/devloop cases may need explicit config", "False sense of same-user protection"],
      "confidence": "high",
      "provenance": "Security hardening vet: Host/Origin gap confirmed and strengthened with DNS-rebinding prior art."
    },
    {
      "proposed_id": "p6-03",
      "change_kind": "new",
      "target_bead": null,
      "title": "Harden browser-capture receiver auth and spool",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:browser-capture", "area:security", "area:daemon"],
      "description": "The browser-capture receiver can accept forged loopback posts when no token is configured, and large repeated posts can fill the spool. This can poison the archive with fabricated browser captures.",
      "design": "Auto-mint a 0600 receiver token, deliver it through the extension handshake, compare with hmac.compare_digest, keep explicit override for dev, and add per-origin/body/spool quotas with backpressure. Mark capture artifacts untrusted until provenance checks pass.",
      "acceptance_criteria": [
        "A tokenless curl POST to the receiver is rejected by default.",
        "The extension receives and uses the minted token in an end-to-end test.",
        "Bearer comparison uses constant-time compare.",
        "Spool quota/rate-limit tests fail closed and emit an ops event.",
        "Docs state that same-user processes remain out of scope."
      ],
      "dependencies": ["p6-02"],
      "parent": "polylogue-kwsb",
      "risks": ["Token injection failure can silently break dogfood capture", "Quota defaults can drop legitimate bulk captures", "Local same-user residual may be overclaimed"],
      "confidence": "high",
      "provenance": "Security/privacy red-team: forged capture and spool-DoS risks confirmed."
    },
    {
      "proposed_id": "p6-04",
      "change_kind": "update",
      "target_bead": "polylogue-w8db",
      "title": "Runtime settings as guarded assertion state",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:config", "tier:user", "area:assertions"],
      "description": "Deployment config already has a layered resolver, but runtime preferences lack a wired db layer. The user_settings table exists but is dead; assertions have the columns needed for scoped settings.",
      "design": "Represent runtime preferences as AssertionKind.SETTING or an assertion-compatible state row, backed by a typed registry. Enforce deployment/runtime partition, secret-key exclusion, batched resolution, scope specificity, actor class, pinning, and judge-gated learned defaults.",
      "acceptance_criteria": [
        "No secret deployment key can be written as a runtime setting.",
        "Resolver precedence flag > env > file > db > default is property-tested.",
        "Scope specificity global < repo < origin < surface is deterministic.",
        "A single batched db query resolves all settings for a context.",
        "Agent-authored setting suggestions land as candidate and never auto-apply."
      ],
      "dependencies": ["polylogue-37t.11"],
      "parent": "polylogue-w8db",
      "risks": ["Conflating preferences with epistemic assertions", "Secret leakage through MCP/web settings", "Per-key DB lookups hurt CLI startup"],
      "confidence": "high",
      "provenance": "Config/runtime-preferences vet: user_settings dead; assertion substrate suitable with guards."
    },
    {
      "proposed_id": "p6-05",
      "change_kind": "new",
      "target_bead": null,
      "title": "Migrate or retire dead user_settings table",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:config", "tier:user", "schema:migration"],
      "description": "user_settings exists in user.db but has no read/write helpers. Keeping a dead durable table creates schema ambiguity and encourages a second settings store.",
      "design": "Add user.db migration 005 that proves the table empty or copy-forwards rows into SETTING assertions with backup manifest, then retires the table or marks it legacy-inert. Regenerate schemas and user audit surfaces.",
      "acceptance_criteria": [
        "Migration rejects without verified backup manifest.",
        "Seeded pre-migration user_settings rows copy forward exactly into setting assertions.",
        "No live code reads or writes user_settings after migration.",
        "render openapi, cli-output-schemas, and user_audit are clean."
      ],
      "dependencies": ["p6-04"],
      "parent": "polylogue-w8db",
      "risks": ["Durable migration error loses operator settings", "Copy-forward natural key chosen poorly", "Schema bump contention with other user v5 work"],
      "confidence": "med",
      "provenance": "Config vet: dead table confirmed; migration shape corrected to prove/copy before drop."
    },
    {
      "proposed_id": "p6-06",
      "change_kind": "update",
      "target_bead": "polylogue-gjg",
      "title": "Capture compaction snapshots with honest tiers",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:compaction", "tier:source", "area:provenance"],
      "description": "Compaction boundaries exist as session events, but the assembled pre/post context and loss forensics are not durable enough to audit what survived compaction.",
      "design": "Add source-tier compaction_snapshots storing content-addressed blobs and capture_tier; derive compaction_loss in index; store user-tier records as refs only. Never claim exact-context fidelity unless the PreCompact payload proves it. Add retention/purge policy for snapshot blobs.",
      "acceptance_criteria": [
        "Synthetic two-compaction fixture records pre/post snapshot refs and loss rows.",
        "ops reset --index regenerates compaction_loss byte-identically while durable refs survive.",
        "If PreCompact lacks assembled context, capture_tier reports boundary-only fidelity.",
        "Snapshot blobs are protected from GC while referenced and reclaimed after explicit purge.",
        "MCP/CLI resolve context-snapshot and loss refs."
      ],
      "dependencies": ["polylogue-4ts.5", "polylogue-37t.11"],
      "parent": "polylogue-gjg",
      "risks": ["PreCompact hook may not expose assembled context", "Structural loss can be misread as semantic forgetting", "Snapshot blobs contain high-sensitivity context"],
      "confidence": "high",
      "provenance": "Compaction lifecycle vet: current event marker exists but durable snapshot/loss forensics are missing."
    },
    {
      "proposed_id": "p6-07",
      "change_kind": "new",
      "target_bead": null,
      "title": "Represent compaction re-grounding as provenance",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:compaction", "area:provenance", "area:context"],
      "description": "A loss record is only actionable if later sessions show that the lost fact had to be rediscovered or reintroduced. That requires a provenance link, not just a count.",
      "design": "Model re-grounding as edges from downstream messages/actions/assertions/query-runs back to compaction_loss refs. Use provenance-style entity/activity/agent fields and keep confidence tiered. Do not infer forgetting from structural absence alone.",
      "acceptance_criteria": [
        "A fixture where a lost decision is later reintroduced creates a re-grounding edge with evidence refs.",
        "Transformed-but-retained facts do not count as re-grounded loss.",
        "Projection can render loss -> re-grounding -> downstream use as an evidence card.",
        "Every edge carries extractor version and confidence tier."
      ],
      "dependencies": ["p6-06"],
      "parent": "polylogue-gjg",
      "risks": ["Over-mining prose creates false re-grounding", "Graph can grow noisy", "Operators may overinterpret confidence"],
      "confidence": "med",
      "provenance": "Compaction vet plus provenance prior art: loss needs downstream evidence, not extractor opinion."
    },
    {
      "proposed_id": "p6-08",
      "change_kind": "new",
      "target_bead": null,
      "title": "Entity-mention unit with trust tiers",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:units", "tier:index", "area:search"],
      "description": "Polylogue lacks a queryable entity-mention unit for commits, PRs, issues, paths, refs, models, and project identifiers. Regex-only mining risks fabricating facts.",
      "design": "Add entity_mentions as an index table with structural-vs-candidate provenance. Structural mentions come from typed fields and object refs; prose regex mentions are candidate-only with capped confidence. Defer tree-sitter code-symbol mining until a follow-up.",
      "acceptance_criteria": [
        "Structural commit/PR/path refs materialize with provenance=structural.",
        "Prose hex-like matches materialize as candidate with confidence cap and evidence span.",
        "No candidate mention is used as fact by default in reports or recall.",
        "Rebuild row counts and index sizes are measured before exposing bulk terminals."
      ],
      "dependencies": ["polylogue-37t.11"],
      "parent": null,
      "risks": ["Regex false positives", "Table size explosion", "Candidate facts laundering into context"],
      "confidence": "med",
      "provenance": "Missing-units vet: entity mention useful only if trust-tiered."
    },
    {
      "proposed_id": "p6-09",
      "change_kind": "new",
      "target_bead": null,
      "title": "World-effect view from observed events to actions",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:units", "tier:index", "area:provenance"],
      "description": "Observed events such as commits, file writes, and test results should be queryable as effects caused by specific actions when structural evidence exists.",
      "design": "Create a world_effects runtime query/view over session_observed_events joined through evidence_refs_json to blocks and actions. Use only structural evidence for the primary lane; optional temporal/path fallback must be lower-trust and explicitly marked.",
      "acceptance_criteria": [
        "A tool action producing an observed event appears with cause_action_block_id, tool command, exit code, and object refs.",
        "Events without structural evidence do not get a fabricated cause.",
        "Optional fallback rows carry lower confidence and detection_type.",
        "Query DSL can select world-effects and project cause/effect fields."
      ],
      "dependencies": [],
      "parent": null,
      "risks": ["evidence_refs_json shape drift", "Temporal fallback overclaims causality", "Persistent cross-tier view trap if assertions are joined later"],
      "confidence": "med",
      "provenance": "Missing-units vet: world-effect is valuable when grounded in observed_event evidence refs."
    },
    {
      "proposed_id": "p6-10",
      "change_kind": "new",
      "target_bead": null,
      "title": "Verification-run unit from structural tool outcomes",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:units", "area:claim-vs-evidence", "tier:index"],
      "description": "Verification commands and their pass/fail outcomes are central to claim-vs-evidence, but should be queryable as a unit rather than re-mined in every analysis.",
      "design": "Add a verification-run projection over actions/tool outcomes using a central command classifier. Include command, cwd/path, exit code, stderr snippet refs, pass/fail/unknown, and confidence. Avoid broad LIKE-only classification except as candidate tier.",
      "acceptance_criteria": [
        "pytest/npm/nix/devtools fixtures classify correctly with structural pass/fail from exit_code.",
        "Unknown verification-like commands remain unknown or candidate, not pass/fail.",
        "Claim-vs-evidence reports consume verification-run unit instead of local regex mining.",
        "Unit appears in query schema and MCP resource/query output."
      ],
      "dependencies": [],
      "parent": null,
      "risks": ["Command classifier misses project-specific test runners", "False pass/fail from wrapper commands", "Duplicate logic with claim-vs-evidence"],
      "confidence": "high",
      "provenance": "Missing-units vet: verification-run is high-value and structurally grounded in actions."
    },
    {
      "proposed_id": "p6-11",
      "change_kind": "new",
      "target_bead": null,
      "title": "Episode embeddings and false-merge scorer",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:episodes", "tier:embeddings", "area:lineage"],
      "description": "Episodes need session-level vectors and a conservative scorer to group distinct sessions into logical tasks without conflating merely similar work.",
      "design": "Add session_embeddings in embeddings.db v2 using message_centroid as the default and summary_text fallback. Implement pure four-signal scorer over repo, time, embedding, and artifacts with a hard false-merge floor and negative evidence. No DB writes in scorer.",
      "acceptance_criteria": [
        "Eligible sessions receive session vectors without extra Voyage calls when message embeddings exist.",
        "High-embedding/same-title/different-repo fixture never exceeds candidate tier.",
        "Shared SHA or shared error fingerprint cross-provider fixture reaches linked/corroborated as configured.",
        "Scorer is pure and unit-tested with monotonicity and negative-evidence fixtures."
      ],
      "dependencies": ["polylogue-mhx.6", "polylogue-7xv"],
      "parent": null,
      "risks": ["Embedding similarity over-merges themes", "Session-vector quality varies by provider", "ANN blocking can drop true positives"],
      "confidence": "high",
      "provenance": "Episode vet: session_embeddings gap confirmed; false-merge floor is load-bearing."
    },
    {
      "proposed_id": "p6-12",
      "change_kind": "new",
      "target_bead": null,
      "title": "Episode materializer and stitch judgments",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:episodes", "tier:index", "tier:user"],
      "description": "Episode groups need deterministic derived materialization plus durable human/operator judgments that survive index rebuilds.",
      "design": "Add episodes, episode_members, episode_edges, and episode_produced_refs to index.db; add ObjectRefKind episode and AssertionKind EPISODE_CONFIRM/SPLIT/REJECT with edge-keyed targets. Build components only from corroborated+ edges and apply split/reject constraints before write.",
      "acceptance_criteria": [
        "Same source rebuild yields identical episode_id and member_set_hash.",
        "EPISODE_REJECT, CONFIRM, and SPLIT survive ops reset --index and are replayed.",
        "Transitive false-merge fixture A-B-C with A/C negative evidence is cut or candidate-only.",
        "Episode cost rollup uses logical sessions and tail-only prefix-sharing fixture.",
        "CLI/MCP/JSON read surfaces render episodes from one descriptor."
      ],
      "dependencies": ["p6-11", "polylogue-4ts"],
      "parent": null,
      "risks": ["Connected components can transitive-merge unrelated work", "Index v25 rebuild cost", "Judgment refs drift if session ids change"],
      "confidence": "med",
      "provenance": "Episode vet: materializer design valid with added component consistency guard."
    },
    {
      "proposed_id": "p6-13",
      "change_kind": "new",
      "target_bead": null,
      "title": "Search relevance spine and why-explain",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:search", "area:dsl", "area:honesty"],
      "description": "Search relevance is split across paths and lacks a single explainable ranking surface. Ranking changes cannot be trusted without shared resolution and score decomposition.",
      "design": "Collapse session resolution into one helper used by CLI/API/MCP/daemon, preserve RRF score components, lineage/variant collapse, and add search_result why/explain payloads. Do not add new ranker knobs in this bead.",
      "acceptance_criteria": [
        "CLI/API/MCP queries with same DSL return the same ordered refs and route_state.",
        "Each hit can explain lexical/vector/RRF/recency/lineage contributions when present.",
        "Lineage siblings collapse consistently and expose physical/logical counts.",
        "Regression test covers known duplicate-sibling and path-query drift class."
      ],
      "dependencies": ["polylogue-t46", "polylogue-fnm.11"],
      "parent": "polylogue-fnm",
      "risks": ["Facade and daemon paths remain subtly divergent", "Explain payload becomes too heavy", "Existing golden outputs change"],
      "confidence": "high",
      "provenance": "Search vet: collapse and explain first; ranking knobs after eval."
    },
    {
      "proposed_id": "p6-14",
      "change_kind": "new",
      "target_bead": null,
      "title": "Relevance judgments and search-eval gate",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:search", "area:evaluation", "tier:user"],
      "description": "Ranking changes need labels and repeatable metrics; otherwise MMR, recency boosts, or learned weights become subjective churn.",
      "design": "Add AssertionKind.RELEVANCE or equivalent judgment rows for query-result pairs, a small labeled corpus, and an eval lane reporting nDCG/MRR/coverage. Ranking changes must pass the gate or stay experimental.",
      "acceptance_criteria": [
        "Operators can mark relevant/irrelevant search hits with query/result refs.",
        "Eval runner reports nDCG, MRR, label count, query count, and coverage.",
        "Below minimum label coverage, eval returns insufficient evidence.",
        "Ranker changes cannot be promoted without eval comparison artifact."
      ],
      "dependencies": ["p6-13"],
      "parent": "polylogue-fnm",
      "risks": ["Too few labels to guide ranking", "Goodharting narrow labeled set", "Judgment UI friction prevents adoption"],
      "confidence": "high",
      "provenance": "Search relevance vet: eval must precede MMR/LTR/weight tuning."
    },
    {
      "proposed_id": "p6-15",
      "change_kind": "new",
      "target_bead": null,
      "title": "Recall(task_hint) facade and MCP tool",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:mcp", "area:recall", "area:context"],
      "description": "Agents need a mid-session way to ask for similar prior work from a free-text task hint. Existing similarity flows are session-seeded; VectorProvider.query already supports text-anchored message search.",
      "design": "Add Polylogue.recall_report and MCP recall(task_hint). Embed task_hint through VectorProvider.query when available, aggregate to sessions, attach refs and active assertions/pathologies, and fall back honestly to FTS when query embedding is unavailable.",
      "acceptance_criteria": [
        "Given a task hint, recall returns ranked sessions with retrieval_lane and score/floor metadata.",
        "No hit below min_similarity is padded into the result.",
        "If Voyage/query embedding is unavailable, FTS fallback is explicit in payload and rendered report.",
        "MCP discovery, tool contract, and schema tests pass.",
        "Demo corpus smoke test runs without private data."
      ],
      "dependencies": ["polylogue-mhx.4", "polylogue-37t.11"],
      "parent": "polylogue-s7ae",
      "risks": ["Provider key unavailable", "Weak FTS fallback disappoints", "Cross-repo recall returns sensitive/private contexts"],
      "confidence": "high",
      "provenance": "Recall vet: text-to-vector primitive confirmed; free-text recall gap is real."
    },
    {
      "proposed_id": "p6-16",
      "change_kind": "new",
      "target_bead": null,
      "title": "Recall trust classes and budgeted evidence packs",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:recall", "area:recursive-safety", "area:context"],
      "description": "Recall can launder agent-authored claims into future agent context unless every item carries trust class, evidence refs, and budget/degradation state.",
      "design": "Return structured RecallPayload with OPERATOR/QUOTED/SYSTEM trust class, evidence refs, token budget, dropped/ref-only reasons, and rendered Markdown. Candidate/agent-authored assertions must never render as directives.",
      "acceptance_criteria": [
        "Prompt-injection fixture cannot cause QUOTED text to render as OPERATOR instruction.",
        "Over-budget recall returns ref-only/dropped-with-reason entries, not silent truncation.",
        "Every recalled assertion includes author_kind, status, evidence refs, and trust class.",
        "Payload and Markdown renderer are golden-tested."
      ],
      "dependencies": ["p6-15", "polylogue-37t.11"],
      "parent": "polylogue-s7ae",
      "risks": ["Trust classifier false promotion", "Renderer strips safety metadata", "Budgeting drops the only useful item"],
      "confidence": "high",
      "provenance": "Recall vet: trust-class laundering is the main safety risk."
    },
    {
      "proposed_id": "p6-17",
      "change_kind": "new",
      "target_bead": null,
      "title": "Collapse MCP reads into query/resources/prompts",
      "issue_type": "epic",
      "priority": 1,
      "labels": ["area:mcp", "area:api-contract", "area:agent-affordance"],
      "description": "The MCP surface has overlapping read tools, static resources, and hardcoded prompts. Agents need a small navigable surface: query for composition, resources for ref fetches, prompts for saved workflows.",
      "design": "Define target MCP shape: query/explain/get/assert/retract/maintenance plus resource fetches and prompts. Move read-heavy get/list operations to polylogue:// resources where possible; keep side effects as tools. Generate an equivalence table before deleting any tool.",
      "acceptance_criteria": [
        "Current MCP tools are mapped to keep/collapse/delete with role-gating preserved.",
        "At least one search/list/aggregate family is served through query(expression) with golden parity.",
        "polylogue:// resource refs round-trip from query hits to fetch.",
        "No tool is removed until an equivalence golden proves replacement behavior."
      ],
      "dependencies": ["polylogue-t46", "p6-13"],
      "parent": "polylogue-s7ae",
      "risks": ["Silent capability loss", "Agent prompts trained on old tool names fail", "Resources lack enough context for existing use cases"],
      "confidence": "med",
      "provenance": "MCP vet: exact tool count stale, but surface-collapse direction confirmed."
    },
    {
      "proposed_id": "p6-18",
      "change_kind": "new",
      "target_bead": null,
      "title": "Dynamic MCP prompts from saved views",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:mcp", "area:prompts", "area:user-state"],
      "description": "Saved recall packs and views are useful only if agents discover and invoke them naturally. Hardcoded prompts do not reflect the operator's curated context.",
      "design": "Expose saved recall packs/views as dynamic MCP prompts, with prompt listChanged notifications and URI refs to underlying resources. A prompt is a named query/projection/render recipe, not arbitrary hidden code.",
      "acceptance_criteria": [
        "Saving a recall pack or view can register a corresponding MCP prompt.",
        "Prompt list_changed fires when prompt-backed saved views change.",
        "Prompt invocation returns structured messages with refs, not opaque prose only.",
        "Deleting or renaming saved view updates the prompt catalog."
      ],
      "dependencies": ["p6-17", "polylogue-fnm.12"],
      "parent": "polylogue-s7ae",
      "risks": ["Prompt catalog grows noisy", "Saved query changes surprise agents", "Hidden prompt behavior becomes hard to audit"],
      "confidence": "med",
      "provenance": "MCP surface vet: prompts should be user-authored saved workflows."
    },
    {
      "proposed_id": "p6-19",
      "change_kind": "update",
      "target_bead": "polylogue-cpf",
      "title": "Temporal doctrine and clock seam",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:temporal", "area:query", "area:honesty"],
      "description": "Relative dates, bucket boundaries, provider time, ingest time, sort keys, and unknown timezones can silently distort analytics and search windows.",
      "design": "Land a time doctrine and code seam: query-time clock injection, UTC epoch-ms canon, half-open intervals, explicit time_kind/time_confidence, tz-unknown default, skew band, and no import-time relative base. Add deny-lint for datetime.now in query-sensitive code.",
      "acceptance_criteria": [
        "since:7d uses injected clock in tests and cannot be anchored at import time.",
        "Date windows are half-open [since, until).",
        "Timeless/synthetic sessions surface time_confidence instead of sorting as 1970 without warning.",
        "Calendar bucket tests cover operator timezone, UTC boundary, DST, and backfilled data.",
        "New durable DDL with naive TEXT timestamps is rejected by policy lint."
      ],
      "dependencies": [],
      "parent": "polylogue-cpf",
      "risks": ["Changing sort semantics breaks cursors", "Backfill order shifts surprise users", "Temporal confidence metadata leaks into many payloads"],
      "confidence": "high",
      "provenance": "Temporal vet: clock/relative-date failure modes are launch-grade honesty defects."
    },
    {
      "proposed_id": "p6-20",
      "change_kind": "new",
      "target_bead": null,
      "title": "Public SDK namespace and stable models",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:sdk", "area:api-contract", "area:python"],
      "description": "External consumers bypass Polylogue internals because the public Python surface is too broad and not clearly versioned. The issue is stable ergonomics, not simply absence of sync access.",
      "design": "Create polylogue.sdk and polylogue.models as a narrow versioned public API with frozen DTO re-exports, sync-friendly entry points, schema pin/warn, and explicit internal/private boundary. Add query object ergonomics and columnar export after the core surface is stable.",
      "acceptance_criteria": [
        "A consumer can list/search/read/query_units without importing polylogue.archive, storage, or insights internals.",
        "Public models are importable from one namespace and have semver/schema version metadata.",
        "Layering lint blocks external examples from using internal modules.",
        "A migration guide replaces raw sqlite/profile examples with SDK calls."
      ],
      "dependencies": ["polylogue-t46"],
      "parent": null,
      "risks": ["Freezing unstable DTOs too early", "SDK duplicates facade instead of narrowing it", "Sync wrapper hides async lifecycle bugs"],
      "confidence": "med",
      "provenance": "Python API vet: async-only diagnosis corrected; stable public SDK gap remains."
    },
    {
      "proposed_id": "p6-21",
      "change_kind": "update",
      "target_bead": "polylogue-3tl",
      "title": "First-value docs and executable demo path",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:docs", "area:demo", "launch-gate"],
      "description": "A cold reader cannot reach a real cited result quickly. Docs are subsystem-oriented, examples are not fully executed, and the query DSL is taught as grammar before use.",
      "design": "Add docs/first-value.md using only polylogue demo commands; make verify_doc_commands execute tagged golden-path blocks against a throwaway demo archive; add task how-tos, query ladder, polylogue explain, and narrated tour.",
      "acceptance_criteria": [
        "A clean machine can run the first-value doc against demo data without private archive setup.",
        "Tagged example blocks execute in CI and diff expected stdout/artifact shape.",
        "search.md starts with worked queries before formal grammar.",
        "polylogue explain returns AST/lane/unit/lowering explanation for at least three demo queries.",
        "tour --narrate teaches search, audit, recall, and evidence refs."
      ],
      "dependencies": ["polylogue-3tl.11", "polylogue-3tl.12"],
      "parent": "polylogue-3tl",
      "risks": ["Docs become another launch blocker", "Demo corpus drifts", "Narrated tour adds maintenance cost"],
      "confidence": "high",
      "provenance": "Docs/onboarding vet: these are high-value execution tasks, not new product surface."
    },
    {
      "proposed_id": "p6-22",
      "change_kind": "update",
      "target_bead": "polylogue-3tl",
      "title": "Launch scope guard and release staircase",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:launch", "area:positioning", "scope-guard"],
      "description": "The strongest product risk is substituting more substrate for one external artifact. Launch needs a narrow staircase: release tag, cold install proof, README de-meta, one regenerated finding, and one short recording.",
      "design": "Add a launch gate that blocks new non-critical capability work until the first external artifact is shipped. Keep identity as local-first personal substrate/system of record; avoid memory-uplift claims until measured. Treat release packaging as decision/execution, not feature R&D.",
      "acceptance_criteria": [
        "First release tag is cut or explicitly deferred with owner/date.",
        "Install matrix is proven on a cold machine and docs no longer say packaged path unavailable if packages exist.",
        "README retained claims each link to a demo command or cited finding.",
        "One flagship finding is regenerated on current archive/schema and published with reproduction command.",
        "New frontier beads are marked deferred unless they unblock the artifact."
      ],
      "dependencies": ["p6-21", "polylogue-cfk"],
      "parent": "polylogue-3tl",
      "risks": ["Scope guard becomes process theater", "Release exposes unstable UX", "Artifact delayed by correctness blockers"],
      "confidence": "med",
      "provenance": "Whole-product/red-team vet: 80/20 critique accepted as launch discipline, not product deletion."
    },
    {
      "proposed_id": "p6-23",
      "change_kind": "new",
      "target_bead": null,
      "title": "Replay plan from actions and repo evidence",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:replay", "area:provenance", "tier:ops"],
      "description": "Polylogue can reconstruct work better if tool calls, repo commits, generated artifacts, and observed events are linked into a dry replay plan. Exact replay should not be promised yet.",
      "design": "Add ops-tier replay_plan over actions, observed events, session_commits, artifacts, and verification-runs. Phase one emits dry plans and completeness gaps only; gated execution with network-deny/frozen-clock is follow-up. Align shape with provenance concepts rather than one-off report tables.",
      "acceptance_criteria": [
        "A session with shell actions and commits renders an ordered dry replay plan with cwd, command, exit code, and produced refs.",
        "Missing cwd/env/file snapshot information is surfaced as incompleteness, not ignored.",
        "No command is executed in phase one.",
        "Plan can be attached as evidence to a report or analysis run."
      ],
      "dependencies": ["p6-09", "p6-10"],
      "parent": null,
      "risks": ["Overpromising reproducibility", "Sensitive commands rendered by default", "Incomplete provenance misleads users"],
      "confidence": "med",
      "provenance": "Compaction/replay/repo-analysis vet plus ReproZip prior art: start with dry reconstruction."
    },
    {
      "proposed_id": "p6-24",
      "change_kind": "new",
      "target_bead": null,
      "title": "Search ranker knobs gated by eval",
      "issue_type": "task",
      "priority": 3,
      "labels": ["area:search", "area:evaluation", "defer-until-evidence"],
      "description": "MMR, recency boosts, learned weights, and diversity reranking are tempting but should not ship before the relevance-eval corpus exists.",
      "design": "Create a deferred bead that records the allowed ranker experiments and their promotion gate. It may prototype RRF constant, recency weighting, diversity/MMR, and model/source weights only behind eval comparison and coverage floors.",
      "acceptance_criteria": [
        "Bead remains blocked until p6-14 reports sufficient labeled queries.",
        "Each ranker variant emits before/after nDCG/MRR and query coverage.",
        "MMR/session-diversity variant depends on session embeddings or explicitly stays message-level.",
        "No ranker variant changes default production ranking without eval artifact."
      ],
      "dependencies": ["p6-14", "p6-11"],
      "parent": "polylogue-fnm",
      "risks": ["Eval set too small", "Knobs create unmaintainable ranking behavior", "User-facing relevance worsens despite metric win"],
      "confidence": "high",
      "provenance": "Search vet: ranking experiments are valuable but must be evidence-gated."
    }
  ]
}
```
