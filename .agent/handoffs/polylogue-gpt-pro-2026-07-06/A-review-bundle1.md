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

## [assistant 337]
bash -lc cd /mnt/data/poly_tree/polylogue && python3 - <<'PY'
from pathlib import Path
files=['polylogue/insights/rigor.py','polylogue/insights/audit.py','polylogue/insights/temporal_source.py','polylogue/insights/transforms.py','polylogue/insights/registry.py','polylogue/insights/archive.py','polylogue/insights/archive_summaries.py','polylogue/insights/archive_rollups.py','polylogue/storage/sqlite/archive_tiers/user.py','polylogue/storage/sqlite/archive_tiers/user_write.py','polylogue/core/enums.py','polylogue/core/refs.py','polylogue/archive/query/metadata.py','polylogue/storage/sqlite/archive_tiers/index.py','polylogue/storage/sqlite/archive_tiers/source.py','polylogue/archive/query/expression.py','polylogue/daemon/http.py']
for f in files:
 p=Path(f)
 print(f, p.exists(), p.stat().st_size if p.exists() else '')
PY

## [assistant 398]
bash -lc cd /mnt/data/poly_pkg/polylogue && python3 - <<'PY'
import json, re
issues={}
for l in open('polylogue-beads-export.jsonl'):
 d=json.loads(l)
 if d.get('_type')=='issue': issues[d['id']]=d
ids=['polylogue-9e5','polylogue-37t','polylogue-l4kf','polylogue-fs1.5','polylogue-9l5','polylogue-3tl','polylogue-bby','polylogue-4smp','polylogue-mhx','polylogue-fnm.13','polylogue-4p1','polylogue-s7ae','polylogue-f2qv','polylogue-83u','polylogue-1xc','polylogue-cpf','polylogue-7aw','polylogue-rii','polylogue-0v9p','polylogue-jgw','polylogue-gjg','polylogue-dmp','polylogue-p5g','polylogue-212','polylogue-3v1']
for id in ids:
 b=issues.get(id)
 print('\n',id,'EXISTS' if b else 'missing')
 if b:
  print(b.get('title'), 'P',b.get('priority'),'status',b.get('status'),'type',b.get('issue_type'))
PY

## [assistant 402]
bash -lc cd /mnt/data/poly_pkg/polylogue && python3 - <<'PY'
import json,re,textwrap
issues=[]
for l in open('polylogue-beads-export.jsonl'):
 d=json.loads(l)
 if d.get('_type')=='issue': issues.append(d)
terms=['RigorFieldContract','FINDING','standing query','TRIAGED','worth_reviewing','polylogue-export','CIF','Atropos','Recursive-safety','provenance_state','FTS coherence','fts drift','fidelity','parser_fingerprint','content-variant','citation anchor','evidence basket','measurement registry','measure registry','query object','result_sets','finding']
for term in terms:
 print('\nTERM',term)
 count=0
 for b in issues:
  txt=' '.join(str(b.get(k,'')) for k in ['id','title','description','design','acceptance_criteria','labels'])
  if term.lower() in txt.lower():
   print(b['id'], 'P'+str(b.get('priority')), b.get('status'), b.get('issue_type'), b['title'][:100])
   count+=1
   if count>=12: break
PY

## [assistant 414]
I treated file 1’s unique Part 2 as the only payload to turn into recommendations. I used the shared preamble only to understand naming and cross-cutting context, then checked the concrete Part‑2 claims against the uploaded code snapshot where possible. The file identifies itself as “file 1 of 6” and says the unique payload is the 14 subagent outputs after the divider, which is the scope I kept to. fileciteturn2file5

One caveat: I could verify many file/path claims against the uploaded `polylogue-all` code snapshot, but I did not have the live `/realm/project/polylogue` tree or the original `/tasks/*.output` files. For anything that depends on the current branch after July 5, the code should be re-opened before patching.

## Executive verdict

The strongest parts of this file are real and important: honesty-audit completeness, recursive-safety, FTS coherence, query/finding objects, and the idea of Polylogue becoming a producer of external standards rather than only a consumer of exports. Those should survive.

The weakest parts are where the designs talk as if a content hash, Merkle root, `AssertionKind`, or generated variant magically solves more than it does. Merkle roots do not give membership queries. New assertion kinds are schema-free at SQLite level but not free at surface/schema/audit level. Mechanical variants are not necessarily faithful. Query canonicalization must be conservative because ranking, lanes, time, macros, and implicit defaults can make superficially equivalent expressions non-equivalent.

My prioritized take: land safety and honesty first, then query/finding objects, then FTS/ingest invariants, then interop/export surfaces. Do not build the large measure catalog or evidence cockpit before the substrate produces one or two citable, regenerated findings.

## Confirmed hard problems

The honesty-audit incompleteness is real. The current registry has more number-bearing insight products than the rigor matrix covers, and the audit path iterates the contracts rather than reconciling against the registry. That means an insight product with no contract can disappear from the audit instead of showing `coverage_status=uncovered`. The payload’s proposed fix is right: iterate the product registry, emit explicit uncovered rows, and make number-over-empty and text-derived provenance checks executable rather than prose. The unique file-1 material names the uncovered cost/coverage/tool/debt cluster and the contract iteration failure directly. fileciteturn2file17

The temporal provenance bug is also real. The aggregate high-water mark classifier collapses input provenance toward `provider_ts` instead of propagating the weakest contributing source. The correction should be stronger than “fix one helper”: the leaf classifier also needs audit because any aggregate fix built over already-laundered leaf values is only half a fix. The file-1 material itself calls out that risk. fileciteturn2file2

The forensic transform honesty problem is real. The code distinguishes tool-result exit status from prose-derived artifacts for some axes, but commit SHAs, decisions, caveats, and similar fields can still be mined from prose without being carried as `text_derived` or unverified. The design direction is correct: payload models, not just bundle prose, must expose which fields are structural evidence and which are text-derived inference. fileciteturn2file17

The recursive-safety diagnosis is confirmed and should be treated as a top-tier safety bug, not a nice-to-have. The durable `assertions` table already has `author_kind`, `status`, `evidence_refs_json`, and `context_policy_json`; candidate promotion already flips judged rows to `author_kind='user'`; but the injection gate does not require `author_kind='user'`, does not resolve material-origin grounding, and can therefore let agent-authored active rows become future context if any write path stamps them incorrectly. The payload identifies this as the self-ingesting archive’s laundering failure mode: auto-captured R&D can become assertions, then future context, without a human or external grounding break. fileciteturn2file0

The `blackboard_post` class of bug is particularly plausible: if any path calls `upsert_assertion` with `author_kind=agent`, `status=ACTIVE`, and `inject:true`, the current central write layer will accept it unless guarded upstream. The right fix is a single `upsert_assertion` chokepoint coercing every non-user author to `CANDIDATE` plus `inject:false`, while preserving terminal judged rows. This exact invariant is also surfaced in the shared later-wave bug list, but it belongs here because the recursive-safety Part‑2 payload makes it operational. fileciteturn2file1

The FTS coherence design is grounded. The rowid identity `messages_fts.rowid == blocks.rowid == messages_fts_docsize.id` is the keystone, and the current ledger/repair path is a real substrate to build on. The proposed metamorphic testing strategy is unusually good: generate block insert/update/delete sequences through the real write path, assert no missing/excess rows, test empty-text transitions, repair convergence, and ledger-vs-exact agreement. fileciteturn2file13

The FTS risk section is also correct and sharper than the average design note. Rowid reuse means a ghost FTS row can accidentally bind to a different block; count agreement alone is insufficient. The coherence check must join on rowid and confirm `block_id`. Ledger-only O(1) checks are acceptable for startup and Prometheus, but periodic exact reconciliation remains necessary because the ledger can be the thing that drifted. fileciteturn2file10

The query/finding-object gap is real. The current object-ref vocabulary has saved views, recall packs, assertions, transforms, runs, context snapshots, observed events, and similar objects, but no first-class `query`, `result_set`, or `finding`. Saved queries are still mutable names / saved-view assertions, not stable query identities. The file-1 model—`query:<hash>`, result-set snapshots, query DAG, `FINDING` lifecycle, findings-as-tests—is the right architecture. fileciteturn2file4

The standing-query design is pointed in the right direction. A standing query should be a saved query with notification/delta semantics and should write candidate findings or query-delta assertions rather than directly notifying as truth. The payload is correct that dedup should be by stable identity sets, not raw events, because re-ingest storms should not generate repeated notifications for the same underlying finding. fileciteturn2file13

The triage frontier is a plausible and useful read-model. The proposed score uses existing `session_profiles`, terminal/workflow weights, logical-session collapse, gates, and read-time staleness. The key correction in the payload is right: hard-zero only truly disposable sessions; in-flight and superseded branches should be visible demotions, not hidden drops. fileciteturn2file12

## Refuted or corrected points

“Add `AssertionKind.FINDING` with no migration” is true only at the SQLite-table level. The durable table stores `kind` as text, so a new kind does not require a user-tier DDL migration. But the project embeds assertion kinds into OpenAPI, CLI output schemas, user-audit surfaces, and tests. The file itself notes those gotchas, and the bead work must include them; otherwise the change will pass storage and fail surfaces. fileciteturn2file12

The query-hash canonicalization rule is too broad if read literally. `auth and test` can commute only inside a pure Boolean predicate fragment. A full Polylogue query can include ranking, retrieval lanes, semantic/FTS/hybrid policy, limits, time-relative predicates, projection/render defaults, macros, set-algebra operand order, and future stage metadata. The safe canonicalization rule is: normalize pure predicate ASTs conservatively, but do not commute operations that affect rank, cost, scope, or time semantics.

The Merkle-root recommendation is half right. A Merkle root over sorted member keys is excellent for change detection and cheap equality testing. It is not sufficient for “cheap intersect/except membership.” To compute intersections, diffs, or example rows, Polylogue needs either `result_set_members`, a compressed membership structure, or a re-executable query. The right storage split is metadata + Merkle root for every snapshot, and exact member rows only for capped/saved/promoted snapshots that need dereference, citation, or set operations.

“Result sets belong in derived index.db” is right for ordinary snapshots, but not for every user-facing result. A result set referenced by a promoted finding must either be reproducible from durable query+corpus hash or have a durable citation manifest sufficient to explain what was cited after a derived reset. The durable object is not the whole member table; it is the decision/finding plus its evidence anchors and the query/result fingerprint.

The standing-query substrate should not ship before query identity and finding identity exist. Saved-view plus `notify_on` JSON is a good migration path, but stable no-storm semantics depend on stable query IDs, stable result-set/finding IDs, and content-hash anchors. Without those, a standing query can still be an event replay notifier dressed as a finding detector.

The CIF/polylogue-export direction is correct, but `polylogue-export` must preserve two identities at once: the embedded original origin and the fact that the transport envelope itself came from Polylogue. Re-ingesting `export(A)` should be idempotent relative to the original content, but the archive must not lose provenance that the bytes came through a Polylogue export package. The package needs an envelope identity, source manifest, origin-specific parser fingerprint, content hash algorithm, and collision policy.

The mechanical-vs-generative variant axis is useful, but “mechanical” cannot mean “trusted 100%.” OCR can be wrong; transliteration can be lossy; language detection can be ambiguous; captions can omit critical content. Mechanical should mean “deterministic or non-generative transform with measurable coverage/confidence,” not “truth-equivalent.”

The RL/eval export lane is good, but it must stay conservative. The action stream gives strong structural labels for tool outcomes, but reward modeling over corrections is weak if corrections remain session-scoped. Any AUC/accuracy report must include base rate, `n`, correction granularity, and coverage. The file’s warning to verify the live Atropos schema first is necessary: Atropos is a moving RL trajectory/evaluation framework, and recent releases reference format fixes around nested messages and `jsonl2html.py`, so a pinned round-trip test is mandatory. citeturn194146search1turn194146search0

The measure catalog should not be allowed to become a museum. The 5-tuple algebra is the right product shape, but the first implementation should be a registry and single aggregate path for a small number of live measures: count, cost-per-outcome, tool-error recovery latency, retry storm, and maybe delegation yield. The full 16-measure catalog should be backfilled only after the registry proves it prevents lies and reduces duplicate implementation.

## External context and missing prior art

For query/findings-as-objects, the closest product analogues are semantic metric layers and data-quality checks, not just saved searches. dbt’s Semantic Layer centralizes metric definitions so metrics are consistently interpreted across tools, and MetricFlow generates SQL for metrics over dimensions; Cube likewise models measures as aggregate definitions such as sum/count/average/rolling windows. Polylogue’s measure registry should steal that split: metric definition is centralized, query execution is generated, and surfaces are not allowed to invent their own denominator. citeturn117276search4turn117276search0turn117276search1

For construct-validity gates, Dagster asset checks and Great Expectations checkpoints are relevant patterns: they turn quality expectations into executable checks attached to data assets. Polylogue’s twist is stronger evidence semantics: a number is invalid not merely because a check fails, but because its backing rows are missing, text-derived, stale, or using the wrong unit frame. citeturn117276search3turn117276search2

For citation anchors and report baskets, the W3C Web Annotation Data Model is directly relevant. It is designed to share and reuse annotations across systems, including linking arbitrary content and selected segments. Polylogue should not copy the model wholesale, but the `target/body/selector/state` vocabulary is useful when designing evidence baskets, content anchors, variant alignments, and external report exports. citeturn493599search2turn493599search6

For interop outputs, SARIF is the obvious standard for exporting pathologies or agent-behavior defects into GitHub/code-scanning style surfaces. SARIF is an OASIS standard format for static-analysis results; a Polylogue “agent pathology as SARIF” export is a low-cost way to make its findings visible in existing developer workflows. citeturn493599search1turn493599search13

For MCP, the Part‑2 idea that sessions/messages/blocks should become resources and recall packs/saved views should become prompts is aligned with the protocol’s own primitive split. MCP resources are application-controlled contextual data identified by URIs, with optional subscriptions/list-changed support; prompts are reusable prompt templates/instructions exposed by servers. That maps neatly to Polylogue: sessions and evidence packs are resources, recall packs and saved analysis recipes are prompts, and mutations/search remain tools. citeturn799248search0turn493599search4

For OpenTelemetry interop, the right lesson is not “export everything as OTel.” OpenTelemetry semantic conventions define shared meanings for attributes, metric instruments, units, and spans; the GenAI conventions already include provider/model/tool/evaluation-oriented attributes. Polylogue should import/export selected spans/events where useful, while preserving its richer archive-specific evidence model internally. citeturn493599search3turn493599search15

## Strategic synthesis

The file’s real center of gravity is not “more features.” It is a transition from archive search to evidence-bearing analytical objects.

The correct product ladder is:

raw sessions and artifacts → typed units → query objects → result relations → findings/assertions → standing rechecks → evidence reports.

Honesty-audit, recursive-safety, FTS coherence, ingestion fidelity, CIF/export, variants, evidence cockpit, RL export, and the measure catalog all fit that ladder. Anything that does not fit should be treated as optional interop or demo polish.

The first thing to land is the safety/honesty substrate. Agent-authored assertions must not self-inject. Every number-bearing insight product must either have a rigor contract or emit `uncovered`. Text-derived fields must be typed as text-derived. Weakest temporal provenance must propagate. FTS drift must become a measured, tested invariant rather than a boolean readiness hope.

The second thing is query/finding identity. Without `query:<hash>` and result/finding objects, standing queries, evidence reports, findings-as-tests, live recall packs, and analysis recipes all lack stable handles.

The third thing is ingestion and citation stability. `polylogue-export`, CIF, block content hashes, parser fingerprints, raw fidelity, and content variants are not side quests; they make later claims reproducible. But they should be staged so the initial proof artifact can ship.

The fourth thing is external surfaces: SARIF, git notes, MCP resources/prompts, `.well-known/ai-sessions`, Datasette/SQLite publication. These are useful because they make Polylogue’s results legible outside Polylogue. They are not a substitute for the internal proof path.

## Prioritized take

Priority 0: safety and lies. Harden `upsert_assertion`, the inject gate, generated-context-pack exclusion, rigor coverage, text-derived fields, weakest-source propagation, and FTS exact-vs-ledger drift. These are credibility issues.

Priority 1: object identity. Add `query`, `result_set`, and `finding` refs; conservative canonical query hashing; result-set metadata/member storage; content-hash evidence qualifiers; re-runnable findings.

Priority 2: standing analysis. Saved queries with `notify_on`, delta findings, finding drift, triage frontier, and analysis/evidence packs should all build on the object identity layer.

Priority 3: standards and interop. CIF/polylogue-export, SARIF, MCP resources/prompts, git notes, and web evidence cockpit are valuable after stable evidence refs exist.

Priority 4: broad measure catalog, variants, RL export, and “champion the standard” campaigns. These are high upside but should not displace the first citable finding and proof loop.

```json
{
  "file": 1,
  "proposed_beads": [
    {
      "proposed_id": "p1-01",
      "change_kind": "new",
      "target_bead": null,
      "title": "Insight rigor registry coverage gate",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:insights", "area:honesty", "tier:index", "construct-validity"],
      "description": "The insight rigor audit currently proves only that declared contracts work; products with no contract can be silently omitted. The registry has number-bearing products outside the current rigor matrix, so cost, coverage, tool, debt, and timeline surfaces can escape audit entirely.",
      "design": "Make the audit iterate the insight registry, not only the contract list. Emit explicit coverage_status=uncovered rows for registry products lacking contracts. Add RIGOR_EXEMPT with inline justification for genuinely non-number products. Keep this as derived/index logic; no durable migration.",
      "acceptance_criteria": ["Audit output contains one row for every registered insight product or a justified RIGOR_EXEMPT entry.", "A test monkeypatching one contract out of the matrix emits coverage_status=uncovered rather than omitting the product.", "CLI, MCP, and rendered schemas include the new uncovered/covered status fields.", "devtools lab policy insight-honesty fails on an uncovered number-bearing product."],
      "dependencies": [],
      "parent": "polylogue-9e5",
      "risks": ["False confidence if contracts are declared but wrong", "Over-broad exemptions can recreate silent skip"],
      "confidence": "high",
      "provenance": "Honesty-audit completeness: registry-vs-rigor-matrix gap and silent contract iteration."
    },
    {
      "proposed_id": "p1-02",
      "change_kind": "new",
      "target_bead": null,
      "title": "Number-field evidence contracts and null gates",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:insights", "area:honesty", "construct-validity"],
      "description": "A product can emit a quantitative value even when backing rows or payload fields are empty, NULL, or aggregate-only. Structural coverage alone does not prove the number has evidence.",
      "design": "Introduce field-level RigorFieldContract entries for number fields, with provenance class, nullable_when_ungrounded, denominator/unit frame, and evidence-tier metadata. Add number-over-empty gates that force None/uncovered instead of 0.0 when backing evidence is absent.",
      "acceptance_criteria": ["Every number-bearing rigor contract declares number_fields with denominator/unit frame and provenance class.", "Property tests generate all-NULL coverage/cost/tool rows and assert emitted totals are None or uncovered, never 0.0.", "The audit report distinguishes absent evidence, zero value, and not-applicable.", "A rendered insight cannot include a quantitative claim whose backing rows are empty unless the field contract explicitly permits it."],
      "dependencies": ["p1-01"],
      "parent": "polylogue-9e5",
      "risks": ["Byte-compat break in consumers expecting numeric zero", "Wrong contracts can still bless bad numbers"],
      "confidence": "high",
      "provenance": "Honesty-audit completeness: number-over-empty and aggregate-only products need executable evidence gates."
    },
    {
      "proposed_id": "p1-03",
      "change_kind": "new",
      "target_bead": null,
      "title": "Weakest-source temporal provenance propagation",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:temporal", "area:insights", "area:honesty"],
      "description": "Aggregate high-water-mark source classification can launder weak timestamp provenance into provider_ts, making dashboards and freshness logic look better grounded than they are.",
      "design": "Represent aggregate source inputs as TemporalSource values rather than only timestamp strings. Implement weakest_source over the provenance lattice and thread it through archive_summaries, archive_rollups, and materializer payloads. Add a follow-up audit of leaf classify_profile_hwm_source so aggregate truth is not built over already-laundered leaves.",
      "acceptance_criteria": ["Table-driven unit tests cover every TemporalSource pair and assert the weakest source wins.", "An aggregate over provider_ts plus fallback_date emits fallback_date.", "Materializer tests prove archive_summaries and archive_rollups retain input_high_water_mark_source.", "A leaf-source audit reports any updated_at path that cannot justify provider_ts."],
      "dependencies": ["p1-01"],
      "parent": "polylogue-9e5",
      "risks": ["Truth surfacing may change recency sorting and staleness UX", "Leaf provenance may be weaker than expected"],
      "confidence": "high",
      "provenance": "Honesty-audit completeness: temporal_source aggregate laundering and leaf-classifier caveat."
    },
    {
      "proposed_id": "p1-04",
      "change_kind": "new",
      "target_bead": null,
      "title": "Mark prose-mined forensic fields as text-derived",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:insights", "area:forensics", "area:honesty"],
      "description": "Forensic transforms can mine commit SHAs, decisions, caveats, and pass-count-like material from prose while only the exit-code axis is structurally grounded. These fields need to carry unverified provenance in the payload model.",
      "design": "Add text_derived_fields and evidence_class/verified markers to ToolSummary, DecisionCandidate, ForensicIndexEntry, and successor payloads. Keep structural tool outcome fields raw_evidence; mark regex/prose-mined fields as text_derived or inference. Render caveats on bundles and block machine promotion without evidence refs.",
      "acceptance_criteria": ["A digest built from prose containing a SHA and decision marks commit_refs and decision fields as text_derived.", "Tool exit-code outcome remains structural raw_evidence.", "Forensic bundle rendering visibly labels text-derived fields.", "A policy test fails if a quantitative or forensic conclusion is rendered from text-derived fields without caveat or evidence refs."],
      "dependencies": ["p1-02"],
      "parent": "polylogue-9e5",
      "risks": ["Payload model version bump may affect consumers", "Over-labeling can make reports noisy"],
      "confidence": "high",
      "provenance": "Honesty-audit completeness: transforms.py prose mining versus structural exit-code evidence."
    },
    {
      "proposed_id": "p1-05",
      "change_kind": "new",
      "target_bead": null,
      "title": "Coerce non-user assertions at write chokepoint",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:safety", "area:assertions", "area:context"],
      "description": "Any agent or detector path that writes an active injectable assertion can launder its own claim into future context. The central assertion write path currently has the right columns but not the invariant.",
      "design": "Add a coerce_agent_authored chokepoint inside upsert_assertion. For author_kind != user, force status=CANDIDATE and context_policy.inject=false unless the row is a terminal judged row that must not be resurrected. Surface role stamping at MCP/blackboard/recipe/distillery writers but enforce centrally.",
      "acceptance_criteria": ["An agent-written inject:true assertion is stored as candidate with inject:false.", "A user-written active inject:true assertion remains active and injectable.", "judge_assertion_candidate accept still promotes by creating/stamping a user-authored active row.", "Regression test covers blackboard_post and at least one recipe/import path.", "No terminal rejected/deleted/superseded row is resurrected by the coercion path."],
      "dependencies": [],
      "parent": "polylogue-37t",
      "risks": ["Breaks existing tests that assumed detector active writes", "Needs careful migration of old unsafe rows"],
      "confidence": "high",
      "provenance": "Recursive-safety subsystem: author_kind differential trust and blackboard-style ACTIVE hole."
    },
    {
      "proposed_id": "p1-06",
      "change_kind": "new",
      "target_bead": null,
      "title": "Recursive-safety provenance and inject gate",
      "issue_type": "feature",
      "priority": 0,
      "labels": ["area:safety", "area:context", "tier:user", "tier:index"],
      "description": "A self-ingesting archive needs a provenance graph gate: agent-authored claims grounded only in other agent-authored/generated material must not become future context without human or external grounding.",
      "design": "Add user-tier provenance_state and safety_json, reusing TopologyEdgeStatus vocabulary. Harden list_assertion_claims/context preamble injection to require status active, inject true, author_kind user, non-stale, non-quarantined evidence, and no generated_context_pack-only grounding. Implement a daemon convergence stage that detects laundering, assertion cycles, and content-hash drift.",
      "acceptance_criteria": ["Migrated and fresh user.db archives expose provenance_state and safety_json with resolved defaults.", "Injection tests independently fail if author_kind, quarantine, staleness, or generated_context_pack clauses are removed.", "A closed-loop agent-only evidence graph is quarantined and not injected.", "Evidence hash drift downgrades an active assertion to candidate and inject:false.", "Human judgment can repair a quarantined row with provenance_state repaired."],
      "dependencies": ["p1-05"],
      "parent": "polylogue-37t",
      "risks": ["Too-strict quarantine can block legitimate agent R&D", "Cross-tier evidence resolution is complex and must not become a persistent ATTACHed view"],
      "confidence": "high",
      "provenance": "Recursive-safety subsystem: closed-loop laundering, provenance-cycle quarantine, generated-context-pack exclusion."
    },
    {
      "proposed_id": "p1-07",
      "change_kind": "new",
      "target_bead": null,
      "title": "Query ObjectRef and canonical query identity",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:objects", "tier:index"],
      "description": "Arbitrary queries currently lack stable addressable identity; saved views are mutable names and opaque values. Findings, standing queries, query edges, and reusable cohorts need query:<hash> refs.",
      "design": "Add query ObjectRef kind and a conservative canonicalization pipeline over lowered planned AST after macro expansion. Canonicalize only pure predicate fragments where semantics are proven commutative. Include retrieval lane, time-base policy, projection-affecting defaults, and macro expansion fingerprint in the hash input.",
      "acceptance_criteria": ["ObjectRef parsing/resolution accepts query:<hash> and rejects malformed hashes.", "Equivalent pure predicate orderings hash identically in golden tests.", "Queries differing in retrieval lane, limit/rank semantics, time base, macro expansion, or projection-relevant defaults hash differently.", "explain_query_expression can return the canonical query id and normalized representation.", "OpenAPI, CLI output schemas, MCP resolve_ref, and user_audit know the query kind."],
      "dependencies": ["polylogue-fnm.13", "polylogue-4p1"],
      "parent": "polylogue-fnm",
      "risks": ["Over-canonicalization can merge non-equivalent queries", "Hash drift can orphan later result sets"],
      "confidence": "high",
      "provenance": "Query/finding object model: query:<hash> over lowered macro-expanded AST, corrected for conservative semantics."
    },
    {
      "proposed_id": "p1-08",
      "change_kind": "new",
      "target_bead": null,
      "title": "Result relations and query DAG snapshots",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:storage", "tier:index", "area:analytics"],
      "description": "Query runs need durable-enough result identities for standing queries, set-algebra explanations, findings-as-tests, and report citations. A Merkle root alone detects changes but cannot support membership or evidence dereference.",
      "design": "Add derived-tier result_sets metadata with result_set_id, query_id, grain, corpus_epoch, member_count, ranking_json, computed_at_ms, and merkle_root. Add result_set_members only for capped, saved, promoted, or explicitly snapshotted sets. Persist query_edges from set-algebra/explain nodes.",
      "acceptance_criteria": ["A query execution can return result_set_id, grain, member_count, corpus_epoch, and merkle_root.", "Saved/promoted snapshots store exact result_set_members and can be read after process restart.", "Root-only snapshots are clearly marked non-dereferenceable for membership.", "query_edges captures operand-of/refines/supersedes/derived-from edges for set-algebra queries.", "ops reset --index drops derived snapshots cleanly while promoted findings retain enough metadata to report stale/missing derived evidence."],
      "dependencies": ["p1-07"],
      "parent": "polylogue-fnm",
      "risks": ["Large member sets can bloat index.db", "Derived reset can surprise users if durable report semantics are unclear"],
      "confidence": "high",
      "provenance": "Query/finding object model: result_sets in derived tier, query_edges DAG, correction that Merkle root is not membership."
    },
    {
      "proposed_id": "p1-09",
      "change_kind": "new",
      "target_bead": null,
      "title": "Finding assertions and findings-as-tests",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:findings", "area:assertions", "area:honesty", "tier:user"],
      "description": "Polylogue needs a first-class way to store claims derived from queries, keep their evidence anchored, and re-test them as the corpus changes.",
      "design": "Add FINDING as an assertion kind at the surface/model level, targetable at query/result/session/insight refs. Machine findings default candidate and inject:false through the central assertion gate. Evidence refs carry content-hash qualifiers. A StandingFindingStage reruns expected-value findings and emits finding-drift candidates on divergence.",
      "acceptance_criteria": ["FINDING is registered in Python enums/schemas/surfaces/user_audit without user-tier DDL beyond text values.", "Machine-authored findings are candidate and inject:false; user-promoted findings can become active.", "A finding with expected count/statistic reruns against the current query id and emits a candidate drift assertion on mismatch.", "A content-hash evidence mismatch marks the finding stale or candidate before any context injection.", "read finding:<id> resolves to claim, query, result-set metadata, evidence refs, staleness, and last-test result."],
      "dependencies": ["p1-06", "p1-07", "p1-08"],
      "parent": "polylogue-9e5",
      "risks": ["New assertion kind touches many generated surfaces", "False drift if canonical query or corpus epoch semantics are wrong"],
      "confidence": "high",
      "provenance": "Query/finding object model plus recursive-safety: FINDING lifecycle and re-runnable invariants."
    },
    {
      "proposed_id": "p1-10",
      "change_kind": "new",
      "target_bead": null,
      "title": "Standing query subscription stage",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:daemon", "area:notifications", "area:query", "tier:ops"],
      "description": "Saved queries can become change detectors, but event-based notifications can storm on re-ingest and stale identity. Delta computation must be based on stable finding/result identity sets.",
      "design": "Represent subscriptions as SAVED_QUERY assertions with notify_on JSON, plus an ops cursor/membership table. The daemon quiet-window stage evaluates only queries whose conservative dependency signature advanced, compares stable result/finding identity sets, and writes candidate query-delta findings rather than authoritative notices.",
      "acceptance_criteria": ["Saved query with notify_on baseline materializes without notifying on first run.", "Re-ingesting identical content does not emit a duplicate notification.", "Unknown or complex dependency signatures fall back to depends_on=* rather than silent skip.", "Heavy query evaluation is budgeted and over-budget work becomes convergence debt.", "Generated_context_pack and notice.* self-trigger sources are excluded from standing-query producers."],
      "dependencies": ["p1-08", "p1-09"],
      "parent": "polylogue-37t",
      "risks": ["Dependency under-approximation causes false quiet", "Heavy standing queries can starve ingest if not budgeted"],
      "confidence": "med",
      "provenance": "Standing-query substrate: SAVED_QUERY notify_on, identity-set dedup, quiet-window evaluation."
    },
    {
      "proposed_id": "p1-11",
      "change_kind": "new",
      "target_bead": null,
      "title": "FTS drift gauges and metamorphic coherence tests",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:fts", "area:storage", "area:observability", "tier:index", "tier:ops"],
      "description": "FTS readiness is currently too boolean. Operators need drift magnitude, and tests need to prove trigger coherence across arbitrary block mutations, including rowid reuse.",
      "design": "Reuse index.db fts_freshness_state for O(1) current gauges and add ops.db fts_drift_samples for disposable history. Add metamorphic tests over block insert/update/delete sequences through real triggers. Exact checks must validate rowid and block_id, not rowid alone.",
      "acceptance_criteria": ["Prometheus exposes polylogue_fts_drift_rows, source_rows, indexed_rows, and drift_exact from the ledger without COUNT scans on scrape.", "ops.db stores fts_drift_samples with retention trimming.", "Hypothesis/property tests assert missing=0 and excess=0 after arbitrary block op sequences.", "A delete/reinsert rowid-reuse case fails unless block_id equality is checked.", "Startup uses the ledger only when freshness_ready_record_trusted holds; periodic exact reconciliation can update ledger truth."],
      "dependencies": [],
      "parent": "polylogue-1xc",
      "risks": ["Exact scans can stall the single writer on a large archive", "Trusting the ledger can hide the precise drift it is meant to report"],
      "confidence": "high",
      "provenance": "FTS coherence and drift: rowid identity, drift gauges, ops sampling, rowid-reuse risk."
    },
    {
      "proposed_id": "p1-12",
      "change_kind": "new",
      "target_bead": null,
      "title": "Polylogue-export origin and CIF round trip",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:interop", "area:ingest", "area:origins", "tier:source"],
      "description": "Polylogue needs a self-describing export/import package that can round-trip without duplicating or corrupting original origin identities, while remaining useful as a federation/interchange primitive.",
      "design": "Add POLYLOGUE_EXPORT origin and a CIF-like envelope with content_hash_algo, embedded original origin, source manifest, parser fingerprint, fidelity declaration, blob/hash inventory, and export-package provenance. Import reconstructs embedded-origin session IDs where safe, while preserving that the transport package was Polylogue export.",
      "acceptance_criteria": ["Origin enum, provider/origin mapping, parser registry, completeness spec, docs, and tests include POLYLOGUE_EXPORT.", "import(export(A)) yields identical content-derived session/message/block IDs for supported origins.", "The archive records package provenance separately from embedded original origin.", "A hash algorithm mismatch or unknown parser fingerprint fails with a clear fidelity error.", "Two packages with same origin/native_id but different content are quarantined or collision-reported rather than merged silently."],
      "dependencies": [],
      "parent": "polylogue-l4kf",
      "risks": ["Identity collision across machines/origins", "Hash-algorithm drift can break federation"],
      "confidence": "high",
      "provenance": "CIF/polylogue-export lane: Origin enum lacks POLYLOGUE_EXPORT and round-trip export is the invariant."
    },
    {
      "proposed_id": "p1-13",
      "change_kind": "new",
      "target_bead": null,
      "title": "AI-session federation manifest",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:interop", "area:sync", "area:origins"],
      "description": "Selective cross-machine or cross-user exchange needs a discovery manifest that advertises available AI-session archives, content-hash algorithms, origin support, and export endpoints without implying cloud service ownership.",
      "design": "Define .well-known/ai-sessions manifest format with archive id, public key optional, supported origins, content_hash_algo, export profiles, freshness timestamp, and privacy disclaimers. Treat it as a companion to polylogue-export/CIF, not a replacement for sync.",
      "acceptance_criteria": ["Manifest schema is documented and has JSON schema tests.", "polylogue export can emit a manifest for a fixture archive.", "Importer validates content_hash_algo and supported profile before fetching packages.", "Manifest never exposes private absolute archive paths or secret local metadata.", "Round-trip fixture proves manifest-discovered package imports through the same polylogue-export path."],
      "dependencies": ["p1-12"],
      "parent": "polylogue-l4kf",
      "risks": ["Premature standardization without users", "Privacy leaks in published manifests"],
      "confidence": "med",
      "provenance": "Standards lane: .well-known/ai-sessions federation manifest and selective content-hash sync."
    },
    {
      "proposed_id": "p1-14",
      "change_kind": "new",
      "target_bead": null,
      "title": "RL eval trajectory export profile",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:eval", "area:export", "area:actions"],
      "description": "The actions stream can produce useful offline RL/eval trajectories, but external schemas such as Atropos are moving and corrections are mostly session-scoped.",
      "design": "Create a pure read projection exporting action/message trajectories to a pinned JSONL profile. Verify the current Atropos schema at implementation time via round-trip html/viewer tooling. Include base rate, n, correction scope, and label coverage in every evaluation report.",
      "acceptance_criteria": ["A fixture archive exports valid trajectory JSONL for at least one supported profile.", "The selected external schema version is pinned in docs and tested with a round-trip renderer or validator.", "Reward/eval reports include n, base rate, correction granularity, and coverage.", "Session-scoped corrections are labeled as weak supervision, not fine-grained ground truth.", "No write/mutation path is used; export is a read projection only."],
      "dependencies": [],
      "parent": "polylogue-fs1.5",
      "risks": ["External schema drift", "Reward labels too coarse to support claimed model-quality conclusions"],
      "confidence": "med",
      "provenance": "RL/eval export lane: Atropos JSONL warning, actions stream, PROMPT_EVAL kind."
    },
    {
      "proposed_id": "p1-15",
      "change_kind": "new",
      "target_bead": null,
      "title": "Attention triage frontier score",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:analytics", "area:triage", "tier:index", "tier:user"],
      "description": "Polylogue can surface sessions worth human review, but current resume logic is repo-scoped and inverted toward continuation. A triage frontier should rank unresolved, blocker-like, or decision-dense sessions without hiding uncertainty.",
      "design": "Add worth_reviewing_score and breakdown to session_profiles or a derived companion table, collapsed by logical_session_id. Store time-invariant value; apply staleness at read time. Add a TRIAGED assertion kind or status marker in user.db, queried via runtime method rather than persistent cross-tier view.",
      "acceptance_criteria": ["Frontier query returns logical-session representatives with score breakdown and confidence/support levels.", "In-flight and superseded sessions are demoted into visible gated buckets rather than hidden.", "TRIAGED/snooze assertions remove or delay rows through a runtime query method.", "Low-confidence enrichment factors are visibly marked and down-weighted.", "A fixture proves disposable chat clean-finish rows hard-zero while blocker/question sessions surface."],
      "dependencies": [],
      "parent": "polylogue-9l5",
      "risks": ["Heuristic enrichment noise can dominate ranking", "A queue that hides rows can falsely look empty"],
      "confidence": "med",
      "provenance": "Attention/Triage Frontier: score function, gates, triage assertion, and confidence-weighting risks."
    },
    {
      "proposed_id": "p1-16",
      "change_kind": "new",
      "target_bead": null,
      "title": "Measure registry minimal algebra",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:analytics", "area:measures", "construct-validity"],
      "description": "Analytics features are converging on a measure algebra, but implementing a 16-measure museum before a single aggregate path exists risks more surfaces inventing their own denominators.",
      "design": "Implement a minimal measure registry with reducer, unit frame, grouping, window, comparison, uncertainty policy, construct validity metadata, and coverage preconditions. Start with count, cost-per-outcome, retry-storm, tool-error recovery latency, and delegation-yield. Refuse rather than fabricate when coverage gates fail.",
      "acceptance_criteria": ["The five starter measures execute through one aggregate path.", "Each measure declares unit frame, denominator, evidence tier, top confounds, and coverage precondition.", "LOGICAL_SESSION is the default frame unless explicitly overridden.", "Wilson/bootstrap or exact/no-uncertainty policy is declared per measure.", "CLI/MCP render measure metadata and refusal reasons, not just numbers."],
      "dependencies": ["p1-01", "p1-02"],
      "parent": "polylogue-9l5",
      "risks": ["Overbuilt catalog delays proof artifact", "Uncertainty computation can be expensive on local SQLite"],
      "confidence": "high",
      "provenance": "Measure catalog and composition algebra: central registry, construct-validity metadata, and no fabricated aggregates."
    },
    {
      "proposed_id": "p1-17",
      "change_kind": "new",
      "target_bead": null,
      "title": "Ingest fidelity parser fingerprints",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:ingest", "area:fidelity", "tier:source", "tier:index"],
      "description": "Import correctness needs explicit fidelity accounting: parser changes, decode failures, unparsed keys, and round-trip reconstruction should be visible rather than inferred from successful ingest.",
      "design": "Add source-tier parser_fingerprint and decode_failure_class where durable, and derived raw_fidelity records with byte-fidelity ratio bands, unparsed-key census, misclassification tripwires, and parser-improvement reprocess flags. Treat fidelity ratio as diagnostic, with structural round-trip equality as the stronger bar.",
      "acceptance_criteria": ["Raw session acquisition stores parser_fingerprint and decode_failure_class or a documented reason it cannot.", "A fixture with unknown keys reports them in an unparsed-key census.", "Parser fingerprint change can enqueue reprocess-on-improvement without duplicating raw evidence.", "Round-trip reconstruction tests assert structural equality for at least two origins.", "Fidelity reports distinguish ratio band, structural failure, and parser misclassification."],
      "dependencies": [],
      "parent": "polylogue-rii",
      "risks": ["Durable source migration must be batched", "Ratio may be misread as correctness score"],
      "confidence": "med",
      "provenance": "Ingestion robustness/fidelity lane: parser fingerprint, byte-fidelity, unparsed-key census, round-trip bar."
    },
    {
      "proposed_id": "p1-18",
      "change_kind": "new",
      "target_bead": null,
      "title": "Evidence basket and report integrity verifier",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:web", "area:evidence", "area:reports"],
      "description": "A web evidence cockpit needs more than a result list: users should be able to collect blocks/evidence into a report and verify that every cited number, quote, and anchor still resolves.",
      "design": "Add evidence basket/report objects over content-hash anchors, with Markdown/HTML twin rendering, anchor resolver states, report-level integrity verification, and degraded/refusal manifest. Use W3C-style annotation concepts as a design reference but keep Polylogue refs native.",
      "acceptance_criteria": ["User can add sessions/messages/blocks/findings to an evidence basket and render a Markdown report from fixtures.", "Every report citation has an anchor resolution state: exact, drifted, deleted, ambiguous, or unavailable.", "Integrity verifier fails the report if a required anchor no longer resolves or a number lacks query/finding backing.", "Rendered report includes corpus datasheet/fingerprint and refusal/degraded manifest.", "Browser and CLI render the same report object from one payload."],
      "dependencies": ["p1-09"],
      "parent": "polylogue-bby",
      "risks": ["Duplicated block hashes can create ambiguous anchors", "Report UX can absorb effort before proof findings ship"],
      "confidence": "med",
      "provenance": "Evidence cockpit lane: report basket, citation anchors, integrity verifier, citable numbers."
    },
    {
      "proposed_id": "p1-19",
      "change_kind": "new",
      "target_bead": null,
      "title": "Content variants with coverage and staleness",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:variants", "area:content", "area:search", "tier:user"],
      "description": "Translations, summaries, OCR, captions, and simplifications should be attached as variants without confusing them with original evidence. Mechanical provenance is not equivalent to perfect fidelity.",
      "design": "Add content variant records with source_content_hash, variant kind, mechanical/generative provenance, coverage spans, confidence, alignment nodes, and stale/orphaned state. Require coverage>0 for summary-like variants and never auto-repaint stale variants after source drift.",
      "acceptance_criteria": ["Variant write fails for summary-like variants with zero cited coverage.", "Source content hash change marks variant stale without altering original evidence.", "Mechanical OCR/transliteration variants carry confidence/coverage and are not treated as raw evidence.", "Render can show original, variant, or side-by-side with variant provenance disclosure.", "Search can include or exclude variants explicitly."],
      "dependencies": [],
      "parent": "polylogue-4smp",
      "risks": ["False-stale storms from hash/fingerprint drift", "Users may misread generated summaries as original evidence"],
      "confidence": "med",
      "provenance": "Content transform lane: mechanical vs generative provenance, coverage-gap rendering, source_content_hash staleness."
    },
    {
      "proposed_id": "p1-20",
      "change_kind": "new",
      "target_bead": null,
      "title": "SARIF export for agent pathologies",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:interop", "area:audit", "area:pathologies"],
      "description": "Polylogue pathologies and claim-vs-evidence defects should be exportable into developer-native review surfaces rather than trapped in Polylogue-only reports.",
      "design": "Map accepted or candidate pathology findings to SARIF results with rule ids, severity, evidence refs, related locations, and markdown help. Preserve local-first privacy by using stable Polylogue refs and optional redacted snippets rather than absolute private paths unless explicitly requested.",
      "acceptance_criteria": ["Fixture pathologies export to SARIF accepted by a standard validator or GitHub code-scanning fixture.", "Each SARIF result links back to a Polylogue evidence ref and rule id.", "Candidate/unverified findings are marked distinctly from accepted findings.", "No private absolute paths or raw secrets appear in default export.", "Docs describe the mapping from Polylogue pathology kinds to SARIF rules."],
      "dependencies": ["p1-09", "p1-04"],
      "parent": "polylogue-9e5",
      "risks": ["SARIF consumers expect file locations; session evidence needs careful mapping", "Export could imply stronger verification than candidate status warrants"],
      "confidence": "med",
      "provenance": "Standards lane: export mined pathologies as SARIF and make findings portable."
    },
    {
      "proposed_id": "p1-21",
      "change_kind": "new",
      "target_bead": null,
      "title": "MCP resources and prompts for archive refs",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:mcp", "area:interop", "area:context"],
      "description": "Polylogue's MCP surface should expose stable sessions/messages/blocks/evidence packs as resources and recall packs/saved views as prompts, rather than forcing every use through model-controlled tools.",
      "design": "Add MCP resources for session/message/block/query/finding/evidence-pack refs with resource:// URIs, list_changed support where feasible, and read payloads from the same resolver as CLI. Expose recall packs and saved views as MCP prompts that expand through the context scheduler and trust-class gates.",
      "acceptance_criteria": ["MCP lists and reads at least session, message, block, query, finding, and evidence-pack resources from a fixture archive.", "Recall pack and saved view prompts are discoverable and render through the same context compiler as CLI.", "Resources carry staleness/degraded metadata and never bypass recursive-safety inject gates.", "Tool list does not duplicate resource/prompt capabilities except for mutation/query execution verbs.", "MCP schema/output docs regenerated and golden tests prove resource resolver parity with CLI read."],
      "dependencies": ["p1-07", "p1-09", "p1-06"],
      "parent": "polylogue-fs1",
      "risks": ["Capability sprawl if resources/prompts/tools overlap", "Prompt expansion can become an injection path if trust gates are bypassed"],
      "confidence": "med",
      "provenance": "Standards lane: MCP sessions/messages/blocks as resources and recall packs/saved views as prompts."
    },
    {
      "proposed_id": "p1-22",
      "change_kind": "new",
      "target_bead": null,
      "title": "Git notes and PR provenance footers",
      "issue_type": "feature",
      "priority": 3,
      "labels": ["area:interop", "area:git", "area:provenance"],
      "description": "Polylogue can make AI-session provenance visible in ordinary Git workflows by writing outbound citations to commits, PRs, and issues.",
      "design": "Implement opt-in polylogue cite commands that emit git notes under refs/notes/polylogue and GitHub PR/issue provenance footers linking commits to session/query/finding refs. Treat this as an outbound projection; inbound git linkage remains separately owned.",
      "acceptance_criteria": ["polylogue cite commit <sha> writes a git note with session/query/finding refs and can show it via git log --notes=polylogue.", "polylogue cite pr renders a footer body without mutating GitHub unless an explicit apply flag is passed.", "Round-trip tests parse the note/footer back into refs.", "Default output avoids private absolute paths and marks missing evidence refs.", "Docs explain outbound cite versus inbound commit detection."],
      "dependencies": ["p1-09"],
      "parent": "polylogue-l4kf",
      "risks": ["GitHub mutation UX can surprise users", "Public footers may leak private session refs if not redacted"],
      "confidence": "med",
      "provenance": "Standards lane: git notes and PR/issue provenance footer as outbound evidence projection."
    }
  ]
}
```
