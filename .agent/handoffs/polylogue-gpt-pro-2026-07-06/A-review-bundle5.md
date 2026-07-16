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

## [assistant 324]
bash -lc python3 - <<'PY'
from pathlib import Path
import re, textwrap
text=Path('/mnt/data/rnd-bundle-5-of-6.md').read_text(errors='ignore')
part=text.split('=====================================================================\n  PART 2 - PAYLOAD',1)[1]
parts=re.split(r'\n---\n\n## \[([0-9a-f]+)\] ', part)
out=[]
for n in range(1,len(parts),2):
    idx=n//2+1; id=parts[n]; rest=parts[n+1]
    title=rest.split('\n',1)[0]
    body=rest.split('\n',1)[1]
    print('\n============================')
    print(idx,id,title)
    # print first 60 headings and bullets with Schema/Tier/Risk/Bead
    for line in body.splitlines():
        if line.startswith('#') or line.startswith('## ') or line.startswith('### ') or line.startswith('- **') or line.startswith('**Tier') or 'Bead' in line[:80] or line.startswith('Risk') or line.startswith('Verdict'):
            print(line[:250])
    print('--- first 500 chars:', textwrap.shorten(body.replace('\n',' | '), width=900))
PY

## [assistant 448]
I treated file 5 as its own slice. I used the master synthesis only to understand vocabulary and extant bead IDs; the substance below is anchored to Part 2’s sixteen outputs. I also checked the uploaded Polylogue code snapshot where possible. I did not run the daemon, the live archive, or heavy tests, and I could not verify cross-repo claims about Lynchpin without Lynchpin’s source tree.

## Executive verdict

File 5’s unique payload is not one topic; it is a coherent “make Polylogue useful to itself and to agents” slice. It covers: intent units (`phase-segment`, `goal`, `decision`), migration/scale correctness, executable demos, content variants, SDK/public boundary, activation/adoption, cross-machine sync, prompt distillation, missing insight units, MCP collapse, cost correctness, daily workflow verbs, release/schema discipline, config doctrine, and self-observability.

The good news: most of the strongest diagnoses are real. The code already has the primitives the designs want to expose: work events distinct from phases, a unified assertions table with candidate/accepted/superseded lifecycle, split-tier storage discipline, sync/async API surfaces, rich MCP tooling, and cost code whose exact-cost and provider-lane problems are concrete. The file is strongest when it says “wire what exists, expose it through the algebra, and gate derived claims.”

The bad news: several designs are overconfident about naming, tiering, or scope. The recurring correction is that Polylogue needs fewer new nouns and more typed provenance objects. “Goal,” “decision,” “prompt template,” “demo finding,” “variant,” and “saved query” should all converge on the same model: durable assertion/judgment/recipe objects, derived candidates, explicit evidence refs, candidate-by-default agent output, and queryable result/analysis objects. The file often gets the storage regime right, but under-specifies the object model that would make these analyses composable.

## Hard vetting: confirmed

The `phase-segment` correction is right. In the code snapshot, `SessionPhase` is explicitly time-gap segmented and its docstring says the intent `kind` field was removed because phases represent when activity happened, not what kind of activity it was. The intent-classified substrate is `session_work_events`, with labels such as planning, implementation, debugging, review, testing, research, configuration, documentation, refactoring, data_analysis, and session; the index DDL has `session_work_events` rows with `work_event_type`, confidence, duration, file paths, tools, and evidence. Exposing `phase-segment` as a query/projection over work events, not as a new `session_phases.kind`, is the correct construct-validity choice. The proposed confidence gate and “unlabeled” bucket are also right: low-confidence inferred work type must not become a crisp workflow shape. fileciteturn1file4turn1file14

The `goal` and `decision-object` direction is mostly right: mined intent must go through the existing assertion lifecycle. I verified `AssertionKind` already includes `DECISION`, `JUDGMENT`, `RUN_STATE`, `PROMPT_EVAL`, `TRANSFORM_CANDIDATE`, and `PATHOLOGY`, and `AssertionStatus` includes `ACTIVE`, `CANDIDATE`, `ACCEPTED`, `REJECTED`, `DEFERRED`, `SUPERSEDED`, `DELETED`, and `INACTIVE`. The assertion table is schema-free enough for new kinds, with `evidence_refs_json`, `context_policy_json`, `supersedes_json`, author fields, status, visibility, and confidence. The design’s “mined claim never ACTIVE/injectable until judged” rule is exactly the right invariant. fileciteturn2file4

The migration-fuzz and scale-regression line is strategically right even though I would size it as an infrastructure bead, not product work. The code has a durable migration runner with backup-manifest checks, per-tier migrations for source/user, and rebuild rather than in-place migration for derived tiers. That means fuzzing start versions, sparse migration chains, backup gating, and durable-versus-derived policy is not theoretical; it directly protects the split-tier archive. The correction is that fuzz tests must model the real sparse historical chains, not assume contiguous `001..N`.

The `.polydemo` proof-engine design is strong. Its best idea is not “demo runner”; it is that a demo finding should be content-addressed over claim, metric, evidence anchor, sorted refs, and corpus datasheet hash, then recomputed in CI so drift breaks the build. That turns public demos into reproducible findings rather than screenshots. The parser/runner/resolver/refusal/construct-validity tests are the right failure gates. fileciteturn1file2turn1file17

The content-variant design is directionally correct: variants must not overwrite original evidence, must carry alignment edges, coverage status, source-content hashes, staleness state, and explicit “dark matter” for unaccounted source children. The design is also correct that generated variants must never be auto-repainted in place when source evidence drifts; recovery should be a new variant that supersedes the old one. fileciteturn1file13

The SDK diagnosis is partly confirmed from Polylogue’s side. The code snapshot has `polylogue.api.sync`, but no stable `polylogue.models` package. Public models are scattered across archive/session/message/insight modules. A curated SDK boundary would be valuable. I could not verify the Lynchpin raw `SELECT ... FROM conversations` claim because Lynchpin source was not in the uploaded Polylogue package; treat that specific cross-repo diagnosis as plausible but unconfirmed until the Lynchpin tree is read. fileciteturn2file6

The cross-machine sync tiering is broadly right: sync durable source/user data, never derived index/embeddings/ops files. I verified `deterministic_raw_session_id` includes `source_path`, so the file’s warning about machine-local raw IDs is real. Same logical capture can become two physical raw sessions across machines if paths differ. That must be surfaced as physical-union behavior, not hidden as “deduped sync.” fileciteturn1file6

The MCP collapse target is right, but the count is approximate. The uploaded code snapshot’s static `EXPECTED_TOOL_NAMES` count is 96, not 95; dynamic insight registration plausibly brings the live server into the ~110–130 range. The conclusion still stands: the MCP surface is too wide and should collapse around algebraic verbs like `query`, `get`, `explain`, `context`, `correlate`, `coordinate`, `assert`, `retract`, and `maintenance`, with resources/prompts used for non-tool affordances. The migration must be golden-tested, not big-bang deleted. fileciteturn1file3turn1file9

The cost-correctness issues are real and urgent. I verified `_session_level_estimate` returns `None`; subscription pricing lacks the newer Opus keys named in the file; `cost_compute.py` hardcodes a Pro-tier `$20 / 21,700,000 credits` conversion; `cost_enrichment.py` can replace a stored exact estimate with a re-derived priced estimate; day/week summaries aggregate physical profiles while logical counts are tracked separately. The file’s provider-lane warnings are exactly the kind of bug that invalidates public findings if left unresolved. fileciteturn2file12turn2file8

The daily workflow section is valuable because it shifts from substrate nouns to operator journeys: resume work, postmortem failures, bundle external context, cite motivating sessions, compare attempts, recall exact errors, save/run queries, and check freshness. These are not “nice UX.” They are the practical surfaces that make the archive worth using every day. fileciteturn2file18

The observability/SLO ideas are good: Polylogue needs to distinguish idle-healthy from stalled, keep ingest/convergence success rates, detect poison rows stuck under retry/`false_means_pending`, and retain ops-history for regression baselining. This fits the disposable `ops.db` tier and should not pollute user evidence. fileciteturn2file9

## Hard vetting: refuted or corrected

`Goal` should not introduce `author_kind="operator"` casually if the recursive-safety rule elsewhere requires `author_kind='user'` for injection. The actual invariant should be trust-class based: agent/miner/authored-by-model rows are candidate and `inject:false`; human/operator-promoted rows may become active/injectable. If the enum uses `operator`, then the injection gate must know `operator` is the trusted human class. If it uses `user`, the goal spec should say `user`, not `operator`. This mismatch is small but load-bearing.

`GOAL` as a new `AssertionKind` is reasonable, but I would first ask whether it should be `kind=RUN_STATE` or `kind=JUDGMENT` with `key=goal`. New kinds are cheap at the database layer, but not free at the product layer: generated schemas, audit surfaces, OpenAPI, CLI schemas, MCP contracts, and UI filters all need updates. The file knows this trap; the correction is to require a “new kind justification” checklist before adding `GOAL`, `PROMPT_TEMPLATE`, or any new assertion vocabulary.

`decision-object` should not store supersession only as `value.superseded_by` long-term. That is acceptable as phase one, but read performance, auditing, and graph queries will want a normalized edge view or runtime edge extractor with cycle checks. Since SQLite persistent views cannot cross attached tiers, the right boundary is a runtime query method over user assertions, not a persistent cross-tier view.

`.polydemo` finding IDs should not hash mutable human phrasing as the only identity. Hashing `claim` catches drift, which is good for CI, but it also makes harmless wording changes look like new findings. I would split this into two IDs: a stable `finding_semantic_id` over metric, corpus datasheet, construct, evidence refs, and anchor; and a `finding_render_id` or artifact hash over the human claim text. CI can decide which drift is fatal.

Content variants should not be only a user-tier table if variant bodies can be large. Durable metadata belongs in `user.db`, but generated text bodies and alignment payloads may need blob storage or a body table with content hashes. Otherwise translations/summaries of long transcripts will bloat user.db and complicate backup/merge semantics. The design should explicitly split metadata from variant payload storage.

Cross-machine sync should not keep both `user_settings` and “settings as assertions.” The file’s sync design includes user settings merge, while the broader config direction says the `user_settings` table is dead and settings should unify into assertions. Pick one. My recommendation: new settings use assertions; legacy `user_settings` gets a copy-forward/deprecation path and is excluded from new sync semantics except for migration compatibility.

The prompt/meta-workflow distillery is high-risk and should be later than the file’s implementation detail implies. The design is honest about candidate-only birth, A/B evaluator floors, and recursive contamination. But observational prompt induction is a causal minefield; treat it as hypothesis mining, not as “distillation” until controlled treatment data exists. Candidate templates must be excluded from their own evidence loops. fileciteturn2file2turn2file15

The missing insight units need stricter priority. `project` is likely useful soon because it helps scope everything else. `topic-cluster` and `cross-origin-thread` are riskier: embeddings, hub-merge, candidate explosion, and label stability will make them expensive. The file already names hard-signal floors and blocking; the correction is to ship them as insight-registry candidates before DSL units, unless a live query demands them. fileciteturn1file12turn1file3

MCP collapse should not be treated as purely a tool-count cleanup. It is an agent-attention and compatibility migration. The current MCP standard separates tools, resources, and prompts; resources are for data/context, prompts are reusable templates/workflows, and tools are executable functions. That supports moving many current read affordances away from tools, but only with compatibility shims and usage telemetry. citeturn837479search0turn837479search1

The release/versioning section’s “decision, not build” stance is only partly true. Local files confirm version `0.1.0` and docs that say PyPI/Homebrew/container/browser-store are not user install paths yet. I did not verify “zero git tags” against a live remote. Even if distribution machinery exists, cold-machine install verification, docs reconciliation, generated metadata, and release artifact provenance are real work.

## External context that should change the designs

Polylogue’s analysis objects should borrow from W3C PROV: entities, activities, agents, usage, generation, derivation, and association. That maps almost perfectly to sessions, queries, query runs, agents, evidence packs, generated reports, content variants, and prompt templates. This is stronger than inventing isolated “demo finding,” “analysis run,” “variant,” and “prompt eval” graphs separately. citeturn643610search0turn643610search4

RO-Crate is relevant for portable evidence bundles: it packages research objects with metadata about files, workflows, people/organizations, provenance, and context entities. A Polylogue evidence pack or `.polydemo` artifact should be able to export into an RO-Crate-like shape later, even if the first implementation is internal JSON/Markdown. citeturn643610search1turn643610search17

The Web Annotation Data Model supports motivations such as classifying, commenting, describing, editing, highlighting, and questioning. That argues for a structured `motivation`/`kind`/`purpose` axis on external annotations and assertions rather than stuffing everything into prose notes. citeturn643610search2turn643610search14

SQLite’s own documentation reinforces the single-writer rule: multiple processes can read, but only one writer can commit at a time; WAL also uses a shared-memory wal-index, so readers and writers need same-machine assumptions. This matters for cross-machine sync: do not rsync live WAL-era SQLite files and do not allow peer merge paths that bypass the daemon/single-writer boundary. citeturn643610search7turn643610search3

Litestream and LiteFS are useful comparisons but not direct substitutes. Litestream continuously copies WAL pages for async disaster recovery; LiteFS replicates SQLite through a distributed filesystem model. Polylogue’s desired sync is not block/WAL replication; it is semantic durable-tier merge with source idempotency and user assertion conflict rules. citeturn779583search1turn779583search0

Hypothesis-style property and stateful testing is the right tool for migration fuzz, sync merge commutativity, and state-machine candidate lifecycles. The important point is to model operation sequences, not only generate values. citeturn837479search7turn837479search3

OpenTelemetry’s semantic conventions are a useful prior for naming observability measurements, but Polylogue should not blindly adopt unstable conventions. Use the pattern—standardized names, units, attributes, and status semantics—not necessarily every metric name. citeturn837479search4turn837479search8

Provider pricing and token semantics must be treated as live external facts, not as static internal doctrine. Anthropic’s official pricing docs distinguish cache writes and cache reads; OpenAI’s pricing exposes input, cached input, and output; Gemini’s docs say output price includes thinking tokens and context caching has separate pricing/storage. That supports the file’s disjoint-lane cost model, but also means rates and inclusions need a live-source update path and tests. citeturn754182search0turn754182search1turn754182search2

## Design synthesis

The missing unifier is an **analysis/provenance object model**. File 5 independently proposes `.polydemo`, prompt evals, adoption experiments, query recall, saved views, content variants, sync manifests, and daily workflow commands. These should not become separate silos.

The shared model should be:

`query_definition → query_run → result_relation/cohort → evidence_pack → annotation_batch/assertions → analysis_run → report/demo artifact`

A query definition is the reusable dynamic object. A query run is a specific execution against an archive epoch. A result relation has grain, identity policy, exactness, and member refs. A cohort is a named dynamic or snapshot relation. An annotation batch imports external/agent judgments as assertions under a schema. An analysis run ties queries, cohorts, annotation batches, evidence packs, reports, and artifacts into a provenance DAG. This single model makes `.polydemo`, prompt distillery, activation A/B tests, Fable-style delegation analysis, and daily saved queries all composable.

The second unifier is **candidate-by-default for derived human-meaning claims**. Goals, decisions, prompt templates, topic clusters, cross-origin threads, and mined findings should all begin as candidate/private/non-injectable unless they come from a structured human/operator command or are explicitly judged. This is not just safety. It preserves construct validity.

The third unifier is **tier discipline**. Source and user data are durable and mergeable only through explicit migration/single-writer paths. Index/embeddings are rebuilt or blue-green swapped. Ops is disposable and ideal for query runs, context compile runs, SLO samples, adoption telemetry, and daemon state. The file mostly respects this, but some designs blur user.db into a general blob/artifact store and blur sync into settings/config.

## Prioritized take

Tier 0: fix cost truth and assertion safety first. The cost issues are concrete and public-finding-breaking. The assertion safety chokepoint is a recursive-contamination risk now that Polylogue captures its own R&D.

Tier 1: land `.polydemo`/finding reproducibility and adoption/install/doctor. These turn the system into an externally inspectable proof engine and make agent use measurable.

Tier 2: thin the API/MCP/SDK boundary. MCP collapse, SDK boundary, and daily CLI workflows should route through the same query/read/action contract. Do not build more surfaces that duplicate substrate logic.

Tier 3: add query/result/cohort/analysis objects. This is the general substrate behind saved queries, demos, annotation batches, prompt evals, and evidence reports.

Tier 4: content variants and project unit. These are useful, but must respect payload storage and provenance boundaries.

Tier 5: cross-machine sync, topic clustering, cross-origin threads, and prompt distillery. These are promising but higher-risk, and should be gated by evidence, small pilots, and clear false-merge/causal-validity floors.

```json
{
  "file": 5,
  "proposed_beads": [
    {
      "proposed_id": "p5-01",
      "change_kind": "update",
      "target_bead": "polylogue-37t.12",
      "title": "Enforce assertion trust chokepoint for agent writes",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:user-db", "area:recursive-safety", "area:context", "kind:assertions"],
      "description": "Agent-authored or mined assertions can only be safe if every write path coerces non-human authors to candidate/private/inject:false. File 5 repeatedly depends on this invariant for goals, decisions, prompt templates, annotation recipes, and context injection.",
      "design": "Implement a single coerce_agent_authored path inside upsert_assertion or its narrowest shared helper. Preserve terminal judged rows; never downgrade ACCEPTED/REJECTED/SUPERSEDED. Treat trusted human/operator authors explicitly and reject ambiguous author_kind values. Add regression fixtures for blackboard, transform candidates, prompt-template candidates, goal candidates, and decision candidates.",
      "acceptance_criteria": [
        "All non-human assertion writes are stored with status CANDIDATE or equivalent non-active state, visibility private, and context_policy.inject false.",
        "Existing terminal-judged rows are not resurrected or downgraded by re-materialization.",
        "Context compile / recall / successor-context tests prove candidate agent-authored rows are never injected.",
        "A single test fixture covers blackboard, transform/pathology, goal, decision, prompt-template, and annotation-import writers."
      ],
      "dependencies": ["polylogue-37t.11"],
      "parent": "polylogue-37t",
      "risks": ["Breaking legitimate automation that currently writes ACTIVE rows", "Ambiguous user/operator author_kind policy", "Hidden write path bypasses the chokepoint"],
      "confidence": "high",
      "provenance": "File 5 goal/decision/prompt-distillery specs all rely on candidate-by-default mined assertions; code snapshot confirms the unified assertions table and status lifecycle exist."
    },
    {
      "proposed_id": "p5-02",
      "change_kind": "new",
      "target_bead": null,
      "title": "Expose phase-segment query unit over work-events",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:query-dsl", "area:insights", "construct:workflow"],
      "description": "Session phases are time-gap segments and must not become intent-classified. The intent substrate already exists as session_work_events; Polylogue needs a queryable phase-segment projection over that table.",
      "design": "Register a phase-segment DSL unit source backed by session_work_events. Add a deterministic WorkEventHeuristicLabel to macro phase_kind mapping at read time, not as a stored session_phases column. Gate weak/low-confidence events to unlabeled. Add aggregation outputs that expose unlabeled_fraction.",
      "acceptance_criteria": [
        "phase-segment appears in query schema/explain output and supports phase_kind, work_event_type, confidence, duration_ms, session_id, started_at_ms, file_paths, and tools_used fields.",
        "A guard test proves session_phases has no intent kind column.",
        "All WorkEventHeuristicLabel values map to exactly one macro phase or unlabeled.",
        "A weak or confidence <0.5 work event projects to unlabeled.",
        "workflow_shape_distribution can be reproduced from the phase-segment pipeline with a regression snapshot."
      ],
      "dependencies": ["polylogue-fnm.2", "polylogue-fnm.11"],
      "parent": "polylogue-fnm",
      "risks": ["Reintroducing the phase/work-event construct error", "Workflow-shape drift between existing insight code and DSL pipeline", "Adding fields shifts CLI positional behavior"],
      "confidence": "high",
      "provenance": "File 5 Unit A; code confirms SessionPhase is time-gap only and session_work_events already carries intent-labeled rows."
    },
    {
      "proposed_id": "p5-03",
      "change_kind": "new",
      "target_bead": null,
      "title": "Construct-gated goal candidates and abandonment join",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:insights", "area:user-db", "area:query-dsl", "construct:goal"],
      "description": "Polylogue can mine likely session objectives, but unjudged goals must not become active memory. Abandoned-session surfaces should show the unmet objective only when its provenance/status is explicit.",
      "design": "Mine goal candidates from explicit operator goal commands, run-state headings, and first human-authored messages in descending trust order. Use instruction-dump rejection and candidate-by-default assertion rows for inferred goals. Add goal_source, goal_confidence, and goal_status to the enrichment payload. Pair abandonment results with goal text/status at read time.",
      "acceptance_criteria": [
        "Explicit structured goal commands may become trusted goals; inferred goals are candidate/private/inject:false.",
        "Runtime protocol/user-role dumps and pasted instruction blobs do not become active goals.",
        "find_abandoned_sessions includes goal_text, goal_source, and goal_status where available.",
        "Re-materialization never downgrades a promoted goal assertion.",
        "Generated schemas and user_audit include any new assertion kind or key used."
      ],
      "dependencies": ["p5-01"],
      "parent": "polylogue-37t",
      "risks": ["Pasted mandates masquerade as user goals", "New AssertionKind overhead if key-based assertion would suffice", "Goal confidence treated as truth"],
      "confidence": "med",
      "provenance": "File 5 Unit B; verified existing enrichment/assertion substrate but corrected the author_kind/trust-class ambiguity."
    },
    {
      "proposed_id": "p5-04",
      "change_kind": "new",
      "target_bead": null,
      "title": "Unify mined decisions with cycle-safe supersession",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:user-db", "area:insights", "construct:decision"],
      "description": "Decision candidates are already mined, but decisions need one queryable kind, explicit provenance, candidate-by-default safety, and acyclic supersession.",
      "design": "Mirror mined decision candidates into AssertionKind.DECISION with author_kind/source metadata and status CANDIDATE. Store supersession as value.superseded_by in phase one, with a runtime chain reader and cycle/quarantine checks. Keep user-authored and mined decisions separable by author/status.",
      "acceptance_criteria": [
        "Mined decisions are queryable under decision kind but never ACTIVE/injectable before human judgment.",
        "User-authored decisions and mined candidates are distinguishable by author_kind, author_ref, and status.",
        "Supersession attempts that create cycles are refused or quarantined.",
        "A bounded chain reader returns active head and lineage without unbounded recursion.",
        "Recovery-digest regression proves prose-mined decisions do not appear as facts."
      ],
      "dependencies": ["p5-01"],
      "parent": "polylogue-37t",
      "risks": ["Mined candidate looks like operator decision", "JSON supersession edges become hard to query", "Cycle checks missed in alternate promotion path"],
      "confidence": "high",
      "provenance": "File 5 Unit C; code confirms DECISION and SUPERSEDED already exist, but runtime edge handling is missing."
    },
    {
      "proposed_id": "p5-05",
      "change_kind": "new",
      "target_bead": null,
      "title": "Migration and scale regression fuzz harness",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:storage", "area:migrations", "area:testing", "area:scale"],
      "description": "Durable tier migrations and large-archive behavior are load-bearing; sparse migration chains, backup manifests, source/user regimes, and scale-only pathologies need executable tests.",
      "design": "Build a property/stateful harness for real historical source/user migration start versions, backup-manifest gates, PRAGMA user_version advancement, derived-tier no-upgrade policy, and scale fixtures. Add targeted scale-regression fixtures for high-cardinality sessions, large WAL-risk batches, and lineage/debt edge cases.",
      "acceptance_criteria": [
        "Fuzz harness maps start_version only to real historical DDL states and accepts sparse migration numbering.",
        "Durable migrations fail without verified backup manifest and advance user_version exactly once.",
        "Derived index/embeddings changes are rejected if implemented as upgrade helpers.",
        "Scale-regression lane includes at least one fixture that would catch one-shot large transaction or never-draining debt behavior.",
        "Tests use frozen_clock and temp archive roots."
      ],
      "dependencies": ["polylogue-1xc"],
      "parent": "polylogue-1xc",
      "risks": ["Synthetic fixtures miss live 38GB archive behavior", "Fuzz harness assumes false contiguous history", "Slow tests become skipped"],
      "confidence": "high",
      "provenance": "File 5 migration-fuzz spec; code confirms durable migration runner and split-tier regimes."
    },
    {
      "proposed_id": "p5-06",
      "change_kind": "update",
      "target_bead": "polylogue-212",
      "title": "Executable polydemo proof engine",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:demo", "area:construct-validity", "area:citations", "area:ci"],
      "description": "Public demos should be reproducible findings, not screenshots. A .polydemo runner can seed a deterministic corpus, run product CLI steps, resolve evidence refs, enforce construct-validity/refusal gates, and fail CI on finding drift.",
      "design": "Implement .polydemo parser, corpus datasheet hashing, finding ID computation, evidence-ref resolver gate, construct-validity pre-render gate, refusal manifest, runner artifacts, and devtools verify integration. Split stable metric/evidence identity from rendered human-claim artifact hash so wording edits do not masquerade as metric drift.",
      "acceptance_criteria": [
        "DemoScript parses frontmatter, corpus seed/datasheet hash, ordered steps, findings, refusals, and budgets.",
        "demo run executes only product polylogue argv steps and writes report.json, report.md, PUBLIC_REPRODUCTION.md, and COLD_READER_GATE.md.",
        "Dangling or non-round-tripping evidence refs fail before rendering.",
        "Findings anchored to unsupported prose/text constructs fail the construct-validity gate.",
        "CI recomputes finding identities and fails on evidence/metric/corpus drift.",
        "Human claim text drift is reported separately from metric/evidence identity drift."
      ],
      "dependencies": ["polylogue-sru.1"],
      "parent": "polylogue-212",
      "risks": ["Demo runner becomes a Python monolith instead of product-step executor", "Claim text hashing creates noisy ID churn", "Seed corpus drift breaks demos without clear explanation"],
      "confidence": "high",
      "provenance": "File 5 demo/proof engine spec; external RO-Crate/PROV review supports provenance-first artifacts."
    },
    {
      "proposed_id": "p5-07",
      "change_kind": "new",
      "target_bead": null,
      "title": "Content variants with alignment and payload storage",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:variants", "area:user-db", "area:projection", "area:citations"],
      "description": "Translations, summaries, simplifications, and alternate renderings must be attached to original evidence without replacing it, with coverage, alignment, staleness, and dark-matter accounting.",
      "design": "Add durable variant metadata and alignment edges, but store large generated bodies by content hash/blob rather than bloating user.db. Add source_content_hash, variant_status, coverage classification, staleness convergence stage, and render policies for original/variant/side-by-side.",
      "acceptance_criteria": [
        "Variant metadata records target_ref, transform type, provenance, author/model/prompt, source_content_hash, status, and supersession.",
        "Large variant bodies are stored by content hash or blob reference, not inline-only user.db text.",
        "Coverage validator classifies complete, partial, sparse, and dark_matter gaps.",
        "Re-ingest source drift marks variants stale/orphaned but never auto-regenerates text.",
        "Projection/render surfaces disclose when variant text substitutes for original evidence."
      ],
      "dependencies": ["polylogue-4smp", "p5-01"],
      "parent": "polylogue-4smp",
      "risks": ["False stale storms from fingerprint changes", "Generated variant text confused with raw evidence", "User.db backup size grows unbounded"],
      "confidence": "med",
      "provenance": "File 5 content-variant spec; corrected tiering by separating metadata from large payload storage."
    },
    {
      "proposed_id": "p5-08",
      "change_kind": "new",
      "target_bead": null,
      "title": "Curated polylogue.sdk and public models boundary",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:sdk", "area:api", "area:contracts", "area:layering"],
      "description": "External consumers should not import archive internals or reimplement models. The code has sync API pieces but no stable polylogue.models package or schema pin contract.",
      "design": "Create polylogue.sdk with SyncClient/AsyncClient, typed errors, schema pin-and-warn, lazy Query object, and curated frozen public models. Re-export/adapt canonical domain models rather than redefining them. Add layering lint preventing external/surface code from importing storage/archive internals except through sdk/api/contracts.",
      "acceptance_criteria": [
        "polylogue.sdk exposes a documented curated __all__ and version independent enough for compatibility warnings.",
        "polylogue.models exports stable frozen Session, Message, Action, Cost, WorkEvent, and readiness models or explicit equivalents.",
        "SDK checks index/user/source schema compatibility and emits actionable warnings.",
        "Layering test fails if non-substrate surfaces or external bridges import storage/archive internals directly.",
        "A smoke consumer migrates off raw sqlite access to the SDK."
      ],
      "dependencies": ["polylogue-t46"],
      "parent": "polylogue-t46",
      "risks": ["Freezing provider/origin vocabulary mid-migration", "SDK becomes a second facade instead of a curated facade", "Cross-repo migration reveals expected output deltas"],
      "confidence": "med",
      "provenance": "File 5 SDK spec; Polylogue snapshot confirms no polylogue.models and scattered public models, but Lynchpin raw-sqlite claim remains unverified without Lynchpin source."
    },
    {
      "proposed_id": "p5-09",
      "change_kind": "new",
      "target_bead": null,
      "title": "Activation install doctor and adoption telemetry",
      "issue_type": "epic",
      "priority": 1,
      "labels": ["area:adoption", "area:hooks", "area:ops", "area:context"],
      "description": "The substrate has no value if agents do not discover or use it. File 5 emphasizes one-command harness wiring, precompact/sessionstart context, liveness diagnosis, and measuring whether injected context changes behavior.",
      "design": "Implement polylogue install for idempotent hook wiring with backup/diff, polylogue doctor for why-zero-usage diagnosis, hook liveness heartbeats in ops.db, relevance-controlled adoption rate, and A/B activation experiments over context preambles. Keep agent-authored outcomes out of trusted memory until judged.",
      "acceptance_criteria": [
        "polylogue install safely adds/updates supported Claude Code/Codex hooks without clobbering foreign hooks.",
        "doctor distinguishes not-installed, hook-not-firing, wrong archive root, daemon down, irrelevant repo, and no-use-despite-available.",
        "ops.db records hook liveness and adoption telemetry with bounded retention.",
        "Adoption metrics include a relevance denominator so repos where Polylogue is irrelevant do not count as failures.",
        "A/B preamble variants can be measured without injecting candidate agent-authored claims as trusted memory."
      ],
      "dependencies": ["polylogue-37t.11", "p5-01"],
      "parent": "polylogue-3tl",
      "risks": ["False adoption alarms train operator to ignore doctor", "Hook installer damages user harness config", "A/B telemetry contaminates context scheduler attention"],
      "confidence": "high",
      "provenance": "File 5 activation/adoption specs and daily-workflow journeys."
    },
    {
      "proposed_id": "p5-10",
      "change_kind": "new",
      "target_bead": null,
      "title": "Durable-tier cross-machine sync pilot",
      "issue_type": "epic",
      "priority": 3,
      "labels": ["area:sync", "area:source-db", "area:user-db", "area:daemon"],
      "description": "Cross-machine sync should merge durable evidence and user overlays without transmitting derived tiers or violating single-writer SQLite rules. Raw IDs currently include machine-local source_path, so duplicates must be handled honestly.",
      "design": "Build a daemon-routed pilot: manifest and bucket digests over source/user only, source INSERT OR IGNORE by content hash, user assertions LWW/natural-key merge with supersession respect, version skew abort, derived convergence after merge. Explicitly exclude index/embeddings/ops and decide user_settings deprecation before syncing settings.",
      "acceptance_criteria": [
        "Manifest exposes no derived-tier data and equal archives produce equal top-level digest.",
        "One differing durable row flips exactly one bucket digest and pulls only that row/blob.",
        "Source merge is idempotent and commutative for identical content hashes.",
        "User assertion merge converges across three peers regardless of order with deterministic tie-break.",
        "Durable user_version/source_version skew aborts before writing.",
        "Sync routes through daemon/single-writer path; no live SQLite file copying.",
        "Docs state that source_path-derived raw_id can create physical duplicates resolved only downstream by lineage."
      ],
      "dependencies": ["polylogue-4ts"],
      "parent": null,
      "risks": ["Single-writer violation", "Machine-local raw IDs inflate physical rows", "Settings merge conflicts with settings-as-assertions plan"],
      "confidence": "med",
      "provenance": "File 5 cross-machine sync spec; code confirms raw_id includes source_path and external SQLite docs argue against live-file sync."
    },
    {
      "proposed_id": "p5-11",
      "change_kind": "new",
      "target_bead": null,
      "title": "Prompt-template candidates and eval lifecycle",
      "issue_type": "feature",
      "priority": 3,
      "labels": ["area:prompting", "area:eval", "area:user-db", "area:recursive-safety"],
      "description": "Polylogue can mine reusable prompt patterns, but any claim that a prompt would improve outcomes is causal and high-confound. Templates must be hypotheses with evidence and treatment tracking.",
      "design": "Add PROMPT_TEMPLATE candidate assertions or a justified assertion key, plus PROMPT_EVAL producer rows. Mine templates deterministically from evidence-rich cells, require evidence refs, mark candidate/private/inject:false, exclude template-influenced sessions from the template's own induction set, and evaluate with SUPPORTED/REFUTED/INSUFFICIENT_EVIDENCE.",
      "acceptance_criteria": [
        "No prompt template candidate can be created without at least one resolving evidence ref.",
        "Candidates are private and non-injectable until explicit judgment.",
        "Evaluation below min_sample returns INSUFFICIENT_EVIDENCE, never a win.",
        "Sessions influenced by a template are excluded from that template's future induction evidence.",
        "Public/exported template text redacts local paths and provider-sensitive details.",
        "user_audit and generated schemas cover any new kind/ref."
      ],
      "dependencies": ["p5-01"],
      "parent": "polylogue-37t",
      "risks": ["Observational causal overclaim", "Recursive prompt laundering", "Private path/provider leakage"],
      "confidence": "med",
      "provenance": "File 5 prompt/meta-workflow distillery; analysis corrected priority and causal framing."
    },
    {
      "proposed_id": "p5-12",
      "change_kind": "new",
      "target_bead": null,
      "title": "Durable project unit and membership invalidation",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:insights", "area:user-db", "area:projects", "area:query-dsl"],
      "description": "Project identity is the most immediately useful missing unit for scoping recall, adoption, config, and daily workflows. It is the only missing insight unit in this file that clearly deserves durable user-tier rules.",
      "design": "Add a durable project unit with user-authored or operator-confirmed rules; derive membership from repo/cwd/gizmo/source hints in index. Editing project rules invalidates derived membership caches via converger. Expose project filters in query and context compile surfaces only after confidence/freshness is available.",
      "acceptance_criteria": [
        "Project rules live in durable user tier with provenance and author metadata.",
        "Derived project membership is rebuildable and invalidated when rules change.",
        "cwd/path matching is path-segment aligned, not substring based.",
        "Provider vocabulary collapse is handled without non-injective project IDs.",
        "Query/filter surfaces expose project with confidence/freshness metadata."
      ],
      "dependencies": [],
      "parent": null,
      "risks": ["Substring cwd false positives", "Provider/gizmo extraction prerequisite missing", "Durable rules and derived membership drift"],
      "confidence": "med",
      "provenance": "File 5 missing insight units; project is the durable-tier unit with broad cross-cutting utility."
    },
    {
      "proposed_id": "p5-13",
      "change_kind": "new",
      "target_bead": null,
      "title": "Topic-cluster candidate insight pilot",
      "issue_type": "feature",
      "priority": 3,
      "labels": ["area:insights", "area:embeddings", "construct:candidate"],
      "description": "Topic clusters can help browsing and analysis, but unsupervised labels are unstable and easy to over-trust. Ship as candidate insight first, not DSL truth.",
      "design": "Build topic clustering as rebuildable index insight with deterministic content-addressed IDs, embed-coverage gate, fallback clustering when HDBSCAN unavailable, and candidate labels/provenance. Do not expose as authoritative query unit until stability and usefulness are measured.",
      "acceptance_criteria": [
        "Cluster IDs are deterministic across rebuilds for unchanged inputs within documented tolerance.",
        "Embed coverage below threshold marks output unavailable or degraded, not silently low quality.",
        "Cluster labels are candidate/provenance-tagged and not injected as facts.",
        "Regression tests cover timeline stability and rebuild determinism.",
        "Insight registry exposes degraded state and coverage."
      ],
      "dependencies": ["polylogue-mhx.3"],
      "parent": null,
      "risks": ["Unstable labels confuse users", "Embedding drift rekeys clusters", "Goodharting topic labels as ground truth"],
      "confidence": "low",
      "provenance": "File 5 topic-cluster unit; analysis recommends insight-registry candidate before DSL exposure."
    },
    {
      "proposed_id": "p5-14",
      "change_kind": "new",
      "target_bead": null,
      "title": "Cross-origin thread candidate insight",
      "issue_type": "feature",
      "priority": 3,
      "labels": ["area:lineage", "area:insights", "area:cross-provider", "construct:candidate"],
      "description": "Logical conversations can span providers without lineage identity, but false merges are expensive and embarrassing. Cross-origin threads need hard-signal floors and candidate status.",
      "design": "Materialize candidate cross_origin_thread_edges and cross_origin_threads in index after project/topic/embedding prerequisites. Block candidate pairs by repo/time/entity, subtract existing lineage/session_links, require differing origins and hard_signal, use hub/component caps, and expose as candidate insight before general DSL unit.",
      "acceptance_criteria": [
        "No cross_origin_thread_edges row coincides with a resolved lineage/session_link pair.",
        "Every edge has differing origins and a hard_signal justification.",
        "Temporal-only pairs do not merge.",
        "Hub guard fixture prevents giant transitive over-merge.",
        "Embeddings-disabled mode degrades explicitly to reduced-confidence entity/temporal candidates.",
        "Candidate generation uses blocking and never full cross-join over sessions."
      ],
      "dependencies": ["p5-12", "p5-13", "polylogue-4ts"],
      "parent": null,
      "risks": ["False merge across unrelated sessions", "O(n^2) candidate blowup", "Embedding availability silently changes semantics"],
      "confidence": "med",
      "provenance": "File 5 cross-origin-thread unit; corrected to candidate insight first because false-merge risk dominates."
    },
    {
      "proposed_id": "p5-15",
      "change_kind": "update",
      "target_bead": "polylogue-t46",
      "title": "Stage MCP collapse behind algebraic verbs",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:mcp", "area:contracts", "area:query-dsl", "area:agent-ux"],
      "description": "The MCP tool surface is too large for agent attention and contract maintenance. Collapse to algebraic verbs, but with telemetry, aliases, and equivalence goldens so capability is not silently lost.",
      "design": "Introduce query/get/explain/context/correlate/coordinate/assert/retract/maintenance verbs and move read-only data to resources/prompts where appropriate. Keep old tools as deprecated aliases until equivalence tests and usage telemetry prove safe deletion. Tighten delete_session to admin unless operator decides otherwise.",
      "acceptance_criteria": [
        "New verb set covers all existing read/write/admin capabilities through typed DTOs or documented resources/prompts.",
        "Per-old-tool equivalence golden maps representative calls to new verb invocations.",
        "EXPECTED_TOOL_NAMES and TOOL_CONTRACT updated intentionally; discovery tests pass.",
        "Deprecated aliases emit warnings and telemetry before removal.",
        "Role gating is preserved or tightened: read verbs cannot mutate, write verbs cannot delete durable archive content unless explicitly allowed."
      ],
      "dependencies": ["polylogue-fnm.11", "polylogue-t46"],
      "parent": "polylogue-t46",
      "risks": ["Silent capability loss", "Agents trained on old tool names break", "Resources/prompts underused, leaving too much as tools"],
      "confidence": "high",
      "provenance": "File 5 MCP-collapse spec; code confirms large static tool allowlist and role-gated registration."
    },
    {
      "proposed_id": "p5-16",
      "change_kind": "update",
      "target_bead": "polylogue-f2qv",
      "title": "Fix provider cost truth before public findings",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:cost", "area:lineage", "area:honesty", "construct:measure"],
      "description": "Provider/session cost numbers currently risk being wrong through dead exact-cost reads, missing model credit rates, hardcoded subscription tier, exact-to-catalog downgrade, physical lineage double-counting, and disjoint-lane mistakes.",
      "design": "Implement a precedence lattice provider_reported > catalog_priced > heuristic; wire session-reported costs; add live-source pricing table update path; parameterize subscription tier via settings/assertions; roll up costs at logical-session grain; union parallel intervals; assert disjoint token lanes at parse. Numbers without coverage must refuse or mark unknown.",
      "acceptance_criteria": [
        "_session_level_estimate or replacement reads provider-reported session costs when present.",
        "Stored provider_reported/exact estimates are never downgraded by re-enrichment.",
        "MODEL_CREDIT_RATES coverage test includes curated live Anthropic model keys or fails with actionable update instructions.",
        "Subscription dollar conversion is tier-configurable, not hardcoded to Pro.",
        "Day/week/provider rollups use logical-session grain for costs/duration/messages unless explicitly physical.",
        "Codex/OpenAI/Anthropic/Gemini lane tests prove cached/input/output/reasoning lanes are disjoint according to current provider docs.",
        "Unknown model/rate yields unknown/degraded status, not silent zero."
      ],
      "dependencies": ["polylogue-4ts"],
      "parent": "polylogue-f2qv",
      "risks": ["Provider pricing changes after merge", "Logical representative selection changes totals unexpectedly", "Backward consumers expect zero for unknown cost"],
      "confidence": "high",
      "provenance": "File 5 cost-correctness lane; code verification confirmed the main pathologies."
    },
    {
      "proposed_id": "p5-17",
      "change_kind": "new",
      "target_bead": null,
      "title": "Daily workflow verbs over existing substrate",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:cli", "area:ux", "area:query-dsl", "area:daily-use"],
      "description": "File 5 identifies high-frequency operator journeys that are currently possible only by composing multiple MCP/CLI calls manually. These should become thin CLI verbs over existing read/query/assertion surfaces.",
      "design": "Add a small journey layer: polylogue resume, analyze --pathology stuck-tests, read --view bundle --budget, polylogue cite, analyze compare, find --error, read --view postmortem, find ... then save <name>, polylogue run <name>, and polylogue fresh. Implement as thin adapters over query/read/projection/assertion/freshness contracts; do not create siloed engines.",
      "acceptance_criteria": [
        "resume prints cwd-scoped most-recent unfinished logical-session brief with last authored user turn, last assistant action, unresolved errors, and next action.",
        "fresh returns cwd-scoped green/amber/red verdict over ingest age, convergence debt, and embedding lag.",
        "cite emits a stable resolver-backed citation token and one-line quote.",
        "save/run creates and executes durable saved query/view objects from the CLI.",
        "Every journey command has JSON output and a golden fixture using the same substrate contract as MCP/API."
      ],
      "dependencies": ["polylogue-t46", "polylogue-fnm.12"],
      "parent": "polylogue-fnm",
      "risks": ["Journey verbs become bespoke silos", "Citations use unstable message-position refs", "Freshness verdict hides degraded archive state"],
      "confidence": "high",
      "provenance": "File 5 operator daily-workflow journeys."
    },
    {
      "proposed_id": "p5-18",
      "change_kind": "update",
      "target_bead": "polylogue-w8db",
      "title": "Settings-as-assertions config resolver",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:config", "area:user-db", "area:runtime", "area:contracts"],
      "description": "Config/runtime preferences should use one resolver and one durable representation. The existing user_settings table is a dead-end if new settings are assertions.",
      "design": "Implement scoped config resolution over assertions with scope global/repo/origin/surface, actor operator/agent/harness, and precedence flag > env > file > db > default. Provide migration/copy-forward for legacy user_settings, then deprecate direct use. Learned defaults exclude config/judge telemetry from their own training data.",
      "acceptance_criteria": [
        "A setting can be read from assertions through a resolver with explicit scope, actor, source, precedence, and provenance.",
        "Legacy user_settings data is either copied forward or explicitly ignored with migration notes.",
        "No new feature writes user_settings directly.",
        "Config telemetry excludes config/judge operations from learned defaults.",
        "CLI/API/MCP expose a config audit surface showing why a value resolved."
      ],
      "dependencies": ["p5-01"],
      "parent": "polylogue-w8db",
      "risks": ["Two config stores drift", "Agents learn defaults from their own config/judgment writes", "Resolver precedence surprises operator"],
      "confidence": "med",
      "provenance": "File 5 config/runtime-preferences ideas plus code confirmation that user_settings exists separately from assertions."
    },
    {
      "proposed_id": "p5-19",
      "change_kind": "new",
      "target_bead": null,
      "title": "Ops SLO samples and stalled-vs-idle verdict",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:ops", "area:daemon", "area:observability", "area:slo"],
      "description": "A local daemon needs self-observability that distinguishes healthy quiet from silent failure. File 5 names missing SLOs around lag, convergence debt, retry poison rows, and baseline regression.",
      "design": "Add ops.db slo_samples with bounded retention and reducers for level, rate, quantile, slope, eta, and burn. Compute idle-healthy vs stalled from watcher activity, last successful attempts, backlog, quiet deferral, and work offered. Expose status and CLI doctor surfaces without alerting on raw backlog alone.",
      "acceptance_criteria": [
        "ops.db records periodic SLO samples with bounded retention/cardinality.",
        "status exposes idle_healthy, stalled, degraded, and warming states with reasons.",
        "Ingest success rate and repeated-failure signals are separate from lag.",
        "false_means_pending/convergence debt items over retry/age ceiling escalate as pathological.",
        "Regression baseline can answer whether a stage is slower than recent history.",
        "Bulk export ingestion is excluded from live-tail ingest-latency SLOs."
      ],
      "dependencies": ["polylogue-1xc"],
      "parent": "polylogue-1xc",
      "risks": ["Self-telemetry grows without bound", "Backlog count false-alarms on healthy idle archives", "Telemetry self-ingest contaminates user-facing analysis"],
      "confidence": "high",
      "provenance": "File 5 observability/SLO ideas; external OpenTelemetry research supports standardized metric naming/units but not blindly importing unstable conventions."
    },
    {
      "proposed_id": "p5-20",
      "change_kind": "update",
      "target_bead": "polylogue-3tl",
      "title": "Release cutover and docs truth gate",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:release", "area:docs", "area:distribution", "area:legibility"],
      "description": "Local files indicate Polylogue is still versioned 0.1.0 and docs say packaged install paths are not available, while the distribution stack is described as mostly wired. External legibility needs a release truth gate, not more capability work.",
      "design": "Create a release cutover checklist: verify package metadata, tag policy, PyPI/Homebrew/GHCR/FlakeHub/Nix paths, cold-machine install, docs reconciliation, README de-meta pass, and one live-cited finding/demo artifact. Keep monetization/signup/telemetry out of scope.",
      "acceptance_criteria": [
        "A cold machine can install and run the documented path or docs explicitly state it is unavailable.",
        "Package version, git tag, release notes, docs, and generated schemas agree.",
        "README first screen contains no unmeasured memory-uplift claims and links to one reproducible finding or demo.",
        "Release checklist records which distribution channels are live versus planned.",
        "No signup/cloud telemetry/monetization machinery is added."
      ],
      "dependencies": ["p5-06", "polylogue-3tl.4"],
      "parent": "polylogue-3tl",
      "risks": ["Release docs overclaim unverified channels", "Launch waits for unrelated new capability", "Public finding depends on unfixed cost/lineage bugs"],
      "confidence": "med",
      "provenance": "File 5 release/versioning section; local code confirms 0.1.0/version-doc mismatch but public tag state was not verified."
    }
  ]
}
```
