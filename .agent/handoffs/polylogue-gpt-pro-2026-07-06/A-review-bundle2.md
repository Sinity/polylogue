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

## [assistant 323]
bash -lc python3 - <<'PY'
from pathlib import Path
text=Path('/mnt/data/rnd-bundle-2-of-6.md').read_text(errors='replace')
idx=text.find('PART 2 - PAYLOAD')
lines=text[idx:].splitlines()
heads=[]
for i,l in enumerate(lines,1):
    if l.startswith('## ['):
        heads.append((i,l))
for j,(i,l) in enumerate(heads):
    end=heads[j+1][0]-1 if j+1<len(heads) else len(lines)
    sec='\n'.join(lines[i-1:end])
    print('\n'+'='*100)
    print(f'{j+1:02d} {l}')
    # Print lines with key section labels + bead titles + bugs risks
    for k,line in enumerate(sec.splitlines(),1):
        if line.startswith('#') or line.startswith('##') or line.startswith('###') or 'Top-3 risks' in line or 'Risk' in line or (line.strip().startswith('- **') and len(line)<300) or line.strip().startswith('1.') and '—' in line:
            if k<15 or line.startswith('###') or line.startswith('##') or 'Risk' in line or 'Top-3' in line or 'bead' in line.lower() or '—' in line:
                print(f'  L{i+k-1}: {line[:220]}')
PY

## [assistant 339]
bash -lc cd /mnt/data/poly_tree/polylogue && python3 - <<'PY'
from pathlib import Path
checks=[
 ('storage/sqlite/archive_tiers/index.py','CREATE VIEW IF NOT EXISTS actions'),
 ('archive/query/metadata.py','QueryUnitDescriptor'),
 ('storage/sqlite/connection.py','ATTACH'),
 ('core/enums.py','class MaterialOrigin'),
 ('storage/sqlite/archive_tiers/archive.py','def query_files'),
 ('core/refs.py','ObjectRefKind'),
 ('storage/sqlite/archive_tiers/user_write.py','def upsert_assertion'),
 ('storage/sqlite/archive_tiers/user_write.py','def upsert_saved_view'),
 ('storage/sqlite/archive_tiers/user.py','CREATE TABLE IF NOT EXISTS assertions'),
 ('storage/sqlite/archive_tiers/ops.py','OPS_SCHEMA_VERSION'),
 ('daemon/events.py','def emit_daemon_event'),
]
for file, pat in checks:
    p=Path('polylogue')/file
    if not p.exists():
        p=Path(file)
    txt=p.read_text(errors='ignore').splitlines()
    print('\n---',file)
    found=False
    for i,l in enumerate(txt,1):
        if pat in l:
            print(i, l[:180]); found=True; break
    if not found: print('NOT FOUND',pat)
PY

## [assistant 345]
bash -lc cd /mnt/data/poly_work2/polylogue && python3 - <<'PY'
from pathlib import Path
checks=[
 ('polylogue/storage/sqlite/archive_tiers/index.py','CREATE VIEW IF NOT EXISTS actions'),
 ('polylogue/archive/query/metadata.py','QueryUnitDescriptor'),
 ('polylogue/storage/sqlite/connection.py','ATTACH'),
 ('polylogue/core/enums.py','class MaterialOrigin'),
 ('polylogue/storage/sqlite/archive_tiers/archive.py','def query_files'),
 ('polylogue/core/refs.py','ObjectRefKind'),
 ('polylogue/storage/sqlite/archive_tiers/user_write.py','def upsert_assertion'),
 ('polylogue/storage/sqlite/archive_tiers/user_write.py','def upsert_saved_view'),
 ('polylogue/storage/sqlite/archive_tiers/user.py','CREATE TABLE IF NOT EXISTS assertions'),
 ('polylogue/storage/sqlite/archive_tiers/ops.py','OPS_SCHEMA_VERSION'),
 ('polylogue/daemon/events.py','def emit_daemon_event'),
 ('polylogue/storage/sqlite/archive_tiers/user.py','user_settings'),
 ('polylogue/storage/sqlite/archive_tiers/index.py','session_phases'),
 ('polylogue/storage/sqlite/archive_tiers/index.py','session_work_events'),
 ('polylogue/storage/sqlite/archive_tiers/index.py','session_commits'),
]
for file, pat in checks:
    p=Path(file)
    txt=p.read_text(errors='ignore').splitlines()
    print('\n---',file)
    found=False
    for i,l in enumerate(txt,1):
        if pat in l:
            print(i, l[:180]); found=True
            break
    if not found: print('NOT FOUND',pat)
PY

## [assistant 424]
According to the file 2 bundle from 2026-07-05, I treated Part 1 only as orientation and anchored the deliverable to Part 2’s unique payload. I also checked the uploaded working-tree snapshot where the payload made concrete file/path claims. That means “confirmed” below means confirmed against the attached source snapshot, not necessarily the live repo if it has moved since this bundle was packed. fileciteturn2file4

## Executive verdict

File 2 is one of the stronger slices. Most of its tier/regime reasoning is correct: new analytical units mostly belong in rebuildable `index.db`; cross-tier assertion joins must remain runtime query methods rather than stored views; user-authored and agent-authored overlay state should stay in `user.db.assertions`; and proof/demo/relevance work should be made executable rather than prose-only. The best parts are the `turn-pair` / `artifact` / `correction-edge` unit design, the annotation-recipe substrate, the measure-registry discipline, Polish recall, search relevance, and demo/finding proof loop. fileciteturn1file16 fileciteturn2file12 fileciteturn1file0 fileciteturn1file13

The main corrections are important. The `turn-pair` pairing law as written can duplicate one assistant answer across multiple preceding prompts. The assertion “query-back” gap in annotation recipes is stale for the uploaded source: assertions are already a registered query unit with `query_assertions`. The observability write-path diagnosis is also partially stale: `daemon_events` already has writer/query infrastructure in the snapshot, so the real gap is SLO sampling and stall/idle verdicts, not basic event recording. The artifact design is directionally right but overconfident about “strictly richer than files” until shell redirections, bash commands, relative paths, and path identity are handled honestly. The measure-registry is conceptually excellent but should not ship as a 16-measure “museum” before the registry/count/coverage gate proves itself.

My strongest strategic correction: **make file 2’s payload cohere around three platform moves** rather than many parallel feature tracks:

1. **Typed analysis units**: turn-pair, artifact, correction-edge, spec-card, TQI, SLO sample, relevance judgment.
2. **Safe external judgment loop**: annotation recipes, assertion batches, query-back, recursive-safety at `upsert_assertion`, evidence/provenance refs.
3. **Executable proof and evaluation**: `.polydemo`, finding provenance, search-eval, measure-registry, replay/spec-card/RL export.

That is the non-overlapping contribution of file 2.

## Hard vet: confirmed

### `turn-pair`, `artifact`, and `correction-edge` are the right missing units

The design correctly identifies `actions` as the model pattern: a read relation pairing tool-use blocks with tool-result blocks. A `turn-pair` view over message adjacency is a natural analogue, and it belongs in the rebuildable index tier as a view plus a covering index. The query-unit machinery is indeed descriptor-driven: new units require a closed `QueryUnitName` update, descriptor, payload model, query method, structural registry entry, generated output schemas, OpenAPI, and topology projection regeneration. The payload’s “miss any one and discovery/render tests fail” warning is accurate. fileciteturn1file16

The `artifact` distinction is also correct. Polylogue already has `files` as a path aggregation over actions, while `raw_artifacts` in the source tier is an ingest taxonomy concept. A session-created/edited/read artifact relation should not reuse `raw_artifacts` terminology, and a real artifact unit needs operation provenance and later lineage, not only “this path appeared in the session.” The proposed Phase A `artifact_touches` view and Phase B `session_artifacts` / `artifact_lineage` materialization are a sound staged design. fileciteturn2file12

The `correction-edge` tiering is exactly right. It must join durable `user.db.assertions` to index-tier blocks/messages, but not as a persistent cross-database SQLite view. SQLite does not allow a view in one attached database/schema to reference objects in another attached database, so the runtime-query-method decision is not taste; it is a platform constraint. The uploaded payload calls this out, and external SQLite guidance confirms the constraint. fileciteturn1file1 citeturn717370search20

### The annotation substrate is real, but the missing product layer is batch/schema/agent ergonomics

The `assertions` table already has the right raw shape: target/scope, kind/key/value/body, author metadata, evidence refs, lifecycle status, confidence, staleness, context policy, supersession. New assertion kinds are schema-free text, so annotation recipes do not need a durable schema migration just to add vocabulary. fileciteturn1file0

The missing layer is not storage. It is: schema registry, JSONL batch import, evidence-ref validation, recursive-safety coercion, review/acceptance surface, query-back examples, and agent-friendly CLI/MCP verbs. External annotators should write candidate assertions in batches, not bespoke JSON files whose only consumer is a report script.

### The recursive-safety write chokepoint is a real bug surface

I verified the uploaded source snapshot’s `upsert_assertion` path: it normalizes status, visibility, author, and refs, but it does not centrally coerce non-user authors into candidate/non-injecting state. `post_blackboard_note` accepts `author_kind`, and the blackboard path can reach assertion insertion. That matches the payload’s diagnosis that agent-authored claims can become authoritative unless the safety gate is enforced in the assertion write chokepoint. The bundle explicitly frames this as the invariant recall/recipe/distillery/goal/decision all assume. fileciteturn1file2

This should be treated as P0/P1 correctness, not “nice safety.” Once Polylogue auto-captures its own R&D and agents can write annotations, self-laundering is the failure mode.

### The measure-registry concept is correct and strategically important

The measure algebra is the right antidote to bespoke `analyze` modes. A measure should declare construct, formula, denominator, unit frame, evidence tier, coverage preconditions, confounds, uncertainty, and output schema. The design’s insistence that `count` route through the same measure path is especially important: otherwise `group by … | count` becomes the bypass through which unfootnoted, construct-invalid numbers leak. fileciteturn1file9 fileciteturn1file10

The default `LOGICAL_SESSION` frame is also right. Physical fork/resume/compaction duplicates are not iid observations. Treating physical sessions as independent shrinks confidence intervals dishonestly and double-counts sums. The bundle’s risk section names this clearly. fileciteturn1file13

### Polish search recall is a real defect class

The payload’s FTS diagnosis is grounded. Current `messages_fts` uses SQLite FTS5 `unicode61`; the payload says both canonical DDL sites need to stay in sync, and that query-side normalization currently relies on the tokenizer rather than doing Polish-specific folding. The specific Polish issue is also real: `ł/Ł` is not handled by ordinary combining-mark removal, so relying on `remove_diacritics` alone leaves a recall hole. The staged plan—shared `pl_fold`, write-side folding into contentless FTS, query-side symmetry tests, then `block_prose_lang`—is sound. fileciteturn2file6 fileciteturn2file2

SQLite’s FTS5 `unicode61` tokenizer is the relevant built-in baseline, and using a Python/SQL folding function avoids writing a custom native FTS tokenizer, which would violate Polylogue’s current “no native deps” posture. citeturn717370search1

### Search relevance needs durable judgments before tuning

The search-relevance slice is strong. It correctly separates “ranking feels off” from the substrate needed to improve it: `RELEVANCE` assertions keyed by query fingerprint and target ref, an eval harness reporting nDCG/MRR/recall, explainability over existing `score_components`, lineage-aware collapse, and ranking-policy versioning. fileciteturn2file11 fileciteturn2file15

Do not tune RRF weights by intuition. First build judgments and an eval harness. Only then add weighted fusion, MMR, weak-evidence tags, or cross-query calibration.

### Demo proof engine and finding provenance are high-leverage

The `.polydemo` / finding-id / evidence-ref resolver / budget telemetry / privacy gate direction is exactly aligned with Polylogue’s credibility story. The important move is that a demo becomes an executable proof object, not a screenshot or README claim. The payload’s proposal for `.polydemo` frontmatter, declared constructs, failure taxonomy, artifact-hash drift, and CI integration should survive. fileciteturn2file0

The “finding provenance as table, not prose” idea is also right. A finding needs a queryable provenance stanza: measurement cursor, measure/DSL/code version, git SHA, sample-frame predicate, run date, and staleness state. That aligns with W3C PROV’s entity/activity/agent framing and RO-Crate’s approach of describing research objects and their contents with machine-readable metadata. fileciteturn1file17 citeturn272480search0turn272480search1

## Hard vet: refuted or corrected

### The `turn-pair` property test is wrong as stated

The proposed SQL pairs each prompt with the next later assistant message. If two human/operator messages occur before the next assistant answer, both prompts can pair to the same assistant. Therefore the proposed property “no assistant message is claimed by two prompts” is not guaranteed by the SQL. fileciteturn1file6

Correct design options:

- If a “turn” means “prompt burst answered by one assistant,” group consecutive human/operator prompts before the assistant into one prompt-set.
- If a “turn” means “one prompt row,” pair only the last prompt before an assistant and mark earlier prompt rows as superseded/unanswered-within-burst.
- If preserving every prompt row matters, expose `answer_claim_count` or `prompt_burst_id` so downstream rates do not assume one-to-one.

I recommend **prompt-burst turn pairs** for analytics and a lower-level message adjacency relation for forensic inspection.

### `artifact_touches` is not yet strictly richer than `files`

It is richer for structured tool-path operations. It is not richer for bash redirections, `tee`, `sed -i`, `cat > file`, generated files without `tool_path`, absolute/relative path aliases, renamed paths, or provider-specific tools that do not map to `Read/Edit/Write`. The payload acknowledges open-ended tool vocabulary risk, but the “strictly richer” statement should be weakened until a tool-operation classifier and shell-path parser exist. fileciteturn2file12

The first landing should call itself “structured artifact touches,” not “all artifacts a session touched.”

### Annotation “query-back” is stale for this source snapshot

The payload/meta-summary says annotation recipes have a “query-back” completeness gap because assertions are MCP-list-only and not in the `find` DSL. That is not true in the uploaded working tree. `assertion` is already a terminal/structural query unit, and `query_assertions` exists as a runtime method over `user_tier.assertions`. The real gap is not queryability; it is batch import, schema validation, review ergonomics, and recursive-safety coercion. fileciteturn2file17 fileciteturn1file0

So: do not create a bead for “make assertions queryable” as if absent. Create one for “make external assertion batches easy and safe.”

### Observability should reuse existing event infrastructure

The steady-state observability design is mostly right: SLO samples, reducers, idle-vs-stalled verdict, live-tail scoping, disposable ops-tier retention. But the current snapshot already has daemon event infrastructure; the missing work is not “create event writing from nothing.” The landing should reuse the existing daemon-events and cursor-lag patterns and add `slo_samples`/reducers/retention. OpenTelemetry and Prometheus both make a useful distinction between raw events, histograms/distributions, and derived time-series; Polylogue should follow that separation. fileciteturn2file18 citeturn272480search2turn272480search3

There is also an unresolved ops-tier policy conflict. Some file-2 designs assume an `ops.db` version bump; others assume self-healing optional telemetry tables. Pick one rule: I recommend **self-heal optional telemetry tables, but version/bump only when a table becomes a required contract validated by doctor/CI**.

### Measure registry is over-scoped if shipped all at once

The 16 named measures are useful, but shipping all 16 before the registry/count/coverage core is a classic “analytics museum” failure. The critical-path material in the bundle itself warns that broad behavioral analytics can be orthogonal until a live query demands them. fileciteturn2file14

First PR should be: `MeasureSpec`, `count`, coverage refusal, tier footnotes, Wilson/bootstrap primitives, and rigor audit. The named measures should land in small batches only when a proof/demo/report uses them.

### TQI must stay reward-shaping, not truth

The spec-card/TQI slice is valuable, but TQI is a composite score over partially heuristic subscores. It should be emitted only with component tiers, NULL/insufficient-coverage gates, base rates, and a Goodhart warning. The file already says this, and I agree. fileciteturn2file1

The stronger design is to make TQI a `MeasureSpec`-backed projection with visible components, not a single ranking number that surfaces by default.

### Trigram fallback and “Polish recall” need measurement before default enablement

Trigram fallback can improve substring/typo/inflection recall, but it can also expand index size and hurt precision. The payload estimates roughly 1.5–2× FTS size and recommends fallback/recall-booster behavior rather than replacing primary FTS. That is the right posture. It should be behind a measured benchmark and an explicit routing rule, not silently enabled for all queries. fileciteturn2file9

### Performance recommendations need sharper “prove then build” gates

The performance slice correctly identifies exact vector scans, repeated per-seed fanout, no query-plan regression lane, no preview cache, and exact counts on interactive paths. fileciteturn2file8

External context shifts the ANN story. `sqlite-vec` now documents binary quantization as a way to reduce vector size, but it still describes the current search path as brute-force; SQLite’s separate Vec1 extension now offers ANN via IVFADC/OPQ. citeturn422173search0turn422173search3

That means the right bead is not “migrate to ANN.” It is: benchmark exact f32, int8/binary quantization + rerank, sqlite-vec metadata partitioning, and Vec1 as an experimental sidecar candidate. Keep “SQLite as sole store” as a constraint until numbers prove otherwise.

## External context and design implications

The local protocol/composer work should borrow from LSP without becoming LSP. LSP’s cancellation mechanism is a useful precedent for keystroke-driven completion/preview: superseded requests should be cancellable by id, not merely ignored client-side. citeturn717370search3

The proof engine should borrow from reproducible research packaging rather than ordinary demos. W3C PROV gives the right conceptual triad—entities, activities, agents—and RO-Crate gives the “research object with machine-readable metadata” pattern. For Polylogue, the minimal adaptation is: finding entity, query/measure activity, archive/corpus entity, agent/operator, evidence refs, generated artifact, and verification status. citeturn272480search0turn272480search1

The MCP-facing parts should use MCP resources and prompts deliberately. Resources are the natural way to expose evidence packs, schemas, query result refs, and report artifacts; prompts are the natural way to publish reusable analysis recipes without hiding them inside a bespoke daemon command. citeturn272480search4turn272480search15

For RL/eval export, Atropos is relevant but must be verified live before locking schema. It is an open-source environment framework for collecting/evaluating LLM rollouts, but file 2’s own caution is correct: the export should be a pure read projection first, not a training system. fileciteturn2file1 citeturn272480search5

## Prioritized take

**Do immediately.** Enforce recursive-safety in `upsert_assertion`; ship annotation batch import; create `turn-pair` corrected to prompt-burst semantics; add `correction-edge` runtime query; build demo/finding proof loop; add search relevance judgments and eval harness; add query-plan regression/perf gates.

**Do next.** Land structured artifact touches; Polish `pl_fold` and FTS symmetry; measure-registry core with `count`; SLO samples and idle-vs-stalled verdict; finding provenance table; spec-card export.

**Do after proof.** TQI as reward-shaping, RL/eval export, trigram fallback, MMR/diversity, vector quantization/ANN, incident/compaction lifecycle analytics.

**Need live source before final implementation.** Re-check current repo HEAD for line drift; confirm whether `s7ae` shipped after the bundle; verify current Atropos schema; measure live archive performance for vector scans, counts, and artifact materialization; sample real provider tool vocab before hardcoding artifact operation classes.

## Proposed bead changes

```json
{
  "file": 2,
  "proposed_beads": [
    {
      "proposed_id": "p2-01",
      "change_kind": "new",
      "target_bead": null,
      "title": "Coerce agent-authored assertions at write chokepoint",
      "issue_type": "bug",
      "priority": 0,
      "labels": ["area:assertions", "area:safety", "area:user-db", "area:agents"],
      "description": "Agent-authored overlay claims can reach assertion insertion paths without a single central coercion rule. In the uploaded source snapshot, upsert_assertion does not force non-user authors to candidate/non-injecting state, and blackboard/annotation paths can pass author_kind through. This is a recursive-safety hole for annotation recipes, context scheduler, blackboard, judgment queue, and self-captured R&D.",
      "design": "Add a single coerce_agent_authored helper inside the assertion write path. If author_kind != user, force status=candidate and context_policy.inject=false unless the existing row is already terminal judged; never resurrect accepted/rejected/superseded/deleted rows. Apply before all user_write assertion upserts, including blackboard, annotation, correction, prompt-eval, pathology, goal, and decision writers. Add tests that every writer inherits the rule.",
      "acceptance_criteria": [
        "A non-user author passed through save_annotation, blackboard_post, record_correction, and any candidate writer lands as status=candidate with inject false.",
        "A user author can still create active/injectable assertions when the caller explicitly requests it.",
        "Terminal judged rows are not resurrected by later agent-authored writes with the same natural key.",
        "A regression test proves context-scheduler injection excludes agent-authored candidate assertions.",
        "Generated OpenAPI and CLI output schemas still render every AssertionKind and status."
      ],
      "dependencies": [],
      "parent": "polylogue-37t.11",
      "risks": [
        "Too-strict coercion may slow legitimate agent-maintained blackboards.",
        "Existing tests may rely on agent notes becoming active.",
        "Natural-key collision handling must not silently overwrite human judgments."
      ],
      "confidence": "high",
      "provenance": "Hard-vetted recursive-safety hole in Part-2 annotation/blackboard substrate."
    },
    {
      "proposed_id": "p2-02",
      "change_kind": "new",
      "target_bead": null,
      "title": "Annotation recipe registry and batch import",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:assertions", "area:analysis", "area:mcp", "area:cli"],
      "description": "External agents should annotate sessions/messages/results as candidate assertions without bespoke JSON glue. The assertion substrate exists, but file 2's actual gap is schema-governed batch import, validation, evidence refs, and review affordances.",
      "design": "Add annotation recipe definitions in git YAML or a recipe registry, plus a CLI/MCP import path for JSONL rows. Each batch records schema_id, author/model/prompt, target grain, scope_ref, evidence refs, row counts, validation errors, and review state. Rows import as candidate assertions using the p2-01 coercion rule. Recipes expose query examples so agents can query their own annotations after import.",
      "acceptance_criteria": [
        "polylogue assertions import --schema <id> file.jsonl validates rows and writes a batch summary.",
        "Invalid target_ref, missing required evidence_refs, unknown schema fields, and wrong target grain produce structured refusal rows, not partial silent success.",
        "Imported rows are queryable through the existing assertion query unit by schema/key/value/scope.",
        "MCP exposes a safe assert/import affordance with the same validation contract.",
        "A fixture recipe annotates messages with two labels and then queries aggregate counts from the imported assertions."
      ],
      "dependencies": ["p2-01"],
      "parent": "polylogue-37t.11",
      "risks": [
        "Schema design can overfit to one demo.",
        "Large batches may need streaming validation.",
        "Agents may treat candidate labels as truth unless render surfaces show status."
      ],
      "confidence": "high",
      "provenance": "Part-2 annotation-recipe substrate plus correction that query-back already exists but batch ergonomics do not."
    },
    {
      "proposed_id": "p2-03",
      "change_kind": "new",
      "target_bead": null,
      "title": "Turn-pair unit with prompt-burst semantics",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:index", "area:analytics", "unit:turn-pair"],
      "description": "Per-turn latency, cost, correction-rate, and answer anchoring need a human/operator prompt to assistant-answer relation. The proposed simple MIN(next assistant) view is close, but can attach multiple preceding prompts to the same answer.",
      "design": "Add an index-tier turn relation that groups consecutive human_authored/operator_command prompts into a prompt_burst before the next assistant_authored active-path answer. Expose prompt_message_ids, answer_message_id, prompt_origin summary, burst_size, latency when timestamps exist, token columns from the answer, and variant/continuation count if available. Register the query unit and generated payloads. Do not treat operator_command as human text silently; expose prompt_origin filters.",
      "acceptance_criteria": [
        "A session with human -> human -> assistant yields one turn-pair with burst_size=2, not two rows claiming the same answer.",
        "A session with human -> tool_result -> assistant pairs correctly and skips runtime/tool rows.",
        "A trailing unanswered prompt burst yields a row with NULL answer fields and abandoned=true.",
        "turn_latency_ms is NULL when either timestamp is NULL and non-negative when both exist.",
        "find 'turn-pairs where answer_model_name:opus' and group-by/count execute across CLI/API/MCP.",
        "OpenAPI, CLI output schemas, query completions, and topology projection are regenerated."
      ],
      "dependencies": [],
      "parent": null,
      "risks": [
        "Provider exports may not preserve enough message timing for latency.",
        "Operator commands can pollute natural-language turn metrics unless filtered.",
        "Variant/fork semantics may require later lineage-aware aggregation."
      ],
      "confidence": "high",
      "provenance": "Units-B turn-pair design confirmed; corrected one-to-one pairing invariant."
    },
    {
      "proposed_id": "p2-04",
      "change_kind": "new",
      "target_bead": null,
      "title": "Structured artifact-touch unit and lineage seed",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:index", "area:artifacts", "unit:artifact"],
      "description": "The existing files unit aggregates tool paths and lacks per-operation provenance. File 2 correctly separates session artifacts from attachments and source-tier raw_artifacts, but the MVP must be honest that only structured tool paths are covered initially.",
      "design": "Add artifact_touches over actions for structured tool_path-bearing operations, then materialize session_artifacts and artifact_lineage in index.db. Use a provider/tool operation classifier table, path normalization with NFC and absolute/relative handling, and confidence fields. Keep shell command path extraction as a separate optional phase. Do not reuse raw_artifacts names.",
      "acceptance_criteria": [
        "Read/Edit/Write style structured tool calls produce per-operation artifact_touches rows with operation and evidence block refs.",
        "session_artifacts aggregates per session/path with read_count, edit_count, create_count, error_count, first/last evidence refs, and confidence.",
        "Attachment-only sessions produce zero session_artifacts rows.",
        "source.raw_artifacts remains untouched and no public ObjectRef kind conflicts with it.",
        "Re-materializing the same session is idempotent and rebuild parity holds after ops reset --index.",
        "Shell redirection operations without reliable path extraction are classified as unknown/touch rather than guessed."
      ],
      "dependencies": [],
      "parent": null,
      "risks": [
        "Tool vocabulary varies heavily by provider.",
        "Path identity is hard across cwd, symlinks, absolute paths, and renames.",
        "Over-eager lineage inference can fabricate provenance."
      ],
      "confidence": "med",
      "provenance": "Units-B artifact design confirmed with stricter MVP boundary."
    },
    {
      "proposed_id": "p2-05",
      "change_kind": "new",
      "target_bead": null,
      "title": "Correction-edge query unit over assertion anchors",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:assertions", "area:analytics", "unit:correction-edge"],
      "description": "Correction assertions need a relation that resolves target_ref anchors to corrected messages/blocks/tools/models so error-rate-per-tool and correction-density measures can be computed. This must not be a stored cross-tier view.",
      "design": "Implement query_correction_edges as a runtime query method that attaches user_tier if present and joins correction assertions by target_ref to blocks/messages/sessions. Return resolution state, corrected_tool_name, corrected_model_name, anchor_kind, session origin/repo, evidence refs, and coarse/null fields for unresolved or session-level anchors. Add a devtools policy comment/check forbidding persistent cross-tier views.",
      "acceptance_criteria": [
        "block: target refs resolve to block, message, session, tool, and model fields.",
        "message: target refs resolve to message/session/model and NULL tool where no block is known.",
        "session: target refs remain coarse and do not pretend tool-level precision.",
        "Unresolved refs emit rows with resolution=unresolved instead of disappearing.",
        "Non-correction assertions do not appear.",
        "The query returns [] when user.db is absent, matching query_assertions behavior."
      ],
      "dependencies": ["p2-01", "p2-03"],
      "parent": null,
      "risks": [
        "Most current correction anchors may be session-coarse, limiting denominator quality.",
        "Future block content-hash anchors may require another resolver layer.",
        "Cross-tier attach order is easy to regress."
      ],
      "confidence": "high",
      "provenance": "Units-B correction-edge design confirmed; SQLite cross-db view constraint externally verified."
    },
    {
      "proposed_id": "p2-06",
      "change_kind": "new",
      "target_bead": null,
      "title": "Coordination message bus with scheduler advisories",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:coordination", "area:daemon", "area:context", "area:agents"],
      "description": "The coordination envelope, agents CLI, daemon route, and web projection are already present in the uploaded source snapshot. The missing Part-2 payload is v2: addressed messages, unread/ack state, TTL, query refs, and scheduler-mediated advisories.",
      "design": "Add coordination_message and coordination_ack in user.db only after the ContextSource trust gate is defined. Messages carry scope, recipients, TTL/expires_at, query:<hash> refs, evidence refs, author_kind, and status. The coordination envelope consumes messages through ContextSource rather than ad-hoc strings so advisory injection is bounded, trust-classed, and ledgered.",
      "acceptance_criteria": [
        "Agents can post a scoped coordination message with recipient, TTL, and query/result refs.",
        "Recipients can list unread messages and ack them idempotently.",
        "Expired messages do not inject into context but remain auditable.",
        "Coordination advisories enter context only through the ContextSource scheduler path.",
        "Two-agent same-repo proof shows conflict detection, message delivery, ack, and handoff rendering.",
        "Agent-authored messages inherit p2-01 recursive-safety behavior."
      ],
      "dependencies": ["p2-01"],
      "parent": "polylogue-s7ae",
      "risks": [
        "Message bus can become a second task tracker.",
        "Unbounded advisories could become context spam.",
        "Durable migration should be batched with other user.db v5 work."
      ],
      "confidence": "med",
      "provenance": "Part-2 coordination substrate: shipped envelope confirmed, v2 bus remains net-new."
    },
    {
      "proposed_id": "p2-07",
      "change_kind": "update",
      "target_bead": "polylogue-9l5.7",
      "title": "Stage measure registry around count and coverage first",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:analytics", "area:honesty", "area:query", "area:measures"],
      "description": "The MeasureSpec algebra is load-bearing, but landing all 16 measures at once risks building an analytics museum before the construct-validity path is proven. The first slice should make count a measure, attach evidence tiers everywhere, and refuse uncovered groups.",
      "design": "Update 9l5.7 to land in phases: MeasureSpec + stats primitives; evaluate_measure with count/sum/mean/proportion/ratio; coverage gate; tier/confound footnotes; query DSL measure/count path; registry audit. Named measures then land only when a demo/report needs them.",
      "acceptance_criteria": [
        "count routes through evaluate_measure and emits a MeasureResult with tier/confound metadata.",
        "A coverage-unmet group renders refused=true and emits no numeric point estimate.",
        "Every rendered measure on CLI/API/MCP carries evidence tier and denominator information.",
        "The rigor audit fails if any MeasureSpec lacks construct, denominator for ratios/proportions, evidence tier, or confounds.",
        "At least one end-to-end query uses group by + measure count with the new path.",
        "The 16-measure backlog is split into later children and not required for the first PR."
      ],
      "dependencies": [],
      "parent": "polylogue-9l5.7",
      "risks": [
        "Legacy aggregate paths may bypass the registry.",
        "Coverage predicates can become too complex if overgeneralized early.",
        "Render surfaces may keep old count formatting."
      ],
      "confidence": "high",
      "provenance": "Part-2 measure-registry value confirmed; scope corrected to avoid premature 16-measure bundle."
    },
    {
      "proposed_id": "p2-08",
      "change_kind": "new",
      "target_bead": null,
      "title": "Ops-tier SLO samples and idle-vs-stalled verdict",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:observability", "area:daemon", "area:ops-db", "area:slo"],
      "description": "Polylogue needs steady-state observability over convergence and ingestion that distinguishes harmless backlog from stalled work. The source snapshot already has daemon event infrastructure, so the missing piece is SLO sampling/reducers and verdicts, not basic event logging.",
      "design": "Add optional ops-tier slo_samples with closed-set labels, retention GC, reducers for level/quantile/slope/eta/burn, and an idle-vs-stalled verdict. Reuse existing daemon-events and cursor-lag patterns. Define an ops-tier schema policy: optional telemetry tables self-heal; only required contract tables bump ops schema.",
      "acceptance_criteria": [
        "slo_samples records bounded samples for selected convergence/ingest/search/daemon metrics.",
        "Reducers emit level, quantile, slope, ETA, and burn-rate where enough history exists.",
        "A backlog is reported stalled only when work is offered and not draining; idle backlog is separate.",
        "Bulk imports are excluded from live-tail ingest latency SLOs by construction.",
        "Metrics export uses low-cardinality labels and does not block the single writer.",
        "Retention GC bounds table size and cold-start baselines degrade honestly to level-only."
      ],
      "dependencies": [],
      "parent": "polylogue-peo",
      "risks": [
        "Telemetry can load the writer if sampled too often.",
        "Disposable ops schema changes can erase baselines.",
        "False stall alerts will be ignored."
      ],
      "confidence": "med",
      "provenance": "Part-2 observability design confirmed but corrected to reuse existing daemon_events."
    },
    {
      "proposed_id": "p2-09",
      "change_kind": "new",
      "target_bead": null,
      "title": "Polish FTS fold and tokenizer symmetry",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:search", "area:fts", "area:index", "lang:pl"],
      "description": "Current FTS relies on unicode61 and does not fold Polish ł/Ł, causing common ASCII queries such as latwo to miss łatwo. The fix belongs in rebuildable index.db and must keep query-side and write-side folding symmetric.",
      "design": "Add shared deterministic pl_fold in Python and as a SQL function registered on index writer connections. Change both FTS DDL sites to unicode61 remove_diacritics 2 and fold search_text before contentless FTS insertion. Normalize FTS query terms with the same pl_fold. Add drift-lock tests for both DDL sites and all MATCH builders.",
      "acceptance_criteria": [
        "Queries latwo/łatwo and zrobilem/zrobiłem hit seeded Polish text containing łatwo/zrobiłem.",
        "pl_fold is idempotent and Python/SQL implementations agree on a seeded corpus.",
        "Both canonical FTS DDL sites use the same tokenizer string and a test fails on drift.",
        "All message FTS MATCH builders pass through the normalizer or are explicitly out of scope.",
        "Writer connections register pl_fold before triggers can fire.",
        "INDEX_SCHEMA_VERSION bump and rebuild plan are documented."
      ],
      "dependencies": [],
      "parent": "polylogue-kwsb",
      "risks": [
        "Missing SQL function registration can stall ingest.",
        "Query/index folding drift silently loses recall.",
        "Non-message FTS tables may need separate follow-up treatment."
      ],
      "confidence": "high",
      "provenance": "Part-2 multilingual Polish search recall; code snapshot confirmed unicode61-only baseline."
    },
    {
      "proposed_id": "p2-10",
      "change_kind": "new",
      "target_bead": null,
      "title": "Prose language facts and cautious trigram recall lane",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:search", "area:index", "area:language", "lang:pl"],
      "description": "Polish recall needs language facts and optional fallback behavior beyond the stroke-fold fix, but trigram fallback can hurt precision and index size if shipped as default.",
      "design": "Materialize block_prose_lang facts with detector/version/confidence/is_mixed and read-side rollups. Add prose_lang query predicate and assertion-backed overrides. Add messages_fts_trigram only as an explicitly measured fallback lane that unions/re-ranks when primary recall is low.",
      "acceptance_criteria": [
        "Mixed Polish/English prose is marked mixed rather than collapsed to one language.",
        "Low-confidence short blocks are lang=und.",
        "prose_lang:pl filters through CLI/API/MCP and explain shows the lowering.",
        "Assertion-backed language override affects read-side language without mutating derived facts.",
        "Trigram fallback is off or gated by a routing rule until benchmarked.",
        "Index-size and precision/recall impact are reported before default enablement."
      ],
      "dependencies": ["p2-09"],
      "parent": "polylogue-kwsb",
      "risks": [
        "Language detection can overstate confidence on short technical blocks.",
        "Trigram index size may be high.",
        "Fallback routing can hide poor primary recall if not measured."
      ],
      "confidence": "med",
      "provenance": "Part-2 Polish recall B3-B8, with fallback scope corrected."
    },
    {
      "proposed_id": "p2-11",
      "change_kind": "new",
      "target_bead": null,
      "title": "Spec-card table and portable benchmark export",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:eval", "area:index", "area:provenance", "area:export"],
      "description": "Sessions that contain implementation specs, commits, outcomes, and verification signals can become portable benchmark/eval rows, but only if outcome grounding and leakage controls are explicit.",
      "design": "Add spec_cards in index.db deriving task title/intent, repo/commit refs, outcome evidence tier, verification command/result, completeness, and leakage-safe export fields. Export JSONL without message bodies by default and include per-field evidence refs and redaction controls.",
      "acceptance_criteria": [
        "build_spec_card produces deterministic rows from profile, commits, actions, and outcome evidence.",
        "pr_merged, explicit_ref, and intent-only outcomes carry different evidence tiers.",
        "polylogue analyze spec-cards --export emits JSONL with no message bodies by default.",
        "Leakage guard rejects private paths/titles unless explicitly redacted or allowed.",
        "ops reset --index rebuilds byte-identical spec_cards on the fixture corpus.",
        "Export schema is documented and generated surfaces are updated."
      ],
      "dependencies": [],
      "parent": "polylogue-fs1",
      "risks": [
        "Commit attribution may be low confidence.",
        "Benchmark export can leak private intent text.",
        "Spec-card completeness can be mistaken for task success."
      ],
      "confidence": "med",
      "provenance": "Part-2 spec-card/TQI slice; spec-cards are the safer first eval export."
    },
    {
      "proposed_id": "p2-12",
      "change_kind": "new",
      "target_bead": null,
      "title": "TQI as MeasureSpec-backed reward-shaping signal",
      "issue_type": "feature",
      "priority": 3,
      "labels": ["area:eval", "area:analytics", "area:measures", "risk:goodhart"],
      "description": "A trajectory quality index can help dashboards and reward-shaping, but it is a composite heuristic and must never masquerade as ground truth.",
      "design": "Implement TQI only after MeasureSpec core exists. Register each component as a tiered measure or derived subscore with confounds, coverage gates, and NULL propagation. Emit a scalar only in reward-shaping/export modes with component breakdown, base-rate context, and Goodhart caveat.",
      "acceptance_criteria": [
        "TQI is declared as a MeasureSpec or composition of MeasureSpecs with evidence tiers and confounds.",
        "Insufficient coverage refuses scalar output instead of fabricating a score.",
        "Rendered TQI always includes component values, tiers, and caveat text.",
        "Reward-shaping export documents sign, normalization, and intended non-authoritative use.",
        "Property tests cover range, monotonicity where valid, and NULL propagation.",
        "No default ranking surface sorts sessions by TQI without an explicit flag."
      ],
      "dependencies": ["p2-07", "p2-11"],
      "parent": "polylogue-9l5.7",
      "risks": [
        "Goodhart pressure if used as sole reward.",
        "Component weights may encode arbitrary taste.",
        "Fragmentation proxies may not measure intent quality."
      ],
      "confidence": "med",
      "provenance": "Part-2 TQI design retained but constrained to reward-shaping."
    },
    {
      "proposed_id": "p2-13",
      "change_kind": "new",
      "target_bead": null,
      "title": "Read-access log and anti-feedback attention model",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:context", "area:ops-db", "area:attention", "area:privacy"],
      "description": "The context scheduler needs an attention signal derived from human/agent reads, but it must not reinforce its own context injections or become durable surveillance by default.",
      "design": "Add disposable read_access_log/attention counters in ops.db with in-process debounce, decayed counters, source/surface labels, and explicit exclusion for context_inject. Expose only aggregated attention to the scheduler. Durable promotion requires a user assertion or analysis run.",
      "acceptance_criteria": [
        "Reading a session/message/artifact through CLI/MCP/web records a debounced ops event.",
        "Repeated context injection does not increment attention.",
        "Attention decays over time and is queryable through a bounded diagnostic payload.",
        "The scheduler can request attention features without reading raw access logs.",
        "Retention policy deletes or compacts old raw access events.",
        "Tests prove injected items cannot self-amplify their future injection score."
      ],
      "dependencies": ["p2-01", "p2-08"],
      "parent": "polylogue-37t.11",
      "risks": [
        "Self-reinforcing memory loops if exclusions are incomplete.",
        "Privacy concern if raw access logs become durable.",
        "High-frequency reads can create noisy counters."
      ],
      "confidence": "med",
      "provenance": "Part-2 read-access/attention slice plus recursive-safety concerns."
    },
    {
      "proposed_id": "p2-14",
      "change_kind": "update",
      "target_bead": "polylogue-3tl.4",
      "title": "Executable polydemo proof engine",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:demo", "area:proof", "area:ci", "area:legibility"],
      "description": "The launch proof should be an executable artifact, not a screenshot or README number. File 2's proof-engine material defines a .polydemo format, citable finding IDs, privacy gates, budget telemetry, and CI verification.",
      "design": "Update 3tl.4 to implement .polydemo as a committable recipe with frontmatter budgets, corpus datasheet hash, declared constructs, product-primitive CLI steps, expected finding IDs, privacy/leakage checks, artifact hash drift checks, and render all --check integration. Restrict steps to polylogue argv, not arbitrary Python scripts.",
      "acceptance_criteria": [
        "uvx --from . polylogue demo run <file.polydemo> reproduces the committed artifact on fixture corpus.",
        "Each step declares expected construct coverage and finding_id or refusal state.",
        "Evidence refs round-trip and dangling refs fail the demo.",
        "Private path/title leakage check runs over markdown/html/transcript/demo frames where applicable.",
        "report.json records per-step wall time and fails if first-result/full-tour budgets are exceeded.",
        "A passing demo can be rerun in CI and produces byte-identical or explicitly versioned output."
      ],
      "dependencies": [],
      "parent": "polylogue-3tl",
      "risks": [
        "Recipe format can grow into another DSL.",
        "Overly strict byte identity can create brittle CI.",
        "Privacy verifier may miss rendered non-text leaks."
      ],
      "confidence": "high",
      "provenance": "Part-2 demo proof engine and reproducible-finding payload."
    },
    {
      "proposed_id": "p2-15",
      "change_kind": "new",
      "target_bead": null,
      "title": "Queryable finding provenance and staleness",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:findings", "area:provenance", "area:index", "area:honesty"],
      "description": "A finding's provenance stanza should be queryable and automatically checked for staleness rather than embedded in prose nobody rereads.",
      "design": "Materialize finding_provenance in the rebuildable/index layer or as a derived view over stored finding assertions and query runs. Capture finding_id, measure id/version, query/result fingerprint, code/git SHA, corpus datasheet hash, sample-frame predicate, run timestamp, evidence refs, and staleness verdict. Recompute staleness on re-ingest/rebuild.",
      "acceptance_criteria": [
        "polylogue finding show <id> displays the provenance row and evidence refs.",
        "findings can be queried by stale/drifted/deleted evidence state.",
        "Changing the sample-frame predicate or corpus version marks affected findings stale or needs-rerun.",
        "A finding cannot render as current if evidence refs no longer resolve.",
        "The provenance table/view is rebuilt or refreshed deterministically.",
        "The .polydemo proof engine consumes the same provenance data."
      ],
      "dependencies": ["p2-07", "p2-14"],
      "parent": "polylogue-cpf",
      "risks": [
        "Index-tier provenance for user-promoted findings may need careful source of truth.",
        "Hash/canonicalization drift can orphan old findings.",
        "Too many stale flags can create noise without prioritization."
      ],
      "confidence": "high",
      "provenance": "Part-2 provenance-stanza-as-table idea, aligned with proof engine."
    },
    {
      "proposed_id": "p2-16",
      "change_kind": "new",
      "target_bead": null,
      "title": "Relevance judgments and search-eval harness",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:search", "area:eval", "area:assertions", "area:ranking"],
      "description": "Search tuning needs durable query-target judgments and a regression harness before changing RRF weights, dedup, MMR, or calibration. The payload identifies score_components and missing relevance-judgment substrate.",
      "design": "Add RELEVANCE assertion rows keyed by query_fingerprint/result target_ref/ranking_policy_version. Provide mark-relevant/irrelevant/best affordances and devtools lab search-eval reporting nDCG@10, MRR, recall@k, and per-lane metrics. Saved queries can carry attached judgments and become relevance fixtures.",
      "acceptance_criteria": [
        "A user can mark a search result relevant, irrelevant, or best from CLI or MCP.",
        "Judgments are stored as assertions with query fingerprint, target_ref, ranking_policy_version, and evidence status.",
        "devtools lab search-eval runs on fixture judgments and reports nDCG@10, MRR, and recall@k.",
        "Changing ranking weights without updating ranking_policy_version is detected by a regression test.",
        "Search tuning changes must show non-regression on the judged set before merge.",
        "Agent-authored relevance judgments are candidate by p2-01."
      ],
      "dependencies": ["p2-01"],
      "parent": null,
      "risks": [
        "Small judged sets can overfit.",
        "Operator judgments may be query-context dependent.",
        "Judgment UI friction can prevent corpus growth."
      ],
      "confidence": "high",
      "provenance": "Part-2 search relevance and explainability payload."
    },
    {
      "proposed_id": "p2-17",
      "change_kind": "new",
      "target_bead": null,
      "title": "Search explain, lineage collapse, and weak-evidence tags",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:search", "area:lineage", "area:ux", "area:honesty"],
      "description": "Hybrid search currently has per-lane score components but lacks an operator-facing explanation, lineage-aware duplicate collapse, and weak-evidence honesty labels.",
      "design": "Build explain search / --why over score_components, add lineage-root collapse with variant_count and best-anchor selection, and tag single-lane deep-rank hits as weak_evidence. Do not add weighted RRF/MMR until p2-16 produces an eval corpus.",
      "acceptance_criteria": [
        "Search hits can render a concise why-ranked explanation from lexical/vector ranks and RRF components.",
        "Lineage siblings can collapse to one result with variant_count and expandable members.",
        "Collapsed ranking chooses a representative by strongest supported member and exposes the policy.",
        "weak_evidence=true appears for hits below a declared lane-support threshold.",
        "Highlight/anchor round-trip test proves matched terms appear in resolved evidence blocks.",
        "Weighted RRF/MMR changes are blocked on p2-16 eval availability."
      ],
      "dependencies": ["p2-16"],
      "parent": null,
      "risks": [
        "Dedup can hide the actually relevant fork.",
        "Weak-evidence threshold may suppress useful long-tail hits.",
        "Explanations can imply calibrated confidence where none exists."
      ],
      "confidence": "med",
      "provenance": "Part-2 search relevance payload extended with sequencing guard."
    },
    {
      "proposed_id": "p2-18",
      "change_kind": "new",
      "target_bead": null,
      "title": "Interactive query-plan regression lane",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:perf", "area:query", "area:sqlite", "area:devtools"],
      "description": "Hot interactive paths can silently flip from indexed SEARCH to table SCAN after schema/query changes. File 2 identifies the lack of EXPLAIN QUERY PLAN goldens for ranked search, actions aggregates, and session lists.",
      "design": "Add a devtools lab perf query-plans lane with a small fixture corpus and golden plan assertions for critical interactive queries. Assert use of expected indexes/search paths without overfitting to every SQLite string. Include plan drift review output for intentional changes.",
      "acceptance_criteria": [
        "Golden plans cover session list, FTS search, hybrid candidate lookup, actions aggregate, query_units, and preview count probes.",
        "A forced missing-index fixture fails the lane by producing SCAN where SEARCH/index use is expected.",
        "The lane tolerates harmless SQLite wording differences but catches material plan regressions.",
        "CI or devtools verify can run the lane quickly on fixtures.",
        "Perf report names the exact query and missing/unused index."
      ],
      "dependencies": [],
      "parent": null,
      "risks": [
        "SQLite plan strings can be brittle across versions.",
        "Over-specific goldens can block legitimate planner improvements.",
        "Fixture plans may not match the 38GB archive."
      ],
      "confidence": "high",
      "provenance": "Part-2 performance slice: no query-plan regression lane."
    },
    {
      "proposed_id": "p2-19",
      "change_kind": "new",
      "target_bead": null,
      "title": "Preview count and keyset pagination fast path",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:perf", "area:query", "area:composer", "area:sqlite"],
      "description": "Interactive preview cannot pay exact COUNT(*) and OFFSET scans on every keystroke. The performance payload identifies exact counts and pagination as a hot-path risk.",
      "design": "Add preview-specific query execution with bounded top-N probes, exact/estimate/capped count precision, keyset pagination for sorted session lists, and prefix-result cache keyed by normalized query/scope/limit/archive epoch. Keep full query execution separate from preview.",
      "acceptance_criteria": [
        "preview returns count_precision in exact, estimate, or capped states.",
        "Large result previews can return >=N without exact COUNT(*) when configured.",
        "Session browser pagination uses keyset cursor on stable sort keys where possible.",
        "Preview cache invalidates on archive/convergence epoch change.",
        "Composer fixture shows repeated d -> da -> dat prefixes avoid redundant full scans.",
        "Render surfaces label capped/estimated counts honestly."
      ],
      "dependencies": ["p2-18"],
      "parent": null,
      "risks": [
        "Approximate counts can be mistaken for exact.",
        "Cache invalidation can show stale previews.",
        "Keyset pagination may interact with NULL/synthetic sort keys."
      ],
      "confidence": "med",
      "provenance": "Part-2 performance ideas around interactive preview and exact counts."
    },
    {
      "proposed_id": "p2-20",
      "change_kind": "new",
      "target_bead": null,
      "title": "Vector search quantization and ANN spike",
      "issue_type": "task",
      "priority": 2,
      "labels": ["area:embeddings", "area:search", "area:perf", "area:sqlite"],
      "description": "The uploaded source uses exact sqlite-vec MATCH scans and per-seed fanout. Current sqlite-vec offers binary quantization but remains brute-force; SQLite Vec1 now provides ANN via IVFADC/OPQ. Polylogue needs numbers before committing to an ANN direction.",
      "design": "Benchmark exact f32 sqlite-vec, binary/int8 quantization with rerank, origin/session metadata partitioning, multi-probe batching, and SQLite Vec1 as experimental sidecar. Use lineage/fork pairs and relevance judgments as recall labels. Keep SQLite as sole store unless evidence forces a sidecar.",
      "acceptance_criteria": [
        "Benchmark reports latency, memory/index size, and recall@k on fixture plus sampled live archive.",
        "Quantization + rerank results are compared against exact f32 baseline.",
        "Per-seed fanout batching reduces query count for session similarity without recall regression.",
        "Vec1 feasibility is documented with build/dependency implications.",
        "No default migration occurs without an eval-gated recommendation.",
        "Results feed search-eval and ranking_policy_version notes."
      ],
      "dependencies": ["p2-16", "p2-18"],
      "parent": "polylogue-mhx",
      "risks": [
        "ANN recall loss may be invisible without good labels.",
        "Native extension/dependency changes may violate packaging posture.",
        "Live archive benchmark may not fit CI."
      ],
      "confidence": "med",
      "provenance": "Part-2 performance/vector-search prompt plus current sqlite-vec/Vec1 research."
    },
    {
      "proposed_id": "p2-21",
      "change_kind": "new",
      "target_bead": null,
      "title": "Compaction boundary and loss-forensics query unit",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:lineage", "area:compaction", "area:query", "area:forensics"],
      "description": "Compaction and postmortem lifecycle ideas in file 2 converge on making compaction boundaries and lost/transformed evidence queryable, not just summarized in prose.",
      "design": "Represent compaction events as source-grounded units with source session, continuation session, compact summary blob, injected context refs, retained/lost/transformed evidence classes, and degradation flags. Provide query/projection surfaces for post-compaction recovery and loss analysis.",
      "acceptance_criteria": [
        "A Claude Code /compact fixture materializes a compaction event with source and continuation refs.",
        "Loss manifest distinguishes retained, lost, transformed, and unknown evidence classes.",
        "Compaction projections can render injected memories/context and omitted refs.",
        "A follow-up session can query its preceding compaction boundary.",
        "Malformed or partial PreCompact payloads degrade honestly rather than fabricating loss coverage.",
        "No provider-specific scratchpad path is special-cased."
      ],
      "dependencies": ["p2-13"],
      "parent": "polylogue-gjg",
      "risks": [
        "Provider compaction formats may differ widely.",
        "Loss-forensics can overclaim if input snapshot lacks assembled context.",
        "Large compacted blobs may need blob-store treatment."
      ],
      "confidence": "med",
      "provenance": "Part-2 incident/compaction lifecycle ideas plus earlier compaction-event direction."
    },
    {
      "proposed_id": "p2-22",
      "change_kind": "new",
      "target_bead": null,
      "title": "Replay-derived RL/eval export prototype",
      "issue_type": "feature",
      "priority": 3,
      "labels": ["area:eval", "area:replay", "area:export", "area:rl"],
      "description": "File 2 identifies CI-passing sessions and spec cards as potential verifiable reward tasks, but the export target schema and replay safety constraints must be verified before building a full environment.",
      "design": "Prototype a pure read export from spec_cards, actions, commits, and verification runs into a minimal JSONL for external eval/reward tooling. Verify live Atropos/verifiers schema first. Do not execute replay in v1; include start_sha, task text, verify command, outcome evidence tier, and known nondeterminism/safety flags.",
      "acceptance_criteria": [
        "The live target schema is checked and documented before finalizing JSONL.",
        "Export includes task, start_sha, verify_cmd, result evidence, and safety flags without message bodies by default.",
        "Sessions without high-confidence commit attribution are refused or marked insufficient.",
        "At least one fixture export round-trips through the target validator or viewer.",
        "The export clearly labels reward-shaping vs verifiable reward.",
        "No network or filesystem replay is executed in this bead."
      ],
      "dependencies": ["p2-11", "p2-12"],
      "parent": "polylogue-fs1",
      "risks": [
        "External schemas may change.",
        "Recorded tasks may not be safely replayable.",
        "Reward labels can encode tool/environment quirks rather than model ability."
      ],
      "confidence": "low",
      "provenance": "Part-2 RL/eval and spec-card material; intentionally limited to export prototype."
    },
    {
      "proposed_id": "p2-23",
      "change_kind": "update",
      "target_bead": "polylogue-peo",
      "title": "Daemon-death forensics via incident timelines",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:daemon", "area:observability", "area:incident", "area:ops-db"],
      "description": "Incident/postmortem ideas in file 2 should be grounded in concrete daemon lifecycle and telemetry, not a broad resilience epic. The useful unit is an incident timeline that reconstructs what happened around daemon death, restart, stalled convergence, or compaction loss.",
      "design": "Update peo to produce incident timelines from daemon_events, slo_samples, convergence status, query failures, compaction events, and process lifecycle. Render a postmortem bundle with evidence refs, not only logs. Feed recurring incident patterns into candidate caveat assertions via p2-02, never direct injection.",
      "acceptance_criteria": [
        "A fixture daemon crash produces an incident timeline with last healthy event, failure event, restart/absence, and affected operations.",
        "Stalled convergence incidents include idle-vs-stalled verdict from p2-08.",
        "Postmortem bundle links to evidence refs and raw event ids.",
        "Recurring incidents can propose candidate caveat assertions but not auto-accept them.",
        "Incident rendering refuses when coverage is insufficient rather than filling gaps with prose."
      ],
      "dependencies": ["p2-02", "p2-08", "p2-21"],
      "parent": "polylogue-peo",
      "risks": [
        "Incident heuristics can infer causality from temporal adjacency.",
        "Ops.db disposability can erase historical incident context.",
        "Too many caveat candidates can become noise."
      ],
      "confidence": "med",
      "provenance": "Part-2 incident/postmortem/resilience ideas grounded into peo."
    },
    {
      "proposed_id": "p2-24",
      "change_kind": "new",
      "target_bead": null,
      "title": "Ops-tier telemetry schema policy",
      "issue_type": "chore",
      "priority": 1,
      "labels": ["area:ops-db", "area:schema", "area:observability", "area:architecture"],
      "description": "File 2 mixes two ops-tier regimes: explicit ops.db version bump for observability additions and self-healing disposable telemetry tables for read-access/attention. Without a single policy, future telemetry beads will disagree on migrations and reset behavior.",
      "design": "Document and enforce an ops-tier schema policy: optional telemetry tables may self-heal with CREATE TABLE IF NOT EXISTS and retention GC; required contract tables that doctor/CI validates require an OPS_SCHEMA_VERSION bump and disposable reset note. Add a lab policy check or architecture doc section that classifies new ops tables.",
      "acceptance_criteria": [
        "Architecture docs define optional vs required ops-tier table regimes.",
        "A devtools policy check flags new ops tables without an explicit regime annotation.",
        "p2-08 and p2-13 are updated to follow the same policy.",
        "Disposable reset behavior and baseline loss are documented for versioned ops changes.",
        "Tests cover self-healing creation for one optional telemetry table."
      ],
      "dependencies": ["p2-08", "p2-13"],
      "parent": null,
      "risks": [
        "Policy can become ceremony for harmless telemetry.",
        "Self-healing tables can drift if not covered by tests.",
        "Version bumps can erase useful baselines."
      ],
      "confidence": "high",
      "provenance": "Vetting correction: Part-2 observability/read-access specs imply conflicting ops.db regimes."
    }
  ]
}
```
