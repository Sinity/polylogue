## [user 307]
▎ *You have the forked context: the polylogue bundle, the swarm2 daemon/composer/thin-client designs, and our discussion of queries-as-objects, delegation units, and annotation recipes. Ground every claim in the bundle (cite file:line / bead id); tag [evidence] vs [proposal]. Be opinionated — one defended recommendation plus the runner-up you rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate change, (3) the full direction. End with Open questions for the operator and What's missing (a claim unverified, a source unread, a modality not run). The snapshot is dated; live code may have moved — flag assumptions to verify locally. Don't re-derive swarm2.*

Analytical [A] over the bundle:

[A] Demo portfolio (212). Specify 8–16 construct-valid demos as executable prompt-files over the demo corpus, each emitting an identical citable artifact; include the honesty/anti-demo.

## [assistant 317]
bash -lc python3 - <<'PY'
import json, os, textwrap
paths=['/mnt/data/poly_pkg/polylogue/polylogue-beads-export.jsonl','/mnt/data/polylogue_pkg/polylogue/polylogue-beads-export.jsonl']
for p in paths:
 print('path',p,os.path.exists(p))
 if os.path.exists(p):
  issues=[]
  for l in open(p):
   d=json.loads(l)
   if d.get('_type')=='issue': issues.append(d)
  for b in issues:
   if b.get('id')=='polylogue-212' or b.get('id','').endswith('-212') or b.get('id')=='212' or 'demo portfolio' in (b.get('title') or '').lower():
    print('FOUND',b['id'],b.get('title'), b.get('status'), b.get('priority'), b.get('issue_type'))
    for k in ['description','design','acceptance_criteria','notes','labels','external_ref']:
     v=b.get(k)
     if v: print('\n'+k,':\n', v if isinstance(v,(list,dict)) else textwrap.shorten(v.replace('\n',' | '), width=2500, placeholder=' ...'))
PY

## [assistant 319]
bash -lc rm -rf /mnt/data/poly_pkg2 /mnt/data/poly_tree2 && mkdir -p /mnt/data/poly_pkg2 && tar -xzf '/mnt/data/polylogue-all.tar(69).gz' -C /mnt/data/poly_pkg2 && ls -la /mnt/data/poly_pkg2/polylogue | sed -n '1,60p' && python3 - <<'PY'
import json, os, textwrap
p='/mnt/data/poly_pkg2/polylogue/polylogue-beads-export.jsonl'
print('exists',os.path.exists(p))
issues=[]
for l in open(p):
 d=json.loads(l)
 if d.get('_type')=='issue': issues.append(d)
print('issues',len(issues))
for b in issues:
 txt=' '.join(str(b.get(k,'')) for k in ['id','title','description','design','acceptance_criteria','notes'])
 if b.get('id')=='polylogue-212' or 'demo portfolio' in txt.lower():
  print('\nFOUND',b['id'],b.get('title'), b.get('status'), b.get('priority'), b.get('issue_type'))
  for k in ['description','design','acceptance_criteria','notes','labels','external_ref']:
   v=b.get(k)
   if v: print('\n'+k,':\n', v if isinstance(v,(list,dict)) else textwrap.shorten(v.replace('\n',' | '), width=5000, placeholder=' ...'))
PY

## [assistant 321]
bash -lc python3 - <<'PY'
import json, textwrap
issues={}
for l in open('/mnt/data/poly_pkg2/polylogue/polylogue-beads-export.jsonl'):
 d=json.loads(l)
 if d.get('_type')=='issue': issues[d['id']]=d
for id in [f'polylogue-212.{i}' for i in range(1,20)]+['polylogue-4bu','polylogue-cfk','polylogue-3tl.4','polylogue-3tl.11','polylogue-3tl.12','polylogue-sru','polylogue-tf2','polylogue-jxe']:
 b=issues.get(id)
 if not b: continue
 print('\n###',id, b.get('priority'),b.get('status'),b.get('issue_type'),b.get('title'))
 print('parent', b.get('parent'))
 for k in ['description','design','acceptance_criteria','notes','labels','external_ref','depends_on']:
  v=b.get(k)
  if v:
   print('\n'+k+':')
   print(v if isinstance(v,(list,dict)) else textwrap.shorten(v.replace('\n',' | '), width=3000, placeholder=' ...'))
PY

## [assistant 340]
Assumption: I am treating the bundle as a dated 2026-07-05 snapshot. I did not verify live `master`, live `bd`, or a running daemon. Some attached chat excerpts say the bead set reached 391 issues on master, while the packaged `polylogue-beads-export.jsonl` I inspected contains a later/different snapshot; verify locally with `bd stats`, `bd show 212 --json`, and `git rev-parse HEAD` before filing or editing.

## Recommendation

[evidence] Bead `polylogue-212` already defines the right constraint: every demo must run on the deterministic seeded corpus, have a live-archive operator variant, resolve every displayed number to structural evidence, and decompose into product primitives rather than bespoke scripts. It explicitly scopes 212 around D1/D2/D4/D5/D8 plus post-hoc forensic Q&A, while leaving D3/D6/D7 to adjacent campaigns. `polylogue-212`.

[proposal] Make 212 a **prompt-file portfolio**, not a demo shelf. Each demo is an executable `PROMPT.md` given to a coding agent, and every prompt emits the same **Demo Finding Packet**. Start with five “today” prompts, including the anti-demo. Then add near-term prompts only when the missing primitive is implemented. The portfolio should optimize for repeatable citable artifacts, not slick one-off recordings.

Runner-up rejected: a single high-polish “agent recovery” flagship video. [evidence] The situation brief says the closest demonstration is browser-captured agent recovery, but also says the shelf’s current failure is not lack of capability; it is lack of one externalized proof that a stranger can inspect, reproduce, and cite. A video alone does not create the structural spine for 8–16 future demos, and risks repeating the deferral pattern where the scaffold absorbs the launch artifact. situation-brief(2).md:65–69, 135–137, 151–153. fileciteturn1file5

## Uniform artifact: Demo Finding Packet

[evidence] The bundle already has the key output constraints. The finding-publishing bead `polylogue-3tl.4` requires `docs/findings/<slug>/finding.yaml` with five provenance fields: archive cursor id/position, measure/query version, git commit SHA, sample-frame predicate, and run date; it also requires stable `/findings/<slug>/` URLs, refusal to render missing provenance, and changelogged living pages. `polylogue-3tl.4`.

[proposal] Every 212 prompt should emit this exact packet:

```text
.agent/demos/212/<demo_id>/
  PROMPT.md                 # executable instruction file for a coding agent
  finding.yaml              # five-part provenance stanza + structured claim
  report.md                 # human-readable citable report
  evidence.ndjson           # one row per cited evidence ref
  queries.ndjson            # query text + lowered/resolved spec if available
  annotations.ndjson        # optional agent judgments/assertions
  checks.json               # pass/fail, unsupported claims, coverage notes
  run.log                   # command transcript, no private rows in public mode
```

The report format should be identical across demos: claim, corpus, method, findings, specimens, counterexamples, limits, reproduce. The anti-demo emits the same packet with `verdict: not_supported`.

[evidence] This aligns with the current shared payload direction: surface payloads are frozen Pydantic models with JSON serialization, and `SearchEnvelope` is already the canonical ranked-result envelope shared across surfaces. polylogue/surfaces/payloads.py:107–113, 1186–1208. The read algebra already defines `QueryProjectionSpec = SelectionSpec × ProjectionSpec × RenderSpec`, but the code comments say the projection bridge is still a contract builder, not an executor. polylogue/surfaces/projection_spec.py:141–218. fileciteturn1file1

## Layer 1 — with today’s substrate

[evidence] Today’s substrate can already support serious read demos: the query layer has boolean predicates, structural units, sequence, pipeline stages, terminal actions, and render formats; the current limitation is that projection is still mostly named views rather than fully composable projection algebra. SWARM_BRIEF.md:9–34. The action view already joins tool-use blocks to tool-result blocks and exposes `tool_name`, command/path/input, `is_error`, and `exit_code`; session tables also carry git branch/repo/commit fields, and repo/commit correlation tables exist. polylogue/storage/sqlite/archive_tiers/index.py:39–82, 324–343, 470–519. fileciteturn1file3

1. `demos/212/d01_receipts_claim_vs_evidence/PROMPT.md`

[evidence] This is bead `polylogue-212.2`: “D1 ‘The receipts’: claim-vs-evidence on a real PR.” It resolves PR → authoring session through `session_commits/session_repos`, renders PR-body claims beside observed action rows, and drills to raw tool-result blocks. `polylogue-212.2`.

[proposal] Prompt claim: “For one merged agent-authored PR, show whether the PR’s claimed verification matches the tool outcomes.” The prompt should select one seeded PR fixture, resolve it to the session, extract claim sentences, extract verification/tool rows, and emit a packet with a two-column table plus drill refs. This should be the first public demo because it directly expresses the wedge: structured tool outcomes versus assistant claims. situation-brief(2).md:57–61.

2. `demos/212/d02_behavioral_archaeology/PROMPT.md`

[evidence] This is bead `polylogue-212.4`: six DSL queries, including SEQ thrash-loop, failure-rate by model, tool breakage by observed-event outcome, semantic probe across providers, abandoned-this-repo-this-quarter, and a query piped into `read`; it must show `explain_query_expression` once. `polylogue-212.4`.

[proposal] Prompt claim: “Polylogue can answer engineering-lead questions that normal chat UIs cannot because it has a queryable archive.” Keep it rapid-fire and citable: six sections, one query each, one output table each, one `explain` block showing the query’s parsed meaning.

3. `demos/212/d03_posthoc_forensic_qa/PROMPT.md`

[evidence] This is bead `polylogue-212.1`: post-hoc forensic Q&A over a completed multi-hour coding-agent session, answering when a bad assumption entered, which file churned before a regression, what evidence was cited for a design choice, and which prior failures resemble the present one; it must include one explicit “we cannot answer X” slide. `polylogue-212.1`.

[proposal] Prompt claim: “A completed session becomes interrogable after the fact.” The output should be a Q&A report where every answer cites session/message/action/commit refs; one question must intentionally fail if evidence is absent.

4. `demos/212/d04_method_trace_swarm_to_beads/PROMPT.md`

[evidence] The situation brief says the process story is the safest, least contested inbound story, because the rare asset is the disciplined agentic-engineering method, not merely “uses coding agents.” situation-brief(2).md:90–99. The attached session shows a large swarm auditing and restructuring the bead set; the later excerpt reports the external-legibility/demo cluster and states that 212 is the demonstrable-evidence layer. Pasted text(34).txt:1991–2006. fileciteturn1file12

[proposal] Prompt claim: “Polylogue plus Beads makes agentic engineering auditable as a process.” The prompt should use the bundle corpus to reconstruct one swarm-to-beads episode: initial directive, subagent launches, findings, applied bead mutations, verification, and final state. This should not be a self-congratulatory velocity demo; it should show mistakes, gates, held changes, and verification.

5. `demos/212/d05_honesty_anti_demo_no_evidence_no_number/PROMPT.md`

[evidence] The situation brief explicitly praises a negative result: “do not force the multi-source demo yet, the upstream data quality is not there,” and says the trap is that every deferral reason is locally correct but the demo remains one step away. situation-brief(2).md:65–69. The 3tl.4 publishing lane is supposed to refuse findings without the provenance stanza. `polylogue-3tl.4`.

[proposal] Prompt claim: “Polylogue refuses to publish unsupported numbers.” The prompt should attempt a tempting finding, such as “multi-source reconstruction proves what the operator was doing minute-by-minute,” and then emit a `verdict: not_supported` packet with missing modalities, missing refs, and the exact query/evidence gap. This is the required honesty/anti-demo. It should be shipped beside the successful demos, not hidden.

## Layer 2 — with a near-term substrate change

[evidence] The near-term substrate change needed here is not the full daemon/composer stack; it is a small set of object types and import surfaces: query/result objects, annotation batches over assertions, and a common packet runner. Assertions are already close: `user.db` has a unified `assertions` table with `scope_ref`, `target_ref`, `kind`, `value_json`, `body_text`, `author_ref`, `author_kind`, `evidence_refs_json`, status, visibility, confidence, staleness, context policy, supersedes, and timestamps. polylogue/storage/sqlite/archive_tiers/user.py:7–31. The enum already includes `annotation`, `saved_query`, `recall_pack`, `judgment`, `run_state`, `prompt_eval`, `transform_candidate`, and `pathology`; assertion statuses include candidate/accepted/rejected/deferred/superseded. polylogue/core/enums.py:399–447. Upsert already accepts target/scope/kind/value/body/author/evidence/status/visibility/confidence/context policy. polylogue/storage/sqlite/archive_tiers/user_write.py:901–999.

6. `demos/212/d06_cost_by_outcome/PROMPT.md`

[evidence] This is bead `polylogue-212.3`: “D2 ‘Where did the money actually go’: cost by outcome.” It wants provider-reported exact versus catalog-priced cost basis, then cost by outcome: monthly spend, percent in abandoned/failing sessions, and five most expensive failures, drillable to exact turns. It explicitly needs the outcome-conditioned join. `polylogue-212.3`.

[proposal] Prompt claim: “The important cost report is not total spend; it is spend by outcome.” Near-term substrate requirement: finish the action-outcome join and cost-honesty lane, then the prompt runs as a normal packet.

7. `demos/212/d07_resume_triage_to_next_action/PROMPT.md`

[evidence] This is bead `polylogue-212.6`: “D8 ‘Pick up where I left off’: abandoned-session triage to live continuation.” It chains `find_abandoned_sessions → get_resume_brief → resume routing`, and explicitly says this proves the archive changes what you do next. `polylogue-212.6`.

[proposal] Prompt claim: “The archive changes the next action, not just the postmortem.” Near-term substrate requirement: make the resume ranking honest, exclude known-dead scorer paths until fixed, and in public mode end at a printed continuation command if live harness routing is not available.

8. `demos/212/d08_self_watching_live_capture/PROMPT.md`

[evidence] This is bead `polylogue-212.5`: “D5 ‘The session that watched itself’: live capture proof.” It queries the archive for the current session mid-session, uses ingest-cursor timestamps for capture latency, and ends by generating the session’s own postmortem. `polylogue-212.5`.

[proposal] Prompt claim: “A session can become queryable while it is still being created.” Near-term substrate requirement: a seeded/public variant needs synthetic live-capture fixtures or a deterministic daemon run. The live operator variant can use the actual daemon.

9. `demos/212/d09_external_annotation_loop/PROMPT.md`

[evidence] Assertions are already queryable as structural fields: assertion field metadata includes author, author kind/ref, body, context, evidence, key, kind, scope/scope_ref, status, target/target_ref, value, visibility, and timestamps. polylogue/archive/query/metadata.py:507–525. The API exposes candidate assertion listing and judgment methods. polylogue/api/archive.py:2079–2206.

[proposal] Prompt claim: “An external agent can annotate sessions/messages, import the judgments, and then query the annotations as evidence-adjacent data.” The prompt should export a small evidence pack, ask the coding agent to classify rows using a fixed JSONL schema, import them as candidate assertions, and produce aggregates plus counterexamples. This is the scalable version of annotation recipes.

10. `demos/212/d10_delegation_rhetoric/PROMPT.md`

[evidence] The bundle’s current query unit vocabulary includes `message`, `action`, `block`, `assertion`, `file`, `run`, `observed-event`, and `context-snapshot`; it does not list a first-class `delegation` unit in the shared read-algebra baseline. SWARM_BRIEF.md:11–18.

[proposal] Prompt claim: “Visible orchestrator-to-subagent delegation can be extracted, classified, and cited.” Near-term substrate requirement: either a temporary annotation recipe over messages/actions, or preferably a first-class `delegation` unit later. The prompt should find candidate delegation instructions, annotate directive intensity/output-contract strictness/control devices, import judgments, and render specimens. This generalizes the Fable-hook idea without making Fable a product silo.

11. `demos/212/d11_composer_relation_preview/PROMPT.md`

[evidence] The composer design says terminal stages preview bounded projections; set algebra previews should show left count, operand count, intersection count, and grain errors before commit; every commit writes a recall entry with query text, resolved spec, result fingerprint, and timestamp. C10_composer_ux.md:142–205. fileciteturn1file13

[proposal] Prompt claim: “Polylogue’s DSL is inspectable while it is being composed.” Near-term substrate requirement: `preview` and `complete` endpoints. The public packet can be generated from a scripted composer trace: keystrokes, preview envelopes, final query run, and rendered result. This is more product-demo than analysis-demo, so it should come after D1/D4/anti-demo.

## Layer 3 — full direction

[evidence] The full direction depends on the hot-daemon/thin-client stack. The swarm brief frames the target as a resident memory-hungry daemon, thin UDS client, no substrate work in CLI/TUI, and live `complete(partial)` / `preview(spec)` in single-digit milliseconds. SWARM_BRIEF.md:47–59. B8 proposes a seven-method service surface: `query`, `read`, `preview`, `complete`, `act`, `status`, and `facets`; `read` should execute `ReadRequest = selection + projection + render`, and `preview`/`complete` are the composer’s reason to require the daemon. B8_contract.md:134–168, 188–219. fileciteturn1file11

12. `demos/212/d12_work_reconstruction_repo_trace/PROMPT.md`

[evidence] The current archive has session git fields, action/tool outcome fields, and session↔repo↔commit correlation tables. polylogue/storage/sqlite/archive_tiers/index.py:39–82, 324–343, 470–519.

[proposal] Prompt claim: “Polylogue can reconstruct the work trace of a coding-agent session across messages, tool calls, files, and commits.” Full substrate requirement: artifact graph and work-trace projection. Do not call it deterministic replay yet. The packet should say “reconstruction” or “reproduction plan,” and it should mark any missing file snapshots/env/state as non-replayable.

13. `demos/212/d13_context_compile_after_compaction/PROMPT.md`

[evidence] The attached Claude Code excerpt shows `/compact` followed by a flat `bd prime --memories-only` injection of 19 persistent memories, including automagic invariants, Beads workflow, demo-regeneration profiling, and live-archive pitfalls. Pasted text(35).txt:28–65. fileciteturn1file4

[proposal] Prompt claim: “A compiled Polylogue context pack can be more relevant and auditable than a flat memory dump.” Full substrate requirement: compaction-boundary units, context-compile runs, selected/omitted memory refs, and outcome tracking. The anti-overclaim guard: the demo should compare what was included/omitted and why; it should not claim improved agent performance unless run as a paired experiment.

14. `demos/212/d14_query_objects_analysis_dag/PROMPT.md`

[evidence] C10 already proposes recall entries with query text, resolved spec, result fingerprint, timestamp, macros, and `@` completion. C10_composer_ux.md:190–205. B8’s service contract would make `query/read/preview/complete/act/status/facets` a single substrate boundary. B8_contract.md:134–168.

[proposal] Prompt claim: “The analysis itself is queryable.” Full substrate requirement: promote recall entries into `query_definition`, `query_run`, `result_relation`, `annotation_batch`, and `analysis_run` objects. The prompt should run a multi-step analysis, store each query/result/annotation import as objects, then generate a report that cites those objects.

## Defended implementation order

[proposal] Implement the portfolio in this order:

First ship **D1 Receipts**, **D4 Behavioral Archaeology**, and **D5 Anti-demo** as a three-packet public mini-portfolio. D1 is the wedge; D4 proves breadth of query; D5 proves honesty. This directly matches the situation brief’s warning that the next artifact matters more than more substrate. situation-brief(2).md:77–81, 135–137.

Second add **D3 Post-hoc Forensic Q&A** and **D4 Method Trace** because they use today’s data and strengthen the “system of record for AI work” story without needing daemon/composer work.

Third, only after the packet runner and assertion import are in place, add **Cost by Outcome**, **Resume Triage**, **Annotation Loop**, and **Delegation Rhetoric**.

Fourth, leave **Work Reconstruction**, **Context Compile**, and **Query Objects Analysis DAG** as full-direction demos. They are valuable, but making them first would recreate the exact deferral pattern the brief warns about. situation-brief(2).md:103–113.

## What this implies for bead 212

[proposal] Update `polylogue-212` from “portfolio of named demos” to “portfolio contract plus children.” The epic should own:

A packet schema child: `212.packet`.

A prompt runner child: `212.runner`.

A public corpus verification child: `212.seed-1843-corpus`.

The existing 212.1–212.6 children.

A new anti-demo child: `212.anti`.

A new annotation-loop child after assertion batch import exists.

A “demo registry manifest” child that lists every prompt file, expected packet path, public/private mode, and required primitives.

[evidence] This stays inside the bead’s existing compositionality rule: shell/python is glue; product primitives own the logic. `polylogue-212`.

## Open questions for the operator

1. Should the first public portfolio be three packets, five packets, or one packet? My recommendation is three: D1 + D4 + anti-demo.

2. Should anti-demo be framed as “no unsupported number is published” or as “multi-source reconstruction is not ready”? The former is more general and less self-defeating; the latter is more concrete.

3. Should `finding.yaml` live only under `docs/findings/<slug>/`, or should `.agent/demos/212/<demo>/finding.yaml` be the source and docs be a rendered projection?

4. Do you want the live-archive variants in the same packet format but private, or should public seeded packets and private operator packets be separate artifact classes?

5. For delegation rhetoric, should the first version be annotation-only over messages/actions, or should it wait for a true `delegation` unit?

## What’s missing

I did not run the demo corpus, the daemon, the CLI, `bd`, or tests. I inspected the uploaded bundle and extracted code/package files only.

I did not verify whether live `master` has already implemented any of the near-term primitives, especially assertion batch import, query-run recording, or a newer 212 state.

I did not inspect every 212 child through live `bd show`; I read the packaged `polylogue-beads-export.jsonl`. Bead notes sidecars can carry important content, and the Claude memory explicitly warns about that trap. Pasted text(35).txt:52–59.

I did not inspect rendered demo artifacts, visual tapes, or docs site output. The modality not run is the actual artifact-generation path.

One claim remains unverified: that seeded corpus `seed 1843` currently contains enough fixtures for every “today” prompt. Bead 212 requires it, but the prompt portfolio should begin by adding a corpus coverage check.
