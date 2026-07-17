## [user 307]
▎ *You have the forked context: the polylogue bundle, the swarm2 daemon/composer/thin-client designs, and our discussion of queries-as-objects, delegation units, and annotation recipes. Ground every claim in the bundle (cite file:line / bead id); tag [evidence] vs [proposal]. Be opinionated — one defended recommendation plus the runner-up you rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate change, (3) the full direction. End with Open questions for the operator and What's missing (a claim unverified, a source unread, a modality not run). The snapshot is dated; live code may have moved — flag assumptions to verify locally. Don't re-derive swarm2.*

Analytical [A] over the bundle:
 [A] Delegation graph as substrate. Design the delegation unit + extraction from Claude-Code/Codex subagent structure + artifact/scratchpad linking + delegation-card projection + analytics DSL. Generalize the Fable demo.

## [assistant 319]
bash -lc cd /mnt/data/polyA_tree/polylogue
printf '--- source raw artifacts ---\n'
nl -ba polylogue/storage/sqlite/archive_tiers/source.py | sed -n '1,180p'
printf '\n--- index actions sessions repos ---\n'
nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '1,115p'
nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '300,360p'
nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '455,525p'
nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '900,1045p'
printf '\n--- user assertions schema ---\n'
nl -ba polylogue/storage/sqlite/archive_tiers/user.py | sed -n '1,60p'
printf '\n--- assertion kinds ---\n'
nl -ba polylogue/core/enums.py | sed -n '395,450p'
printf '\n--- query metadata unit kinds ---\n'
nl -ba polylogue/archive/query/metadata.py | sed -n '470,545p'
nl -ba polylogue/archive/query/metadata.py | sed -n '780,840p'
printf '\n--- projection spec ---\n'
nl -ba polylogue/surfaces/projection_spec.py | sed -n '1,230p'

## [assistant 355]
bash -lc cd /mnt/data/polyA_pkg/polylogue
python3 - <<'PY'
import json, textwrap
issues=[]
for l in open('polylogue-beads-export.jsonl'):
    d=json.loads(l)
    if d.get('_type')=='issue': issues.append(d)
terms=['subagent','delegation','artifact','scratch','report','agent','coordination','context','run','assertion','query object','recall','cohort','analysis','work-evidence','compaction','repo','commit']
for term in terms:
    hits=[]
    for b in issues:
        txt='\n'.join(str(b.get(k,'')) for k in ['id','title','description','design','acceptance_criteria','notes','labels'])
        if term.lower() in txt.lower(): hits.append(b)
    print('\nTERM',term,'hits',len(hits))
    for b in hits[:18]:
        print(b.get('id'), 'P'+str(b.get('priority')), b.get('status'), b.get('issue_type'), b.get('title'))
PY

## [assistant 358]
bash -lc cd /mnt/data/polyA_pkg/polylogue
python3 - <<'PY'
import json, textwrap
issues={}
for l in open('polylogue-beads-export.jsonl'):
    d=json.loads(l)
    if d.get('_type')=='issue': issues[d['id']]=d
for id in ['polylogue-s7ae','polylogue-s7ae.1','polylogue-s7ae.4','polylogue-s7ae.5','polylogue-s7ae.6','polylogue-37t','polylogue-37t.11','polylogue-37t.12','polylogue-fnm','polylogue-fnm.13','polylogue-4p1','polylogue-t46','polylogue-4ts.3','polylogue-gjg','polylogue-x4s','polylogue-7fj','polylogue-fs1','polylogue-fs1.2','polylogue-9l5','polylogue-9l5.7','polylogue-stc','polylogue-3tl.11','polylogue-3tl']:
    b=issues.get(id)
    if not b: continue
    print('\n###',id,b.get('priority'),b.get('status'),b.get('issue_type'),b.get('title'))
    for field in ['description','design','acceptance_criteria','notes']:
        v=b.get(field)
        if v:
            print(field.upper()+':')
            print(textwrap.shorten(v.replace('\n',' | '), width=1800, placeholder=' ...'))
PY

## [assistant 379]
According to the 2026-07-05 bundle snapshot, the right answer is: **Polylogue should add a first-class `delegation` derived unit now, not wait for the full work-replay graph.** The current substrate already sees enough Claude-Code/Codex subagent structure to extract high-value delegation edges, and the existing query/projection/assertion machinery is close enough that this can be made generic rather than a Fable silo. The runner-up is “represent delegation only as assertions over messages/actions.” I reject that because it keeps the central relation invisible to the DSL, projections, set algebra, previews, and analytics; assertions should annotate delegations, not be the delegation graph itself.

I am grounding this in the bundle snapshot; live code and live Beads may have moved after July 5, 2026, so all code/file claims below should be verified locally with `rg`, `bd show`, and current migrations before implementation.

## Recommendation

[evidence] The bundle’s read frame is already relation-shaped: `Query × Projection × Render`, with structural query units including `message`, `action`, `block`, `assertion`, `file`, `run`, `observed-event`, and `context-snapshot`; the pipeline is split on `|`; terminal actions include `read`, `analyze`, `select`, `mark`, `delete`, and `continue`. The same brief says projection is still closed-view-driven and should move toward composable projection/macros. fileciteturn2file3 Code agrees: `QueryUnitName` is declared in `polylogue/archive/query/metadata.py:8-17`, and `QueryProjectionSpec = SelectionSpec × ProjectionSpec × RenderSpec` is in `polylogue/surfaces/projection_spec.py:141-146`.

[proposal] Add `delegation` as the next derived query unit, with a materialized relation in `index.db` or a derived read model backed by existing blocks/runs/topology/artifacts. Then add a `delegation-card` projection preset over that unit, and let external agents annotate delegation rows through the existing assertion/judgment substrate. This is the narrowest substrate change that makes the Fable demo, Codex/Claude comparison, multi-agent work audits, and future work-replay all easier.

[evidence] This is not inventing a new direction. The Beads/system frame already names `s7ae` as the agent coordination substrate: evidence-backed multi-agent work over sessions, topology, repos/worktrees, work items, activity/resource episodes, context injection, messages, handoffs, proof evidence, and no Beads lock-in. The bundle also says `37t.11` is the context scheduler, `37t.12` is the judgment queue, `fnm.13` is set algebra, `t46` is contracts-own-surfaces, and `4p1` is the one read algebra. Bead IDs: `polylogue-s7ae`, `polylogue-s7ae.4`, `polylogue-s7ae.5`, `polylogue-37t.11`, `polylogue-37t.12`, `polylogue-fnm.13`, `polylogue-t46`, `polylogue-4p1`.

[proposal] The defended path is therefore: **derived `delegation` unit first; generic artifact/evidence edges second; annotation batches third; full analysis/work-trace graph later.** This gives a publishable, useful demo without blocking on the entire “replay work from git/tool calls” vision.

## Runner-up rejected

[proposal] Runner-up: keep delegation as an external `annotations.jsonl` / assertions-only workflow, where an agent scans sessions, labels messages/actions, imports judgments, and reports over assertion rows.

[evidence] The assertion substrate is strong enough to support that: `user.db.assertions` already has `target_ref`, `scope_ref`, `key`, `kind`, `value_json`, `body_text`, `author_ref`, `author_kind`, `evidence_refs_json`, `status`, `visibility`, `confidence`, `staleness_json`, and `context_policy_json` (`polylogue/storage/sqlite/archive_tiers/user.py:7-31`). Assertion kinds include `annotation`, `saved_query`, `recall_pack`, `decision`, `lesson`, `handoff`, `judgment`, `run_state`, `prompt_eval`, `transform_candidate`, and `pathology` (`polylogue/core/enums.py:399-428`). The API already exposes listing and judging assertion candidates (`polylogue/api/archive.py:2079-2128`, `2181-2206`), and B8 explicitly names `save_annotation`, `record_correction`, `judge_assertion_candidate`, saved views, workspaces, recall packs, and blackboard as real facade methods not yet unified under the cross-surface contract. fileciteturn2file9

[proposal] I reject assertions-only because it makes the most important noun—delegation—implicit. You would constantly query “actions where tool:Task plus assertions where key:directive_style plus maybe runs where role:subagent,” then reconstruct the graph in scripts. That defeats the point of the DSL/composer direction. Assertions should say “this delegation is stern / failed / under-specified / used by parent,” not define what a delegation is.

## Layer 1 — with today’s substrate

[evidence] Today, Polylogue can already see much of the delegation structure, especially for Claude Code and Codex. Claude Code’s parser treats `agent-*` session files as child sessions, but special-cases `agent-acompact-*` as continuation/compaction rather than real subagent work (`polylogue/sources/parsers/claude/code_parser.py:432-451`). Codex parsing records that `source.subagent.thread_spawn` is positive evidence for subagent lineage, while `forked_from_id` alone proves a parent but not the relationship type; only `source.subagent.thread_spawn` assigns `BranchType.SUBAGENT` (`polylogue/sources/parsers/codex.py:592-599`, `821-837`).

[evidence] The insight layer already extracts “subagent reports.” `SubagentReport` has `subagent_type`, `tool_id`, `task_id`, `child_session_id`, link status/type, prompt preview, final report preview, PR refs, issue refs, test evidence, caveats, and raw refs (`polylogue/insights/transforms.py:214-228`). Extraction pairs `tool_use` with `tool_result` by `tool_id`, recognizes subagent tools, captures prompt/result previews, and records raw refs (`polylogue/insights/transforms.py:1618-1655`). It also enriches reports with child-session links where topology/session links can resolve them (`polylogue/insights/transforms.py:1658-1690`). The detector recognizes a Claude-style `Task` tool or tool inputs carrying `subagent_type` / `agent_type` (`polylogue/insights/transforms.py:2072-2116`).

[evidence] The lower pipeline also already marks subagent spawns: `extract_tool_invocations` sets `is_subagent = tool_name == "Task"` (`polylogue/pipeline/semantic_capture.py:125-138`), and `extract_subagent_spawns` emits agent type, prompt, description, and `run_in_background` from `Task` invocations (`polylogue/pipeline/semantic_capture.py:193-208`).

[evidence] Structured tool evidence is already strong enough for serious analytics: `blocks` store `tool_name`, `tool_id`, `tool_input`, `tool_result_is_error`, `tool_result_exit_code`, generated `tool_command`, generated `tool_path`, and `search_text` (`polylogue/storage/sqlite/archive_tiers/index.py:182-208`). The `actions` view pairs tool use and tool result blocks and exposes `tool_command`, `tool_path`, output text, `is_error`, and `exit_code` (`index.py:324-343`). The code comments explicitly frame tool-result error/exit fields as the anchor for structured outcome reports rather than prose guesses (`index.py:217-224`; `polylogue/insights/transforms.py:1709-1717`).

[evidence] Runtime/run structure also exists. `session_runs` includes `parent_run_ref`, `agent_ref`, `context_snapshot_ref`, provider origin, harness, `role` constrained to `main|subagent`, status, confidence, cwd, git branch, lineage refs, evidence refs, transcript ref, payload, and search text (`polylogue/storage/sqlite/archive_tiers/index.py:913-938`). `session_observed_events` and `session_context_snapshots` add event and context-boundary units tied to runs (`index.py:952-1024`).

[evidence] Repo/commit correlation is partially present. Sessions carry `git_branch`, `git_repository_url`, and `commit_hash` (`index.py:51-55`); `repos`, `session_repos`, and `session_commits` model repo roots, branch names, commit SHAs, detection types (`time_window`, `file_overlap`, `explicit_ref`, `origin_reported`), method, confidence, and evidence JSON (`index.py:477-520`). This supports the earlier raw-log idea of correlating logs with commits/issues/PRs. fileciteturn1file17

[evidence] Artifact observation exists but is too source-tier/local today. `raw_artifacts` records `artifact_id`, `raw_id`, origin, `source_path`, `source_index`, `artifact_kind`, support status, classification reason, parse eligibility, malformed/decode state, `cohort_id`, `link_group_key`, `sidecar_agent_type`, and first/last observed times (`polylogue/storage/sqlite/archive_tiers/source.py:84-102`). This is enough to see sidecar/report-like artifacts, but not enough to query “artifact produced by delegation D and consumed by parent session P.”

[evidence] Current query units do not include `delegation`. The registry has `action`, `block`, `file`, `message`, `assertion`, `run`, `observed-event`, and `context-snapshot`; examples already cover `exists file(...)`, `runs where ...`, and `observed-events where ...`, but there is no delegation source (`polylogue/archive/query/metadata.py:692-728`). The query field metadata does include action fields such as `command`, `exit_code`, `followup_class`, `is_error`, `output`, `path`, `tool`, and `type` (`metadata.py:475-505`), and assertion fields such as `author_kind`, `evidence`, `key`, `kind`, `scope_ref`, `target_ref`, `status`, and `value` (`metadata.py:507-525`).

[proposal] With today’s substrate, a workable demo can be built as an external script/agent prompt:

`query candidate parent sessions → inspect actions/tool_use Task blocks → resolve child session/report refs via existing SubagentReport/session topology → export evidence windows → annotate rows externally → import annotations as assertions → write report`.

[proposal] The demo should avoid claiming “the delegation graph” as a built-in Polylogue feature. The honest present-tense claim is: **Polylogue can mine a candidate delegation inventory from existing subagent/report/runtime evidence, then use assertions to layer external judgments over it.**

[proposal] Example today-ish analysis flow:

```text
1. Find candidate sessions:
   sessions with action:subagent OR action tool:Task OR runs role:subagent.

2. For each candidate:
   read the Task/tool_use block, paired tool_result, SubagentReport, child session link,
   parent follow-up window, artifact paths mentioned in prompt/result.

3. Produce delegations.jsonl externally:
   one row per candidate delegation edge, with evidence_refs to session/message/block/run/artifact refs.

4. Ask an agent to annotate:
   directive_intensity, output_contract_strictness, scope_control, tone, control_devices, result_status.

5. Import as assertions:
   kind=judgment or annotation, key=directive_style_v1.*, target_ref=<block/session ref>,
   scope_ref=analysis:<id>, evidence_refs_json=[...].
```

[evidence] This “external annotation loop” fits the existing assertion model and the `37t` memory loop: `37t` explicitly describes agents declaring claims, the operator judging them, active claims compiling into preambles, and reboot/handoff through evidence packs; `37t.12` exists to make judgment queue bulk review/accept/reject of candidate assertions an owned feature. Bead IDs: `polylogue-37t`, `polylogue-37t.12`.

[proposal] Today’s ceiling: this will be useful but fiddly. The agent/script has to rebuild a graph from actions, reports, sessions, and assertions each time. Projection is manual. The DSL cannot say `delegations where ...`. The result is good enough for one demo, not good enough as a substrate.

## Layer 2 — near-term substrate change

[proposal] Add a derived `delegations` table/read model and make `delegation` a query unit. This should be a derived/index-tier model, not durable source evidence. Raw messages/tool blocks/session files stay source truth; delegation rows are materialized, recomputable claims with extractor version and confidence.

[proposal] Minimal `delegations` row:

```text
delegation_id                 stable deterministic id
parent_session_id             session containing the delegation act
parent_message_id             message containing delegation tool_use or instruction
parent_run_ref                run ref if known
tool_use_block_id             instruction/call block
tool_result_block_id          result block if paired
child_session_id              resolved child session if any
child_run_ref                 resolved child run if any
task_id                       taskId/task_id if present
tool_id                       tool id if present
delegation_kind               subagent | background-agent | sidecar-report | async-task | unknown
harness                       claude-code | codex | gemini-cli | hermes | unknown
subagent_type                 researcher | implementer | general-purpose | ...
subagent_model                if visible
subagent_model_family         if normalized
description                   short description field if present
instruction_preview           bounded preview, not canonical evidence
result_preview                bounded preview, not canonical evidence
status                        spawned | completed | failed | stopped | unresolved | unknown
link_status                   resolved | unresolved | inferred | quarantined
confidence                    raw | inferred plus numeric if useful
evidence_refs_json            block/session/run/artifact refs
artifact_refs_json            generic artifact refs
created_at_ms / updated_at_ms
materializer_version
payload_json                  provider-specific residue
search_text
```

[proposal] Stable identity should prefer `(parent_session_id, tool_use_block_id)` for tool-based delegations. If there is no tool block, use a deterministic hash over parent session + message + child session + instruction ref. Do not use natural-language prompt text alone; identical prompts can be different delegations.

[evidence] This schema mirrors existing patterns. `session_runs` already carries parent/agent/context/evidence fields (`index.py:913-938`). `session_observed_events` already carries subject/object/evidence/payload/search fields (`index.py:952-968`). `SubagentReport` already carries tool/task/child/prompt/report/test/caveat/raw refs (`transforms.py:214-228`). Therefore the new table is a normalization of existing concepts, not a foreign ontology.

[proposal] Extraction rules:

For Claude Code, high-confidence delegation comes from a `Task` tool use or tool input containing `subagent_type` / `agent_type`; pair it with the `tool_result` block via `tool_id`, use `_subagent_child_session_id` extraction when present, and link to `agent-*` child sessions unless the child is `agent-acompact-*`, which must be typed as compaction/continuation, not delegation. Evidence: Claude parser special-cases `agent-acompact-*` as `BranchType.CONTINUATION` and real `agent-*` as `BranchType.SUBAGENT` (`code_parser.py:432-451`); transform extraction recognizes Task/subagent input and child session IDs (`transforms.py:2072-2116`).

For Codex, high-confidence delegation requires `source.subagent.thread_spawn`; `forked_from_id` alone should create a parent/topology relation but not `delegation_kind=subagent`. Evidence: Codex parser says exactly that `source.subagent.thread_spawn` is positive subagent evidence, while `forked_from_id` alone proves parentage but not relationship type (`codex.py:821-837`).

For provider-neutral/local-agent structure, use `session_runs.role = subagent`, `parent_run_ref`, `native_parent_session_id`, and `agent_ref` as candidate evidence. Evidence: `session_runs.role` is constrained to `main|subagent`, and the row carries parent/agent/evidence fields (`index.py:913-938`).

For artifact linking, do not special-case `.agent/scratch`. Use generic `artifact_ref` edges with edge types: `produced`, `consumed`, `mentioned`, `reported_by`, `derived_from`, `evidence_for`. Raw source already has `source_path`, `artifact_kind`, `link_group_key`, and `sidecar_agent_type` (`source.py:84-102`); the missing piece is a derived graph edge from delegation/session/action/run/query/analysis to artifact.

[proposal] Add two derived relations:

```text
delegation_artifact_edges(
  delegation_id,
  artifact_ref,
  edge_type,        -- produced|consumed|mentioned|reported_by|derived_from|evidence_for
  evidence_refs_json,
  confidence,
  extractor_version
)

artifact_observations_public(
  artifact_ref,
  source_path,
  artifact_kind,
  content_hash/blob_ref if available,
  repo_ref,
  commit_ref,
  first_observed_at_ms,
  last_observed_at_ms
)
```

[proposal] Then add the DSL/source:

```text
delegations where parent.repo:polylogue and harness:claude-code
delegations where subagent_type:researcher and status:failed
delegations where instruction.text:"do not edit" | group by subagent_model_family | count
delegations where artifact.kind:markdown and artifact.path:swarm2 | read delegation-card
delegations where parent.model:Fable | except (delegations where child.family:claude)
```

[evidence] This fits the existing query model because unit query execution already centralizes terminal unit-source requests (`QueryUnitRequest`) and resolves unit descriptors into payload models (`polylogue/archive/query/unit_results.py:57-79`, `158-184`). It also fits C10’s composer/set-preview design: set operations are intended to show operand sizes and grain mismatch before commit, and recall/macros are intended to make composed sets reusable. fileciteturn2file0

[proposal] Add `DelegationQueryRowPayload` and register `delegation` in `QueryUnitName`, structural field metadata, terminal source list, and unit-row execution. First fields:

```text
parent, parent_session, parent_run, child, child_session, child_run
harness, parent_model, child_model, child_family
subagent_type, task_id, tool_id, description
instruction, result, status, link_status, confidence
artifact, artifact_kind, artifact_path
repo, branch, commit, time
annotation_key, annotation_value   -- joins assertions later, optional
```

[proposal] The `delegation-card` projection should render:

```text
header: delegation id, harness, parent→child, status, confidence
instruction: full or bounded instruction/tool_use prompt
parent context window: N messages before/after delegation
child output: child report/result/session summary
parent use: messages/actions after result where parent consumed/synthesized it
artifacts: linked report/files/commits
annotations: assertions/judgments over the delegation
provenance: exact evidence refs, extractor version, caveats
```

[evidence] Current projection families include `sessions`, `messages`, `blocks`, `actions`, `raw`, `context`, `chronicle`, `neighbors`, `correlation`, `temporal`, and `assertions`, with body policies `full`, `omit-tool-outputs`, `authored-dialogue`, and `metadata-only` (`projection_spec.py:18-40`). Named views are mapped to family bundles, and the file explicitly says this is a contract builder, not executor (`projection_spec.py:149-218`). This is why `delegation-card` should start as a server-side preset but be designed as a projection program, not a permanent special view.

[proposal] For external annotations, add `target_kind=delegation` to user-state target registries and allow assertions to target `delegation:<id>`. The current user-state target registry supports session, message, work_event, thread, block, attachment, and paste_span, but not delegation, query, artifact, or analysis (`polylogue/core/user_state_targets.py:29-118`). Even though raw `target_ref` is a free text field in `assertions`, surface affordances, recall packs, workspaces, and validation need a registered kind.

[proposal] Near-term implementation sequence:

1. `delegation` derived model/materializer over existing subagent reports, Task/action blocks, session_runs, and session topology.
2. `delegation` query unit + `DelegationQueryRowPayload`.
3. Generic artifact edge model; use raw artifacts as source, not `.agent/scratch` as a special case.
4. `target_kind=delegation` and annotation-batch import into assertions.
5. `delegation-card` projection preset and evidence-pack renderer.
6. Minimal demo prompt that runs the loop and produces `cohort.json`, `annotations.jsonl`, and `report.md`.

## Layer 3 — full direction

[proposal] The full direction is a **work graph**, not only a delegation graph:

```text
session
  -> message
  -> block/action
  -> run
  -> delegation
  -> artifact
  -> repo file / commit / PR / bead / issue
  -> query_run / result_relation / cohort
  -> annotation_batch / assertion
  -> analysis_run / report
```

[evidence] The bundle already argues for this broader framing: Polylogue is characterized as a “system of record for AI work,” with durable source/user tiers, derived index/embedding tiers, disposable ops tier, and program epics around coordination, content variants, lineage truth, attachment/blob integrity, scale-hardening, and external legibility. fileciteturn2file1 The situation brief says the strategic gate is an externalized proof that an agent/operator does a real task better with the substrate, not more internal machinery. fileciteturn2file16

[proposal] In the full design, `delegation` is one edge family inside `work_trace`. Work trace supports reconstructing, and sometimes reproducing, what happened: tool calls, command exits, file paths, repo/commit links, generated artifacts, handoffs, query runs, and reports. Do not promise deterministic replay by default; promise **reconstruction** and **reproducibility bundles**. Replay is a privileged, unsafe, partial operation.

[evidence] The raw-log already contains the seed of this: correlate Polylogue logs with git commits/issues/PRs, use temporal analysis to distinguish editing vs builds/tests, and make work more observable. fileciteturn1file17 Current code stores sessions↔repos↔commits with confidence/evidence (`index.py:477-520`), action rows with commands/paths/exit status (`index.py:324-343`), and run/event/context units (`index.py:913-1024`).

[proposal] Query objects become essential in the full direction. A Fable/delegation analysis should be able to say:

```text
analysis:a_fable_rhetoric
  step 1: query_run:q1 -> result_relation:r1 -> cohort:@candidate_delegations
  step 2: evidence_pack:e1 -> annotation_batch:b1
  step 3: query_run:q2 over assertions -> cohort:@strict_delegations
  step 4: report:artifact:rpt1
```

[evidence] C10 already proposes recall entries in `user.db` with query text, resolved spec, result fingerprint, timestamp, and macros via `@name`; it also says recalled queries should store resolved specs rather than just text. fileciteturn2file0 B8 proposes a single `PolylogueService` contract with `query`, `read`, `preview`, `complete`, `act`, `status`, and possibly `facets`, replacing scattered facade/CLI/HTTP paths. fileciteturn1file15

[proposal] In the full design, `cohort` means a named relation over a grain. A dynamic cohort is a saved query/spec; a snapshot cohort is frozen member ids at an archive epoch. Delegation analytics wants both: `@fable_candidate_delegations` should be dynamic while exploring; `@fable_report_v1_specimens` should be a snapshot for publication.

[proposal] External annotation recipes should compile to actual database objects, not only YAML. A recipe/prompt is portable intent. The durable record is `analysis_run`, with child `query_run`, `result_relation`, `cohort`, `evidence_pack`, `annotation_batch`, `assertion`, and `report` refs. This makes complex agent loops general without forcing every analysis into a rigid YAML workflow language.

[proposal] Full analytics examples:

```text
delegations where harness:claude-code
| group by subagent_type, status
| count

delegations where parent.model:Fable
| project delegation-card
| annotate schema:directive_style_v1

assertions where scope:analysis:fable and key:directive_intensity and value >= 4
| lift delegations
| group by child_model_family
| count

delegations where status:failed
| join parent.actions after result
| project failure-recovery-card

cohort:@strict_fable_delegations
| except (delegations where parent_use:none)
| render specimen-gallery
```

## Fable demo generalized

[evidence] The Fable idea is only the public hook. The bundle’s own strategic frame says the valuable demonstration is externally inspectable proof, and the project’s category is “system of record for AI work,” not a chat viewer. fileciteturn2file16

[proposal] Generalize the demo as **orchestrator delegation analysis**:

```text
Question:
How does an orchestrator instruct, constrain, route, and consume subagent work?

Units:
delegations, child runs/sessions, artifacts, parent-use windows, annotations.

Metrics:
fanout, model-family routing, task type, directive intensity, output-contract strictness,
scope-control devices, completion/failure status, parent-use/ignore rate, retry/escalation.

Specimens:
delegation-card projections with exact evidence refs.

External agent role:
annotate delegation rows under schema directive_style_v1, import assertions,
then re-query aggregates and outliers.
```

[proposal] This makes Fable one cohort, not one product feature:

```text
@fable_delegations
@claude_code_delegations
@codex_subagent_delegations
@strict_output_contracts
@failed_but_used
@non_claude_children
```

[proposal] A demo prompt should ask for three artifacts:

```text
cohort.json          candidate delegations + evidence refs
annotations.jsonl   directive_style_v1 judgments targeting delegation:<id>
report.md           citable report with specimen cards, counts, caveats, reproduce command
```

[proposal] The citable claim should be modest: “In this local archive slice, Fable-visible orchestration uses these delegation/control patterns.” It should not claim hidden Fable internals or generalize beyond visible local evidence.

## Concrete bead recommendations

[evidence] Existing owner beads: `polylogue-s7ae` for coordination substrate, `polylogue-s7ae.4` for archive session-tree/topology/proof/context-flow composition into coordination envelope, `polylogue-s7ae.5` for live two-agent proof, `polylogue-37t.12` for judgment queue, `polylogue-fnm.10` for fields/select, `polylogue-fnm.13` for set algebra, `polylogue-4p1` for read algebra, `polylogue-t46` for contract-owned surfaces, `polylogue-9l5` for construct-valid analytics, `polylogue-83u` for artifact/blob integrity, and `polylogue-4ts` for lineage truth.

[proposal] Add one new P1/P2 child under `s7ae`, not a new top-level epic:

**`s7ae.delegation-unit` — First-class delegation graph unit**

Design: materialize provider-neutral `delegation` rows from Claude Code Task blocks, Codex `source.subagent.thread_spawn`, `session_runs.role=subagent`, SubagentReport extraction, and session topology. Include parent/child refs, instruction/result block refs, link confidence, status, artifact refs, and evidence refs. Do not special-case `.agent/scratch`; artifacts are generic produced/consumed/mentioned/derived_from edges.

Acceptance: `delegations where ...` works as a terminal query unit; fixture tests cover Claude Task, Claude acompact exclusion, Codex subagent spawn, unresolved child, and no false subagent from plain `forked_from_id`; rows carry evidence refs; result is queryable via CLI/API/MCP; devtools verify green.

[proposal] Add a second child, probably under `83u` or `s7ae` depending on ownership:

**`artifact-edge-derived` — Generic artifact links from sessions/actions/runs/delegations**

Acceptance: derived artifact edges link raw artifacts to sessions/actions/runs/delegations with edge type and evidence refs; no path name is special; `.agent/scratch`, report markdown, JSON findings, exported evidence packs, generated PR summaries, and raw sidecars all use the same relation.

[proposal] Add a child under `37t.12` or a sibling:

**`annotation-batch-import` — Agent-written JSONL judgments into assertions**

Acceptance: schema-registered JSONL imports candidate assertions targeting session/message/block/run/delegation/query-result refs; rows require evidence refs; batch metadata records model/prompt/analysis; CLI/MCP can list/accept/reject/import; accepted assertions are queryable via `assertions where ...`.

[proposal] Add a child under `4p1` / projection surface:

**`delegation-card-projection`**

Acceptance: read/projection renders a bounded delegation card with instruction, parent context, child result, parent-use window, artifacts, annotations, and provenance; output works in markdown and JSON; no custom Fable view.

[proposal] Add a child under `fnm`:

**`delegation-query-unit`**

Acceptance: `delegations where ...` supports rows/count/group/select, joins assertion labels by target, respects grain rules, and participates in set algebra.

## Open questions for the operator

[proposal] Should every delegation attempt get a row, even if no child session is resolved? My recommendation: yes, with `link_status=unresolved`; otherwise failed/unlinked delegation behavior disappears.

[proposal] Should `agent-acompact-*` become a separate `compaction` unit rather than excluded entirely? My recommendation: yes later; for delegation extraction, exclude it now to avoid false Fable/Claude subagent counts.

[proposal] Should artifact edges live in `index.db` only, or should some artifact annotations live in `user.db`? My recommendation: raw/derived artifact observations and edges in source/index; human/agent judgments about artifact meaning in assertions/user.db.

[proposal] Should annotation imports default to `candidate` or `active`? My recommendation: agent-written qualitative labels default to `candidate`; deterministic extractor facts can be `active` if schema-owned and tested.

[proposal] How sensitive are delegation prompts? Should `delegation-card` show full prompts by default? My recommendation: JSON evidence pack can include full text under explicit opt-in; public markdown defaults to bounded prompt excerpts plus evidence refs.

[proposal] Should “Fable” be a normalized model/provider token or an analysis-level cohort label? My recommendation: normalize model/provider if present in source; otherwise keep “Fable-visible” as a cohort assertion with evidence, not a source fact.

## What’s missing

[evidence] I did not inspect a live archive for actual Fable sessions or real Fable delegation examples. The design is grounded in the code bundle and uploaded reports, not in observed Fable rows.

[evidence] I did not run the daemon, migrations, query unit tests, or materializers. All implementation estimates assume the July 5 snapshot code paths still exist and have not moved.

[evidence] I did not inspect raw Claude/Codex fixture payloads beyond parser/materializer code. The extraction design should be validated against real source files before bead acceptance is finalized.

[evidence] I did not verify the live Beads state after the snapshot. The bundle contains post-swarm bead changes, but the repository may have moved; verify with `bd show polylogue-s7ae polylogue-37t.12 polylogue-fnm.13 polylogue-4p1 polylogue-t46`.

[evidence] I did not run multimodal artifact analysis. If delegation artifacts include screenshots, PDFs, or image reports, this design only covers their refs/metadata, not content extraction.
