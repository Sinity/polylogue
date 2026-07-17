## [user 307]
▎ *You have the forked context: the polylogue bundle, the swarm2 daemon/composer/thin-client designs, and our discussion of queries-as-objects, delegation units, and annotation recipes. Ground every claim in the bundle (cite file:line / bead id); tag [evidence] vs [proposal]. Be opinionated — one defended recommendation plus the runner-up you rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate change, (3) the full direction. End with Open questions for the operator and What's missing (a claim unverified, a source unread, a modality not run). The snapshot is dated; live code may have moved — flag assumptions to verify locally. Don't re-derive swarm2.*

Analytical [A] over the bundle:
[A] The "so what" analytics engine (9l5). Design the substrate for temporal decomposition (edit/compile/test/idle spans), agent-efficiency scoring, and delegation analytics. Which new units/materializers/DSL surfaces are needed; what's one materializer away from existing insights.

## [assistant 319]
bash -lc cd /mnt/data/poly_analysis/polylogue && python3 - <<'PY'
import json, textwrap
issues={}
for l in open('polylogue-beads-export.jsonl'):
 d=json.loads(l)
 if d.get('_type')=='issue': issues[d['id']]=d
print('issues',len(issues))
terms=['9l5','efficiency','analytics','temporal','delegation','run','work event','construct','materializer','agent']
for term in terms:
 print('\nTERM',term)
 n=0
 for id,d in issues.items():
  txt=' '.join(str(d.get(k,'')) for k in ['id','title','description','design','acceptance_criteria','notes','labels'])
  if term.lower() in txt.lower():
   print(id,d.get('priority'),d.get('status'),d.get('issue_type'),d.get('title'))
   n+=1
   if n>=25: break

for id in sorted([i for i in issues if i.startswith('polylogue-9l5') or i in ['polylogue-f2qv','polylogue-f2qv.1','polylogue-f2qv.2','polylogue-f2qv.3','polylogue-f2qv.4','polylogue-s7ae','polylogue-s7ae.1','polylogue-s7ae.4','polylogue-s7ae.5','polylogue-37t.12']]):
 d=issues[id]
 print('\n###',id,d.get('priority'),d.get('status'),d.get('issue_type'),d.get('title'))
 for f in ['description','design','acceptance_criteria','notes']:
  v=d.get(f)
  if v:
   print(f.upper()+':')
   print(textwrap.shorten(v.replace('\n',' | '), width=1800, placeholder=' ...'))
PY

## [assistant 322]
bash -lc cd /mnt/data/poly_analysis/polylogue && python3 - <<'PY'
import json, textwrap
issues={}
for l in open('polylogue-beads-export.jsonl'):
 d=json.loads(l)
 if d.get('_type')=='issue': issues[d['id']]=d
ids=['polylogue-9l5','polylogue-9l5.1','polylogue-9l5.2','polylogue-9l5.3','polylogue-9l5.4','polylogue-9l5.5','polylogue-9l5.6','polylogue-9l5.7','polylogue-9l5.7.1','polylogue-9l5.8','polylogue-9l5.9','polylogue-9l5.10','polylogue-9l5.11','polylogue-9l5.12','polylogue-b0b.1','polylogue-stc','polylogue-h10','polylogue-s7ae','polylogue-s7ae.1','polylogue-s7ae.4','polylogue-s7ae.5','polylogue-37t.12','polylogue-f2qv']
for id in ids:
 d=issues.get(id)
 if not d:
  print('MISSING',id); continue
 print('\n---',id,'---')
 for k in ['id','title','priority','status','issue_type','parent_id','labels','depends_on','external_ref']:
  print(k,':',d.get(k))
 for f in ['description','design','acceptance_criteria','notes']:
  v=d.get(f)
  if v:
   print(f+':')
   print(v[:5000])
PY

## [assistant 325]
bash -lc cd /mnt/data/poly_analysis/tree/polylogue && 
# key line excerpts
for f in polylogue/storage/sqlite/archive_tiers/index.py polylogue/archive/query/metadata.py polylogue/surfaces/projection_spec.py polylogue/core/enums.py polylogue/storage/sqlite/archive_tiers/user.py polylogue/storage/sqlite/archive_tiers/user_write.py polylogue/coordination/payloads.py polylogue/coordination/envelope.py polylogue/pipeline/payload_types.py; do echo '---' $f; done
nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '1,120p'
printf '\n--- actions ---\n'; nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '300,360p'
printf '\n--- repos/commits ---\n'; nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '450,530p'
printf '\n--- runs/observed/context ---\n'; nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '900,1035p'
printf '\n--- query metadata units ---\n'; nl -ba polylogue/archive/query/metadata.py | sed -n '470,535p'; nl -ba polylogue/archive/query/metadata.py | sed -n '780,835p'
printf '\n--- projection spec ---\n'; nl -ba polylogue/surfaces/projection_spec.py | sed -n '1,80p'; nl -ba polylogue/surfaces/projection_spec.py | sed -n '130,180p'
printf '\n--- assertions enums/user table/write ---\n'; nl -ba polylogue/core/enums.py | sed -n '399,447p'; nl -ba polylogue/storage/sqlite/archive_tiers/user.py | sed -n '1,45p'; nl -ba polylogue/storage/sqlite/archive_tiers/user_write.py | sed -n '901,1005p'
printf '\n--- coordination payloads/envelope ---\n'; nl -ba polylogue/coordination/payloads.py | sed -n '1,260p'; nl -ba polylogue/coordination/envelope.py | sed -n '600,940p'
printf '\n--- pipeline payload types ---\n'; nl -ba polylogue/pipeline/payload_types.py | sed -n '1,125p'

## [assistant 360]
According to the 2026-07-05 bundle snapshot, I would treat 9l5 as **almost ready for a first serious “so what” engine**, but not by building a big dashboard first. My defended recommendation is:

**[proposal] Build `activity_spans_v1` as the next analytic substrate: a derived, queryable relation over time-bounded work spans, backed by existing actions, messages, phases, work-events, observed-events, runs, and subagent projections. Then register measures over it.**

The runner-up I reject is: **start with the full 9l5.7 measure registry/statistics layer first.** It is directionally right and already beaded as the keystone, but if implemented in full before any new analytic relation, it risks becoming another internal substrate artifact. The situation brief explicitly warns that internal scaffold can substitute for the external artifact, and says the missing gate is a finished, inspectable demonstration of real task uplift. `activity_spans_v1` gives immediate publishable numbers; the full measure registry can harden them next. fileciteturn1file6

Snapshot caveat: this is grounded in the uploaded bundle and extracted code snapshot. Live code and Beads may have moved; verify locally with `bd show polylogue-9l5 --json`, `bd show polylogue-9l5.6 --json`, and current `rg` before filing.

## Layer 1 — with today’s substrate

[evidence] Today’s Polylogue already has more than enough raw structure to produce primitive analytics. The read algebra already exposes units including `observed-event`, `context-snapshot`, `message`, `action`, `block`, `assertion`, `file`, and `run`, with pipelines split on `|` and stages such as group/count/terminal actions. That is in the swarm brief at `SWARM_BRIEF.md:9-18`. The same brief says projection is currently still named-view-heavy, not fully composable, at `SWARM_BRIEF.md:19-25`. fileciteturn1file8

[evidence] The action substrate is strong. The SQLite `actions` view pairs `tool_use` blocks with `tool_result` blocks and exposes `tool_name`, `semantic_type`, `tool_command`, `tool_path`, `output_text`, `is_error`, and `exit_code` at `polylogue/storage/sqlite/archive_tiers/index.py:324-343`. Query rows preserve time ordering via `messages.occurred_at_ms` and expose `is_error`, `exit_code`, and `followup_class` at `archive.py:4960-5002`. The query metadata explicitly exposes action fields `exit_code`, `followup_class`, `is_error`, `output`, `path`, `tool`, and `duration_ms` at `polylogue/archive/query/metadata.py:474-505`.

[evidence] The run/delegation substrate exists but is not yet a first-class “delegation” unit. `session_runs` stores `role` as `main` or `subagent`, plus `parent_run_ref`, `agent_ref`, `provider_origin`, `harness`, status, cwd, branch, lineage refs, and evidence refs at `polylogue/storage/sqlite/archive_tiers/index.py:913-938`. The run projection builds subagent runs from extracted subagent reports, assigns `role="subagent"`, links parent/child run refs, and emits `subagent_started` / `subagent_finished` observed events at `polylogue/insights/run_projection.py:307-368`. Subagent extraction currently recognizes `Task` tools or `subagent_type` / `agent_type`, prompt/task ids, child session ids, final report previews, PR/issue refs, tests, caveats, and raw refs at `polylogue/insights/transforms.py:1618-1655` and `2072-2138`.

[evidence] Temporal material already exists, but at the wrong granularity for edit/compile/test/idle decomposition. `session_work_events` stores weakly labelled message ranges with start/end/duration, file paths, tools used, evidence, inference, and search text at `polylogue/storage/sqlite/archive_tiers/index.py:689-708`. `session_phases` stores deterministic time-gap intervals with message ranges, duration, tool counts, word count, and evidence at `index.py:731-748`. The phase extractor explicitly says phases are time-gap segmented intervals, not intent-classified, at `polylogue/archive/phase/extraction.py:27-34`. The work-event labels include planning, implementation, debugging, review, testing, research, configuration, documentation, refactoring, data_analysis, and session at `polylogue/archive/session/extraction.py:89-102`; their classification is heuristic, based on action counts and text signals, at `extraction.py:305-359`.

[evidence] There is also already an honest timing posture. Session profiles store total, engaged, tool-active, wall, thinking, output, and tool duration fields, plus workflow shape, terminal state, cost, token lanes, and timing provenance at `polylogue/storage/sqlite/archive_tiers/index.py:799-868`. OTLP correlation can enrich work events with exact spans when present, otherwise falling back to message-gap estimates with evidence provenance, at `polylogue/insights/otlp_correlation.py:1-6`, `193-260`, and `274-318`.

[proposal] With no schema change, today’s substrate can already answer weak but useful questions:

`actions where exit_code > 0 | group by tool | count`

`actions where is_error:true | group by followup_class | count`

`observed-events where kind:test_failed | group by session.origin | count`

`runs where role:subagent | group by harness | count`

`actions where session.repo:polylogue | group by action | count`

Those are “so what” adjacent, but not yet the real engine. They tell you which tools fail and where subagents appear. They do **not** yet tell you “this session spent 42% in edit, 18% in test, 31% idle/wait, then delegated twice and only used one result.”

[evidence] Bead `polylogue-9l5` already frames analytics exactly as “the archive answers ‘so what’ questions” and describes Layer 0 as existing substrate: profiles, work events, phases, threads, cost rollups, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes. Its acceptance criteria require analytics to land as registered measures over the query algebra, not bespoke modes. In the bundled export, `polylogue-beads-export.jsonl:211` is the 9l5 epic. `polylogue-9l5.1` is outcome-conditioned analytics; `polylogue-9l5.6` is tool episodes; `polylogue-9l5.7` is measure registry; `polylogue-9l5.8` is temporal analytics; `polylogue-9l5.10` is process mining; `polylogue-9l5.12` is graph/information-theoretic measures (`polylogue-beads-export.jsonl:156-157`, `207-211`, `293-296`, `357-359`).

## Layer 2 — near-term substrate change

[proposal] Add one derived relation first:

`activity_spans`

This is the missing bridge. It is not a dashboard. It is not a model. It is a materialized, evidence-tiered relation over bounded work intervals.

A row should look roughly like:

```text
span_ref
session_id
run_ref
parent_span_ref
start_message_id
end_message_id
start_action_ref
end_action_ref
started_at_ms
ended_at_ms
duration_ms
turn_start
turn_end
kind
subkind
outcome
confidence
evidence_tier
timing_provenance
classification_provenance
tool_episode_refs_json
action_refs_json
observed_event_refs_json
file_refs_json
commit_refs_json
delegation_refs_json
evidence_refs_json
payload_json
```

[evidence] The needed source fields are already there: action rows carry tool, command/path, timestamp, is_error/exit_code/followup_class (`archive.py:4960-5002`); phases give gap-bounded intervals (`index.py:731-748`; `phase/extraction.py:23-34`); work-events give weak semantic labels and evidence (`index.py:689-708`; `session/extraction.py:505-543`); observed events give structured outcomes including `tool_finished`, `command_failed`, `test_failed`, and subagent start/finish (`run_projection.py:22-33`, `171-192`, `268-305`, `346-368`); session profiles already store active/tool/wall durations and workflow/terminal state (`index.py:799-868`).

[proposal] Span kinds should start coarse and construct-valid:

`read_search`, `edit`, `build_compile`, `test`, `debug`, `review_vcs`, `delegate`, `synthesize`, `idle_gap`, `tool_wait`, `llm_wait`, `unknown`.

The important design choice: **separate “span kind” from “evidence tier.”** A span classified as `test` from a `pytest` command with exit code is stronger than a span classified as `debug` from user text containing “debug.” This follows the 9l5.7.1 gap: existing Layer 0 payloads expose confidence floats but not an explicit evidence-tier distinction, while that bead proposes tagging structural/provider-reported/derived/heuristic confidence (`polylogue-beads-export.jsonl:263`, bead `polylogue-9l5.7.1`).

[proposal] Build `activity_spans_v1` as a materializer over existing tables, not as a new capture source. The algorithm:

First, normalize action rows into atomic events. Use `semantic_type`, `tool_name`, `tool_command`, `tool_path`, `is_error`, `exit_code`, and `occurred_at_ms`. Commands that look like `pytest`, `devtools test`, `devtools verify`, `ruff`, or `mypy` already have related recognition in transforms (`polylogue/insights/transforms.py:2227-2228`); extend that classifier into a versioned alphabet rather than burying it in ad hoc analytics.

Second, split the session timeline at large gaps using the phase threshold. Existing phases use a 5-minute gap threshold at `polylogue/archive/phase/extraction.py:23-24`, and the phase table stores the threshold per row at `index.py:731-748`. A gap larger than threshold should produce an `idle_gap` or `human_absence` span, not disappear into duration math.

Third, merge adjacent atomic events into spans when their kind is stable and no significant gap intervenes. This mirrors the current work-event merge behavior for adjacent same-label events at `polylogue/archive/session/extraction.py:474-502`, but span classification must be stricter and more structural.

Fourth, attach outcome. For edit/test loops, the outcome is often not the edit itself but the following test/build/verify action. The span materializer should attach “next structural outcome” and “next diagnostic outcome” so we can compute recovery and loop metrics.

Fifth, emit caveats. If timestamps are missing, use turn-axis only. If classification comes from text signals, mark `evidence_tier=heuristic`. If a tool has no result, mark outcome unknown. This matches the 9l5.6 tool-episode AC, which explicitly says a tool_use with no paired result must yield an unknown-outcome episode rather than being dropped (`polylogue-beads-export.jsonl:207`, bead `polylogue-9l5.6`).

[proposal] I would slightly revise the current 9l5 design: make `tool_episodes` (`polylogue-9l5.6`) the **atomic episode layer**, and add `activity_spans` as the **temporal composition layer** above it. `tool_episodes` answers “what happened at this call/result boundary?” `activity_spans` answers “how did work unfold over time?”

[proposal] New query units:

`tool-episode` — mostly already specified by bead `polylogue-9l5.6`.

`activity-span` — new. Queryable fields: `kind`, `subkind`, `outcome`, `duration_ms`, `evidence_tier`, `timing_provenance`, `tool`, `path`, `repo`, `run`, `agent`, `session.origin`, `session.repo`.

`delegation` — new or materialized view over run/subagent_report/observed_event. Queryable fields: `parent_run`, `child_run`, `subagent_type`, `model_family`, `instruction_text`, `result_status`, `return_latency_ms`, `used_by_parent`, `outcome`, `evidence_tier`.

[proposal] Minimum DSL surfaces:

```text
activity-spans where session.repo:polylogue
  | group by kind
  | sum duration_ms
```

```text
tool-episodes where outcome:failed
  | group by tool, followup_class
  | count
```

```text
activity-spans where kind:test AND outcome:failed
  | seq(test_failed -> edit -> test_passed)
```

```text
delegations where parent.origin:fable
  | group by child.model_family, result_status
  | count
```

```text
activity-spans where session.repo:polylogue
  | measure active_time_ratio by session.origin window week
```

[evidence] This is aligned with the existing read-algebra direction, because the current design explicitly wants one query grammar and composable projection/render rather than multiplying bespoke verbs (`SWARM_BRIEF.md:9-34`, `63-65`). It also aligns with B8’s diagnosis that `QueryProjectionSpec = Selection × Projection × Render` exists but is not yet the executor, and that read still runs through named views (`B8_contract.md:66-75`). fileciteturn1file8 fileciteturn0file2

## Layer 3 — full direction

[proposal] The full 9l5 direction should be a **measure algebra over typed relations**, not an “analytics page.”

The core objects:

`tool_episode` — one tool call/result pair, including unknown-result cases.

`activity_span` — bounded work span with kind, duration, outcome, evidence tier.

`delegation` — parent agent assigns child/subagent work, with instruction/result/use.

`query_run` — stored Polylogue query execution, so analytics can cite the analysis process itself.

`result_relation` / `cohort` — reusable result set by grain.

`annotation_batch` — agent/human external judgments imported as assertions.

`analysis_run` — provenance DAG tying query runs, cohorts, annotation batches, evidence packs, and reports.

[evidence] Existing object-ref kinds already include `work_event`, `phase`, `file`, `branch`, `commit`, `agent`, `repo`, `run`, `context-snapshot`, `observed-event`, `assertion`, `saved_view`, `recall_pack`, `tool-call`, and `subagent-report` at `polylogue/core/refs.py:8-38`. That means the ref vocabulary is already broad enough for this direction, but it lacks explicit `query_run`, `result_relation`, `activity_span`, `delegation`, and `analysis_run`.

[proposal] Agent-efficiency scoring should **not** be one magic score. It should be a vector of registered measures, with an optional composite view. The composite can exist, but it should never be the source of truth.

Core measures:

`active_time_ratio = active_span_duration / wall_duration`

`idle_gap_ratio = idle_gap_duration / wall_duration`

`edit_test_cycle_count`

`failure_recovery_latency = time from failed test/command to next structural success`

`silent_proceed_after_failure_rate = failed episode followed by non-acknowledging assistant action`

`verification_after_edit_rate = edit spans followed by test/build/verify before final answer`

`cost_per_clean_session`

`cost_per_structural_success`

`tool_failure_rate`

`delegation_fanout`

`delegation_return_latency`

`delegation_used_result_rate`

`delegation_rework_rate`

`subagent_parallelism_width`

`context_churn_ratio`

[evidence] These correspond to existing or beaded analytics. `polylogue-9l5.1` wants cost/duration/retries/tools by structural success; `polylogue-9l5.6` wants tool episodes; `polylogue-9l5.8` wants series/trends/baselines/changepoints; `polylogue-9l5.10` wants process mining over event sequences; `polylogue-9l5.12` wants fan-out/depth distributions and co-occurrence networks (`polylogue-beads-export.jsonl:156`, `207`, `210`, `293`, `295`). The situation brief also says the process story should carry a quantitative spine, but that raw commit volume can mislead because high-commit mechanical sweeps may be low-value compared with architectural collapse days (`situation-brief(2).md:97-107`). fileciteturn1file6

[proposal] Delegation analytics should be its own unit, not just `runs where role:subagent`. The current run projection can count subagent runs and start/finish events, but an analysis like “Fable rules subagents with an iron fist” needs the parent instruction, child identity/model family, output contract, child result, parent use of the result, retry/critique, and return latency. That is a `delegation` relation.

A minimal `delegation` materializer:

```text
delegation_ref
parent_session_id
parent_run_ref
child_run_ref
subagent_report_ref
instruction_block_ref
result_block_ref
subagent_type
child_model
child_model_family
instruction_preview
result_preview
status
started_at_ms
finished_at_ms
return_latency_ms
used_by_parent
parent_use_ref
evidence_tier
evidence_refs_json
payload_json
```

`used_by_parent` is tricky: v1 should be `unknown|mentioned|quoted|synthesized|ignored` with evidence-tier labels. It can be structural when the parent cites the child result by ref; derived/heuristic when inferred from text similarity or temporal adjacency. This should explicitly use assertions/annotation batches for richer qualitative labels instead of pretending the derived label is raw truth.

[proposal] The full scoring layer should be:

`raw evidence → tool episodes → activity spans → delegation edges → measures → series/comparisons → reports`.

A “score” only renders at the last stage. It should decompose into visible components:

```text
agent_efficiency_score
  = structural_success_quality
  + verification_discipline
  + low_idle_ratio
  + low_rework_ratio
  + good_delegation_yield
  - coverage_penalties
```

But the report should show all components, their evidence tiers, and sample frame. This is the construct-valid version of “which model/agent is better.”

## What is one materializer away?

[proposal] The best one-materializer target is:

**`activity_spans_v1` over the existing `actions`, `messages`, `session_phases`, `session_work_events`, `session_observed_events`, `session_runs`, and `session_profiles`.**

Why this one:

[evidence] Existing work events and phases already give time-bounded intervals, but phases are explicitly not intent labels (`polylogue/archive/phase/extraction.py:27-34`) and work-events are weak heuristic labels (`polylogue/archive/session/extraction.py:89-102`, `305-359`). Existing action rows have the stronger structural data: tool call/result pairing, exit code/error flag, path/command, timestamps, followup class (`polylogue/storage/sqlite/archive_tiers/index.py:324-343`; `archive.py:4960-5002`). Existing observed events already include structured command/test outcomes (`polylogue/insights/run_projection.py:22-33`). So the missing materializer is a normalizer/composer, not a new capture pipeline.

[proposal] Once `activity_spans_v1` exists, five immediately useful insights become simple grouped measures:

Edit/test/idle decomposition by model, repo, and harness.

Failure recovery latency after test/build failures.

Wasted-loop detection as a continuous measure, not just a pathology flag.

Delegation overhead: time before/after subagent dispatch and parent integration.

Cost per active minute / cost per structural success.

That is the shortest route from existing substrate to “so what.”

## New or changed beads I would file

[proposal] Add under `polylogue-9l5`:

**`polylogue-9l5.x — Activity spans materializer: edit/test/build/idle/delegate intervals over tool episodes and phases`**

Design: derived `activity_spans` table in `index.db`; builds from action rows, observed events, phases, work-events, run projections; emits span kind, duration, outcome, evidence tier, timing provenance, action/tool/delegation refs; versioned classifier alphabet; no LLM labels.

Acceptance: seeded corpus produces spans; every span has evidence refs; gaps over threshold become idle spans; structural test failures produce `kind=test outcome=failed`; broad query `activity-spans where session.repo:polylogue | group by kind | sum duration_ms` works; no span with heuristic-only classification renders without `evidence_tier=heuristic`.

**`polylogue-9l5.x — Delegation unit: parent instruction → child run/report → parent use`**

Design: derived `delegations` unit over subagent reports/run projections; includes instruction/result refs, subagent type/model family, status, return latency, parent-use classification, evidence tier.

Acceptance: `delegations where role:subagent` equivalent row count matches subagent run count within documented caveats; every row links parent and child evidence; parent-use can be unknown and is not guessed as used.

**`polylogue-9l5.x — Efficiency measure pack v1`**

Design: MeasureSpec registrations over `activity_spans` and `tool_episodes`: active_time_ratio, idle_gap_ratio, test_failure_recovery_latency, verification_after_edit_rate, tool_failure_rate, delegation_return_latency, delegation_used_result_rate.

Acceptance: each measure declares construct, formula, sample frame, evidence tier, confounds; output refuses unlabeled cross-origin comparison if coverage tier missing, matching 9l5.7 doctrine.

**`polylogue-fnm.x — Span/delegation terminal unit DSL`**

Design: add `activity-spans`, `tool-episodes`, `delegations` terminal sources, fields, group keys, and completions. Pipeline remains hand-split outside Lark, so new stages need care but not a full grammar rewrite; the compaction memory also warns that new terminals containing `:` must respect terminal priority relative to `FIELD_CLAUSE` (`Pasted text(35).txt`, query DSL memory). fileciteturn1file4

**`polylogue-bby.x / jnj.x — Analytics evidence-card projection`**

Design: projection preset for span/episode/delegation evidence cards. This is how a group-count cell drills down to actual evidence.

## Rejected runner-up

[proposal] Rejected runner-up: “Implement full 9l5.7 first: stats module, registry, DSL measure/compare, multiple-comparison controls, renderer footnotes, then build analytics.”

Reason: 9l5.7 is correct but over-frontloaded. The bead itself even has a “contract-first split” note saying slice 1 should be small: MeasureSpec dataclass + registry + Wilson interval + one registered measure, because downstream beads depend on the spec shape, not the complete primitive set (`polylogue-beads-export.jsonl:157`, bead `polylogue-9l5.7`). That is the path I’d take: implement the **thin MeasureSpec skeleton** alongside `activity_spans_v1`, not the full statistical edifice first.

[evidence] This also follows the situation brief’s product constraint: the next thing should become a finished artifact proving usefulness, not another internally satisfying scaffold (`situation-brief(2).md:133-137`, `157-161`). fileciteturn1file6

## Open questions for the operator

What should count as “success” for agent-efficiency v1: final structural terminal state, test/verify pass, merged commit, closed bead, or report accepted by parent? My recommendation: v1 uses structural tool/test success and terminal_state; PR/bead outcomes become optional stronger labels.

Should idle mean “no messages/tools for >5 minutes,” “human absence,” “tool wait,” or “agent thinking”? My recommendation: separate `idle_gap`, `tool_wait`, `llm_wait`, and `human_absence` when evidence supports it; do not collapse them into one idle score.

Should delegation “used_by_parent” be structural-only at first, or may it use heuristic text similarity? My recommendation: allow heuristic as candidate assertion, not as raw field. Structural mention/citation gets `evidence_tier=structural`; inferred use gets `derived|heuristic`.

Do you want an explicit composite “agent_efficiency_score,” or only a scorecard of measures? My recommendation: scorecard first; composite only as a named view that decomposes into components.

Should `activity_spans` live under 9l5 as analytics substrate, or under a more general “work-trace” epic that also serves context/replay/delegation? My recommendation: file it under 9l5 now, but design it as a reusable `work-trace` relation.

## What’s missing

I did not run live archive queries or materializers. All claims are from the uploaded 2026-07-05 bundle, extracted code snapshot, and Beads export. Live `master` may have moved.

I did not verify the current Beads state after the later “391 issues” / “370 issues” transcript updates except through uploaded logs. The bundle export I inspected contains 398 issue records; the transcript mentions later applied states. Verify locally.

I did not inspect every raw-log line; I used the relevant retrieved chunks and the uploaded design reports.

I did not run modality-specific checks for Sinex/ActivityWatch/OTLP availability. Exact idle/tool/LLM timing depends heavily on whether OTLP spans or other capture modalities are present; otherwise v1 must use message-gap estimates with caveats.

I did not inspect the live daemon or current HTTP/UDS implementation. The swarm2 daemon/composer documents were treated as design context, not re-derived.
