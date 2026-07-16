## [user 307]
▎ *You have the forked context: the polylogue bundle, the swarm2 daemon/composer/thin-client designs, and our discussion of queries-as-objects, delegation units, and annotation recipes. Ground every claim in the bundle (cite file:line / bead id); tag [evidence] vs [proposal]. Be opinionated — one defended recommendation plus the runner-up you rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate change, (3) the full direction. End with Open questions for the operator and What's missing (a claim unverified, a source unread, a modality not run). The snapshot is dated; live code may have moved — flag assumptions to verify locally. Don't re-derive swarm2.*

Analytical [A] over the bundle:
[A] Annotation-recipe substrate. Audit how complete assertion-based annotation is today (from the bundle); name what's missing for an external agent to annotate sessions/messages and query them back; design the recipe format — decide DB-object vs YAML.

## [assistant 382]
According to the 2026-07-05 bundle snapshot, my defended recommendation is: **make analysis recipes DB-native runtime objects, with YAML as import/export/source-control serialization, and keep annotation rows as assertions.** [proposal] YAML-only prompt files are the runner-up; I reject them as the canonical substrate because they cannot themselves be queried, targeted by assertions, linked to query runs/results, resumed, judged, or audited as part of the archive. [proposal] The third tempting option, “store recipes themselves as assertions,” is also wrong: assertions are authored claims about targets; recipes/runs are procedure/state. The bundle’s own schema separates `user_settings` from `assertions`, which supports this distinction. [evidence: `polylogue/storage/sqlite/archive_tiers/user.py:7-49`] fileciteturn3file0

## Layer 1 — with today’s substrate

[evidence] The assertion storage substrate is substantially real. `user.db` has a strict `assertions` table with `assertion_id`, `scope_ref`, `target_ref`, `key`, `kind`, `value_json`, `body_text`, `author_ref`, `author_kind`, `evidence_refs_json`, `status`, `visibility`, `confidence`, `staleness_json`, `context_policy_json`, `supersedes_json`, and timestamps; it also has indexes by target/kind/status/visibility. [evidence: `polylogue/storage/sqlite/archive_tiers/user.py:7-40`] fileciteturn3file0

[evidence] The runtime vocabulary is already broad enough for external annotations: `AssertionKind` includes `annotation`, `correction`, `saved_query`, `recall_pack`, `note`, `decision`, `caveat`, `lesson`, `blocker`, `handoff`, `judgment`, `run_state`, `prompt_eval`, `transform_candidate`, and `pathology`; lifecycle states include `active`, `candidate`, `accepted`, `rejected`, `deferred`, `superseded`, `deleted`, and `inactive`. [evidence: `polylogue/core/enums.py:399-447`] fileciteturn2file12

[evidence] The low-level writer is capable: `upsert_assertion()` accepts target, kind, scope, key, JSON value, body text, author ref/kind, evidence refs, status, visibility, confidence, staleness, context policy, and supersedes; it normalizes target/scope/author/evidence refs before insert/update. [evidence: `polylogue/storage/sqlite/archive_tiers/user_write.py:901-999`] fileciteturn3file10

[evidence] Querying annotations back is partially real, not aspirational. The query grammar includes `assertion` as a structural unit, the query metadata exposes assertion fields such as `author`, `author_kind`, `author_ref`, `body`, `context`, `evidence`, `key`, `kind`, `scope_ref`, `status`, `target_ref`, `text`, `value`, and `visibility`, and `ArchiveStore.query_assertions()` executes unit-scoped assertion predicates against `user_tier.assertions`. [evidence: `SWARM_BRIEF.md:11-18`; `polylogue/archive/query/metadata.py:507-525`; `polylogue/storage/sqlite/archive_tiers/archive.py:5393-5447`] fileciteturn0file9 fileciteturn3file8

[evidence] Candidate review is also real. The storage layer has `list_assertion_candidate_reviews()` and `judge_assertion_candidate()`, where decisions are `accept`, `reject`, `defer`, or `supersede`; accepted/superseded candidates can be promoted into active assertions, and the judgment itself is recorded as an assertion targeting the candidate assertion. [evidence: `polylogue/storage/sqlite/archive_tiers/user_write.py:1185-1370`] fileciteturn3file3

[evidence] The API facade exposes read/review methods for claims and candidates: `list_assertion_claims`, `list_assertion_claim_payloads`, `list_assertion_candidates`, `list_assertion_candidate_reviews`, and `judge_assertion_candidate`. [evidence: `polylogue/api/archive.py:2079-2206`] fileciteturn3file12

[evidence] The shared payload layer has typed JSON-ish envelopes for assertion claims, judgments, judgment results, candidate-review items, and candidate-review lists; payload validators normalize object refs and public evidence refs. [evidence: `polylogue/surfaces/payloads.py:1465-1518`, `1554-1696`] fileciteturn3file4

[evidence] Existing high-level write APIs are still uneven. `save_annotation()` exists but is a legacy/narrow annotation method over `annotation_id`, `session_id`, `note_text`, target type/id, and optional `message_id`; it does not expose the full assertion shape to agents. `post_blackboard_note()` is richer — it accepts `author_ref`, `author_kind`, `evidence_refs`, `staleness`, and `context_policy` — but it is still semantically a blackboard note surface, not a general annotation-batch writer. [evidence: `polylogue/api/archive.py:4717-4746`, `5284-5345`] fileciteturn3file12

[evidence] The B8 contract audit says the facade exposes many loose methods, including `save_annotation`, `record_correction`, `judge_assertion_candidate`, `save_view`, `save_workspace`, and `post_blackboard_note`, but those write/read surfaces are not yet declared as a complete cross-surface contract; B8 explicitly calls out that annotate/correction/saved-view/workspace/blackboard/assertion-judgment are missing from the protocol family. [evidence: `B8_contract.md:84-124`] fileciteturn3file12

[proposal] Practical verdict for today: **storage is about 75% ready, query-back is about 55% ready, agent ergonomics are about 25% ready, and recipe substrate is about 5–10% ready.** An external agent can already produce assertion-shaped JSON and, with privileged/in-process access or a narrow existing API, store candidate annotations. It can query back through `assertions where ...` for basic fields. But it cannot yet have a smooth, stable loop of “export evidence pack → annotate JSONL under schema → import batch → query annotations → review candidates → render report” through one daemon-fast CLI/MCP/API contract.

### What is missing today for external agent annotation

[proposal] Missing 1: **general assertion import/write surface.** The agent needs `assertions import` / `act(kind=assertions.import)` / MCP equivalent that accepts JSONL rows with `target_ref`, `scope_ref`, `kind`, `key`, `value`, `body_text`, `evidence_refs`, `confidence`, `author_ref`, `author_kind`, and `status=candidate`. The low-level helper exists, but the stable external surface does not. This is exactly the kind of gap B8 says should move behind a single `act` verb rather than remaining loose facade/storage plumbing. [evidence: `B8_contract.md:151-155`, `117-124`] fileciteturn3file7

[proposal] Missing 2: **annotation schema registry.** Today `key`/`value_json` can hold arbitrary label data, but nothing declares that `directive_style_v1.directive_intensity` is an integer 0–5, requires at least one evidence ref, targets messages or delegations, has an abstain value, or was produced by a particular prompt/model. Without a schema registry, annotations remain queryable blobs rather than safe analytical variables.

[proposal] Missing 3: **annotation batches.** The unit of import should not be “many unrelated assertions.” It should be `annotation_batch:<id>` with schema id, source query/result, actor/model/prompt, import time, counts, validation failures, and review state. Each row is still an assertion; the batch is the provenance container.

[proposal] Missing 4: **analysis/run/query/result refs.** Current `ObjectRefKind` includes sessions/messages/blocks/files/commits/runs/context-snapshots/observed-events/assertions/saved_views/recall_packs/transforms/tool-calls/subagent-reports/GitHub refs, but not `analysis`, `analysis-run`, `annotation-schema`, `annotation-batch`, `query-run`, `result-relation`, or `cohort`. [evidence: `polylogue/core/refs.py:8-38`] Those refs are needed so assertions can target not only sessions/messages but also the analysis process itself.

[proposal] Missing 5: **typed JSON-value querying.** Current assertion fields include a broad `value` text/JSON substring field. That is enough for “find rows whose value mentions strict,” but not enough for “directive_intensity >= 4” or “control_devices contains read-only.” For annotation recipes, Polylogue needs schema-aware virtual fields or JSON-path predicates over assertion values.

[proposal] Missing 6: **evidence-pack export.** The annotation agent needs bounded evidence packs with stable refs, not ad hoc transcript excerpts. Existing read algebra can point in this direction, but B8 says read still executes through named views rather than the projection algebra executor. [evidence: `B8_contract.md:66-75`, `113-115`] fileciteturn3file12

[proposal] Missing 7: **operator-grade candidate review.** The storage/API candidate review exists, but the bead set still treats the actual ergonomic queue as open work: `polylogue-37t.12` is “Judgment queue: operator bulk review/accept/reject of candidate assertions”; `polylogue-p5g` is “polylogue judge: interactive candidate triage in the terminal”; `polylogue-dmp` is “polylogue note: zero-friction memory capture from the terminal.” Those are the missing affordances that make candidate assertions not rot.

## Layer 2 — near-term substrate change

[proposal] The near-term design should be: **assertion rows remain the annotation primitive; add first-class `annotation_schema`, `annotation_batch`, `analysis_recipe`, and `analysis_run` objects in `user.db`; expose them through the B8 `act/query/read/preview/complete/status/facets` shape; YAML is only a portable serialization.**

[proposal] Minimal tables or equivalent user-tier objects:

```text
annotation_schemas(
  schema_id,
  version,
  target_grain,
  value_schema_json,
  required_evidence_policy_json,
  allowed_keys_json,
  validity_note,
  created_at_ms,
  updated_at_ms
)

analysis_recipes(
  recipe_id,
  name,
  version,
  definition_json,
  source_artifact_ref,
  created_by_ref,
  created_at_ms,
  updated_at_ms
)

analysis_runs(
  analysis_run_id,
  recipe_id,
  status,
  actor_ref,
  archive_epoch,
  started_at_ms,
  finished_at_ms,
  input_refs_json,
  query_run_refs_json,
  annotation_batch_refs_json,
  artifact_refs_json,
  degraded_json
)

annotation_batches(
  batch_id,
  analysis_run_ref,
  schema_id,
  source_result_ref,
  target_grain,
  author_ref,
  author_kind,
  prompt_ref,
  model_ref,
  row_count,
  accepted_count,
  rejected_count,
  validation_json,
  created_at_ms
)
```

[proposal] Assertion rows imported by an agent should look like this:

```json
{
  "scope_ref": "annotation-batch:fable-rhetoric-v1-batch-001",
  "target_ref": "message:claude-code-session:abc123:m45",
  "kind": "judgment",
  "key": "directive_style_v1.directive_intensity",
  "value": {"score": 4, "scale": "0..5"},
  "body_text": "Strict bounded delegation with explicit non-overreach instruction.",
  "author_ref": "agent:gpt-5.5-pro",
  "author_kind": "agent",
  "evidence_refs": ["claude-code-session:abc123::m45::0"],
  "status": "candidate",
  "confidence": 0.82,
  "context_policy": {"inject": false}
}
```

[proposal] The near-term CLI/MCP affordances should be these, all daemon-backed:

```text
polylogue recipe import recipe.yaml
polylogue recipe run <recipe-id> --json
polylogue evidence-pack export --query '<query>' --projection <preset> --out evidence.jsonl
polylogue assertions import annotations.jsonl --schema directive_style_v1 --scope analysis-run:<id>
polylogue assertions query 'assertions where scope_ref:analysis-run:<id> AND key:directive_style_v1.directive_intensity'
polylogue judge --list --scope analysis-run:<id>
polylogue report render --analysis-run <id> --format markdown
```

[proposal] This uses the B8 service-contract direction rather than inventing a sidecar runner. B8’s recommended protocol already has `query`, `read`, `preview`, `complete`, `act`, `status`, and optional `facets`; `act` is explicitly the place to gather mutations such as save-annotation, correction, saved-view, assertion-judgment, and ingest. [evidence: `B8_contract.md:134-155`] fileciteturn3file7

[proposal] Recipe YAML should exist, but only as source/export. A recipe file should be importable, hash-pinned, and rendered back out from the DB. Example:

```yaml
api_version: polylogue.analysis/v1
recipe_id: fable-delegation-rhetoric
version: 1
question: >
  How does the orchestrator phrase delegation instructions to subagents?
inputs:
  corpus:
    query: 'sessions where model:Fable OR text:"subagent"'
    grain: session
evidence_pack:
  projection: delegation-card
  limit_per_target: 3
annotations:
  schema: directive_style_v1
  target_grain: message
  require_evidence_refs: true
  initial_status: candidate
  prompt_ref: file:prompts/directive-style-v1.md
post_import_queries:
  - name: strict_by_model_family
    query: 'assertions where key:directive_style_v1.directive_intensity AND value.score>=4 | group by target.session.model_family'
review:
  require_operator_judgment: true
report:
  renderer: finding-report
  outputs:
    - markdown
    - json
```

[proposal] DB-object canonicalization is better because the live object can be resumed, cited, queried, targeted by assertions, and inspected by later agents. YAML-only is attractive because it is simple, portable, and agent-readable, but it loses the whole benefit of Polylogue tracking its own analytical work. Use YAML as a declaration artifact; use the database as the execution record.

[proposal] Current bead coverage already implies this but does not name it cleanly. Relevant open/closed beads: `polylogue-37t` covers the declared-claims → judgment → preamble → reboot loop; `polylogue-37t.1` covers assertion consumer/lifecycle tightening; `polylogue-37t.2` covers agent-authored inline annotation markers; `polylogue-37t.12` covers judgment queue; `polylogue-dmp` covers zero-friction note capture; `polylogue-p5g` covers terminal candidate triage; `polylogue-fnm.12` covers user-defined query macros; `polylogue-fnm.13` covers set algebra; `polylogue-pj8` covers agent query cookbook / skill recipes; `polylogue-27p` being closed suggests MCP write access has landed in some form. The missing bead is the one that binds these into **analysis recipe + annotation batch substrate**, not another Fable-specific demo bead. [evidence: bead ids from `polylogue-beads-export.jsonl` in bundle]

## Layer 3 — full direction

[proposal] The full direction is: **analysis is a first-class graph over query runs, result relations, evidence packs, annotation batches, cohorts, reports, and artifacts.** Assertions remain the claim layer, not the workflow layer.

[proposal] Full object graph:

```text
analysis_recipe
  -> analysis_run
    -> query_run*
      -> result_relation*
        -> evidence_pack_artifact*
    -> annotation_batch*
      -> assertion*
    -> cohort*
    -> report_artifact*
```

[proposal] A **query run** is a specific execution with actor, surface, query text, lowered spec, archive epoch, degraded state, result grain, count/exactness, timing, and output refs. This generalizes C10’s recall-entry idea, where every committed composer run writes query text, resolved spec, result fingerprint, and timestamp to `user.db`. [evidence: `C10_composer_ux.md:202-219`] fileciteturn1file10

[proposal] A **result relation** is the durable/semi-durable set returned by a query, with grain and identity policy. This is the thing a recipe passes to an annotation agent. It can be exact, capped, sampled, or snapshot. It is the cleaner successor to “parse stdout and keep going.”

[proposal] A **cohort** is a named result relation. Dynamic cohort = saved query/spec; snapshot cohort = frozen refs from one archive epoch. Cohorts are needed because annotation recipes need stable input populations: “these 47 delegation messages were labeled in report v1.” They also become operands for set algebra, which is already on the query roadmap as `fnm.13` and in the composer design as live set-op preview. [evidence: `C10_composer_ux.md:183-200`; bead `polylogue-fnm.13`] fileciteturn1file10

[proposal] A **recipe** should be able to loop. The YAML example above is only the declarative first pass. The actual `analysis_run` should record that an agent queried, inspected partial results, refined the query, exported a second evidence pack, imported another annotation batch, and then rendered a report. That is why DB-object beats YAML-only: complex analyses are interactive DAGs, not static phase lists.

[proposal] A **delegation unit** belongs in this full direction, but not in the annotation substrate itself. Delegation should be a typed evidence/read unit — parent run/message/action → subordinate run/report/output → parent use — and external annotations can label delegations or their instruction messages. Do not encode delegation as a special recipe. Make it a queryable unit, then recipes can use it.

[proposal] A **measure/validity registry** should connect annotation schemas to analytics. Bead `polylogue-9l5.7` already names the right doctrine: analytics need uncertainty primitives and construct-validity metadata; every measure should declare construct, formula, evidence tier, sample-frame requirements, and confounds. Annotation schemas should reuse that discipline: a label is not just a JSON key; it is an operationalization with caveats.

## The defended recommendation

[proposal] Build **DB-native analysis recipes + annotation batches** now. Use assertions as the row-level annotation primitive. Keep YAML as import/export and as a nice demo artifact. Do not build a separate “recipe runner” silo; put it behind the same daemon-fast CLI/MCP/API contract.

Why this wins:

[evidence] The durable user tier already has an assertion model designed for authored/interpretive overlays, with status, evidence refs, staleness, context policy, and visibility. [evidence: `polylogue/storage/sqlite/archive_tiers/user.py:7-40`; `polylogue/core/enums.py:399-447`] fileciteturn3file0

[evidence] The query DSL already treats assertions as a query unit, and the read algebra frame already wants `Query × Projection × Render`, not siloed commands. [evidence: `SWARM_BRIEF.md:9-34`] fileciteturn0file9

[evidence] The service-contract work says all surfaces should converge on typed request/response DTOs and a small set of verbs. Recipe/annotation import belongs there, not as a shell-only script. [evidence: `B8_contract.md:134-155`] fileciteturn3file7

[proposal] Runner-up rejected: **YAML-only recipe files executed by external coding agents.** This is useful for demos, but insufficient as substrate. It cannot answer “which query produced this batch?”, “which model labeled these rows?”, “which assertions came from report v1?”, “what did the agent see?”, “which batches were superseded?”, or “rerun this recipe against archive epoch N and diff outputs.” Those questions require DB objects and refs.

[proposal] Also rejected as a secondary runner-up: **everything as assertions.** An annotation row is an assertion; a recipe is not. A recipe is procedure/config. A run is execution state. A batch is provenance. If all of those become assertion kinds, the epistemic layer gets overloaded and `assertions where ...` becomes a dumping ground.

## Open questions for the operator

[proposal] Should accepted external-agent annotations become `status=active` assertions automatically for trusted schemas, or should all external-agent rows enter as `candidate` and require `polylogue judge`? My lean: default `candidate`; allow schema-level `trusted_actor_refs` later.

[proposal] Should annotation schemas live in `user.db` only, or can built-in schemas ship as code/config and seed into `user.db`? My lean: built-ins as versioned bundled schemas, user overrides/copies in `user.db`.

[proposal] What is the first target grain for the demo: `session`, `message`, `block`, or a new `delegation` unit? My lean: begin with `message` plus evidence windows; add `delegation` once artifact/run linking is ready.

[proposal] Should recipe runs write every query-run automatically, or only promoted/significant query-runs? My lean: all query runs in `ops.db`; promoted/cited/query-result-producing runs in `user.db`.

[proposal] Is `analysis_run` a durable user object by default, or does it start in ops and get promoted? My lean: if launched through `recipe run`, durable by default; casual CLI search stays ops by default.

[proposal] Should YAML recipes be hand-written, generated from composer sessions, or both? My lean: both; the composer can “save as recipe,” and hand-authored prompt files can import into the same representation.

## What’s missing / assumptions to verify locally

[evidence] The snapshot is dated 2026-07-05; live code and beads may have moved. Verify locally before implementing: `git status`, `rg "class AssertionKind" polylogue/core/enums.py`, `rg "def judge_assertion_candidate" polylogue/storage/sqlite/archive_tiers/user_write.py`, `bd show polylogue-37t.12`, and current MCP/CLI command inventory.

[evidence] I did not run the daemon, MCP server, CLI, or tests. So “assertions are queryable” is grounded in code/DSL descriptors and storage methods, not a live command execution against the archive. [evidence: `polylogue/archive/query/metadata.py:792-805`; `polylogue/storage/sqlite/archive_tiers/archive.py:5393-5447`] fileciteturn3file8

[evidence] I did not fully audit the MCP write tool list in live code. `polylogue-27p` is closed and B8 names write methods, but the exact external-agent affordance available today needs local verification. [evidence: `B8_contract.md:84-124`; bead `polylogue-27p`] fileciteturn3file12

[evidence] I did not inspect every user-tier migration or parity test. The DDL and helper code show the assertion substrate exists, but migration completeness across all old overlays should be verified by the existing user-audit tests before relying on it for external annotations. [evidence: `polylogue/storage/sqlite/archive_tiers/user.py:7-49`; `polylogue/storage/sqlite/archive_tiers/user_write.py:1591-1660`] fileciteturn3file0

[proposal] The unverified claim most relevant to this design is: “a recipe-run object belongs in `user.db`, not `ops.db`.” I defended it for cited/reproducible analyses, but retention policy is an operator decision.

[proposal] The source I did not read deeply here is the current live `mcp/server` command/tool registration. That matters because the fastest path for external annotation agents may be MCP rather than CLI, even if the CLI is daemon-fast.

[proposal] The modality not run is a live annotation roundtrip: export a small evidence pack, have an agent label 5 messages, import as candidate assertions, query them back, accept one, and render it. That should be the acceptance demo for the new bead.
