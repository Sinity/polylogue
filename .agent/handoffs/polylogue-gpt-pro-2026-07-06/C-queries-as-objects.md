## [user 307]
▎ *You have the forked context: the polylogue bundle, the swarm2 daemon/composer/thin-client designs, and our discussion of queries-as-objects, delegation units, and annotation recipes. Ground every claim in the bundle (cite file:line / bead id); tag [evidence] vs [proposal]. Be opinionated — one defended recommendation plus the runner-up you rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate change, (3) the full direction. End with Open questions for the operator and What's missing (a claim unverified, a source unread, a modality not run). The snapshot is dated; live code may have moved — flag assumptions to verify locally. Don't re-derive swarm2.*

Analytical [A] over the bundle:
[A] Queries-as-first-class-objects. Full model: persist queries with ids, hierarchical query→result-set, a query referencing another, assertions attached to query objects, self-tracked usage. Storage tier, schema, DSL syntax, MCP affordances. Make our frontier concrete.

## [assistant 323]
bash -lc cd /mnt/data/poly_tree/polylogue
printf '\nUSER TABLE\n'; nl -ba polylogue/storage/sqlite/archive_tiers/user.py | sed -n '1,42p'
printf '\nASSERTION ENUMS\n'; nl -ba polylogue/core/enums.py | sed -n '399,447p'
printf '\nUPSERT ASSERTION\n'; nl -ba polylogue/storage/sqlite/archive_tiers/user_write.py | sed -n '901,1015p'
printf '\nAPI ASSERTION METHODS\n'; nl -ba polylogue/api/archive.py | sed -n '2079,2216p'
printf '\nSAVE VIEW WORKSPACE RECALL\n'; nl -ba polylogue/api/archive.py | sed -n '2271,2510p'
printf '\nASSERTION METADATA FIELDS\n'; nl -ba polylogue/archive/query/metadata.py | sed -n '500,535p'
printf '\nQUERY RUN METADATA\n'; nl -ba polylogue/archive/query/metadata.py | sed -n '780,835p'
printf '\nPROJECTION SPEC\n'; nl -ba polylogue/surfaces/projection_spec.py | sed -n '141,218p'
printf '\nSEARCH ENVELOPE\n'; nl -ba polylogue/surfaces/payloads.py | sed -n '1186,1240p'

## [assistant 337]
According to the 2026-07-05 bundle snapshot, I would make **queries-as-first-class-objects** a near-term substrate bead, not just a composer nicety. The defended recommendation is:

**[proposal] Store every committed query execution as an `ops.db` `query_run` with a stable ref and result-relation ref; promote only named/cited/saved results into durable `user.db` objects.** Do not make every query a `user.db` assertion. Assertions should attach to query/query-run/result objects; they should not be the only storage mechanism for those objects.

**Runner-up rejected:** use the existing `AssertionKind.SAVED_QUERY` / `RECALL_PACK` machinery as the whole model. It is tempting because those enum values already exist (`polylogue/core/enums.py:415-416`) and recall-pack HTTP endpoints already exist (`polylogue/daemon/user_state_http.py:445-508`). I reject it as the main model because it collapses three different things — query definition, execution event, and result relation — into a user-overlay table. It also cannot yet attach cleanly to query objects because the public object-ref vocabulary has `saved_view` and `recall_pack`, but no `query`, `query_run`, or `result_relation` kind (`polylogue/core/refs.py:8-38`, `polylogue/core/refs.py:197-213`).

I am assuming the code snapshot in `polylogue-all.tar(69).gz` is the intended bundle state, not necessarily live HEAD. The local tree should verify all path:line references before turning this into Beads or code.

## 0. Frontier in one paragraph

**[evidence]** The read side already thinks algebraically: the swarm brief states the baseline as `Query × Projection × Render`; query has boolean predicates, field predicates, FTS, semantic, structural `exists <unit>(...)`, `seq(...)`, and pipeline stages, while projection/render are separate layers. It also names `fnm.12` macros and `fnm.13` set algebra as roadmap items. `SWARM_BRIEF.md:9-34` fileciteturn1file13L9-L34

**[evidence]** The hot-daemon direction says the client should do no substrate work, should speak UDS to a warm daemon, and the composer should use `complete(partial)` and `preview(spec)` in single-digit ms. `SWARM_BRIEF.md:48-65` fileciteturn1file13L48-L65

**[evidence]** B8 already proposes a `query` verb returning `SearchEnvelope`, a `read` verb, `preview`, `complete`, `act`, and `status`, with `facets` likely separate. `B8_contract.md:134-168` fileciteturn1file9L134-L168

**[proposal]** The missing frontier is: every committed `query`/`read`/`preview→commit` should create a typed object graph:

```text
query_definition?  →  query_run  →  result_relation  →  render_artifact?
                         │              │
                         │              └─ result_members / fingerprint / sample
                         └─ assertions / annotations / analysis_step refs
```

The important jump is not “save query text.” It is **make the question, its execution, and its resulting relation addressable separately.**

## 1. With today’s substrate

### What is already possible

**[evidence]** `user.db` already has a unified assertions table. It stores `scope_ref`, `target_ref`, `kind`, `value_json`, `body_text`, `author_ref`, `author_kind`, `evidence_refs_json`, `status`, `visibility`, `confidence`, `staleness_json`, `context_policy_json`, `supersedes_json`, and timestamps. The table comment explicitly says marks, annotations, corrections, suppressions, tags, metadata, saved views, recall packs, workspaces, and blackboard notes are represented there. `polylogue/storage/sqlite/archive_tiers/user.py:7-31`

**[evidence]** The assertion kind vocabulary already includes `SAVED_QUERY`, `RECALL_PACK`, `JUDGMENT`, `RUN_STATE`, `PROMPT_EVAL`, `TRANSFORM_CANDIDATE`, and `PATHOLOGY`. Lifecycle states include `active`, `candidate`, `accepted`, `rejected`, `deferred`, `superseded`, and `deleted`. `polylogue/core/enums.py:399-447`

**[evidence]** The write path can upsert assertions with explicit target, scope, author, evidence refs, status, visibility, confidence, staleness, context policy, and supersession. `polylogue/storage/sqlite/archive_tiers/user_write.py:901-999`

**[evidence]** The API already exposes assertion read/review affordances: `list_assertion_claims`, `list_assertion_claim_payloads`, `list_assertion_candidates`, `list_assertion_candidate_reviews`, and `judge_assertion_candidate`. `polylogue/api/archive.py:2079-2206`

**[evidence]** Assertion rows are queryable as terminal query units: `QueryUnitKind` includes `assertion`, and `AssertionQueryRowPayload` exposes target/scope/kind/key/body/value/author/status/visibility/evidence/staleness/context fields. `polylogue/surfaces/payloads.py:1252-1261`, `polylogue/surfaces/payloads.py:1407-1426`

**[evidence]** The assertion query metadata includes structural fields for `author`, `author_kind`, `body`, `context`, `evidence`, `key`, `kind`, `scope`, `scope_ref`, `status`, `target`, `target_ref`, `text`, `time`, `value`, and `visibility`. `polylogue/archive/query/metadata.py:507-525`

So today, an external demo can already fake the workflow:

1. Run a query through CLI/API/MCP.
2. Save the result-set manually as a recall pack, or as an external JSON artifact.
3. Ask an agent to annotate sessions/messages/actions/assertions.
4. Import the annotations as assertion rows targeting existing refs.
5. Query `assertions where ...` for aggregate analysis.

This is enough for a crude Fable/delegation demo, but not a clean substrate demo.

### What is blocked today

**[evidence]** Public object refs include `session`, `message`, `block`, `attachment`, `paste_span`, `work_event`, `phase`, `thread`, `file`, `branch`, `commit`, `check-run`, `workspace`, `agent`, `user`, `repo`, `insight`, `run`, `context-snapshot`, `observed-event`, `assertion`, `saved_view`, `recall_pack`, `transform`, `tool-call`, `subagent-report`, and GitHub refs. There is no `query`, `query_run`, `query_definition`, `result_relation`, `cohort`, `analysis`, or `annotation_batch`. `polylogue/core/refs.py:8-38`

**[evidence]** `normalize_object_ref_text` parses and validates object refs against that closed kind map, so a string like `query_run:abc` is not merely undocumented; it should fail as an unsupported object-ref kind until the registry expands. `polylogue/core/refs.py:197-213`

**[evidence]** `ops.db` currently has operational tables such as `ingest_cursor`, `ingest_attempts`, `convergence_debt`, `cursor_lag_samples`, `daemon_stage_events`, `daemon_events`, `embedding_catchup_runs`, and OTLP tables, but no `query_runs` / `result_relations` table. `polylogue/storage/sqlite/archive_tiers/ops.py:10-158`

**[evidence]** User-state target kinds are still reader-oriented: session, message, work_event, thread, block, attachment, paste_span, plus some storage-only kinds. There is no query or result target there either. `polylogue/core/user_state_targets.py:1-22`, `polylogue/core/user_state_targets.py:52-115`

**[evidence]** C10’s composer design already proposed recall entries: every commit writes query text, resolved spec, result fingerprint, and timestamp to `user.db`; naming a recall entry promotes it to a macro; recalled entries store resolved specs, not only text. `C10_composer_ux.md:221-230` fileciteturn1file15L221-L230

**[proposal]** Today’s limitation is therefore precise: Polylogue can store **claims about existing archive objects**, and can store **recall packs**, but it cannot yet make the **query act itself** a typed, ref-addressable archive object. That means no clean `assertion.target_ref = query_run:...`, no query-run lineage, no query-object composition, no “from this prior result” syntax, and no self-audit of query usage beyond generic daemon/OTLP events.

## 2. Near-term substrate change

This is the concrete implementation frontier I would turn into a bead.

### Storage-tier decision

**[proposal]** Use **`ops.db` for automatic query telemetry** and **`user.db` for promoted durable analytical objects**.

`ops.db` should receive every committed query run by default because query executions are operational events. They are high-volume, disposable-ish, and useful for latency/debug/self-observation. This matches the existing role of `ops.db` as disposable daemon telemetry and operational state (`polylogue/storage/sqlite/archive_tiers/ops.py:1-158`).

`user.db` should receive durable objects only when the operator or an agent intentionally saves/promotes them: saved query definitions, named cohorts, cited result snapshots, analysis runs, reports, and assertions about any of those. This matches `user.db`’s role as irreplaceable overlay state, and it avoids turning ordinary `polylogue find ...` usage into permanent clutter.

The doctrine from the compaction memory agrees with this shape: durable `source.db`/`user.db` use explicit additive migrations, derived tiers rebuild, and `ops.db` remains disposable bootstrap/ALTER territory. `Pasted text(35).txt:73-77` fileciteturn0file0L73-L77

### Minimal schema

**[proposal]** Add these `ops.db` tables:

```sql
CREATE TABLE query_runs (
    query_run_id          TEXT PRIMARY KEY,
    parent_query_run_id   TEXT,
    actor_ref             TEXT,
    actor_kind            TEXT,
    surface               TEXT NOT NULL,       -- cli|mcp|tui|web|api|agent
    verb                  TEXT NOT NULL,       -- query|read|preview_commit|facets
    input_text            TEXT,
    query_request_json    TEXT NOT NULL,
    lowered_spec_json     TEXT,
    projection_spec_json  TEXT,
    render_spec_json      TEXT,
    archive_root_hash     TEXT,
    archive_epoch_json    TEXT,
    started_at_ms         INTEGER NOT NULL,
    finished_at_ms        INTEGER,
    status                TEXT NOT NULL,       -- running|completed|failed|cancelled
    degraded_json         TEXT NOT NULL DEFAULT '{}',
    diagnostics_json      TEXT NOT NULL DEFAULT '{}',
    result_relation_id    TEXT,
    render_artifact_refs_json TEXT NOT NULL DEFAULT '[]',
    trace_id              TEXT,
    span_id               TEXT
) STRICT;
```

```sql
CREATE TABLE result_relations (
    result_relation_id    TEXT PRIMARY KEY,
    query_run_id          TEXT NOT NULL,
    grain                 TEXT NOT NULL,       -- session|message|action|block|assertion|file|run|...
    identity_policy       TEXT NOT NULL,       -- physical|logical|content|lineage-root
    member_storage        TEXT NOT NULL,       -- fingerprint|sample|exact_inline|exact_table|dynamic
    total_count           INTEGER,
    count_precision       TEXT NOT NULL,       -- exact|estimate|capped|unknown
    result_fingerprint    TEXT,
    sample_refs_json      TEXT NOT NULL DEFAULT '[]',
    capped_at             INTEGER,
    created_at_ms         INTEGER NOT NULL
) STRICT;
```

```sql
CREATE TABLE result_relation_members (
    result_relation_id    TEXT NOT NULL,
    ordinal               INTEGER NOT NULL,
    object_ref            TEXT NOT NULL,
    score                 REAL,
    payload_json          TEXT,
    PRIMARY KEY(result_relation_id, ordinal)
) STRICT;
```

Use `result_relation_members` only for exact or deliberately snapshotted relations. Most routine query runs should store a fingerprint + sample refs, not the entire member set.

**[proposal]** Add these durable `user.db` tables or assertion-backed equivalents:

```sql
CREATE TABLE query_definitions (
    query_definition_id   TEXT PRIMARY KEY,
    name                  TEXT UNIQUE,
    input_text            TEXT NOT NULL,
    normalized_spec_json  TEXT NOT NULL,
    default_projection_json TEXT,
    default_render_json   TEXT,
    author_ref            TEXT NOT NULL,
    visibility            TEXT NOT NULL DEFAULT 'private',
    created_at_ms         INTEGER NOT NULL,
    updated_at_ms         INTEGER NOT NULL
) STRICT;
```

```sql
CREATE TABLE cohorts (
    cohort_id             TEXT PRIMARY KEY,
    name                  TEXT UNIQUE,
    kind                  TEXT NOT NULL,       -- dynamic|snapshot
    grain                 TEXT NOT NULL,
    identity_policy       TEXT NOT NULL,
    query_definition_id   TEXT,
    source_query_run_id   TEXT,
    result_relation_id    TEXT,
    member_count          INTEGER,
    member_fingerprint    TEXT,
    description           TEXT,
    created_at_ms         INTEGER NOT NULL,
    updated_at_ms         INTEGER NOT NULL
) STRICT;
```

I would still allow `AssertionKind.SAVED_QUERY` and `AssertionKind.RECALL_PACK` as compatibility/projection rows, but the durable primary keys should be typed rows if we want a serious algebra.

### Object-ref expansion

**[proposal]** Add public ref kinds:

```text
query-definition:<id>
query-run:<id>
result-relation:<id>
cohort:<id>
analysis:<id>
analysis-step:<id>
annotation-batch:<id>
report:<id>
```

Minimum first slice: `query-run`, `result-relation`, `query-definition`, `cohort`.

This unlocks assertions attached to queries:

```json
{
  "target_ref": "query-run:qr_20260705_abc",
  "scope_ref": "analysis:fable_delegate_tone_v1",
  "kind": "judgment",
  "key": "methodology.validity",
  "value_json": {"label": "biased_sample", "severity": "medium"},
  "evidence_refs_json": ["result-relation:rr_20260705_def"],
  "author_ref": "agent:gpt-5.5-pro",
  "author_kind": "agent",
  "status": "candidate"
}
```

Strictly, `evidence_refs_json` currently normalizes public refs through `normalize_public_ref_text`, which today accepts object refs or evidence refs. Once `result-relation` is an object-ref kind, it can participate. `polylogue/storage/sqlite/archive_tiers/user_write.py:941-946`, `polylogue/core/refs.py:190-206`

### Service contract change

**[evidence]** B8 wants all client/substrate communication through typed DTO verbs and specifically says `query` should collapse list/search/count into one response envelope, while `act` should absorb mutations including annotations/corrections/saved views/judgments/ingest. `B8_contract.md:151-168`, `B8_contract.md:231-241` fileciteturn1file9L151-L168 fileciteturn1file9L231-L241

**[proposal]** Extend those DTOs, not the method count.

`query(QueryRequest) -> QueryResponse` should return:

```json
{
  "query_run_ref": "query-run:...",
  "result_relation_ref": "result-relation:...",
  "grain": "session",
  "total": 142,
  "count_precision": "exact",
  "items": [...],
  "diagnostics": {...},
  "route_state": {...}
}
```

For backward compatibility, this can initially be `SearchEnvelope` plus optional `query_run_ref` and `result_relation_ref`.

`preview(QueryRequest)` should not persist every keystroke by default. It can return an ephemeral `preview_id` for cancellation/debugging, but only **commit** creates a `query_run` unless `debug_record_previews=true`.

`read(ReadRequest)` should accept `selection_ref` as an alternative to inline query:

```json
{
  "selection_ref": "result-relation:rr_...",
  "projection": {...},
  "render": {...}
}
```

`act(ActionRequest)` should gain these kinds:

```text
save_query_definition
save_cohort
promote_result_relation
attach_assertion
import_annotation_batch
delete_query_run        -- probably ops/admin only
pin_query_run           -- promote from ops to user-visible
```

### DSL syntax

**[evidence]** Existing design already has `@` macro completion and `@arm_pack | except (@arm_raw)` examples for composed cohorts/macros. `C10_composer_ux.md:223-230` fileciteturn1file15L223-L230

**[proposal]** Keep `@name` for dynamic saved query definitions/macros. Do not overload it for every result.

Recommended DSL:

```text
@macro
```

expands a dynamic saved query definition.

```text
from query-definition:<id>
```

re-evaluates the saved query definition.

```text
from query-run:<id>
```

uses the result relation produced by that specific run if retained; otherwise errors with “run has no retained result relation.”

```text
from result-relation:<id>
```

uses a stored relation directly.

```text
from cohort:<name-or-id>
```

uses a durable named cohort. If the cohort is dynamic, it re-evaluates; if snapshot, it yields stored members.

Examples:

```text
from query-run:qr_fable_candidates_001
  | intersect (@strict_directives)
  | read with assertions
```

```text
from result-relation:rr_20260705_fable_delegations
  | group by assertion.value.directive_intensity
  | count
```

```text
@fable_delegations
  | except (from cohort:claude_family_subagents)
  | read layout:delegation-card
```

**[proposal]** A query referencing another query should lower to a typed AST node:

```text
RefOperand(kind=query_definition|query_run|result_relation|cohort, ref=...)
```

Do not textual-expand everything. Textual macro expansion is fine for `@macro`, but `from query-run:<id>` should remain a provenance-preserving reference node so `explain` can show the parent query/run/result lineage.

### MCP affordances

**[proposal]** MCP should expose ergonomic tools even if internally they all call the B8 `query/read/act` verbs.

Minimum:

```text
polylogue_query
  input: QueryRequest
  output: QueryResponse with query_run_ref/result_relation_ref

polylogue_read
  input: ReadRequest, including selection_ref
  output: ReadEnvelope/rendered artifact ref

polylogue_save_query
  input: name, query_run_ref or input_text/spec
  output: query_definition_ref

polylogue_save_cohort
  input: name, result_relation_ref, kind=snapshot|dynamic
  output: cohort_ref

polylogue_assert
  input: target_ref, scope_ref, kind, key, value/body, evidence_refs
  output: assertion_ref

polylogue_import_annotation_batch
  input: schema_id, scope_ref, rows[]
  output: annotation_batch_ref + assertion refs

polylogue_get_query_run
  input: query_run_ref
  output: full envelope, parent refs, result refs

polylogue_explain_ref
  input: query_definition|query_run|result_relation|cohort ref
  output: provenance/explain tree
```

This makes the agent loop easy:

```text
query → get result_relation_ref
read evidence pack for that relation
annotate rows
import annotation batch
query assertions scoped to the batch
save cohort/report
```

That is the scalable version of the external annotation recipe you described.

## 3. Full direction

The full design is not “query history.” It is **analysis provenance**.

**[proposal]** The object graph should eventually include:

```text
analysis
  ├─ analysis_step
  │    ├─ query_run
  │    │    └─ result_relation
  │    ├─ evidence_pack / render_artifact
  │    ├─ annotation_batch
  │    │    └─ assertions
  │    └─ report
  └─ final cohort(s)
```

Queries become one node type in a larger analysis DAG. That matters because complex analyses are loops, not static YAML:

1. Agent runs broad query.
2. Sees noisy results.
3. Saves a narrowed query.
4. Exports evidence pack.
5. Annotates evidence.
6. Queries annotations.
7. Finds counterexamples.
8. Revises schema.
9. Writes report.

A YAML recipe can describe the intended loop, but the durable truth is the run graph.

**[proposal]** This graph also solves “Polylogue tracks its own use.” The CLI, TUI, web workbench, MCP, and external agents all become actors that produce query runs, reads, actions, annotations, cohorts, reports, and context packs. Polylogue can then analyze:

```text
query-runs where actor_kind:agent and status:failed
```

```text
query-runs where degraded:true | group by degraded.reason | count
```

```text
analysis-runs where report_ref:* and query_run.count_precision:capped
```

```text
assertions where target_ref:query-run:* and key:methodology.caveat
```

**[proposal]** The full DSL should treat prior query results as first-class relations, not as saved text snippets. A query result has grain. That makes `result-relation:...` algebraically valid only when the downstream operation accepts its grain. A message-result relation cannot silently flow into a session-only action. This follows the set-algebra / composer doctrine in C10: grain mismatch should fail closed and surface suggestions rather than broaden silently. `C10_composer_ux.md:206-230` fileciteturn1file15L206-L230

**[proposal]** Continuous queries are then just installed query definitions with triggers:

```text
continuous_query {
  query_definition_ref
  schedule_or_event
  action: save_result_relation | notify | create_candidate_assertions | refresh_cohort
}
```

Do not build continuous queries in the first slice. The first slice should only make committed query runs and saved result relations real.

## Concrete bead I would create

**Title:** First-class query runs and result relations

**Parent:** likely `polylogue-fnm` or `polylogue-t46`; I lean `t46` if the first deliverable is protocol DTOs, `fnm` if the first deliverable is DSL/reference syntax. It also relates to `37t.12` because external annotation batches need query/result targets. Existing relevant bead IDs: `polylogue-fnm.12` user-defined macros, `polylogue-fnm.13` set algebra, `polylogue-4p1` one read algebra, `polylogue-t46` contracts own surfaces, `polylogue-37t.12` judgment queue.

**Description:** Queries currently execute as transient operations or composer recall entries. Agents and operators need addressable query definitions, query runs, and result relations so analysis can be iterative, citable, annotatable, and composable.

**Design:**

1. Add ObjectRef kinds `query-definition`, `query-run`, `result-relation`, `cohort`.
2. Add `ops.db.query_runs`, `ops.db.result_relations`, and optional `ops.db.result_relation_members`.
3. Extend `SearchEnvelope` / query response with `query_run_ref`, `result_relation_ref`, `grain`, `count_precision`.
4. Add `user.db.query_definitions` and `user.db.cohorts`, or an equivalent assertion-backed compatibility layer with typed rows as primary.
5. Add `act` kinds `save_query_definition`, `save_cohort`, `promote_result_relation`, `attach_assertion`, `import_annotation_batch`.
6. Add DSL support for `from query-definition:<id>`, `from query-run:<id>`, `from result-relation:<id>`, and `from cohort:<id-or-name>`.
7. Add MCP tools for query, save-query, save-cohort, attach/import assertions, and explain-ref.
8. Add retention policy: previews not recorded by default; committed queries recorded in ops; promoted objects durable in user.

**Acceptance criteria:**

A CLI/MCP query returns a stable `query_run_ref` and `result_relation_ref`.

`polylogue read --selection result-relation:<id>` renders the same relation with a different projection.

`assertions` can target `query-run:<id>` and `result-relation:<id>`.

`@macro` still expands saved query definitions; `from query-run:<id>` does not textual-expand and preserves provenance in `explain`.

A result relation can be promoted to a snapshot cohort and later used as a set-op operand.

Routine preview keystrokes are not persisted unless a debug flag is set.

A migration test proves `user.db` additive migration preserves existing assertions and recall packs.

A parity test proves CLI and MCP receive the same refs for the same committed query through the shared service contract.

## Why this is the right first move

**[evidence]** The strategic brief says finished demos prove the substrate is honest, but not yet that an agent/operator completes a real task better with it; the missing gate is a committed, reproducible artifact. Query objects directly serve that gate because they let the artifact cite exactly which queries, result sets, annotation batches, and reports produced the analysis. `situation-brief(2).md:67-79`, `situation-brief(2).md:133-137` fileciteturn0file10L67-L79 fileciteturn0file10L133-L137

**[evidence]** The B7 thinning audit says the CLI currently has a second engine and must shrink to formatter/client over shared payloads; it also says query schema/parse contract and projection-as-input are first blockers. `B7_thinning.md:24-46`, `B7_thinning.md:153-168` fileciteturn0file7L24-L46 fileciteturn0file7L153-L168

**[proposal]** Query objects are a low-risk bridge between today’s evidence/archive strength and tomorrow’s analysis workbench. They do not require solving the whole recipe language, work trace replay, or continuous-query engine. They only require making the already-central operation — querying — leave behind a typed, reusable object.

## Open questions for the operator

1. Should **every committed query** be recorded in `ops.db` by default, or only queries with `--record` / agent/MCP callers? I recommend every committed query, with retention policy, because self-observation is the point; do not record every preview keystroke.

2. Should durable saved query definitions live in a new `user.db.query_definitions` table, or should the existing `AssertionKind.SAVED_QUERY` be the primary storage? I recommend a dedicated table plus assertion compatibility/projection, because query definitions are state/programs, while assertions are claims/judgments.

3. Should `@name` resolve both macros and cohorts, or should cohorts use a visibly different namespace? I recommend `@name` for dynamic query macros and `from cohort:<name>` for cohorts at first, because conflating dynamic query definitions and snapshot cohorts will create surprising freshness behavior.

4. How much of a result relation should be stored by default: fingerprint + sample, exact member list, or only exact member list when promoted? I recommend fingerprint + sample in `ops.db`; exact members only for bounded results or promoted snapshot cohorts.

5. Is `query-run:<id>` allowed as a long-lived public ref if the underlying `ops.db` row may expire? I recommend yes, but expired refs should resolve to a typed “expired operational ref” payload, not silently disappear.

6. Should annotation batches be a first-class table in the first query-object slice, or should imports just create assertions with a shared `scope_ref`? I recommend a first-class `annotation_batch` soon after query runs, but not necessarily in the same PR.

## What’s missing

I did not run the daemon, tests, or live CLI; this is static analysis of the uploaded snapshot. Live HEAD may already have moved.

I did not inspect the full `docs/design/query-set-algebra.md`; I relied on the swarm brief and C10 summary for set-op semantics.

I did not verify whether `AssertionKind.SAVED_QUERY` is currently surfaced by CLI/MCP as a real user-facing saved-query feature or only exists in the enum/storage substrate.

I did not verify Beads after the final swarm apply pass against live `bd`; the bead IDs cited here come from the bundled export/snapshot and the transcript, so priority/status may be stale.

I did not inspect generated OpenAPI/MCP tool schemas, so MCP tool names above are proposed affordances, not confirmed current tools.

The one unverified claim most worth checking locally is whether `normalize_object_ref_text("query-run:...")` fails exactly as expected in current HEAD; based on `polylogue/core/refs.py:8-38` and `:197-213`, it should, but live code may have added the kind.
