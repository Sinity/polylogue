# Polylogue MCP surface collapse design

Status: append-only proposal. No source edits are implied. This design assumes the current `tests/infra/mcp.py` surface is the compatibility baseline: 96 tools, four static resource URIs in the fixture, one resource-template URI in the fixture, eight resources/templates in `server_resources.py`, and six prompts in `server_prompts.py`.

## Design stance

The public surface should shrink to a small verb algebra, but the migration must not make agents blind. Polylogue should ship three surfaces for a deprecation window:

| Surface mode | Registered tools | Use case |
|---|---:|---|
| `legacy` | current 96 names | exact compatibility, rollback, contract pinning |
| `hybrid` | verbs plus legacy wrappers | default during telemetry; wrappers emit shadow mappings |
| `verbs` | 6 role-gated verbs plus the `search` alias | target surface for new agent configs |

The target read surface is `search`, `query`, `get`, `context`, `insight`, and `explain`. `state` appears only when the server role allows write operations. `maintain` appears only when the server role allows admin operations. Keeping `state` and `maintain` role-gated avoids a read-only client seeing mutating verbs with write actions in the schema.

`search` stays as a first-class transitional verb, not because the old `search` contract is ideal, but because agents and humans pattern-match on it. It should be an alias for `query(kind="sessions", mode="search")`, with one compact schema and a short description. After telemetry proves `query` adoption, `search` can remain as a tiny always-load affordance rather than a full legacy wrapper.

Generated insight tools remain descriptor-driven. The current `INSIGHT_REGISTRY` should become an `InsightKindCatalog`: the registry continues to define CLI, MCP, and JSON contracts, but MCP exposes a single `insight(kind=..., params=...)` dispatcher plus a cacheable `polylogue://insights/catalog` resource. In hybrid mode, generated per-insight tool names keep calling the same catalog entries and log adoption telemetry.

## Target verb signatures

All verbs return the same `PolylogueMCPResponse` envelope. The existing payload builders remain authoritative for domain fields: `build_search_envelope`, `archive_session_list_payload`, `archive_search_payload`, `archive_query_unit_payload`, `archive_messages_payload`, `MCPPaginatedQueryResultPayload`, `MCPMessagesListPayload`, and `MCPErrorPayload` are wrapped rather than replaced.

```ts
type DetailLevel = "metadata" | "summary" | "compact" | "full";

type PageSpec = {
  limit?: number;           // clamped by server policy
  offset?: number;          // retained for current archive builders
  cursor?: string;          // opaque continuation handle for new clients
  offset_from?: "start" | "end";
  tail?: boolean;
};

type BudgetSpec = {
  detail?: DetailLevel;     // default: "summary"
  max_items?: number;       // default per kind
  max_bytes?: number;       // hard serialized payload ceiling
  max_tokens?: number;      // advisory ceiling for text-bearing fields
  include_bodies?: boolean; // default false except explicit message reads
  include_diagnostics?: boolean;
};

type SessionScope = {
  query?: string;
  expression?: string;
  origin?: string;
  exclude_origin?: string[];
  tag?: string[];
  exclude_tag?: string[];
  repo?: string[];
  project?: string[];
  cwd_prefix?: string;
  referenced_path?: string[];
  has_type?: string[];
  action?: string[];
  exclude_action?: string[];
  action_sequence?: string[];
  action_text?: string[];
  tool?: string[];
  exclude_tool?: string[];
  mcp_server?: string;
  model?: string;
  provider?: string;        // serialized as origin where current contracts expose origin
  title?: string;
  since?: string;
  until?: string;
  since_ms?: number;
  has_tool_use?: boolean;
  has_thinking?: boolean;
  has_paste_evidence?: boolean;
  typed_only?: boolean;
  message_type?: string;
  min_messages?: number;
  max_messages?: number;
  min_words?: number;
  max_words?: number;
  similar_session_id?: string;
  similar_text?: string;
  retrieval_lane?: "auto" | "dialogue" | "actions" | "hybrid";
  sort?: string;
  reverse?: boolean;
};

tool search(
  text?: string,
  scope?: SessionScope,
  page?: PageSpec,
  budget?: BudgetSpec
): PolylogueMCPResponse;

tool query(
  kind: "sessions" | "units" | "facets" | "neighbors" | "archive_debt",
  mode?: "search" | "list" | "aggregate" | "sample",
  scope?: SessionScope,
  unit?: "session" | "message" | "action" | "path" | "tag" | "origin",
  group_by?: string | string[],
  page?: PageSpec,
  budget?: BudgetSpec
): PolylogueMCPResponse;

tool get(
  kind:
    | "session" | "messages" | "lineage" | "raw_artifacts" | "metadata"
    | "stats" | "status" | "capabilities" | "usage" | "blackboard"
    | "assertion_claims" | "ref" | "read_view_profiles",
  ref?: string,
  view?: string,
  scope?: SessionScope,
  selector?: Record<string, unknown>,
  page?: PageSpec,
  budget?: BudgetSpec
): PolylogueMCPResponse;

tool context(
  kind: "compiled" | "image" | "preamble" | "resume_brief" | "resume_candidates" | "agent_coordination",
  seed_ref?: string,
  seed_query?: string,
  scope?: SessionScope,
  repo_path?: string,
  cwd?: string,
  recent_files?: string[],
  read_views?: string[],
  max_sessions?: number,
  max_tokens?: number,
  page?: PageSpec,
  budget?: BudgetSpec
): PolylogueMCPResponse;

tool insight(
  kind: InsightKind,
  params?: Record<string, unknown>,
  scope?: SessionScope,
  ref?: string,
  session_ids?: string[],
  group_by?: string | string[],
  page?: PageSpec,
  budget?: BudgetSpec
): PolylogueMCPResponse;

tool explain(
  kind: "import" | "query_expression" | "query_completion" | "schema" | "ref",
  input?: string,
  path?: string,
  source_name?: string,
  source_path?: string,
  query?: string,
  incomplete?: string,
  unit?: string,
  field?: string,
  budget?: BudgetSpec
): PolylogueMCPResponse;

tool state(
  kind:
    | "tag" | "mark" | "annotation" | "saved_view" | "recall_pack" | "workspace"
    | "metadata" | "correction" | "blackboard" | "session",
  action: "list" | "get" | "add" | "remove" | "save" | "delete" | "set" | "clear" | "post" | "bulk_tag",
  target?: { session_id?: string; ids?: string[]; name?: string; key?: string; ref?: string },
  payload?: Record<string, unknown>,
  confirm?: boolean,
  author?: { agent?: string; run_id?: string },
  page?: PageSpec,
  budget?: BudgetSpec
): PolylogueMCPResponse;

tool maintain(
  action: "list" | "preview" | "execute" | "status" | "rebuild_index" | "update_index" | "rebuild_session_insights",
  targets?: string[],
  scope?: SessionScope,
  dry_run?: boolean,
  operation_id?: string,
  confirm?: boolean,
  budget?: BudgetSpec
): PolylogueMCPResponse;
```

## Uniform response-budget contract

Every target verb returns this envelope. The same envelope wraps successful and failed responses; on errors the existing `MCPErrorPayload` fills `error` and `diagnostics` while `ok=false`.

```ts
type PolylogueMCPResponse = {
  ok: boolean;
  contract_version: "polylogue.mcp.response.v1";
  verb: "search" | "query" | "get" | "context" | "insight" | "explain" | "state" | "maintain";
  kind?: string;
  role: "read" | "write" | "admin";
  request: {
    normalized: Record<string, unknown>;
    redactions?: string[];
  };
  summary: {
    title: string;
    status: "complete" | "partial" | "empty" | "error";
    count?: number;
    total?: number;
    range?: string;
    omitted?: number;
    metadata_only: boolean;
    bullets: string[];        // 0-5 concise bullets, not boilerplate
  };
  data?: unknown;             // current payload builder output or bounded items
  page?: {
    limit?: number;
    offset?: number;
    next_offset?: number | null;
    cursor?: string | null;
    next_cursor?: string | null;
    has_more: boolean;
    continuation?: {
      token: string;
      verb: string;
      args: Record<string, unknown>;
      reason: string;
      expires_at?: string;
    };
  };
  refs?: Array<{ ref: string; kind: string; label?: string; uri?: string }>;
  omissions?: Array<{ reason: string; count?: number; next?: string }>;
  affordances?: Array<{
    label: string;
    verb: string;
    args: Record<string, unknown>;
    safe: boolean;
    requires_role?: "write" | "admin";
  }>;
  diagnostics?: Record<string, unknown>;
  telemetry?: { shadow_id?: string; legacy_tool?: string; mapped_to?: string };
};
```

Budget rules:

1. Default `detail="summary"`, `max_items=20`, and `include_bodies=false` for searches, list views, insights, resources, and affordances. Message reads are capped to the current `MCPMessagesListPayload` page semantics and must include `next_offset`, `offset_from`, and a cursor.
2. `metadata_only=true` means the payload contains stable identifiers, titles, timestamps, counts, origins, tags, short snippets, route state, and continuation handles, but not full message bodies or raw artifacts.
3. Every payload with truncation must set `summary.status="partial"`, `page.has_more=true`, and one continuation that exactly replays the next safe read.
4. “Next safe action” affordances are bounded. Return at most three actions. Read-only verbs only suggest read-only affordances. `state` and `maintain` actions must appear with `requires_role` and `safe=false` unless `dry_run=true` or `confirm=true` has already passed validation.
5. `archive_support.py` remains the canonical place for archive pagination semantics. The new cursor is an opaque encoding of the existing offset/filter/sort tuple plus response-shape version. Offset remains visible for compatibility.
6. `payloads.py` remains the canonical place for JSON payload models. New verbs should use a `response_budget.apply(payload, budget)` layer immediately before `hooks.json_payload`, not ad hoc trimming inside individual tools.
7. Oversized diagnostics are opt-in. `include_diagnostics=false` returns route/state names and counters only; `true` can include ranking stage details, resolver candidates, or maintenance preview internals.
8. Legacy wrappers in hybrid mode continue returning their old payload shape unless a wrapper already returns an envelope. They still emit `telemetry.shadow_id`, compare the old and new payload budgets internally, and report shape drift in ops telemetry.

## Taxonomy table: all 96 current tools

Every current name appears below. The “delete” classification always means delete only after shadow telemetry, migration warnings, and a rollback surface. No current tool is removed in the first hybrid release.

| Current tool(s) | Role today | Classification | Target public surface |
|---|---|---|---|
| `search` | read | keep-as-verb; also fold implementation into verb algebra | `search(text, scope, page, budget)` aliasing `query(kind="sessions", mode="search")` |
| `list_sessions`, `archive_list_sessions` | read | fold into verb; `polylogue://sessions` resource for default list; `archive_list_sessions` delete-after-telemetry candidate | `query(kind="sessions", mode="list", scope, page, budget)` |
| `archive_search_sessions` | read | fold into verb; delete-after-telemetry candidate | `query(kind="sessions", mode="search", scope, page, budget)` |
| `query_units` | read | fold into verb | `query(kind="units", unit, scope, page, budget)` |
| `facets` | read | fold into verb; global/default facets also resource candidate | `query(kind="facets", scope, group_by, page, budget)` |
| `neighbor_candidates` | read | fold into verb | `query(kind="neighbors", scope={similar_session_id or similar_text}, page, budget)` |
| `archive_debt` | read/generated-special | fold into verb; also insight-catalog entry | `query(kind="archive_debt", scope, page, budget)` or `insight(kind="archive_debt")` if kept in registry |
| `get_session`, `get_session_summary`, `archive_get_session` | read | fold into verb; `polylogue://session/{conv_id}` resource for summary; `get_session_summary` and `archive_get_session` delete-after-telemetry candidates | `get(kind="session", ref, view="summary|full", budget)` |
| `get_messages` | read | fold into verb; `polylogue://messages/{conv_id}` resource for default first page | `get(kind="messages", ref, page, budget)` |
| `get_session_tree`, `get_session_topology`, `get_logical_session` | read | fold into verb; session-tree/topology/logical resources for cacheable views | `get(kind="lineage", ref, view="tree|topology|logical", budget)` |
| `raw_artifacts` | read | fold into verb; full output requires explicit body budget | `get(kind="raw_artifacts", ref, selector, page, budget={detail:"full"})` |
| `resolve_ref` | read | fold into verb; optional explain companion for ambiguous refs | `get(kind="ref", ref, budget)` |
| `stats`, `get_stats_by` | read | fold into verb; `polylogue://stats` resource remains canonical for default overview | `get(kind="stats", view="overview|by", selector={by}, scope, budget)` |
| `readiness_check` | read | fold into verb; `polylogue://readiness` resource remains canonical for compact readiness | `get(kind="status", view="readiness", budget)` |
| `embedding_status`, `embedding_preflight` | read | fold into verb; status resource candidates | `get(kind="status", view="embedding|embedding_preflight", selector, budget)` |
| `provider_usage` | read | fold into verb; aggregate resource candidate | `get(kind="usage", view="provider", scope, budget)` |
| `list_read_view_profiles`, `action_affordances` | read | demote cacheable read surfaces to resources; fold into `get` for dynamic selectors | `get(kind="read_view_profiles")`; `get(kind="capabilities", view="action_affordances")` |
| `blackboard_list`, `list_assertion_claims` | read | fold into verb; blackboard status can be resource when cacheable | `get(kind="blackboard", view="list", page, budget)`; `get(kind="assertion_claims", scope, page, budget)` |
| `compile_context`, `build_context_image`, `compose_context_preamble` | read | fold into verb; `compile_context` delete-after-telemetry candidate once `context(kind="compiled")` is adopted; preamble prompt companion | `context(kind="compiled|image|preamble", seed_ref, seed_query, scope, budget)` |
| `agent_coordination`, `get_resume_brief`, `find_resume_candidates` | read | fold into verb; prompt companions for guided flows | `context(kind="agent_coordination|resume_brief|resume_candidates", scope, repo_path, cwd, budget)` |
| `explain_import`, `explain_query_expression`, `query_completions` | read | fold into verb; query-builder prompt companion | `explain(kind="import|query_expression|query_completion", input/query/incomplete, budget)` |
| `session_profiles`, `session_work_events`, `session_phases`, `threads`, `session_tag_rollups`, `archive_coverage`, `tool_usage`, `session_costs`, `cost_rollups`, `usage_timeline` | read/generated insight | fold into generated `insight` dispatcher; per-kind tool wrappers delete-after-telemetry once descriptor catalog adoption is proven | `insight(kind=<registry name>, params, scope, page, budget)` |
| `session_profile`, `session_latency_profile`, `tool_call_latency_distribution`, `session_tool_timing` | read/special insight | fold into `insight`; expose latency/tool timing schemas in insight catalog | `insight(kind="session_profile|session_latency_profile|tool_call_latency_distribution|session_tool_timing", params, budget)` |
| `find_stuck_sessions`, `workflow_shape_distribution`, `find_abandoned_sessions` | read/special insight | fold into `insight`; metadata-only default | `insight(kind="find_stuck_sessions|workflow_shape_distribution|find_abandoned_sessions", params, page, budget)` |
| `aggregate_sessions`, `compare_sessions`, `find_similar_sessions`, `correlate_sessions`, `correlate_session` | read/special insight | fold into `insight`; `compare_sessions` prompt remains guided flow | `insight(kind="aggregate_sessions|compare_sessions|find_similar_sessions|correlate_sessions|correlate_session", params, budget)` |
| `cost_outlook`, `insight_rigor_audit`, `get_postmortem_bundle`, `get_pathologies` | read/special insight | fold into `insight`; postmortem/pathology prompts become guided flows | `insight(kind="cost_outlook|insight_rigor_audit|get_postmortem_bundle|get_pathologies", params, budget)` |
| `list_tags`, `add_tag`, `remove_tag`, `bulk_tag_sessions` | write | fold into `state`; `polylogue://tags` resource remains default list; granular wrappers delete-after-telemetry candidates | `state(kind="tag", action="list|add|remove|bulk_tag", target, payload, confirm)` |
| `list_marks`, `add_mark`, `remove_mark` | write | fold into `state`; granular wrappers delete-after-telemetry candidates | `state(kind="mark", action="list|add|remove", target, payload, confirm)` |
| `list_annotations`, `save_annotation`, `delete_annotation` | write | fold into `state`; granular wrappers delete-after-telemetry candidates | `state(kind="annotation", action="list|save|delete", target, payload, confirm)` |
| `list_saved_views`, `save_saved_view`, `delete_saved_view` | write | fold into `state`; read-only saved-view catalog can later become resource; granular wrappers delete-after-telemetry candidates | `state(kind="saved_view", action="list|save|delete", target, payload, confirm)` |
| `list_recall_packs`, `save_recall_pack`, `delete_recall_pack` | write | fold into `state`; granular wrappers delete-after-telemetry candidates | `state(kind="recall_pack", action="list|save|delete", target, payload, confirm)` |
| `list_workspaces`, `save_workspace`, `delete_workspace` | write | fold into `state`; granular wrappers delete-after-telemetry candidates | `state(kind="workspace", action="list|save|delete", target, payload, confirm)` |
| `get_metadata`, `set_metadata`, `delete_metadata` | write/read-write mixed today | fold into `state` for writes and `get` for reads; delete write wrappers only after telemetry | `get(kind="metadata", target)`; `state(kind="metadata", action="set|delete", target, payload, confirm)` |
| `record_correction`, `list_corrections`, `clear_corrections` | write | fold into `state`; granular wrappers delete-after-telemetry candidates | `state(kind="correction", action="list|save|clear", target, payload, confirm)` |
| `blackboard_post` | write | fold into `state` | `state(kind="blackboard", action="post", target, payload, confirm)` |
| `delete_session` | write/destructive | fold into `state`; require `confirm=true` and preview affordance; granular wrapper delete-after-telemetry candidate | `state(kind="session", action="delete", target, confirm=true)` |
| `maintenance_list`, `maintenance_preview`, `maintenance_execute`, `maintenance_status` | admin | fold into `maintain`; granular wrappers delete-after-telemetry candidates | `maintain(action="list|preview|execute|status", targets, operation_id, dry_run, confirm)` |
| `rebuild_index`, `update_index`, `rebuild_session_insights` | admin | fold into `maintain`; granular wrappers delete-after-telemetry candidates | `maintain(action="rebuild_index|update_index|rebuild_session_insights", targets, scope, dry_run, confirm)` |

## Resources strategy

MCP resources should cover stable, URI-addressable, cacheable read state. Tools remain for model-selected operations, expensive analysis, cross-object joins, and mutations. Prompts remain for user-invoked guided workflows.

Keep and normalize the resources already implemented:

| Resource | Current backing tool or payload | Disposition |
|---|---|---|
| `polylogue://stats` | `stats` / `MCPArchiveStatsPayload` | keep; compact resource and `get(kind="stats")` for dynamic views |
| `polylogue://sessions` | `list_sessions` / `archive_session_list_payload` | keep; default list only |
| `polylogue://session/{conv_id}` | `get_session_summary` / `archive_summary_payload` | keep; summary only |
| `polylogue://tags` | `list_tags` | keep; cacheable tag counts |
| `polylogue://messages/{conv_id}` | `get_messages` / `archive_messages_payload` | keep; default first page only, no full-body dump |
| `polylogue://session-tree/{conv_id}` | `get_session_tree` / `session_tree_payload` | keep; add to expected template fixture |
| `polylogue://origin/{name}/recent` | list sessions with origin filter | keep; add to expected template fixture |
| `polylogue://readiness` | `readiness_check` / `MCPReadinessReportPayload` | keep |

Add these resources after the envelope contract exists:

| Proposed resource | Replaces default read of | Notes |
|---|---|---|
| `polylogue://read-view-profiles` | `list_read_view_profiles` | cacheable schema/capability catalog |
| `polylogue://action-affordances` | `action_affordances` | compact catalog; dynamic next actions still in verb response |
| `polylogue://embedding/status` | `embedding_status` | compact status, no preflight details |
| `polylogue://provider-usage/summary` | `provider_usage` | bounded aggregate resource |
| `polylogue://facets/global` | `facets` with no query scope | cacheable default facets only |
| `polylogue://session/{conv_id}/topology` | `get_session_topology` | cacheable lineage view |
| `polylogue://session/{conv_id}/logical` | `get_logical_session` | cacheable logical-session view |
| `polylogue://insights/catalog` | generated insight schemas | exposes `InsightKind`, parameter schemas, default budgets |
| `polylogue://mcp/aliases` | migration map | maps every legacy tool to verb/kind/action |
| `polylogue://mcp/response-contract` | design contract | machine-readable response-budget defaults |
| `polylogue://coordination/status` | `agent_coordination(view="status")` | cacheable status snapshot only |

Do not demote `raw_artifacts`, full message bodies, maintenance previews, or mutable user state to resources by default. They are too large, too dynamic, or too permission-sensitive.

## Prompts strategy

Keep all existing prompts: `analyze_errors`, `summarize_week`, `extract_code`, `compare_sessions`, `extract_patterns`, and `agent_coordination_brief`.

Add guided prompts where the human intent is a workflow rather than a data fetch:

| Prompt | Backing verbs | Why prompt instead of tool |
|---|---|---|
| `resume_work` | `context(kind="resume_candidates")`, then `context(kind="resume_brief")` | asks the model to choose a restart path |
| `session_handoff` | `context(kind="preamble")`, `get(kind="session")`, `get(kind="messages")` | converts archive state into a handoff narrative |
| `query_builder` | `explain(kind="query_expression")`, `explain(kind="query_completion")`, `query(kind="facets")` | guided construction/recovery of query expressions |
| `postmortem_review` | `insight(kind="get_postmortem_bundle")`, `insight(kind="get_pathologies")` | interpretive review, not just retrieval |
| `maintenance_plan_review` | `maintain(action="preview", dry_run=true)` | forces preview-first admin behavior |
| `insight_rigor_review` | `insight(kind="insight_rigor_audit")` | guided audit narrative with caveats |

## Migration mechanics

### Shadow telemetry

Create `mcp_tool_shadow_usage` in the disposable ops tier. The ops tier is correct because these counters are operational, lossy, and safe to rebuild.

Suggested columns:

```sql
CREATE TABLE IF NOT EXISTS mcp_tool_shadow_usage (
  id INTEGER PRIMARY KEY,
  occurred_at TEXT NOT NULL,
  server_version TEXT NOT NULL,
  surface_mode TEXT NOT NULL,          -- legacy | hybrid | verbs
  role TEXT NOT NULL,                  -- read | write | admin
  client_name TEXT,
  client_version TEXT,
  legacy_tool TEXT,                    -- null for direct verb calls
  target_verb TEXT NOT NULL,
  target_kind TEXT,
  target_action TEXT,
  request_shape_hash TEXT NOT NULL,    -- normalized schema/path hash, no raw prompt text
  response_shape_hash TEXT,
  status TEXT NOT NULL,                -- ok | error | validation_error | denied
  error_code TEXT,
  duration_ms INTEGER,
  result_count INTEGER,
  response_bytes INTEGER,
  metadata_only INTEGER NOT NULL,
  continuation_returned INTEGER NOT NULL,
  continuation_consumed INTEGER,       -- filled by joining token use
  affordance_count INTEGER,
  budget_detail TEXT,
  budget_max_items INTEGER,
  sampled INTEGER NOT NULL DEFAULT 0
);
```

What counts as usage:

1. Tool discovery does not count.
2. A completed tool call counts, even if validation fails.
3. A legacy wrapper call records `legacy_tool`, the normalized target verb/kind/action, status, result count, response bytes, and whether a continuation was returned.
4. A direct verb call records `legacy_tool=NULL` and the same target fields.
5. A continuation token consumed by a later call links back by token hash. This measures whether the budget contract is usable instead of just smaller.
6. Mutation/admin calls include authorization outcome and `confirm`/`dry_run`, but never raw payload bodies.

Adoption gates before deleting any old name from the default surface:

| Gate | Requirement |
|---|---|
| Functional equivalence | Legacy wrapper and target verb produce the same stable identifiers, counts, next offsets/cursors, and error codes under golden fixtures |
| Adoption | For each legacy family, direct verb calls are at least 80% of equivalent calls for 14 consecutive operator-days, or the operator explicitly accepts removal |
| Low legacy dependence | Fewer than 3 calls/week to the legacy name in the daily Claude/Codex workflow, excluding test runs |
| Error neutrality | Target verb has no higher validation/error rate than its legacy equivalent over the same window |
| Budget usability | Continuation-consumption rate and follow-up success do not drop versus legacy pagination flows |
| Rollback | `POLYLOGUE_MCP_SURFACE=legacy` remains supported for one additional release after default removal |

### Deprecation window

1. Release N: add verbs/resources/prompts; default `hybrid`; old names unchanged; telemetry off by default unless `POLYLOGUE_MCP_TELEMETRY=shadow`.
2. Release N+1: telemetry on by default for local ops tier; no raw text; legacy tool descriptions start with “Prefer `<verb>`”; alias map resource available.
3. Release N+2: default `verbs` if adoption gates pass; `legacy` and `hybrid` still supported by env/config.
4. Release N+3 or later: delete low-value aliases from default package only after evidence. Keep compatibility shim package or legacy mode for one more release.

### Downstream agent configs

Keep the MCP server name stable. Do not require Claude Code, Codex, or FastMCP hosts to rename the server. Config survival comes from aliasing, not from a server identity change.

Server instructions should begin with a short, self-contained line:

> Use `search`, `query`, `get`, `context`, `insight`, and `explain` for read work. Use `state` only for write-role mutation. Use `maintain` only for admin maintenance. Prefer resources for `stats`, `sessions`, `tags`, `readiness`, and session summaries.

Claude Code config should mark only `search`, `query`, `get`, `context`, `insight`, `explain`, `state`, and `maintain` as always-load candidates. Legacy wrappers should not be always-load. Codex config should keep the same server stanza and receive concise server instructions plus the `polylogue://mcp/aliases` resource. FastMCP clients should pin tests against `verbs` mode and use explicit calls rather than agent name matching.

## Test-impact matrix

| Test area | Current pin | Proposed update |
|---|---|---|
| `tests/infra/mcp.py::EXPECTED_TOOL_NAMES` | one 96-name set | split into `EXPECTED_LEGACY_TOOL_NAMES`, `EXPECTED_VERB_TOOL_NAMES`, `EXPECTED_HYBRID_TOOL_NAMES`; keep the 96-name fixture unchanged for `legacy` |
| Resource fixtures | currently under-specifies implemented templates | add `polylogue://messages/{conv_id}`, `polylogue://session-tree/{conv_id}`, and `polylogue://origin/{name}/recent` to expected templates; add new resources only behind a feature flag until implemented |
| Prompt fixtures | six names | keep six; add new prompt names only with prompt tests and docs |
| `register_tools` tests | role-gated family registration | parametrize by surface mode and role: read gets 6/96/102-ish depending mode; write adds `state` or write wrappers; admin adds `maintain` or admin wrappers |
| Per-tool contracts | individual contracts and `TOOL_MATRIX` rows | keep legacy contract tests in legacy/hybrid; add `VERB_KIND_MATRIX` and `STATE_ACTION_MATRIX`; generated insights feed `INSIGHT_KIND_MATRIX` from registry |
| Read-role authorization | read role omits every mutation/admin tool | read role must omit `state` and `maintain`; in legacy mode still omits every mutation/admin wrapper |
| Mutation tests | granular wrappers | add dual-path assertions: wrapper call and `state(kind/action)` produce same stable store effects and errors |
| Maintenance tests | granular admin tools | add dual-path assertions: wrapper call and `maintain(action)` produce same preview/status/execute payloads |
| Envelope tests | mixed payloads plus error payload | add `PolylogueMCPResponse` schema tests for every verb/kind; legacy wrappers may retain old shape until deletion |
| Insight contract tests | generated MCP tools from registry | assert `INSIGHT_REGISTRY` generates `InsightKindCatalog`, `insight.kind` enum, and per-kind param schemas; wrappers become compatibility tests |
| Query/read tests | search/list/session/message envelopes | assert `query` and `get` wrap current archive payload builders without losing `total`, `limit`, `offset`, `next_offset`, route state, or diagnostics |
| Telemetry tests | none | add ops-tier migration test, no raw text capture, one row per wrapper/direct call, continuation token correlation |
| Downstream config tests | none or ad hoc | add golden snippets for Claude Code, Codex, and FastMCP examples using stable server name and alias resource |

## Client-reality check, mid-2026

Knowledge:

1. MCP tools are model-controlled capabilities identified by name, description, input schema, annotations, and output schema. Tool names and schemas are therefore part of the selection surface.
2. MCP resources are URI-addressed, application-controlled context. They are better than tools for stable catalogs, bounded summaries, and cacheable state.
3. MCP prompts are user-controlled structured templates. They are the correct home for guided workflows like resume, postmortem, and query repair.
4. Claude Code has MCP resource access through `@` mentions and fuzzy search, prompt access through slash commands, dynamic list-change refresh, and tool search/lazy schema loading. This reduces schema-token pressure for Claude Code specifically, but it does not solve confusing names, large responses, or other hosts.
5. Codex supports MCP servers in CLI and IDE, including stdio and streamable HTTP. Its docs emphasize concise server instructions, with the first 512 characters self-contained. This argues for a compact verb story and a short alias resource.
6. The Claude API MCP connector path currently emphasizes tool calls; prompts/resources require local/client-side SDK handling. A resource/prompt demotion is therefore safe only if every demoted surface still has a tool path via `get`, `context`, `insight`, or `explain`.
7. FastMCP is a server/client framework that generates schemas from functions and supports deterministic explicit client calls. It is good for regression tests but is not proof that autonomous clients choose the right tool from 96 names.

Assumptions:

1. A flat 96-tool list is still costly for hosts that eagerly inject full schemas or do weak tool search. Even when the token cost is hidden, tool-choice accuracy suffers from near-duplicate names such as `get_session`, `get_session_summary`, `archive_get_session`, `list_sessions`, `archive_list_sessions`, and `archive_search_sessions`.
2. Parameterized verbs improve context economy only if `kind` enums are short, examples are strong, and schemas avoid giant `oneOf` blocks. A single huge dispatcher schema recreates the problem.
3. The best compromise is not “7 verbs only”; it is “7 verbs plus resources/prompts plus `search` as a high-signal alias plus telemetry-backed wrappers.”
4. Claude Code’s lazy tool loading makes deletion less urgent for context budget, but not for operator ergonomics or cross-client portability.

## Paste-ready rewrite of `polylogue-t46.8` design and acceptance criteria

### Design

Collapse Polylogue MCP from a flat 96-tool compatibility surface into a small role-gated verb algebra, without deleting any existing tool until shadow telemetry proves replacement adoption.

Target verbs:

- `search`: compatibility-first high-signal alias for session search.
- `query`: session/unit/facet/neighbor/archive-debt queries.
- `get`: bounded reads of sessions, messages, lineage, metadata, stats, status, usage, capabilities, blackboard, assertion claims, and refs.
- `context`: context image, compiled context, preamble, resume candidates/brief, and agent coordination.
- `insight`: registry-driven insight dispatcher. `INSIGHT_REGISTRY` remains the source of truth for kind names, parameter schemas, CLI contracts, MCP contracts, and JSON payload shapes.
- `explain`: import diagnostics, query-expression explanation, query completion, and schema/help views.
- `state`: write-role user-state and mutation dispatcher.
- `maintain`: admin-role maintenance dispatcher.

Expose stable read surfaces as MCP resources: stats, sessions, session summary, tags, readiness, messages, session tree, origin recent, read-view profiles, action affordances, embedding status, provider usage summary, global facets, insight catalog, alias map, and response contract. Keep full body reads, raw artifacts, dynamic joins, mutation, and maintenance as tools.

Keep existing prompts and add guided prompts for resume work, session handoff, query building, postmortem/pathology review, maintenance plan review, and insight rigor review.

All new verbs return `PolylogueMCPResponse v1`: `ok`, `verb`, `kind`, normalized request, metadata-only summary, bounded data, pagination/cursor, refs, omissions, next safe affordances, diagnostics, and telemetry. Default detail is metadata/summary, not full bodies. Every partial response carries a replayable continuation. Legacy wrappers keep their old response shape during hybrid mode but call the new verb implementation internally and emit telemetry.

Shadow telemetry lands in `ops.db` only. Count completed tool calls, not discovery. Store no raw prompt text or message bodies. Record legacy name, target verb/kind/action, status, duration, result count, response bytes, budget detail, whether a continuation was returned, and whether it was consumed.

### Acceptance criteria

1. `POLYLOGUE_MCP_SURFACE=legacy` registers exactly the current 96 expected tool names and passes all existing contract and authorization tests unchanged.
2. `POLYLOGUE_MCP_SURFACE=verbs` registers `search`, `query`, `get`, `context`, `insight`, and `explain` for read role; adds `state` for write role; adds `maintain` for admin role.
3. `POLYLOGUE_MCP_SURFACE=hybrid` registers verbs plus legacy wrappers, with wrappers delegating through the verb implementation where feasible and emitting shadow telemetry.
4. Every current tool name has an alias-map entry to one target verb/kind/action, including generated insight tools and role-gated mutation/admin tools.
5. `INSIGHT_REGISTRY` generates `InsightKindCatalog`, `insight.kind` enum coverage, and per-kind parameter-schema tests. No insight kind is hand-maintained in the MCP dispatcher.
6. All target verbs return `PolylogueMCPResponse v1`; all partial responses include either `next_offset` or `next_cursor` plus a continuation object.
7. Default responses respect the response budget: summary/metadata by default, no full message bodies unless explicitly requested, max three next-safe affordances, diagnostics compact unless requested.
8. Existing archive payload builders continue to own domain pagination and payload semantics. The response-budget wrapper must not drop stable identifiers, totals, offsets, route state, or error codes.
9. Resources include the currently implemented resource templates in test fixtures, especially messages, session tree, and origin recent. New resource fixtures are added only when implemented.
10. Read-role tests prove `state`, `maintain`, and all legacy write/admin wrappers are absent from read-only discovery.
11. Write/admin tests prove `state` and `maintain` enforce the same authorization, dry-run, confirmation, and error semantics as legacy wrappers.
12. Telemetry tests prove one ops-tier row per direct verb call and legacy wrapper call, no raw text capture, stable request-shape hashes, and continuation-token correlation.
13. Deprecation cannot remove a legacy name from the default surface until adoption gates pass: functional equivalence, 80% direct-verb adoption for the family over 14 operator-days, fewer than 3 non-test legacy calls/week, error neutrality, continuation usability, and rollback availability.
14. Downstream Claude Code, Codex, and FastMCP config examples keep the same server name and document the alias map, resources, prompts, and target verbs.
15. The bead remains append-only: it documents migration, telemetry, and acceptance criteria before any source deletion.
