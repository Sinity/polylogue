# Evidence and usage-ranking method

## Authority and observed facts

| Evidence | Observed fact | Consequence |
|---|---|---|
| Snapshot `polylogue-manifest.json` | Generated 2026-07-17T18:09:50Z from `master` `536a53efac0c`, `dirty=true`; the extracted checkout was clean at inspection. | Current extracted source is primary; no claim is made about a live daemon/archive. |
| `tests/infra/mcp.py:16-128` | Frozen baseline and declaration-derived admin set contain 104 names. | The request's 103 count is stale; map 104. |
| Git commit `b6c78adfc` | Added read-role `named_source_freshness` on 2026-07-16. | This is the 104th row and was absent from the older usage census. |
| Git commit `ed44be18f448` | Landed declaration registry/generated map for 104 tools and explicitly called itself “foundation, not migration.” | Reuse the declarations; do not mistake them for runtime parity evidence. |
| `polylogue/mcp/declarations/registry.py:3376-3449` | Target has seven default reads because `graph` is a separately named transaction with verb `get`. | Fold graph into `get(view="graph.*")` to meet six-tool contract without losing continuation semantics. |
| `polylogue/mcp/declarations/registry.py:3451-3488`, `polylogue/mcp/declarations/models.py:11-18` | Privileged transactions are `write`, `judge`, `run`, `maintenance`; roles form read < write < review < admin. | Expected exact role surfaces are 6/8/9/10. |
| `docs/generated/mcp-equivalence.json` | Role counts: {'read': 66, 'write': 29, 'review': 2, 'admin': 7}; semantics counts: {'single_object': 14, 'mutation': 23, 'aggregate': 16, 'exhaustive_page': 23, 'top_k': 7, 'bounded_context': 11, 'recursive_graph': 3, 'maintenance': 7}. Declaration `observed_use`: {'unknown': 99, 'not_observed': 2, 'observed': 3}. | Hand-coded usage metadata is materially under-classified: only three names are observed and two not observed. Generate it from a census artifact. |
| Beads `polylogue-t46.8.1` | Still open although its foundation landed; AC forbids deletion in that slice. | Tracker should be reconciled, but source state controls the report. |
| Beads `polylogue-t46.8.2`, `.2.1` | Read migration remains open; `.2.1` records zero filter gap and zero captured use for two archive aliases. | Those two rows are mapping-only and first deletion candidates after the atomic cutover. |
| Bead `polylogue-t46.8.3` | Requires candidate/judged state, dry-run/auth/apply/receipt/reconcile, idempotency, and role isolation. | Privileged comparison is lifecycle/state equivalence, not byte equality. |
| Bead `polylogue-s1kr` | Operation-level Python parity matrix remains open. | Regenerate MCP/Python binding evidence after names collapse. |

## MCP call-log schema and its limits

`polylogue/storage/sqlite/archive_tiers/ops.py:172-190` defines `mcp_call_log(call_id, tool_name, session_id, started_at_ms, finished_at_ms, duration_ms, success, error_detail)` with tool/time and start indexes. `polylogue/storage/sqlite/archive_tiers/ops.py:216-224` adds call-to-session membership. `polylogue/storage/sqlite/archive_tiers/ops_write.py:908-995` records idempotently by `call_id`, records primary/member refs, and prunes rows older than 90 days whenever a new call is written. `polylogue/storage/sqlite/archive_tiers/ops_write.py:998-1026` supports newest-first tool/session reads.

`polylogue/mcp/call_log.py:32-43` shows the event contains no arguments and no result body. Therefore telemetry can rank tools and failure/latency burden, but it **cannot** recreate production inputs or capture old outputs. Goldens must come from a deterministic fixture and explicit case manifest. The outbox status exposes pending/quarantined debt (`polylogue/mcp/call_log.py:53-64`, `polylogue/mcp/call_log.py:182-207`); a zero-call result is valid removal evidence only when delivery debt is empty or adjudicated.

The server spools calls through `_safe_call`/`_async_safe_call` (`polylogue/mcp/server_support.py:439-471`, `polylogue/mcp/server_support.py:524-637`) using bare server tool names. Historical session action evidence uses qualified client names such as `mcp__polylogue__search`; analytics needs a normalization map, but runtime aliases are prohibited.

Retention is a maximum, not guaranteed horizon completeness: pruning happens on writes, old rows may remain if traffic stops, and a fresh installation may have fewer than 90 days. Every census must report min/max timestamps, row count, outbox pending/quarantined counts, and the actual observed-day horizon. “Never called” means zero calls over the complete available retention horizon after delivery debt is resolved—not merely zero in one client log.

## Exact ranking query

Materialize `temp.mcp_tool_inventory` from `MCP_TOOL_DECLARATIONS` plus a reviewed `criticality` classification, then run this read-only query against `<archive-root>/ops.db`:

```sql
-- The harness first creates temp.mcp_tool_inventory from the executable declarations:
-- (tool_name TEXT PRIMARY KEY, minimum_role TEXT, criticality TEXT)
-- criticality is one of ordinary, continuity, incident, destructive, authority.
WITH
params AS (
  SELECT CAST(strftime('%s','now','-90 days') AS INTEGER) * 1000 AS since_ms
),
calls AS (
  SELECT c.*
  FROM mcp_call_log AS c, params AS p
  WHERE c.started_at_ms >= p.since_ms
),
call_rollup AS (
  SELECT
    tool_name,
    COUNT(*) AS call_count,
    SUM(success) AS success_count,
    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failure_count,
    COUNT(DISTINCT date(started_at_ms / 1000, 'unixepoch')) AS active_days,
    MAX(started_at_ms) AS last_call_at_ms,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms
  FROM calls
  GROUP BY tool_name
),
session_membership AS (
  SELECT call_id, session_id FROM calls WHERE session_id IS NOT NULL
  UNION
  SELECT r.call_id, r.session_id
  FROM mcp_call_session_refs AS r
  JOIN calls AS c ON c.call_id = r.call_id
),
session_rollup AS (
  SELECT c.tool_name, COUNT(DISTINCT m.session_id) AS distinct_sessions
  FROM calls AS c
  LEFT JOIN session_membership AS m ON m.call_id = c.call_id
  GROUP BY c.tool_name
),
duration_rank AS (
  SELECT
    tool_name,
    duration_ms,
    row_number() OVER (PARTITION BY tool_name ORDER BY duration_ms) AS rn,
    count(*) OVER (PARTITION BY tool_name) AS n
  FROM calls
),
p95 AS (
  SELECT tool_name, MIN(duration_ms) AS p95_duration_ms
  FROM duration_rank
  WHERE rn >= CAST((n * 95 + 99) / 100 AS INTEGER)
  GROUP BY tool_name
)
SELECT
  i.tool_name,
  i.minimum_role,
  i.criticality,
  COALESCE(c.call_count, 0) AS call_count,
  COALESCE(c.success_count, 0) AS success_count,
  COALESCE(c.failure_count, 0) AS failure_count,
  COALESCE(s.distinct_sessions, 0) AS distinct_sessions,
  COALESCE(c.active_days, 0) AS active_days,
  c.last_call_at_ms,
  ROUND(c.avg_duration_ms, 1) AS avg_duration_ms,
  p.p95_duration_ms,
  c.max_duration_ms,
  CASE
    WHEN COALESCE(c.call_count, 0) > 0 THEN 'parity-golden'
    WHEN i.criticality <> 'ordinary' OR i.minimum_role IN ('review','admin') THEN 'parity-golden'
    ELSE 'mapping-only'
  END AS recommended_disposition
FROM temp.mcp_tool_inventory AS i
LEFT JOIN call_rollup AS c ON c.tool_name = i.tool_name
LEFT JOIN session_rollup AS s ON s.tool_name = i.tool_name
LEFT JOIN p95 AS p ON p.tool_name = i.tool_name
ORDER BY
  CASE
    WHEN COALESCE(c.call_count, 0) > 0 THEN 0
    WHEN i.criticality <> 'ordinary' OR i.minimum_role IN ('review','admin') THEN 1
    ELSE 2
  END,
  COALESCE(c.call_count, 0) DESC,
  COALESCE(s.distinct_sessions, 0) DESC,
  COALESCE(c.active_days, 0) DESC,
  COALESCE(c.failure_count, 0) DESC,
  c.last_call_at_ms DESC,
  i.tool_name;
```

The ordering is intentionally lexicographic rather than an opaque weighted score:

1. Actually called tools first.
2. Zero-call but continuity/incident/destructive/authority-critical or review/admin tools next.
3. Ordinary zero-call tools last.
4. Within a tier: calls, distinct sessions, active days, failures, recency, then name.

Disposition rule: any observed call is parity-golden. A zero-call critical/privileged operation remains parity-golden for authorization, idempotency, and recovery. An ordinary zero-call operation is mapping-only. Any mapping-only row found in live telemetry is automatically promoted before deletion.

## Committed usage evidence available in the snapshot

`.agent/demos/agent-affordance-usage/README.md` and `surface-inventory.csv` were generated 2026-07-05 from `/home/sinity/.local/share/polylogue`, using the `grouped-tool-name-recent-window` action scope. They are real captured archive evidence, but not the current `ops.db` 90-day census.

For current tools, that artifact records **30 names with actions**, **65 with zero actions**, and **9 current names absent because they were newer**. Examples: `search` 50 actions/16 sessions; `stats` 8/6; `query_units` 5/1; `get_session_summary` 5/4. The committed `surface-inventory.csv` is authoritative for the exact table values used in `REPORT.md`; `tool-counts.csv` is a narrower recent ranking and records 4 summary calls in its displayed window. This difference is why the final gate must use one declared census query/horizon.

The declaration registry currently labels only `search`, `query_units`, and `list_sessions` observed; `archive_list_sessions` and `archive_search_sessions` not observed; 99 unknown. At least 27 other tools have observed actions in the committed census. `observed_use` should become a generated census reference (timestamp, horizon, count, artifact hash), not a manually maintained enum.

## Contract and completeness evidence

| Contract owner | What it proves now | Required cutover change |
|---|---|---|
| `tests/infra/mcp.py::MCP_TOOL_NAME_BASELINE` | Independent frozen 104-name compatibility oracle. | Move old list into parity manifest; replace runtime baseline with exact role-scoped 6/8/9/10 sets. |
| `tests/unit/mcp/test_envelope_contracts.py::TOOL_CONTRACT` | Every old tool has an output-kind classification and stale/missing rows fail. | Add canonical envelope contract for 10 tools; archive old classifications with goldens. |
| `tests/unit/mcp/test_tool_contracts.py` | Live input schemas/tool descriptions. | Assert exact six/four schemas and generated manual. |
| `tests/unit/mcp/test_per_tool_contracts.py` | Happy/invalid mutation and per-tool contract cases. | Convert required old cases into parity manifest rows before deleting handlers. |
| `tests/unit/mcp/test_tool_declarations.py` | Exact old role counts, migration ownership, and current seven-read declaration. | Assert six reads, four privileged, exact resources/prompts, and no retired runtime registration. |
| `tests/unit/mcp/test_server_surfaces.py` | Tools exact; resources/prompts only missing-entry checks. | Make all role-scoped discovery sets exact and test prompt dependency role validity. |
| `devtools/render_mcp_equivalence.py` | Generated JSON follows registry. | Add disposition, census artifact reference/hash, case IDs, normalization profile, and target runtime binding. |
| `devtools/render_mcp_tool_index.py` + docs coverage | Every current tool literal appears in docs. | Generate current six/four index; keep old names only in a historical parity section excluded from current-surface claims. |

The declaration commit's focused command reported 384 passing tests and `devtools verify --quick` at that commit. This analysis independently reran `PYTHONPATH=. python -m devtools render mcp-equivalence --check` and `render mcp-tool-index --check`; both returned `sync OK`. A focused pytest collection attempt did not start because the analysis container lacks `hypothesis`, so no new test-pass claim is made. The full suite and live archive remain implementation-lane gates.

## Prompt/resource contradictions

Runtime resources in `polylogue/mcp/server_resources.py` include stats, sessions, one-session, tags, two capability resources, messages-by-session, session tree, origin recent, readiness, and raw-authority continuation resources. The target declaration instead names exact session/message/block/action/file/query/result-set/recall-pack resources. The foundation commit explicitly says it did not wire target resources/prompts. Exact parity must be implemented and tested; generated declarations are not runtime proof.

Runtime has 12 prompts (`tests/infra/mcp.py:143-156`) while `TARGET_PROMPTS` declares seven. The five server-side data prompts have no usage telemetry in `mcp_call_log`. Retiring them in the tool cutover would be unsupported. The seven workflow prompts also hard-code old tool names in `polylogue/mcp/server_prompts.py:455-546` and must be rewritten.

`unacknowledged_failures` is read-visible but calls `list_marks` and `list_annotations`, both minimum role `write`. This is a concrete role/discovery defect. Falsifying this finding requires either source showing those tools are read-authorized at cutover or a rewritten prompt that no longer depends on them.

## Missing evidence and falsification conditions

Missing from the attachment: the operator's live `ops.db`, call-log outbox state, Sinnix repository/configuration, installed client profiles, Polylogue skill files, actual SessionStart scripts, MCP client discovery caches, live old-surface output capture, transition shadow telemetry, and cold-model trial results.

The main recommendations should change only with concrete evidence:

- A current exact inventory other than 104 changes the row set.
- A live 90-day census with complete delivery can promote/demote noncritical rows.
- Proof that a mapping drops a filter, ordering rule, authority gate, continuation, or state transition blocks that row's retirement.
- Proof that a separate `graph` tool materially improves blind route selection without adding confusion could retain seven reads, but must beat the six-tool cold-model trials.
- Evidence that a current prompt/resource is unused and has a fully declared successor can justify separate retirement; tool call logs alone cannot establish prompt/resource non-use.
