# Agent Affordance Usage

Generated: 2026-07-05T08:42:39.982266+00:00
Archive root: `/home/sinity/.local/share/polylogue`
Index schema: v24
Action scope: `grouped-tool-name-recent-window`

## Top Families

- serena: 20 actions across 3 raw tool name(s); 5 distinct session(s); errors=1, nonzero_exits=0.
- polylogue: 19 actions across 8 raw tool name(s); 6 distinct session(s); errors=3, nonzero_exits=0.
- lynchpin: 3 actions across 2 raw tool name(s); 1 distinct session(s); errors=0, nonzero_exits=0.
- codebase-memory: 2 actions across 2 raw tool name(s); 1 distinct session(s); errors=0, nonzero_exits=0.

## Recent Window (7 days)

- serena: serena/find_symbol — 12 actions across 4 session(s), failure_rate=0.083.
- serena: serena/get_symbols_overview — 7 actions across 3 session(s), failure_rate=0.0.
- polylogue: polylogue/query_units — 5 actions across 1 session(s), failure_rate=0.4.
- polylogue: polylogue/get_session_summary — 4 actions across 3 session(s), failure_rate=0.0.
- lynchpin: lynchpin/lynchpin_project — 2 actions across 1 session(s), failure_rate=0.0.
- polylogue: polylogue/get_pathologies — 2 actions across 2 session(s), failure_rate=0.0.
- polylogue: polylogue/get_resume_brief — 2 actions across 2 session(s), failure_rate=0.0.
- polylogue: polylogue/list_sessions — 2 actions across 1 session(s), failure_rate=0.0.

## Surface Inventory Classification

- cli_command keep: 45 surface(s), observed_actions=20.
- cli_command kill: 34 surface(s), observed_actions=0.
- cli_command promote: 15 surface(s), observed_actions=959.
- mcp_tool keep: 32 surface(s), observed_actions=108.
- mcp_tool kill: 59 surface(s), observed_actions=0.
- mcp_tool promote: 5 surface(s), observed_actions=35.

## Kill Candidates

These are zero-use non-operator surfaces in the captured archive evidence. They are review candidates, not automatic removals.

- mcp_tool `action_affordances` — zero captured agent use in this archive window; review before removal
- mcp_tool `add_mark` — zero captured agent use in this archive window; review before removal
- mcp_tool `add_tag` — zero captured agent use in this archive window; review before removal
- mcp_tool `agent_coordination` — zero captured agent use in this archive window; review before removal
- mcp_tool `aggregate_sessions` — zero captured agent use in this archive window; review before removal
- mcp_tool `archive_get_session` — zero captured agent use in this archive window; review before removal
- mcp_tool `archive_list_sessions` — zero captured agent use in this archive window; review before removal
- mcp_tool `archive_search_sessions` — zero captured agent use in this archive window; review before removal
- mcp_tool `blackboard_list` — zero captured agent use in this archive window; review before removal
- mcp_tool `blackboard_post` — zero captured agent use in this archive window; review before removal
- mcp_tool `build_context_image` — zero captured agent use in this archive window; review before removal
- mcp_tool `bulk_tag_sessions` — zero captured agent use in this archive window; review before removal

## Interpretation

- Family-normalized counts avoid treating plugin-prefixed tool names as separate affordances.
- The default action scope is the recent-session window; use --all-time for the intentionally broader scan.
- Recent windows are required for newly-added affordances such as Serena and codebase-memory.
- Failure rates are structured tool-result signals; they identify friction, not necessarily low utility.
- The surface inventory left-joins observed usage against every registered MCP tool and CLI command.
- Operator-only rows are kept even when unused; the classification caveat is part of the data.

## Notes

- Default family counts used an indexed grouped tool-name path over blocks.
- Command and input text bodies are not scanned unless --detail-pattern is supplied.
- Samples are omitted on this fast grouped path to avoid materializing every matching action row.

## Files

- `family-counts.csv`
- `evidence-kind-counts.csv`
- `tool-counts.csv`
- `tool-by-origin.csv`
- `recent-7d-tool-counts.csv`
- `tool-samples.csv`
- `surface-inventory.csv`
- `surface-classification-summary.csv`
- `affordance-usage.report.json`
- `summary.json`
