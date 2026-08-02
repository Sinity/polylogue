# Agent Affordance Usage

Generated: 2026-08-02T13:51:38.483581+00:00
Archive root: `/path/to/demo-archive`
Index schema: v54
Action scope: `grouped-tool-name-recent-window`

## Top Families


## Recent Window (7 days)


## Surface Inventory Classification

- cli_command keep: 64 surface(s), observed_actions=0.
- cli_command kill: 54 surface(s), observed_actions=0.
- mcp_tool keep: 4 surface(s), observed_actions=0.
- mcp_tool kill: 6 surface(s), observed_actions=0.

## Kill Candidates

These are zero-use non-operator surfaces in the captured archive evidence. They are review candidates, not automatic removals.

- mcp_tool `context` — zero captured agent use in this archive window; review before removal
- mcp_tool `explain` — zero captured agent use in this archive window; review before removal
- mcp_tool `get` — zero captured agent use in this archive window; review before removal
- mcp_tool `query` — zero captured agent use in this archive window; review before removal
- mcp_tool `read` — zero captured agent use in this archive window; review before removal
- mcp_tool `status` — zero captured agent use in this archive window; review before removal
- cli_command `agent` — zero captured agent use in this archive window; review before removal
- cli_command `agent doctor` — zero captured agent use in this archive window; review before removal
- cli_command `agent install` — zero captured agent use in this archive window; review before removal
- cli_command `agent manifest` — zero captured agent use in this archive window; review before removal
- cli_command `agent manual` — zero captured agent use in this archive window; review before removal
- cli_command `agent status` — zero captured agent use in this archive window; review before removal

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
