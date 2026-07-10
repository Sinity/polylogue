# MCP Reference

Polylogue exposes an MCP (Model Context Protocol) server that coding agents can use to query
AI chat archives during sessions. This is the primary continuity surface — richer than the
CLI or Python API for agent-facing recall, corrections, and context assembly.

## Tools

~100 tools registered across `polylogue/mcp/server_*.py`, gated by capability role (see
Configuration below). A representative sample by category:

- **Search / list / get**: `search`, `list_sessions`, `get_messages`,
  `get_session_tree`, `get_session_summary`, `get_logical_session`, `facets`.
- **Insights**: `get_stats_by`, `session_costs`, `cost_rollups`, `usage_timeline`,
  `tool_usage`, `workflow_shape_distribution`, `session_latency_profile`.
- **Continuity / recall**: `compile_context`, `build_context_image`,
  `compose_context_preamble`, `get_resume_brief`, `find_resume_candidates`,
  `find_abandoned_sessions`, `find_stuck_sessions`.
- **Correlation / topology**: `correlate_session`, `correlate_sessions`,
  `get_session_topology`, `neighbor_candidates`, `find_similar_sessions`,
  `compare_sessions`.
- **Postmortem / pathology**: `get_postmortem_bundle`, `get_pathologies`,
  `insight_rigor_audit`.
- **Corrections / assertions** (write role): `add_tag`, `bulk_tag_sessions`,
  `blackboard_post`.
- **Maintenance** (admin role): `maintenance_preview`, `maintenance_execute`,
  `rebuild_index`, `update_index`, `rebuild_session_insights`.

The exhaustive, currently-registered tool name set is a test-enforced contract, not hand
duplicated here: `tests/infra/mcp.py:EXPECTED_TOOL_NAMES`. Adding a tool requires updating
that set plus its tool contract (see `CLAUDE.md` § MCP gotchas).

## Resources

- `polylogue://stats` — Archive-wide summary stats.
- `polylogue://sessions` — Recent session list.
- `polylogue://session/{conv_id}` — Individual session by ID.
- `polylogue://tags` — Known tag vocabulary.
- `polylogue://messages/{conv_id}` — Messages for a session.
- `polylogue://session-tree/{conv_id}` — Lineage-composed session tree.
- `polylogue://origin/{name}/recent` — Recent sessions for one origin.
- `polylogue://readiness` — Daemon/archive readiness snapshot.

## Configuration

The MCP server is a standalone console script (`polylogue.mcp.cli:main`), not a `polylogue`
subcommand. `--role` gates capability: `read` (default, omits mutation/maintenance tools),
`write` (adds corrections/tagging), or `admin` (adds maintenance/rebuild tools).

Add to your Claude Code `.mcp.json`:

```json
{
  "mcpServers": {
    "polylogue": {
      "command": "polylogue-mcp",
      "args": ["--role", "read"]
    }
  }
}
```

See `docs/mcp-integration.md` for the full client-integration walkthrough
(Claude Code, Codex, other MCP clients).
