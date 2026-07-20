[← Back to README](../README.md)

# MCP Integration

Polylogue provides an MCP (Model Context Protocol) server for integration with Claude Desktop, Claude Code, and other MCP clients.

**Primary use case**: Claude Code's `/history` command can use Polylogue to
search past sessions semantically and retrieve archived sessions directly.

## Starting the Server

```bash
polylogue-mcp
```

Runs in stdio mode (standard for MCP), read-only by default. Logs to stderr. Write, judge,
and maintenance capability are independent config opt-ins (`polylogue.toml` `[mcp]` keys or
`POLYLOGUE_MCP_WRITE_ENABLED`/`POLYLOGUE_MCP_JUDGE_ENABLED`/
`POLYLOGUE_MCP_MAINTENANCE_ENABLED`) — see `docs/mcp-reference.md#configuration`. There is no
`--role` flag.

## Claude Code Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "polylogue": {
      "command": "polylogue-mcp"
    }
  }
}
```

## Claude Desktop Configuration

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "polylogue": {
      "command": "polylogue-mcp"
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search` | Search sessions by text query with optional provider/date filters; returns session summaries plus match rank, lane, surface, message id, and snippet evidence. Drive/Gemini provider attachment ids and stored `provider_id`, `id`, `fileId`, or `driveId` metadata return `match_surface=attachment` hits. |
| `list_sessions` | List recent sessions, optionally filtered |
| `get_session_summary` | Get a single session summary by ID (supports prefix matching) |
| `stats` | Archive statistics: totals, provider breakdown, database size |
| `get_postmortem_bundle` | Distilled postmortem bundle over a matched scope (#2380): top sessions by cost, repos touched, tool/work-kind rollups, failure signals. Read-only. |
| `get_pathologies` | Deterministic agent-workflow pathology distribution over a matched scope (#2383): consecutive failure-streak and stale-context findings with drillable evidence. Read-only. |
## Available Resources

| Resource | Description |
|----------|-------------|
| `polylogue://stats` | Archive statistics |
| `polylogue://sessions` | List all sessions (up to 1000) |
| `polylogue://session/{id}` | Single session content |

## Available Prompts

| Prompt | Description |
|--------|-------------|
| `analyze_errors` | Analyze error patterns and solutions across sessions |
| `summarize_week` | Summarize key insights from the past week |
| `extract_code` | Extract and organize code snippets by language |

---

**See also:** [CLI Reference](cli-reference.md) · [Configuration](configuration.md) · [Library API](library-api.md)
