[← Back to README](../README.md)

# MCP Integration

Polylogue provides an MCP (Model Context Protocol) server for integration with Claude Desktop, Claude Code, and other MCP clients.

**Primary use case**: Claude Code's `/history` command can use polylogue to search past sessions semantically, rather than just grepping JSONL files.

## Starting the Server

```bash
polylogue mcp
```

Runs in stdio mode (standard for MCP). Logs to stderr.

## Claude Code Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "polylogue": {
      "command": "polylogue",
      "args": ["mcp"]
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
      "command": "polylogue",
      "args": ["mcp"]
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search` | Search conversations by text query with optional provider/date filters |
| `list_conversations` | List recent conversations, optionally filtered |
| `get_conversation` | Get a single conversation by ID (supports prefix matching) |
| `stats` | Archive statistics: totals, provider breakdown, database size |

## Available Resources

| Resource | Description |
|----------|-------------|
| `polylogue://stats` | Archive statistics |
| `polylogue://conversations` | List all conversations (up to 1000) |
| `polylogue://conversation/{id}` | Single conversation content |

## Available Prompts

| Prompt | Description |
|--------|-------------|
| `analyze_errors` | Analyze error patterns and solutions across conversations |
| `summarize_week` | Summarize key insights from the past week |
| `extract_code` | Extract and organize code snippets by language |

---

**See also:** [CLI Reference](cli-reference.md) · [Configuration](configuration.md) · [Library API](library-api.md)
