# MCP Reference

Polylogue exposes an MCP (Model Context Protocol) server that coding agents can use to query
AI chat archives during sessions.

## Tools

- `list_sessions` — List recent sessions, filterable by path and sort order.
- `get_session` — Retrieve a full session by ID with optional prose-only projection.
- `search` — Full-text search across sessions with filter chains.

## Resources

- `polylogue://sessions/recent` — URI-addressable recent session list.
- `polylogue://sessions/{id}` — URI-addressable individual session.

## Configuration

Add to your Claude Code `.mcp.json`:

```json
{
  "mcpServers": {
    "polylogue": {
      "command": "uv",
      "args": ["run", "polylogue", "mcp"]
    }
  }
}
```
