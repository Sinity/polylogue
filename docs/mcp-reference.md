# MCP Reference

Polylogue exposes an MCP (Model Context Protocol) server that coding agents can use to query
AI chat archives during sessions.

## Tools

- `list_conversations` — List recent conversations, filterable by path and sort order.
- `get_conversation` — Retrieve a full conversation by ID with optional prose-only projection.
- `search` — Full-text search across conversations with filter chains.

## Resources

- `polylogue://conversations/recent` — URI-addressable recent conversation list.
- `polylogue://conversations/{id}` — URI-addressable individual conversation.

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
