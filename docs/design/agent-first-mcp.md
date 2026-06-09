# Agent-First MCP Surface

Coding agents are polylogue's primary users — more so than humans. The MCP
surface should be designed for agent consumption first, with the CLI and web
UI as secondary surfaces over the same tools.

## Principles

1. **Predictable schemas** — every response has a stable typed shape. Agents
   should be able to destructure responses without probing for optional fields.
2. **Structured errors** — errors carry machine-readable codes and retry hints,
   not just human-readable messages.
3. **Tool chaining** — tool A's output can be passed directly to tool B's input
   without the agent parsing and reconstructing.
4. **Pagination as a protocol** — every list response carries `total`, `limit`,
   `offset` so agents can navigate without counting.
5. **Projection over fetching** — agents can request only the fields they need.
6. **Graceful degradation** — when the archive is empty or a feature is
   unavailable, tools return structured empty states, not errors.

## Current Gaps

From the audit (U8 agent), 8 tool gaps:

| Gap | Current State | Fix |
|-----|--------------|-----|
| Help tool | No `get_help` or tool descriptions via MCP | Add `polylogue_help` tool returning tool schemas |
| Auth | `--role` flag, self-declared, no enforcement | Per-role tool visibility; write tools require token |
| Pagination | `list_sessions` returns bare arrays | Add `total`/`limit`/`offset` to all list responses |
| Chaining | Tool outputs use different ID formats | Standardize on `session_id` across all tools |
| Count | No way to count without fetching all | Add `count_sessions` / `count_messages` tools |
| Projection | All fields returned, no field selection | Add `fields` parameter to list/search tools |
| Graceful shutdown | MCP server has no shutdown handshake | Add `polylogue_shutdown` tool with flush guarantee |
| Export fidelity | #811 — content_blocks dropped from exports | Fix export to include all message content |

## Tool Catalog

### Read tools (always available)

| Tool | Purpose | Chaining |
|------|---------|----------|
| `polylogue_search` | Full-text + filter search | Returns `session_ids[]` → feed to `polylogue_get_session` |
| `polylogue_list_sessions` | Paginated list with filters | Returns `session_ids[]` |
| `polylogue_get_session` | Single session with messages | Accepts `session_id` from search/list |
| `polylogue_get_messages` | Messages for a session | Accepts `session_id`, optional `role`/`limit` filter |
| `polylogue_get_session_profile` | Profile for a session | Accepts `session_id`, returns structured profile |
| `polylogue_get_project_memory` | Project memory entries | Accepts `project_path`, optional `kind` filter |
| `polylogue_search_project_memory` | FTS5 over project memory | Returns `entry_ids[]` |
| `polylogue_get_stats` | Archive statistics | No input needed |
| `polylogue_get_cost` | Cost breakdown | Accepts `session_id` or `--since` filter |
| `polylogue_count` | Count sessions/messages matching filters | Same filter params as search |

### Write tools (require write role + auth)

| Tool | Purpose |
|------|---------|
| `polylogue_tag_session` | Add/remove tags |
| `polylogue_record_project_memory` | Create a project memory entry |
| `polylogue_update_metadata` | Update session metadata |
| `polylogue_reset` | Re-ingest a source |

### Admin tools (require admin role + auth)

| Tool | Purpose |
|------|---------|
| `polylogue_shutdown` | Graceful daemon shutdown |
| `polylogue_health` | Full health check with component status |
| `polylogue_rebuild_embeddings` | Trigger re-embedding |

## Response Envelope

Every response wraps data in a consistent envelope:

```json
{
  "ok": true,
  "data": { ... },
  "meta": {
    "total": 5529,
    "limit": 20,
    "offset": 0,
    "ms": 42
  }
}
```

Error responses:

```json
{
  "ok": false,
  "error": {
    "code": "ARCHIVE_EMPTY",
    "message": "No sessions match the query.",
    "retryable": false,
    "hint": "Try broadening your filters or check that ingestion is running."
  }
}
```

Error codes are stable and documented. Agents can branch on `error.code` without
parsing `error.message`.

## Agent Workflow Patterns

### Pattern 1: Session continuity (SessionStart hook)

Agent receives `additionalContext` with recent sessions + project memory. No
tool calls needed. The hook runs before the agent's first message.

### Pattern 2: Context injection (mid-session)

Agent searches for relevant past sessions, reads the top 3, extracts relevant
patterns. 3 MCP calls, each <500ms:
1. `polylogue_search("error: foreign key constraint", project_path=...)`
2. `polylogue_get_session(top_result_id, prose_only=true)`
3. `polylogue_get_session_profile(top_result_id)`

### Pattern 3: Self-audit (post-session)

Agent reviews its own session for patterns:
1. `polylogue_get_session_profile(current_session_id)`
2. Compare tool efficiency against project baseline from `polylogue_get_stats`
3. Record any discoveries via `polylogue_record_project_memory`

### Pattern 4: Decision support (during implementation)

Agent checks project memory before making architectural choices:
1. `polylogue_search_project_memory("schema rebuild")`
2. If results found, read the relevant Decisions
3. Apply or consciously deviate

## Hard Limit

Agents must not spend more than 3 calls / 2 seconds per decision step. The MCP
surface is a support system, not a research library. If the answer isn't in the
first 3 calls, the agent should proceed with available information and record
uncertainty for later review.
