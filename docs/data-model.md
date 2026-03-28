[← Back to README](../README.md)

# Data Model

## Conversation

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique ID (`provider:provider_id`) |
| `provider` | str | Source provider (`chatgpt`, `claude`, `claude-code`, `gemini`) |
| `original_title` | str? | Provider's original title |
| `created_at` | datetime? | Creation timestamp |
| `updated_at` | datetime? | Last update timestamp |
| `content_hash` | str | SHA-256 for deduplication |
| `metadata` | dict | User metadata (k:v, see below) |

**Metadata** (unified k:v storage):

| Key | Type | Description |
|-----|------|-------------|
| `title` | str | User-set title (overrides original) |
| `summary` | str | User or LLM-generated summary |
| `tags` | list[str] | Tags (`important`, `project:foo`) |
| (custom) | str | Any user-defined key |

Display title precedence: `metadata.title` > `original_title` > truncated ID.

## Message

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Message ID |
| `role` | str | `user`, `assistant`, `system`, `tool` |
| `text` | str? | Message content |
| `timestamp` | datetime? | Message timestamp |
| `parent_id` | str? | Parent message (for branching) |
| `provider_meta` | dict? | Provider-specific data (content_blocks, cost, duration, etc.) |
| `attachments` | list | File attachments |

## Attachments

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Attachment ID |
| `name` | str? | Filename |
| `mime_type` | str? | MIME type |
| `size_bytes` | int? | File size |
| `path` | str? | Local path (if downloaded) |

Attachments are stored as references. For Drive sources, attachments can be downloaded on demand.

## Branching

Conversations may have branching structure (e.g., ChatGPT "edit and regenerate"). The `parent_id` field links messages to their parent, forming a tree.

**Current behavior**: Messages are flattened to a list in creation order. Branch structure is preserved in `provider_meta` for future use.

## Provider-Specific Metadata

Some providers include additional metadata:

**Claude Code**:

- `cost_usd`: API cost in USD
- `duration_ms`: Response generation time
- `model`: Model used (e.g., `claude-3-opus`)

Access via `message.provider_meta` or convenience properties:

```python
msg.cost_usd      # float or None
msg.duration_ms   # int or None
conv.total_cost_usd    # Sum of all message costs
conv.total_duration_ms # Sum of all durations
```

## Semantic Classification

Messages have classification properties derived from content and metadata:

| Property | Meaning |
|----------|---------|
| `is_user` | From user |
| `is_assistant` | From assistant |
| `is_system` | System prompt |
| `is_tool_use` | Tool call or result |
| `is_thinking` | Reasoning/thinking trace |
| `is_context_dump` | Pasted file content, context |
| `is_noise` | Tool use, context dump, or system |
| `is_substantive` | Real dialogue (not noise, not thinking) |

**Provider-specific detection**:

- **ChatGPT**: Thinking detected via `content_type: "thoughts"` or `"reasoning_recap"` in metadata
- **Claude Code**: Thinking via `content_blocks` with `type: "thinking"`; tool use via `type: "tool_use"` or `"tool_result"`
- **Gemini**: Thinking via `isThought` marker in chunk metadata
- **Claude (web)**: No structured thinking (simple text messages)

## Tags

Tags support `key:value` notation for namespacing:

```
important              # Simple tag
project:polylogue      # Namespaced
status:wip             # Namespaced
```

---

**See also:** [Library API](library-api.md) · [CLI Reference](cli-reference.md) · [Configuration](configuration.md)
