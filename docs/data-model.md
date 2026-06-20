[← Back to README](../README.md)

# Data Model

This page describes the **public Python domain models** returned by the library
API, CLI, and MCP surfaces. These are read models hydrated from the archive;
for the on-disk storage shape see [Schema](schema.md), and for the conceptual
rings see [Architecture](architecture.md). The authoritative field definitions
live in `polylogue/archive/session/domain_models.py`,
`polylogue/archive/message/models.py`, and
`polylogue/archive/attachment/models.py` — read those if a field here is
ambiguous.

## Origin, not provider

Public read surfaces are keyed by **`origin`** (the `Origin` enum in
`polylogue/core/enums.py`): `claude-code-session`, `claude-ai-export`,
`chatgpt-export`, `codex-session`, `gemini-cli-session`, `aistudio-drive`,
`hermes-session`, `antigravity-session`, `unknown-export`. The provider-wire
`Provider` enum (`chatgpt`, `claude-code`, …) is retained only at the
parsing/schema boundary and is not the public filter token. Filter and query
surfaces use `origin`.

## Session

`Session` (and its message-less sibling `SessionSummary`) is the primary read
entity.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `SessionId` (str) | Composite ID, `origin:native_id` |
| `origin` | `Origin` | Source-origin token (see above) |
| `title` | `str?` | Parsed session title |
| `created_at` | `datetime?` | Creation timestamp |
| `updated_at` | `datetime?` | Last update timestamp |
| `messages` | `MessageCollection` | Eagerly or lazily materialized messages (`Session` only) |
| `metadata` | `dict[str, object]` | User-metadata overlay (see below) |
| `tags_m2m` | `tuple[str, ...]` | Tags hydrated from the `session_tags` table |
| `working_directories` | `tuple[str, ...]` | Working dirs observed for the session |
| `git_branch` | `str?` | Git branch, when known |
| `git_repository_url` | `str?` | Git remote URL, when known |
| `parent_id` | `SessionId?` | Parent session (continuation/fork/sidechain/subagent) |
| `branch_type` | `BranchType?` | `continuation`, `sidechain`, `fork`, `subagent` |
| `session_events` | `tuple[SessionEvent, ...]` | Structured session events (e.g. compaction) — `Session` only |
| `attachments` | `list[Attachment]` | Session-level attachments not bound to a message |

`SessionSummary` carries the same identity/metadata fields plus precomputed
`message_count` and `dialogue_count`, but omits `messages`, `session_events`,
and `attachments`.

### Metadata overlay and derived properties

`metadata` is the user-owned key/value overlay projected from `user.db`
metadata assertions. It holds the user title override, a user/LLM summary, and
any custom keys. It is **excluded from the content hash**, so editing it never
triggers re-import. Tags are not stored in this overlay: user tags are tag
assertions, while auto-tags are rebuildable `index.db.session_tags` rows and
surface through `tags_m2m`.

Convenience properties resolve these:

- `display_title` → `metadata["title"]` (user override) if set, else `title`, else `id[:8]`.
- `tags` → `tags_m2m` when hydrated, else any tags in `metadata`.
- `summary` → `metadata["summary"]`.
- `is_continuation` → `branch_type == continuation`.

## Message

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Message ID, `session_id:native_id` (or `session_id:position.variant`) |
| `role` | `Role` | `user`, `assistant`, `system`, `tool`, `unknown` |
| `text` | `str?` | Flattened message text |
| `timestamp` | `datetime?` | Message timestamp |
| `message_type` | `MessageType` | `message`, `summary`, `tool_use`, `tool_result`, `thinking`, `context`, `protocol` |
| `provider` | `Provider?` | Provider-wire identity (parse-boundary only; prefer the session `origin`) |
| `content_blocks` | `list[dict]` | Structured content blocks (text/thinking/tool_use/tool_result/image/code/document) |
| `attachments` | `list[Attachment]` | File attachments referenced by the message |
| `parent_id` | `str?` | Parent message, for branching |
| `branch_index` | `int` | Branch position among sibling variants |
| `has_tool_use` / `has_thinking` / `has_paste` | `bool` | Precomputed content flags projected from storage |
| `input_tokens` / `output_tokens` | `int` | Token counts, when reported |
| `cache_read_tokens` / `cache_write_tokens` | `int` | Cache token counts, when reported |
| `duration_ms` | `int` | Reported generation duration |
| `model_name` | `str?` | Model that produced the message |

Cost and per-model token rollups are **not** message-level properties. They are
materialized at the session level in `session_model_usage`,
`session_reported_costs`, and `session_profiles`, and surfaced through the
session insight reads (e.g. `session_costs`, `cost_rollups`). See
[Architecture § Derived Read Models](architecture.md#2-derived-read-models).

### Semantic classification

`Message` exposes derived boolean properties (in
`polylogue/archive/message/model_runtime.py`):

| Property | Meaning |
|----------|---------|
| `is_user` / `is_assistant` / `is_system` | Role-based |
| `is_dialogue` | User or assistant turn |
| `is_tool_use` | Tool call or result |
| `is_thinking` | Reasoning/thinking content |
| `is_context_dump` | Pasted file content / context dump |
| `is_protocol_artifact` | Provider protocol noise |
| `is_noise` | Tool use, context dump, protocol, or system |
| `is_substantive` | Real dialogue (not noise, not thinking) |
| `is_branch` | Has a parent message (non-linear) |

## Attachment

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Attachment ID |
| `name` | `str?` | Filename / display name |
| `mime_type` | `str?` | MIME type |
| `size_bytes` | `int?` | Byte size |
| `path` | `str?` | Local path, if downloaded |
| `source_url` | `str?` | Upload/source URL, if known |
| `caption` | `str?` | Caption, if provided |

Attachments are content-addressed in the blob store and joined to messages
through `attachment_refs`. For Drive/Gemini sources, attachment lookup keys on
explicit identity fields preserved at ingest (the provider attachment id, plus
`fileId`/`driveId`); those identifiers are indexed for exact lookup and search
reports the match as attachment evidence.

## Branching and topology

Sessions form trees and cross-session lineages: continuations, forks,
sidechains, and subagent sessions. Within a session, `Message.parent_id` links
messages into a branch tree, while the materialized `messages.text` is the
flattened active-path content. Across sessions, parent/child references are
persisted as typed rows in `session_links` (even when the parent has not been
ingested yet), and the resolved logical root is materialized as
`session_profiles.logical_session_id`. See
[Internals § Topology Edges](internals.md#topology-edges-1258) and
[Internals § Logical Session Identity](internals.md#logical-session-identity-866).

## Tags

Tags support `key:value` notation for namespacing:

```
important              # Simple tag
repo:polylogue         # Namespaced
status:wip             # Namespaced
```

User tags are assertion rows in `user.db`; auto-tags remain heuristic
`session_tags` rows in `index.db`. The user side is irreplaceable durable
overlay state; the auto side is rebuildable read-model state.

---

**See also:** [Schema](schema.md) · [Library API](library-api.md) · [CLI Reference](cli-reference.md) · [Configuration](configuration.md)
