[← Back to Docs](README.md)

# Database Schema

Polylogue stores all data in a single SQLite database with WAL mode enabled. The
schema is versioned and uses fresh-first initialization: version mismatches are
rejected unless an explicit upgrade migration exists for that exact transition.

**Current schema version: 6.**

## Core Tables

### conversations

The primary archive entity. One row per imported conversation.

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | TEXT PK | Composite ID (`provider:provider_id`) |
| `provider_name` | TEXT | Provider enum value |
| `provider_conversation_id` | TEXT | Provider's native ID |
| `title` | TEXT | Conversation title |
| `created_at` | TEXT | Creation timestamp (ISO 8601) |
| `updated_at` | TEXT | Last update timestamp |
| `sort_key` | REAL | Sortable timeline key |
| `content_hash` | TEXT | SHA-256 over normalized content (dedup key) |
| `provider_meta` | TEXT | Provider-specific metadata (JSON) |
| `metadata` | TEXT | User metadata: tags, summaries, titles (JSON) |
| `version` | INTEGER | Schema version at write time |
| `parent_conversation_id` | TEXT FK | Parent conversation (branches/continuations) |
| `branch_type` | TEXT | `continuation`, `sidechain`, `fork`, `subagent` |
| `raw_id` | TEXT FK | Source raw record |

### messages

Individual messages within a conversation.

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | TEXT PK | Unique message ID |
| `conversation_id` | TEXT FK | Parent conversation |
| `role` | TEXT | `user`, `assistant`, `system`, `tool` |
| `text` | TEXT | Message text content |
| `sort_key` | REAL | Message order within conversation |
| `content_hash` | TEXT | SHA-256 for dedup |
| `parent_message_id` | TEXT FK | Parent message (branching) |
| `branch_index` | INTEGER | Branch position |
| `provider_name` | TEXT | Denormalized for index-only queries |
| `word_count` | INTEGER | Precomputed word count |
| `has_tool_use` | INTEGER | Precomputed: message contains tool calls |
| `has_thinking` | INTEGER | Precomputed: message contains thinking blocks |
| `has_paste` | INTEGER | Precomputed: message contains pasted content |
| `message_type` | TEXT | `message` or `action` |

Covering index `idx_messages_provider_stats` enables index-only GROUP BY
queries for provider analytics.

### content_blocks

First-class structured content within messages. One row per block.

| Column | Type | Description |
|--------|------|-------------|
| `block_id` | TEXT PK | Unique block ID |
| `message_id` | TEXT FK | Parent message |
| `conversation_id` | TEXT FK | Denormalized parent conversation |
| `block_index` | INTEGER | Order within message |
| `type` | TEXT | `text`, `thinking`, `tool_use`, `tool_result`, `image`, `code`, `document` |
| `text` | TEXT | Block text content |
| `tool_name` | TEXT | Tool name (for `tool_use` blocks) |
| `tool_id` | TEXT | Tool call ID |
| `tool_input` | TEXT | Tool input JSON |
| `media_type` | TEXT | MIME type for images/documents |
| `semantic_type` | TEXT | Inferred semantic type: `file_read`, `file_write`, `file_edit`, `shell`, `git`, `search`, `web`, `agent`, `subagent`, `thinking`, `other` |

### action_events

Normalized action records derived from content blocks.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | TEXT PK | Unique event ID |
| `conversation_id` | TEXT FK | Parent conversation |
| `message_id` | TEXT FK | Parent message |
| `kind` | TEXT | Action kind (same vocabulary as `semantic_type`) |
| `tool_name` | TEXT | Normalized tool name |
| `summary` | TEXT | Action description |
| `start_time` | TEXT | Action start timestamp |
| `end_time` | TEXT | Action end timestamp |
| `file_paths_json` | TEXT | Referenced file paths (JSON array) |

### conversation_stats

Precomputed per-conversation aggregates, updated atomically with message writes.

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | TEXT PK FK | Parent conversation |
| `provider_name` | TEXT | Denormalized |
| `message_count` | INTEGER | Total messages |
| `word_count` | INTEGER | Total words |
| `tool_use_count` | INTEGER | Messages with tool use |
| `thinking_count` | INTEGER | Messages with thinking |
| `paste_count` | INTEGER | Messages with pasted content |

Used for pushdown filters (`--min-messages`, `--min-words`, `--has-tool-use`,
`--has-thinking`).

### Other Core Tables

| Table | Purpose |
|-------|---------|
| `attachments` / `attachment_refs` | File attachments with M:N message refs |
| `tags` / `conversation_tags` | M:N tag assignments (replaced JSON `metadata.tags`) |
| `raw_conversations` | Raw import records before parsing |
| `runs` | Pipeline run history with state snapshots |

## FTS5 Tables

| Virtual Table | Content | Tokenizer |
|---------------|---------|-----------|
| `messages_fts` | Message text | `unicode61` |
| `action_events_fts` | Action event text | `unicode61` |
| `session_profiles_fts` | Session profile search text | `unicode61` |
| `session_profile_evidence_fts` | Profile evidence text | `unicode61` |
| `session_profile_inference_fts` | Profile inference text | `unicode61` |
| `session_profile_enrichment_fts` | Profile enrichment text | `unicode61` |
| `session_work_events_fts` | Work event search text | `unicode61` |
| `work_threads_fts` | Thread search text | `unicode61` |

All FTS tables use `unicode61` tokenizer (no porter stemmer). Tables are
maintained via AFTER INSERT/UPDATE/DELETE triggers on their base tables.

## Vector Table

`message_embeddings` uses the `vec0` virtual table extension:

```
message_id TEXT PRIMARY KEY
embedding float[1024]
+provider_name TEXT
+conversation_id TEXT
```

1024-dimensional float embeddings. Populated when `VOYAGE_API_KEY` is set and
the embed pipeline stage runs. Used by the `--similar` filter and `hybrid`
retrieval lane.

## Insight Tables

| Table | Description |
|-------|-------------|
| `session_profiles` | Per-session aggregates: repos, tools, costs, durations, message counts |
| `session_work_events` | File-level work events within sessions |
| `session_phases` | Session segments: planning, implementation, verification, exploration |
| `work_threads` | Multi-session work groupings |
| `session_tag_rollups` | Pre-aggregated tag usage stats |

Each insight table includes a `materializer_version` for cache invalidation and
a `search_text` column for FTS indexing.

## Auxiliary Tables

| Table | Purpose |
|-------|---------|
| `artifact_observations` | Schema inference pipeline: per-artifact analysis records |
| `publications` | Site/exports generation history |
| `pending_blob_refs` | Blob store leases (prevents GC races) |
| `gc_generations` | Garbage collection generation tracking |
| `source_file_cursors` | Per-file ingestion progress (idempotent resume) |
| `identity_ledger` | Conversation content-hash identity ledger |

## Schema Versioning

Polylogue uses fresh-first schema initialization:

- **Version match**: open database normally
- **New database**: create all tables at current `SCHEMA_VERSION`
- **Version mismatch**: rejected with an error unless an explicit migration
  exists for that exact version transition

Schema version is declared in `polylogue/storage/sqlite/schema_ddl.py` as
`SCHEMA_VERSION`. The bootstrap branching logic in `schema_bootstrap.py` handles
both sync and async backends.

## Schema Drift Detection

Run `polylogue check` to verify schema integrity:

```bash
polylogue check                     # Schema health + FK integrity
polylogue check --schemas           # Schema conformance check
polylogue check --schemas --schema-samples all   # Full schema audit
```

The `devtools verify-schema-roundtrip` command verifies provider schema
packages reload and roundtrip cleanly through typed models.

## Blob Store

Content-addressed blob storage for attachment payloads. Blobs are keyed by
SHA-256 hash and stored under `~/.local/share/polylogue/blobs/`. The
`pending_blob_refs` table provides lease-based garbage collection that prevents
races with concurrent ingestion.
