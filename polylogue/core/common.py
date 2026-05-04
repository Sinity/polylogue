"""Canonical shared utilities and SQL templates.

Single-source of truth for:
- SQL upsert statements used by both sync (ingest_batch.py) and async
  (storage/sqlite/queries/*.py) write paths.
- Generic utility functions duplicated across modules (chunked, _json_object).

Sync path (ingest_batch.py) is the canonical source for SQL — battle-tested
through tens of thousands of bulk ingest operations. Async path MUST match.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from polylogue.core.json import json_document

# ---------------------------------------------------------------------------
# SQL templates — canonical source, used by sync and async write paths
# ---------------------------------------------------------------------------

_CONVERSATION_UPSERT_SQL = """
INSERT INTO conversations (
    conversation_id, provider_name, provider_conversation_id, title,
    created_at, updated_at, sort_key, content_hash,
    provider_meta, metadata, version,
    parent_conversation_id, branch_type, raw_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
    title = excluded.title,
    created_at = excluded.created_at,
    updated_at = excluded.updated_at,
    sort_key = excluded.sort_key,
    content_hash = excluded.content_hash,
    provider_meta = excluded.provider_meta,
    metadata = COALESCE(excluded.metadata, conversations.metadata),
    parent_conversation_id = excluded.parent_conversation_id,
    branch_type = excluded.branch_type,
    raw_id = COALESCE(excluded.raw_id, conversations.raw_id)
WHERE
    content_hash != excluded.content_hash
    OR IFNULL(title, '') != IFNULL(excluded.title, '')
    OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
    OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
    OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
    OR IFNULL(parent_conversation_id, '') != IFNULL(excluded.parent_conversation_id, '')
    OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
    OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
    OR IFNULL(sort_key, 0) != IFNULL(excluded.sort_key, 0)
"""

_MESSAGE_UPSERT_SQL = """
INSERT INTO messages (
    message_id, conversation_id, provider_message_id, role, text,
    sort_key, content_hash, version, parent_message_id, branch_index,
    provider_name, word_count, has_tool_use, has_thinking, has_paste, message_type
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(message_id) DO UPDATE SET
    role = excluded.role,
    text = excluded.text,
    sort_key = excluded.sort_key,
    content_hash = excluded.content_hash,
    parent_message_id = excluded.parent_message_id,
    branch_index = excluded.branch_index,
    provider_name = excluded.provider_name,
    word_count = excluded.word_count,
    has_tool_use = excluded.has_tool_use,
    has_thinking = excluded.has_thinking,
    has_paste = excluded.has_paste,
    message_type = excluded.message_type
WHERE
    content_hash != excluded.content_hash
    OR IFNULL(role, '') != IFNULL(excluded.role, '')
    OR IFNULL(text, '') != IFNULL(excluded.text, '')
    OR IFNULL(sort_key, 0) != IFNULL(excluded.sort_key, 0)
    OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
    OR branch_index != excluded.branch_index
    OR word_count != excluded.word_count
    OR has_tool_use != excluded.has_tool_use
    OR has_thinking != excluded.has_thinking
    OR has_paste != excluded.has_paste
    OR message_type != excluded.message_type
"""

_CONTENT_BLOCK_UPSERT_SQL = """
INSERT INTO content_blocks (
    block_id, message_id, conversation_id, block_index,
    type, text, tool_name, tool_id, tool_input,
    media_type, metadata, semantic_type
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(message_id, block_index) DO UPDATE SET
    type = excluded.type,
    text = excluded.text,
    tool_name = excluded.tool_name,
    tool_id = excluded.tool_id,
    tool_input = excluded.tool_input,
    media_type = excluded.media_type,
    metadata = excluded.metadata,
    semantic_type = excluded.semantic_type
"""

_STATS_UPSERT_SQL = """
INSERT INTO conversation_stats
    (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count, paste_count)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
    provider_name  = excluded.provider_name,
    message_count  = excluded.message_count,
    word_count     = excluded.word_count,
    tool_use_count = excluded.tool_use_count,
    thinking_count = excluded.thinking_count,
    paste_count    = excluded.paste_count
"""

_ACTION_EVENT_INSERT_SQL = """
INSERT INTO action_events (
    event_id, conversation_id, message_id, materializer_version,
    source_block_id, timestamp, sort_key, sequence_index,
    provider_name, action_kind, tool_name, normalized_tool_name, tool_id,
    affected_paths_json, cwd_path, branch_names_json,
    command, query_text, url, output_text, search_text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_ATTACHMENT_UPSERT_SQL = """
INSERT INTO attachments (
    attachment_id, mime_type, size_bytes, path, ref_count, provider_meta
) VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(attachment_id) DO UPDATE SET
    mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
    size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
    path = COALESCE(excluded.path, attachments.path),
    provider_meta = COALESCE(excluded.provider_meta, attachments.provider_meta)
"""

_ATTACHMENT_REF_INSERT_SQL = """
INSERT OR IGNORE INTO attachment_refs (
    ref_id, attachment_id, conversation_id, message_id, provider_meta
) VALUES (?, ?, ?, ?, ?)
"""


# Public exports — both sync and async write paths import these directly.
SQL_CONVERSATION_UPSERT = _CONVERSATION_UPSERT_SQL
SQL_MESSAGE_UPSERT = _MESSAGE_UPSERT_SQL
SQL_CONTENT_BLOCK_UPSERT = _CONTENT_BLOCK_UPSERT_SQL
SQL_STATS_UPSERT = _STATS_UPSERT_SQL
SQL_ACTION_EVENT_INSERT = _ACTION_EVENT_INSERT_SQL
SQL_ATTACHMENT_UPSERT = _ATTACHMENT_UPSERT_SQL
SQL_ATTACHMENT_REF_INSERT = _ATTACHMENT_REF_INSERT_SQL

# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    """Yield successive chunks from a sequence of items."""
    for index in range(0, len(items), size):
        yield items[index : index + size]


def json_object(value: object) -> dict[str, object]:
    """Convert a JSON-compatible value to a plain dict of str->object.

    Used by publication and run record mappers to convert manifest/plan
    documents into dict form without retaining orjson-specific types.
    """
    document = json_document(value)
    result: dict[str, object] = {}
    for key, item in document.items():
        result[key] = item
    return result


def format_malformed_jsonl_error(*, malformed_lines: int, malformed_detail: str | None) -> str:
    """Format a human-readable error for malformed JSONL input."""
    message = f"Malformed JSONL lines: {malformed_lines}"
    if malformed_detail:
        return f"{message} (first bad {malformed_detail})"
    return message


__all__ = [
    "SQL_ACTION_EVENT_INSERT",
    "SQL_ATTACHMENT_REF_INSERT",
    "SQL_ATTACHMENT_UPSERT",
    "SQL_CONTENT_BLOCK_UPSERT",
    "SQL_CONVERSATION_UPSERT",
    "SQL_MESSAGE_UPSERT",
    "SQL_STATS_UPSERT",
    "chunked",
    "format_malformed_jsonl_error",
    "json_object",
]
