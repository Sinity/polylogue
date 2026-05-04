"""SQL constants, protocol classes, and dataclasses for batch ingest."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from polylogue.pipeline.services.ingest_worker import ConversationData
from polylogue.storage.raw.models import RawConversationStateUpdate

if TYPE_CHECKING:
    import aiosqlite

_DEFAULT_INGEST_WORKER_LIMIT = 8
_INGEST_SOFT_BLOB_LIMIT_BYTES = 48 * 1024 * 1024
_INGEST_HIGH_BLOB_LIMIT_BYTES = 96 * 1024 * 1024
_INGEST_EXTREME_BLOB_LIMIT_BYTES = 256 * 1024 * 1024

_IDENTITY_LEDGER_UPSERT_SQL = """
INSERT OR IGNORE INTO identity_ledger (
    provider, source, source_path, provider_conversation_id, raw_hash, current_conversation_id
) VALUES (?, ?, ?, ?, ?, ?)
"""


class _RawStateRepositoryLike(Protocol):
    async def update_raw_state(self, raw_id: str, *, state: RawConversationStateUpdate) -> object: ...


class _ParsingServiceRawStateLike(Protocol):
    @property
    def repository(self) -> _RawStateRepositoryLike: ...


class _BulkConnectionBackendLike(Protocol):
    def bulk_connection(self) -> AbstractAsyncContextManager[object]: ...


class _ConnectionBackendLike(Protocol):
    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...


@dataclass(slots=True)
class _RawIngestOutcome:
    raw_id: str
    payload_provider: str | None
    validation_status: str
    validation_error: str | None
    parse_error: str | None
    error: str | None
    had_conversations: bool


@dataclass(slots=True)
class _IngestBatchSummary:
    outcomes: dict[str, _RawIngestOutcome] = field(default_factory=dict)
    failed_raw_ids: dict[str, str] = field(default_factory=dict)
    skipped_raw_ids: set[str] = field(default_factory=set)
    processed_ids: set[str] = field(default_factory=set)
    changed_conversation_ids: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }
    )
    changed_counts: dict[str, int] = field(
        default_factory=lambda: {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
        }
    )
    parse_failures: int = 0
    total_msgs: int = 0
    total_convos: int = 0
    raw_record_count: int = 0
    worker_count: int = 0
    total_blob_mb: float = 0.0
    total_result_bytes: int = 0
    max_result_bytes: int = 0
    max_result_raw_id: str | None = None
    elapsed_s: float = 0.0
    setup_elapsed_s: float = 0.0
    max_current_rss_mb: float | None = None
    result_wait_s: float = 0.0
    drain_elapsed_s: float = 0.0
    write_elapsed_s: float = 0.0
    max_write_elapsed_s: float = 0.0
    flush_elapsed_s: float = 0.0
    commit_elapsed_s: float = 0.0
    teardown_elapsed_s: float = 0.0


@dataclass(frozen=True, slots=True)
class _IngestWorkerRequest:
    archive_root_str: str
    blob_root_str: str
    validation_mode: str
    measure_ingest_result_size: bool


_ConversationEntry = tuple[str, ConversationData]


# ---------------------------------------------------------------------------
# SQL statements (copied from async query modules — sync versions)
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
