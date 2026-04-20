"""Shared storage-record builders and DB seeding helpers for tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias
from uuid import uuid4

from polylogue.lib.branch_type import BranchType
from polylogue.lib.roles import Role
from polylogue.pipeline.prepare import _timestamp_sort_key
from polylogue.storage.backends.connection import connection_context, open_connection
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
    RunRecord,
    _json_or_none,
    _make_ref_id,
)
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
    ValidationMode,
    ValidationStatus,
)

if TYPE_CHECKING:
    from polylogue.lib.conversation_models import Conversation

JSONRecord: TypeAlias = dict[str, object]
MessageMapping: TypeAlias = Mapping[str, object]


class _AutoTimestampSentinel:
    """Marker for builders that should synthesize a fresh timestamp."""


class _AutoMessageIdSentinel:
    """Marker for builders that should target the most recent message."""


# Thread-safety lock for writes (matches store.py pattern)
_WRITE_LOCK = threading.Lock()
_AUTO_TIMESTAMP: Final = _AutoTimestampSentinel()
_AUTO_MESSAGE_ID: Final = _AutoMessageIdSentinel()


def _conversation_id(value: str) -> ConversationId:
    return ConversationId(value)


def _message_id(value: str) -> MessageId:
    return MessageId(value)


def _attachment_id(value: str) -> AttachmentId:
    return AttachmentId(value)


def _content_hash(value: str) -> ContentHash:
    return ContentHash(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_dict(value: object) -> JSONRecord | None:
    if not isinstance(value, Mapping):
        return None
    result: JSONRecord = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _coerce_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _coerce_int(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _coerce_sort_key(value: object, default: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_content_hash(value: object, default: str) -> ContentHash:
    return _content_hash(_coerce_str(value, default))


def _resolve_timestamp(value: str | None | _AutoTimestampSentinel) -> str | None:
    return datetime.now(timezone.utc).isoformat() if isinstance(value, _AutoTimestampSentinel) else value


def _resolve_attachment_message_id(
    *,
    value: str | None | _AutoMessageIdSentinel,
    messages: list[MessageRecord],
) -> str | None:
    if isinstance(value, _AutoMessageIdSentinel):
        return str(messages[-1].message_id) if messages else None
    return value


def _coerce_builder_timestamp(value: object) -> str | None | _AutoTimestampSentinel:
    if isinstance(value, str) or value is None:
        return value
    return _AUTO_TIMESTAMP


def _content_block_record(
    *,
    message_id: str,
    conversation_id: str,
    block_index: int,
    block_type: str,
    text: str | None = None,
    tool_name: str | None = None,
    tool_id: str | None = None,
    tool_input: str | None = None,
    media_type: str | None = None,
    metadata: str | None = None,
    semantic_type: str | None = None,
) -> ContentBlockRecord:
    return ContentBlockRecord(
        block_id=ContentBlockRecord.make_id(message_id, block_index),
        message_id=_message_id(message_id),
        conversation_id=_conversation_id(conversation_id),
        block_index=block_index,
        type=ContentBlockType.from_string(block_type),
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        media_type=media_type,
        metadata=metadata,
        semantic_type=None if semantic_type is None else SemanticBlockType.from_string(semantic_type),
    )


def _content_block_from_mapping(
    *,
    block: MessageMapping,
    message_id: str,
    conversation_id: str,
    block_index: int,
) -> ContentBlockRecord:
    raw_tool_input = block.get("tool_input", block.get("input"))
    raw_metadata = block.get("metadata")
    return _content_block_record(
        message_id=message_id,
        conversation_id=conversation_id,
        block_index=block_index,
        block_type=_optional_str(block.get("type")) or "text",
        text=_optional_str(block.get("text")),
        tool_name=_optional_str(block.get("tool_name")) or _optional_str(block.get("name")),
        tool_id=_optional_str(block.get("tool_id")) or _optional_str(block.get("id")),
        tool_input=(
            raw_tool_input
            if isinstance(raw_tool_input, str)
            else json.dumps(raw_tool_input)
            if raw_tool_input is not None
            else None
        ),
        media_type=_optional_str(block.get("media_type")),
        metadata=(
            raw_metadata
            if isinstance(raw_metadata, str)
            else json.dumps(raw_metadata)
            if raw_metadata is not None
            else None
        ),
        semantic_type=_optional_str(block.get("semantic_type")),
    )


def _normalize_content_blocks(
    *,
    raw_blocks: object,
    message_id: str,
    conversation_id: str,
) -> list[ContentBlockRecord]:
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[ContentBlockRecord] = []
    for idx, raw_block in enumerate(raw_blocks):
        if isinstance(raw_block, ContentBlockRecord):
            blocks.append(raw_block)
            continue
        if isinstance(raw_block, Mapping):
            blocks.append(
                _content_block_from_mapping(
                    block=raw_block,
                    message_id=message_id,
                    conversation_id=conversation_id,
                    block_index=idx,
                )
            )
    return blocks


def make_content_block(
    *,
    message_id: str,
    conversation_id: str,
    block_index: int,
    block_type: str = "text",
    text: str | None = None,
    tool_name: str | None = None,
    tool_id: str | None = None,
    tool_input: str | None = None,
    media_type: str | None = None,
    metadata: str | None = None,
    semantic_type: str | None = None,
) -> ContentBlockRecord:
    return _content_block_record(
        message_id=message_id,
        conversation_id=conversation_id,
        block_index=block_index,
        block_type=block_type,
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        media_type=media_type,
        metadata=metadata,
        semantic_type=semantic_type,
    )


# =============================================================================
# STORE FUNCTIONS (moved from store.py for testing)
# =============================================================================


def _prune_attachment_refs(conn: sqlite3.Connection, conversation_id: str, keep_ref_ids: set[str]) -> None:
    """Prune old attachment references for a conversation."""
    query = "SELECT ref_id, attachment_id FROM attachment_refs WHERE conversation_id = ?"
    params: list[str] = [conversation_id]
    if keep_ref_ids:
        placeholders = ", ".join("?" for _ in keep_ref_ids)
        query += f" AND ref_id NOT IN ({placeholders})"
        params.extend(sorted(keep_ref_ids))
    rows = conn.execute(query, tuple(params)).fetchall()
    if not rows:
        return

    ref_ids = [row["ref_id"] for row in rows]
    attachments = {row["attachment_id"] for row in rows}

    # Use SAVEPOINT for atomic multi-step ref_count operations
    # If interrupted, all changes rollback to prevent incorrect ref_count
    conn.execute("SAVEPOINT prune_attachment_refs")
    try:
        placeholders = ", ".join("?" for _ in ref_ids)
        conn.execute(
            f"DELETE FROM attachment_refs WHERE ref_id IN ({placeholders})",
            tuple(ref_ids),
        )

        # Recalculate ref_count from actual attachment_refs table
        # This is race-safe: instead of decrementing (which could race),
        # we recompute from source of truth using COUNT(*)
        # Single UPDATE query with IN clause instead of N individual queries
        if attachments:
            att_placeholders = ", ".join("?" for _ in attachments)
            conn.execute(
                f"""
                UPDATE attachments
                SET ref_count = (
                    SELECT COUNT(*)
                    FROM attachment_refs
                    WHERE attachment_refs.attachment_id = attachments.attachment_id
                )
                WHERE attachment_id IN ({att_placeholders})
                """,
                tuple(attachments),
            )
        conn.execute("DELETE FROM attachments WHERE ref_count <= 0")
        conn.execute("RELEASE SAVEPOINT prune_attachment_refs")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT prune_attachment_refs")
        raise


def upsert_conversation(conn: sqlite3.Connection, record: ConversationRecord) -> bool:
    """Upsert a conversation record."""
    res = conn.execute(
        """
        INSERT INTO conversations (
            conversation_id,
            provider_name,
            provider_conversation_id,
            title,
            created_at,
            updated_at,
            sort_key,
            content_hash,
            provider_meta,
            metadata,
            version,
            parent_conversation_id,
            branch_type,
            raw_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            sort_key = excluded.sort_key,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta,
            metadata = excluded.metadata,
            parent_conversation_id = excluded.parent_conversation_id,
            branch_type = excluded.branch_type,
            raw_id = COALESCE(excluded.raw_id, conversations.raw_id)
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(title, '') != IFNULL(excluded.title, '')
            OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
            OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
            OR IFNULL(metadata, '') != IFNULL(excluded.metadata, '')
            OR IFNULL(parent_conversation_id, '') != IFNULL(excluded.parent_conversation_id, '')
            OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
            OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
        """,
        (
            record.conversation_id,
            record.provider_name,
            record.provider_conversation_id,
            record.title,
            record.created_at,
            record.updated_at,
            record.sort_key,
            record.content_hash,
            _json_or_none(record.provider_meta),
            _json_or_none(record.metadata),
            record.version,
            record.parent_conversation_id,
            record.branch_type,
            record.raw_id,
        ),
    )
    return bool(res.rowcount > 0)


def upsert_message(conn: sqlite3.Connection, record: MessageRecord) -> bool:
    """Upsert a message record."""
    res = conn.execute(
        """
        INSERT INTO messages (
            message_id,
            conversation_id,
            provider_message_id,
            role,
            text,
            sort_key,
            content_hash,
            version,
            parent_message_id,
            branch_index,
            provider_name,
            word_count,
            has_tool_use,
            has_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            has_thinking = excluded.has_thinking
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(role, '') != IFNULL(excluded.role, '')
            OR IFNULL(text, '') != IFNULL(excluded.text, '')
            OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
            OR branch_index != excluded.branch_index
        """,
        (
            record.message_id,
            record.conversation_id,
            record.provider_message_id,
            record.role,
            record.text,
            record.sort_key,
            record.content_hash,
            record.version,
            record.parent_message_id,
            record.branch_index,
            record.provider_name,
            record.word_count,
            record.has_tool_use,
            record.has_thinking,
        ),
    )
    updated = bool(res.rowcount > 0)

    # Persist content blocks if any
    for blk in record.content_blocks:
        conn.execute(
            """
            INSERT INTO content_blocks (
                block_id, message_id, conversation_id, block_index,
                type, text, tool_name, tool_id, tool_input, media_type, metadata, semantic_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, block_index) DO UPDATE SET
                type = excluded.type,
                text = excluded.text,
                tool_name = excluded.tool_name,
                tool_id = excluded.tool_id,
                tool_input = excluded.tool_input,
                semantic_type = excluded.semantic_type
            """,
            (
                blk.block_id,
                blk.message_id,
                blk.conversation_id,
                blk.block_index,
                blk.type,
                blk.text,
                blk.tool_name,
                blk.tool_id,
                blk.tool_input,
                blk.media_type,
                blk.metadata,
                blk.semantic_type,
            ),
        )

    return updated


def upsert_attachment(conn: sqlite3.Connection, record: AttachmentRecord) -> bool:
    """Upsert an attachment record."""
    # Ensure attachment metadata exists (idempotent, doesn't touch ref_count)
    conn.execute(
        """
        INSERT INTO attachments (
            attachment_id,
            mime_type,
            size_bytes,
            path,
            ref_count,
            provider_meta
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(attachment_id) DO UPDATE SET
            mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
            size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
            path = COALESCE(excluded.path, attachments.path),
            provider_meta = COALESCE(excluded.provider_meta, attachments.provider_meta)
        """,
        (
            record.attachment_id,
            record.mime_type,
            record.size_bytes,
            record.path,
            0,
            _json_or_none(record.provider_meta),
        ),
    )

    # Atomically insert ref and increment count in a single statement
    # This prevents race conditions where multiple threads could increment simultaneously
    ref_id = _make_ref_id(record.attachment_id, record.conversation_id, record.message_id)
    res = conn.execute(
        """
        INSERT OR IGNORE INTO attachment_refs (
            ref_id,
            attachment_id,
            conversation_id,
            message_id,
            provider_meta
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            ref_id,
            record.attachment_id,
            record.conversation_id,
            record.message_id,
            _json_or_none(record.provider_meta),
        ),
    )

    # Only increment if we actually inserted a new ref
    # Use atomic increment to avoid read-modify-write race
    if res.rowcount > 0:
        conn.execute(
            "UPDATE attachments SET ref_count = ref_count + 1 WHERE attachment_id = ?",
            (record.attachment_id,),
        )
        return True
    return False


def record_run(conn: sqlite3.Connection, record: RunRecord) -> None:
    """Record a pipeline run."""
    conn.execute(
        """
        INSERT INTO runs (
            run_id,
            timestamp,
            plan_snapshot,
            counts_json,
            drift_json,
            indexed,
            duration_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.run_id,
            record.timestamp,
            _json_or_none(record.plan_snapshot),
            _json_or_none(record.counts),
            _json_or_none(record.drift),
            int(record.indexed) if record.indexed is not None else None,
            record.duration_ms,
        ),
    )


def store_records(
    *,
    conversation: ConversationRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    """Store conversation records (conversation, messages, attachments).

    Thread-safe with write lock. Returns count of inserted/updated records.
    """
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    with connection_context(conn) as db_conn, _WRITE_LOCK:
        if upsert_conversation(db_conn, conversation):
            counts["conversations"] += 1
        else:
            counts["skipped_conversations"] += 1
        for message in messages:
            if upsert_message(db_conn, message):
                counts["messages"] += 1
            else:
                counts["skipped_messages"] += 1
        seen_ref_ids: set[str] = set()
        for attachment in attachments:
            ref_id = _make_ref_id(attachment.attachment_id, attachment.conversation_id, attachment.message_id)
            seen_ref_ids.add(ref_id)
            if upsert_attachment(db_conn, attachment):
                counts["attachments"] += 1
            else:
                counts["skipped_attachments"] += 1
        _prune_attachment_refs(db_conn, conversation.conversation_id, seen_ref_ids)
        # Commit inside lock to ensure atomic transaction boundaries
        db_conn.commit()

    return counts


# =============================================================================
# DATABASE SETUP UTILITIES
# =============================================================================


def db_setup(workspace_env: Mapping[str, Path]) -> Path:
    """Initialize database path in workspace environment."""
    db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# =============================================================================
# MESSAGE/CONVERSATION BUILDERS (Fluent API)
# =============================================================================


class ConversationBuilder:
    """Fluent builder for creating conversations in test databases."""

    def __init__(self, db_path: Path, conversation_id: str) -> None:
        self.db_path = db_path
        now = datetime.now(timezone.utc).isoformat()
        self.conv = ConversationRecord(
            conversation_id=_conversation_id(conversation_id),
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test Conversation",
            created_at=now,
            updated_at=now,
            sort_key=_timestamp_sort_key(now),
            content_hash=_content_hash(uuid4().hex),
        )
        self.messages: list[MessageRecord] = []
        self.attachments: list[AttachmentRecord] = []

    def title(self, title: str | None) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"title": title})
        return self

    def provider(self, provider: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"provider_name": provider})
        return self

    def created_at(self, created_at: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"created_at": created_at})
        return self

    def updated_at(self, updated_at: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"updated_at": updated_at, "sort_key": _timestamp_sort_key(updated_at)})
        return self

    def metadata(self, metadata: JSONRecord | None) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"metadata": metadata})
        return self

    def parent_conversation(self, parent_id: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"parent_conversation_id": _conversation_id(parent_id)})
        return self

    def branch_type(self, branch_type: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"branch_type": BranchType(branch_type)})
        return self

    def add_message(
        self,
        message_id: str | None = None,
        role: str | None = "user",
        text: str = "Test message",
        timestamp: str | None | _AutoTimestampSentinel = _AUTO_TIMESTAMP,
        **kwargs: object,
    ) -> ConversationBuilder:
        msg_id = f"m{len(self.messages) + 1}" if message_id is None else message_id
        ts = _resolve_timestamp(timestamp)

        provider_meta = _optional_dict(kwargs.pop("provider_meta", None))
        extra_blocks = _normalize_content_blocks(
            raw_blocks=provider_meta.get("content_blocks") if provider_meta is not None else None,
            message_id=msg_id,
            conversation_id=str(self.conv.conversation_id),
        )
        existing_blocks = _normalize_content_blocks(
            raw_blocks=kwargs.pop("content_blocks", []),
            message_id=msg_id,
            conversation_id=str(self.conv.conversation_id),
        )
        all_blocks = [*extra_blocks, *existing_blocks]

        block_types = {blk.type for blk in all_blocks}
        role_value = None if role is None else Role.normalize(role)
        word_count = len(text.split()) if text.strip() else 0
        has_tool_use = (
            1
            if (block_types & {ContentBlockType.TOOL_USE, ContentBlockType.TOOL_RESULT}) or role_value is Role.TOOL
            else 0
        )
        has_thinking = 1 if ContentBlockType.THINKING in block_types else 0
        default_sort_key = _timestamp_sort_key(ts) if ts is not None else None
        default_content_hash = uuid4().hex[:16]

        payload: JSONRecord = {
            "message_id": _message_id(msg_id),
            "conversation_id": self.conv.conversation_id,
            "role": role_value,
            "text": text,
            "sort_key": _coerce_sort_key(
                kwargs.pop("sort_key", default_sort_key),
                default_sort_key,
            ),
            "content_hash": _coerce_content_hash(
                kwargs.pop("content_hash", default_content_hash), default_content_hash
            ),
            "content_blocks": all_blocks,
            "word_count": _coerce_int(kwargs.pop("word_count", word_count), word_count),
            "has_tool_use": _coerce_int(kwargs.pop("has_tool_use", has_tool_use), has_tool_use),
            "has_thinking": _coerce_int(kwargs.pop("has_thinking", has_thinking), has_thinking),
        }
        payload.update(kwargs)
        msg = MessageRecord.model_validate(payload)
        self.messages.append(msg)
        return self

    def add_attachment(
        self,
        attachment_id: str | None = None,
        message_id: str | None | _AutoMessageIdSentinel = _AUTO_MESSAGE_ID,
        mime_type: str = "application/octet-stream",
        size_bytes: int = 1024,
        path: str | None = None,
        provider_meta: JSONRecord | None = None,
    ) -> ConversationBuilder:
        att_id = f"att{len(self.attachments) + 1}" if attachment_id is None else attachment_id
        resolved_message_id = _resolve_attachment_message_id(value=message_id, messages=self.messages)
        att = AttachmentRecord(
            attachment_id=_attachment_id(att_id),
            conversation_id=self.conv.conversation_id,
            message_id=None if resolved_message_id is None else _message_id(resolved_message_id),
            mime_type=mime_type,
            size_bytes=size_bytes,
            path=path,
            provider_meta=provider_meta,
        )
        self.attachments.append(att)
        return self

    def save(self) -> ConversationRecord:
        with open_connection(self.db_path) as conn:
            store_records(
                conversation=self.conv,
                messages=self.messages,
                attachments=self.attachments,
                conn=conn,
            )
        return self.conv

    async def build(self) -> Conversation | None:
        from polylogue.storage.backends.async_sqlite import SQLiteBackend
        from polylogue.storage.repository import ConversationRepository

        self.save()
        async with ConversationRepository(backend=SQLiteBackend(db_path=self.db_path)) as repo:
            return await repo.get(self.conv.conversation_id)


# =============================================================================
# QUICK BUILDERS (For simple cases)
# =============================================================================


def make_hash(s: str) -> str:
    """Create a 16-char content hash for test data."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def make_conversation(
    conversation_id: str = "conv1",
    provider_name: str = "test",
    title: str = "Test Conversation",
    created_at: str | None = None,
    updated_at: str | None = None,
    **kwargs: object,
) -> ConversationRecord:
    now = datetime.now(timezone.utc).isoformat()
    default_content_hash = uuid4().hex
    payload: JSONRecord = {
        "conversation_id": _conversation_id(conversation_id),
        "provider_name": provider_name,
        "provider_conversation_id": _coerce_str(
            kwargs.pop("provider_conversation_id", f"ext-{conversation_id}"),
            f"ext-{conversation_id}",
        ),
        "title": title,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
        "content_hash": _coerce_content_hash(kwargs.pop("content_hash", default_content_hash), default_content_hash),
    }
    payload.update(kwargs)
    return ConversationRecord.model_validate(payload)


def make_message(
    message_id: str = "m1",
    conversation_id: str = "conv1",
    role: str = "user",
    text: str | None = "Test message",
    timestamp: str | None = None,
    **kwargs: object,
) -> MessageRecord:
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    provider_meta = _optional_dict(kwargs.pop("provider_meta", None))
    extra_blocks = _normalize_content_blocks(
        raw_blocks=provider_meta.get("content_blocks") if provider_meta is not None else None,
        message_id=message_id,
        conversation_id=conversation_id,
    )
    existing_blocks = _normalize_content_blocks(
        raw_blocks=kwargs.pop("content_blocks", []),
        message_id=message_id,
        conversation_id=conversation_id,
    )
    all_blocks = [*extra_blocks, *existing_blocks]

    block_types = {blk.type for blk in all_blocks}
    role_value = Role.normalize(role)
    word_count = len(text.split()) if isinstance(text, str) and text.strip() else 0
    has_tool_use = (
        1 if (block_types & {ContentBlockType.TOOL_USE, ContentBlockType.TOOL_RESULT}) or role_value is Role.TOOL else 0
    )
    has_thinking = 1 if ContentBlockType.THINKING in block_types else 0
    default_sort_key = _timestamp_sort_key(ts) if ts is not None else None
    default_content_hash = uuid4().hex[:16]

    payload: JSONRecord = {
        "message_id": _message_id(message_id),
        "conversation_id": _conversation_id(conversation_id),
        "role": role_value,
        "text": text,
        "sort_key": _coerce_sort_key(
            kwargs.pop("sort_key", default_sort_key),
            default_sort_key,
        ),
        "content_hash": _coerce_content_hash(kwargs.pop("content_hash", default_content_hash), default_content_hash),
        "content_blocks": all_blocks,
        "word_count": _coerce_int(kwargs.pop("word_count", word_count), word_count),
        "has_tool_use": _coerce_int(kwargs.pop("has_tool_use", has_tool_use), has_tool_use),
        "has_thinking": _coerce_int(kwargs.pop("has_thinking", has_thinking), has_thinking),
    }
    payload.update(kwargs)
    return MessageRecord.model_validate(payload)


def make_attachment(
    attachment_id: str = "att1",
    conversation_id: str = "conv1",
    message_id: str | None = None,
    mime_type: str = "application/octet-stream",
    size_bytes: int = 1024,
    name: str | None = None,
    **kwargs: object,
) -> AttachmentRecord:
    provider_meta = _optional_dict(kwargs.pop("provider_meta", None))
    if name and provider_meta is None:
        provider_meta = {"name": name}

    payload: JSONRecord = {
        "attachment_id": _attachment_id(attachment_id),
        "conversation_id": _conversation_id(conversation_id),
        "message_id": None if message_id is None else _message_id(message_id),
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "provider_meta": provider_meta,
    }
    payload.update(kwargs)
    return AttachmentRecord.model_validate(payload)


def make_raw_conversation(
    raw_id: str = "raw1",
    provider_name: str = "test",
    source_path: str = "/tmp/test.json",
    *,
    blob_size: int = 2,
    acquired_at: str | None = None,
    payload_provider: str | Provider | None = None,
    validation_status: str | ValidationStatus | None = None,
    validation_provider: str | Provider | None = None,
    validation_mode: str | ValidationMode | None = None,
    **kwargs: object,
) -> RawConversationRecord:
    timestamp = acquired_at or datetime.now(timezone.utc).isoformat()
    payload: JSONRecord = {
        "raw_id": raw_id,
        "provider_name": provider_name,
        "source_path": source_path,
        "blob_size": blob_size,
        "acquired_at": timestamp,
        "payload_provider": (
            payload_provider
            if isinstance(payload_provider, Provider) or payload_provider is None
            else Provider.from_string(payload_provider)
        ),
        "validation_status": (
            validation_status
            if isinstance(validation_status, ValidationStatus) or validation_status is None
            else ValidationStatus.from_string(validation_status)
        ),
        "validation_provider": (
            validation_provider
            if isinstance(validation_provider, Provider) or validation_provider is None
            else Provider.from_string(validation_provider)
        ),
        "validation_mode": (
            validation_mode
            if isinstance(validation_mode, ValidationMode) or validation_mode is None
            else ValidationMode.from_string(validation_mode)
        ),
    }
    payload.update(kwargs)
    return RawConversationRecord.model_validate(payload)


class DbFactory:
    """Low-ceremony DB seeder built on top of ConversationBuilder."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def create_conversation(
        self,
        id: str | None = None,
        provider: str = "test",
        title: str = "Test Conversation",
        messages: list[JSONRecord] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        metadata: JSONRecord | None = None,
    ) -> str:
        cid = id or str(uuid4())
        created_iso = (created_at or datetime.now(timezone.utc)).isoformat()
        updated_iso = (updated_at or datetime.now(timezone.utc)).isoformat()

        builder = (
            ConversationBuilder(self.db_path, cid)
            .provider(provider)
            .title(title)
            .created_at(created_iso)
            .updated_at(updated_iso)
            .metadata(metadata)
        )

        for msg in messages or []:
            message_id = _optional_str(msg.get("id"))
            text_value = _optional_str(msg.get("text"))
            if text_value is None:
                text_value = _optional_str(msg.get("content")) or "hello"
            message_kwargs: dict[str, object] = {
                "provider_message_id": _optional_str(msg.get("provider_message_id")),
                "parent_message_id": _optional_str(msg.get("parent_message_id")),
                "branch_index": _optional_int(msg.get("branch_index")) or 0,
                "content_blocks": msg.get("content_blocks", []),
            }
            provider_meta = _optional_dict(msg.get("provider_meta"))
            if provider_meta is not None:
                message_kwargs["provider_meta"] = provider_meta
            if (word_count := _optional_int(msg.get("word_count"))) is not None:
                message_kwargs["word_count"] = word_count
            if (has_tool_use := _optional_int(msg.get("has_tool_use"))) is not None:
                message_kwargs["has_tool_use"] = has_tool_use
            if (has_thinking := _optional_int(msg.get("has_thinking"))) is not None:
                message_kwargs["has_thinking"] = has_thinking

            builder.add_message(
                message_id=message_id,
                role=_optional_str(msg.get("role")) or "user",
                text=text_value,
                timestamp=_coerce_builder_timestamp(msg.get("timestamp", _AUTO_TIMESTAMP)),
                **message_kwargs,
            )

            attachments = msg.get("attachments")
            if not isinstance(attachments, list):
                continue
            for raw_attachment in attachments:
                if not isinstance(raw_attachment, Mapping):
                    continue
                builder.add_attachment(
                    attachment_id=_optional_str(raw_attachment.get("id")),
                    message_id=message_id if message_id is not None else _AUTO_MESSAGE_ID,
                    mime_type=_optional_str(raw_attachment.get("mime_type")) or "application/octet-stream",
                    size_bytes=_optional_int(raw_attachment.get("size_bytes")) or 1024,
                    path=_optional_str(raw_attachment.get("path")),
                    provider_meta=(
                        _optional_dict(raw_attachment.get("meta"))
                        or _optional_dict(raw_attachment.get("provider_meta"))
                    ),
                )

        builder.save()
        return cid
