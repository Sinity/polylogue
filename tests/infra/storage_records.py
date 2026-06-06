"""Shared storage-record builders and DB seeding helpers for tests."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias, cast
from uuid import uuid4

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.json import dumps, loads, require_json_document, require_json_value
from polylogue.pipeline.prepare import _timestamp_sort_key
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
)
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    MessageRecord,
    RawSessionRecord,
    SessionRecord,
    _json_or_none,
    _make_ref_id,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection import connection_context
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    MessageId,
    Provider,
    SemanticBlockType,
    SessionId,
    ValidationMode,
    ValidationStatus,
)

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session

JSONRecord: TypeAlias = dict[str, object]
MessageMapping: TypeAlias = Mapping[str, object]
RecordPayload: TypeAlias = dict[str, object]


class _AutoTimestampSentinel:
    """Marker for builders that should synthesize a fresh timestamp."""


class _AutoMessageIdSentinel:
    """Marker for builders that should target the most recent message."""


# Thread-safety lock for writes (matches store.py pattern)
_WRITE_LOCK = threading.Lock()
_AUTO_TIMESTAMP: Final = _AutoTimestampSentinel()
_AUTO_MESSAGE_ID: Final = _AutoMessageIdSentinel()


def _session_id(value: str) -> SessionId:
    return SessionId(value)


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


def _optional_json_document(value: object, *, context: str = "JSON object") -> JSONRecord | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return cast(JSONRecord, dict(require_json_document(dict(value), context=context)))


def _json_string_or_none(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return dumps(require_json_value(value, context=context))


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


def _merge_media_type_into_metadata(metadata: str | None, media_type: str | None) -> str | None:
    """#1240: store media_type inside the block-metadata JSON envelope."""
    if not media_type:
        return metadata
    base: dict[str, object] = {}
    if metadata:
        try:
            parsed = loads(metadata)
        except Exception:
            return metadata
        if isinstance(parsed, dict):
            base.update(parsed)
    base.setdefault("media_type", media_type)
    return dumps(base)


def _content_block_record(
    *,
    message_id: str,
    session_id: str,
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
    # #1240: media_type is now stored inside the block-metadata JSON.
    merged_metadata = _merge_media_type_into_metadata(metadata, media_type)
    return ContentBlockRecord(
        block_id=ContentBlockRecord.make_id(message_id, block_index),
        message_id=_message_id(message_id),
        session_id=_session_id(session_id),
        block_index=block_index,
        type=ContentBlockType.from_string(block_type),
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        metadata=merged_metadata,
        semantic_type=None if semantic_type is None else SemanticBlockType.from_string(semantic_type),
    )


def _content_block_from_mapping(
    *,
    block: MessageMapping,
    message_id: str,
    session_id: str,
    block_index: int,
) -> ContentBlockRecord:
    raw_tool_input = block.get("tool_input", block.get("input"))
    raw_metadata = block.get("metadata")
    return _content_block_record(
        message_id=message_id,
        session_id=session_id,
        block_index=block_index,
        block_type=_optional_str(block.get("type")) or "text",
        text=_optional_str(block.get("text")),
        tool_name=_optional_str(block.get("tool_name")) or _optional_str(block.get("name")),
        tool_id=_optional_str(block.get("tool_id")) or _optional_str(block.get("id")),
        tool_input=_json_string_or_none(raw_tool_input, context="content block tool input"),
        media_type=_optional_str(block.get("media_type")),
        metadata=_json_string_or_none(raw_metadata, context="content block metadata"),
        semantic_type=_optional_str(block.get("semantic_type")),
    )


def _normalize_content_blocks(
    *,
    raw_blocks: object,
    message_id: str,
    session_id: str,
) -> list[ContentBlockRecord]:
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[ContentBlockRecord] = []
    for idx, raw_block in enumerate(raw_blocks):
        if isinstance(raw_block, ContentBlockRecord):
            blocks.append(raw_block)
            continue
        if isinstance(raw_block, Mapping):
            if not all(isinstance(key, str) for key in raw_block):
                raise TypeError("content block keys must be strings")
            blocks.append(
                _content_block_from_mapping(
                    block=cast(MessageMapping, raw_block),
                    message_id=message_id,
                    session_id=session_id,
                    block_index=idx,
                )
            )
    return blocks


def make_content_block(
    *,
    message_id: str,
    session_id: str,
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
        session_id=session_id,
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


def _prune_attachment_refs(conn: sqlite3.Connection, session_id: str, keep_ref_ids: set[str]) -> None:
    """Prune old attachment references for a session."""
    query = "SELECT ref_id, attachment_id FROM attachment_refs WHERE session_id = ?"
    params: list[str] = [session_id]
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


def upsert_session(conn: sqlite3.Connection, record: SessionRecord) -> bool:
    """Upsert a session record."""
    res = conn.execute(
        """
        INSERT INTO sessions (
            session_id,
            source_name,
            provider_session_id,
            title,
            created_at,
            updated_at,
            sort_key,
            content_hash,
            provider_meta,
            metadata,
            version,
            parent_session_id,
            branch_type,
            raw_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            sort_key = excluded.sort_key,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta,
            metadata = excluded.metadata,
            parent_session_id = excluded.parent_session_id,
            branch_type = excluded.branch_type,
            raw_id = COALESCE(excluded.raw_id, sessions.raw_id)
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(title, '') != IFNULL(excluded.title, '')
            OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
            OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
            OR IFNULL(metadata, '') != IFNULL(excluded.metadata, '')
            OR IFNULL(parent_session_id, '') != IFNULL(excluded.parent_session_id, '')
            OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
            OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
        """,
        (
            record.session_id,
            record.source_name,
            record.provider_session_id,
            record.title,
            record.created_at,
            record.updated_at,
            record.sort_key,
            record.content_hash,
            _json_or_none(record.provider_meta),
            _json_or_none(record.metadata),
            record.version,
            record.parent_session_id,
            record.branch_type,
            record.raw_id,
        ),
    )
    return bool(res.rowcount > 0)


def upsert_message(conn: sqlite3.Connection, record: MessageRecord) -> bool:
    """Upsert a message record — must match the canonical _MESSAGE_UPSERT_SQL column set."""
    res = conn.execute(
        """
        INSERT INTO messages (
            message_id,
            session_id,
            provider_message_id,
            role,
            text,
            sort_key,
            content_hash,
            version,
            parent_message_id,
            branch_index,
            source_name,
            word_count,
            has_tool_use,
            has_thinking,
            has_paste,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            model_name,
            message_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            role = excluded.role,
            text = excluded.text,
            sort_key = excluded.sort_key,
            content_hash = excluded.content_hash,
            parent_message_id = excluded.parent_message_id,
            branch_index = excluded.branch_index,
            source_name = excluded.source_name,
            word_count = excluded.word_count,
            has_tool_use = excluded.has_tool_use,
            has_thinking = excluded.has_thinking,
            has_paste = excluded.has_paste,
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            cache_read_tokens = excluded.cache_read_tokens,
            cache_write_tokens = excluded.cache_write_tokens,
            model_name = excluded.model_name,
            message_type = excluded.message_type
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(role, '') != IFNULL(excluded.role, '')
            OR IFNULL(text, '') != IFNULL(excluded.text, '')
            OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
            OR branch_index != excluded.branch_index
            OR has_paste != excluded.has_paste
            OR input_tokens != excluded.input_tokens
            OR output_tokens != excluded.output_tokens
            OR cache_read_tokens != excluded.cache_read_tokens
            OR cache_write_tokens != excluded.cache_write_tokens
            OR IFNULL(model_name, '') != IFNULL(excluded.model_name, '')
            OR IFNULL(message_type, '') != IFNULL(excluded.message_type, '')
        """,
        (
            record.message_id,
            record.session_id,
            record.provider_message_id,
            record.role,
            record.text,
            record.sort_key,
            record.content_hash,
            record.version,
            record.parent_message_id,
            record.branch_index,
            record.source_name,
            record.word_count,
            record.has_tool_use,
            record.has_thinking,
            record.has_paste,
            record.input_tokens,
            record.output_tokens,
            record.cache_read_tokens,
            record.cache_write_tokens,
            record.model_name,
            record.message_type.value,
        ),
    )
    updated = bool(res.rowcount > 0)

    # Persist content blocks if any
    for blk in record.content_blocks:
        conn.execute(
            """
            INSERT INTO content_blocks (
                block_id, message_id, session_id, block_index,
                type, text, tool_name, tool_id, tool_input, metadata, semantic_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                blk.session_id,
                blk.block_index,
                blk.type,
                blk.text,
                blk.tool_name,
                blk.tool_id,
                blk.tool_input,
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
    ref_id = _make_ref_id(record.attachment_id, record.session_id, record.message_id)
    res = conn.execute(
        """
        INSERT INTO attachment_refs (
            ref_id,
            attachment_id,
            session_id,
            message_id,
            provider_meta
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ref_id) DO NOTHING
        """,
        (
            ref_id,
            record.attachment_id,
            record.session_id,
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


def store_records(
    *,
    session: SessionRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    """Store session records (session, messages, attachments).

    Thread-safe with write lock. Returns count of inserted/updated records.
    """
    counts = {
        "sessions": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_sessions": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    with connection_context(conn) as db_conn, _WRITE_LOCK:
        if upsert_session(db_conn, session):
            counts["sessions"] += 1
        else:
            counts["skipped_sessions"] += 1
        for message in messages:
            if upsert_message(db_conn, message):
                counts["messages"] += 1
            else:
                counts["skipped_messages"] += 1
        seen_ref_ids: set[str] = set()
        for attachment in attachments:
            ref_id = _make_ref_id(attachment.attachment_id, attachment.session_id, attachment.message_id)
            seen_ref_ids.add(ref_id)
            if upsert_attachment(db_conn, attachment):
                counts["attachments"] += 1
            else:
                counts["skipped_attachments"] += 1
        _prune_attachment_refs(db_conn, session.session_id, seen_ref_ids)
        # Mirror the production write path's stats projection so that
        # surfaces relying on session_stats (the stats-join pushdown
        # filters: min_messages/max_messages/min_words/has_tool_use/…)
        # see the same precomputed values as production ingest.  Without
        # this, the test builders produced rows but no stats, and any
        # parity test exercising those filters would observe an empty
        # result set on every surface.
        _upsert_session_stats_sync(db_conn, session=session, messages=messages)
        # Mirror save_via_backend's identity-preserving repoint pass (#1114) so
        # tests that exercise reset/reimport via the sync builder see the same
        # rebinding behaviour as production async ingest.
        _repoint_user_state_by_identity_sync(db_conn, session.session_id)
        # Commit inside lock to ensure atomic transaction boundaries
        db_conn.commit()

    return counts


def _repoint_user_state_by_identity_sync(conn: sqlite3.Connection, session_id: str) -> None:
    """Sync mirror of ``repoint_user_state_by_identity`` for #1114 test parity."""
    conv_key = f"session:{session_id}"
    msg_key_prefix = f"message:{session_id}:"
    conn.execute(
        """
        UPDATE user_marks SET session_id = ?
        WHERE identity_key = ? AND target_type = 'session'
          AND (session_id IS NULL OR session_id != ?)
        """,
        (session_id, conv_key, session_id),
    )
    conn.execute(
        """
        UPDATE user_annotations SET session_id = ?
        WHERE identity_key = ? AND target_type = 'session'
          AND (session_id IS NULL OR session_id != ?)
        """,
        (session_id, conv_key, session_id),
    )
    conn.execute(
        """
        UPDATE user_marks SET session_id = ?, message_id = target_id
        WHERE identity_key LIKE ? AND target_type = 'message'
          AND EXISTS (SELECT 1 FROM messages m WHERE m.message_id = user_marks.target_id AND m.session_id = ?)
          AND (session_id IS NULL OR message_id IS NULL)
        """,
        (session_id, f"{msg_key_prefix}%", session_id),
    )
    conn.execute(
        """
        UPDATE user_annotations SET session_id = ?, message_id = target_id
        WHERE identity_key LIKE ? AND target_type = 'message'
          AND EXISTS (SELECT 1 FROM messages m WHERE m.message_id = user_annotations.target_id AND m.session_id = ?)
          AND (session_id IS NULL OR message_id IS NULL)
        """,
        (session_id, f"{msg_key_prefix}%", session_id),
    )


def _upsert_session_stats_sync(
    conn: sqlite3.Connection,
    *,
    session: SessionRecord,
    messages: list[MessageRecord],
) -> None:
    """Sync mirror of ``upsert_session_stats`` for test seeding.

    Production routes stats through the async backend; the test
    ``SessionBuilder`` uses a sync sqlite3 connection, so we share
    the same SQL statement constant but execute it synchronously here.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.common import SQL_STATS_UPSERT

    message_count = len(messages)
    word_count = sum(m.word_count for m in messages)
    tool_use_count = sum(1 for m in messages if m.has_tool_use)
    thinking_count = sum(1 for m in messages if m.has_thinking)
    paste_count = sum(1 for m in messages if m.has_paste)
    user_msg_count = sum(1 for m in messages if m.role == Role.USER)
    assistant_msg_count = sum(1 for m in messages if m.role == Role.ASSISTANT)
    system_msg_count = sum(1 for m in messages if m.role == Role.SYSTEM)
    tool_msg_count = sum(1 for m in messages if m.role == Role.TOOL)
    user_word_count = sum(m.word_count for m in messages if m.role == Role.USER)
    assistant_word_count = sum(m.word_count for m in messages if m.role == Role.ASSISTANT)
    conn.execute(
        SQL_STATS_UPSERT,
        (
            session.session_id,
            session.source_name,
            message_count,
            word_count,
            tool_use_count,
            thinking_count,
            paste_count,
            user_msg_count,
            assistant_msg_count,
            system_msg_count,
            tool_msg_count,
            user_word_count,
            assistant_word_count,
        ),
    )


# =============================================================================
# DATABASE SETUP UTILITIES
# =============================================================================


def db_setup(workspace_env: Mapping[str, Path]) -> Path:
    """Return the archive's index database path inside the workspace.

    Seeding and reads both resolve to the configured archive root, so the
    builders write the same store the facade/CLI/MCP read.
    """
    root = workspace_env["archive_root"]
    root.mkdir(parents=True, exist_ok=True)
    return root / "index.db"


def _archive_root_for(db_path: Path) -> Path:
    """Resolve the archive root that contains ``db_path``.

    ``db_path`` is the index database file (``.../index.db``); the archive
    root is its parent directory, where the rest of the store lives.
    """
    return db_path.parent


def _record_to_parsed_session(
    session: SessionRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
) -> ParsedSession:
    """Convert builder records into the parser envelope the archive ingests."""

    def _maybe_json_object(value: object) -> dict[str, object] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str) and value:
            parsed = loads(value)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return None

    def _blocks(message: MessageRecord) -> list[ParsedContentBlock]:
        return [
            ParsedContentBlock(
                type=block.type,
                text=block.text,
                tool_name=block.tool_name,
                tool_id=block.tool_id,
                tool_input=_maybe_json_object(block.tool_input),
                metadata=_maybe_json_object(block.metadata),
            )
            for block in (message.content_blocks or [])
        ]

    parsed_messages = [
        ParsedMessage(
            provider_message_id=str(message.provider_message_id or message.message_id),
            role=message.role if message.role is not None else Role.USER,
            text=message.text,
            content_blocks=_blocks(message),
            message_type=message.message_type,
            parent_message_provider_id=(
                str(message.parent_message_id) if message.parent_message_id is not None else None
            ),
            position=position,
            branch_index=message.branch_index,
            variant_index=message.branch_index,
            occurred_at_ms=(int(message.sort_key * 1000) if message.sort_key is not None else None),
            input_tokens=message.input_tokens,
            output_tokens=message.output_tokens,
            cache_read_tokens=message.cache_read_tokens,
            cache_write_tokens=message.cache_write_tokens,
            model_name=message.model_name,
        )
        for position, message in enumerate(messages)
    ]

    parsed_attachments = [
        ParsedAttachment(
            provider_attachment_id=str(attachment.attachment_id),
            message_provider_id=(str(attachment.message_id) if attachment.message_id is not None else None),
            name=None,
            mime_type=attachment.mime_type,
            size_bytes=attachment.size_bytes,
            path=attachment.path,
            provider_meta=attachment.provider_meta,
        )
        for attachment in attachments
    ]

    return ParsedSession(
        source_name=session.provider,
        provider_session_id=session.provider_session_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=parsed_messages,
        attachments=parsed_attachments,
        provider_meta=session.provider_meta,
        parent_session_provider_id=(str(session.parent_session_id) if session.parent_session_id is not None else None),
        branch_type=session.branch_type,
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
    )


# =============================================================================
# MESSAGE/SESSION BUILDERS (Fluent API)
# =============================================================================


class SessionBuilder:
    """Fluent builder for creating sessions in test databases."""

    def __init__(self, db_path: Path, session_id: str) -> None:
        self.db_path = db_path
        now = datetime.now(timezone.utc).isoformat()
        self.conv = SessionRecord(
            session_id=_session_id(session_id),
            source_name="test",
            provider_session_id=f"ext-{session_id}",
            title="Test Session",
            created_at=now,
            updated_at=now,
            sort_key=_timestamp_sort_key(now),
            content_hash=_content_hash(uuid4().hex),
        )
        self.messages: list[MessageRecord] = []
        self.attachments: list[AttachmentRecord] = []

    def title(self, title: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"title": title})
        return self

    def provider(self, provider: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"source_name": provider})
        return self

    def created_at(self, created_at: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"created_at": created_at})
        return self

    def updated_at(self, updated_at: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"updated_at": updated_at, "sort_key": _timestamp_sort_key(updated_at)})
        return self

    def metadata(self, metadata: JSONRecord | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"metadata": metadata})
        return self

    def provider_meta(self, provider_meta: JSONRecord | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"provider_meta": provider_meta})
        return self

    def parent_session(self, parent_id: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"parent_session_id": _session_id(parent_id)})
        return self

    def branch_type(self, branch_type: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"branch_type": BranchType(branch_type)})
        return self

    def add_message(
        self,
        message_id: str | None = None,
        role: str | None = "user",
        text: str = "Test message",
        timestamp: str | None | _AutoTimestampSentinel = _AUTO_TIMESTAMP,
        **kwargs: object,
    ) -> SessionBuilder:
        msg_id = f"m{len(self.messages) + 1}" if message_id is None else message_id
        ts = _resolve_timestamp(timestamp)

        provider_meta = _optional_json_document(kwargs.pop("provider_meta", None), context="message provider_meta")
        extra_blocks = _normalize_content_blocks(
            raw_blocks=provider_meta.get("content_blocks") if provider_meta is not None else None,
            message_id=msg_id,
            session_id=str(self.conv.session_id),
        )
        existing_blocks = _normalize_content_blocks(
            raw_blocks=kwargs.pop("content_blocks", []),
            message_id=msg_id,
            session_id=str(self.conv.session_id),
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

        payload: RecordPayload = {
            "message_id": _message_id(msg_id),
            "session_id": self.conv.session_id,
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
    ) -> SessionBuilder:
        att_id = f"att{len(self.attachments) + 1}" if attachment_id is None else attachment_id
        resolved_message_id = _resolve_attachment_message_id(value=message_id, messages=self.messages)
        att = AttachmentRecord(
            attachment_id=_attachment_id(att_id),
            session_id=self.conv.session_id,
            message_id=None if resolved_message_id is None else _message_id(resolved_message_id),
            mime_type=mime_type,
            size_bytes=size_bytes,
            path=path,
            provider_meta=provider_meta,
        )
        self.attachments.append(att)
        return self

    def save(self) -> SessionRecord:
        parsed = _record_to_parsed_session(self.conv, self.messages, self.attachments)
        with _WRITE_LOCK, ArchiveStore(_archive_root_for(self.db_path)) as archive:
            archive.write_parsed(parsed)
        return self.conv

    def native_session_id(self) -> str:
        """The archive's deterministic session id for the built session."""
        from polylogue.core.identity_law import session_id as archive_session_id
        from polylogue.core.sources import origin_from_provider

        origin = origin_from_provider(self.conv.provider)
        return archive_session_id(origin.value, self.conv.provider_session_id)

    async def build(self) -> Session | None:
        from polylogue.api import Polylogue

        self.save()
        root = _archive_root_for(self.db_path)
        async with Polylogue(archive_root=root, db_path=root / "index.db") as plg:
            return await plg.get_session(self.native_session_id())


# =============================================================================
# QUICK BUILDERS (For simple cases)
# =============================================================================


def make_hash(s: str) -> str:
    """Create a 16-char content hash for test data."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def make_session(
    session_id: str = "conv1",
    source_name: str = "test",
    title: str = "Test Session",
    created_at: str | None = None,
    updated_at: str | None = None,
    **kwargs: object,
) -> SessionRecord:
    now = datetime.now(timezone.utc).isoformat()
    default_content_hash = uuid4().hex
    payload: RecordPayload = {
        "session_id": _session_id(session_id),
        "source_name": source_name,
        "provider_session_id": _coerce_str(
            kwargs.pop("provider_session_id", f"ext-{session_id}"),
            f"ext-{session_id}",
        ),
        "title": title,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
        "content_hash": _coerce_content_hash(kwargs.pop("content_hash", default_content_hash), default_content_hash),
    }
    payload.update(kwargs)
    return SessionRecord.model_validate(payload)


def make_message(
    message_id: str = "m1",
    session_id: str = "conv1",
    role: str = "user",
    text: str | None = "Test message",
    timestamp: str | None = None,
    **kwargs: object,
) -> MessageRecord:
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    provider_meta = _optional_json_document(kwargs.pop("provider_meta", None), context="message provider_meta")
    extra_blocks = _normalize_content_blocks(
        raw_blocks=provider_meta.get("content_blocks") if provider_meta is not None else None,
        message_id=message_id,
        session_id=session_id,
    )
    existing_blocks = _normalize_content_blocks(
        raw_blocks=kwargs.pop("content_blocks", []),
        message_id=message_id,
        session_id=session_id,
    )
    all_blocks = [*extra_blocks, *existing_blocks]

    block_types = {blk.type for blk in all_blocks}
    role_value = Role.normalize(role)
    word_count = len(text.split()) if isinstance(text, str) and text.strip() else 0
    has_tool_use = (
        1 if (block_types & {ContentBlockType.TOOL_USE, ContentBlockType.TOOL_RESULT}) or role_value is Role.TOOL else 0
    )
    has_thinking = 1 if ContentBlockType.THINKING in block_types else 0
    default_sort_key = _timestamp_sort_key(ts)
    default_content_hash = uuid4().hex[:16]

    payload: RecordPayload = {
        "message_id": _message_id(message_id),
        "session_id": _session_id(session_id),
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
    session_id: str = "conv1",
    message_id: str | None = None,
    mime_type: str = "application/octet-stream",
    size_bytes: int = 1024,
    name: str | None = None,
    **kwargs: object,
) -> AttachmentRecord:
    provider_meta = _optional_json_document(kwargs.pop("provider_meta", None), context="attachment provider_meta")
    if name and provider_meta is None:
        provider_meta = {"name": name}

    payload: RecordPayload = {
        "attachment_id": _attachment_id(attachment_id),
        "session_id": _session_id(session_id),
        "message_id": None if message_id is None else _message_id(message_id),
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "provider_meta": provider_meta,
    }
    payload.update(kwargs)
    return AttachmentRecord.model_validate(payload)


def make_raw_session(
    raw_id: str = "raw1",
    source_name: str = "test",
    source_path: str = "/tmp/test.json",
    *,
    blob_size: int = 2,
    acquired_at: str | None = None,
    payload_provider: str | Provider | None = None,
    validation_status: str | ValidationStatus | None = None,
    validation_provider: str | Provider | None = None,
    validation_mode: str | ValidationMode | None = None,
    **kwargs: object,
) -> RawSessionRecord:
    timestamp = acquired_at or datetime.now(timezone.utc).isoformat()
    payload: RecordPayload = {
        "raw_id": raw_id,
        "source_name": source_name,
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
    return RawSessionRecord.model_validate(payload)


class DbFactory:
    """Low-ceremony DB seeder built on top of SessionBuilder."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        id: str | None = None,
        provider: str = "test",
        title: str = "Test Session",
        messages: list[JSONRecord] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        metadata: JSONRecord | None = None,
    ) -> str:
        cid = id or str(uuid4())
        created_iso = (created_at or datetime.now(timezone.utc)).isoformat()
        updated_iso = (updated_at or datetime.now(timezone.utc)).isoformat()

        builder = (
            SessionBuilder(self.db_path, cid)
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
            provider_meta = _optional_json_document(msg.get("provider_meta"), context="factory message provider_meta")
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
                        _optional_json_document(raw_attachment.get("meta"), context="factory attachment meta")
                        or _optional_json_document(
                            raw_attachment.get("provider_meta"), context="factory attachment provider_meta"
                        )
                    ),
                )

        builder.save()
        return cid
