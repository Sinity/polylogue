"""Shared storage-record builders and DB seeding helpers for tests."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast
from uuid import uuid4

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Origin, Provider, SemanticBlockType, ValidationMode, ValidationStatus
from polylogue.core.json import dumps, loads, require_json_document, require_json_value
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.core.timestamps import _timestamp_sort_key
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
)
from polylogue.storage.runtime import (
    AttachmentRecord,
    BlockRecord,
    MessageRecord,
    RawSessionRecord,
    SessionRecord,
    _make_ref_id,
)
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import connection_context, open_connection
from polylogue.types import AttachmentId, ContentHash, MessageId, SessionId

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
    return ContentHash(_writer_hash(value))


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


def _origin_value(provider: str) -> Origin:
    return origin_from_provider(Provider.from_string(provider))


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
    tool_result_is_error: int | None = None,
    tool_result_exit_code: int | None = None,
) -> BlockRecord:
    # #1240: media_type is now stored inside the block-metadata JSON.
    merged_metadata = _merge_media_type_into_metadata(metadata, media_type)
    return BlockRecord(
        block_id=BlockRecord.make_id(message_id, block_index),
        message_id=_message_id(message_id),
        session_id=_session_id(session_id),
        block_index=block_index,
        type=BlockType.from_string(block_type),
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        metadata=merged_metadata,
        semantic_type=None if semantic_type is None else SemanticBlockType.from_string(semantic_type),
        tool_result_is_error=tool_result_is_error,
        tool_result_exit_code=tool_result_exit_code,
    )


def _content_block_from_mapping(
    *,
    block: MessageMapping,
    message_id: str,
    session_id: str,
    block_index: int,
) -> BlockRecord:
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
        tool_result_is_error=_optional_int(block.get("tool_result_is_error")),
        tool_result_exit_code=_optional_int(block.get("tool_result_exit_code")),
    )


def _normalize_content_blocks(
    *,
    raw_blocks: object,
    message_id: str,
    session_id: str,
) -> list[BlockRecord]:
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[BlockRecord] = []
    for idx, raw_block in enumerate(raw_blocks):
        if isinstance(raw_block, BlockRecord):
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
) -> BlockRecord:
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
    from polylogue.storage.sqlite.archive_tiers.write import _timestamp_ms

    res = conn.execute(
        """
        INSERT INTO sessions (
            native_id,
            origin,
            title,
            content_hash,
            parent_session_id,
            branch_type,
            raw_id,
            git_branch,
            git_repository_url,
            provider_project_ref,
            created_at_ms,
            updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(origin, native_id) DO UPDATE SET
            title = excluded.title,
            content_hash = excluded.content_hash,
            parent_session_id = excluded.parent_session_id,
            branch_type = excluded.branch_type,
            raw_id = COALESCE(excluded.raw_id, sessions.raw_id),
            git_branch = excluded.git_branch,
            git_repository_url = excluded.git_repository_url,
            provider_project_ref = excluded.provider_project_ref,
            created_at_ms = COALESCE(sessions.created_at_ms, excluded.created_at_ms),
            updated_at_ms = excluded.updated_at_ms
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(title, '') != IFNULL(excluded.title, '')
            OR IFNULL(parent_session_id, '') != IFNULL(excluded.parent_session_id, '')
            OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
            OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
            OR IFNULL(git_branch, '') != IFNULL(excluded.git_branch, '')
            OR IFNULL(git_repository_url, '') != IFNULL(excluded.git_repository_url, '')
            OR IFNULL(provider_project_ref, '') != IFNULL(excluded.provider_project_ref, '')
            OR IFNULL(updated_at_ms, 0) != IFNULL(excluded.updated_at_ms, 0)
        """,
        (
            record.native_id,
            record.origin.value,
            record.title,
            bytes.fromhex(_writer_hash(record.content_hash)),
            record.parent_session_id,
            record.branch_type.value if record.branch_type is not None else None,
            record.raw_id,
            record.git_branch,
            record.git_repository_url,
            record.provider_project_ref,
            _timestamp_ms(record.created_at),
            _timestamp_ms(record.updated_at),
        ),
    )
    return bool(res.rowcount > 0)


def upsert_message(conn: sqlite3.Connection, record: MessageRecord) -> bool:
    """Upsert a message record into the current archive schema."""
    session_row = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ? OR native_id = ? ORDER BY session_id LIMIT 2",
        (record.session_id, record.session_id),
    ).fetchall()
    if len(session_row) != 1:
        raise ValueError(f"Cannot write message for unknown or ambiguous session {record.session_id!r}")
    session_id = str(session_row[0]["session_id"])
    native_id = record.provider_message_id or str(record.message_id).removeprefix(f"{session_id}:")
    existing = conn.execute(
        "SELECT position, variant_index FROM messages WHERE session_id = ? AND native_id = ?",
        (session_id, native_id),
    ).fetchone()
    if existing is None:
        position_row = conn.execute(
            "SELECT COALESCE(MAX(position) + 1, 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        position = int(position_row[0] or 0)
        variant_index = record.branch_index
    else:
        position = int(existing["position"])
        variant_index = int(existing["variant_index"])
    res = conn.execute(
        """
        INSERT INTO messages (
            session_id,
            native_id,
            parent_message_id,
            position,
            role,
            message_type,
            model_name,
            has_tool_use,
            has_thinking,
            has_paste,
            paste_boundary,
            variant_index,
            word_count,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            duration_ms,
            content_hash,
            occurred_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, position, variant_index) DO UPDATE SET
            native_id = excluded.native_id,
            parent_message_id = excluded.parent_message_id,
            role = excluded.role,
            message_type = excluded.message_type,
            model_name = excluded.model_name,
            has_tool_use = excluded.has_tool_use,
            has_thinking = excluded.has_thinking,
            has_paste = excluded.has_paste,
            paste_boundary = excluded.paste_boundary,
            word_count = excluded.word_count,
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            cache_read_tokens = excluded.cache_read_tokens,
            cache_write_tokens = excluded.cache_write_tokens,
            duration_ms = excluded.duration_ms,
            content_hash = excluded.content_hash,
            occurred_at_ms = excluded.occurred_at_ms
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(role, '') != IFNULL(excluded.role, '')
            OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
            OR has_paste != excluded.has_paste
            OR input_tokens != excluded.input_tokens
            OR output_tokens != excluded.output_tokens
            OR cache_read_tokens != excluded.cache_read_tokens
            OR cache_write_tokens != excluded.cache_write_tokens
            OR IFNULL(duration_ms, -1) != IFNULL(excluded.duration_ms, -1)
            OR IFNULL(model_name, '') != IFNULL(excluded.model_name, '')
            OR IFNULL(message_type, '') != IFNULL(excluded.message_type, '')
        """,
        (
            session_id,
            native_id,
            record.parent_message_id,
            position,
            record.role.value if record.role is not None else "unknown",
            record.message_type.value,
            record.model_name,
            record.has_tool_use,
            record.has_thinking,
            record.has_paste,
            record.paste_boundary_state,
            variant_index,
            record.word_count,
            record.input_tokens,
            record.output_tokens,
            record.cache_read_tokens,
            record.cache_write_tokens,
            record.duration_ms,
            bytes.fromhex(_writer_hash(record.content_hash)),
            int(record.sort_key * 1000) if record.sort_key is not None else None,
        ),
    )
    updated = bool(res.rowcount > 0)
    row = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? AND position = ? AND variant_index = ?",
        (session_id, position, variant_index),
    ).fetchone()
    if row is None:
        raise RuntimeError("message upsert did not produce a message_id")
    message_id = str(row["message_id"])

    if record.text:
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text)
            VALUES (?, ?, 0, 'text', ?)
            ON CONFLICT(message_id, position) DO UPDATE SET text = excluded.text
            """,
            (message_id, session_id, record.text),
        )

    # Persist message blocks if any.
    for blk in record.blocks:
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type,
                text, tool_name, tool_id, tool_input, semantic_type,
                tool_result_is_error, tool_result_exit_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, position) DO UPDATE SET
                block_type = excluded.block_type,
                text = excluded.text,
                tool_name = excluded.tool_name,
                tool_id = excluded.tool_id,
                tool_input = excluded.tool_input,
                semantic_type = excluded.semantic_type,
                tool_result_is_error = excluded.tool_result_is_error,
                tool_result_exit_code = excluded.tool_result_exit_code
            """,
            (
                message_id,
                session_id,
                blk.block_index,
                blk.type.value,
                blk.text,
                blk.tool_name,
                blk.tool_id,
                blk.tool_input,
                blk.semantic_type.value if blk.semantic_type is not None else None,
                blk.tool_result_is_error,
                blk.tool_result_exit_code,
            ),
        )

    return updated


def upsert_attachment(conn: sqlite3.Connection, record: AttachmentRecord) -> bool:
    """Upsert an attachment record."""
    if record.message_id is None:
        raise ValueError("attachment refs require message_id in the current archive schema")
    session_row = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ? OR native_id = ? ORDER BY session_id LIMIT 2",
        (record.session_id, record.session_id),
    ).fetchall()
    if len(session_row) != 1:
        raise ValueError(f"Cannot write attachment for unknown or ambiguous session {record.session_id!r}")
    session_id = str(session_row[0]["session_id"])
    message_row = conn.execute(
        """
        SELECT message_id FROM messages
        WHERE message_id = ? OR (session_id = ? AND native_id = ?)
        ORDER BY message_id LIMIT 2
        """,
        (record.message_id, session_id, record.message_id),
    ).fetchall()
    if len(message_row) != 1:
        raise ValueError(f"Cannot write attachment for unknown or ambiguous message {record.message_id!r}")
    message_id = str(message_row[0]["message_id"])
    attachment_id = _writer_hash(record.attachment_id)

    # Ensure attachment metadata exists (idempotent, doesn't touch ref_count)
    conn.execute(
        """
        INSERT INTO attachments (
            attachment_id,
            display_name,
            media_type,
            byte_count,
            blob_hash,
            ref_count
        ) VALUES (?, ?, ?, ?, ?, 0)
        ON CONFLICT(attachment_id) DO UPDATE SET
            display_name = COALESCE(excluded.display_name, attachments.display_name),
            media_type = COALESCE(excluded.media_type, attachments.media_type),
            byte_count = MAX(attachments.byte_count, excluded.byte_count),
            blob_hash = excluded.blob_hash
        """,
        (
            attachment_id,
            Path(record.path).name if record.path else None,
            record.mime_type,
            record.size_bytes or 0,
            bytes.fromhex(attachment_id),
        ),
    )

    ref_position_row = conn.execute(
        "SELECT COALESCE(MAX(position) + 1, 0) FROM attachment_refs WHERE message_id = ?",
        (message_id,),
    ).fetchone()
    ref_position = int(ref_position_row[0] or 0)
    existing_ref = conn.execute(
        """
        SELECT ar.ref_id FROM attachment_refs ar
        JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id
        WHERE ar.message_id = ? AND ani.id_kind = 'attachment' AND ani.native_id = ?
        """,
        (message_id, record.attachment_id),
    ).fetchone()
    if existing_ref is not None:
        ref_id = str(existing_ref["ref_id"])
        res = conn.execute("SELECT 0")
    else:
        res = conn.execute(
            """
            INSERT INTO attachment_refs (
                attachment_id,
                session_id,
                message_id,
                position,
                upload_origin,
                source_url
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                attachment_id,
                session_id,
                message_id,
                ref_position,
                record.upload_origin,
                record.path if record.path and record.path.startswith(("http://", "https://")) else None,
            ),
        )
        ref_id = f"{message_id}:attachment:{ref_position}"
        native_rows = [(ref_id, "attachment", str(record.attachment_id))]
        if record.file_native_id:
            native_rows.append((ref_id, "file", record.file_native_id))
        if record.drive_native_id:
            native_rows.append((ref_id, "drive", record.drive_native_id))
        if record.path:
            native_rows.append((ref_id, "source", record.path))
        conn.executemany(
            """
            INSERT OR IGNORE INTO attachment_native_ids (ref_id, id_kind, native_id)
            VALUES (?, ?, ?)
            """,
            native_rows,
        )

    # Only increment if we actually inserted a new ref
    # Use atomic increment to avoid read-modify-write race
    if res.rowcount > 0:
        conn.execute(
            "UPDATE attachments SET ref_count = ref_count + 1 WHERE attachment_id = ?",
            (attachment_id,),
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
        columns = {str(row["name"]) for row in db_conn.execute("PRAGMA table_xinfo(sessions)").fetchall()}
        if {"origin", "native_id", "session_id"}.issubset(columns):
            existing = db_conn.execute(
                "SELECT lower(hex(content_hash)) AS content_hash FROM sessions WHERE origin = ? AND native_id = ?",
                (session.origin.value, session.native_id),
            ).fetchone()
            new_hash = _writer_hash(session.content_hash)
            parsed = _record_to_parsed_session(session, messages, attachments)
            write_parsed_session_to_archive(
                db_conn,
                parsed,
                content_hash=new_hash,
            )
            db_conn.commit()
            if existing is not None and str(existing["content_hash"]) == new_hash:
                written_attachments = sum(1 for attachment in attachments if attachment.message_id is not None)
                return {
                    "sessions": 0,
                    "messages": 0,
                    "attachments": written_attachments,
                    "skipped_sessions": 1,
                    "skipped_messages": len(messages),
                    "skipped_attachments": len(attachments) - written_attachments,
                }
            written_attachments = sum(1 for attachment in attachments if attachment.message_id is not None)
            return {
                "sessions": 1,
                "messages": len(messages),
                "attachments": written_attachments,
                "skipped_sessions": 0,
                "skipped_messages": 0,
                "skipped_attachments": len(attachments) - written_attachments,
            }
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
        # Mirror the production write path's aggregate projection so filters
        # on min_messages/max_messages/min_words/has_tool_use see the same
        # precomputed session values as production ingest.
        _upsert_session_stats_sync(db_conn, session=session, messages=messages)
        # User marks/annotations are keyed by the deterministic public target id
        # (origin:native_id), which is stable across reset+reimport, so there is
        # no identity-repoint pass to mirror (#1114 obsoleted by deterministic IDs).
        # Commit inside lock to ensure atomic transaction boundaries
        db_conn.commit()

    return counts


def _upsert_session_stats_sync(
    conn: sqlite3.Connection,
    *,
    session: SessionRecord,
    messages: list[MessageRecord],
) -> None:
    """Sync mirror of aggregate-column maintenance for test seeding."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import MaterialOrigin

    message_count = len(messages)
    word_count = sum(m.word_count for m in messages)
    tool_use_count = sum(1 for m in messages if m.has_tool_use)
    thinking_count = sum(1 for m in messages if m.has_thinking)
    paste_count = sum(1 for m in messages if m.has_paste)
    user_msg_count = sum(1 for m in messages if m.role == Role.USER)
    authored_user_msg_count = sum(1 for m in messages if m.material_origin == MaterialOrigin.HUMAN_AUTHORED)
    assistant_msg_count = sum(1 for m in messages if m.role == Role.ASSISTANT)
    system_msg_count = sum(1 for m in messages if m.role == Role.SYSTEM)
    tool_msg_count = sum(1 for m in messages if m.role == Role.TOOL)
    user_word_count = sum(m.word_count for m in messages if m.role == Role.USER)
    authored_user_word_count = sum(m.word_count for m in messages if m.material_origin == MaterialOrigin.HUMAN_AUTHORED)
    assistant_word_count = sum(m.word_count for m in messages if m.role == Role.ASSISTANT)
    conn.execute(
        """
        UPDATE sessions
        SET message_count = ?,
            word_count = ?,
            tool_use_count = ?,
            thinking_count = ?,
            paste_count = ?,
            user_message_count = ?,
            authored_user_message_count = ?,
            assistant_message_count = ?,
            system_message_count = ?,
            tool_message_count = ?,
            user_word_count = ?,
            authored_user_word_count = ?,
            assistant_word_count = ?
        WHERE session_id = ?
        """,
        (
            message_count,
            word_count,
            tool_use_count,
            thinking_count,
            paste_count,
            user_msg_count,
            authored_user_msg_count,
            assistant_msg_count,
            system_msg_count,
            tool_msg_count,
            user_word_count,
            authored_user_word_count,
            assistant_word_count,
            session.session_id,
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

    def _provider_message_id(value: object | None) -> str | None:
        if value is None:
            return None
        text = str(value)
        prefix = f"{session.session_id}:"
        return text[len(prefix) :] if text.startswith(prefix) else text

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
                is_error=None if block.tool_result_is_error is None else bool(block.tool_result_is_error),
                exit_code=block.tool_result_exit_code,
            )
            for block in (message.blocks or [])
        ]

    parsed_messages = [
        ParsedMessage(
            provider_message_id=_provider_message_id(message.provider_message_id or message.message_id)
            or str(message.message_id),
            role=message.role if message.role is not None else Role.USER,
            text=message.text,
            blocks=_blocks(message),
            message_type=message.message_type,
            material_origin=message.material_origin,
            parent_message_provider_id=_provider_message_id(message.parent_message_id),
            position=position,
            branch_index=message.branch_index,
            variant_index=message.branch_index,
            occurred_at_ms=(int(message.sort_key * 1000) if message.sort_key is not None else None),
            input_tokens=message.input_tokens,
            output_tokens=message.output_tokens,
            cache_read_tokens=message.cache_read_tokens,
            cache_write_tokens=message.cache_write_tokens,
            duration_ms=message.duration_ms,
            model_name=message.model_name,
        )
        for position, message in enumerate(messages)
    ]

    parsed_attachments = [
        ParsedAttachment(
            provider_attachment_id=str(attachment.attachment_native_id or attachment.attachment_id),
            message_provider_id=_provider_message_id(attachment.message_id),
            name=attachment.display_name,
            mime_type=attachment.mime_type,
            size_bytes=attachment.size_bytes,
            path=attachment.path,
            source_url=attachment.source_url,
            caption=attachment.caption,
        )
        for attachment in attachments
    ]
    working_directories_raw = session.working_directories_json
    parsed_wds = loads(working_directories_raw) if isinstance(working_directories_raw, str) else None
    working_directories = [item for item in parsed_wds if isinstance(item, str)] if isinstance(parsed_wds, list) else []

    return ParsedSession(
        source_name=provider_from_origin(session.origin),
        provider_session_id=session.native_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=parsed_messages,
        attachments=parsed_attachments,
        parent_session_provider_id=(str(session.parent_session_id) if session.parent_session_id is not None else None),
        branch_type=session.branch_type,
        reported_duration_ms=None,
        working_directories=working_directories,
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        provider_project_ref=session.provider_project_ref,
    )


def _writer_hash(value: object) -> str:
    text = str(value)
    try:
        raw = bytes.fromhex(text)
    except ValueError:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text if len(raw) == 32 else hashlib.sha256(text.encode("utf-8")).hexdigest()


async def save_current_archive_records(
    repository: Any,
    *,
    session: SessionRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
) -> dict[str, int]:
    """Seed current archive rows through the parsed-session writer."""

    parsed = _record_to_parsed_session(session, messages, attachments)
    result: dict[str, int] = await repository.save_parsed_session(parsed, _writer_hash(session.content_hash))
    return result


async def save_session_to_archive(
    backend: Any,
    *,
    session: SessionRecord,
    messages: Sequence[MessageRecord] = (),
    attachments: Sequence[AttachmentRecord] = (),
) -> dict[str, int]:
    """Seed a session into a SQLiteBackend through the live archive writer.

    Backend-based twin of :func:`save_current_archive_records`. Wraps the
    backend in a repository so population goes through the one production
    writer (``write_parsed_session_to_archive``) rather than any record-level
    backend write path. Content blocks must be attached to their
    ``MessageRecord.content_blocks`` (no separate block-write step exists).

    ``raw_id`` is not propagated by the parsed-session writer path
    (``ArchiveStore.write_parsed`` never receives it), so a follow-up UPDATE
    keyed on ``(origin, native_id)`` patches the column when the session
    record carries one.
    """
    from polylogue.storage.repository import SessionRepository

    result = await save_current_archive_records(
        SessionRepository(backend=backend),
        session=session,
        messages=list(messages),
        attachments=list(attachments),
    )

    if session.raw_id is not None:
        async with backend.connection() as conn:
            await conn.execute(
                "UPDATE sessions SET raw_id = ? WHERE origin = ? AND native_id = ?",
                (session.raw_id, session.origin.value, session.native_id),
            )
            await conn.commit()

    return result


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
            native_id=f"ext-{session_id}",
            origin=_origin_value("test"),
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
        self.conv = self.conv.model_copy(update={"origin": _origin_value(provider)})
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

    def working_directories(self, paths: list[str]) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"working_directories_json": dumps(paths)})
        return self

    def git_branch(self, branch: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"git_branch": branch})
        return self

    def git_repository_url(self, repository_url: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"git_repository_url": repository_url})
        return self

    def provider_project_ref(self, project_ref: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"provider_project_ref": project_ref})
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

        existing_blocks = _normalize_content_blocks(
            raw_blocks=kwargs.pop("blocks", []),
            message_id=msg_id,
            session_id=str(self.conv.session_id),
        )
        all_blocks = existing_blocks

        block_types = {blk.type for blk in all_blocks}
        role_value = None if role is None else Role.normalize(role)
        word_count = len(text.split()) if text.strip() else 0
        has_tool_use = (
            1 if (block_types & {BlockType.TOOL_USE, BlockType.TOOL_RESULT}) or role_value is Role.TOOL else 0
        )
        has_thinking = 1 if BlockType.THINKING in block_types else 0
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
            "blocks": all_blocks,
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
        display_name: str | None = None,
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
            display_name=display_name,
            attachment_native_id=att_id,
        )
        self.attachments.append(att)
        return self

    def save(self) -> SessionRecord:
        parsed = _record_to_parsed_session(self.conv, self.messages, self.attachments)
        with _WRITE_LOCK, open_connection(self.db_path) as conn:
            write_parsed_session_to_archive(
                conn,
                parsed,
                content_hash=_writer_hash(self.conv.content_hash),
            )
        return self.conv

    def native_session_id(self) -> str:
        """The archive's deterministic session id for the built session."""
        from polylogue.core.identity_law import session_id as archive_session_id

        return archive_session_id(self.conv.origin.value, self.conv.native_id)

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
        "origin": _origin_value(source_name),
        "native_id": _coerce_str(
            kwargs.pop("provider_session_id", session_id),
            session_id,
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
    if "provider_meta" in kwargs:
        require_json_value(kwargs["provider_meta"], context="message provider_meta")
    existing_blocks = _normalize_content_blocks(
        raw_blocks=kwargs.pop("blocks", []),
        message_id=message_id,
        session_id=session_id,
    )
    all_blocks = existing_blocks

    block_types = {blk.type for blk in all_blocks}
    role_value = Role.normalize(role)
    word_count = len(text.split()) if isinstance(text, str) and text.strip() else 0
    has_tool_use = 1 if (block_types & {BlockType.TOOL_USE, BlockType.TOOL_RESULT}) or role_value is Role.TOOL else 0
    has_thinking = 1 if BlockType.THINKING in block_types else 0
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
        "blocks": all_blocks,
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
    payload: RecordPayload = {
        "attachment_id": _attachment_id(attachment_id),
        "session_id": _session_id(session_id),
        "message_id": None if message_id is None else _message_id(message_id),
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "display_name": name,
        "attachment_native_id": attachment_id,
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
                "blocks": msg.get("blocks", []),
            }
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
                    display_name=(
                        _optional_str(raw_attachment.get("name"))
                        or _optional_str(raw_attachment.get("title"))
                        or _optional_str(raw_attachment.get("display_name"))
                    ),
                )

        builder.save()
        return cid
