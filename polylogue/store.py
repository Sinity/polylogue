from __future__ import annotations

import hashlib
import sqlite3
import threading

from pydantic import BaseModel, field_validator

from polylogue.core.json import dumps as json_dumps

# Type aliases for semantic clarity (full migration pending)
# from polylogue.types import ConversationId, MessageId, AttachmentId, ContentHash

_WRITE_LOCK = threading.Lock()


class ConversationRecord(BaseModel):
    conversation_id: str  # TODO: migrate to ConversationId
    provider_name: str  # TODO: migrate to Provider
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    content_hash: str  # TODO: migrate to ContentHash
    provider_meta: dict | None = None
    version: int = 1

    @field_validator("conversation_id", "provider_name", "provider_conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class MessageRecord(BaseModel):
    message_id: str  # TODO: migrate to MessageId
    conversation_id: str  # TODO: migrate to ConversationId
    provider_message_id: str | None = None
    role: str | None = None
    text: str | None = None
    timestamp: str | None = None
    content_hash: str  # TODO: migrate to ContentHash
    provider_meta: dict | None = None
    version: int = 1

    @field_validator("message_id", "conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class AttachmentRecord(BaseModel):
    attachment_id: str  # TODO: migrate to AttachmentId
    conversation_id: str  # TODO: migrate to ConversationId
    message_id: str | None = None  # TODO: migrate to MessageId | None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict | None = None

    @field_validator("attachment_id", "conversation_id")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("size_bytes")
    @classmethod
    def non_negative_size(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("size_bytes cannot be negative")
        return v


class RunRecord(BaseModel):
    run_id: str
    timestamp: str
    plan_snapshot: dict | None = None
    counts: dict | None = None
    drift: dict | None = None
    indexed: bool | None = None
    duration_ms: int | None = None


def _json_or_none(value: dict | None) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _make_ref_id(attachment_id: str, conversation_id: str, message_id: str | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


def _prune_attachment_refs(conn, conversation_id: str, keep_ref_ids: set[str]) -> None:
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


def upsert_conversation(conn, record: ConversationRecord) -> bool:
    res = conn.execute(
        """
        INSERT INTO conversations (
            conversation_id,
            provider_name,
            provider_conversation_id,
            title,
            created_at,
            updated_at,
            content_hash,
            provider_meta,
            version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(title, '') != IFNULL(excluded.title, '')
            OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
            OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
        """,
        (
            record.conversation_id,
            record.provider_name,
            record.provider_conversation_id,
            record.title,
            record.created_at,
            record.updated_at,
            record.content_hash,
            _json_or_none(record.provider_meta),
            record.version,
        ),
    )
    return res.rowcount > 0


def upsert_message(conn, record: MessageRecord) -> bool:
    res = conn.execute(
        """
        INSERT INTO messages (
            message_id,
            conversation_id,
            provider_message_id,
            role,
            text,
            timestamp,
            content_hash,
            provider_meta,
            version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            role = excluded.role,
            text = excluded.text,
            timestamp = excluded.timestamp,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(role, '') != IFNULL(excluded.role, '')
            OR IFNULL(text, '') != IFNULL(excluded.text, '')
            OR IFNULL(timestamp, '') != IFNULL(excluded.timestamp, '')
            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
        """,
        (
            record.message_id,
            record.conversation_id,
            record.provider_message_id,
            record.role,
            record.text,
            record.timestamp,
            record.content_hash,
            _json_or_none(record.provider_meta),
            record.version,
        ),
    )
    return res.rowcount > 0


def upsert_attachment(conn, record: AttachmentRecord) -> bool:
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
            mime_type = excluded.mime_type,
            size_bytes = excluded.size_bytes,
            path = COALESCE(excluded.path, path),
            provider_meta = excluded.provider_meta
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
    if res.rowcount > 0:
        conn.execute(
            "UPDATE attachments SET ref_count = ref_count + 1 WHERE attachment_id = ?",
            (record.attachment_id,),
        )
        return True
    return False


def record_run(conn, record: RunRecord) -> None:
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
) -> dict:
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    from .db import connection_context

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


__all__ = [
    "ConversationRecord",
    "MessageRecord",
    "AttachmentRecord",
    "RunRecord",
    "record_run",
    "store_records",
]
