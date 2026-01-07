from __future__ import annotations

import hashlib
import json
import sqlite3

from pydantic import BaseModel


class ConversationRecord(BaseModel):
    conversation_id: str
    provider_name: str
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    content_hash: str
    provider_meta: dict | None = None
    version: int = 1


class MessageRecord(BaseModel):
    message_id: str
    conversation_id: str
    provider_message_id: str | None = None
    role: str | None = None
    text: str | None = None
    timestamp: str | None = None
    content_hash: str
    provider_meta: dict | None = None
    version: int = 1


class AttachmentRecord(BaseModel):
    attachment_id: str
    conversation_id: str
    message_id: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict | None = None


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
    return json.dumps(value, sort_keys=True)


def _make_ref_id(attachment_id: str, conversation_id: str, message_id: str | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


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
        WHERE content_hash != excluded.content_hash OR title != excluded.title
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
        WHERE content_hash != excluded.content_hash
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

    with connection_context(conn) as db_conn:
        if upsert_conversation(db_conn, conversation):
            counts["conversations"] += 1
        else:
            counts["skipped_conversations"] += 1
        for message in messages:
            if upsert_message(db_conn, message):
                counts["messages"] += 1
            else:
                counts["skipped_messages"] += 1
        for attachment in attachments:
            if upsert_attachment(db_conn, attachment):
                counts["attachments"] += 1
            else:
                counts["skipped_attachments"] += 1

    return counts


__all__ = [
    "ConversationRecord",
    "MessageRecord",
    "AttachmentRecord",
    "RunRecord",
    "record_run",
    "store_records",
]
