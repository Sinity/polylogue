from __future__ import annotations

import json
from dataclasses import dataclass
import hashlib
from typing import Optional

from .db import open_connection


@dataclass
class ConversationRecord:
    conversation_id: str
    provider_name: str
    provider_conversation_id: str
    title: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    content_hash: str
    provider_meta: Optional[dict]
    version: int = 1


@dataclass
class MessageRecord:
    message_id: str
    conversation_id: str
    provider_message_id: Optional[str]
    role: Optional[str]
    text: Optional[str]
    timestamp: Optional[str]
    content_hash: str
    provider_meta: Optional[dict]
    version: int = 1


@dataclass
class AttachmentRecord:
    attachment_id: str
    conversation_id: str
    message_id: Optional[str]
    mime_type: Optional[str]
    size_bytes: Optional[int]
    path: Optional[str]
    provider_meta: Optional[dict]


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    plan_snapshot: Optional[dict]
    counts: Optional[dict]
    drift: Optional[dict]
    indexed: Optional[bool]
    duration_ms: Optional[int]


def _json_or_none(value: Optional[dict]) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _make_ref_id(attachment_id: str, conversation_id: str, message_id: Optional[str]) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


def upsert_conversation(conn, record: ConversationRecord) -> bool:
    row = conn.execute(
        """
        SELECT title, created_at, updated_at, content_hash, provider_meta
        FROM conversations
        WHERE conversation_id = ?
        """,
        (record.conversation_id,),
    ).fetchone()
    if row:
        updates = {}
        if record.title is not None and record.title != row["title"]:
            updates["title"] = record.title
        if record.created_at is not None and record.created_at != row["created_at"]:
            updates["created_at"] = record.created_at
        if record.updated_at is not None and record.updated_at != row["updated_at"]:
            updates["updated_at"] = record.updated_at
        if record.content_hash != row["content_hash"]:
            updates["content_hash"] = record.content_hash
        new_meta = _json_or_none(record.provider_meta)
        if new_meta is not None and new_meta != row["provider_meta"]:
            updates["provider_meta"] = new_meta
        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"UPDATE conversations SET {assignments} WHERE conversation_id = ?",
                (*updates.values(), record.conversation_id),
            )
        return False
    conn.execute(
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
    return True


def upsert_message(conn, record: MessageRecord) -> bool:
    row = conn.execute(
        """
        SELECT provider_message_id, role, text, timestamp, content_hash, provider_meta
        FROM messages
        WHERE message_id = ?
        """,
        (record.message_id,),
    ).fetchone()
    if row:
        updates = {}
        if record.provider_message_id is not None and record.provider_message_id != row["provider_message_id"]:
            updates["provider_message_id"] = record.provider_message_id
        if record.role is not None and record.role != row["role"]:
            updates["role"] = record.role
        if record.text is not None and record.text != row["text"]:
            updates["text"] = record.text
        if record.timestamp is not None and record.timestamp != row["timestamp"]:
            updates["timestamp"] = record.timestamp
        if record.content_hash != row["content_hash"]:
            updates["content_hash"] = record.content_hash
        new_meta = _json_or_none(record.provider_meta)
        if new_meta is not None and new_meta != row["provider_meta"]:
            updates["provider_meta"] = new_meta
        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"UPDATE messages SET {assignments} WHERE message_id = ?",
                (*updates.values(), record.message_id),
            )
        return False
    conn.execute(
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
    return True


def upsert_attachment(conn, record: AttachmentRecord) -> bool:
    row = conn.execute(
        "SELECT mime_type, size_bytes, path, provider_meta FROM attachments WHERE attachment_id = ?",
        (record.attachment_id,),
    ).fetchone()
    if row:
        updates = {}
        if record.mime_type is not None and record.mime_type != row["mime_type"]:
            updates["mime_type"] = record.mime_type
        if record.size_bytes is not None and record.size_bytes != row["size_bytes"]:
            updates["size_bytes"] = record.size_bytes
        new_path = record.path or row["path"]
        if new_path != row["path"]:
            updates["path"] = new_path
        new_meta = _json_or_none(record.provider_meta)
        if new_meta is not None and new_meta != row["provider_meta"]:
            updates["provider_meta"] = new_meta
        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"UPDATE attachments SET {assignments} WHERE attachment_id = ?",
                (*updates.values(), record.attachment_id),
            )
    else:
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
    existing_ref = conn.execute(
        "SELECT 1 FROM attachment_refs WHERE ref_id = ?",
        (ref_id,),
    ).fetchone()
    if existing_ref:
        return False
    conn.execute(
        """
        INSERT INTO attachment_refs (
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
    conn.execute(
        "UPDATE attachments SET ref_count = ref_count + 1 WHERE attachment_id = ?",
        (record.attachment_id,),
    )
    return True


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
) -> dict:
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    with open_connection(None) as conn:
        if upsert_conversation(conn, conversation):
            counts["conversations"] += 1
        else:
            counts["skipped_conversations"] += 1
        for message in messages:
            if upsert_message(conn, message):
                counts["messages"] += 1
            else:
                counts["skipped_messages"] += 1
        for attachment in attachments:
            if upsert_attachment(conn, attachment):
                counts["attachments"] += 1
            else:
                counts["skipped_attachments"] += 1
        conn.commit()
    return counts


__all__ = [
    "ConversationRecord",
    "MessageRecord",
    "AttachmentRecord",
    "RunRecord",
    "record_run",
    "store_records",
]
