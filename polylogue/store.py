from __future__ import annotations

import json
from dataclasses import dataclass
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
    profile: Optional[str]


def _json_or_none(value: Optional[dict]) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def upsert_conversation(conn, record: ConversationRecord) -> bool:
    existing = conn.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = ?",
        (record.conversation_id,),
    ).fetchone()
    if existing:
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
    existing = conn.execute(
        "SELECT 1 FROM messages WHERE message_id = ?",
        (record.message_id,),
    ).fetchone()
    if existing:
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
        "SELECT ref_count, path FROM attachments WHERE attachment_id = ?",
        (record.attachment_id,),
    ).fetchone()
    if row:
        new_path = record.path or row["path"]
        conn.execute(
            "UPDATE attachments SET ref_count = ref_count + 1, path = ? WHERE attachment_id = ?",
            (new_path, record.attachment_id),
        )
        return False
    conn.execute(
        """
        INSERT INTO attachments (
            attachment_id,
            conversation_id,
            message_id,
            mime_type,
            size_bytes,
            path,
            ref_count,
            provider_meta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.attachment_id,
            record.conversation_id,
            record.message_id,
            record.mime_type,
            record.size_bytes,
            record.path,
            1,
            _json_or_none(record.provider_meta),
        ),
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
            duration_ms,
            profile
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.run_id,
            record.timestamp,
            _json_or_none(record.plan_snapshot),
            _json_or_none(record.counts),
            _json_or_none(record.drift),
            int(record.indexed) if record.indexed is not None else None,
            record.duration_ms,
            record.profile,
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
