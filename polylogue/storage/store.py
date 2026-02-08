from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from polylogue.lib.json import dumps as json_dumps
from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId

# Valid provider name pattern: starts with letter, contains only letters, numbers, hyphens, underscores
_PROVIDER_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

# Maximum reasonable file size (1TB)
MAX_ATTACHMENT_SIZE = 1024 * 1024 * 1024 * 1024

_WRITE_LOCK = threading.Lock()


class ConversationRecord(BaseModel):
    conversation_id: ConversationId
    provider_name: str
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    content_hash: ContentHash
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] | None = None
    version: int = 1
    # Branching support: links conversations in session trees
    parent_conversation_id: ConversationId | None = None
    branch_type: str | None = None  # "continuation", "sidechain", "fork"
    # Link to raw source data (FK to raw_conversations.raw_id)
    raw_id: str | None = None

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("provider_name cannot be empty")
        v = v.strip()
        if not _PROVIDER_NAME_PATTERN.match(v):
            raise ValueError(
                f"provider_name '{v}' is invalid. Must start with a letter and "
                "contain only letters, numbers, hyphens, and underscores."
            )
        return v

    @field_validator("conversation_id", "provider_conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class MessageRecord(BaseModel):
    message_id: MessageId
    conversation_id: ConversationId
    provider_message_id: str | None = None
    role: str | None = None
    text: str | None = None
    timestamp: str | None = None
    content_hash: ContentHash
    provider_meta: dict[str, object] | None = None
    version: int = 1
    # Branching support: links messages in conversation trees
    parent_message_id: MessageId | None = None
    branch_index: int = 0  # 0 = mainline, >0 = branch sibling position

    @field_validator("message_id", "conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class AttachmentRecord(BaseModel):
    attachment_id: AttachmentId
    conversation_id: ConversationId
    message_id: MessageId | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict[str, object] | None = None

    @field_validator("attachment_id", "conversation_id")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("path")
    @classmethod
    def sanitize_path(cls, v: str | None) -> str | None:
        """Sanitize path to prevent traversal attacks and other security issues."""
        if v is None:
            return v

        original_v = v

        # Remove null bytes
        v = v.replace("\x00", "")

        # Remove control characters (ASCII < 32 and 127)
        v = "".join(c for c in v if ord(c) >= 32 and ord(c) != 127)

        # Detect threats:
        # 1. Traversal attempts (..)
        # 2. Symlinks in path (potential traversal bypass)
        has_traversal = ".." in original_v

        # Check for symlinks in the path by checking path components
        has_symlink = False
        try:
            p = Path(v)
            # Check each parent in the path to see if it's a symlink
            # This prevents traversal via symlinks
            for parent in [p] + list(p.parents):
                if parent.is_symlink():
                    has_symlink = True
                    break
        except Exception:
            # If we can't check, assume it's safe
            pass

        # If traversal or symlinks were detected, hash to prevent re-assembly
        if has_traversal or has_symlink:
            import hashlib
            # Hash the original to prevent reconstruction
            original_hash = hashlib.sha256(original_v.encode()).hexdigest()[:12]
            v = f"_blocked_{original_hash}"
        # For safe paths, just clean up but don't strip leading /
        # (preserve absolute vs relative structure)
        else:
            try:
                parts = []
                for component in v.split("/"):
                    component = component.strip()
                    # Skip empty or special dot components
                    if component and component not in (".", ".."):
                        parts.append(component)
                # Rebuild path, preserving leading / if it was there
                if original_v.startswith("/"):
                    v = "/" + "/".join(parts) if parts else "/"
                else:
                    v = "/".join(parts) if parts else v
            except Exception:
                pass

        return v if v else None

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 0:
            raise ValueError("size_bytes cannot be negative")
        if v > MAX_ATTACHMENT_SIZE:
            raise ValueError(f"size_bytes exceeds maximum ({MAX_ATTACHMENT_SIZE} bytes / 1TB)")
        return v


class RunRecord(BaseModel):
    run_id: str
    timestamp: str
    plan_snapshot: dict[str, Any] | None = None
    counts: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None
    indexed: bool | None = None
    duration_ms: int | None = None


class RawConversationRecord(BaseModel):
    """Record storing original raw JSON/JSONL bytes before parsing.

    This enables honest, database-driven testing by preserving the exact
    input data that was parsed into conversations and messages.

    Note: The link to parsed conversations goes the OTHER way:
    conversations.raw_id → raw_conversations.raw_id
    This matches the data flow: acquire raw first, then parse.
    """
    raw_id: str  # SHA256 of raw_content
    provider_name: str
    source_name: str | None = None  # Config source name (e.g., "inbox"), distinct from provider
    source_path: str
    source_index: int | None = None  # Position in bundle (e.g., conversations[3])
    raw_content: bytes  # Full JSON/JSONL bytes
    acquired_at: str  # ISO timestamp of acquisition
    file_mtime: str | None = None  # File modification time if available

    @field_validator("raw_id", "provider_name", "source_path")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("raw_content")
    @classmethod
    def non_empty_bytes(cls, v: bytes) -> bytes:
        if not v:
            raise ValueError("raw_content cannot be empty")
        return v


class PlanResult(BaseModel):
    timestamp: int
    counts: dict[str, int]
    sources: list[str]
    cursors: dict[str, dict[str, Any]]


class RunResult(BaseModel):
    run_id: str
    counts: dict[str, int]
    drift: dict[str, dict[str, int]]
    indexed: bool
    index_error: str | None
    duration_ms: int
    render_failures: list[dict[str, str]] = []


class ExistingConversation(BaseModel):
    conversation_id: str
    content_hash: str


def _json_or_none(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _make_ref_id(attachment_id: AttachmentId, conversation_id: ConversationId, message_id: MessageId | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


def _prune_attachment_refs(conn: sqlite3.Connection, conversation_id: ConversationId, keep_ref_ids: set[str]) -> None:
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
            version,
            parent_conversation_id,
            branch_type,
            raw_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta,
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
            record.parent_conversation_id,
            record.branch_type,
            record.raw_id,
        ),
    )
    return bool(res.rowcount > 0)


def upsert_message(conn: sqlite3.Connection, record: MessageRecord) -> bool:
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
            version,
            parent_message_id,
            branch_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            role = excluded.role,
            text = excluded.text,
            timestamp = excluded.timestamp,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta,
            parent_message_id = excluded.parent_message_id,
            branch_index = excluded.branch_index
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(role, '') != IFNULL(excluded.role, '')
            OR IFNULL(text, '') != IFNULL(excluded.text, '')
            OR IFNULL(timestamp, '') != IFNULL(excluded.timestamp, '')
            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
            OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
            OR branch_index != excluded.branch_index
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
            record.parent_message_id,
            record.branch_index,
        ),
    )
    return bool(res.rowcount > 0)


def upsert_attachment(conn: sqlite3.Connection, record: AttachmentRecord) -> bool:
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
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    from .backends.sqlite import connection_context

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
    "RawConversationRecord",
    "MAX_ATTACHMENT_SIZE",
    "record_run",
    "store_records",
]
