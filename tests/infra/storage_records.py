"""Shared storage-record builders and DB seeding helpers for tests."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from polylogue.storage.backends.connection import connection_context, open_connection
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
    _json_or_none,
    _make_ref_id,
)

# Thread-safety lock for writes (matches store.py pattern)
_WRITE_LOCK = threading.Lock()

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
            version,
            parent_conversation_id,
            branch_type,
            raw_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            sort_key = excluded.sort_key,
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
            record.sort_key,
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
                blk.block_id, blk.message_id, blk.conversation_id, blk.block_index,
                blk.type, blk.text, blk.tool_name, blk.tool_id, blk.tool_input,
                blk.media_type, blk.metadata, blk.semantic_type,
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


def db_setup(workspace_env) -> Path:
    """Initialize database path in workspace environment.

    Usage in tests:
        db_path = db_setup(workspace_env)
        builder = ConversationBuilder(db_path, "test-conv")
    """
    db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# =============================================================================
# MESSAGE/CONVERSATION BUILDERS (Fluent API)
# =============================================================================


class ConversationBuilder:
    """Fluent builder for creating conversations in test database.

    Example:
        (ConversationBuilder(db_path, "test-conv")
         .title("My Test")
         .provider("chatgpt")
         .add_message(msg1)
         .add_message(msg2)
         .add_attachment(att1)
         .save())

    Simplifies:
        - Creating ConversationRecord
        - Adding messages/attachments
        - Calling store_records with proper open_connection
    """

    def __init__(self, db_path: Path, conversation_id: str):
        self.db_path = db_path
        now = datetime.now(timezone.utc).isoformat()

        from polylogue.pipeline.prepare import _timestamp_sort_key

        self.conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test Conversation",
            created_at=now,
            updated_at=now,
            sort_key=_timestamp_sort_key(now),
            content_hash=uuid4().hex,
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
        from polylogue.pipeline.prepare import _timestamp_sort_key

        self.conv = self.conv.model_copy(update={
            "updated_at": updated_at,
            "sort_key": _timestamp_sort_key(updated_at),
        })
        return self

    def metadata(self, metadata: dict[str, Any] | None) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"metadata": metadata})
        return self

    def parent_conversation(self, parent_id: str) -> ConversationBuilder:
        """Set parent conversation for continuations/sidechains."""
        self.conv = self.conv.model_copy(update={"parent_conversation_id": parent_id})
        return self

    def branch_type(self, branch_type: str) -> ConversationBuilder:
        """Set branch type: 'continuation', 'sidechain', or 'fork'."""
        self.conv = self.conv.model_copy(update={"branch_type": branch_type})
        return self

    def add_message(
        self,
        message_id: str | None = None,
        role: str = "user",
        text: str = "Test message",
        timestamp: str | None | object = ...,  # ... = auto-generate, None = no timestamp
        **kwargs,
    ) -> ConversationBuilder:
        """Add a message to the conversation.

        Can pass MessageRecord directly or use kwargs to build one.

        Usage:
            .add_message("m1", role="user", text="Hello!")
            .add_message("m2", timestamp=None)  # Explicitly no timestamp
            .add_message(message_record)
        """
        msg_id = f"m{len(self.messages) + 1}" if message_id is None else message_id

        # Handle timestamp: ... = auto-generate, None = keep None, str = use value
        ts = datetime.now(timezone.utc).isoformat() if timestamp is ... else timestamp

        from polylogue.pipeline.prepare import _timestamp_sort_key

        # Extract content_blocks from provider_meta if provided (legacy test format)
        provider_meta = kwargs.pop("provider_meta", None)
        raw_blocks = (provider_meta or {}).get("content_blocks") or []
        extra_blocks: list[ContentBlockRecord] = []
        for idx, blk in enumerate(raw_blocks):
            extra_blocks.append(ContentBlockRecord(
                block_id=f"blk-{msg_id}-{idx}",
                message_id=msg_id,
                conversation_id=self.conv.conversation_id,
                block_index=idx,
                type=blk.get("type", "text"),
                text=blk.get("text"),
                tool_name=blk.get("tool_name"),
                tool_id=blk.get("tool_id"),
                tool_input=(
                    blk["input"] if isinstance(blk.get("input"), str)
                    else __import__("json").dumps(blk["input"]) if blk.get("input") is not None
                    else None
                ),
                semantic_type=blk.get("semantic_type"),
            ))

        # Merge with any content_blocks already in kwargs
        existing_blocks = kwargs.pop("content_blocks", [])
        all_blocks = extra_blocks + list(existing_blocks)

        # Compute analytics fields from content_blocks (same logic as prepare.py)
        _block_types = {blk.type for blk in all_blocks}
        _word_count = len(text.split()) if text and text.strip() else 0
        _has_tool_use = 1 if (_block_types & {"tool_use", "tool_result"}) or role == "tool" else 0
        _has_thinking = 1 if "thinking" in _block_types else 0

        msg = MessageRecord(
            message_id=msg_id,
            conversation_id=self.conv.conversation_id,
            role=role,
            text=text,
            sort_key=_timestamp_sort_key(ts) if ts is not None else None,
            content_hash=uuid4().hex[:16],
            content_blocks=all_blocks,
            word_count=kwargs.pop("word_count", _word_count),
            has_tool_use=kwargs.pop("has_tool_use", _has_tool_use),
            has_thinking=kwargs.pop("has_thinking", _has_thinking),
            **kwargs,
        )
        self.messages.append(msg)
        return self

    def add_attachment(
        self,
        attachment_id: str | None = None,
        message_id: str | None | object = ...,  # ... = auto-assign to last message, None = orphaned
        mime_type: str = "application/octet-stream",
        size_bytes: int = 1024,
        path: str | None = None,
        provider_meta: dict | None = None,
    ) -> ConversationBuilder:
        """Add an attachment to the conversation.

        Args:
            message_id: ... (default) = attach to last message, None = orphaned attachment
        """
        att_id = f"att{len(self.attachments) + 1}" if attachment_id is None else attachment_id

        # Handle message_id: ... = auto-assign to last message, None = orphaned
        msg_id = (self.messages[-1].message_id if self.messages else None) if message_id is ... else message_id

        att = AttachmentRecord(
            attachment_id=att_id,
            conversation_id=self.conv.conversation_id,
            message_id=msg_id,
            mime_type=mime_type,
            size_bytes=size_bytes,
            path=path,
            provider_meta=provider_meta,
        )
        self.attachments.append(att)
        return self

    def save(self) -> ConversationRecord:
        """Save conversation, messages, and attachments to database."""
        with open_connection(self.db_path) as conn:
            store_records(
                conversation=self.conv,
                messages=self.messages,
                attachments=self.attachments,
                conn=conn,
            )
        return self.conv

    async def build(self):
        """Save to the DB and return the hydrated domain conversation."""
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
    **kwargs,
) -> ConversationRecord:
    """Quick conversation record creation without storing.

    Usage:
        conv = make_conversation("conv1", provider_name="claude", title="My Conv")
    """
    now = datetime.now(timezone.utc).isoformat()
    return ConversationRecord(
        conversation_id=conversation_id,
        provider_name=provider_name,
        provider_conversation_id=kwargs.pop("provider_conversation_id", f"ext-{conversation_id}"),
        title=title,
        created_at=created_at or now,
        updated_at=updated_at or now,
        content_hash=kwargs.pop("content_hash", uuid4().hex),
        **kwargs,
    )


def make_message(
    message_id: str = "m1",
    conversation_id: str = "conv1",
    role: str = "user",
    text: str = "Test message",
    timestamp: str | None = None,
    **kwargs,
) -> MessageRecord:
    """Quick message creation without builder.

    Usage:
        msg = make_message("m1", role="assistant", text="Reply")
        msg = make_message("m1", content_hash="explicit-hash")  # Override hash
        msg = make_message("m1", provider_meta={"content_blocks": [{"type": "thinking"}]})
    """
    import json as _json

    from polylogue.pipeline.prepare import _timestamp_sort_key

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    # Extract content_blocks from provider_meta if provided (legacy test format)
    provider_meta = kwargs.pop("provider_meta", None)
    raw_blocks = (provider_meta or {}).get("content_blocks") or []
    extra_blocks: list[ContentBlockRecord] = []
    for idx, blk in enumerate(raw_blocks or []):
        if not isinstance(blk, dict):
            continue
        tool_input = blk.get("input")
        extra_blocks.append(ContentBlockRecord(
            block_id=f"blk-{message_id}-{idx}",
            message_id=message_id,
            conversation_id=conversation_id,
            block_index=idx,
            type=blk.get("type", "text"),
            text=blk.get("text"),
            tool_name=blk.get("tool_name") or blk.get("name"),
            tool_id=blk.get("tool_id") or blk.get("id"),
            tool_input=(
                tool_input if isinstance(tool_input, str)
                else _json.dumps(tool_input) if tool_input is not None
                else None
            ),
        ))

    existing_blocks = kwargs.pop("content_blocks", [])
    all_blocks = extra_blocks + list(existing_blocks)

    # Compute analytics fields from content_blocks (same logic as prepare.py)
    _block_types = {blk.type for blk in all_blocks}
    _word_count = len(text.split()) if text and text.strip() else 0
    _has_tool_use = 1 if (_block_types & {"tool_use", "tool_result"}) or role == "tool" else 0
    _has_thinking = 1 if "thinking" in _block_types else 0

    return MessageRecord(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        text=text,
        sort_key=kwargs.pop("sort_key", _timestamp_sort_key(ts)),
        content_hash=kwargs.pop("content_hash", uuid4().hex[:16]),
        content_blocks=all_blocks,
        word_count=kwargs.pop("word_count", _word_count),
        has_tool_use=kwargs.pop("has_tool_use", _has_tool_use),
        has_thinking=kwargs.pop("has_thinking", _has_thinking),
        **kwargs,
    )


def make_attachment(
    attachment_id: str = "att1",
    conversation_id: str = "conv1",
    message_id: str | None = None,
    mime_type: str = "application/octet-stream",
    size_bytes: int = 1024,
    name: str | None = None,
    **kwargs,
) -> AttachmentRecord:
    """Quick attachment creation.

    Usage:
        att = make_attachment("att1", name="file.pdf")
    """
    provider_meta = kwargs.pop("provider_meta", None)
    if name and provider_meta is None:
        provider_meta = {"name": name}

    return AttachmentRecord(
        attachment_id=attachment_id,
        conversation_id=conversation_id,
        message_id=message_id,
        mime_type=mime_type,
        size_bytes=size_bytes,
        provider_meta=provider_meta,
        **kwargs,
    )


class DbFactory:
    """Low-ceremony DB seeder built on top of ConversationBuilder."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def create_conversation(
        self,
        id: str | None = None,
        provider: str = "test",
        title: str = "Test Conversation",
        messages: list[dict[str, Any]] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a conversation with simple dict-shaped messages and attachments."""
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
            message_id = msg.get("id")
            message_kwargs: dict[str, Any] = {
                "provider_message_id": msg.get("provider_message_id"),
                "parent_message_id": msg.get("parent_message_id"),
                "branch_index": msg.get("branch_index", 0),
                "content_blocks": msg.get("content_blocks", []),
            }
            if "provider_meta" in msg:
                message_kwargs["provider_meta"] = msg["provider_meta"]
            if "word_count" in msg:
                message_kwargs["word_count"] = msg["word_count"]
            if "has_tool_use" in msg:
                message_kwargs["has_tool_use"] = msg["has_tool_use"]
            if "has_thinking" in msg:
                message_kwargs["has_thinking"] = msg["has_thinking"]

            builder.add_message(
                message_id=message_id,
                role=msg.get("role", "user"),
                text=msg.get("text", msg.get("content", "hello")),
                timestamp=msg.get("timestamp", ...),
                **message_kwargs,
            )

            for att in msg.get("attachments", []):
                builder.add_attachment(
                    attachment_id=att.get("id"),
                    message_id=message_id if message_id is not None else ...,
                    mime_type=att.get("mime_type", "application/octet-stream"),
                    size_bytes=att.get("size_bytes", 1024),
                    path=att.get("path"),
                    provider_meta=att.get("meta") or att.get("provider_meta"),
                )

        builder.save()
        return cid
