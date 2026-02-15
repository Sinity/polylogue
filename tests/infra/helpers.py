"""Test utilities and builders for Polylogue test suite.

These helpers reduce boilerplate in parametrized tests and provide consistent
test data generation across the test suite.

Usage:
    from tests.infra.helpers import ConversationBuilder, make_message, db_setup

Created during aggressive test consolidation to eliminate repeated patterns.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from polylogue.storage.backends.sqlite import connection_context, open_connection
from polylogue.storage.store import (
    AttachmentRecord,
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
    """Upsert a message record."""
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

        self.conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test Conversation",
            created_at=now,
            updated_at=now,
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
        self.conv = self.conv.model_copy(update={"updated_at": updated_at})
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

        msg = MessageRecord(
            message_id=msg_id,
            conversation_id=self.conv.conversation_id,
            role=role,
            text=text,
            timestamp=ts,
            content_hash=uuid4().hex[:16],
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

    def build(self):
        """Save to database and return a full Conversation domain object."""
        from polylogue import Polylogue

        self.save()
        p = Polylogue(db_path=self.db_path)
        return p.repository.get(self.conv.conversation_id)


# =============================================================================
# QUICK BUILDERS (For simple cases)
# =============================================================================


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
    """
    return MessageRecord(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        text=text,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        content_hash=kwargs.pop("content_hash", uuid4().hex[:16]),
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


# =============================================================================
# PARAMETRIZED TEST HELPERS
# =============================================================================


def assert_messages_ordered(markdown_text: str, *expected_order: str):
    """Assert messages appear in given order in markdown output.

    Usage:
        assert_messages_ordered(result.markdown_text, "First", "Second", "Third")
    """
    indices = []
    for text in expected_order:
        try:
            idx = markdown_text.index(text)
            indices.append((idx, text))
        except ValueError as exc:
            raise AssertionError(f"Expected text '{text}' not found in markdown") from exc

    # Verify order
    for i in range(len(indices) - 1):
        if indices[i][0] >= indices[i + 1][0]:
            raise AssertionError(
                f"Order violation: '{indices[i][1]}' (index {indices[i][0]}) "
                f"should come before '{indices[i + 1][1]}' (index {indices[i + 1][0]})"
            )


def assert_contains_all(text: str, *expected: str):
    """Assert text contains all expected substrings.

    Usage:
        assert_contains_all(result, "user", "assistant", "Hello")
    """
    for expected_text in expected:
        assert expected_text in text, f"Expected '{expected_text}' not found in text"


def assert_not_contains_any(text: str, *unexpected: str):
    """Assert text does NOT contain any of the given substrings.

    Usage:
        assert_not_contains_any(result, "ERROR", "FAIL", "```json")
    """
    for unexpected_text in unexpected:
        assert unexpected_text not in text, f"Unexpected '{unexpected_text}' found in text"


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


def make_chatgpt_node(
    msg_id: str,
    role: str,
    content_parts: list[str],
    children: list[str] | None = None,
    timestamp: float | None = None,
    metadata: dict | None = None,
    parent: str | None = None,
) -> dict[str, Any]:
    """Generate ChatGPT mapping node for parser tests.

    Usage:
        node = make_chatgpt_node("msg1", "user", ["Hello"], timestamp=1704067200)
        node = make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1")
    """
    node = {
        "id": msg_id,
        "message": {
            "id": msg_id,
            "author": {"role": role},
            "content": {"parts": content_parts},
        },
    }
    if children:
        node["children"] = children
    if parent:
        node["parent"] = parent
    if timestamp:
        node["message"]["create_time"] = timestamp
    if metadata:
        node["message"]["metadata"] = metadata
    return node


# =============================================================================
# LEGACY FACTORY (dict-based API, simpler for quick seeding)
# =============================================================================


class DbFactory:
    """Helper to seed the database with consistent records via dict-based API.

    For new tests, prefer ConversationBuilder (fluent API).
    """

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
        """Create a conversation with messages in the DB."""
        cid = id or str(uuid4())
        created_iso = (created_at or datetime.now(timezone.utc)).isoformat()
        updated_iso = (updated_at or datetime.now(timezone.utc)).isoformat()

        conv_rec = ConversationRecord(
            conversation_id=cid,
            provider_name=provider,
            provider_conversation_id=f"ext-{cid}",
            title=title,
            created_at=created_iso,
            updated_at=updated_iso,
            content_hash=uuid4().hex,
            metadata=metadata,
        )

        msg_recs = []
        att_recs = []

        if messages:
            for msg in messages:
                mid = msg.get("id") or str(uuid4())
                m_rec = MessageRecord(
                    message_id=mid,
                    conversation_id=cid,
                    role=msg.get("role", "user"),
                    text=msg.get("text", "hello"),
                    timestamp=msg.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    content_hash=uuid4().hex,
                )
                msg_recs.append(m_rec)

                if "attachments" in msg:
                    for att in msg["attachments"]:
                        aid = att.get("id") or str(uuid4())
                        att_recs.append(
                            AttachmentRecord(
                                attachment_id=aid,
                                conversation_id=cid,
                                message_id=mid,
                                mime_type=att.get("mime_type", "application/octet-stream"),
                                size_bytes=att.get("size_bytes", 1024),
                                provider_meta=att.get("meta"),
                            )
                        )

        with open_connection(self.db_path) as conn:
            store_records(
                conversation=conv_rec,
                messages=msg_recs,
                attachments=att_recs,
                conn=conn,
            )
        return cid


def make_claude_chat_message(
    uuid: str,
    sender: str,
    text: str,
    attachments: list[dict] | None = None,
    files: list[dict] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Generate Claude AI chat_messages entry for parser tests.

    Usage:
        msg = make_claude_chat_message("u1", "human", "Hello")
    """
    msg = {
        "uuid": uuid,
        "text": text,
    }

    if sender:
        msg["sender"] = sender
    if attachments:
        msg["attachments"] = attachments
    if files:
        msg["files"] = files
    if timestamp:
        msg["created_at"] = timestamp

    return msg


# =============================================================================
# FILE DATA BUILDERS (For creating test inbox files)
# =============================================================================


class InboxBuilder:
    """Fluent builder for creating test inbox directories with files.

    Example:
        inbox = (InboxBuilder(tmp_path)
                 .add_codex_conversation("conv1", messages=[("user", "Hello")])
                 .add_chatgpt_export("conv2", nodes=[...])
                 .add_claude_export("conv3", chat_messages=[...])
                 .build())
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.files: list[tuple[Path, str]] = []

    def add_json_file(self, filename: str, data: Any) -> InboxBuilder:
        """Add a raw JSON file."""
        import json
        path = self.base_path / filename
        self.files.append((path, json.dumps(data, indent=2)))
        return self

    def add_jsonl_file(self, filename: str, entries: list[Any]) -> InboxBuilder:
        """Add a JSONL file with multiple entries."""
        import json
        path = self.base_path / filename
        content = "\n".join(json.dumps(entry) for entry in entries) + "\n"
        self.files.append((path, content))
        return self

    def add_codex_conversation(
        self,
        conv_id: str,
        title: str | None = None,
        messages: list[tuple[str, str]] | None = None,
        filename: str | None = None,
    ) -> InboxBuilder:
        """Add a simple Codex/generic format conversation.

        Args:
            conv_id: Conversation ID
            title: Optional title
            messages: List of (role, content) tuples
            filename: Custom filename (default: {conv_id}.json)
        """
        if messages is None:
            messages = [("user", "Hello"), ("assistant", "Hi there!")]

        payload: dict[str, Any] = {
            "id": conv_id,
            "messages": [
                {"id": f"m{i+1}", "role": role, "content": content}
                for i, (role, content) in enumerate(messages)
            ],
        }
        if title:
            payload["title"] = title

        fname = filename or f"{conv_id}.json"
        return self.add_json_file(fname, payload)

    def add_chatgpt_export(
        self,
        conv_id: str,
        title: str | None = None,
        nodes: list[dict] | None = None,
        filename: str | None = None,
    ) -> InboxBuilder:
        """Add a ChatGPT export format conversation.

        Args:
            conv_id: Conversation ID
            title: Optional title
            nodes: List of mapping node dicts (use make_chatgpt_node())
            filename: Custom filename
        """
        if nodes is None:
            nodes = [
                make_chatgpt_node("n1", "user", ["Hello"], timestamp=1704067200),
                make_chatgpt_node("n2", "assistant", ["Hi there!"], timestamp=1704067201),
            ]

        # Build mapping from nodes
        mapping = {}
        for node in nodes:
            mapping[node["id"]] = node

        payload: dict[str, Any] = {
            "id": conv_id,
            "mapping": mapping,
        }
        if title:
            payload["title"] = title

        fname = filename or f"chatgpt_{conv_id}.json"
        return self.add_json_file(fname, payload)

    def add_claude_export(
        self,
        conv_id: str,
        name: str | None = None,
        chat_messages: list[dict] | None = None,
        filename: str | None = None,
        wrap_in_conversations: bool = True,
    ) -> InboxBuilder:
        """Add a Claude AI export format conversation.

        Args:
            conv_id: Conversation ID
            name: Conversation name/title
            chat_messages: List of message dicts (use make_claude_chat_message())
            filename: Custom filename
            wrap_in_conversations: Wrap in {"conversations": [...]} structure
        """
        if chat_messages is None:
            chat_messages = [
                make_claude_chat_message("m1", "human", "Hello"),
                make_claude_chat_message("m2", "assistant", "Hi there!"),
            ]

        conversation = {
            "id": conv_id,
            "chat_messages": chat_messages,
        }
        if name:
            conversation["name"] = name

        payload = {"conversations": [conversation]} if wrap_in_conversations else conversation

        fname = filename or f"claude_{conv_id}.json"
        return self.add_json_file(fname, payload)

    def build(self) -> Path:
        """Write all files and return the inbox path."""
        for path, content in self.files:
            path.write_text(content, encoding="utf-8")
        return self.base_path

    def get_file_path(self, filename: str) -> Path:
        """Get path to a specific file in the inbox."""
        return self.base_path / filename


class ChatGPTExportBuilder:
    """Builder for ChatGPT export format with proper node structure.

    Example:
        export = (ChatGPTExportBuilder("conv1")
                  .title("My Chat")
                  .add_node("user", "Hello")
                  .add_node("assistant", "Hi there!")
                  .add_node("user", "How are you?", model_slug="gpt-4")
                  .build())
    """

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._nodes: list[dict] = []
        self._node_counter = 0
        self._timestamp = 1704067200.0  # Base timestamp

    def title(self, title: str) -> ChatGPTExportBuilder:
        self._title = title
        return self

    def add_node(
        self,
        role: str,
        *content_parts: str,
        node_id: str | None = None,
        metadata: dict | None = None,
        model_slug: str | None = None,
    ) -> ChatGPTExportBuilder:
        """Add a message node."""
        self._node_counter += 1
        nid = node_id or f"node-{self._node_counter}"

        meta = metadata or {}
        if model_slug:
            meta["model_slug"] = model_slug

        node = make_chatgpt_node(
            nid,
            role,
            list(content_parts),
            timestamp=self._timestamp,
            metadata=meta if meta else None,
        )
        self._nodes.append(node)
        self._timestamp += 1.0
        return self

    def add_system_node(self, content: str, node_id: str | None = None) -> ChatGPTExportBuilder:
        """Add a system message node."""
        return self.add_node("system", content, node_id=node_id)

    def add_tool_node(
        self,
        tool_name: str,
        result: str,
        node_id: str | None = None,
    ) -> ChatGPTExportBuilder:
        """Add a tool result node."""
        return self.add_node(
            "tool",
            result,
            node_id=node_id,
            metadata={"name": tool_name},
        )

    def build(self) -> dict[str, Any]:
        """Build the ChatGPT export structure."""
        mapping = {node["id"]: node for node in self._nodes}
        result: dict[str, Any] = {
            "id": self.conv_id,
            "mapping": mapping,
        }
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        """Build and write to file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class ClaudeExportBuilder:
    """Builder for Claude AI export format.

    Example:
        export = (ClaudeExportBuilder("conv1")
                  .name("My Claude Chat")
                  .add_message("human", "Hello")
                  .add_message("assistant", "Hi!", attachments=[...])
                  .build())
    """

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._name: str | None = None
        self._messages: list[dict] = []
        self._msg_counter = 0
        self._wrap_in_conversations = True

    def name(self, name: str) -> ClaudeExportBuilder:
        self._name = name
        return self

    def unwrapped(self) -> ClaudeExportBuilder:
        """Don't wrap in {"conversations": [...]} structure."""
        self._wrap_in_conversations = False
        return self

    def add_message(
        self,
        sender: str,
        text: str,
        uuid: str | None = None,
        attachments: list[dict] | None = None,
        files: list[dict] | None = None,
        timestamp: str | None = None,
    ) -> ClaudeExportBuilder:
        """Add a chat message."""
        self._msg_counter += 1
        msg_uuid = uuid or f"msg-{self._msg_counter}"

        msg = make_claude_chat_message(
            msg_uuid,
            sender,
            text,
            attachments=attachments,
            files=files,
            timestamp=timestamp,
        )
        self._messages.append(msg)
        return self

    def add_human(self, text: str, **kwargs) -> ClaudeExportBuilder:
        """Shorthand for add_message with sender='human'."""
        return self.add_message("human", text, **kwargs)

    def add_assistant(self, text: str, **kwargs) -> ClaudeExportBuilder:
        """Shorthand for add_message with sender='assistant'."""
        return self.add_message("assistant", text, **kwargs)

    def build(self) -> dict[str, Any]:
        """Build the Claude export structure."""
        conversation = {
            "id": self.conv_id,
            "chat_messages": self._messages,
        }
        if self._name:
            conversation["name"] = self._name

        if self._wrap_in_conversations:
            return {"conversations": [conversation]}
        return conversation

    def write_to(self, path: Path) -> Path:
        """Build and write to file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class GenericConversationBuilder:
    """Builder for simple/Codex format conversations.

    Example:
        conv = (GenericConversationBuilder("conv1")
                .title("Test Chat")
                .add_message("user", "Hello")
                .add_message("assistant", "Hi!")
                .build())
    """

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._messages: list[dict] = []
        self._msg_counter = 0

    def title(self, title: str) -> GenericConversationBuilder:
        self._title = title
        return self

    def add_message(
        self,
        role: str,
        content: str,
        message_id: str | None = None,
        text: str | None = None,  # Alias for content
    ) -> GenericConversationBuilder:
        """Add a message. Uses 'content' key by default, but can use 'text'."""
        self._msg_counter += 1
        msg_id = message_id or f"m{self._msg_counter}"

        msg: dict[str, Any] = {
            "id": msg_id,
            "role": role,
        }
        # Support both content and text keys
        if text is not None:
            msg["text"] = text
        else:
            msg["content"] = content

        self._messages.append(msg)
        return self

    def add_user(self, content: str, **kwargs) -> GenericConversationBuilder:
        """Shorthand for add_message with role='user'."""
        return self.add_message("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs) -> GenericConversationBuilder:
        """Shorthand for add_message with role='assistant'."""
        return self.add_message("assistant", content, **kwargs)

    def build(self) -> dict[str, Any]:
        """Build the conversation structure."""
        result: dict[str, Any] = {
            "id": self.conv_id,
            "messages": self._messages,
        }
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        """Build and write to file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path




