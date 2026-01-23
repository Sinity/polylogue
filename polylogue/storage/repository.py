"""Storage repository for encapsulating database operations."""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

from .db import connection_context
from .store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
    _make_ref_id,
    _prune_attachment_refs,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)

if TYPE_CHECKING:
    from polylogue.protocols import StorageBackend


class StorageRepository:
    """Repository for managing database storage operations.

    Encapsulates the write lock and provides thread-safe methods for
    storing conversations, messages, attachments, and run records.

    This repository owns the write lock and ensures thread-safe access to
    the database. All write operations should go through this repository.

    The repository can optionally use a StorageBackend for database operations,
    enabling backend abstraction (SQLite, PostgreSQL, etc.). When no backend
    is provided, it uses direct SQLite operations for backward compatibility.
    """

    def __init__(self, backend: StorageBackend | None = None) -> None:
        """Initialize the repository.

        Args:
            backend: Optional storage backend. If provided, all database operations
                    will be delegated to this backend. If None, uses direct SQLite
                    operations via connection_context for backward compatibility.
        """
        self._write_lock = threading.Lock()
        self._backend = backend

    def save_conversation(
        self,
        *,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, int]:
        """Save a conversation with its messages and attachments.

        This is the primary write method for ingesting conversation data.
        All operations are performed atomically under the repository's write lock.

        Args:
            conversation: Conversation record to save
            messages: List of message records
            attachments: List of attachment records
            conn: Optional database connection (for transaction control)
                 Only used when backend is None (legacy mode)

        Returns:
            Dictionary with counts:
                - conversations: Number of conversations inserted
                - messages: Number of messages inserted
                - attachments: Number of attachments inserted
                - skipped_conversations: Number already existing (by content hash)
                - skipped_messages: Number already existing (by content hash)
                - skipped_attachments: Number already existing (by ref)
        """
        # Use backend if available, otherwise fall back to legacy SQLite operations
        if self._backend is not None:
            return self._save_via_backend(conversation, messages, attachments)
        else:
            return self._save_via_legacy(conversation, messages, attachments, conn)

    def _save_via_backend(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save via StorageBackend (new abstraction layer)."""
        counts: dict[str, int] = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }

        backend = self._backend
        if backend is None:
            raise RuntimeError("Backend is not initialized")

        with self._write_lock:
            # Use backend transaction for atomicity
            backend.begin()
            try:
                # Check if conversation already exists with same content_hash
                existing = backend.get_conversation(conversation.conversation_id)
                if existing and existing.content_hash == conversation.content_hash:
                    counts["skipped_conversations"] += 1
                else:
                    backend.save_conversation(conversation)
                    counts["conversations"] += 1

                # Check and save messages
                existing_messages = {msg.message_id: msg for msg in backend.get_messages(conversation.conversation_id)}
                for message in messages:
                    existing_msg = existing_messages.get(message.message_id)
                    if existing_msg and existing_msg.content_hash == message.content_hash:
                        counts["skipped_messages"] += 1
                    else:
                        counts["messages"] += 1

                # Save all messages (backend handles upsert)
                if messages:
                    backend.save_messages(messages)

                # Check and save attachments
                existing_attachments = {att.attachment_id: att for att in backend.get_attachments(conversation.conversation_id)}
                for attachment in attachments:
                    if attachment.attachment_id in existing_attachments:
                        counts["skipped_attachments"] += 1
                    else:
                        counts["attachments"] += 1

                # Save all attachments (backend handles refs)
                if attachments:
                    backend.save_attachments(attachments)

                backend.commit()
            except Exception:
                backend.rollback()
                raise

        return counts

    def _save_via_legacy(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
        conn: sqlite3.Connection | None,
    ) -> dict[str, int]:
        """Save via legacy store.py functions (backward compatibility)."""
        counts = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }

        with connection_context(conn) as db_conn, self._write_lock:
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
                ref_id = _make_ref_id(
                    attachment.attachment_id,
                    attachment.conversation_id,
                    attachment.message_id,
                )
                seen_ref_ids.add(ref_id)
                if upsert_attachment(db_conn, attachment):
                    counts["attachments"] += 1
                else:
                    counts["skipped_attachments"] += 1

            _prune_attachment_refs(db_conn, conversation.conversation_id, seen_ref_ids)
            # Commit inside lock to ensure atomic transaction boundaries
            db_conn.commit()

        return counts

    def record_run(
        self,
        record: RunRecord,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Record a pipeline run audit entry.

        Args:
            record: Run record to save
            conn: Optional database connection (for transaction control)
        """
        from .store import _json_or_none

        with connection_context(conn) as db_conn, self._write_lock:
            db_conn.execute(
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
            db_conn.commit()


__all__ = ["StorageRepository"]
