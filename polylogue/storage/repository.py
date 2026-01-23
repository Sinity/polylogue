"""Storage repository for encapsulating database operations."""

from __future__ import annotations

import sqlite3
import threading

from .db import connection_context
from .store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
    _prune_attachment_refs,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)


class StorageRepository:
    """Repository for managing database storage operations.

    Encapsulates the write lock and provides thread-safe methods for
    storing conversations, messages, attachments, and run records.

    This repository owns the write lock and ensures thread-safe access to
    the database. All write operations should go through this repository.
    """

    def __init__(self) -> None:
        """Initialize the repository with its own write lock."""
        self._write_lock = threading.Lock()

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

        Returns:
            Dictionary with counts:
                - conversations: Number of conversations inserted
                - messages: Number of messages inserted
                - attachments: Number of attachments inserted
                - skipped_conversations: Number already existing (by content hash)
                - skipped_messages: Number already existing (by content hash)
                - skipped_attachments: Number already existing (by ref)
        """
        counts = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }

        from .store import _make_ref_id

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
