"""Storage repository for encapsulating database operations."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
)

if TYPE_CHECKING:
    from polylogue.protocols import StorageBackend


class StorageRepository:
    """Repository for managing database storage operations.

    Encapsulates the write lock and provides thread-safe methods for
    storing conversations, messages, attachments, and run records.

    This repository owns the write lock and ensures thread-safe access to
    the database. All write operations go through the StorageBackend abstraction.
    SQLiteBackend is internally thread-safe using threading.local().
    """

    def __init__(
        self,
        backend: StorageBackend,
    ) -> None:
        """Initialize the repository.

        Args:
            backend: Storage backend for all database operations.
                    SQLiteBackend is internally thread-safe using threading.local().
        """
        self._write_lock = threading.Lock()
        self._backend = backend
        # Store db_path for thread workers in IngestionService
        self._db_path = getattr(backend, '_db_path', None)

    def save_conversation(
        self,
        *,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a conversation with its messages and attachments.

        This is the primary write method for ingesting conversation data.
        All operations are performed atomically under the repository's write lock.

        Args:
            conversation: Conversation record to save
            messages: List of message records
            attachments: List of attachment records

        Returns:
            Dictionary with counts:
                - conversations: Number of conversations inserted
                - messages: Number of messages inserted
                - attachments: Number of attachments inserted
                - skipped_conversations: Number already existing (by content hash)
                - skipped_messages: Number already existing (by content hash)
                - skipped_attachments: Number already existing (by ref)
        """
        return self._save_via_backend(conversation, messages, attachments)

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
                # Always save conversation - the backend's SQL handles upsert logic
                # and will update metadata fields (title, updated_at, provider_meta)
                # even when content_hash is unchanged
                existing = backend.get_conversation(conversation.conversation_id)
                backend.save_conversation(conversation)
                if existing and existing.content_hash == conversation.content_hash:
                    counts["skipped_conversations"] += 1
                else:
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

                # Prune attachments no longer in the bundle (handles empty attachments too)
                new_attachment_ids: set[str] = {str(att.attachment_id) for att in attachments}
                backend.prune_attachments(conversation.conversation_id, new_attachment_ids)

                # Save all attachments (backend handles refs)
                if attachments:
                    backend.save_attachments(attachments)

                backend.commit()
            except Exception:
                backend.rollback()
                raise

        return counts

    def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run audit entry.

        Args:
            record: Run record to save
        """
        with self._write_lock:
            self._backend.record_run(record)

    # --- Metadata CRUD ---

    def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata dict for a conversation."""
        return self._backend.get_metadata(conversation_id)

    def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Set a single metadata key."""
        with self._write_lock:
            self._backend.update_metadata(conversation_id, key, value)

    def delete_metadata(self, conversation_id: str, key: str) -> None:
        """Remove a metadata key."""
        with self._write_lock:
            self._backend.delete_metadata(conversation_id, key)

    def add_tag(self, conversation_id: str, tag: str) -> None:
        """Add a tag to the conversation's tags list."""
        with self._write_lock:
            self._backend.add_tag(conversation_id, tag)

    def remove_tag(self, conversation_id: str, tag: str) -> None:
        """Remove a tag from the conversation's tags list."""
        with self._write_lock:
            self._backend.remove_tag(conversation_id, tag)

    def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        """Replace entire metadata dict."""
        with self._write_lock:
            self._backend.set_metadata(conversation_id, metadata)


__all__ = ["StorageRepository"]
