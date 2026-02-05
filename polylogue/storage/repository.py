from __future__ import annotations

import builtins
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from polylogue.lib.log import get_logger
from polylogue.lib.messages import MessageSource
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.types import ConversationId

from .store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
)

if TYPE_CHECKING:
    from polylogue.lib import filters
    from polylogue.protocols import VectorProvider

logger = get_logger(__name__)


class RepositoryMessageSource(MessageSource):
    """Adapter providing MessageSource protocol for lazy message loading."""

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    def iter_messages(self, conversation_id: str) -> Iterator[Message]:
        for record in self._backend.iter_messages(conversation_id):
            yield Message.from_record(record, attachments=[])

    def count_messages(self, conversation_id: str) -> int:
        stats = self._backend.get_conversation_stats(conversation_id)
        return stats.get("total_messages", 0)


class ConversationRepository:
    """Unified repository for conversation read/write operations.

    Encapsulates the write lock for thread-safety and provides methods for
    querying, retrieving, and storing conversation data.
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend
        self._write_lock = threading.Lock()
        # Store db_path for thread workers in IngestionService
        self._db_path = getattr(backend, "_db_path", None)

    @property
    def backend(self) -> SQLiteBackend:
        """Access the underlying storage backend."""
        return self._backend

    # --- Read Methods ---

    def resolve_id(self, id_prefix: str) -> ConversationId | None:
        """Resolve a partial ID to a full conversation ID."""
        resolved = self._backend.resolve_id(id_prefix)
        return ConversationId(resolved) if resolved else None

    def get(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with lazy message loading."""
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        source = RepositoryMessageSource(self._backend)
        return Conversation.from_lazy(conv_record, source)

    def view(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with ID resolution support."""
        full_id = self.resolve_id(conversation_id)
        if not full_id:
            # Try as full ID if resolution fails
            full_id = conversation_id
        return self.get(full_id)

    def get_eager(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with all messages loaded."""
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        msg_records = self._backend.get_messages(conversation_id)
        att_records = self._backend.get_attachments(conversation_id)

        return Conversation.from_records(conv_record, msg_records, att_records)

    def list_summaries(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        source: str | None = None,
    ) -> list[ConversationSummary]:
        """List conversation summaries without loading messages."""
        conv_records = self._backend.list_conversations(
            source=source,
            provider=provider,
            limit=limit,
            offset=offset,
        )
        return [ConversationSummary.from_record(rec) for rec in conv_records]

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
    ) -> list[Conversation]:
        """List conversations with lazy message loading."""
        conv_records = self._backend.list_conversations(
            provider=provider,
            limit=limit,
            offset=offset,
        )
        source = RepositoryMessageSource(self._backend)
        return [Conversation.from_lazy(rec, source) for rec in conv_records]

    def get_parent(self, conversation_id: str) -> Conversation | None:
        """Get the parent conversation if it exists."""
        conv = self.get(conversation_id)
        if conv and conv.parent_id:
            return self.get(str(conv.parent_id))
        return None

    def get_children(self, conversation_id: str) -> list[Conversation]:
        """Get all direct children of this conversation."""
        child_records = self._backend.list_conversations(parent_id=conversation_id)
        source = RepositoryMessageSource(self._backend)
        return [Conversation.from_lazy(rec, source) for rec in child_records]

    def get_root(self, conversation_id: str) -> Conversation:
        """Walk up to find the root conversation."""
        current = self.get(conversation_id)
        if not current:
            raise ValueError(f"Conversation {conversation_id} not found")

        while current.parent_id:
            parent = self.get(str(current.parent_id))
            if not parent:
                break
            current = parent
        return current

    def get_session_tree(self, conversation_id: str) -> list[Conversation]:
        """Get all conversations in the session tree."""
        root = self.get_root(conversation_id)

        tree = []
        queue = [root]

        while queue:
            current = queue.pop(0)
            tree.append(current)
            children = self.get_children(str(current.id))
            queue.extend(children)

        return tree

    def search_summaries(self, query: str, limit: int = 20) -> list[ConversationSummary]:
        """Search conversations and return summaries."""
        ids = self._backend.search_conversations(query, limit=limit)
        summaries = []
        for cid in ids:
            record = self._backend.get_conversation(cid)
            if record:
                summaries.append(ConversationSummary.from_record(record))
        return summaries

    def search(self, query: str, limit: int = 20) -> builtins.list[Conversation]:
        """Search conversations using full-text search."""
        ids = self._backend.search_conversations(query, limit=limit)
        return self._get_many(ids)

    def _get_many(self, conversation_ids: list[str]) -> list[Conversation]:
        """Bulk fetch lazy conversation objects."""
        if not conversation_ids:
            return []
        results = []
        for cid in conversation_ids:
            conv = self.get(cid)
            if conv is not None:
                results.append(conv)
        return results

    def get_conversation_stats(self, conversation_id: str) -> dict[str, int] | None:
        """Get message counts without loading messages."""
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None
        return self._backend.get_conversation_stats(conversation_id)

    def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """Stream messages without loading full conversation."""
        for record in self._backend.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            limit=limit,
        ):
            yield Message.from_record(record, attachments=[])

    def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[Conversation]:
        """Search by semantic similarity."""
        if not vector_provider:
            raise ValueError("Semantic search requires a vector provider.")

        results = vector_provider.query(text, limit=limit * 3)
        if not results:
            return []

        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = self._get_message_conversation_mapping(message_ids)

        conv_scores: dict[str, float] = {}
        for msg_id, score in results:
            conv_id = msg_to_conv.get(msg_id)
            if conv_id:
                conv_scores[conv_id] = max(conv_scores.get(conv_id, 0.0), score)

        ranked_ids = sorted(
            conv_scores.keys(),
            key=lambda x: conv_scores[x],
            reverse=True,
        )[:limit]

        return self._get_many(ranked_ids)

    def _get_message_conversation_mapping(self, message_ids: builtins.list[str]) -> dict[str, str]:
        conn = self._backend._get_connection()
        placeholders = ",".join("?" * len(message_ids))
        query = f"SELECT message_id, conversation_id FROM messages WHERE message_id IN ({placeholders})"
        rows = conn.execute(query, message_ids).fetchall()
        return {row["message_id"]: row["conversation_id"] for row in rows}

    def filter(self) -> filters.ConversationFilter:
        """Create a filter builder for chainable queries."""
        from polylogue.lib import filters

        return filters.ConversationFilter(self)

    # --- Write Methods ---

    def save_conversation(
        self,
        *,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a conversation with its messages and attachments atomically."""
        return self._save_via_backend(conversation, messages, attachments)

    def _save_via_backend(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
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
            backend.begin()
            try:
                existing = backend.get_conversation(conversation.conversation_id)
                backend.save_conversation(conversation)
                if existing and existing.content_hash == conversation.content_hash:
                    counts["skipped_conversations"] += 1
                else:
                    counts["conversations"] += 1

                existing_messages = {msg.message_id: msg for msg in backend.get_messages(conversation.conversation_id)}
                for message in messages:
                    existing_msg = existing_messages.get(message.message_id)
                    if existing_msg and existing_msg.content_hash == message.content_hash:
                        counts["skipped_messages"] += 1
                    else:
                        counts["messages"] += 1

                if messages:
                    backend.save_messages(messages)

                existing_attachments = {
                    att.attachment_id: att for att in backend.get_attachments(conversation.conversation_id)
                }
                for attachment in attachments:
                    if attachment.attachment_id in existing_attachments:
                        counts["skipped_attachments"] += 1
                    else:
                        counts["attachments"] += 1

                new_attachment_ids: set[str] = {str(att.attachment_id) for att in attachments}
                backend.prune_attachments(conversation.conversation_id, new_attachment_ids)

                if attachments:
                    backend.save_attachments(attachments)

                backend.commit()
            except Exception:
                backend.rollback()
                raise

        return counts

    def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run audit entry."""
        with self._write_lock:
            self._backend.record_run(record)

    # --- Metadata CRUD ---

    def get_metadata(self, conversation_id: str) -> dict[str, object]:
        return self._backend.get_metadata(conversation_id)

    def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        with self._write_lock:
            self._backend.update_metadata(conversation_id, key, value)

    def delete_metadata(self, conversation_id: str, key: str) -> None:
        with self._write_lock:
            self._backend.delete_metadata(conversation_id, key)

    def add_tag(self, conversation_id: str, tag: str) -> None:
        with self._write_lock:
            self._backend.add_tag(conversation_id, tag)

    def remove_tag(self, conversation_id: str, tag: str) -> None:
        with self._write_lock:
            self._backend.remove_tag(conversation_id, tag)

    def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        with self._write_lock:
            self._backend.set_metadata(conversation_id, metadata)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._write_lock:
            return self._backend.delete_conversation(conversation_id)


def _records_to_conversation(
    conversation: ConversationRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
) -> Conversation:
    """Convert records to a Conversation model.

    Used by async facades and internal migration tools.
    """
    return Conversation.from_records(conversation, messages, attachments)


__all__ = ["ConversationRepository", "_records_to_conversation"]
