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
    from polylogue.lib.stats import ArchiveStats
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
        full_id = self.resolve_id(conversation_id) or conversation_id
        return self.get(full_id)

    def get_eager(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with all messages loaded."""
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        msg_records = self._backend.get_messages(conversation_id)
        att_records = self._backend.get_attachments(conversation_id)

        return Conversation.from_records(conv_record, msg_records, att_records)

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        """Get a single conversation summary without loading messages."""
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None
        return ConversationSummary.from_record(conv_record)

    def list_summaries(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
    ) -> builtins.list[ConversationSummary]:
        """List conversation summaries without loading messages."""
        conv_records = self._backend.list_conversations(
            source=source,
            provider=provider,
            providers=providers,
            limit=limit,
            offset=offset,
            since=since,
            until=until,
            title_contains=title_contains,
        )
        return [ConversationSummary.from_record(rec) for rec in conv_records]

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
    ) -> builtins.list[Conversation]:
        """List conversations with lazy message loading."""
        conv_records = self._backend.list_conversations(
            provider=provider,
            providers=providers,
            limit=limit,
            offset=offset,
            since=since,
            until=until,
            title_contains=title_contains,
        )
        source = RepositoryMessageSource(self._backend)
        return [Conversation.from_lazy(rec, source) for rec in conv_records]

    def count(
        self,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
    ) -> int:
        """Count conversations matching filters without loading data."""
        return self._backend.count_conversations(
            provider=provider,
            providers=providers,
            since=since,
            until=until,
            title_contains=title_contains,
        )

    def get_parent(self, conversation_id: str) -> Conversation | None:
        """Get the parent conversation if it exists."""
        conv = self.get(conversation_id)
        if conv and conv.parent_id:
            return self.get(str(conv.parent_id))
        return None

    def get_children(self, conversation_id: str) -> builtins.list[Conversation]:
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

    def get_session_tree(self, conversation_id: str) -> builtins.list[Conversation]:
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

    def search_summaries(
        self, query: str, limit: int = 20, providers: builtins.list[str] | None = None
    ) -> builtins.list[ConversationSummary]:
        """Search conversations and return summaries."""
        ids = self._backend.search_conversations(query, limit=limit, providers=providers)
        summaries = []
        for cid in ids:
            record = self._backend.get_conversation(cid)
            if record:
                summaries.append(ConversationSummary.from_record(record))
        return summaries

    def search(
        self, query: str, limit: int = 20, providers: builtins.list[str] | None = None
    ) -> builtins.list[Conversation]:
        """Search conversations using full-text search."""
        ids = self._backend.search_conversations(query, limit=limit, providers=providers)
        return self._get_many(ids)

    def _get_many(self, conversation_ids: builtins.list[str]) -> builtins.list[Conversation]:
        """Bulk fetch lazy conversation objects in a single SQL query."""
        if not conversation_ids:
            return []
        records = self._backend.get_conversations_batch(conversation_ids)
        source = RepositoryMessageSource(self._backend)
        return [Conversation.from_lazy(rec, source) for rec in records]

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
        from polylogue.storage.backends.sqlite import open_connection

        with open_connection(self._db_path) as conn:
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
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a conversation with its messages and attachments atomically."""
        return self._save_via_backend(conversation, messages, attachments)

    def _save_via_backend(
        self,
        conversation: ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
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

    def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        """List all tags with counts. Read-only, no write lock needed."""
        return self._backend.list_tags(provider=provider)

    def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        with self._write_lock:
            self._backend.set_metadata(conversation_id, metadata)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._write_lock:
            return self._backend.delete_conversation(conversation_id)

    # --- Vector Search Methods ---

    def embed_conversation(
        self,
        conversation_id: str,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        """Generate and store embeddings for a conversation.

        Args:
            conversation_id: Conversation to embed
            vector_provider: Optional vector provider (creates default if None)

        Returns:
            Number of messages embedded

        Raises:
            ValueError: If no vector provider available
        """
        if vector_provider is None:
            from polylogue.storage.search_providers import create_vector_provider
            vector_provider = create_vector_provider()

        if vector_provider is None:
            raise ValueError("No vector provider available. Set VOYAGE_API_KEY.")

        messages = self._backend.get_messages(conversation_id)
        if not messages:
            return 0

        vector_provider.upsert(conversation_id, messages)
        return len(messages)

    def similarity_search(
        self,
        query: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[tuple[str, str, float]]:
        """Search conversations by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum results
            vector_provider: Optional vector provider

        Returns:
            List of (conversation_id, message_id, distance) tuples
        """
        if vector_provider is None:
            from polylogue.storage.search_providers import create_vector_provider
            vector_provider = create_vector_provider()

        if vector_provider is None:
            raise ValueError("No vector provider configured")

        results = vector_provider.query(query, limit=limit)
        if not results:
            return []

        # Batch lookup conversation IDs (single query instead of N+1)
        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = self._get_message_conversation_mapping(message_ids)

        return [
            (msg_to_conv[msg_id], msg_id, distance)
            for msg_id, distance in results
            if msg_id in msg_to_conv
        ]

    def get_archive_stats(self) -> ArchiveStats:
        """Get comprehensive archive statistics.

        Returns:
            ArchiveStats with all metrics
        """
        from polylogue.lib.stats import ArchiveStats

        conn = self._backend._get_connection()

        conv_count = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        provider_rows = conn.execute(
            """
            SELECT provider_name, COUNT(*) as count
            FROM conversations
            GROUP BY provider_name
            """
        ).fetchall()
        providers = {row["provider_name"]: row["count"] for row in provider_rows}

        # Check embedding status if table exists
        embedded_convs = 0
        embedded_msgs = 0
        try:
            embedded_convs = conn.execute(
                "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
            ).fetchone()[0]
            embedded_msgs = conn.execute(
                "SELECT COUNT(*) FROM message_embeddings"
            ).fetchone()[0]
        except Exception as exc:
            logger.debug("Embedding stats query failed: %s", exc)

        # Get database size
        db_size = 0
        try:
            import os
            db_size = os.path.getsize(self._db_path) if self._db_path else 0
        except Exception as exc:
            logger.debug("DB size check failed: %s", exc)

        return ArchiveStats(
            total_conversations=conv_count,
            total_messages=msg_count,
            providers=providers,
            embedded_conversations=embedded_convs,
            embedded_messages=embedded_msgs,
            db_size_bytes=db_size,
        )


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
