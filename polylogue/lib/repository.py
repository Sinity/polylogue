"""Repository layer for conversation access.

This module provides the `ConversationRepository` class, which is the primary
interface for querying and retrieving conversations from the database.

The repository now returns **lazy** `Conversation` objects by default, which
stream messages from the database on iteration. This reduces memory usage from
O(n) to O(1) for large conversations.

For use cases requiring full message access (filtering, indexing), use
`get_eager()` instead of `get()`.

All database operations go through the StorageBackend protocol for clean abstraction.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from typing import TYPE_CHECKING

from polylogue.core.log import get_logger
from polylogue.lib.messages import MessageSource
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import ConversationId

if TYPE_CHECKING:
    from polylogue.lib import filters
    from polylogue.protocols import StorageBackend, VectorProvider

logger = get_logger(__name__)


class RepositoryMessageSource(MessageSource):
    """Adapter providing MessageSource protocol for lazy message loading.

    This class bridges the gap between the StorageBackend's streaming API
    and the MessageCollection's lazy loading interface.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize with a storage backend.

        Args:
            backend: Storage backend for database access
        """
        self._backend = backend

    def iter_messages(self, conversation_id: str) -> Iterator[Message]:
        """Stream messages from the database.

        Args:
            conversation_id: ID of the conversation

        Yields:
            Message objects one at a time
        """
        from polylogue.storage.backends.sqlite import SQLiteBackend

        # Use streaming if available
        if isinstance(self._backend, SQLiteBackend):
            for record in self._backend.iter_messages(conversation_id):
                yield Message.from_record(record, attachments=[])
        else:
            # Fallback for other backends: load all messages
            msg_records = self._backend.get_messages(conversation_id)
            att_records = self._backend.get_attachments(conversation_id)

            att_map: dict[str, list[AttachmentRecord]] = {}
            for att in att_records:
                if att.message_id:
                    att_map.setdefault(att.message_id, []).append(att)

            for record in msg_records:
                yield Message.from_record(record, att_map.get(record.message_id, []))

    def count_messages(self, conversation_id: str) -> int:
        """Get message count using efficient COUNT(*) query.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Total number of messages
        """
        from polylogue.storage.backends.sqlite import SQLiteBackend

        if isinstance(self._backend, SQLiteBackend):
            stats = self._backend.get_conversation_stats(conversation_id)
            return stats.get("total_messages", 0)

        # Fallback: load and count (not ideal but maintains API)
        msg_records = self._backend.get_messages(conversation_id)
        return len(msg_records)


def _records_to_conversation(
    conv_record: ConversationRecord,
    msg_records: list[MessageRecord],
    att_records: list[AttachmentRecord],
) -> Conversation:
    """Convert storage records to a Conversation domain object.

    Args:
        conv_record: Conversation record from storage
        msg_records: Message records from storage
        att_records: Attachment records from storage

    Returns:
        Conversation object with semantic projection support
    """
    return Conversation.from_records(conv_record, msg_records, att_records)


class ConversationRepository:
    """Repository for querying and retrieving conversations.

    This is the primary interface for accessing conversation data. It returns
    `Conversation` objects that support semantic projections like:

    - `substantive_only()` - Filter to substantive dialogue
    - `iter_pairs()` - Iterate user/assistant turn pairs
    - `without_noise()` - Remove tool calls, context dumps
    - `to_clean_text()` - Render as clean dialogue text

    All database operations go through a StorageBackend for clean abstraction.

    Example:
        from polylogue.storage.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_path="/path/to/db.db")
        repo = ConversationRepository(backend=backend)
        conv = repo.get("claude:abc123")
        if conv:
            for pair in conv.iter_pairs():
                print(pair.exchange)
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the repository.

        Args:
            backend: Storage backend for all database operations.
                    Use SQLiteBackend for SQLite, or implement the protocol
                    for PostgreSQL, DuckDB, etc.
        """
        self._backend = backend

    @property
    def backend(self) -> StorageBackend:
        """Access the underlying storage backend."""
        return self._backend

    def resolve_id(self, id_prefix: str) -> ConversationId | None:
        """Resolve a partial ID to a full conversation ID.

        Supports both exact matches and prefix matches. If multiple
        conversations match the prefix, returns None (ambiguous).

        Args:
            id_prefix: Full or partial conversation ID.

        Returns:
            The full conversation ID if exactly one match, None otherwise.
        """
        resolved = self._backend.resolve_id(id_prefix)
        return ConversationId(resolved) if resolved else None

    def view(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with full semantic projection support.

        This is the primary API for consumers. The returned Conversation
        has methods like `substantive_only()`, `iter_pairs()`, `without_noise()`.

        Supports partial ID resolution - if a unique prefix is provided,
        it will be resolved to the full ID.

        Args:
            conversation_id: Full or partial conversation ID.

        Returns:
            A Conversation with projection methods, or None if not found.
        """
        full_id = self.resolve_id(conversation_id)
        if not full_id:
            return None
        return self.get(full_id)

    def get(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with lazy message loading (memory-efficient).

        Messages will stream from the database on iteration, using O(1) memory
        regardless of conversation size. Use `get_eager()` if you need full
        message access upfront (e.g., for indexing).

        Note: Attachments are not loaded in lazy mode. If you need attachments,
        use `get_eager()` instead.

        Args:
            conversation_id: Conversation ID to retrieve

        Returns:
            Conversation object with lazy messages, or None if not found
        """
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        source = RepositoryMessageSource(self._backend)
        return Conversation.from_lazy(conv_record, source)

    def get_eager(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with all messages loaded (eager mode).

        This loads all messages and attachments into memory upfront.
        Use this when you need:
        - Indexing access (`conv.messages[0]`)
        - Attachment data
        - In-memory filtering that needs re-iteration

        For iteration-only access, prefer `get()` which uses O(1) memory.

        Args:
            conversation_id: Conversation ID to retrieve

        Returns:
            Conversation object with eager messages, or None if not found
        """
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        msg_records = self._backend.get_messages(conversation_id)
        att_records = self._backend.get_attachments(conversation_id)

        return _records_to_conversation(conv_record, msg_records, att_records)

    def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """Stream messages without loading full conversation.

        This is the memory-efficient way to access large conversations.
        Messages are yielded one at a time, keeping memory usage constant
        regardless of conversation size.

        Note: Attachments are not loaded when streaming (they're set to empty
        lists). If you need attachments, use get() instead.

        Args:
            conversation_id: ID of the conversation to stream
            dialogue_only: If True, only yield user/assistant messages
            limit: Maximum messages to yield. None = no limit.

        Yields:
            Message objects one at a time
        """
        from polylogue.storage.backends.sqlite import SQLiteBackend

        # Only SQLiteBackend supports iter_messages currently
        if not isinstance(self._backend, SQLiteBackend):
            # Fallback: load full conversation and iterate
            conv = self.get(conversation_id)
            if not conv:
                return
            count = 0
            for msg in conv.messages:
                if dialogue_only and not msg.is_dialogue:
                    continue
                yield msg
                count += 1
                if limit is not None and count >= limit:
                    return
            return

        # Use streaming backend method
        for record in self._backend.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            limit=limit,
        ):
            # Convert record to Message without attachments (deferred for memory)
            yield Message.from_record(record, attachments=[])

    def get_conversation_stats(self, conversation_id: str) -> dict[str, int] | None:
        """Get message counts without loading messages.

        Useful for deciding whether to use streaming and for UI display.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Dict with counts, or None if conversation doesn't exist
        """
        from polylogue.storage.backends.sqlite import SQLiteBackend

        # Check conversation exists
        conv_record = self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        if isinstance(self._backend, SQLiteBackend):
            return self._backend.get_conversation_stats(conversation_id)

        # Fallback: load and count (defeats the purpose but provides API consistency)
        conv = self.get(conversation_id)
        if not conv:
            return None
        dialogue_count = sum(1 for m in conv.messages if m.is_dialogue)
        return {
            "total_messages": len(conv.messages),
            "dialogue_messages": dialogue_count,
            "tool_messages": len(conv.messages) - dialogue_count,
        }

    def _get_many(self, conversation_ids: list[str]) -> list[Conversation]:
        """Bulk fetch lazy conversation objects.

        Args:
            conversation_ids: List of conversation IDs to fetch

        Returns:
            List of lazy Conversation objects (may be fewer than requested if some don't exist)
        """
        if not conversation_ids:
            return []

        results = []
        for cid in conversation_ids:
            conv = self.get(cid)
            if conv is not None:
                results.append(conv)
        return results

    def _get_many_eager(self, conversation_ids: list[str]) -> list[Conversation]:
        """Bulk fetch eager conversation objects with all messages loaded.

        Args:
            conversation_ids: List of conversation IDs to fetch

        Returns:
            List of eager Conversation objects (may be fewer than requested if some don't exist)
        """
        if not conversation_ids:
            return []

        results = []
        for cid in conversation_ids:
            conv = self.get_eager(cid)
            if conv is not None:
                results.append(conv)
        return results

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
    ) -> list[Conversation]:
        """List conversations with optional filtering and pagination.

        Returns lazy Conversation objects that stream messages on iteration.
        This is memory-efficient for listing and basic iteration.

        For use cases requiring full message access (filtering, searching),
        use list_eager() or access conversations individually with get_eager().

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            provider: Optional provider filter (e.g., "claude", "chatgpt")

        Returns:
            List of lazy Conversation objects ordered by updated_at DESC
        """
        conv_records = self._backend.list_conversations(
            provider=provider,
            limit=limit,
            offset=offset,
        )
        source = RepositoryMessageSource(self._backend)
        return [Conversation.from_lazy(rec, source) for rec in conv_records]

    def list_eager(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
    ) -> list[Conversation]:
        """List conversations with all messages loaded (eager mode).

        WARNING: This loads full conversations with all messages into memory.
        For large databases, prefer list() (lazy) or list_summaries() unless
        you specifically need full message access.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            provider: Optional provider filter (e.g., "claude", "chatgpt")

        Returns:
            List of Conversation objects with eager messages
        """
        conv_records = self._backend.list_conversations(
            provider=provider,
            limit=limit,
            offset=offset,
        )
        ids = [str(rec.conversation_id) for rec in conv_records]
        return self._get_many_eager(ids)

    def list_summaries(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        source: str | None = None,
    ) -> list[ConversationSummary]:
        """List conversation summaries without loading messages.

        This is the memory-efficient way to list conversations. Returns
        lightweight ConversationSummary objects that contain metadata but
        no message content.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            provider: Optional provider filter (e.g., "claude", "chatgpt")
            source: Optional source filter

        Returns:
            List of ConversationSummary objects ordered by updated_at DESC
        """
        conv_records = self._backend.list_conversations(
            source=source,
            provider=provider,
            limit=limit,
            offset=offset,
        )
        return [ConversationSummary.from_record(rec) for rec in conv_records]

    def search_summaries(self, query: str, limit: int = 20) -> list[ConversationSummary]:
        """Search conversations and return summaries without loading messages.

        Uses FTS to find matching conversations, but only returns lightweight
        summary objects.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of ConversationSummary objects
        """
        ids = self._backend.search_conversations(query, limit=limit)
        summaries = []
        for cid in ids:
            record = self._backend.get_conversation(cid)
            if record:
                summaries.append(ConversationSummary.from_record(record))
        return summaries

    def search(self, query: str, limit: int = 20) -> builtins.list[Conversation]:
        """Search conversations using full-text search.

        Returns lazy Conversation objects for memory efficiency.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default 20)

        Returns:
            List of lazy matching conversations ordered by relevance

        Raises:
            DatabaseError: If search index not available
        """
        ids = self._backend.search_conversations(query, limit=limit)
        return self._get_many(ids)

    def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[Conversation]:
        """Search by semantic similarity using vector embeddings.

        Args:
            text: Query text to find similar conversations for
            limit: Maximum number of results to return
            vector_provider: Optional vector provider (Qdrant). If None, raises error.

        Returns:
            List of conversations ranked by similarity

        Raises:
            ValueError: If vector_provider is None
        """
        if not vector_provider:
            raise ValueError(
                "Semantic search requires a vector provider. Set QDRANT_URL and VOYAGE_API_KEY environment variables."
            )

        # Query returns (message_id, score) tuples
        results = vector_provider.query(text, limit=limit * 3)

        if not results:
            return []

        # Get message->conversation mapping from backend
        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = self._get_message_conversation_mapping(message_ids)

        # Aggregate to conversation level (max score per conversation)
        conv_scores: dict[str, float] = {}
        for msg_id, score in results:
            conv_id = msg_to_conv.get(msg_id)
            if conv_id:
                conv_scores[conv_id] = max(conv_scores.get(conv_id, 0.0), score)

        # Sort by score descending, take top N
        ranked_ids = sorted(
            conv_scores.keys(),
            key=lambda x: conv_scores[x],
            reverse=True,
        )[:limit]

        return self._get_many(ranked_ids)

    def _get_message_conversation_mapping(self, message_ids: builtins.list[str]) -> dict[str, str]:
        """Get conversation IDs for a list of message IDs.

        Args:
            message_ids: List of message IDs

        Returns:
            Mapping of message_id -> conversation_id
        """
        # Access backend's connection to query messages table
        # This is a bit of a hack but avoids adding a new protocol method
        from polylogue.storage.backends.sqlite import SQLiteBackend

        if isinstance(self._backend, SQLiteBackend):
            conn = self._backend._get_connection()
            placeholders = ",".join("?" * len(message_ids))
            query = f"SELECT message_id, conversation_id FROM messages WHERE message_id IN ({placeholders})"
            rows = conn.execute(query, message_ids).fetchall()
            return {row["message_id"]: row["conversation_id"] for row in rows}

        # Fallback: empty mapping
        return {}

    def get_parent(self, conversation_id: str) -> Conversation | None:
        """Get the parent conversation if one exists.

        Args:
            conversation_id: ID of the child conversation

        Returns:
            Parent Conversation or None if no parent
        """
        conv = self.get(conversation_id)
        if not conv or not conv.parent_id:
            return None
        return self.get(str(conv.parent_id))

    def get_children(self, conversation_id: str) -> list[Conversation]:
        """Get all direct children of a conversation.

        Args:
            conversation_id: Parent conversation ID

        Returns:
            List of child Conversation objects
        """
        # Use the backend's list_conversations_by_parent method
        from polylogue.storage.backends.sqlite import SQLiteBackend

        if isinstance(self._backend, SQLiteBackend):
            child_records = self._backend.list_conversations_by_parent(conversation_id)
            return self._get_many([str(r.conversation_id) for r in child_records])
        return []

    def get_session_tree(self, conversation_id: str) -> list[Conversation]:
        """Get the full session tree containing this conversation.

        Returns all conversations in the tree from root to leaves,
        including the given conversation.

        Args:
            conversation_id: Any conversation ID in the tree

        Returns:
            List of all Conversation objects in the tree, ordered root-first
        """
        root = self.get_root(conversation_id)
        if not root:
            return []

        # BFS to collect all nodes
        result = [root]
        queue = [str(root.id)]
        seen = {str(root.id)}

        while queue:
            current_id = queue.pop(0)
            children = self.get_children(current_id)
            for child in children:
                if str(child.id) not in seen:
                    seen.add(str(child.id))
                    result.append(child)
                    queue.append(str(child.id))

        return result

    def get_root(self, conversation_id: str) -> Conversation | None:
        """Get the root conversation of the tree containing this conversation.

        Args:
            conversation_id: Any conversation ID in the tree

        Returns:
            Root Conversation or None if not found
        """
        conv = self.get(conversation_id)
        if not conv:
            return None

        # Walk up to root
        while conv.parent_id:
            parent = self.get(str(conv.parent_id))
            if not parent:
                break
            conv = parent

        return conv

    def filter(self) -> filters.ConversationFilter:
        """Create a filter builder for chainable queries.

        Returns a ConversationFilter that can be used to build complex queries:

            repo.filter()
                .provider("claude")
                .since("2024-01-01")
                .contains("error")
                .limit(10)
                .list()

        Returns:
            ConversationFilter builder for chainable queries
        """
        from polylogue.lib import filters

        return filters.ConversationFilter(self)
