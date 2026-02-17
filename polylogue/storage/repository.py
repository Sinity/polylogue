"""Async storage repository for conversation persistence.

Provides async/await interface for storing and retrieving conversations.
Wraps SQLiteBackend for parallel operations.

All methods are async and use eager loading (Conversation.from_records)
instead of lazy loading, since async I/O already enables efficient parallel
fetching of conversations, messages, and attachments together.
"""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.lib.log import get_logger
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord, RunRecord
from polylogue.types import ConversationId

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.lib import filters
    from polylogue.lib.stats import ArchiveStats
    from polylogue.protocols import VectorProvider

logger = get_logger(__name__)


class ConversationRepository:
    """Async repository for conversation storage operations.

    Wraps SQLiteBackend to provide high-level async storage interface with
    full feature parity to sync ConversationRepository.

    All methods are async. Eager loading (Conversation.from_records) is used
    for fetching conversations, enabling efficient parallel I/O via asyncio.gather()
    for conversations, messages, and attachments.

    Write safety is provided by SQLite's ``BEGIN IMMEDIATE`` transactions
    in the backend layer, combined with asyncio.Lock() serialization.

    Example:
        async with ConversationRepository() as repo:
            conv = await repo.get("claude:abc123")
            convs = await repo.list(limit=10)
            await repo.save_conversation(conv_rec, msgs, atts)
    """

    def __init__(
        self,
        backend: SQLiteBackend | None = None,
        db_path: Path | None = None,
    ) -> None:
        """Initialize async storage repository.

        Args:
            backend: Optional SQLiteBackend instance. If provided, db_path is ignored.
            db_path: Optional path to database file. Used if backend is None.
        """
        if backend is not None:
            self._backend = backend
        else:
            self._backend = SQLiteBackend(db_path=db_path)

        # Expose db_path for schema inference (generate_provider_schema needs it)
        self._db_path = getattr(self._backend, "_db_path", None)

    async def __aenter__(self) -> ConversationRepository:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit async context manager."""
        await self.close()

    @property
    def backend(self) -> SQLiteBackend:
        """Access the underlying async storage backend."""
        return self._backend

    async def close(self) -> None:
        """Close database connections and release resources."""
        await self._backend.close()

    # --- Read Methods ---

    async def resolve_id(self, id_prefix: str) -> ConversationId | None:
        """Resolve a partial ID to a full conversation ID.

        Args:
            id_prefix: Full or partial conversation ID

        Returns:
            Full ConversationId if resolved, None if ambiguous or not found
        """
        resolved = await self._backend.resolve_id(id_prefix)
        return ConversationId(resolved) if resolved else None

    async def get(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with all messages and attachments (eager loading).

        Fetches conversation, messages, and attachments in parallel.

        Args:
            conversation_id: Full conversation ID

        Returns:
            Conversation with all data loaded, or None if not found
        """
        conv_record = await self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None

        # Fetch messages and attachments in parallel
        msg_records, att_records = await asyncio.gather(
            self._backend.get_messages(conversation_id),
            self._backend.get_attachments(conversation_id),
        )

        return Conversation.from_records(conv_record, msg_records, att_records)

    async def view(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with ID resolution support.

        Attempts to resolve a partial ID to a full ID, then fetches the conversation.

        Args:
            conversation_id: Full or partial conversation ID

        Returns:
            Conversation with all data loaded, or None if not found
        """
        full_id = await self.resolve_id(conversation_id) or conversation_id
        return await self.get(str(full_id))

    async def get_eager(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with all messages and attachments.

        Alias for get() - all async operations are eager by design.

        Args:
            conversation_id: Full conversation ID

        Returns:
            Conversation with all data loaded, or None if not found
        """
        return await self.get(conversation_id)

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Get conversation record by ID.

        Args:
            conversation_id: Full conversation ID

        Returns:
            ConversationRecord or None if not found
        """
        return await self._backend.get_conversation(conversation_id)

    async def conversation_exists(self, content_hash: str) -> bool:
        """Check if conversation with given content hash exists.

        Args:
            content_hash: SHA-256 hash of conversation content

        Returns:
            True if conversation exists, False otherwise
        """
        return await self._backend.conversation_exists_by_hash(content_hash)

    async def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        """Get a conversation summary without loading messages.

        Args:
            conversation_id: Full conversation ID

        Returns:
            ConversationSummary, or None if not found
        """
        conv_record = await self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None
        return ConversationSummary.from_record(conv_record)

    async def list_summaries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
    ) -> builtins.list[ConversationSummary]:
        """List conversation summaries without loading messages.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            provider: Filter by single provider name
            providers: Filter by multiple providers
            source: Filter by source name
            since: Filter to conversations updated on/after this ISO date
            until: Filter to conversations updated on/before this ISO date
            title_contains: Filter by title substring (case-insensitive)

        Returns:
            List of ConversationSummary objects
        """
        conv_records = await self._backend.list_conversations(
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

    async def list(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
    ) -> builtins.list[Conversation]:
        """List conversations with eager-loaded messages and attachments.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            provider: Filter by single provider name
            providers: Filter by multiple providers
            since: Filter to conversations updated on/after this ISO date
            until: Filter to conversations updated on/before this ISO date
            title_contains: Filter by title substring (case-insensitive)

        Returns:
            List of Conversation objects with all data eager-loaded
        """
        conv_records = await self._backend.list_conversations(
            provider=provider,
            providers=providers,
            limit=limit,
            offset=offset,
            since=since,
            until=until,
            title_contains=title_contains,
        )

        # Fetch messages and attachments for all conversations in parallel
        if not conv_records:
            return []

        return await self._get_many([rec.conversation_id for rec in conv_records])

    async def count(
        self,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
    ) -> int:
        """Count conversations matching filters.

        Args:
            provider: Filter by single provider
            providers: Filter by multiple providers
            since: Filter to conversations updated on/after this ISO date
            until: Filter to conversations updated on/before this ISO date
            title_contains: Filter by title substring

        Returns:
            Count of matching conversations
        """
        return await self._backend.count_conversations(
            provider=provider,
            providers=providers,
            since=since,
            until=until,
            title_contains=title_contains,
        )

    async def get_source_conversations(self, provider: str) -> builtins.list[str]:
        """Get all conversation IDs for a given provider.

        Args:
            provider: Provider name to filter by

        Returns:
            List of conversation IDs from that provider
        """
        records = await self._backend.list_conversations(provider=provider, limit=None)
        return [rec.conversation_id for rec in records]

    async def get_parent(self, conversation_id: str) -> Conversation | None:
        """Get the parent conversation if it exists.

        Args:
            conversation_id: Child conversation ID

        Returns:
            Parent Conversation, or None if no parent
        """
        conv = await self.get(conversation_id)
        if conv and conv.parent_id:
            return await self.get(str(conv.parent_id))
        return None

    async def get_children(self, conversation_id: str) -> builtins.list[Conversation]:
        """Get all direct children of this conversation.

        Args:
            conversation_id: Parent conversation ID

        Returns:
            List of child Conversation objects with all data eager-loaded
        """
        child_records = await self._backend.list_conversations(parent_id=conversation_id)
        if not child_records:
            return []
        return await self._get_many([rec.conversation_id for rec in child_records])

    async def get_root(self, conversation_id: str) -> Conversation:
        """Walk up the parent chain to find the root conversation.

        Args:
            conversation_id: Any conversation ID in the tree

        Returns:
            The root Conversation object

        Raises:
            ValueError: If the conversation is not found
        """
        current = await self.get(conversation_id)
        if not current:
            raise ValueError(f"Conversation {conversation_id} not found")

        while current.parent_id:
            parent = await self.get(str(current.parent_id))
            if not parent:
                break
            current = parent
        return current

    async def get_session_tree(self, conversation_id: str) -> builtins.list[Conversation]:
        """Get all conversations in the session tree (root + all descendants).

        Args:
            conversation_id: Any conversation in the tree

        Returns:
            List of all Conversation objects in tree (breadth-first order)
        """
        root = await self.get_root(conversation_id)

        tree = []
        queue = [root]

        while queue:
            current = queue.pop(0)
            tree.append(current)
            children = await self.get_children(str(current.id))
            queue.extend(children)

        return tree

    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[ConversationSummary]:
        """Search conversations and return summaries.

        Args:
            query: Full-text search query
            limit: Maximum results
            providers: Optional list of provider names to filter by

        Returns:
            List of ConversationSummary objects matching the query
        """
        ids = await self._backend.search_conversations(query, limit=limit, providers=providers)
        if not ids:
            return []

        # Fetch all matching conversations in parallel
        records = await self._backend.get_conversations_batch(ids)
        return [ConversationSummary.from_record(rec) for rec in records]

    async def search(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[Conversation]:
        """Search conversations using full-text search.

        Args:
            query: Full-text search query
            limit: Maximum results
            providers: Optional list of provider names to filter by

        Returns:
            List of Conversation objects with all data eager-loaded
        """
        ids = await self._backend.search_conversations(query, limit=limit, providers=providers)
        return await self._get_many(ids)

    async def _get_many(self, conversation_ids: builtins.list[str]) -> builtins.list[Conversation]:
        """Bulk fetch conversations with eager-loaded messages and attachments.

        Uses asyncio.gather() to fetch messages and attachments in parallel
        for all conversations.

        Args:
            conversation_ids: List of conversation IDs to fetch

        Returns:
            List of Conversation objects in the order of input IDs
        """
        if not conversation_ids:
            return []

        # Fetch conversation records
        records = await self._backend.get_conversations_batch(conversation_ids)
        if not records:
            return []

        # Build dict for order preservation
        by_id = {rec.conversation_id: rec for rec in records}

        # Fetch all messages and attachments in parallel
        fetch_tasks = [
            (cid, asyncio.gather(
                self._backend.get_messages(cid),
                self._backend.get_attachments(cid),
            ))
            for cid in conversation_ids
            if cid in by_id
        ]

        results: builtins.list[Conversation] = []
        for cid, gather_task in fetch_tasks:
            msg_records, att_records = await gather_task
            conv_record = by_id[cid]
            results.append(Conversation.from_records(conv_record, msg_records, att_records))

        return results

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int] | None:
        """Get message counts without loading messages.

        Args:
            conversation_id: Conversation ID

        Returns:
            Dict with counts (total_messages, dialogue_messages, tool_messages),
            or None if conversation not found
        """
        conv_record = await self._backend.get_conversation(conversation_id)
        if not conv_record:
            return None
        return await self._backend.get_conversation_stats(conversation_id)

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> AsyncIterator[Message]:
        """Stream messages without loading full conversation.

        Memory-efficient iteration for large conversations.

        Args:
            conversation_id: Conversation ID
            dialogue_only: If True, only yield user/assistant messages
            limit: Maximum messages to yield

        Yields:
            Message objects one at a time
        """
        async for record in self._backend.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            limit=limit,
        ):
            yield Message.from_record(record, attachments=[])

    async def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[Conversation]:
        """Search by semantic similarity.

        Args:
            text: Query text
            limit: Maximum results
            vector_provider: Vector provider for embedding

        Returns:
            List of similar Conversation objects

        Raises:
            ValueError: If no vector provider available
        """
        if not vector_provider:
            raise ValueError("Semantic search requires a vector provider.")

        results = vector_provider.query(text, limit=limit * 3)
        if not results:
            return []

        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = await self._get_message_conversation_mapping(message_ids)

        conv_scores: dict[str, float] = {}
        for msg_id, distance in results:
            conv_id = msg_to_conv.get(msg_id)
            if conv_id:
                # Lower distance = more similar; keep the best (lowest) per conversation
                conv_scores[conv_id] = min(conv_scores.get(conv_id, float("inf")), distance)

        ranked_ids = sorted(
            conv_scores.keys(),
            key=lambda x: conv_scores[x],
        )[:limit]

        return await self._get_many(ranked_ids)

    async def _get_message_conversation_mapping(
        self, message_ids: builtins.list[str]
    ) -> dict[str, str]:
        """Fetch mapping of message IDs to conversation IDs.

        Args:
            message_ids: List of message IDs

        Returns:
            Dict mapping message_id -> conversation_id
        """
        if not message_ids:
            return {}

        placeholders = ",".join("?" * len(message_ids))
        query = f"SELECT message_id, conversation_id FROM messages WHERE message_id IN ({placeholders})"

        async with self._backend._get_connection() as conn:
            cursor = await conn.execute(query, message_ids)
            rows = await cursor.fetchall()

        return {row["message_id"]: row["conversation_id"] for row in rows}

    def filter(self) -> filters.ConversationFilter:
        """Create a filter builder for chainable queries.

        Terminal methods (list, first, count, etc.) are async and must be awaited.

        Returns:
            ConversationFilter instance
        """
        from polylogue.lib import filters

        return filters.ConversationFilter(self)

    # --- Write Methods ---

    async def save_conversation(
        self,
        conversation: Conversation | ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a conversation with its messages and attachments atomically.

        Args:
            conversation: Conversation model or ConversationRecord to save
            messages: List of message records
            attachments: List of attachment records

        Returns:
            Dictionary with counts:
                - conversations: New conversations
                - messages: New messages
                - attachments: New attachments
                - skipped_conversations: Unchanged conversations
                - skipped_messages: Unchanged messages
                - skipped_attachments: Existing attachments
        """
        # Convert Conversation model to ConversationRecord if needed
        if isinstance(conversation, Conversation):
            conv_record = self._conversation_to_record(conversation)
        else:
            conv_record = conversation

        return await self._save_via_backend(conv_record, messages, attachments)

    def _conversation_to_record(self, conversation: Conversation) -> ConversationRecord:
        """Convert a Conversation model to a ConversationRecord.

        Args:
            conversation: Conversation model

        Returns:
            ConversationRecord
        """
        from typing import cast

        from polylogue.types import ContentHash, ConversationId

        created_at_str = conversation.created_at.isoformat() if conversation.created_at else None
        updated_at_str = conversation.updated_at.isoformat() if conversation.updated_at else (created_at_str or None)

        # Try to extract provider_id from canonical id (format: provider:id)
        # Fallback to whole ID if pattern doesn't match
        provider_id = str(conversation.id)
        if ":" in provider_id and conversation.provider:
            prefix = f"{conversation.provider}:"
            if provider_id.startswith(prefix):
                provider_id = provider_id[len(prefix) :]

        return ConversationRecord(
            conversation_id=cast(ConversationId, str(conversation.id)),
            provider_name=conversation.provider,
            provider_conversation_id=provider_id,
            title=conversation.title or "",
            created_at=created_at_str,
            updated_at=updated_at_str,
            content_hash=cast(ContentHash, conversation.metadata.get("content_hash", "")),
            provider_meta=cast(dict[str, object], conversation.metadata.get("provider_meta", {})),
            metadata=conversation.metadata,
        )

    async def _save_via_backend(
        self,
        conversation: ConversationRecord,
        messages: builtins.list[MessageRecord],
        attachments: builtins.list[AttachmentRecord],
    ) -> dict[str, int]:
        """Internal implementation of save_conversation via backend.

        Handles transaction control and skip detection.

        Args:
            conversation: Conversation record
            messages: List of message records
            attachments: List of attachment records

        Returns:
            Dictionary with operation counts
        """
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

        async with backend.transaction():
            # Check existing conversation
            existing = await backend.get_conversation(conversation.conversation_id)
            await backend.save_conversation_record(conversation)
            if existing and existing.content_hash == conversation.content_hash:
                counts["skipped_conversations"] += 1
            else:
                counts["conversations"] += 1

            # Check existing messages
            existing_messages = {
                msg.message_id: msg for msg in await backend.get_messages(conversation.conversation_id)
            }
            for message in messages:
                existing_msg = existing_messages.get(message.message_id)
                if existing_msg and existing_msg.content_hash == message.content_hash:
                    counts["skipped_messages"] += 1
                else:
                    counts["messages"] += 1

            if messages:
                await backend.save_messages(messages)

            # Check existing attachments
            existing_attachments = {
                att.attachment_id: att
                for att in await backend.get_attachments(conversation.conversation_id)
            }
            for attachment in attachments:
                if attachment.attachment_id in existing_attachments:
                    counts["skipped_attachments"] += 1
                else:
                    counts["attachments"] += 1

            new_attachment_ids: set[str] = {str(att.attachment_id) for att in attachments}
            await backend.prune_attachments(conversation.conversation_id, new_attachment_ids)

            if attachments:
                await backend.save_attachments(attachments)

        return counts

    async def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run audit entry.

        Args:
            record: Run record to persist
        """
        await self._backend.record_run(record)

    # --- Metadata CRUD ---

    async def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata dictionary for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Metadata dictionary (empty if not found)
        """
        return await self._backend.get_metadata(conversation_id)

    async def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Set a single metadata key.

        Args:
            conversation_id: Conversation ID
            key: Metadata key
            value: Value to set
        """
        await self._backend.update_metadata(conversation_id, key, value)

    async def delete_metadata(self, conversation_id: str, key: str) -> None:
        """Remove a metadata key.

        Args:
            conversation_id: Conversation ID
            key: Metadata key to delete
        """
        await self._backend.delete_metadata(conversation_id, key)

    async def add_tag(self, conversation_id: str, tag: str) -> None:
        """Add a tag to the conversation's tags list.

        Args:
            conversation_id: Conversation ID
            tag: Tag to add
        """
        await self._backend.add_tag(conversation_id, tag)

    async def remove_tag(self, conversation_id: str, tag: str) -> None:
        """Remove a tag from the conversation's tags list.

        Args:
            conversation_id: Conversation ID
            tag: Tag to remove
        """
        await self._backend.remove_tag(conversation_id, tag)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        """List all tags with counts.

        Args:
            provider: Optional provider filter

        Returns:
            Dict of tag -> count, sorted by count descending
        """
        return await self._backend.list_tags(provider=provider)

    async def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        """Replace entire metadata dictionary.

        Args:
            conversation_id: Conversation ID
            metadata: New metadata dictionary
        """
        await self._backend.set_metadata(conversation_id, metadata)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and all related records.

        Removes conversation, messages, attachment refs, and FTS index entries.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if deleted, False if not found
        """
        return await self._backend.delete_conversation(conversation_id)

    # --- Vector Search Methods ---

    async def embed_conversation(
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

        messages = await self._backend.get_messages(conversation_id)
        if not messages:
            return 0

        vector_provider.upsert(conversation_id, messages)
        return len(messages)

    async def similarity_search(
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

        Raises:
            ValueError: If no vector provider configured
        """
        if vector_provider is None:
            from polylogue.storage.search_providers import create_vector_provider

            vector_provider = create_vector_provider()

        if vector_provider is None:
            raise ValueError("No vector provider configured")

        results = vector_provider.query(query, limit=limit)
        if not results:
            return []

        # Batch lookup conversation IDs
        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = await self._get_message_conversation_mapping(message_ids)

        return [
            (msg_to_conv[msg_id], msg_id, distance)
            for msg_id, distance in results
            if msg_id in msg_to_conv
        ]

    async def get_archive_stats(self) -> ArchiveStats:
        """Get comprehensive archive statistics.

        Returns:
            ArchiveStats with all metrics
        """
        from polylogue.lib.stats import ArchiveStats

        async with self._backend._get_connection() as conn:
            # Total counts
            cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
            conv_count = (await cursor.fetchone())[0]

            cursor = await conn.execute("SELECT COUNT(*) FROM messages")
            msg_count = (await cursor.fetchone())[0]

            cursor = await conn.execute("SELECT COUNT(*) FROM attachments")
            att_count = (await cursor.fetchone())[0]

            # Provider breakdown
            cursor = await conn.execute(
                """
                SELECT provider_name, COUNT(*) as count
                FROM conversations
                GROUP BY provider_name
                """
            )
            provider_rows = await cursor.fetchall()
            providers = {row["provider_name"]: row["count"] for row in provider_rows}

            # Check embedding status if table exists
            embedded_convs = 0
            embedded_msgs = 0
            try:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
                )
                embedded_convs = (await cursor.fetchone())[0]

                cursor = await conn.execute("SELECT COUNT(*) FROM message_embeddings")
                embedded_msgs = (await cursor.fetchone())[0]
            except Exception as exc:
                logger.warning("Embedding stats query failed: %s", exc)

        # Get database size
        db_size = 0
        try:
            from pathlib import Path

            db_size = Path(self._db_path).stat().st_size if self._db_path else 0
        except Exception as exc:
            logger.warning("DB size check failed: %s", exc)

        return ArchiveStats(
            total_conversations=conv_count,
            total_messages=msg_count,
            total_attachments=att_count,
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

    Used by facades and internal migration tools.
    """
    return Conversation.from_records(conversation, messages, attachments)


__all__ = ["ConversationRepository"]
