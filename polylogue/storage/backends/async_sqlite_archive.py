"""Archive conversation/message/query methods for the async SQLite backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from polylogue.storage.backends.queries import attachments as attachments_q
from polylogue.storage.backends.queries import conversations as conversations_q
from polylogue.storage.backends.queries import messages as messages_q
from polylogue.storage.backends.queries.stats import AggregateMessageStats
from polylogue.storage.session_product_runtime import SessionProductStatusSnapshot
from polylogue.storage.store import AttachmentRecord, ContentBlockRecord, ConversationRecord, MessageRecord

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.storage.backends.query_store import SQLiteQueryStore


class SQLiteArchiveMixin:
    """Conversation/message/archive-query methods for ``SQLiteBackend``."""

    if TYPE_CHECKING:
        queries: SQLiteQueryStore
        _transaction_depth: int

        def _get_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID."""
        return await self.queries.get_conversation(conversation_id)

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        """Retrieve multiple conversations in a single query.

        Preserves the order of input IDs. Missing IDs are silently skipped.
        """
        return await self.queries.get_conversations_batch(ids)

    async def aggregate_message_stats(
        self,
        conversation_ids: list[str] | None = None,
    ) -> AggregateMessageStats:
        """Compute aggregate message statistics via SQL."""
        return await self.queries.aggregate_message_stats(conversation_ids)

    async def conversation_exists_by_hash(self, content_hash: str) -> bool:
        """Check if conversation with given content hash exists."""
        return await self.queries.conversation_exists_by_hash(content_hash)

    async def save_conversation_record(self, record: ConversationRecord) -> None:
        """Persist a conversation record with upsert semantics."""
        async with self._get_connection() as conn:
            await conversations_q.save_conversation_record(conn, record, self._transaction_depth)

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Get all messages for a conversation, with content_blocks attached."""
        return await self.queries.get_messages(conversation_id)

    async def get_messages_batch(self, conversation_ids: list[str]) -> dict[str, list[MessageRecord]]:
        """Get messages for multiple conversations in a single query, with content_blocks."""
        return await self.queries.get_messages_batch(conversation_ids)

    @staticmethod
    def _topo_sort_messages(records: list[MessageRecord]) -> list[MessageRecord]:
        """Sort messages so parents come before children (for FK constraint)."""
        return messages_q.topo_sort_messages(records)

    async def save_messages(self, records: list[MessageRecord]) -> None:
        """Persist multiple message records using bulk insert."""
        async with self._get_connection() as conn:
            await messages_q.save_messages(conn, records, self._transaction_depth)

    async def save_content_blocks(self, records: list[ContentBlockRecord]) -> None:
        """Persist content block records using bulk insert."""
        async with self._get_connection() as conn:
            await attachments_q.save_content_blocks(conn, records, self._transaction_depth)

    async def get_content_blocks(self, message_ids: list[str]) -> dict[str, list[ContentBlockRecord]]:
        """Get content blocks for a list of message IDs."""
        return await self.queries.get_content_blocks(message_ids)

    async def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Get all attachments for a conversation."""
        return await self.queries.get_attachments(conversation_id)

    async def get_attachments_batch(self, conversation_ids: list[str]) -> dict[str, list[AttachmentRecord]]:
        """Get attachments for multiple conversations in a single query."""
        return await self.queries.get_attachments_batch(conversation_ids)

    async def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records with reference counting."""
        async with self._get_connection() as conn:
            await attachments_q.save_attachments(conn, records, self._transaction_depth)

    async def prune_attachments(self, conversation_id: str, keep_attachment_ids: set[str]) -> None:
        """Remove attachment refs not in keep set and clean up orphaned attachments."""
        async with self._get_connection() as conn:
            await attachments_q.prune_attachments(conn, conversation_id, keep_attachment_ids, self._transaction_depth)

    async def list_conversations_by_parent(self, parent_id: str) -> list[ConversationRecord]:
        """List all conversations that have the given conversation as parent."""
        async with self._get_connection() as conn:
            return await conversations_q.list_conversations_by_parent(conn, parent_id)

    async def resolve_id(self, id_prefix: str) -> str | None:
        """Resolve a partial conversation ID to a full ID."""
        return await self.queries.resolve_id(id_prefix)

    async def get_last_sync_timestamp(self) -> str | None:
        """Return the timestamp of the most recent ingestion run, or None."""
        return await self.queries.get_last_sync_timestamp()

    def _conversation_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped conversation-ID query."""
        return self.queries.conversation_id_query(source_names=source_names)

    async def count_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        """Count conversation IDs, optionally scoped to source names."""
        return await self.queries.count_conversation_ids(source_names=source_names)

    async def iter_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate conversation IDs in bounded fetch batches."""
        async for cid in self.queries.iter_conversation_ids(source_names=source_names, page_size=page_size):
            yield cid

    async def get_session_product_status(self) -> SessionProductStatusSnapshot:
        """Return materialized session-product coverage counters."""
        return await self.queries.get_session_product_status()

    async def search_conversations(self, query: str, limit: int = 100, providers: list[str] | None = None) -> list[str]:
        """Search conversations using the canonical ranked FTS conversation query."""
        return await self.queries.search_conversations(query, limit, providers)

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        chunk_size: int = 100,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        """Stream messages in chunks instead of loading all at once."""
        if chunk_size != 100:
            async with self._get_connection() as conn:
                async for msg in messages_q.iter_messages(
                    conn,
                    conversation_id,
                    chunk_size=chunk_size,
                    dialogue_only=dialogue_only,
                    limit=limit,
                ):
                    yield msg
            return
        async for msg in self.queries.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            limit=limit,
        ):
            yield msg

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Get message counts without loading messages."""
        return await self.queries.get_conversation_stats(conversation_id)

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]:
        """Get message counts for multiple conversations in a single query."""
        return await self.queries.get_message_counts_batch(conversation_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        """Get conversation counts grouped by provider, month, or year."""
        return await self.queries.get_stats_by(group_by)

    async def get_provider_conversation_counts(self) -> list[dict[str, object]]:
        """Return conversation counts per provider."""
        return await self.queries.get_provider_conversation_counts()

    async def get_provider_metrics_rows(self) -> list[dict[str, object]]:
        """Return raw provider aggregation rows for analytics reporting."""
        return await self.queries.get_provider_metrics_rows()


__all__ = ["SQLiteArchiveMixin"]
