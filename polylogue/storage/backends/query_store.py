"""Low-level read/query surface for the SQLite archive backend."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager

import aiosqlite

from polylogue.storage.backends.queries import (
    attachments as attachments_q,
)
from polylogue.storage.backends.queries import (
    conversations as conversations_q,
)
from polylogue.storage.backends.queries import (
    messages as messages_q,
)
from polylogue.storage.backends.queries import (
    publications as publications_q,
)
from polylogue.storage.backends.queries import (
    raw as raw_queries,
)
from polylogue.storage.backends.queries import (
    runs as runs_q,
)
from polylogue.storage.backends.queries import (
    stats as stats_q,
)
from polylogue.storage.store import (
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    PublicationRecord,
    RunRecord,
)


class SQLiteQueryStore:
    """Canonical low-level read/query API for SQLite archive state."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]],
    ) -> None:
        self._connection_factory = connection_factory

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation record by ID."""
        async with self._connection_factory() as conn:
            return await conversations_q.get_conversation(conn, conversation_id)

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        """Retrieve multiple conversation records in input order."""
        async with self._connection_factory() as conn:
            return await conversations_q.get_conversations_batch(conn, ids)

    async def list_conversations(
        self,
        *,
        source: str | None = None,
        provider: str | None = None,
        providers: list[str] | None = None,
        parent_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        has_file_ops: bool = False,
        has_git_ops: bool = False,
        has_subagent: bool = False,
    ) -> list[ConversationRecord]:
        """List conversation records with filtering and pagination."""
        async with self._connection_factory() as conn:
            return await conversations_q.list_conversations(
                conn,
                source=source,
                provider=provider,
                providers=providers,
                parent_id=parent_id,
                since=since,
                until=until,
                title_contains=title_contains,
                limit=limit,
                offset=offset,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                has_file_ops=has_file_ops,
                has_git_ops=has_git_ops,
                has_subagent=has_subagent,
            )

    async def count_conversations(
        self,
        *,
        source: str | None = None,
        provider: str | None = None,
        providers: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        has_file_ops: bool = False,
        has_git_ops: bool = False,
        has_subagent: bool = False,
    ) -> int:
        """Count conversation records matching the given filters."""
        async with self._connection_factory() as conn:
            return await conversations_q.count_conversations(
                conn,
                source=source,
                provider=provider,
                providers=providers,
                since=since,
                until=until,
                title_contains=title_contains,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                has_file_ops=has_file_ops,
                has_git_ops=has_git_ops,
                has_subagent=has_subagent,
            )

    async def conversation_exists_by_hash(self, content_hash: str) -> bool:
        """Check whether a conversation with the given content hash exists."""
        async with self._connection_factory() as conn:
            return await conversations_q.conversation_exists_by_hash(conn, content_hash)

    async def resolve_id(self, id_prefix: str) -> str | None:
        """Resolve a partial conversation ID to a full ID."""
        async with self._connection_factory() as conn:
            return await conversations_q.resolve_id(conn, id_prefix)

    async def search_conversations(
        self, query: str, limit: int = 100, providers: list[str] | None = None
    ) -> list[str]:
        """Return ranked conversation IDs for the given search query."""
        async with self._connection_factory() as conn:
            return await conversations_q.search_conversations(conn, query, limit, providers)

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Get message records for a conversation with content blocks attached."""
        async with self._connection_factory() as conn:
            messages = await messages_q.get_messages(conn, conversation_id)
        if not messages:
            return []
        blocks_by_message = await self.get_content_blocks([message.message_id for message in messages])
        return [
            message.model_copy(
                update={"content_blocks": blocks_by_message.get(message.message_id, [])}
            )
            for message in messages
        ]

    async def get_messages_batch(self, conversation_ids: list[str]) -> dict[str, list[MessageRecord]]:
        """Get message records for multiple conversations with content blocks attached."""
        if not conversation_ids:
            return {}
        async with self._connection_factory() as conn:
            result, all_messages = await messages_q.get_messages_batch(conn, conversation_ids)
        if not all_messages:
            return result
        blocks_by_message = await self.get_content_blocks([message.message_id for message in all_messages])
        return {
            conversation_id: [
                message.model_copy(
                    update={"content_blocks": blocks_by_message.get(message.message_id, [])}
                )
                for message in records
            ]
            for conversation_id, records in result.items()
        }

    async def get_content_blocks(
        self, message_ids: list[str]
    ) -> dict[str, list[ContentBlockRecord]]:
        """Get content block records keyed by message ID."""
        async with self._connection_factory() as conn:
            return await attachments_q.get_content_blocks(conn, message_ids)

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Get lightweight message statistics for one conversation."""
        async with self._connection_factory() as conn:
            return await messages_q.get_conversation_stats(conn, conversation_id)

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]:
        """Get total message counts for multiple conversations."""
        async with self._connection_factory() as conn:
            return await messages_q.get_message_counts_batch(conn, conversation_ids)

    async def aggregate_message_stats(
        self, conversation_ids: list[str] | None = None
    ) -> dict[str, int]:
        """Compute archive-wide or scoped aggregate message statistics."""
        async with self._connection_factory() as conn:
            return await stats_q.aggregate_message_stats(conn, conversation_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        """Get conversation counts grouped by provider, month, or year."""
        async with self._connection_factory() as conn:
            return await stats_q.get_stats_by(conn, group_by)

    async def get_provider_conversation_counts(self) -> list[dict[str, object]]:
        """Return archive conversation counts grouped by provider."""
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_conversation_counts(conn)

    async def get_provider_metrics_rows(self) -> list[dict[str, object]]:
        """Return raw per-provider metrics rows."""
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_metrics_rows(conn)

    async def get_last_sync_timestamp(self) -> str | None:
        """Return the timestamp of the latest ingestion run, if any."""
        async with self._connection_factory() as conn:
            return await conversations_q.get_last_sync_timestamp(conn)

    def conversation_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped conversation-ID query."""
        return conversations_q.conversation_id_query(source_names=source_names)

    async def count_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        """Count conversation IDs within the optional source scope."""
        async with self._connection_factory() as conn:
            return await conversations_q.count_conversation_ids(conn, source_names=source_names)

    async def iter_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate scoped conversation IDs in bounded batches."""
        async with self._connection_factory() as conn:
            async for conversation_id in conversations_q.iter_conversation_ids(
                conn, source_names=source_names, page_size=page_size
            ):
                yield conversation_id

    def raw_id_query(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped raw-ID query."""
        return raw_queries.raw_id_query(
            source_names=source_names,
            provider_name=provider_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
        )

    async def iter_raw_ids(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate raw conversation IDs for a processing slice."""
        async with self._connection_factory() as conn:
            async for raw_id in raw_queries.iter_raw_ids(
                conn,
                source_names=source_names,
                provider_name=provider_name,
                require_unparsed=require_unparsed,
                require_unvalidated=require_unvalidated,
                validation_statuses=validation_statuses,
                page_size=page_size,
            ):
                yield raw_id

    async def get_known_source_mtimes(self) -> dict[str, str]:
        """Return persisted source mtimes keyed by source path."""
        async with self._connection_factory() as conn:
            return await raw_queries.get_known_source_mtimes(conn)

    async def get_latest_run(self) -> RunRecord | None:
        """Fetch the most recent pipeline run record."""
        async with self._connection_factory() as conn:
            return await runs_q.get_latest_run(conn)

    async def get_latest_publication(
        self,
        publication_kind: str,
    ) -> PublicationRecord | None:
        """Fetch the most recent publication record for one publication kind."""
        async with self._connection_factory() as conn:
            return await publications_q.get_latest_publication(conn, publication_kind)


__all__ = ["SQLiteQueryStore"]
