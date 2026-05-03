"""Archive/message/archive-search query band for SQLiteQueryStore."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)
from polylogue.storage.search.models import ConversationSearchEvidenceHit, ConversationSearchResult
from polylogue.storage.sqlite.queries import attachments as attachments_q
from polylogue.storage.sqlite.queries import conversations as conversations_q
from polylogue.storage.sqlite.queries import messages as messages_q
from polylogue.storage.sqlite.queries import stats as stats_q
from polylogue.storage.sqlite.queries.messages import MessageTypeName
from polylogue.storage.sqlite.queries.stats import (
    AggregateMessageStats,
    ProviderConversationCountRow,
    ProviderMetricsRow,
)


class SQLiteQueryStoreArchiveMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        async with self._connection_factory() as conn:
            return await conversations_q.get_conversation(conn, conversation_id)

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        async with self._connection_factory() as conn:
            return await conversations_q.get_conversations_batch(conn, ids)

    async def list_conversations(
        self,
        request: ConversationRecordQuery,
    ) -> list[ConversationRecord]:
        async with self._connection_factory() as conn:
            return await conversations_q.list_conversations(conn, **request.to_list_kwargs())

    async def list_conversation_summaries(
        self,
        request: ConversationRecordQuery,
    ) -> list[ConversationRecord]:
        async with self._connection_factory() as conn:
            return await conversations_q.list_conversation_summaries(conn, **request.to_list_kwargs())

    async def count_conversations(
        self,
        request: ConversationRecordQuery,
    ) -> int:
        async with self._connection_factory() as conn:
            return await conversations_q.count_conversations(conn, **request.to_count_kwargs())

    async def conversation_exists_by_hash(self, content_hash: str) -> bool:
        async with self._connection_factory() as conn:
            return await conversations_q.conversation_exists_by_hash(conn, content_hash)

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> str | None:
        async with self._connection_factory() as conn:
            return await conversations_q.resolve_id(conn, id_prefix, strict=strict)

    async def search_conversations(self, query: str, limit: int = 100, providers: list[str] | None = None) -> list[str]:
        return (await self.search_conversation_hits(query, limit=limit, providers=providers)).conversation_ids()

    async def search_action_conversations(
        self, query: str, limit: int = 100, providers: list[str] | None = None
    ) -> list[str]:
        return (await self.search_action_conversation_hits(query, limit=limit, providers=providers)).conversation_ids()

    async def search_conversation_hits(
        self,
        query: str,
        limit: int = 100,
        providers: list[str] | None = None,
    ) -> ConversationSearchResult:
        async with self._connection_factory() as conn:
            return await conversations_q.search_conversation_hits(conn, query, limit, providers)

    async def search_conversation_evidence_hits(
        self,
        query: str,
        limit: int = 100,
        providers: list[str] | None = None,
        since: str | None = None,
    ) -> list[ConversationSearchEvidenceHit]:
        async with self._connection_factory() as conn:
            return await conversations_q.search_conversation_evidence_hits(conn, query, limit, providers, since)

    async def search_attachment_identity_evidence_hits(
        self,
        query: str,
        limit: int = 100,
        providers: list[str] | None = None,
        since: str | None = None,
    ) -> list[ConversationSearchEvidenceHit]:
        async with self._connection_factory() as conn:
            return await attachments_q.search_attachment_identity_evidence_hits(conn, query, limit, providers, since)

    async def search_action_conversation_hits(
        self,
        query: str,
        limit: int = 100,
        providers: list[str] | None = None,
    ) -> ConversationSearchResult:
        async with self._connection_factory() as conn:
            return await conversations_q.search_action_conversation_hits(conn, query, limit, providers)

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        async with self._connection_factory() as conn:
            messages = await messages_q.get_messages(conn, conversation_id)
        if not messages:
            return []
        blocks_by_message = await self.get_content_blocks([message.message_id for message in messages])
        return [
            message.model_copy(update={"content_blocks": blocks_by_message.get(message.message_id, [])})
            for message in messages
        ]

    async def get_messages_paginated(
        self,
        conversation_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MessageRecord], int]:
        async with self._connection_factory() as conn:
            messages, total = await messages_q.get_messages_paginated(
                conn,
                conversation_id,
                message_role=message_role,
                message_type=message_type,
                limit=limit,
                offset=offset,
            )
        if not messages:
            return [], total
        blocks_by_message = await self.get_content_blocks([message.message_id for message in messages])
        return [
            message.model_copy(update={"content_blocks": blocks_by_message.get(message.message_id, [])})
            for message in messages
        ], total

    async def get_messages_batch(self, conversation_ids: list[str]) -> dict[str, list[MessageRecord]]:
        if not conversation_ids:
            return {}
        async with self._connection_factory() as conn:
            result, all_messages = await messages_q.get_messages_batch(conn, conversation_ids)
        if not all_messages:
            return result
        blocks_by_message = await self.get_content_blocks([message.message_id for message in all_messages])
        return {
            conversation_id: [
                message.model_copy(update={"content_blocks": blocks_by_message.get(message.message_id, [])})
                for message in records
            ]
            for conversation_id, records in result.items()
        }

    async def get_content_blocks(self, message_ids: list[str]) -> dict[str, list[ContentBlockRecord]]:
        async with self._connection_factory() as conn:
            return await attachments_q.get_content_blocks(conn, message_ids)

    async def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        async with self._connection_factory() as conn:
            return await attachments_q.get_attachments(conn, conversation_id)

    async def get_attachments_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]:
        async with self._connection_factory() as conn:
            return await attachments_q.get_attachments_batch(conn, conversation_ids)

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        async with self._connection_factory() as conn:
            async for record in messages_q.iter_messages(
                conn,
                conversation_id,
                dialogue_only=dialogue_only,
                message_roles=message_roles,
                limit=limit,
            ):
                yield record

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        async with self._connection_factory() as conn:
            return await messages_q.get_conversation_stats(conn, conversation_id)

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]:
        async with self._connection_factory() as conn:
            return await messages_q.get_message_counts_batch(conn, conversation_ids)

    async def aggregate_message_stats(self, conversation_ids: list[str] | None = None) -> AggregateMessageStats:
        async with self._connection_factory() as conn:
            return await stats_q.aggregate_message_stats(conn, conversation_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        async with self._connection_factory() as conn:
            return await stats_q.get_stats_by(conn, group_by)

    async def get_provider_conversation_counts(self) -> list[ProviderConversationCountRow]:
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_conversation_counts(conn)

    async def get_provider_metrics_rows(self) -> list[ProviderMetricsRow]:
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_metrics_rows(conn)

    async def get_last_sync_timestamp(self) -> str | None:
        async with self._connection_factory() as conn:
            return await conversations_q.get_last_sync_timestamp(conn)

    def conversation_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        return conversations_q.conversation_id_query(source_names=source_names)

    async def count_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        async with self._connection_factory() as conn:
            return await conversations_q.count_conversation_ids(conn, source_names=source_names)

    async def iter_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        async with self._connection_factory() as conn:
            async for conversation_id in conversations_q.iter_conversation_ids(
                conn, source_names=source_names, page_size=page_size
            ):
                yield conversation_id
