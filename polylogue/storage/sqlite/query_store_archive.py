"""Archive/message/archive-search query band for SQLiteQueryStore."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.runtime import (
    AttachmentRecord,
    BlockRecord,
    MessageRecord,
    SessionEventRecord,
    SessionRecord,
)
from polylogue.storage.search.models import SessionSearchEvidenceRow, SessionSearchResult
from polylogue.storage.sqlite.queries import attachments as attachments_q
from polylogue.storage.sqlite.queries import messages as messages_q
from polylogue.storage.sqlite.queries import session_events as session_events_q
from polylogue.storage.sqlite.queries import session_links as session_links_q
from polylogue.storage.sqlite.queries import sessions as sessions_q
from polylogue.storage.sqlite.queries import stats as stats_q
from polylogue.storage.sqlite.queries import tool_usage as tool_usage_q
from polylogue.storage.sqlite.queries.messages import MaterialOriginFilter, MessageTypeName
from polylogue.storage.sqlite.queries.stats import (
    AggregateMessageStats,
    ProviderMetricsRow,
    ProviderSessionCountRow,
)
from polylogue.storage.sqlite.queries.tool_usage import (
    ToolUsageProviderCoverageRow,
    ToolUsageRow,
)


def _hydrate_message_text_from_blocks(message: MessageRecord) -> None:
    """Reconstruct aggregate display text when storage keeps blocks canonical."""
    if message.text:
        return
    parts = [block.text for block in message.blocks if block.text]
    if parts:
        message.text = "\n".join(parts)


class SQLiteQueryStoreArchiveMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with self._connection_factory() as conn:
            return await sessions_q.get_session(conn, session_id)

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]:
        async with self._connection_factory() as conn:
            return await sessions_q.get_sessions_batch(conn, ids)

    async def list_session_links_for_session(self, session_id: str) -> list[dict[str, object]]:
        async with self._connection_factory() as conn:
            return await session_links_q.list_session_links_for_session(conn, session_id)

    async def list_sessions(
        self,
        request: SessionRecordQuery,
    ) -> list[SessionRecord]:
        async with self._connection_factory() as conn:
            return await sessions_q.list_sessions(conn, **request.to_list_kwargs())

    async def list_session_summaries(
        self,
        request: SessionRecordQuery,
    ) -> list[SessionRecord]:
        async with self._connection_factory() as conn:
            return await sessions_q.list_session_summaries(conn, **request.to_list_kwargs())

    async def count_sessions(
        self,
        request: SessionRecordQuery,
    ) -> int:
        async with self._connection_factory() as conn:
            return await sessions_q.count_sessions(conn, **request.to_count_kwargs())

    async def session_exists_by_hash(self, content_hash: str) -> bool:
        async with self._connection_factory() as conn:
            return await sessions_q.session_exists_by_hash(conn, content_hash)

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> str | None:
        async with self._connection_factory() as conn:
            return await sessions_q.resolve_id(conn, id_prefix, strict=strict)

    async def search_sessions(self, query: str, limit: int = 100, origins: list[str] | None = None) -> list[str]:
        return (await self.search_session_hits(query, limit=limit, origins=origins)).session_ids()

    async def search_action_sessions(self, query: str, limit: int = 100, origins: list[str] | None = None) -> list[str]:
        return (await self.search_action_session_hits(query, limit=limit, origins=origins)).session_ids()

    async def search_session_hits(
        self,
        query: str,
        limit: int = 100,
        origins: list[str] | None = None,
    ) -> SessionSearchResult:
        async with self._connection_factory() as conn:
            return await sessions_q.search_session_hits(conn, query, limit, origins)

    async def search_session_evidence_hits(
        self,
        query: str,
        limit: int = 100,
        origins: list[str] | None = None,
        since: str | None = None,
    ) -> list[SessionSearchEvidenceRow]:
        async with self._connection_factory() as conn:
            return await sessions_q.search_session_evidence_hits(conn, query, limit, origins, since)

    async def search_attachment_identity_evidence_hits(
        self,
        query: str,
        limit: int = 100,
        origins: list[str] | None = None,
        since: str | None = None,
    ) -> list[SessionSearchEvidenceRow]:
        async with self._connection_factory() as conn:
            return await attachments_q.search_attachment_identity_evidence_hits(conn, query, limit, origins, since)

    async def search_action_session_hits(
        self,
        query: str,
        limit: int = 100,
        origins: list[str] | None = None,
    ) -> SessionSearchResult:
        async with self._connection_factory() as conn:
            return await sessions_q.search_action_session_hits(conn, query, limit, origins)

    async def get_messages(self, session_id: str) -> list[MessageRecord]:
        async with self._connection_factory() as conn:
            messages = await messages_q.get_messages(conn, session_id)
        if not messages:
            return []
        blocks_by_message = await self.get_blocks([message.message_id for message in messages])
        # In-place attachment avoids constructing a second pydantic instance
        # per message in the hot hydration path (#1314). The MessageRecord
        # instances were just constructed by _row_to_message and aren't shared.
        for message in messages:
            message.blocks = blocks_by_message.get(message.message_id, [])
            _hydrate_message_text_from_blocks(message)
        return messages

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MessageRecord], int]:
        async with self._connection_factory() as conn:
            messages, total = await messages_q.get_messages_paginated(
                conn,
                session_id,
                message_role=message_role,
                message_type=message_type,
                limit=limit,
                offset=offset,
            )
        if not messages:
            return [], total
        blocks_by_message = await self.get_blocks([message.message_id for message in messages])
        for message in messages:
            message.blocks = blocks_by_message.get(message.message_id, [])
            _hydrate_message_text_from_blocks(message)
        return messages, total

    async def get_message_edge_windows(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        material_origin: MaterialOriginFilter | None = None,
        edge_limit: int = 8,
    ) -> tuple[list[MessageRecord], list[MessageRecord], int]:
        async with self._connection_factory() as conn:
            first, last, total = await messages_q.get_message_edge_windows(
                conn,
                session_id,
                message_role=message_role,
                message_type=message_type,
                material_origin=material_origin,
                edge_limit=edge_limit,
            )
        messages = [*first, *last]
        if not messages:
            return first, last, total
        blocks_by_message = await self.get_blocks([message.message_id for message in messages])
        for message in messages:
            message.blocks = blocks_by_message.get(message.message_id, [])
            _hydrate_message_text_from_blocks(message)
        return first, last, total

    async def get_messages_batch(
        self,
        session_ids: list[str],
        *,
        sort_key_since: float | None = None,
        sort_key_until: float | None = None,
        message_role: MessageRoleFilter = (),
    ) -> dict[str, list[MessageRecord]]:
        if not session_ids:
            return {}
        async with self._connection_factory() as conn:
            result, all_messages = await messages_q.get_messages_batch(
                conn,
                session_ids,
                sort_key_since=sort_key_since,
                sort_key_until=sort_key_until,
                message_role=message_role,
            )
        if not all_messages:
            return result
        blocks_by_message = await self.get_blocks([message.message_id for message in all_messages])
        for message in all_messages:
            message.blocks = blocks_by_message.get(message.message_id, [])
            _hydrate_message_text_from_blocks(message)
        return result

    async def get_blocks(self, message_ids: list[str]) -> dict[str, list[BlockRecord]]:
        async with self._connection_factory() as conn:
            return await attachments_q.get_blocks(conn, message_ids)

    async def get_attachments(self, session_id: str) -> list[AttachmentRecord]:
        async with self._connection_factory() as conn:
            return await attachments_q.get_attachments(conn, session_id)

    async def get_attachments_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]:
        async with self._connection_factory() as conn:
            return await attachments_q.get_attachments_batch(conn, session_ids)

    async def get_session_events(self, session_id: str) -> list[SessionEventRecord]:
        async with self._connection_factory() as conn:
            return await session_events_q.get_session_events(conn, session_id)

    async def get_session_events_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[SessionEventRecord]]:
        async with self._connection_factory() as conn:
            return await session_events_q.get_session_events_batch(conn, session_ids)

    async def iter_messages(
        self,
        session_id: str,
        *,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        async with self._connection_factory() as conn:
            async for record in messages_q.iter_messages(
                conn,
                session_id,
                message_roles=message_roles,
                limit=limit,
            ):
                yield record

    async def get_session_stats(self, session_id: str) -> dict[str, int]:
        async with self._connection_factory() as conn:
            return await messages_q.get_session_stats(conn, session_id)

    async def get_message_counts_batch(self, session_ids: list[str]) -> dict[str, int]:
        async with self._connection_factory() as conn:
            return await messages_q.get_message_counts_batch(conn, session_ids)

    async def aggregate_message_stats(self, session_ids: list[str] | None = None) -> AggregateMessageStats:
        async with self._connection_factory() as conn:
            return await stats_q.aggregate_message_stats(conn, session_ids)

    async def get_stats_by(self, group_by: str = "origin") -> dict[str, int]:
        async with self._connection_factory() as conn:
            return await stats_q.get_stats_by(conn, group_by)

    async def get_provider_session_counts(self) -> list[ProviderSessionCountRow]:
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_session_counts(conn)

    async def get_provider_metrics_rows(self) -> list[ProviderMetricsRow]:
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_metrics_rows(conn)

    async def get_tool_usage_rows(self) -> list[ToolUsageRow]:
        async with self._connection_factory() as conn:
            return await tool_usage_q.get_tool_usage_rows(conn)

    async def get_tool_usage_provider_coverage_rows(
        self,
    ) -> list[ToolUsageProviderCoverageRow]:
        async with self._connection_factory() as conn:
            return await tool_usage_q.get_tool_usage_provider_coverage_rows(conn)

    async def get_last_sync_timestamp(self) -> str | None:
        async with self._connection_factory() as conn:
            return await sessions_q.get_last_sync_timestamp(conn)

    def session_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        return sessions_q.session_id_query(source_names=source_names)

    async def count_session_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        async with self._connection_factory() as conn:
            return await sessions_q.count_session_ids(conn, source_names=source_names)

    async def iter_session_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        async with self._connection_factory() as conn:
            async for session_id in sessions_q.iter_session_ids(conn, source_names=source_names, page_size=page_size):
                yield session_id
