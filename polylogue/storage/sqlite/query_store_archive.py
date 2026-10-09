"""Archive/message/archive-search query band for SQLiteQueryStore."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.runtime import (
    AttachmentRecord,
    BlockRecord,
    FileEditRecord,
    LineageCompleteness,
    MessageRecord,
    SessionCommitRecord,
    SessionEventRecord,
    SessionRecord,
    SessionRefRecord,
    WebContentConstructRecord,
)
from polylogue.storage.search.models import SessionSearchEvidenceRow, SessionSearchResult
from polylogue.storage.sqlite.archive_tiers.write import ArchiveAgentPolicy
from polylogue.storage.sqlite.queries import attachments as attachments_q
from polylogue.storage.sqlite.queries import file_edits as file_edits_q
from polylogue.storage.sqlite.queries import messages as messages_q
from polylogue.storage.sqlite.queries import session_agent_policies as session_agent_policies_q
from polylogue.storage.sqlite.queries import session_commits as session_commits_q
from polylogue.storage.sqlite.queries import session_events as session_events_q
from polylogue.storage.sqlite.queries import session_links as session_links_q
from polylogue.storage.sqlite.queries import session_refs as session_refs_q
from polylogue.storage.sqlite.queries import sessions as sessions_q
from polylogue.storage.sqlite.queries import stats as stats_q
from polylogue.storage.sqlite.queries import tool_usage as tool_usage_q
from polylogue.storage.sqlite.queries import web_content_constructs as web_content_constructs_q
from polylogue.storage.sqlite.queries.messages import MaterialOriginFilter, MessageTypeName
from polylogue.storage.sqlite.queries.stats import (
    AggregateMessageStats,
    OriginMetricsRow,
    OriginSessionCountRow,
)
from polylogue.storage.sqlite.queries.tool_usage import (
    ToolUsageOriginCoverageRow,
    ToolUsageRow,
)


def _hydrate_message_text_from_blocks(message: MessageRecord) -> None:
    """Reconstruct aggregate display text when storage keeps blocks canonical."""
    if message.text:
        return
    parts = [block.text for block in message.blocks if block.text]
    if parts:
        message.text = "\n".join(parts)


@asynccontextmanager
async def _message_snapshot(conn: aiosqlite.Connection) -> AsyncIterator[None]:
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        async with conn.execute("BEGIN DEFERRED"):
            pass
    try:
        yield
    finally:
        if owns_snapshot:
            async with conn.execute("ROLLBACK"):
                pass


async def _hydrate_message_rows(conn: aiosqlite.Connection, messages: list[MessageRecord]) -> None:
    ids: list[str] = [message.message_id for message in messages]
    blocks = await attachments_q.get_blocks(conn, ids)
    attachments = await attachments_q.get_message_attachments(conn, ids)
    for message in messages:
        message.blocks = blocks.get(message.message_id, [])
        message.attachments = attachments.get(message.message_id, [])
        _hydrate_message_text_from_blocks(message)


class SQLiteQueryStoreArchiveMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def get_session_tags_batch(self, session_ids: list[str]) -> dict[str, tuple[str, ...]]:
        """Read tags through the same connection owner as dependent metadata."""
        if not session_ids:
            return {}
        result: dict[str, list[str]] = {session_id: [] for session_id in session_ids}
        async with self._connection_factory() as conn:
            for table in ("session_tags", "tags"):
                async with conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
                ) as cursor:
                    if await cursor.fetchone() is None:
                        return dict.fromkeys(session_ids, ())
            placeholders = ",".join("?" for _ in session_ids)
            async with conn.execute(
                f"SELECT ct.session_id,t.name FROM session_tags ct JOIN tags t ON t.id=ct.tag_id "
                f"WHERE ct.session_id IN ({placeholders}) ORDER BY t.name",
                session_ids,
            ) as cursor:
                for row in await cursor.fetchall():
                    result[str(row[0])].append(str(row[1]))
        return {session_id: tuple(names) for session_id, names in result.items()}

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with self._connection_factory() as conn:
            return await sessions_q.get_session(conn, session_id)

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]:
        async with self._connection_factory() as conn:
            return await sessions_q.get_sessions_batch(conn, ids)

    async def list_session_links_for_session(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, object]]:
        async with self._connection_factory() as conn:
            return await session_links_q.list_session_links_for_session(conn, session_id, limit=limit)

    async def list_session_links_to_session(self, session_id: str, *, limit: int | None) -> list[dict[str, object]]:
        async with self._connection_factory() as conn:
            return await session_links_q.list_session_links_to_session(conn, session_id, limit=limit)

    async def list_session_links(self) -> list[dict[str, object]]:
        """Return the canonical relation used by the topology projection."""
        async with self._connection_factory() as conn:
            return await session_links_q.list_session_links(conn)

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

    async def count_actions(self, *, origin: str | None = None) -> int:
        async with self._connection_factory() as conn:
            return await sessions_q.count_actions(conn, origin=origin)

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
        async with self._connection_factory() as conn, _message_snapshot(conn):
            messages = await messages_q.get_messages(conn, session_id)
            await _hydrate_message_rows(conn, messages)
            return messages

    async def get_effective_context(self, session_id: str, at_position: int | None = None) -> list[MessageRecord]:
        async with self._connection_factory() as conn, _message_snapshot(conn):
            messages = await messages_q.get_effective_context(conn, session_id, at_position)
            await _hydrate_message_rows(conn, messages)
            return messages

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MessageRecord], int, LineageCompleteness]:
        async with self._connection_factory() as conn, _message_snapshot(conn):
            messages, total, completeness = await messages_q.get_messages_paginated(
                conn,
                session_id,
                message_role=message_role,
                message_type=message_type,
                limit=limit,
                offset=offset,
            )
            await _hydrate_message_rows(conn, messages)
            return messages, total, completeness

    async def get_lineage_completeness(self, session_id: str) -> LineageCompleteness:
        async with self._connection_factory() as conn:
            return await messages_q.get_lineage_completeness(conn, session_id)

    async def get_message_edge_windows(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        material_origin: MaterialOriginFilter | None = None,
        edge_limit: int = 8,
    ) -> tuple[list[MessageRecord], list[MessageRecord], int]:
        async with self._connection_factory() as conn, _message_snapshot(conn):
            first, last, total = await messages_q.get_message_edge_windows(
                conn,
                session_id,
                message_role=message_role,
                message_type=message_type,
                material_origin=material_origin,
                edge_limit=edge_limit,
            )
            messages = [*first, *last]
            await _hydrate_message_rows(conn, messages)
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
        async with self._connection_factory() as conn, _message_snapshot(conn):
            result, all_messages = await messages_q.get_messages_batch(
                conn,
                session_ids,
                sort_key_since=sort_key_since,
                sort_key_until=sort_key_until,
                message_role=message_role,
            )
            await _hydrate_message_rows(conn, all_messages)
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

    async def get_attachment_library_page(
        self,
        *,
        limit: int,
        offset: int,
        mime_filter: str = "",
        session_filter: str = "",
        state_filter: str = "",
        blob_store: BlobStore | None = None,
    ) -> list[tuple[AttachmentRecord, str, str | None]]:
        async with self._connection_factory() as conn, _message_snapshot(conn):
            return await attachments_q.get_attachment_library_page(
                conn,
                limit=limit,
                offset=offset,
                mime_filter=mime_filter,
                session_filter=session_filter,
                state_filter=state_filter,
                blob_store=blob_store,
            )

    async def get_session_events(self, session_id: str) -> list[SessionEventRecord]:
        async with self._connection_factory() as conn:
            return await session_events_q.get_session_events(conn, session_id)

    async def get_session_events_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[SessionEventRecord]]:
        async with self._connection_factory() as conn:
            return await session_events_q.get_session_events_batch(conn, session_ids)

    async def get_session_agent_policies(self, session_id: str) -> list[ArchiveAgentPolicy]:
        async with self._connection_factory() as conn:
            return await session_agent_policies_q.get_session_agent_policies(conn, session_id)

    async def get_session_agent_policies_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[ArchiveAgentPolicy]]:
        async with self._connection_factory() as conn:
            return await session_agent_policies_q.get_session_agent_policies_batch(conn, session_ids)

    async def get_file_edits_for_session(self, session_id: str) -> list[FileEditRecord]:
        async with self._connection_factory() as conn:
            return await file_edits_q.get_file_edits_for_session(conn, session_id)

    async def get_file_edits_for_session_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[FileEditRecord]]:
        async with self._connection_factory() as conn:
            return await file_edits_q.get_file_edits_for_session_batch(conn, session_ids)

    async def get_session_refs(self, session_id: str) -> list[SessionRefRecord]:
        async with self._connection_factory() as conn:
            return await session_refs_q.get_session_refs(conn, session_id)

    async def get_session_refs_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[SessionRefRecord]]:
        async with self._connection_factory() as conn:
            return await session_refs_q.get_session_refs_batch(conn, session_ids)

    async def get_session_commits(self, session_id: str) -> list[SessionCommitRecord]:
        async with self._connection_factory() as conn:
            return await session_commits_q.get_session_commits(conn, session_id)

    async def get_web_content_constructs(
        self,
        session_id: str,
        *,
        construct_type: str | None = None,
    ) -> list[WebContentConstructRecord]:
        async with self._connection_factory() as conn:
            return await web_content_constructs_q.get_web_content_constructs_for_session(
                conn, session_id, construct_type=construct_type
            )

    async def get_web_content_constructs_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[WebContentConstructRecord]]:
        async with self._connection_factory() as conn:
            return await web_content_constructs_q.get_web_content_constructs_for_session_batch(conn, session_ids)

    async def iter_messages(
        self,
        session_id: str,
        *,
        chunk_size: int = 100,
        message_roles: MessageRoleFilter = (),
        material_origin: MaterialOriginFilter | None = None,
        limit: int | None = None,
    ) -> AsyncGenerator[MessageRecord, None]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        # Hydrate one bounded page before yielding, using the same held
        # connection as the message stream rather than one query per row.
        async with (
            self._connection_factory() as conn,
            _message_snapshot(conn),
            aclosing(
                messages_q.iter_messages(
                    conn,
                    session_id,
                    chunk_size=chunk_size,
                    message_roles=message_roles,
                    material_origin=material_origin,
                    limit=limit,
                )
            ) as records,
        ):
            batch: list[MessageRecord] = []
            async for record in records:
                batch.append(record)
                if len(batch) == chunk_size:
                    await _hydrate_message_rows(conn, batch)
                    for row in batch:
                        yield row
                    batch.clear()
            if batch:
                await _hydrate_message_rows(conn, batch)
                for row in batch:
                    yield row

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

    async def get_origin_session_counts(self) -> list[OriginSessionCountRow]:
        async with self._connection_factory() as conn:
            return await stats_q.get_origin_session_counts(conn)

    async def get_origin_metrics_rows(self) -> list[OriginMetricsRow]:
        async with self._connection_factory() as conn:
            return await stats_q.get_origin_metrics_rows(conn)

    async def get_tool_usage_rows(self) -> list[ToolUsageRow]:
        async with self._connection_factory() as conn:
            return await tool_usage_q.get_tool_usage_rows(conn)

    async def get_tool_usage_origin_coverage_rows(
        self,
    ) -> list[ToolUsageOriginCoverageRow]:
        async with self._connection_factory() as conn:
            return await tool_usage_q.get_tool_usage_origin_coverage_rows(conn)

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
