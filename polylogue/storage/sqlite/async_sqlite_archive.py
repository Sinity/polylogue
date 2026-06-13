"""Archive session/message/query methods for the async SQLite backend.

This mixin (SQLiteArchiveMixin) serves the SQLiteBackend with read AND
write capability. Most reads delegate to self.queries (a SQLiteQueryStore),
while writes use self._get_connection() directly.

Relationship with SQLiteQueryStoreArchiveMixin (query_store_archive.py):

- SQLiteArchiveMixin is the *backend* layer: write-capable, schema-ensured
  connections. Reads usually delegate to the underlying query store.
- SQLiteQueryStoreArchiveMixin is the *query store* layer: read-only, uses
  plain connection factory, implements the SQLiteQueryStore public API.

Intentional divergences (10 known, all architectural):

1. Naming: _session_id_query (private delegate) vs
   session_id_query (public query-store API). Same function, different
   callers.

2. search_sessions: delegates to queries (backend) vs delegates to
   search_session_hits().session_ids() (query store). Same result.

3. get_messages: content_blocks are pre-attached by the query store layer;
   the backend inherits this. The query store does a two-step load+merge
   because it IS the canonical implementation.

4. Connection management: _get_connection() ensures schema before every use
   (backend); _connection_factory provides pre-configured read-only
   connections (query store). Both correct for their context.

5. Write methods (save_session_record, save_messages, etc.) exist only
   on SQLiteArchiveMixin. The query store is deliberately read-only.

6. Query API methods (list_sessions, count_sessions,
   search_action_*, search_session_evidence_hits) exist only on
   SQLiteQueryStoreArchiveMixin. The backend accesses them through
   self.queries (a SQLiteQueryStore instance).

7. get_session_insight_status exists only on SQLiteArchiveMixin. The query
   store has its own implementation in query_store.py.

8. get_messages_batch: both have equivalent behavior. The backend delegates
   to queries; the query store adds an explicit empty-session_ids early
   exit for clarity.

9. iter_messages: the backend has a chunk_size=100 fast path that delegates
   to the query store; both call messages_q.iter_messages underneath.

10. search_session_hits: backend delegates to self.queries; query store
    opens a direct connection. Same destination through different layers.

These divergences reflect the different roles, not bugs. The backend is a
write-capable full DB owner; the query store is a composable read-only
component usable across multiple backend types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.runtime import (
    AttachmentRecord,
    BlockRecord,
    MessageRecord,
    SessionEventRecord,
    SessionRecord,
)
from polylogue.storage.search.models import SessionSearchResult
from polylogue.storage.sqlite.queries import messages as messages_q
from polylogue.storage.sqlite.queries import sessions as sessions_q
from polylogue.storage.sqlite.queries.stats import (
    AggregateMessageStats,
    ProviderMetricsRow,
    ProviderSessionCountRow,
)
from polylogue.storage.sqlite.queries.tool_usage import (
    ToolUsageProviderCoverageRow,
    ToolUsageRow,
)

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.storage.sqlite.queries.messages import MessageTypeName
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class SQLiteArchiveMixin:
    """Session/message/archive-query methods for ``SQLiteBackend``."""

    if TYPE_CHECKING:
        queries: SQLiteQueryStore
        _transaction_depth: int
        _db_path: Path

        def _get_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    async def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve a session by ID."""
        return await self.queries.get_session(session_id)

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]:
        """Retrieve multiple sessions in a single query.

        Preserves the order of input IDs. Missing IDs are silently skipped.
        """
        return await self.queries.get_sessions_batch(ids)

    async def aggregate_message_stats(
        self,
        session_ids: list[str] | None = None,
    ) -> AggregateMessageStats:
        """Compute aggregate message statistics via SQL."""
        return await self.queries.aggregate_message_stats(session_ids)

    async def session_exists_by_hash(self, content_hash: str) -> bool:
        """Check if session with given content hash exists."""
        return await self.queries.session_exists_by_hash(content_hash)

    async def get_messages(self, session_id: str) -> list[MessageRecord]:
        """Get all messages for a session, with content_blocks attached."""
        return await self.queries.get_messages(session_id)

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MessageRecord], int]:
        """Get paginated messages for a session with optional filters.

        Returns (messages, total_count) where total_count is the count of
        messages matching the filters before pagination.
        """
        return await self.queries.get_messages_paginated(
            session_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )

    async def get_messages_batch(
        self,
        session_ids: list[str],
        *,
        sort_key_since: float | None = None,
        sort_key_until: float | None = None,
        message_role: MessageRoleFilter = (),
    ) -> dict[str, list[MessageRecord]]:
        """Get messages for multiple sessions in a single query, with content_blocks."""
        return await self.queries.get_messages_batch(
            session_ids,
            sort_key_since=sort_key_since,
            sort_key_until=sort_key_until,
            message_role=message_role,
        )

    async def get_blocks(self, message_ids: list[str]) -> dict[str, list[BlockRecord]]:
        """Get content blocks for a list of message IDs."""
        return await self.queries.get_blocks(message_ids)

    async def get_attachments(self, session_id: str) -> list[AttachmentRecord]:
        """Get all attachments for a session."""
        return await self.queries.get_attachments(session_id)

    async def get_attachments_batch(self, session_ids: list[str]) -> dict[str, list[AttachmentRecord]]:
        """Get attachments for multiple sessions in a single query."""
        return await self.queries.get_attachments_batch(session_ids)

    async def get_session_events(self, session_id: str) -> list[SessionEventRecord]:
        """Get timeline events for a session."""
        return await self.queries.get_session_events(session_id)

    async def get_session_events_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[SessionEventRecord]]:
        """Get timeline events for multiple sessions."""
        return await self.queries.get_session_events_batch(session_ids)

    async def list_sessions_by_parent(self, parent_id: str) -> list[SessionRecord]:
        """List all sessions that have the given session as parent."""
        async with self._get_connection() as conn:
            return await sessions_q.list_sessions_by_parent(conn, parent_id)

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> str | None:
        """Resolve a partial session ID to a full ID."""
        return await self.queries.resolve_id(id_prefix, strict=strict)

    async def get_last_sync_timestamp(self) -> str | None:
        """Return the timestamp of the most recent ingestion run, or None."""
        return await self.queries.get_last_sync_timestamp()

    def _session_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped session-ID query."""
        return self.queries.session_id_query(source_names=source_names)

    async def count_session_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        """Count session IDs, optionally scoped to source names."""
        return await self.queries.count_session_ids(source_names=source_names)

    async def iter_session_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate session IDs in bounded fetch batches."""
        async for cid in self.queries.iter_session_ids(source_names=source_names, page_size=page_size):
            yield cid

    async def get_session_insight_status(self, *, verify_freshness: bool = True) -> SessionInsightStatusSnapshot:
        """Return materialized session-insight coverage counters."""
        return await self.queries.get_session_insight_status(verify_freshness=verify_freshness)

    async def search_sessions(self, query: str, limit: int = 100, providers: list[str] | None = None) -> list[str]:
        """Search sessions using the canonical ranked FTS session query."""
        return await self.queries.search_sessions(query, limit, providers)

    async def search_session_hits(
        self,
        query: str,
        limit: int = 100,
        providers: list[str] | None = None,
    ) -> SessionSearchResult:
        """Search sessions while preserving ordered session-hit metadata."""
        return await self.queries.search_session_hits(query, limit, providers)

    async def iter_messages(
        self,
        session_id: str,
        *,
        chunk_size: int = 100,
        dialogue_only: bool = False,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        """Stream messages in chunks instead of loading all at once."""
        if chunk_size != 100:
            async with self._get_connection() as conn:
                async for msg in messages_q.iter_messages(
                    conn,
                    session_id,
                    chunk_size=chunk_size,
                    dialogue_only=dialogue_only,
                    message_roles=message_roles,
                    limit=limit,
                ):
                    yield msg
            return
        async for msg in self.queries.iter_messages(
            session_id,
            dialogue_only=dialogue_only,
            message_roles=message_roles,
            limit=limit,
        ):
            yield msg

    async def get_session_stats(self, session_id: str) -> dict[str, int]:
        """Get message counts without loading messages."""
        return await self.queries.get_session_stats(session_id)

    async def get_message_counts_batch(self, session_ids: list[str]) -> dict[str, int]:
        """Get message counts for multiple sessions in a single query."""
        return await self.queries.get_message_counts_batch(session_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        """Get session counts grouped by provider, month, or year."""
        return await self.queries.get_stats_by(group_by)

    async def get_provider_session_counts(self) -> list[ProviderSessionCountRow]:
        """Return session counts per provider."""
        return await self.queries.get_provider_session_counts()

    async def get_provider_metrics_rows(self) -> list[ProviderMetricsRow]:
        """Return raw provider aggregation rows for analytics reporting."""
        return await self.queries.get_provider_metrics_rows()

    async def get_tool_usage_rows(self) -> list[ToolUsageRow]:
        """Return per-(provider, tool, action_kind) tool usage rows."""
        return await self.queries.get_tool_usage_rows()

    async def get_tool_usage_provider_coverage_rows(
        self,
    ) -> list[ToolUsageProviderCoverageRow]:
        """Return per-provider tool-data coverage signals."""
        return await self.queries.get_tool_usage_provider_coverage_rows()


__all__ = ["SQLiteArchiveMixin"]
