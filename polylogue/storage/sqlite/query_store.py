"""Low-level SQLite query store composed from explicit concern bands."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import aiosqlite

from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.query_models import (
    SessionTagRollupListQuery,
    ThreadListQuery,
)
from polylogue.storage.runtime import (
    SessionTagRollupRecord,
    ThreadRecord,
)
from polylogue.storage.sqlite.queries import (
    session_insight_summary_queries as session_insight_summaries_q,
)
from polylogue.storage.sqlite.queries import (
    session_insight_thread_queries as session_insight_threads_q,
)
from polylogue.storage.sqlite.query_store_archive import SQLiteQueryStoreArchiveMixin
from polylogue.storage.sqlite.query_store_insight_profiles import (
    SQLiteQueryStoreInsightProfilesMixin,
)
from polylogue.storage.sqlite.query_store_insight_timelines import (
    SQLiteQueryStoreInsightTimelinesMixin,
)
from polylogue.storage.sqlite.query_store_maintenance import SQLiteQueryStoreMaintenanceMixin


class SQLiteQueryStore(
    SQLiteQueryStoreArchiveMixin,
    SQLiteQueryStoreInsightProfilesMixin,
    SQLiteQueryStoreInsightTimelinesMixin,
    SQLiteQueryStoreMaintenanceMixin,
):
    """Canonical low-level read/query API for SQLite archive state."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]],
    ) -> None:
        self._connection_factory = connection_factory

    # -- Insight status (formerly query_store_insight_status.py) ------------

    async def get_session_insight_status(self, *, verify_freshness: bool = True) -> SessionInsightStatusSnapshot:
        from polylogue.storage.insights.session.status import session_insight_status_async

        async with self._connection_factory() as conn:
            return await session_insight_status_async(conn, verify_freshness=verify_freshness)

    # -- Threads (formerly query_store_insight_threads.py) ------------------

    async def get_thread(self, thread_id: str) -> ThreadRecord | None:
        async with self._connection_factory() as conn:
            return await session_insight_threads_q.get_thread(conn, thread_id)

    async def _list_threads_query(
        self,
        query: ThreadListQuery,
    ) -> list[ThreadRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_threads_q.list_threads(conn, query)

    async def list_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[ThreadRecord]:
        return await self._list_threads_query(
            ThreadListQuery(
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )
        )

    # -- Summaries (formerly query_store_insight_summaries.py) --------------

    async def _list_session_tag_rollup_rows_query(
        self,
        query: SessionTagRollupListQuery,
    ) -> list[SessionTagRollupRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_summaries_q.list_session_tag_rollup_rows(conn, query)

    async def list_session_tag_rollup_rows(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ) -> list[SessionTagRollupRecord]:
        return await self._list_session_tag_rollup_rows_query(
            SessionTagRollupListQuery(
                provider=provider,
                since=since,
                until=until,
                query=query,
            )
        )


__all__ = ["SQLiteQueryStore"]
