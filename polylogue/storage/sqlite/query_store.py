"""Low-level SQLite query store composed from explicit concern bands."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import aiosqlite

from polylogue.storage.derived.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.runtime import (
    ThreadRecord,
)
from polylogue.storage.sqlite.queries import (
    session_insight_thread_queries as session_insight_threads_q,
)
from polylogue.storage.sqlite.query_store_archive import SQLiteQueryStoreArchiveMixin
from polylogue.storage.sqlite.query_store_insight_profiles import (
    SQLiteQueryStoreInsightProfilesMixin,
)
from polylogue.storage.sqlite.query_store_maintenance import SQLiteQueryStoreMaintenanceMixin
from polylogue.storage.sqlite.query_store_work_evidence import SQLiteQueryStoreWorkEvidenceMixin


class SQLiteQueryStore(
    SQLiteQueryStoreArchiveMixin,
    SQLiteQueryStoreInsightProfilesMixin,
    SQLiteQueryStoreWorkEvidenceMixin,
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
        from polylogue.storage.derived.session.status import session_insight_status_async

        async with self._connection_factory() as conn:
            return await session_insight_status_async(conn, verify_freshness=verify_freshness)

    # -- Threads (formerly query_store_insight_threads.py) ------------------

    async def get_thread(self, thread_id: str) -> ThreadRecord | None:
        async with self._connection_factory() as conn:
            return await session_insight_threads_q.get_thread(conn, thread_id)

    # -- Summaries (formerly query_store_insight_summaries.py) --------------


__all__ = ["SQLiteQueryStore"]
