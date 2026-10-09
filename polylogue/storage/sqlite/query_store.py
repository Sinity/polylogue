"""Low-level SQLite query store composed from explicit concern bands."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import aiosqlite

from polylogue.analysis.topology import SessionTopology
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

    @asynccontextmanager
    async def read_snapshot(self) -> AsyncIterator[SQLiteQueryStore]:
        """Own one connection and snapshot for a composed repository read.

        The returned store belongs to this operation's task and lifetime.
        An existing caller transaction remains its caller's responsibility.
        """
        from polylogue.storage.sqlite.query_store_archive import _message_snapshot

        owner = asyncio.current_task()
        active = True
        async with self._connection_factory() as conn, _message_snapshot(conn):

            @asynccontextmanager
            async def pinned_connection() -> AsyncIterator[aiosqlite.Connection]:
                if not active or asyncio.current_task() is not owner:
                    raise RuntimeError("snapshot queries require their active operation owner")
                yield conn

            try:
                yield SQLiteQueryStore(connection_factory=pinned_connection)
            finally:
                active = False

    async def get_session_topology(
        self,
        session_id: str,
        *,
        node_offset: int = 0,
        node_limit: int | None = 200,
        edge_limit: int | None = 500,
    ) -> SessionTopology | None:
        """Read one graph page in one SQLite snapshot, including its root walk."""
        from polylogue.storage.derived.topology.derivation import derive_session_topology_async

        async with self._connection_factory() as conn:
            # A bulk caller may already own a transaction. Do not commit or
            # roll back that caller's work; ordinary reads own a deferred,
            # read-only snapshot rather than acquiring the writer lease.
            owns_snapshot = not conn.in_transaction

            @asynccontextmanager
            async def pinned_connection() -> AsyncIterator[aiosqlite.Connection]:
                yield conn

            try:
                if owns_snapshot:
                    await conn.execute("BEGIN")
                return await derive_session_topology_async(
                    SQLiteQueryStore(connection_factory=pinned_connection),
                    session_id,
                    node_offset=node_offset,
                    node_limit=node_limit,
                    edge_limit=edge_limit,
                )
            finally:
                if owns_snapshot:
                    await conn.rollback()

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
