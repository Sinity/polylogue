"""Durable work-thread read band for SQLiteQueryStore."""

from __future__ import annotations

from polylogue.storage.backends.queries import (
    session_product_thread_queries as session_product_threads_q,
)
from polylogue.storage.store import WorkThreadRecord


class SQLiteQueryStoreProductThreadsMixin:
    async def get_work_thread(self, thread_id: str) -> WorkThreadRecord | None:
        async with self._connection_factory() as conn:
            return await session_product_threads_q.get_work_thread(conn, thread_id)

    async def list_work_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThreadRecord]:
        async with self._connection_factory() as conn:
            return await session_product_threads_q.list_work_threads(
                conn,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )


__all__ = ["SQLiteQueryStoreProductThreadsMixin"]
