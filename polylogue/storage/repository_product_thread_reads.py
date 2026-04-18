"""Hydrated work-thread durable product reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.threads import WorkThread
from polylogue.storage.session_product_threads import hydrate_work_thread
from polylogue.storage.store import WorkThreadRecord

if TYPE_CHECKING:
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryProductThreadReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_work_thread_record(self, thread_id: str) -> WorkThreadRecord | None:
        return await self.queries.get_work_thread(thread_id)

    async def get_work_thread(self, thread_id: str) -> WorkThread | None:
        record = await self.get_work_thread_record(thread_id)
        return hydrate_work_thread(record) if record is not None else None

    async def list_work_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThread]:
        records = await self.queries.list_work_threads(
            since=since,
            until=until,
            limit=limit,
            offset=offset,
            query=query,
        )
        return [hydrate_work_thread(record) for record in records]

    async def list_work_thread_records(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThreadRecord]:
        return await self.queries.list_work_threads(
            since=since,
            until=until,
            limit=limit,
            offset=offset,
            query=query,
        )


__all__ = ["RepositoryProductThreadReadMixin"]
