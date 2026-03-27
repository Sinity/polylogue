"""Hydrated work-thread durable product reads for the repository."""

from __future__ import annotations

from polylogue.storage.session_product_thread_rows import hydrate_work_thread


class RepositoryProductThreadReadMixin:
    async def get_work_thread_record(self, thread_id: str):
        return await self.queries.get_work_thread(thread_id)

    async def get_work_thread(self, thread_id: str):
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
    ):
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
    ):
        return await self.queries.list_work_threads(
            since=since,
            until=until,
            limit=limit,
            offset=offset,
            query=query,
        )


__all__ = ["RepositoryProductThreadReadMixin"]
