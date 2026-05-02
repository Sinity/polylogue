"""Hydrated work-thread durable insight reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.conversation.threads import WorkThread
from polylogue.storage.insights.insight_read_support import hydrate_optional, hydrate_sequence
from polylogue.storage.insights.session.threads import hydrate_work_thread
from polylogue.storage.query_models import WorkThreadListQuery
from polylogue.storage.runtime import WorkThreadRecord

if TYPE_CHECKING:
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryInsightThreadReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_work_thread_record(self, thread_id: str) -> WorkThreadRecord | None:
        return await self.queries.get_work_thread(thread_id)

    async def get_work_thread(self, thread_id: str) -> WorkThread | None:
        record = await self.get_work_thread_record(thread_id)
        return hydrate_optional(record, hydrate_work_thread)

    async def _list_work_thread_records_query(
        self,
        query: WorkThreadListQuery,
    ) -> list[WorkThreadRecord]:
        return await self.queries._list_work_threads_query(query)

    async def list_work_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThread]:
        records = await self._list_work_thread_records_query(
            WorkThreadListQuery(
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )
        )
        return hydrate_sequence(records, hydrate_work_thread)

    async def list_work_thread_records(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThreadRecord]:
        return await self._list_work_thread_records_query(
            WorkThreadListQuery(
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )
        )


__all__ = ["RepositoryInsightThreadReadMixin"]
