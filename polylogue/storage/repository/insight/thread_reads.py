"""Hydrated thread durable insight reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.session.threads import Thread
from polylogue.storage.insights.insight_read_support import hydrate_optional, hydrate_sequence
from polylogue.storage.insights.session.threads import hydrate_thread
from polylogue.storage.query_models import ThreadListQuery
from polylogue.storage.runtime import ThreadRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryInsightThreadReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_thread_record(self, thread_id: str) -> ThreadRecord | None:
        return await self.queries.get_thread(thread_id)

    async def get_thread(self, thread_id: str) -> Thread | None:
        record = await self.get_thread_record(thread_id)
        return hydrate_optional(record, hydrate_thread)

    async def _list_thread_records_query(
        self,
        query: ThreadListQuery,
    ) -> list[ThreadRecord]:
        return await self.queries._list_threads_query(query)

    async def list_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[Thread]:
        records = await self._list_thread_records_query(
            ThreadListQuery(
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )
        )
        return hydrate_sequence(records, hydrate_thread)

    async def list_thread_records(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[ThreadRecord]:
        return await self._list_thread_records_query(
            ThreadListQuery(
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )
        )


__all__ = ["RepositoryInsightThreadReadMixin"]
