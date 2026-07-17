"""Durable insight aggregate reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.storage.query_models import SessionTagRollupListQuery
from polylogue.storage.runtime import SessionTagRollupRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryInsightSummaryReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def _list_session_tag_rollup_rows_query(
        self,
        query: SessionTagRollupListQuery,
    ) -> list[SessionTagRollupRecord]:
        return await self.queries._list_session_tag_rollup_rows_query(query)

    async def list_session_tag_rollup_records(
        self,
        *,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ) -> list[SessionTagRollupRecord]:
        return await self._list_session_tag_rollup_rows_query(
            SessionTagRollupListQuery(
                origin=origin,
                since=since,
                until=until,
                query=query,
            )
        )


__all__ = ["RepositoryInsightSummaryReadMixin"]
