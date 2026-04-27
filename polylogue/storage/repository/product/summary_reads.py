"""Durable product aggregate reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.storage.query_models import (
    DaySessionSummaryListQuery,
    SessionTagRollupListQuery,
)
from polylogue.storage.runtime import DaySessionSummaryRecord, SessionTagRollupRecord

if TYPE_CHECKING:
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryProductSummaryReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def _list_session_tag_rollup_rows_query(
        self,
        query: SessionTagRollupListQuery,
    ) -> list[SessionTagRollupRecord]:
        return await self.queries._list_session_tag_rollup_rows_query(query)

    async def _list_day_session_summary_records_query(
        self,
        query: DaySessionSummaryListQuery,
    ) -> list[DaySessionSummaryRecord]:
        return await self.queries._list_day_session_summaries_query(query)

    async def list_session_tag_rollup_records(
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

    async def list_day_session_summary_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[DaySessionSummaryRecord]:
        return await self._list_day_session_summary_records_query(
            DaySessionSummaryListQuery(
                provider=provider,
                since=since,
                until=until,
            )
        )


__all__ = ["RepositoryProductSummaryReadMixin"]
