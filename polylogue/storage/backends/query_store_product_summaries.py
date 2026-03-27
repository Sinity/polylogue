"""Durable product rollup/summary read band for SQLiteQueryStore."""

from __future__ import annotations

from polylogue.storage.backends.queries import (
    session_product_summary_queries as session_product_summaries_q,
)
from polylogue.storage.store import DaySessionSummaryRecord, SessionTagRollupRecord


class SQLiteQueryStoreProductSummariesMixin:
    async def list_session_tag_rollup_rows(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ) -> list[SessionTagRollupRecord]:
        async with self._connection_factory() as conn:
            return await session_product_summaries_q.list_session_tag_rollup_rows(
                conn,
                provider=provider,
                since=since,
                until=until,
                query=query,
            )

    async def list_day_session_summaries(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[DaySessionSummaryRecord]:
        async with self._connection_factory() as conn:
            return await session_product_summaries_q.list_day_session_summaries(
                conn,
                provider=provider,
                since=since,
                until=until,
            )


__all__ = ["SQLiteQueryStoreProductSummariesMixin"]
