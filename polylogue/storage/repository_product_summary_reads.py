"""Durable product aggregate reads for the repository."""

from __future__ import annotations


class RepositoryProductSummaryReadMixin:
    async def list_session_tag_rollup_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ):
        return await self.queries.list_session_tag_rollup_rows(
            provider=provider,
            since=since,
            until=until,
            query=query,
        )

    async def list_day_session_summary_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ):
        return await self.queries.list_day_session_summaries(
            provider=provider,
            since=since,
            until=until,
        )


__all__ = ["RepositoryProductSummaryReadMixin"]
