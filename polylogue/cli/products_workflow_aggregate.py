"""Aggregate archive-product workflow helpers."""

from __future__ import annotations

from polylogue.archive_products import (
    DaySessionSummaryProductQuery,
    ProviderAnalyticsProductQuery,
    SessionTagRollupQuery,
    WeekSessionSummaryProductQuery,
)
from polylogue.cli.types import AppEnv
from polylogue.sync_bridge import run_coroutine_sync


def list_session_tag_rollup_products(
    env: AppEnv,
    *,
    provider: str | None,
    since: str | None,
    until: str | None,
    query: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_session_tag_rollup_products(
            SessionTagRollupQuery(
                provider=provider,
                since=since,
                until=until,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )


def list_day_session_summary_products(
    env: AppEnv,
    *,
    provider: str | None,
    since: str | None,
    until: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_day_session_summary_products(
            DaySessionSummaryProductQuery(
                provider=provider,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            )
        )
    )


def list_week_session_summary_products(
    env: AppEnv,
    *,
    provider: str | None,
    since: str | None,
    until: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_week_session_summary_products(
            WeekSessionSummaryProductQuery(
                provider=provider,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            )
        )
    )


def list_provider_analytics_products(
    env: AppEnv,
    *,
    provider: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_provider_analytics_products(
            ProviderAnalyticsProductQuery(
                provider=provider,
                limit=limit,
                offset=offset,
            )
        )
    )


__all__ = [
    "list_day_session_summary_products",
    "list_provider_analytics_products",
    "list_session_tag_rollup_products",
    "list_week_session_summary_products",
]
