"""Workflow helpers for archive-product CLI commands."""

from __future__ import annotations

from polylogue.archive_products import (
    ArchiveDebtProductQuery,
    DaySessionSummaryProductQuery,
    MaintenanceRunProductQuery,
    ProviderAnalyticsProductQuery,
    SessionEnrichmentProductQuery,
    SessionPhaseProductQuery,
    SessionProfileProductQuery,
    SessionTagRollupQuery,
    SessionWorkEventProductQuery,
    WeekSessionSummaryProductQuery,
    WorkThreadProductQuery,
)
from polylogue.cli.types import AppEnv
from polylogue.sync_bridge import run_coroutine_sync


def get_products_status(env: AppEnv) -> tuple[dict[str, int | bool], list[object]]:
    status = run_coroutine_sync(env.operations.get_session_product_status())
    debt_items = run_coroutine_sync(
        env.operations.list_archive_debt_products(ArchiveDebtProductQuery())
    )
    return status, debt_items


def list_session_profile_products(
    env: AppEnv,
    *,
    provider: str | None,
    since: str | None,
    until: str | None,
    first_message_since: str | None,
    first_message_until: str | None,
    session_date_since: str | None,
    session_date_until: str | None,
    tier: str,
    query: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_session_profile_products(
            SessionProfileProductQuery(
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                tier=tier,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )


def list_session_work_event_products(
    env: AppEnv,
    *,
    conversation_id: str | None,
    provider: str | None,
    since: str | None,
    until: str | None,
    kind: str | None,
    query: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_session_work_event_products(
            SessionWorkEventProductQuery(
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )


def list_session_enrichment_products(
    env: AppEnv,
    *,
    provider: str | None,
    since: str | None,
    until: str | None,
    first_message_since: str | None,
    first_message_until: str | None,
    session_date_since: str | None,
    session_date_until: str | None,
    refined_work_kind: str | None,
    query: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_session_enrichment_products(
            SessionEnrichmentProductQuery(
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                refined_work_kind=refined_work_kind,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )


def list_session_phase_products(
    env: AppEnv,
    *,
    conversation_id: str | None,
    provider: str | None,
    since: str | None,
    until: str | None,
    kind: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_session_phase_products(
            SessionPhaseProductQuery(
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
            )
        )
    )


def list_work_thread_products(
    env: AppEnv,
    *,
    since: str | None,
    until: str | None,
    query: str | None,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_work_thread_products(
            WorkThreadProductQuery(
                since=since,
                until=until,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )


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


def list_maintenance_run_products(env: AppEnv, *, limit: int) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_maintenance_run_products(
            MaintenanceRunProductQuery(limit=limit)
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


def list_archive_debt_products(
    env: AppEnv,
    *,
    category: str | None,
    actionable_only: bool,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_archive_debt_products(
            ArchiveDebtProductQuery(
                category=category,
                only_actionable=actionable_only,
                limit=limit,
                offset=offset,
            )
        )
    )


__all__ = [
    "get_products_status",
    "list_archive_debt_products",
    "list_day_session_summary_products",
    "list_maintenance_run_products",
    "list_provider_analytics_products",
    "list_session_enrichment_products",
    "list_session_phase_products",
    "list_session_profile_products",
    "list_session_tag_rollup_products",
    "list_session_work_event_products",
    "list_week_session_summary_products",
    "list_work_thread_products",
]
