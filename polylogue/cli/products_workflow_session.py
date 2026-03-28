"""Session-scoped archive-product workflow helpers."""

from __future__ import annotations

from polylogue.archive_products import (
    SessionEnrichmentProductQuery,
    SessionPhaseProductQuery,
    SessionProfileProductQuery,
    SessionWorkEventProductQuery,
    WorkThreadProductQuery,
)
from polylogue.cli.types import AppEnv
from polylogue.sync_bridge import run_coroutine_sync


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


__all__ = [
    "list_session_enrichment_products",
    "list_session_phase_products",
    "list_session_profile_products",
    "list_session_work_event_products",
    "list_work_thread_products",
]
