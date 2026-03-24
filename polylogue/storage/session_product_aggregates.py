"""Aggregate refresh helpers for day summaries and tag rollups."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.archive_product_builders import (
    build_day_session_summary_records,
    build_session_tag_rollup_records,
    date_from_iso,
)
from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.session_product_rows import hydrate_session_profile
from polylogue.storage.session_product_storage import (
    replace_day_session_summaries_sync,
    replace_session_tag_rollup_rows_sync,
)
from polylogue.storage.store import SessionProfileRecord

_PROFILE_BUCKET_DAY_SQL = (
    "COALESCE(sp.canonical_session_date, "
    "date(COALESCE(sp.first_message_at, json_extract(sp.evidence_payload_json, '$.created_at'), sp.source_updated_at, sp.last_message_at)))"
)


def replace_day_session_summaries_async(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    day: str,
    records,
    transaction_depth: int,
):
    from polylogue.storage.backends.queries.session_product_summary_queries import (
        replace_day_session_summaries as replace_day_session_summaries_async_query,
    )

    return replace_day_session_summaries_async_query(
        conn,
        provider_name=provider_name,
        day=day,
        records=records,
        transaction_depth=transaction_depth,
    )


def replace_session_tag_rollup_rows_async(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    bucket_day: str,
    records,
    transaction_depth: int,
):
    from polylogue.storage.backends.queries.session_product_summary_queries import (
        replace_session_tag_rollup_rows as replace_session_tag_rollup_rows_async_query,
    )

    return replace_session_tag_rollup_rows_async_query(
        conn,
        provider_name=provider_name,
        bucket_day=bucket_day,
        records=records,
        transaction_depth=transaction_depth,
    )


def load_sync_provider_day_profile_records(
    conn: sqlite3.Connection,
    *,
    provider_name: str,
    bucket_day: str,
) -> list[SessionProfileRecord]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM session_profiles sp
        WHERE sp.provider_name = ?
          AND {_PROFILE_BUCKET_DAY_SQL} = ?
        ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id
        """,
        (provider_name, bucket_day),
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


async def load_async_provider_day_profile_records(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    bucket_day: str,
) -> list[SessionProfileRecord]:
    rows = await (
        await conn.execute(
            f"""
            SELECT *
            FROM session_profiles sp
            WHERE sp.provider_name = ?
              AND {_PROFILE_BUCKET_DAY_SQL} = ?
            ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id
            """,
            (provider_name, bucket_day),
        )
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


def refresh_sync_provider_day_aggregates(
    conn: sqlite3.Connection,
    groups: set[tuple[str, str]],
) -> None:
    for provider_name, bucket_day in groups:
        profile_records = load_sync_provider_day_profile_records(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
        )
        profiles = [hydrate_session_profile(record) for record in profile_records]
        day_rows = build_day_session_summary_records(profiles) if profiles else []
        tag_rows = build_session_tag_rollup_records(profiles) if profiles else []
        replace_day_session_summaries_sync(
            conn,
            provider_name=provider_name,
            day=bucket_day,
            records=[row for row in day_rows if row.provider_name == provider_name and row.day == bucket_day],
        )
        replace_session_tag_rollup_rows_sync(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
            records=[
                row
                for row in tag_rows
                if row.provider_name == provider_name and row.bucket_day == bucket_day
            ],
        )


async def refresh_async_provider_day_aggregates(
    conn: aiosqlite.Connection,
    groups: set[tuple[str, str]],
    *,
    transaction_depth: int,
) -> None:
    for provider_name, bucket_day in groups:
        profile_records = await load_async_provider_day_profile_records(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
        )
        profiles = [hydrate_session_profile(record) for record in profile_records]
        day_rows = build_day_session_summary_records(profiles) if profiles else []
        tag_rows = build_session_tag_rollup_records(profiles) if profiles else []
        await replace_day_session_summaries_async(
            conn,
            provider_name=provider_name,
            day=bucket_day,
            records=[row for row in day_rows if row.provider_name == provider_name and row.day == bucket_day],
            transaction_depth=transaction_depth,
        )
        await replace_session_tag_rollup_rows_async(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
            records=[
                row
                for row in tag_rows
                if row.provider_name == provider_name and row.bucket_day == bucket_day
            ],
            transaction_depth=transaction_depth,
        )


def profile_provider_day(record: SessionProfileRecord | None) -> tuple[str, str] | None:
    if record is None:
        return None
    if record.canonical_session_date:
        return (record.provider_name, record.canonical_session_date)
    day_candidates = [
        record.first_message_at,
        (
            str(record.evidence_payload.get("created_at"))
            if isinstance(record.evidence_payload, dict) and record.evidence_payload.get("created_at")
            else None
        ),
        record.source_updated_at,
        record.last_message_at,
    ]
    bucket_day: str | None = None
    for candidate in day_candidates:
        if not candidate:
            continue
        try:
            bucket_day = date_from_iso(str(candidate)[:10]).isoformat()
            break
        except ValueError:
            continue
    if bucket_day is None:
        return None
    return (record.provider_name, bucket_day)


__all__ = [
    "_PROFILE_BUCKET_DAY_SQL",
    "load_async_provider_day_profile_records",
    "load_sync_provider_day_profile_records",
    "profile_provider_day",
    "refresh_async_provider_day_aggregates",
    "refresh_sync_provider_day_aggregates",
]
