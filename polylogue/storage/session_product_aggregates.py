"""Aggregate refresh helpers for day summaries and tag rollups."""

from __future__ import annotations

import sqlite3
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any

import aiosqlite

from polylogue.archive_product_rollups import build_session_tag_rollup_records
from polylogue.archive_product_summaries import build_day_session_summary_records
from polylogue.archive_products import date_from_iso
from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.session_product_profiles import hydrate_session_profile
from polylogue.storage.session_product_storage import (
    replace_day_session_summaries_sync,
    replace_session_tag_rollup_rows_sync,
)
from polylogue.storage.store import (
    DaySessionSummaryRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
)

_PROFILE_BUCKET_DAY_SQL = (
    "COALESCE(sp.canonical_session_date, "
    "date(COALESCE(sp.first_message_at, json_extract(sp.evidence_payload_json, '$.created_at'), sp.source_updated_at, sp.last_message_at)))"
)
_PROVIDER_DAY_PROFILE_RECORDS_SQL_TEMPLATE = f"""
    WITH target_groups(provider_name, bucket_day) AS (
        VALUES {{values}}
    )
    SELECT
        tg.provider_name AS group_provider_name,
        tg.bucket_day AS group_bucket_day,
        sp.*
    FROM target_groups tg
    JOIN session_profiles sp
      ON sp.provider_name = tg.provider_name
     AND {_PROFILE_BUCKET_DAY_SQL} = tg.bucket_day
    ORDER BY tg.provider_name, tg.bucket_day, COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id
"""
_GROUP_BATCH_SIZE = 100
_DISTINCT_PROVIDER_DAY_GROUPS_SQL = f"""
    SELECT DISTINCT
        sp.provider_name AS provider_name,
        {_PROFILE_BUCKET_DAY_SQL} AS bucket_day
    FROM session_profiles sp
    WHERE sp.provider_name IS NOT NULL
      AND {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL
    ORDER BY provider_name, bucket_day
"""


def replace_day_session_summaries_async(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    day: str,
    records: list[DaySessionSummaryRecord],
    transaction_depth: int,
) -> Coroutine[Any, Any, None]:
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
    records: list[SessionTagRollupRecord],
    transaction_depth: int,
) -> Coroutine[Any, Any, None]:
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
    return load_sync_provider_day_profile_records_by_groups(
        conn,
        [(provider_name, bucket_day)],
    ).get((provider_name, bucket_day), [])


def list_sync_provider_day_groups(
    conn: sqlite3.Connection,
) -> list[tuple[str, str]]:
    return [
        (str(row["provider_name"]), str(row["bucket_day"]))
        for row in conn.execute(_DISTINCT_PROVIDER_DAY_GROUPS_SQL).fetchall()
    ]


async def load_async_provider_day_profile_records(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    bucket_day: str,
) -> list[SessionProfileRecord]:
    return (
        await load_async_provider_day_profile_records_by_groups(
            conn,
            [(provider_name, bucket_day)],
        )
    ).get((provider_name, bucket_day), [])


async def list_async_provider_day_groups(
    conn: aiosqlite.Connection,
) -> list[tuple[str, str]]:
    return [
        (str(row["provider_name"]), str(row["bucket_day"]))
        for row in await (await conn.execute(_DISTINCT_PROVIDER_DAY_GROUPS_SQL)).fetchall()
    ]


def _normalize_provider_day_groups(
    groups: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        dict.fromkeys(
            (str(provider_name), str(bucket_day))
            for provider_name, bucket_day in groups
            if str(provider_name) and str(bucket_day)
        )
    )


def _chunk_provider_day_groups(
    groups: Sequence[tuple[str, str]],
    *,
    size: int = _GROUP_BATCH_SIZE,
) -> list[tuple[tuple[str, str], ...]]:
    return [
        tuple(groups[index : index + size]) for index in range(0, len(groups), size) if groups[index : index + size]
    ]


def _empty_provider_day_profile_groups(
    groups: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], list[SessionProfileRecord]]:
    return {group: [] for group in groups}


def _group_profile_records_by_provider_day(
    rows: Iterable[sqlite3.Row] | Iterable[aiosqlite.Row],
    *,
    groups: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], list[SessionProfileRecord]]:
    grouped: dict[tuple[str, str], list[SessionProfileRecord]] = _empty_provider_day_profile_groups(groups)
    for row in rows:
        group = (str(row["group_provider_name"]), str(row["group_bucket_day"]))
        grouped[group].append(_row_to_session_profile_record(row))
    return grouped


def _aggregate_rows_for_profile_records(
    profile_records: Sequence[SessionProfileRecord],
) -> tuple[list[DaySessionSummaryRecord], list[SessionTagRollupRecord]]:
    if not profile_records:
        return [], []
    profiles = [hydrate_session_profile(record) for record in profile_records]
    return (
        build_day_session_summary_records(profiles),
        build_session_tag_rollup_records(profiles),
    )


def load_sync_provider_day_profile_records_by_groups(
    conn: sqlite3.Connection,
    groups: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], list[SessionProfileRecord]]:
    normalized_groups = _normalize_provider_day_groups(groups)
    if not normalized_groups:
        return {}
    grouped: dict[tuple[str, str], list[SessionProfileRecord]] = _empty_provider_day_profile_groups(normalized_groups)
    for group_chunk in _chunk_provider_day_groups(normalized_groups):
        values = ", ".join("(?, ?)" for _ in group_chunk)
        params = tuple(value for group in group_chunk for value in group)
        rows = conn.execute(
            _PROVIDER_DAY_PROFILE_RECORDS_SQL_TEMPLATE.format(values=values),
            params,
        ).fetchall()
        for group, records in _group_profile_records_by_provider_day(rows, groups=group_chunk).items():
            grouped[group].extend(records)
    return grouped


async def load_async_provider_day_profile_records_by_groups(
    conn: aiosqlite.Connection,
    groups: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], list[SessionProfileRecord]]:
    normalized_groups = _normalize_provider_day_groups(groups)
    if not normalized_groups:
        return {}
    grouped: dict[tuple[str, str], list[SessionProfileRecord]] = _empty_provider_day_profile_groups(normalized_groups)
    for group_chunk in _chunk_provider_day_groups(normalized_groups):
        values = ", ".join("(?, ?)" for _ in group_chunk)
        params = tuple(value for group in group_chunk for value in group)
        rows = await (
            await conn.execute(
                _PROVIDER_DAY_PROFILE_RECORDS_SQL_TEMPLATE.format(values=values),
                params,
            )
        ).fetchall()
        for group, records in _group_profile_records_by_provider_day(rows, groups=group_chunk).items():
            grouped[group].extend(records)
    return grouped


def refresh_sync_provider_day_aggregates(
    conn: sqlite3.Connection,
    groups: set[tuple[str, str]],
) -> None:
    normalized_groups = _normalize_provider_day_groups(sorted(groups))
    for group_chunk in _chunk_provider_day_groups(normalized_groups):
        profile_records_by_group = load_sync_provider_day_profile_records_by_groups(conn, group_chunk)
        for provider_name, bucket_day in group_chunk:
            day_rows, tag_rows = _aggregate_rows_for_profile_records(
                profile_records_by_group.get((provider_name, bucket_day), []),
            )
            replace_day_session_summaries_sync(
                conn,
                provider_name=provider_name,
                day=bucket_day,
                records=day_rows,
            )
            replace_session_tag_rollup_rows_sync(
                conn,
                provider_name=provider_name,
                bucket_day=bucket_day,
                records=tag_rows,
            )


async def refresh_async_provider_day_aggregates(
    conn: aiosqlite.Connection,
    groups: set[tuple[str, str]],
    *,
    transaction_depth: int,
) -> None:
    normalized_groups = _normalize_provider_day_groups(sorted(groups))
    for group_chunk in _chunk_provider_day_groups(normalized_groups):
        profile_records_by_group = await load_async_provider_day_profile_records_by_groups(
            conn,
            group_chunk,
        )
        for provider_name, bucket_day in group_chunk:
            day_rows, tag_rows = _aggregate_rows_for_profile_records(
                profile_records_by_group.get((provider_name, bucket_day), []),
            )
            await replace_day_session_summaries_async(
                conn,
                provider_name=provider_name,
                day=bucket_day,
                records=day_rows,
                transaction_depth=transaction_depth,
            )
            await replace_session_tag_rollup_rows_async(
                conn,
                provider_name=provider_name,
                bucket_day=bucket_day,
                records=tag_rows,
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
    "load_async_provider_day_profile_records_by_groups",
    "list_async_provider_day_groups",
    "list_sync_provider_day_groups",
    "load_sync_provider_day_profile_records",
    "load_sync_provider_day_profile_records_by_groups",
    "profile_provider_day",
    "refresh_async_provider_day_aggregates",
    "refresh_sync_provider_day_aggregates",
]
