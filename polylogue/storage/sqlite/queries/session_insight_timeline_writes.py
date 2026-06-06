"""Timeline-oriented durable session-insight write queries."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.insights.session.storage import (
    session_phase_insert_columns,
    session_phase_insert_values,
    session_work_event_insert_columns,
    session_work_event_insert_values,
)
from polylogue.storage.runtime import SessionPhaseRecord, SessionWorkEventRecord
from polylogue.storage.sqlite.queries._bulk_replace import replace_insight_rows

__all__ = [
    "replace_session_phases",
    "replace_session_phases_bulk",
    "replace_session_work_events",
    "replace_session_work_events_bulk",
]

_ASYNC_COLUMN_CACHE: dict[tuple[int, str], bool] = {}


async def _table_has_column(conn: aiosqlite.Connection, table: str, column: str) -> bool:
    key = (id(conn), f"{table}.{column}")
    cached = _ASYNC_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    found = any(str(row["name"] if "name" in row else row[1]) == column for row in rows)
    _ASYNC_COLUMN_CACHE[key] = found
    return found


async def replace_session_work_events(
    conn: aiosqlite.Connection,
    session_id: str,
    records: list[SessionWorkEventRecord],
    transaction_depth: int,
) -> None:
    await replace_session_work_events_bulk(
        conn,
        [session_id],
        records,
        transaction_depth,
    )


async def replace_session_work_events_bulk(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    records: Sequence[SessionWorkEventRecord],
    transaction_depth: int,
) -> None:
    has_fallback_payload = await _table_has_column(conn, "session_work_events", "payload_json") if records else False
    columns = session_work_event_insert_columns(has_fallback_payload=has_fallback_payload)
    await replace_insight_rows(
        conn,
        table="session_work_events",
        id_column="session_id",
        id_values=session_ids,
        columns=columns,
        records=records,
        extractor=lambda r: session_work_event_insert_values(r, has_fallback_payload=has_fallback_payload),
        transaction_depth=transaction_depth,
    )


async def replace_session_phases(
    conn: aiosqlite.Connection,
    session_id: str,
    records: list[SessionPhaseRecord],
    transaction_depth: int,
) -> None:
    await replace_session_phases_bulk(
        conn,
        [session_id],
        records,
        transaction_depth,
    )


async def replace_session_phases_bulk(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    records: Sequence[SessionPhaseRecord],
    transaction_depth: int,
) -> None:
    has_fallback_payload = await _table_has_column(conn, "session_phases", "payload_json") if records else False
    columns = session_phase_insert_columns(has_fallback_payload=has_fallback_payload)
    await replace_insight_rows(
        conn,
        table="session_phases",
        id_column="session_id",
        id_values=session_ids,
        columns=columns,
        records=records,
        extractor=lambda r: session_phase_insert_values(r, has_fallback_payload=has_fallback_payload),
        transaction_depth=transaction_depth,
    )
