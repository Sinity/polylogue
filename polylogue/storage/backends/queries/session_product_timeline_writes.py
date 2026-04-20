"""Timeline-oriented durable session-product write queries."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.session_product_storage import (
    build_insert_sql,
    session_phase_insert_columns,
    session_phase_insert_values,
    session_work_event_insert_columns,
    session_work_event_insert_values,
)
from polylogue.storage.store import SessionPhaseRecord, SessionWorkEventRecord

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
    conversation_id: str,
    records: list[SessionWorkEventRecord],
    transaction_depth: int,
) -> None:
    await replace_session_work_events_bulk(
        conn,
        [conversation_id],
        records,
        transaction_depth,
    )


async def replace_session_work_events_bulk(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    records: Sequence[SessionWorkEventRecord],
    transaction_depth: int,
) -> None:
    if conversation_ids:
        placeholders = ", ".join("?" for _ in conversation_ids)
        await conn.execute(
            f"DELETE FROM session_work_events WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        )
    if records:
        has_legacy_payload = await _table_has_column(conn, "session_work_events", "payload_json")
        columns = session_work_event_insert_columns(has_legacy_payload=has_legacy_payload)
        await conn.executemany(
            build_insert_sql("session_work_events", columns),
            [
                session_work_event_insert_values(
                    record,
                    has_legacy_payload=has_legacy_payload,
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()


async def replace_session_phases(
    conn: aiosqlite.Connection,
    conversation_id: str,
    records: list[SessionPhaseRecord],
    transaction_depth: int,
) -> None:
    await replace_session_phases_bulk(
        conn,
        [conversation_id],
        records,
        transaction_depth,
    )


async def replace_session_phases_bulk(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    records: Sequence[SessionPhaseRecord],
    transaction_depth: int,
) -> None:
    if conversation_ids:
        placeholders = ", ".join("?" for _ in conversation_ids)
        await conn.execute(
            f"DELETE FROM session_phases WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        )
    if records:
        has_legacy_payload = await _table_has_column(conn, "session_phases", "payload_json")
        columns = session_phase_insert_columns(has_legacy_payload=has_legacy_payload)
        await conn.executemany(
            build_insert_sql("session_phases", columns),
            [
                session_phase_insert_values(
                    record,
                    has_legacy_payload=has_legacy_payload,
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()
