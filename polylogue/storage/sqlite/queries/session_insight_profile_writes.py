"""Profile-oriented durable session-insight write queries."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.insights.session.storage import (
    session_profile_insert_columns,
    session_profile_insert_values,
)
from polylogue.storage.runtime import SessionProfileRecord
from polylogue.storage.sqlite.queries._bulk_replace import replace_insight_rows

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


__all__ = ["replace_session_profile", "replace_session_profiles_bulk"]


async def replace_session_profiles_bulk(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    records: Sequence[SessionProfileRecord],
    transaction_depth: int,
) -> None:
    has_legacy_payload = await _table_has_column(conn, "session_profiles", "payload_json") if records else False
    columns = session_profile_insert_columns(has_legacy_payload=has_legacy_payload)
    await replace_insight_rows(
        conn,
        table="session_profiles",
        id_column="conversation_id",
        id_values=conversation_ids,
        columns=columns,
        records=records,
        extractor=lambda r: session_profile_insert_values(r, has_legacy_payload=has_legacy_payload),
        transaction_depth=transaction_depth,
    )


async def replace_session_profile(
    conn: aiosqlite.Connection,
    record: SessionProfileRecord,
    transaction_depth: int,
) -> None:
    await replace_session_profiles_bulk(
        conn,
        [record.conversation_id],
        [record],
        transaction_depth,
    )
