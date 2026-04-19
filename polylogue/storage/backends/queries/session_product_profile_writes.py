"""Profile-oriented durable session-product write queries."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.session_product_storage import (
    session_profile_insert_columns,
    session_profile_insert_values,
)
from polylogue.storage.store import SessionProfileRecord

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
    if conversation_ids:
        placeholders = ", ".join("?" for _ in conversation_ids)
        await conn.execute(
            f"DELETE FROM session_profiles WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        )
    if records:
        has_legacy_payload = await _table_has_column(conn, "session_profiles", "payload_json")
        columns = session_profile_insert_columns(has_legacy_payload=has_legacy_payload)
        placeholders = ", ".join("?" for _ in columns)
        await conn.executemany(
            f"""
            INSERT INTO session_profiles (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            [
                session_profile_insert_values(
                    record,
                    has_legacy_payload=has_legacy_payload,
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()


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
