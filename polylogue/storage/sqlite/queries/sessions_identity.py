"""Session identity, scope, and metadata helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.core.json import JSONDocument
from polylogue.storage.sqlite.connection import _build_source_scope_filter


async def resolve_id(conn: aiosqlite.Connection, id_prefix: str, *, strict: bool = False) -> str | None:
    cursor = await conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (id_prefix,),
    )
    row = await cursor.fetchone()
    if row:
        return str(row["session_id"])

    if strict:
        return None

    cursor = await conn.execute(
        "SELECT session_id FROM sessions WHERE session_id LIKE ? LIMIT 2",
        (f"{id_prefix}%",),
    )
    rows = list(await cursor.fetchall())
    if len(rows) == 1:
        return str(rows[0]["session_id"])
    return None


async def get_last_sync_timestamp(conn: aiosqlite.Connection) -> str | None:
    cursor = await conn.execute("SELECT datetime(MAX(updated_at_ms) / 1000, 'unixepoch') as last FROM sessions")
    row = await cursor.fetchone()
    return row["last"] if row and row["last"] else None


def session_id_query(
    *,
    source_names: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="origin",
    )
    sql = "SELECT session_id FROM sessions"
    if predicate:
        sql += f" WHERE {predicate}"
    sql += " ORDER BY sort_key_ms DESC, session_id ASC"
    return sql, tuple(params)


async def count_session_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
) -> int:
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="origin",
    )
    sql = "SELECT COUNT(*) AS count FROM sessions"
    if predicate:
        sql += f" WHERE {predicate}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["count"]) if row is not None else 0


async def iter_session_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
    page_size: int = 1000,
) -> AsyncIterator[str]:
    sql, params = session_id_query(source_names=source_names)
    cursor = await conn.execute(sql, params)
    while True:
        rows = list(await cursor.fetchmany(page_size))
        if not rows:
            break
        for row in rows:
            yield str(row["session_id"])


async def get_metadata(conn: aiosqlite.Connection, session_id: str) -> JSONDocument:
    cursor = await conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
    row = await cursor.fetchone()
    return {} if row is not None else {}


async def update_metadata_raw(
    conn: aiosqlite.Connection,
    session_id: str,
    metadata: JSONDocument,
) -> None:
    del metadata
    await conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))


async def set_metadata(
    conn: aiosqlite.Connection,
    session_id: str,
    metadata: JSONDocument,
    transaction_depth: int,
) -> None:
    del metadata
    await conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
    if transaction_depth == 0:
        await conn.commit()


async def list_tags(conn: aiosqlite.Connection, *, provider: str | None = None) -> dict[str, int]:
    """List all tags with session counts, optionally filtered by provider.

    Reads from the normalized ``tags`` + ``session_tags`` tables.
    Falls back to JSON metadata extraction when the normalized tables are empty
    (e.g. before rebuild has run).
    """
    params: tuple[str, ...] = ()
    join = "JOIN sessions c ON ct.session_id = c.session_id"
    where = ""
    if provider:
        where = " AND c.origin = ?"
        params = (provider,)
    cursor = await conn.execute(
        f"""
        SELECT t.name AS tag_name, COUNT(DISTINCT ct.session_id) AS cnt
        FROM session_tags ct
        JOIN tags t ON t.id = ct.tag_id
        {join}
        WHERE 1=1{where}
        GROUP BY t.name
        ORDER BY cnt DESC
        """,
        params,
    )
    rows = list(await cursor.fetchall())
    # #1240: M2M (tags + session_tags) is the canonical read surface;
    # the fallback JSON metadata fallback was removed with SCHEMA_VERSION 3.
    return {row["tag_name"]: row["cnt"] for row in rows}


__all__ = [
    "count_session_ids",
    "get_last_sync_timestamp",
    "get_metadata",
    "iter_session_ids",
    "list_tags",
    "resolve_id",
    "session_id_query",
    "set_metadata",
    "update_metadata_raw",
]
