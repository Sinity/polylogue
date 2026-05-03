"""Conversation identity, scope, and metadata helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.json import dumps as json_dumps
from polylogue.storage.sqlite.connection import _build_source_scope_filter
from polylogue.storage.sqlite.queries.mappers import _parse_json


async def resolve_id(conn: aiosqlite.Connection, id_prefix: str, *, strict: bool = False) -> str | None:
    cursor = await conn.execute(
        "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
        (id_prefix,),
    )
    row = await cursor.fetchone()
    if row:
        return str(row["conversation_id"])

    if strict:
        return None

    cursor = await conn.execute(
        "SELECT conversation_id FROM conversations WHERE conversation_id LIKE ? LIMIT 2",
        (f"{id_prefix}%",),
    )
    rows = list(await cursor.fetchall())
    if len(rows) == 1:
        return str(rows[0]["conversation_id"])
    return None


async def get_last_sync_timestamp(conn: aiosqlite.Connection) -> str | None:
    cursor = await conn.execute("SELECT MAX(timestamp) as last FROM runs")
    row = await cursor.fetchone()
    return row["last"] if row and row["last"] else None


def conversation_id_query(
    *,
    source_names: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    sql = "SELECT conversation_id FROM conversations"
    if predicate:
        sql += f" WHERE {predicate}"
    sql += " ORDER BY sort_key DESC, conversation_id ASC"
    return sql, tuple(params)


async def count_conversation_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
) -> int:
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    sql = "SELECT COUNT(*) AS count FROM conversations"
    if predicate:
        sql += f" WHERE {predicate}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["count"]) if row is not None else 0


async def iter_conversation_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
    page_size: int = 1000,
) -> AsyncIterator[str]:
    sql, params = conversation_id_query(source_names=source_names)
    cursor = await conn.execute(sql, params)
    while True:
        rows = await cursor.fetchmany(page_size)
        if not rows:
            break
        for row in rows:
            yield str(row["conversation_id"])


async def get_metadata(conn: aiosqlite.Connection, conversation_id: str) -> JSONDocument:
    cursor = await conn.execute(
        "SELECT metadata FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return {}
    return json_document(_parse_json(row["metadata"], field="metadata", record_id=conversation_id))


async def update_metadata_raw(
    conn: aiosqlite.Connection,
    conversation_id: str,
    metadata: JSONDocument,
) -> None:
    await conn.execute(
        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
        (json_dumps(metadata), conversation_id),
    )


async def set_metadata(
    conn: aiosqlite.Connection,
    conversation_id: str,
    metadata: JSONDocument,
    transaction_depth: int,
) -> None:
    await conn.execute(
        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
        (json_dumps(metadata), conversation_id),
    )
    if transaction_depth == 0:
        await conn.commit()


async def list_tags(conn: aiosqlite.Connection, *, provider: str | None = None) -> dict[str, int]:
    where = "WHERE metadata IS NOT NULL AND json_extract(metadata, '$.tags') IS NOT NULL"
    params: tuple[str, ...] = ()
    if provider:
        where += " AND provider_name = ?"
        params = (provider,)
    cursor = await conn.execute(
        f"""
        SELECT tag.value AS tag_name, COUNT(*) AS cnt
        FROM conversations,
             json_each(json_extract(metadata, '$.tags')) AS tag
        {where}
        GROUP BY tag.value
        ORDER BY cnt DESC
        """,
        params,
    )
    rows = await cursor.fetchall()
    return {row["tag_name"]: row["cnt"] for row in rows}


__all__ = [
    "conversation_id_query",
    "count_conversation_ids",
    "get_last_sync_timestamp",
    "get_metadata",
    "iter_conversation_ids",
    "list_tags",
    "resolve_id",
    "set_metadata",
    "update_metadata_raw",
]
