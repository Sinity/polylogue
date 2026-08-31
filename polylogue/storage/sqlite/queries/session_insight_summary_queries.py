"""Summary/rollup durable session-insight queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.query_models import SessionTagRollupListQuery
from polylogue.storage.runtime import SessionTagRollupRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_session_tag_rollup_record

__all__ = [
    "list_session_tag_rollup_rows",
]


async def list_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    query: SessionTagRollupListQuery,
) -> list[SessionTagRollupRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.origin:
        where.append("source_name = ?")
        params.append(query.origin)
    if query.since:
        where.append("bucket_day >= date(?)")
        params.append(query.since)
    if query.until:
        where.append("bucket_day <= date(?)")
        params.append(query.until)
    if query.query:
        where.append("LOWER(tag) LIKE ?")
        params.append(f"%{query.query.strip().lower()}%")

    sql = """
        SELECT *
        FROM session_tag_rollups
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY bucket_day DESC, source_name, tag"

    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_tag_rollup_record(row) for row in rows]
