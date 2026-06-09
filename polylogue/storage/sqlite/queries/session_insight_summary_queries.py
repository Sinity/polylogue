"""Summary/rollup durable session-insight queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.insights.session.storage import session_tag_rollup_insert_values
from polylogue.storage.query_models import SessionTagRollupListQuery
from polylogue.storage.runtime import SessionTagRollupRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_session_tag_rollup_record

__all__ = [
    "list_session_tag_rollup_rows",
    "replace_session_tag_rollup_rows",
]


async def list_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    query: SessionTagRollupListQuery,
) -> list[SessionTagRollupRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.provider:
        where.append("source_name = ?")
        params.append(query.provider)
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


async def replace_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    *,
    source_name: str,
    bucket_day: str,
    records: list[SessionTagRollupRecord],
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM session_tag_rollups WHERE source_name = ? AND bucket_day = ?",
        (source_name, bucket_day),
    )
    if records:
        await conn.executemany(
            """
            INSERT INTO session_tag_rollups (
                tag,
                bucket_day,
                source_name,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                input_high_water_mark,
                input_high_water_mark_source,
                input_row_count,
                session_count,
                logical_session_count,
                logical_session_ids_json,
                explicit_count,
                auto_count,
                repo_breakdown_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [session_tag_rollup_insert_values(record) for record in records],
        )
    if transaction_depth == 0:
        await conn.commit()
