"""Read queries over source-derived and materialized run-projection relations."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.query_models import RunProjectionListQuery
from polylogue.storage.runtime import (
    SessionContextSnapshotRecord,
    SessionObservedEventRecord,
    SessionRunRecord,
)
from polylogue.storage.sqlite.run_projection_relations import (
    context_snapshot_relation_sql,
    observed_event_relation_sql,
    row_to_session_context_snapshot_record,
    row_to_session_observed_event_record,
    row_to_session_run_record,
    run_relation_sql,
)

__all__ = [
    "list_context_snapshots",
    "list_observed_events",
    "list_runs",
]


def _order_by(query: RunProjectionListQuery, alias: str, pk_column: str) -> str:
    if query.sort == "recency":
        return f"ORDER BY COALESCE({alias}.source_updated_at, {alias}.materialized_at) DESC, {alias}.position, {alias}.{pk_column}"
    if query.sort != "position":
        raise ValueError(f"unsupported run-projection sort: {query.sort!r} (expected 'position' or 'recency')")
    return f"ORDER BY {alias}.session_id, {alias}.position, {alias}.{pk_column}"


def _apply_limit(sql: str, params: list[object], query: RunProjectionListQuery) -> str:
    if query.limit is not None:
        params.extend([query.limit, query.offset])
        return sql + " LIMIT ? OFFSET ?"
    return sql


async def _table_exists(conn: aiosqlite.Connection, table_name: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return await cursor.fetchone() is not None


async def list_runs(
    conn: aiosqlite.Connection,
    query: RunProjectionListQuery,
) -> list[SessionRunRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.session_id:
        where.append("r.session_id = ?")
        params.append(query.session_id)
    if query.run_ref:
        where.append("r.run_ref = ?")
        params.append(query.run_ref)
    if query.harness:
        where.append("r.harness = ?")
        params.append(query.harness)
    if query.role:
        where.append("r.role = ?")
        params.append(query.role)
    if query.status:
        where.append("r.status = ?")
        params.append(query.status)
    if query.query:
        where.append("r.search_text LIKE ?")
        params.append(f"%{query.query}%")
    relation_sql = run_relation_sql(include_materialized=await _table_exists(conn, "session_runs"))
    sql = f"{relation_sql} SELECT r.* FROM runs r"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " " + _order_by(query, "r", "run_ref")
    sql = _apply_limit(sql, params, query)
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [row_to_session_run_record(row) for row in rows]


async def list_observed_events(
    conn: aiosqlite.Connection,
    query: RunProjectionListQuery,
) -> list[SessionObservedEventRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.session_id:
        where.append("e.session_id = ?")
        params.append(query.session_id)
    if query.run_ref:
        where.append("e.run_ref = ?")
        params.append(query.run_ref)
    if query.kind:
        where.append("e.kind = ?")
        params.append(query.kind)
    if query.delivery_state:
        where.append("e.delivery_state = ?")
        params.append(query.delivery_state)
    if query.query:
        where.append("e.search_text LIKE ?")
        params.append(f"%{query.query}%")
    source_where = "1=1"
    relation_sql = observed_event_relation_sql(
        source_where=source_where,
        include_materialized=await _table_exists(conn, "session_observed_events"),
    )
    sql = f"{relation_sql} SELECT e.* FROM observed_events e"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " " + _order_by(query, "e", "event_ref")
    sql = _apply_limit(sql, params, query)
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [row_to_session_observed_event_record(row) for row in rows]


async def list_context_snapshots(
    conn: aiosqlite.Connection,
    query: RunProjectionListQuery,
) -> list[SessionContextSnapshotRecord]:
    params: list[object] = []
    where: list[str] = []
    if query.session_id:
        where.append("c.session_id = ?")
        params.append(query.session_id)
    if query.run_ref:
        where.append("c.run_ref = ?")
        params.append(query.run_ref)
    if query.boundary:
        where.append("c.boundary = ?")
        params.append(query.boundary)
    if query.inheritance_mode:
        where.append("c.inheritance_mode = ?")
        params.append(query.inheritance_mode)
    if query.query:
        where.append("c.search_text LIKE ?")
        params.append(f"%{query.query}%")
    relation_sql = context_snapshot_relation_sql(
        include_materialized=await _table_exists(conn, "session_context_snapshots")
    )
    sql = f"{relation_sql} SELECT c.* FROM context_snapshots c"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " " + _order_by(query, "c", "snapshot_ref")
    sql = _apply_limit(sql, params, query)
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [row_to_session_context_snapshot_record(row) for row in rows]
