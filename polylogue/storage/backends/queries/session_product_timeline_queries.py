"""Timeline-oriented durable session-product queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import (
    _row_to_session_phase_record,
    _row_to_session_work_event_record,
)
from polylogue.storage.store import (
    SessionPhaseRecord,
    SessionWorkEventRecord,
    _json_array_or_none,
    _json_or_none,
)

_ASYNC_COLUMN_CACHE: dict[tuple[int, str], bool] = {}

__all__ = [
    "get_session_phases",
    "get_work_events",
    "list_session_phases",
    "list_work_events",
    "replace_session_phases",
    "replace_session_work_events",
]


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


async def get_work_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[SessionWorkEventRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM session_work_events
        WHERE conversation_id = ?
        ORDER BY event_index
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session_work_event_record(row) for row in rows]


async def get_session_phases(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[SessionPhaseRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM session_phases
        WHERE conversation_id = ?
        ORDER BY phase_index
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session_phase_record(row) for row in rows]


async def list_work_events(
    conn: aiosqlite.Connection,
    *,
    conversation_id: str | None = None,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    kind: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
    query: str | None = None,
) -> list[SessionWorkEventRecord]:
    params: list[object] = []
    if query:
        from_clause = """
            FROM session_work_events swe
            JOIN session_work_events_fts
              ON session_work_events_fts.event_id = swe.event_id
        """
        where = ["session_work_events_fts MATCH ?"]
        params.append(query)
        order_by = "ORDER BY bm25(session_work_events_fts), COALESCE(swe.source_sort_key, 0) DESC, swe.event_index"
    else:
        from_clause = "FROM session_work_events swe"
        where = []
        order_by = "ORDER BY COALESCE(swe.source_sort_key, 0) DESC, swe.event_index"

    if conversation_id:
        where.append("swe.conversation_id = ?")
        params.append(conversation_id)
    if provider:
        where.append("swe.provider_name = ?")
        params.append(provider)
    if kind:
        where.append("swe.kind = ?")
        params.append(kind)
    if since:
        where.append("COALESCE(swe.end_time, swe.start_time, swe.source_updated_at, swe.materialized_at) >= ?")
        params.append(since)
    if until:
        where.append("COALESCE(swe.start_time, swe.end_time, swe.source_updated_at, swe.materialized_at) <= ?")
        params.append(until)

    sql = "SELECT swe.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_work_event_record(row) for row in rows]


async def list_session_phases(
    conn: aiosqlite.Connection,
    *,
    conversation_id: str | None = None,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    kind: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
) -> list[SessionPhaseRecord]:
    params: list[object] = []
    where: list[str] = []
    if conversation_id:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if provider:
        where.append("provider_name = ?")
        params.append(provider)
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if since:
        where.append("COALESCE(end_time, start_time, source_updated_at, materialized_at) >= ?")
        params.append(since)
    if until:
        where.append("COALESCE(start_time, end_time, source_updated_at, materialized_at) <= ?")
        params.append(until)

    sql = """
        SELECT *
        FROM session_phases
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(start_time, end_time, materialized_at) DESC, phase_index"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_phase_record(row) for row in rows]


async def replace_session_work_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
    records: list[SessionWorkEventRecord],
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM session_work_events WHERE conversation_id = ?",
        (conversation_id,),
    )
    if records:
        has_legacy_payload = await _table_has_column(conn, "session_work_events", "payload_json")
        columns = [
            "event_id",
            "conversation_id",
            "materializer_version",
            "materialized_at",
            "source_updated_at",
            "source_sort_key",
            "provider_name",
            "event_index",
            "kind",
            "confidence",
            "start_index",
            "end_index",
            "start_time",
            "end_time",
            "duration_ms",
            "canonical_session_date",
            "summary",
            "file_paths_json",
            "tools_used_json",
        ]
        if has_legacy_payload:
            columns.append("payload_json")
        columns.extend(
            [
                "evidence_payload_json",
                "inference_payload_json",
                "search_text",
                "inference_version",
                "inference_family",
            ]
        )
        placeholders = ", ".join("?" for _ in columns)
        await conn.executemany(
            f"""
            INSERT INTO session_work_events (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            [
                tuple(
                    [
                        record.event_id,
                        record.conversation_id,
                        record.materializer_version,
                        record.materialized_at,
                        record.source_updated_at,
                        record.source_sort_key,
                        record.provider_name,
                        record.event_index,
                        record.kind,
                        record.confidence,
                        record.start_index,
                        record.end_index,
                        record.start_time,
                        record.end_time,
                        record.duration_ms,
                        record.canonical_session_date,
                        record.summary,
                        _json_array_or_none(record.file_paths),
                        _json_array_or_none(record.tools_used),
                    ]
                    + (
                        [
                            _json_or_none(
                                {
                                    **record.evidence_payload,
                                    **record.inference_payload,
                                }
                            )
                        ]
                        if has_legacy_payload
                        else []
                    )
                    + [
                        _json_or_none(record.evidence_payload),
                        _json_or_none(record.inference_payload),
                        record.search_text,
                        record.inference_version,
                        record.inference_family,
                    ]
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
    await conn.execute(
        "DELETE FROM session_phases WHERE conversation_id = ?",
        (conversation_id,),
    )
    if records:
        has_legacy_payload = await _table_has_column(conn, "session_phases", "payload_json")
        columns = [
            "phase_id",
            "conversation_id",
            "materializer_version",
            "materialized_at",
            "source_updated_at",
            "source_sort_key",
            "provider_name",
            "phase_index",
            "kind",
            "start_index",
            "end_index",
            "start_time",
            "end_time",
            "duration_ms",
            "canonical_session_date",
            "confidence",
            "evidence_reasons_json",
            "tool_counts_json",
            "word_count",
        ]
        if has_legacy_payload:
            columns.append("payload_json")
        columns.extend(
            [
                "evidence_payload_json",
                "inference_payload_json",
                "search_text",
                "inference_version",
                "inference_family",
            ]
        )
        placeholders = ", ".join("?" for _ in columns)
        await conn.executemany(
            f"""
            INSERT INTO session_phases (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            [
                tuple(
                    [
                        record.phase_id,
                        record.conversation_id,
                        record.materializer_version,
                        record.materialized_at,
                        record.source_updated_at,
                        record.source_sort_key,
                        record.provider_name,
                        record.phase_index,
                        record.kind,
                        record.start_index,
                        record.end_index,
                        record.start_time,
                        record.end_time,
                        record.duration_ms,
                        record.canonical_session_date,
                        record.confidence,
                        _json_array_or_none(record.evidence_reasons),
                        _json_or_none(record.tool_counts),
                        record.word_count,
                    ]
                    + (
                        [
                            _json_or_none(
                                {
                                    **record.evidence_payload,
                                    **record.inference_payload,
                                }
                            )
                        ]
                        if has_legacy_payload
                        else []
                    )
                    + [
                        _json_or_none(record.evidence_payload),
                        _json_or_none(record.inference_payload),
                        record.search_text,
                        record.inference_version,
                        record.inference_family,
                    ]
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()
