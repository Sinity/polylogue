"""Queries for durable semantic/session product read models."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import (
    _row_to_day_session_summary_record,
    _row_to_session_phase_record,
    _row_to_session_profile_record,
    _row_to_session_tag_rollup_record,
    _row_to_session_work_event_record,
    _row_to_work_thread_record,
)
from polylogue.storage.store import (
    DaySessionSummaryRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
    _json_array_or_none,
    _json_or_none,
)

__all__ = [
    "get_session_profile",
    "get_session_profiles_batch",
    "get_session_phases",
    "get_work_events",
    "get_work_thread",
    "list_day_session_summaries",
    "list_session_phases",
    "list_session_profiles",
    "list_session_tag_rollup_rows",
    "list_work_events",
    "list_work_threads",
    "replace_day_session_summaries",
    "replace_session_phases",
    "replace_session_profile",
    "replace_session_tag_rollup_rows",
    "replace_session_work_events",
    "replace_work_thread",
]


async def get_session_profile(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> SessionProfileRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM session_profiles WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    return _row_to_session_profile_record(row) if row else None


async def get_session_profiles_batch(
    conn: aiosqlite.Connection,
    conversation_ids: list[str],
) -> dict[str, SessionProfileRecord]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    cursor = await conn.execute(
        f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
        tuple(conversation_ids),
    )
    rows = await cursor.fetchall()
    return {str(row["conversation_id"]): _row_to_session_profile_record(row) for row in rows}


async def list_session_profiles(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    first_message_since: str | None = None,
    first_message_until: str | None = None,
    session_date_since: str | None = None,
    session_date_until: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
    query: str | None = None,
) -> list[SessionProfileRecord]:
    params: list[object] = []
    if query:
        from_clause = """
            FROM session_profiles sp
            JOIN session_profiles_fts
              ON session_profiles_fts.conversation_id = sp.conversation_id
        """
        where = ["session_profiles_fts MATCH ?"]
        params.append(query)
        order_by = "ORDER BY bm25(session_profiles_fts), COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"
    else:
        from_clause = "FROM session_profiles sp"
        where = []
        order_by = "ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"

    if provider:
        where.append("sp.provider_name = ?")
        params.append(provider)
    if since:
        where.append(
            "COALESCE(sp.last_message_at, sp.source_updated_at, sp.first_message_at) >= ?"
        )
        params.append(since)
    if until:
        where.append(
            "COALESCE(sp.first_message_at, sp.source_updated_at, sp.last_message_at) <= ?"
        )
        params.append(until)
    if first_message_since:
        where.append("sp.first_message_at >= ?")
        params.append(first_message_since)
    if first_message_until:
        where.append("sp.first_message_at <= ?")
        params.append(first_message_until)
    if session_date_since:
        where.append("sp.canonical_session_date >= date(?)")
        params.append(session_date_since)
    if session_date_until:
        where.append("sp.canonical_session_date <= date(?)")
        params.append(session_date_until)

    sql = "SELECT sp.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


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


async def get_work_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
) -> WorkThreadRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM work_threads WHERE thread_id = ?",
        (thread_id,),
    )
    row = await cursor.fetchone()
    return _row_to_work_thread_record(row) if row else None


async def list_work_threads(
    conn: aiosqlite.Connection,
    *,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
    query: str | None = None,
) -> list[WorkThreadRecord]:
    params: list[object] = []
    if query:
        from_clause = """
            FROM work_threads wt
            JOIN work_threads_fts
              ON work_threads_fts.thread_id = wt.thread_id
        """
        where = ["work_threads_fts MATCH ?"]
        params.append(query)
        order_by = "ORDER BY bm25(work_threads_fts), COALESCE(wt.end_time, wt.start_time, wt.materialized_at) DESC, wt.thread_id"
    else:
        from_clause = "FROM work_threads wt"
        where = []
        order_by = "ORDER BY COALESCE(wt.end_time, wt.start_time, wt.materialized_at) DESC, wt.thread_id"
    if since:
        where.append("COALESCE(wt.end_time, wt.start_time, wt.materialized_at) >= ?")
        params.append(since)
    if until:
        where.append("COALESCE(wt.start_time, wt.end_time, wt.materialized_at) <= ?")
        params.append(until)
    sql = "SELECT wt.* " + from_clause
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by}"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_work_thread_record(row) for row in rows]


async def list_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    query: str | None = None,
) -> list[SessionTagRollupRecord]:
    params: list[object] = []
    where: list[str] = []
    if provider:
        where.append("provider_name = ?")
        params.append(provider)
    if since:
        where.append("bucket_day >= date(?)")
        params.append(since)
    if until:
        where.append("bucket_day <= date(?)")
        params.append(until)
    if query:
        where.append("LOWER(tag) LIKE ?")
        params.append(f"%{query.strip().lower()}%")

    sql = """
        SELECT *
        FROM session_tag_rollups
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY bucket_day DESC, provider_name, tag"

    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session_tag_rollup_record(row) for row in rows]


async def list_day_session_summaries(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[DaySessionSummaryRecord]:
    params: list[object] = []
    where: list[str] = []
    if provider:
        where.append("provider_name = ?")
        params.append(provider)
    if since:
        where.append("day >= date(?)")
        params.append(since)
    if until:
        where.append("day <= date(?)")
        params.append(until)

    sql = """
        SELECT *
        FROM day_session_summaries
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY day DESC, provider_name"

    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_day_session_summary_record(row) for row in rows]


async def replace_session_profile(
    conn: aiosqlite.Connection,
    record: SessionProfileRecord,
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM session_profiles WHERE conversation_id = ?",
        (record.conversation_id,),
    )
    await conn.execute(
        """
        INSERT INTO session_profiles (
            conversation_id,
            materializer_version,
            materialized_at,
            source_updated_at,
            source_sort_key,
            provider_name,
            title,
            first_message_at,
            last_message_at,
            canonical_session_date,
            primary_work_kind,
            repo_paths_json,
            canonical_projects_json,
            tags_json,
            auto_tags_json,
            message_count,
            work_event_count,
            phase_count,
            word_count,
            tool_use_count,
            thinking_count,
            total_cost_usd,
            total_duration_ms,
            engaged_duration_ms,
            wall_duration_ms,
            payload_json,
            search_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.conversation_id,
            record.materializer_version,
            record.materialized_at,
            record.source_updated_at,
            record.source_sort_key,
            record.provider_name,
            record.title,
            record.first_message_at,
            record.last_message_at,
            record.canonical_session_date,
            record.primary_work_kind,
            _json_array_or_none(record.repo_paths),
            _json_array_or_none(record.canonical_projects),
            _json_array_or_none(record.tags),
            _json_array_or_none(record.auto_tags),
            record.message_count,
            record.work_event_count,
            record.phase_count,
            record.word_count,
            record.tool_use_count,
            record.thinking_count,
            record.total_cost_usd,
            record.total_duration_ms,
            record.engaged_duration_ms,
            record.wall_duration_ms,
            _json_or_none(record.payload),
            record.search_text,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


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
        await conn.executemany(
            """
            INSERT INTO session_work_events (
                event_id,
                conversation_id,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                provider_name,
                event_index,
                kind,
                confidence,
                start_index,
                end_index,
                start_time,
                end_time,
                duration_ms,
                canonical_session_date,
                summary,
                file_paths_json,
                tools_used_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
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
                    _json_or_none(record.payload),
                    record.search_text,
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
        await conn.executemany(
            """
            INSERT INTO session_phases (
                phase_id,
                conversation_id,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                provider_name,
                phase_index,
                kind,
                start_index,
                end_index,
                start_time,
                end_time,
                duration_ms,
                canonical_session_date,
                tool_counts_json,
                word_count,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
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
                    _json_or_none(record.tool_counts),
                    record.word_count,
                    _json_or_none(record.payload),
                    record.search_text,
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()


async def replace_work_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
    record: WorkThreadRecord | None,
    transaction_depth: int,
) -> None:
    await conn.execute("DELETE FROM work_threads WHERE thread_id = ?", (thread_id,))
    if record is not None:
        await conn.execute(
            """
            INSERT INTO work_threads (
                thread_id,
                root_id,
                materializer_version,
                materialized_at,
                start_time,
                end_time,
                dominant_project,
                session_ids_json,
                session_count,
                depth,
                branch_count,
                total_messages,
                total_cost_usd,
                wall_duration_ms,
                work_event_breakdown_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.thread_id,
                record.root_id,
                record.materializer_version,
                record.materialized_at,
                record.start_time,
                record.end_time,
                record.dominant_project,
                _json_array_or_none(record.session_ids),
                record.session_count,
                record.depth,
                record.branch_count,
                record.total_messages,
                record.total_cost_usd,
                record.wall_duration_ms,
                _json_or_none(record.work_event_breakdown or {}),
                _json_or_none(record.payload),
                record.search_text,
            ),
    )
    if transaction_depth == 0:
        await conn.commit()


async def replace_session_tag_rollup_rows(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    bucket_day: str,
    records: list[SessionTagRollupRecord],
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM session_tag_rollups WHERE provider_name = ? AND bucket_day = ?",
        (provider_name, bucket_day),
    )
    if records:
        await conn.executemany(
            """
            INSERT INTO session_tag_rollups (
                tag,
                bucket_day,
                provider_name,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                conversation_count,
                explicit_count,
                auto_count,
                project_breakdown_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.tag,
                    record.bucket_day,
                    record.provider_name,
                    record.materializer_version,
                    record.materialized_at,
                    record.source_updated_at,
                    record.source_sort_key,
                    record.conversation_count,
                    record.explicit_count,
                    record.auto_count,
                    _json_or_none(record.project_breakdown),
                    record.search_text,
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()


async def replace_day_session_summaries(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    day: str,
    records: list[DaySessionSummaryRecord],
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM day_session_summaries WHERE provider_name = ? AND day = ?",
        (provider_name, day),
    )
    if records:
        await conn.executemany(
            """
            INSERT INTO day_session_summaries (
                day,
                provider_name,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                conversation_count,
                total_cost_usd,
                total_duration_ms,
                total_wall_duration_ms,
                total_messages,
                total_words,
                work_event_breakdown_json,
                projects_active_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.day,
                    record.provider_name,
                    record.materializer_version,
                    record.materialized_at,
                    record.source_updated_at,
                    record.source_sort_key,
                    record.conversation_count,
                    record.total_cost_usd,
                    record.total_duration_ms,
                    record.total_wall_duration_ms,
                    record.total_messages,
                    record.total_words,
                    _json_or_none(record.work_event_breakdown),
                    _json_array_or_none(record.projects_active),
                    _json_or_none(record.payload),
                    record.search_text,
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()
