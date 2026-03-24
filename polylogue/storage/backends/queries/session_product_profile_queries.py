"""Profile-oriented durable session-product queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.store import (
    SessionProfileRecord,
    _json_array_or_none,
    _json_or_none,
)

_ASYNC_COLUMN_CACHE: dict[tuple[int, str], bool] = {}

__all__ = [
    "get_session_profile",
    "get_session_profiles_batch",
    "list_session_profiles",
    "replace_session_profile",
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
    tier: str = "merged",
    refined_work_kind: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
    query: str | None = None,
) -> list[SessionProfileRecord]:
    params: list[object] = []
    if query:
        fts_table = {
            "evidence": "session_profile_evidence_fts",
            "inference": "session_profile_inference_fts",
            "enrichment": "session_profile_enrichment_fts",
            "merged": "session_profiles_fts",
        }.get(tier, "session_profiles_fts")
        from_clause = f"""
            FROM session_profiles sp
            JOIN {fts_table}
              ON {fts_table}.conversation_id = sp.conversation_id
        """
        where = [f"{fts_table} MATCH ?"]
        params.append(query)
        order_by = f"ORDER BY bm25({fts_table}), COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id"
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
    if refined_work_kind:
        where.append(
            "COALESCE(NULLIF(json_extract(sp.enrichment_payload_json, '$.refined_work_kind'), ''), sp.primary_work_kind) = ?"
        )
        params.append(refined_work_kind)

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


async def replace_session_profile(
    conn: aiosqlite.Connection,
    record: SessionProfileRecord,
    transaction_depth: int,
) -> None:
    await conn.execute(
        "DELETE FROM session_profiles WHERE conversation_id = ?",
        (record.conversation_id,),
    )
    payload_json = _json_or_none(
        {
            **record.evidence_payload,
            **record.inference_payload,
            "conversation_id": str(record.conversation_id),
            "provider": record.provider_name,
            "title": record.title,
        }
    )
    columns = [
        "conversation_id",
        "materializer_version",
        "materialized_at",
        "source_updated_at",
        "source_sort_key",
        "provider_name",
        "title",
        "first_message_at",
        "last_message_at",
        "canonical_session_date",
        "primary_work_kind",
        "repo_paths_json",
        "canonical_projects_json",
        "tags_json",
        "auto_tags_json",
        "message_count",
        "substantive_count",
        "attachment_count",
        "work_event_count",
        "phase_count",
        "word_count",
        "tool_use_count",
        "thinking_count",
        "total_cost_usd",
        "total_duration_ms",
        "engaged_duration_ms",
        "wall_duration_ms",
        "cost_is_estimated",
    ]
    values: list[object] = [
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
        record.substantive_count,
        record.attachment_count,
        record.work_event_count,
        record.phase_count,
        record.word_count,
        record.tool_use_count,
        record.thinking_count,
        record.total_cost_usd,
        record.total_duration_ms,
        record.engaged_duration_ms,
        record.wall_duration_ms,
        int(record.cost_is_estimated),
    ]
    if await _table_has_column(conn, "session_profiles", "payload_json"):
        columns.append("payload_json")
        values.append(payload_json)
    columns.extend(
        [
            "evidence_payload_json",
            "inference_payload_json",
            "enrichment_payload_json",
            "search_text",
            "evidence_search_text",
            "inference_search_text",
            "enrichment_search_text",
            "enrichment_version",
            "enrichment_family",
            "inference_version",
            "inference_family",
        ]
    )
    values.extend(
        [
            _json_or_none(record.evidence_payload),
            _json_or_none(record.inference_payload),
            _json_or_none(record.enrichment_payload),
            record.search_text,
            record.evidence_search_text,
            record.inference_search_text,
            record.enrichment_search_text,
            record.enrichment_version,
            record.enrichment_family,
            record.inference_version,
            record.inference_family,
        ]
    )
    placeholders = ", ".join("?" for _ in columns)
    await conn.execute(
        f"""
        INSERT INTO session_profiles (
            {", ".join(columns)}
        ) VALUES ({placeholders})
        """,
        tuple(values),
    )
    if transaction_depth == 0:
        await conn.commit()
