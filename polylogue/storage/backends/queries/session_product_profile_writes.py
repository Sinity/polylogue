"""Profile-oriented durable session-product write queries."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.store import SessionProfileRecord, _json_array_or_none, _json_or_none

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


def _session_profile_insert_columns(
    *,
    has_legacy_payload: bool,
) -> list[str]:
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
        "repo_paths_json",
        "repo_names_json",
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
    if has_legacy_payload:
        columns.append("payload_json")
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
    return columns


def _session_profile_insert_values(
    record: SessionProfileRecord,
    *,
    has_legacy_payload: bool,
) -> tuple[object, ...]:
    payload_json = _json_or_none(
        {
            **record.evidence_payload,
            **record.inference_payload,
            "conversation_id": str(record.conversation_id),
            "provider": record.provider_name,
            "title": record.title,
        }
    )
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
        _json_array_or_none(record.repo_paths),
        _json_array_or_none(record.repo_names),
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
    if has_legacy_payload:
        values.append(payload_json)
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
    return tuple(values)


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
        columns = _session_profile_insert_columns(has_legacy_payload=has_legacy_payload)
        placeholders = ", ".join("?" for _ in columns)
        await conn.executemany(
            f"""
            INSERT INTO session_profiles (
                {", ".join(columns)}
            ) VALUES ({placeholders})
            """,
            [
                _session_profile_insert_values(
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
