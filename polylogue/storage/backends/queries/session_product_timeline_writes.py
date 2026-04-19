"""Timeline-oriented durable session-product write queries."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.store import (
    SessionPhaseRecord,
    SessionWorkEventRecord,
    _json_array_or_none,
    _json_or_none,
)

__all__ = [
    "replace_session_phases",
    "replace_session_phases_bulk",
    "replace_session_work_events",
    "replace_session_work_events_bulk",
]

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


async def replace_session_work_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
    records: list[SessionWorkEventRecord],
    transaction_depth: int,
) -> None:
    await replace_session_work_events_bulk(
        conn,
        [conversation_id],
        records,
        transaction_depth,
    )


async def replace_session_work_events_bulk(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    records: Sequence[SessionWorkEventRecord],
    transaction_depth: int,
) -> None:
    if conversation_ids:
        placeholders = ", ".join("?" for _ in conversation_ids)
        await conn.execute(
            f"DELETE FROM session_work_events WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
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
                                    **record.evidence_payload.model_dump(mode="json"),
                                    **record.inference_payload.model_dump(mode="json"),
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
    await replace_session_phases_bulk(
        conn,
        [conversation_id],
        records,
        transaction_depth,
    )


async def replace_session_phases_bulk(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    records: Sequence[SessionPhaseRecord],
    transaction_depth: int,
) -> None:
    if conversation_ids:
        placeholders = ", ".join("?" for _ in conversation_ids)
        await conn.execute(
            f"DELETE FROM session_phases WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
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
                        _json_array_or_none(record.evidence_reasons) or "[]",
                        _json_or_none(record.tool_counts),
                        record.word_count,
                    ]
                    + (
                        [
                            _json_or_none(
                                {
                                    **record.evidence_payload.model_dump(mode="json"),
                                    **record.inference_payload.model_dump(mode="json"),
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
