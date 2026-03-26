"""Timeline-row storage writes for session products."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from polylogue.storage.session_product_storage_support import table_has_column
from polylogue.storage.store import _json_array_or_none, _json_or_none


def replace_session_work_events_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: Sequence[object],
) -> None:
    conn.execute("DELETE FROM session_work_events WHERE conversation_id = ?", (conversation_id,))
    if records:
        has_legacy_payload = table_has_column(conn, "session_work_events", "payload_json")
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
        conn.executemany(
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


def replace_session_phases_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: Sequence[object],
) -> None:
    conn.execute("DELETE FROM session_phases WHERE conversation_id = ?", (conversation_id,))
    if records:
        has_legacy_payload = table_has_column(conn, "session_phases", "payload_json")
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
        conn.executemany(
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


__all__ = [
    "replace_session_phases_sync",
    "replace_session_work_events_sync",
]
