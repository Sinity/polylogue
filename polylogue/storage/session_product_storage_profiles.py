"""Profile-row storage writes for session products."""

from __future__ import annotations

import sqlite3

from polylogue.storage.session_product_storage_support import table_has_column
from polylogue.storage.store import SessionProfileRecord, _json_array_or_none, _json_or_none


def replace_session_profile_sync(conn: sqlite3.Connection, record: SessionProfileRecord) -> None:
    conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (record.conversation_id,))
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
    if table_has_column(conn, "session_profiles", "payload_json"):
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
    conn.execute(
        f"""
        INSERT INTO session_profiles (
            {", ".join(columns)}
        ) VALUES ({placeholders})
        """,
        tuple(values),
    )


__all__ = ["replace_session_profile_sync"]
