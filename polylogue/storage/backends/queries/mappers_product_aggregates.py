"""Row mappers for aggregate and governance product records."""

from __future__ import annotations

import sqlite3

from polylogue.storage.backends.queries.mappers_support import _parse_json, _row_get
from polylogue.storage.store import (
    DaySessionSummaryRecord,
    MaintenanceRunRecord,
    SessionTagRollupRecord,
)


def _row_to_session_tag_rollup_record(row: sqlite3.Row) -> SessionTagRollupRecord:
    return SessionTagRollupRecord(
        tag=row["tag"],
        bucket_day=row["bucket_day"],
        provider_name=row["provider_name"],
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_get(row, "source_updated_at"),
        source_sort_key=_row_get(row, "source_sort_key"),
        conversation_count=int(_row_get(row, "conversation_count", 0) or 0),
        explicit_count=int(_row_get(row, "explicit_count", 0) or 0),
        auto_count=int(_row_get(row, "auto_count", 0) or 0),
        project_breakdown=_parse_json(
            row["project_breakdown_json"],
            field="project_breakdown_json",
            record_id=f'{row["provider_name"]}:{row["bucket_day"]}:{row["tag"]}',
        )
        or {},
        search_text=row["search_text"],
    )


def _row_to_day_session_summary_record(row: sqlite3.Row) -> DaySessionSummaryRecord:
    return DaySessionSummaryRecord(
        day=row["day"],
        provider_name=row["provider_name"],
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_get(row, "source_updated_at"),
        source_sort_key=_row_get(row, "source_sort_key"),
        conversation_count=int(_row_get(row, "conversation_count", 0) or 0),
        total_cost_usd=float(_row_get(row, "total_cost_usd", 0.0) or 0.0),
        total_duration_ms=int(_row_get(row, "total_duration_ms", 0) or 0),
        total_wall_duration_ms=int(_row_get(row, "total_wall_duration_ms", 0) or 0),
        total_messages=int(_row_get(row, "total_messages", 0) or 0),
        total_words=int(_row_get(row, "total_words", 0) or 0),
        work_event_breakdown=_parse_json(
            row["work_event_breakdown_json"],
            field="work_event_breakdown_json",
            record_id=f'{row["provider_name"]}:{row["day"]}',
        )
        or {},
        projects_active=tuple(_parse_json(_row_get(row, "projects_active_json")) or []),
        payload=_parse_json(
            row["payload_json"],
            field="payload_json",
            record_id=f'{row["provider_name"]}:{row["day"]}',
        )
        or {},
        search_text=row["search_text"],
    )


def _row_to_maintenance_run_record(row: sqlite3.Row) -> MaintenanceRunRecord:
    return MaintenanceRunRecord(
        maintenance_run_id=row["maintenance_run_id"],
        schema_version=int(_row_get(row, "schema_version", 1) or 1),
        executed_at=row["executed_at"],
        mode=row["mode"],
        preview=bool(_row_get(row, "preview", 0)),
        repair_selected=bool(_row_get(row, "repair_selected", 0)),
        cleanup_selected=bool(_row_get(row, "cleanup_selected", 0)),
        vacuum_requested=bool(_row_get(row, "vacuum_requested", 0)),
        target_names=tuple(_parse_json(_row_get(row, "target_names_json")) or []),
        success=bool(_row_get(row, "success", 0)),
        manifest=_parse_json(row["manifest_json"], field="manifest_json", record_id=row["maintenance_run_id"]) or {},
    )


__all__ = [
    "_row_to_day_session_summary_record",
    "_row_to_maintenance_run_record",
    "_row_to_session_tag_rollup_record",
]
