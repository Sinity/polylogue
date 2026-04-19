"""Row mappers for aggregate product records."""

from __future__ import annotations

import sqlite3

from polylogue.archive_product_models import DaySessionSummaryPayload
from polylogue.storage.backends.queries.mappers import _parse_json, _row_get
from polylogue.storage.store import (
    DaySessionSummaryRecord,
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
        repo_breakdown=_parse_json(
            row["repo_breakdown_json"],
            field="repo_breakdown_json",
            record_id=f"{row['provider_name']}:{row['bucket_day']}:{row['tag']}",
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
            record_id=f"{row['provider_name']}:{row['day']}",
        )
        or {},
        repos_active=tuple(_parse_json(_row_get(row, "repos_active_json")) or []),
        payload=DaySessionSummaryPayload.model_validate(
            _parse_json(
                row["payload_json"],
                field="payload_json",
                record_id=f"{row['provider_name']}:{row['day']}",
            )
            or {}
        ),
        search_text=row["search_text"],
    )


__all__ = [
    "_row_to_day_session_summary_record",
    "_row_to_session_tag_rollup_record",
]
