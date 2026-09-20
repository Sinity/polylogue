"""Row mappers for thread read-model records."""

from __future__ import annotations

import sqlite3

from polylogue.analysis.archive_models import ThreadPayload
from polylogue.core.types import SessionId
from polylogue.storage.runtime import ThreadRecord
from polylogue.storage.sqlite.queries.mappers_support import (
    _json_text_tuple,
    _parse_json,
    _row_float,
    _row_get,
    _row_int,
    _row_text,
)


def _row_to_thread_record(row: sqlite3.Row) -> ThreadRecord:
    return ThreadRecord(
        thread_id=row["thread_id"],
        root_id=SessionId(row["root_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        input_high_water_mark=_row_text(row, "input_high_water_mark"),
        input_high_water_mark_source=_row_text(row, "input_high_water_mark_source"),
        input_row_count=int(_row_int(row, "input_row_count", 0) or 0),
        start_time=_row_text(row, "start_time"),
        end_time=_row_text(row, "end_time"),
        dominant_repo=_row_text(row, "dominant_repo"),
        session_ids=_json_text_tuple(_parse_json(_row_get(row, "session_ids_json"))),
        session_count=int(_row_int(row, "session_count", 0) or 0),
        depth=int(_row_int(row, "depth", 0) or 0),
        branch_count=int(_row_int(row, "branch_count", 0) or 0),
        total_messages=int(_row_int(row, "total_messages", 0) or 0),
        total_cost_usd=float(_row_float(row, "total_cost_usd", 0.0) or 0.0),
        wall_duration_ms=int(_row_int(row, "wall_duration_ms", 0) or 0),
        payload=ThreadPayload.model_validate(
            _parse_json(row["payload_json"], field="payload_json", record_id=row["thread_id"]) or {}
        ),
        search_text=row["search_text"],
    )


__all__ = [
    "_row_to_thread_record",
]
