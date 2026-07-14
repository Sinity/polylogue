"""Derived workload fields for durable live-ingest attempt status."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from polylogue.core.payload_coercion import optional_str as _optional_str
from polylogue.core.payload_coercion import required_str as _required_str
from polylogue.core.payload_coercion import row_float as _row_float
from polylogue.core.payload_coercion import row_int as _row_int


@dataclass(frozen=True, slots=True)
class LiveIngestStageEventInfo:
    archive_write_bytes_delta: int = 0
    total_time_s: float = 0.0
    stage_timings_s: dict[str, float] | None = None


def latest_stage_events(
    conn: sqlite3.Connection,
    attempt_ids: list[str],
) -> dict[str, LiveIngestStageEventInfo]:
    if not attempt_ids:
        return {}
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'live_ingest_stage_event'"
    ).fetchone()
    if has_table is None:
        return {}
    placeholders = ", ".join("?" for _ in attempt_ids)
    rows = conn.execute(
        f"""
        SELECT attempt_id, archive_write_bytes_delta, total_time_s, stage_timings_json
        FROM live_ingest_stage_event
        WHERE attempt_id IN ({placeholders})
        ORDER BY attempt_id, sequence DESC
        """,
        tuple(attempt_ids),
    ).fetchall()
    latest: dict[str, LiveIngestStageEventInfo] = {}
    for row in rows:
        attempt_id = str(row[0])
        latest.setdefault(
            attempt_id,
            LiveIngestStageEventInfo(
                archive_write_bytes_delta=_row_int(row[1]),
                total_time_s=_row_float(row[2]) or 0.0,
                stage_timings_s=_stage_timings(row[3]),
            ),
        )
    return latest


def workload_fields(
    row: sqlite3.Row | tuple[object, ...],
    *,
    stage_event: LiveIngestStageEventInfo | None,
) -> dict[str, object]:
    updated_at = _required_str(row[2])
    input_bytes = _row_int(row[10])
    source_payload_read_bytes = _row_int(row[11])
    cursor_fingerprint_read_bytes = _row_int(row[12])
    total_read_bytes = source_payload_read_bytes + cursor_fingerprint_read_bytes
    total_time_s = (
        stage_event.total_time_s
        if stage_event is not None and stage_event.total_time_s > 0
        else _attempt_elapsed_s(started_at=_required_str(row[1]), ended_at=_optional_str(row[3]) or updated_at)
    )
    return {
        "total_read_bytes": total_read_bytes,
        "read_amplification": round(total_read_bytes / input_bytes, 3) if input_bytes > 0 else 0.0,
        "files_per_second": round(_row_int(row[8]) / total_time_s, 3) if total_time_s > 0 else 0.0,
        "source_mb_per_second": (
            round((source_payload_read_bytes / (1024 * 1024)) / total_time_s, 3) if total_time_s > 0 else 0.0
        ),
        "archive_write_bytes_delta": stage_event.archive_write_bytes_delta if stage_event is not None else 0,
        "total_time_s": total_time_s,
        "stage_timings_s": stage_event.stage_timings_s or {} if stage_event is not None else {},
    }


def _stage_timings(value: object) -> dict[str, float]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): float(item) for key, item in parsed.items() if isinstance(item, int | float)}


def _attempt_elapsed_s(*, started_at: str, ended_at: str) -> float:
    try:
        started = datetime.fromisoformat(started_at)
        ended = datetime.fromisoformat(ended_at)
    except ValueError:
        return 0.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=UTC)
    return max(0.0, round((ended.astimezone(UTC) - started.astimezone(UTC)).total_seconds(), 3))
