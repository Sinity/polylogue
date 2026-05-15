"""Status projection for daemon catch-up progress and throughput."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.storage.sqlite.connection_profile import open_readonly_connection


class CatchupStageEvent(BaseModel):
    attempt_id: str
    sequence: int
    observed_at: str
    phase: str
    status: str
    queued_file_count: int = 0
    needed_file_count: int = 0
    skipped_file_count: int = 0
    succeeded_file_count: int = 0
    failed_file_count: int = 0
    input_bytes: int = 0
    source_payload_read_bytes: int = 0
    cursor_fingerprint_read_bytes: int = 0
    archive_write_bytes_delta: int = 0
    parse_time_s: float = 0.0
    convergence_time_s: float = 0.0
    total_time_s: float = 0.0
    current_source: str | None = None
    current_path: str | None = None
    error: str | None = None


class CatchupStatus(BaseModel):
    mode: str = "idle"
    current_phase: str | None = None
    current_source: str | None = None
    current_path: str | None = None
    queued_file_count: int = 0
    needed_file_count: int = 0
    skipped_file_count: int = 0
    succeeded_file_count: int = 0
    failed_file_count: int = 0
    input_bytes: int = 0
    source_payload_read_bytes: int = 0
    cursor_fingerprint_read_bytes: int = 0
    archive_write_bytes_delta: int = 0
    read_amplification: float = 0.0
    files_per_second: float = 0.0
    source_mb_per_second: float = 0.0
    parse_time_s: float = 0.0
    convergence_time_s: float = 0.0
    total_time_s: float = 0.0
    latest_event_age_s: float | None = None
    recent_events: list[CatchupStageEvent] = Field(default_factory=list)


def catchup_status_info(
    dbf: Path,
    *,
    latest_attempt: object | None,
    convergence: object,
) -> CatchupStatus:
    """Return bounded catch-up/convergence progress and throughput from durable events."""
    events = _recent_stage_events(dbf)
    latest = events[0] if events else None
    now = datetime.now(UTC)
    mode = _catchup_mode(latest, latest_attempt, convergence)
    if latest is not None:
        total_time_s = latest.total_time_s or latest.parse_time_s + latest.convergence_time_s
        return CatchupStatus(
            mode=mode,
            current_phase=latest.phase,
            current_source=latest.current_source,
            current_path=latest.current_path,
            queued_file_count=latest.queued_file_count,
            needed_file_count=latest.needed_file_count,
            skipped_file_count=latest.skipped_file_count,
            succeeded_file_count=latest.succeeded_file_count,
            failed_file_count=latest.failed_file_count,
            input_bytes=latest.input_bytes,
            source_payload_read_bytes=latest.source_payload_read_bytes,
            cursor_fingerprint_read_bytes=latest.cursor_fingerprint_read_bytes,
            archive_write_bytes_delta=latest.archive_write_bytes_delta,
            read_amplification=round(_ratio(latest.source_payload_read_bytes, latest.input_bytes), 4),
            files_per_second=round(_ratio(latest.succeeded_file_count, total_time_s), 3),
            source_mb_per_second=round(_ratio(latest.source_payload_read_bytes / 1_000_000, total_time_s), 3),
            parse_time_s=latest.parse_time_s,
            convergence_time_s=latest.convergence_time_s,
            total_time_s=total_time_s,
            latest_event_age_s=_iso_age_s(latest.observed_at, now=now),
            recent_events=events,
        )
    if latest_attempt is None:
        return CatchupStatus(mode=mode)
    total_time_s = _float_attr(latest_attempt, "parse_time_s") + _float_attr(latest_attempt, "convergence_time_s")
    return CatchupStatus(
        mode=mode,
        current_phase=_str_attr(latest_attempt, "phase"),
        current_source=_str_attr(latest_attempt, "current_source"),
        current_path=_str_attr(latest_attempt, "current_path"),
        queued_file_count=_int_attr(latest_attempt, "queued_file_count"),
        needed_file_count=_int_attr(latest_attempt, "needed_file_count"),
        succeeded_file_count=_int_attr(latest_attempt, "succeeded_file_count"),
        failed_file_count=_int_attr(latest_attempt, "failed_file_count"),
        input_bytes=_int_attr(latest_attempt, "input_bytes"),
        source_payload_read_bytes=_int_attr(latest_attempt, "source_payload_read_bytes"),
        cursor_fingerprint_read_bytes=_int_attr(latest_attempt, "cursor_fingerprint_read_bytes"),
        read_amplification=round(
            _ratio(_int_attr(latest_attempt, "source_payload_read_bytes"), _int_attr(latest_attempt, "input_bytes")),
            4,
        ),
        files_per_second=round(_ratio(_int_attr(latest_attempt, "succeeded_file_count"), total_time_s), 3),
        source_mb_per_second=round(
            _ratio(_int_attr(latest_attempt, "source_payload_read_bytes") / 1_000_000, total_time_s),
            3,
        ),
        parse_time_s=_float_attr(latest_attempt, "parse_time_s"),
        convergence_time_s=_float_attr(latest_attempt, "convergence_time_s"),
        total_time_s=total_time_s,
        latest_event_age_s=_optional_float_attr(latest_attempt, "updated_age_s"),
    )


def format_catchup_status_lines(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    lines = [
        "Catch-up: "
        f"{payload.get('mode', 'idle')} "
        f"{payload.get('succeeded_file_count', 0)}/{payload.get('needed_file_count', 0)} files, "
        f"read amp {payload.get('read_amplification', 0)}x"
    ]
    if phase := payload.get("current_phase"):
        lines.append(
            "  "
            f"phase={phase} source={payload.get('current_source') or '-'} "
            f"source_read={payload.get('source_payload_read_bytes', 0)} bytes "
            f"cursor_read={payload.get('cursor_fingerprint_read_bytes', 0)} bytes"
        )
    return lines


def _recent_stage_events(dbf: Path) -> list[CatchupStageEvent]:
    if not dbf.exists():
        return []
    try:
        conn = open_readonly_connection(dbf)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'live_ingest_stage_event'"
            ).fetchone()
            if has_table is None:
                return []
            rows = conn.execute(
                """
                SELECT attempt_id, sequence, observed_at, phase, status,
                       queued_file_count, needed_file_count, skipped_file_count,
                       succeeded_file_count, failed_file_count, input_bytes,
                       source_payload_read_bytes, cursor_fingerprint_read_bytes,
                       archive_write_bytes_delta, parse_time_s, convergence_time_s,
                       total_time_s, current_source, current_path, error
                FROM live_ingest_stage_event
                ORDER BY observed_at DESC, event_id DESC
                LIMIT 10
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [_catchup_stage_event_from_row(row) for row in rows]


def _catchup_stage_event_from_row(row: sqlite3.Row | tuple[object, ...]) -> CatchupStageEvent:
    return CatchupStageEvent(
        attempt_id=_required_str(row[0]),
        sequence=_row_int(row[1]),
        observed_at=_required_str(row[2]),
        phase=_required_str(row[3]),
        status=_required_str(row[4]),
        queued_file_count=_row_int(row[5]),
        needed_file_count=_row_int(row[6]),
        skipped_file_count=_row_int(row[7]),
        succeeded_file_count=_row_int(row[8]),
        failed_file_count=_row_int(row[9]),
        input_bytes=_row_int(row[10]),
        source_payload_read_bytes=_row_int(row[11]),
        cursor_fingerprint_read_bytes=_row_int(row[12]),
        archive_write_bytes_delta=_row_int(row[13]),
        parse_time_s=_row_float(row[14]) or 0.0,
        convergence_time_s=_row_float(row[15]) or 0.0,
        total_time_s=_row_float(row[16]) or 0.0,
        current_source=_optional_str(row[17]),
        current_path=_optional_str(row[18]),
        error=_optional_str(row[19]),
    )


def _catchup_mode(latest: CatchupStageEvent | None, latest_attempt: object | None, convergence: object) -> str:
    attempt_status = _str_attr(latest_attempt, "status") if latest_attempt is not None else None
    attempt_phase = _str_attr(latest_attempt, "phase") if latest_attempt is not None else None
    if attempt_status == "running":
        return (
            "converging" if attempt_phase in {"convergence", "fts", "insights", "full_worker_wait"} else "catching_up"
        )
    if latest is not None and latest.status == "running":
        return "converging" if latest.phase in {"convergence", "fts", "insights", "full_worker_wait"} else "catching_up"
    if _int_attr(convergence, "retry_due_count") > 0:
        return "debt_retry"
    if _int_attr(convergence, "failed_count") > 0:
        return "degraded"
    return "idle"


def _iso_age_s(value: str, *, now: datetime) -> float | None:
    try:
        observed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    return max(0.0, round((now - observed.astimezone(UTC)).total_seconds(), 3))


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _required_str(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _row_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _row_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_attr(item: object, name: str) -> int:
    return _row_int(getattr(item, name, None))


def _float_attr(item: object, name: str) -> float:
    return _row_float(getattr(item, name, None)) or 0.0


def _optional_float_attr(item: object, name: str) -> float | None:
    return _row_float(getattr(item, name, None))


def _str_attr(item: object, name: str) -> str | None:
    value = getattr(item, name, None)
    return value if isinstance(value, str) else None


__all__ = ["CatchupStageEvent", "CatchupStatus", "catchup_status_info", "format_catchup_status_lines"]
