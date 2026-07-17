"""Status projection for daemon catch-up progress and throughput."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.core.payload_coercion import optional_str as _optional_str
from polylogue.core.payload_coercion import required_str as _required_str
from polylogue.core.payload_coercion import row_float as _row_float
from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.logging import get_logger
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)


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
    ops_events = _archive_recent_stage_events(dbf.with_name("ops.db"))
    if ops_events:
        return ops_events
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
    except sqlite3.Error as exc:
        logger.warning("catchup live stage-event query failed for %s: %s", dbf, exc, exc_info=True)
        return []
    return [_catchup_stage_event_from_row(row) for row in rows]


def _archive_recent_stage_events(ops_db: Path) -> list[CatchupStageEvent]:
    if not ops_db.exists():
        return []
    try:
        conn = open_readonly_connection(ops_db)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daemon_stage_events'"
            ).fetchone()
            if has_table is None:
                return []
            rows = conn.execute(
                """
                SELECT rowid, attempt_id, observed_at_ms, stage, status, payload_json
                FROM daemon_stage_events
                ORDER BY observed_at_ms DESC, rowid DESC
                LIMIT 10
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("catchup archive stage-event query failed for %s: %s", ops_db, exc, exc_info=True)
        return []
    return [_archive_catchup_stage_event_from_row(row) for row in rows]


def _archive_catchup_stage_event_from_row(row: sqlite3.Row | tuple[object, ...]) -> CatchupStageEvent:
    payload = _payload(row[5])
    return CatchupStageEvent(
        attempt_id=_required_str(row[1]),
        sequence=_row_int(row[0]),
        observed_at=_epoch_ms_to_iso(_row_int(row[2])),
        phase=_payload_str(payload, "phase", default=_required_str(row[3])),
        status=_payload_str(payload, "status", default=_required_str(row[4])),
        queued_file_count=_payload_int(payload, "queued_file_count"),
        needed_file_count=_payload_int(payload, "needed_file_count"),
        skipped_file_count=_payload_int(payload, "skipped_file_count"),
        succeeded_file_count=_payload_int(payload, "succeeded_file_count"),
        failed_file_count=_payload_int(payload, "failed_file_count"),
        input_bytes=_payload_int(payload, "input_bytes"),
        source_payload_read_bytes=_payload_int(payload, "source_payload_read_bytes"),
        cursor_fingerprint_read_bytes=_payload_int(payload, "cursor_fingerprint_read_bytes"),
        archive_write_bytes_delta=_payload_int(payload, "archive_write_bytes_delta"),
        parse_time_s=_payload_float(payload, "parse_time_s"),
        convergence_time_s=_payload_float(payload, "convergence_time_s"),
        total_time_s=_payload_float(payload, "total_time_s"),
        current_source=_payload_optional_str(payload, "current_source"),
        current_path=_payload_optional_str(payload, "current_path"),
        error=_payload_optional_str(payload, "error"),
    )


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


def _payload(raw: object) -> dict[str, object]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _payload_int(payload: dict[str, object], key: str, default: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _payload_float(payload: dict[str, object], key: str, default: float = 0.0) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _payload_str(payload: dict[str, object], key: str, *, default: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _payload_optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _epoch_ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(max(value, 0) / 1000.0, tz=UTC).isoformat()


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
