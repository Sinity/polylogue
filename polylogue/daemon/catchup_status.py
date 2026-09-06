"""Status projection for daemon catch-up progress and throughput."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from polylogue.core.payload_coercion import optional_str as _optional_str
from polylogue.core.payload_coercion import required_str as _required_str
from polylogue.core.payload_coercion import row_float as _row_float
from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.core.payload_coercion import row_iso_from_epoch_ms as _iso_from_epoch_ms
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
    ingested_bytes: int = 0
    failed_bytes: int = 0
    refused_bytes: int = 0
    refused_bytes_by_reason: dict[str, int] = Field(default_factory=dict)
    source_payload_read_bytes: int = 0
    cursor_fingerprint_read_bytes: int = 0
    archive_write_bytes_delta: int = 0
    parse_time_s: float = 0.0
    convergence_time_s: float = 0.0
    total_time_s: float = 0.0
    current_source: str | None = None
    current_path: str | None = None
    error: str | None = None


class HaltedSourceStatus(BaseModel):
    """One source whose ingest is stopped until the daemon restarts."""

    source_name: str
    code: str
    message: str
    derived_only: bool = False
    observed_at: str


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
    #: ``input_bytes`` is what the batch was OFFERED; the split below is what
    #: became of it. Throughput is ``ingested_mb_per_second`` -- a rate over
    #: offered bytes counts every declined file as work done.
    input_bytes: int = 0
    ingested_bytes: int = 0
    failed_bytes: int = 0
    refused_bytes: int = 0
    refused_bytes_by_reason: dict[str, int] = Field(default_factory=dict)
    source_payload_read_bytes: int = 0
    cursor_fingerprint_read_bytes: int = 0
    archive_write_bytes_delta: int = 0
    read_amplification: float = 0.0
    files_per_second: float = 0.0
    source_mb_per_second: float = 0.0
    ingested_mb_per_second: float = 0.0
    parse_time_s: float = 0.0
    convergence_time_s: float = 0.0
    total_time_s: float = 0.0
    latest_event_age_s: float | None = None
    #: Sources the running daemon refuses to ingest until restart. Empty is
    #: the healthy answer; a non-empty list means catch-up is skipping that
    #: source's backlog on purpose, not idling.
    halted_sources: list[HaltedSourceStatus] = Field(default_factory=list)
    recent_events: list[CatchupStageEvent] = Field(default_factory=list)


def catchup_status_info(
    dbf: Path,
    *,
    latest_attempt: object | None,
    convergence: object,
    ops_db: Path | None = None,
) -> CatchupStatus:
    """Return bounded catch-up/convergence progress and throughput from durable events."""
    events = _recent_stage_events(dbf, ops_db=ops_db)
    latest = events[0] if events else None
    now = datetime.now(UTC)
    mode = _catchup_mode(latest, latest_attempt, convergence)
    halted = _halted_sources(ops_db if ops_db is not None else dbf.with_name("ops.db"))
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
            ingested_bytes=latest.ingested_bytes,
            failed_bytes=latest.failed_bytes,
            refused_bytes=latest.refused_bytes,
            refused_bytes_by_reason=dict(latest.refused_bytes_by_reason),
            source_payload_read_bytes=latest.source_payload_read_bytes,
            cursor_fingerprint_read_bytes=latest.cursor_fingerprint_read_bytes,
            archive_write_bytes_delta=latest.archive_write_bytes_delta,
            read_amplification=round(_ratio(latest.source_payload_read_bytes, latest.input_bytes), 4),
            files_per_second=round(_ratio(latest.succeeded_file_count, total_time_s), 3),
            source_mb_per_second=round(_ratio(latest.source_payload_read_bytes / 1_000_000, total_time_s), 3),
            ingested_mb_per_second=round(_ratio(latest.ingested_bytes / 1_000_000, total_time_s), 3),
            parse_time_s=latest.parse_time_s,
            convergence_time_s=latest.convergence_time_s,
            total_time_s=total_time_s,
            latest_event_age_s=_iso_age_s(latest.observed_at, now=now),
            halted_sources=halted,
            recent_events=events,
        )
    if latest_attempt is None:
        return CatchupStatus(mode=mode, halted_sources=halted)
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
        halted_sources=halted,
    )


def format_catchup_status_lines(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    lines = [
        "Catch-up: "
        f"{payload.get('mode', 'idle')} "
        f"{payload.get('succeeded_file_count', 0)}/{payload.get('needed_file_count', 0)} files, "
        f"read amp {payload.get('read_amplification', 0)}x, "
        f"{payload.get('ingested_mb_per_second', 0)} MB/s ingested"
    ]
    refused_by_reason = payload.get("refused_bytes_by_reason")
    if isinstance(refused_by_reason, dict) and refused_by_reason:
        lines.append(
            f"  refused {payload.get('refused_bytes', 0)} bytes of "
            f"{payload.get('input_bytes', 0)} offered: "
            + ", ".join(f"{reason}={count}" for reason, count in sorted(refused_by_reason.items()))
        )
    halted = payload.get("halted_sources")
    if isinstance(halted, list) and halted:
        for entry in halted:
            if isinstance(entry, dict):
                lines.append(
                    f"  HALTED {entry.get('source_name', '?')} ({entry.get('code', 'unknown')}): "
                    f"{entry.get('message', '')}"
                )
    if phase := payload.get("current_phase"):
        lines.append(
            "  "
            f"phase={phase} source={payload.get('current_source') or '-'} "
            f"source_read={payload.get('source_payload_read_bytes', 0)} bytes "
            f"cursor_read={payload.get('cursor_fingerprint_read_bytes', 0)} bytes"
        )
    return lines


HALT_EVENT_KIND = "source_ingest_halted"


def _halted_sources(ops_db: Path) -> list[HaltedSourceStatus]:
    """Sources halted by the daemon run that is currently recorded.

    Bounded to the latest ``daemon_lifecycle`` start: a halt is process-local
    and does not survive a restart, so replaying an older run's halt would
    report a source as stopped when it is ingesting.
    """
    if not ops_db.exists():
        return []
    try:
        conn = open_readonly_connection(ops_db, validate_schema=False)
        try:
            if not _has_table(conn, "daemon_events"):
                return []
            floor_ms = 0
            if _has_table(conn, "daemon_lifecycle"):
                row = conn.execute("SELECT MAX(started_at_ms) FROM daemon_lifecycle").fetchone()
                floor_ms = _row_int(row[0]) if row is not None and row[0] is not None else 0
            rows = conn.execute(
                """
                SELECT ts_ms, payload_json
                FROM daemon_events
                WHERE kind = ? AND ts_ms >= ?
                ORDER BY id DESC
                LIMIT 50
                """,
                (HALT_EVENT_KIND, floor_ms),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("halted-source query failed for %s: %s", ops_db, exc, exc_info=True)
        return []
    latest_by_source: dict[str, HaltedSourceStatus] = {}
    for row in rows:
        payload = _payload(row[1])
        source_name = _payload_optional_str(payload, "source_name")
        if source_name is None or source_name in latest_by_source:
            continue
        latest_by_source[source_name] = HaltedSourceStatus(
            source_name=source_name,
            code=_payload_str(payload, "code", default="unknown"),
            message=_payload_str(payload, "message", default=""),
            derived_only=payload.get("derived_only") is True,
            observed_at=cast(str, _iso_from_epoch_ms(max(_row_int(row[0]), 0))),
        )
    return [latest_by_source[name] for name in sorted(latest_by_source)]


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def _recent_stage_events(dbf: Path, *, ops_db: Path | None = None) -> list[CatchupStageEvent]:
    resolved_ops_db = ops_db if ops_db is not None else dbf.with_name("ops.db")
    ops_events = _archive_recent_stage_events(resolved_ops_db)
    if ops_events:
        return ops_events
    if not dbf.exists():
        return []
    try:
        # Status reads tolerate a skewed or unstamped tier; the table probe below
        # already establishes what can be read.
        conn = open_readonly_connection(dbf, validate_schema=False)
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
        # Negative epoch_ms values clamp to the epoch floor, matching the
        # previous _epoch_ms_to_iso behavior; the shared helper's int branch
        # then always returns a string, never None.
        observed_at=cast(str, _iso_from_epoch_ms(max(_row_int(row[2]), 0))),
        phase=_payload_str(payload, "phase", default=_required_str(row[3])),
        status=_payload_str(payload, "status", default=_required_str(row[4])),
        queued_file_count=_payload_int(payload, "queued_file_count"),
        needed_file_count=_payload_int(payload, "needed_file_count"),
        skipped_file_count=_payload_int(payload, "skipped_file_count"),
        succeeded_file_count=_payload_int(payload, "succeeded_file_count"),
        failed_file_count=_payload_int(payload, "failed_file_count"),
        input_bytes=_payload_int(payload, "input_bytes"),
        ingested_bytes=_payload_int(payload, "ingested_bytes"),
        failed_bytes=_payload_int(payload, "failed_bytes"),
        refused_bytes=_payload_int(payload, "refused_bytes"),
        refused_bytes_by_reason=_payload_int_map(payload, "refused_bytes_by_reason"),
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
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    return _row_int(value)


def _payload_int_map(payload: dict[str, object], key: str) -> dict[str, int]:
    value = payload.get(key)
    if not isinstance(value, dict):
        return {}
    return {str(name): _row_int(count) for name, count in value.items() if isinstance(count, int | float | str)}


def _payload_float(payload: dict[str, object], key: str, default: float = 0.0) -> float:
    coerced = _row_float(payload.get(key))
    return default if coerced is None else coerced


def _payload_str(payload: dict[str, object], key: str, *, default: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _payload_optional_str(payload: dict[str, object], key: str) -> str | None:
    return _optional_str(payload.get(key))


def _catchup_mode(latest: CatchupStageEvent | None, latest_attempt: object | None, convergence: object) -> str:
    attempt_status = _str_attr(latest_attempt, "status") if latest_attempt is not None else None
    attempt_phase = _str_attr(latest_attempt, "phase") if latest_attempt is not None else None
    if attempt_status == "running":
        return "converging" if attempt_phase in {"convergence", "fts", "derived", "full_worker_wait"} else "catching_up"
    if latest is not None and latest.status == "running":
        return "converging" if latest.phase in {"convergence", "fts", "derived", "full_worker_wait"} else "catching_up"
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
