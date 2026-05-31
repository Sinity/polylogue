"""Per-file processing cursor for live ingestion.

Cursor state enables content-aware skip decisions: same-size rewrites,
truncation, and parser version changes are detected via content fingerprint
comparison rather than relying on file size alone.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from polylogue.sources.live._lag_sample_ddl import _LAG_SAMPLE_DDL, _LAG_SAMPLE_INDEX_DDL
from polylogue.sources.live.convergence_debt_store import (
    clear_convergence_debt_except_sync,
    record_convergence_debt_sync,
)
from polylogue.sources.live.sqlite_locking import best_effort_cursor_write
from polylogue.storage.sqlite.connection_profile import open_connection

_DDL = """
CREATE TABLE IF NOT EXISTS live_cursor (
    source_path TEXT PRIMARY KEY,
    byte_size INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL DEFAULT 0,
    last_complete_newline INTEGER NOT NULL DEFAULT 0,
    record_count INTEGER NOT NULL DEFAULT 0,
    last_record_ts TEXT,
    parser_fingerprint TEXT,
    content_fingerprint TEXT,
    tail_hash TEXT,
    source_name TEXT,
    -- Stat metadata for fast-path skip (Phase 2 K: #620)
    st_dev INTEGER,
    st_ino INTEGER,
    mtime_ns INTEGER,
    -- Quarantine fields (Phase 2 K: #620)
    source_generation INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    excluded INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
)
"""

_ATTEMPT_DDL = """
CREATE TABLE IF NOT EXISTS live_ingest_attempt (
    attempt_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    queued_file_count INTEGER NOT NULL DEFAULT 0,
    needed_file_count INTEGER NOT NULL DEFAULT 0,
    succeeded_file_count INTEGER NOT NULL DEFAULT 0,
    failed_file_count INTEGER NOT NULL DEFAULT 0,
    input_bytes INTEGER NOT NULL DEFAULT 0,
    source_payload_read_bytes INTEGER NOT NULL DEFAULT 0,
    cursor_fingerprint_read_bytes INTEGER NOT NULL DEFAULT 0,
    parse_time_s REAL NOT NULL DEFAULT 0,
    convergence_time_s REAL NOT NULL DEFAULT 0,
    current_source TEXT,
    current_path TEXT,
    error TEXT,
    rss_current_mb REAL,
    rss_peak_self_mb REAL,
    rss_peak_children_mb REAL,
    cgroup_path TEXT,
    cgroup_memory_current_mb REAL,
    cgroup_memory_peak_mb REAL,
    cgroup_memory_swap_current_mb REAL,
    cgroup_memory_anon_mb REAL,
    cgroup_memory_file_mb REAL,
    cgroup_memory_inactive_file_mb REAL,
    worker_in_flight_count INTEGER,
    worker_completed_count INTEGER,
    worker_total_count INTEGER,
    stale_cursor_write_count INTEGER NOT NULL DEFAULT 0,
    source_paths_json TEXT NOT NULL DEFAULT '[]'
)
"""

_STAGE_EVENT_DDL = """
CREATE TABLE IF NOT EXISTS live_ingest_stage_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    queued_file_count INTEGER,
    needed_file_count INTEGER,
    skipped_file_count INTEGER,
    succeeded_file_count INTEGER,
    failed_file_count INTEGER,
    input_bytes INTEGER,
    source_payload_read_bytes INTEGER,
    cursor_fingerprint_read_bytes INTEGER,
    archive_write_bytes_delta INTEGER,
    parse_time_s REAL,
    convergence_time_s REAL,
    total_time_s REAL,
    current_source TEXT,
    current_path TEXT,
    error TEXT,
    rss_current_mb REAL,
    rss_peak_self_mb REAL,
    rss_peak_children_mb REAL,
    cgroup_path TEXT,
    cgroup_memory_current_mb REAL,
    cgroup_memory_peak_mb REAL,
    cgroup_memory_swap_current_mb REAL,
    cgroup_memory_anon_mb REAL,
    cgroup_memory_file_mb REAL,
    cgroup_memory_inactive_file_mb REAL,
    worker_in_flight_count INTEGER,
    worker_completed_count INTEGER,
    worker_total_count INTEGER,
    stage_timings_json TEXT,
    UNIQUE(attempt_id, sequence)
)
"""

_CONVERGENCE_DEBT_DDL = """
CREATE TABLE IF NOT EXISTS live_convergence_debt (
    stage TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'failed',
    failure_count INTEGER NOT NULL DEFAULT 0,
    first_failed_at TEXT NOT NULL,
    last_failed_at TEXT NOT NULL,
    next_retry_at TEXT,
    materializer_version TEXT,
    last_error TEXT,
    PRIMARY KEY (stage, subject_type, subject_id)
)
"""

# Per-source-family cursor-lag sample history (#1349). Daemon-runtime state,
# not part of SCHEMA_VERSION — same lifecycle as live_cursor / live_convergence_debt.
# DDL is shared with cursor_lag_baseline via polylogue.sources.live._lag_sample_ddl.


@dataclass(frozen=True, slots=True)
class CursorRecord:
    """Stored live cursor state for one source file."""

    source_path: str
    byte_size: int
    byte_offset: int
    last_complete_newline: int
    record_count: int
    updated_at: str
    last_record_ts: str | None = None
    parser_fingerprint: str | None = None
    content_fingerprint: str | None = None
    tail_hash: str | None = None
    source_name: str | None = None
    st_dev: int | None = None
    st_ino: int | None = None
    mtime_ns: int | None = None
    source_generation: int = 0
    failure_count: int = 0
    next_retry_at: str | None = None
    excluded: bool | int = False


@dataclass(frozen=True, slots=True)
class LiveIngestAttempt:
    """Durable live-ingest attempt snapshot for in-flight diagnostics."""

    attempt_id: str
    started_at: str
    updated_at: str
    status: str
    phase: str
    queued_file_count: int
    needed_file_count: int
    succeeded_file_count: int
    failed_file_count: int
    input_bytes: int
    source_payload_read_bytes: int
    cursor_fingerprint_read_bytes: int
    parse_time_s: float
    convergence_time_s: float
    completed_at: str | None = None
    current_source: str | None = None
    current_path: str | None = None
    error: str | None = None
    rss_current_mb: float | None = None
    rss_peak_self_mb: float | None = None
    rss_peak_children_mb: float | None = None
    cgroup_path: str | None = None
    cgroup_memory_current_mb: float | None = None
    cgroup_memory_peak_mb: float | None = None
    cgroup_memory_swap_current_mb: float | None = None
    cgroup_memory_anon_mb: float | None = None
    cgroup_memory_file_mb: float | None = None
    cgroup_memory_inactive_file_mb: float | None = None
    worker_in_flight_count: int | None = None
    worker_completed_count: int | None = None
    worker_total_count: int | None = None
    stale_cursor_write_count: int = 0
    source_paths_json: str = "[]"


@dataclass(frozen=True, slots=True)
class LiveConvergenceDebt:
    """Durable post-ingest convergence failure for one derived subject."""

    stage: str
    subject_type: str
    subject_id: str
    status: str
    failure_count: int
    first_failed_at: str
    last_failed_at: str
    next_retry_at: str | None = None
    materializer_version: str | None = None
    last_error: str | None = None


def _required_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str | bytes | bytearray):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return int(cast(Any, value))


def _optional_int(value: object) -> int | None:
    return None if value is None else _required_int(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _cursor_record_from_row(row: sqlite3.Row | tuple[object, ...]) -> CursorRecord:
    return CursorRecord(
        source_path=str(row[0]),
        byte_size=_required_int(row[1]),
        byte_offset=_required_int(row[2]),
        last_complete_newline=_required_int(row[3]),
        record_count=_required_int(row[4]),
        updated_at=str(row[5]),
        last_record_ts=_optional_str(row[6]),
        parser_fingerprint=_optional_str(row[7]),
        content_fingerprint=_optional_str(row[8]),
        tail_hash=_optional_str(row[9]),
        source_name=_optional_str(row[10]),
        st_dev=_optional_int(row[11]),
        st_ino=_optional_int(row[12]),
        mtime_ns=_optional_int(row[13]),
        source_generation=_required_int(row[14]) if row[14] is not None else 0,
        failure_count=_required_int(row[15]) if row[15] is not None else 0,
        next_retry_at=_optional_str(row[16]),
        excluded=bool(row[17]) if row[17] is not None else False,
    )


class CursorStore:
    """SQLite-backed live cursor store keyed by source path."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_DDL)
            conn.execute(_ATTEMPT_DDL)
            conn.execute(_STAGE_EVENT_DDL)
            conn.execute(_CONVERGENCE_DEBT_DDL)
            conn.execute(_LAG_SAMPLE_DDL)
            conn.execute(_LAG_SAMPLE_INDEX_DDL)
            self._ensure_columns(conn)
            self._mark_interrupted_attempts(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # ``open_connection`` hands back a fresh connection the caller owns and
        # must close. The inner ``with conn`` preserves the prior commit-on-
        # success / rollback-on-exception transaction semantics; the surrounding
        # ``finally`` adds the close every call site previously omitted (``with
        # sqlite3.Connection`` only commits, it never closes — a per-operation
        # connection leak in the live cursor store).
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = open_connection(self._db_path, timeout=10.0)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(live_cursor)")}
        columns = {
            "byte_offset": "INTEGER NOT NULL DEFAULT 0",
            "last_complete_newline": "INTEGER NOT NULL DEFAULT 0",
            "last_record_ts": "TEXT",
            "parser_fingerprint": "TEXT",
            "content_fingerprint": "TEXT",
            "tail_hash": "TEXT",
            "source_name": "TEXT",
            "st_dev": "INTEGER",
            "st_ino": "INTEGER",
            "mtime_ns": "INTEGER",
            "source_generation": "INTEGER NOT NULL DEFAULT 0",
            "failure_count": "INTEGER NOT NULL DEFAULT 0",
            "next_retry_at": "TEXT",
            "excluded": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE live_cursor ADD COLUMN {name} {definition}")
        conn.execute(_ATTEMPT_DDL)
        existing_attempt = {row[1] for row in conn.execute("PRAGMA table_info(live_ingest_attempt)")}
        attempt_columns = {
            "cgroup_path": "TEXT",
            "cgroup_memory_current_mb": "REAL",
            "cgroup_memory_peak_mb": "REAL",
            "cgroup_memory_swap_current_mb": "REAL",
            "cgroup_memory_anon_mb": "REAL",
            "cgroup_memory_file_mb": "REAL",
            "cgroup_memory_inactive_file_mb": "REAL",
            "worker_in_flight_count": "INTEGER",
            "worker_completed_count": "INTEGER",
            "worker_total_count": "INTEGER",
            "stale_cursor_write_count": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in attempt_columns.items():
            if name not in existing_attempt:
                conn.execute(f"ALTER TABLE live_ingest_attempt ADD COLUMN {name} {definition}")
        conn.execute(_STAGE_EVENT_DDL)
        existing_stage_event = {row[1] for row in conn.execute("PRAGMA table_info(live_ingest_stage_event)")}
        stage_event_columns = {
            "cgroup_memory_anon_mb": "REAL",
            "cgroup_memory_file_mb": "REAL",
            "cgroup_memory_inactive_file_mb": "REAL",
        }
        for name, definition in stage_event_columns.items():
            if name not in existing_stage_event:
                conn.execute(f"ALTER TABLE live_ingest_stage_event ADD COLUMN {name} {definition}")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_ingest_stage_event_attempt
            ON live_ingest_stage_event(attempt_id, sequence)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_ingest_stage_event_observed
            ON live_ingest_stage_event(observed_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_cursor_failed_retry
            ON live_cursor(failure_count, next_retry_at)
            WHERE failure_count > 0
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_cursor_excluded
            ON live_cursor(excluded)
            WHERE excluded = 1
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_ingest_attempt_status_updated
            ON live_ingest_attempt(status, updated_at DESC, started_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_ingest_attempt_updated
            ON live_ingest_attempt(updated_at DESC, started_at DESC)
            """
        )
        conn.execute(_CONVERGENCE_DEBT_DDL)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_convergence_debt_status_retry
            ON live_convergence_debt(status, next_retry_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_convergence_debt_subject
            ON live_convergence_debt(subject_type, subject_id)
            """
        )
        conn.commit()

    def _mark_interrupted_attempts(self, conn: sqlite3.Connection) -> None:
        """Close attempts left running by a killed daemon process."""
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """
            UPDATE live_ingest_attempt
            SET updated_at = ?,
                completed_at = ?,
                status = 'abandoned',
                phase = 'interrupted',
                error = COALESCE(error, 'daemon stopped before completing this ingest attempt')
            WHERE status = 'running'
            """,
            (now, now),
        )
        conn.commit()

    def begin_ingest_attempt(
        self,
        *,
        paths: list[Path],
        input_bytes: int,
        queued_file_count: int,
    ) -> str:
        """Record a durable in-flight live-ingest attempt."""
        import json

        now = datetime.now(UTC).isoformat()
        attempt_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_ingest_attempt (
                    attempt_id,
                    started_at,
                    updated_at,
                    status,
                    phase,
                    queued_file_count,
                    needed_file_count,
                    input_bytes,
                    source_paths_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    now,
                    now,
                    "running",
                    "planning",
                    queued_file_count,
                    len(paths),
                    input_bytes,
                    json.dumps([str(path) for path in paths], separators=(",", ":")),
                ),
            )
            conn.commit()
        return attempt_id

    def update_ingest_attempt(
        self,
        attempt_id: str,
        *,
        phase: str,
        status: str = "running",
        succeeded_file_count: int | None = None,
        failed_file_count: int | None = None,
        source_payload_read_bytes: int | None = None,
        cursor_fingerprint_read_bytes: int | None = None,
        parse_time_s: float | None = None,
        convergence_time_s: float | None = None,
        current_source: str | None = None,
        current_path: Path | str | None = None,
        error: str | None = None,
        rss_current_mb: float | None = None,
        rss_peak_self_mb: float | None = None,
        rss_peak_children_mb: float | None = None,
        cgroup_path: str | None = None,
        cgroup_memory_current_mb: float | None = None,
        cgroup_memory_peak_mb: float | None = None,
        cgroup_memory_swap_current_mb: float | None = None,
        cgroup_memory_anon_mb: float | None = None,
        cgroup_memory_file_mb: float | None = None,
        cgroup_memory_inactive_file_mb: float | None = None,
        worker_in_flight_count: int | None = None,
        worker_completed_count: int | None = None,
        worker_total_count: int | None = None,
        stale_cursor_write_count: int | None = None,
    ) -> bool:
        """Update an in-flight attempt without waiting for batch completion."""
        now = datetime.now(UTC).isoformat()
        assignments: list[str] = ["updated_at = ?", "status = ?", "phase = ?"]
        values: list[object] = [now, status, phase]
        optional_fields = {
            "succeeded_file_count": succeeded_file_count,
            "failed_file_count": failed_file_count,
            "source_payload_read_bytes": source_payload_read_bytes,
            "cursor_fingerprint_read_bytes": cursor_fingerprint_read_bytes,
            "parse_time_s": parse_time_s,
            "convergence_time_s": convergence_time_s,
            "current_source": current_source,
            "current_path": str(current_path) if current_path is not None else None,
            "error": error,
            "rss_current_mb": rss_current_mb,
            "rss_peak_self_mb": rss_peak_self_mb,
            "rss_peak_children_mb": rss_peak_children_mb,
            "cgroup_path": cgroup_path,
            "cgroup_memory_current_mb": cgroup_memory_current_mb,
            "cgroup_memory_peak_mb": cgroup_memory_peak_mb,
            "cgroup_memory_swap_current_mb": cgroup_memory_swap_current_mb,
            "cgroup_memory_anon_mb": cgroup_memory_anon_mb,
            "cgroup_memory_file_mb": cgroup_memory_file_mb,
            "cgroup_memory_inactive_file_mb": cgroup_memory_inactive_file_mb,
            "worker_in_flight_count": worker_in_flight_count,
            "worker_completed_count": worker_completed_count,
            "worker_total_count": worker_total_count,
            "stale_cursor_write_count": stale_cursor_write_count,
        }
        for field, value in optional_fields.items():
            if value is None:
                continue
            assignments.append(f"{field} = ?")
            values.append(value)
        values.append(attempt_id)

        def write() -> None:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE live_ingest_attempt SET {', '.join(assignments)} WHERE attempt_id = ?",
                    tuple(values),
                )
                conn.commit()

        return best_effort_cursor_write("live ingest attempt progress", write)

    def record_ingest_stage_event(
        self,
        attempt_id: str,
        *,
        phase: str,
        status: str = "running",
        queued_file_count: int | None = None,
        needed_file_count: int | None = None,
        skipped_file_count: int | None = None,
        succeeded_file_count: int | None = None,
        failed_file_count: int | None = None,
        input_bytes: int | None = None,
        source_payload_read_bytes: int | None = None,
        cursor_fingerprint_read_bytes: int | None = None,
        archive_write_bytes_delta: int | None = None,
        parse_time_s: float | None = None,
        convergence_time_s: float | None = None,
        total_time_s: float | None = None,
        current_source: str | None = None,
        current_path: Path | str | None = None,
        error: str | None = None,
        rss_current_mb: float | None = None,
        rss_peak_self_mb: float | None = None,
        rss_peak_children_mb: float | None = None,
        cgroup_path: str | None = None,
        cgroup_memory_current_mb: float | None = None,
        cgroup_memory_peak_mb: float | None = None,
        cgroup_memory_swap_current_mb: float | None = None,
        cgroup_memory_anon_mb: float | None = None,
        cgroup_memory_file_mb: float | None = None,
        cgroup_memory_inactive_file_mb: float | None = None,
        worker_in_flight_count: int | None = None,
        worker_completed_count: int | None = None,
        worker_total_count: int | None = None,
        stage_timings_json: str | None = None,
    ) -> bool:
        """Append one durable progress event for a live-ingest attempt."""
        now = datetime.now(UTC).isoformat()

        def write() -> None:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM live_ingest_stage_event WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                sequence = int(row[0] or 1)
                conn.execute(
                    """
                    INSERT INTO live_ingest_stage_event (
                    attempt_id,
                    sequence,
                    observed_at,
                    phase,
                    status,
                    queued_file_count,
                    needed_file_count,
                    skipped_file_count,
                    succeeded_file_count,
                    failed_file_count,
                    input_bytes,
                    source_payload_read_bytes,
                    cursor_fingerprint_read_bytes,
                    archive_write_bytes_delta,
                    parse_time_s,
                    convergence_time_s,
                    total_time_s,
                    current_source,
                    current_path,
                    error,
                    rss_current_mb,
                    rss_peak_self_mb,
                    rss_peak_children_mb,
                    cgroup_path,
                    cgroup_memory_current_mb,
                    cgroup_memory_peak_mb,
                    cgroup_memory_swap_current_mb,
                    cgroup_memory_anon_mb,
                    cgroup_memory_file_mb,
                    cgroup_memory_inactive_file_mb,
                    worker_in_flight_count,
                    worker_completed_count,
                    worker_total_count,
                    stage_timings_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        sequence,
                        now,
                        phase,
                        status,
                        queued_file_count,
                        needed_file_count,
                        skipped_file_count,
                        succeeded_file_count,
                        failed_file_count,
                        input_bytes,
                        source_payload_read_bytes,
                        cursor_fingerprint_read_bytes,
                        archive_write_bytes_delta,
                        parse_time_s,
                        convergence_time_s,
                        total_time_s,
                        current_source,
                        str(current_path) if current_path is not None else None,
                        error,
                        rss_current_mb,
                        rss_peak_self_mb,
                        rss_peak_children_mb,
                        cgroup_path,
                        cgroup_memory_current_mb,
                        cgroup_memory_peak_mb,
                        cgroup_memory_swap_current_mb,
                        cgroup_memory_anon_mb,
                        cgroup_memory_file_mb,
                        cgroup_memory_inactive_file_mb,
                        worker_in_flight_count,
                        worker_completed_count,
                        worker_total_count,
                        stage_timings_json,
                    ),
                )
                conn.commit()

        return best_effort_cursor_write("live ingest stage event", write)

    def finish_ingest_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        phase: str,
        error: str | None = None,
    ) -> bool:
        """Mark an ingest attempt complete or failed."""
        now = datetime.now(UTC).isoformat()

        def write() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE live_ingest_attempt
                    SET updated_at = ?,
                        completed_at = ?,
                        status = ?,
                        phase = ?,
                        error = ?
                    WHERE attempt_id = ?
                    """,
                    (now, now, status, phase, error, attempt_id),
                )
                conn.commit()

        return best_effort_cursor_write("live ingest attempt finish", write)

    def recent_ingest_attempts(self, *, limit: int = 5) -> list[LiveIngestAttempt]:
        """Return recent live-ingest attempts for status/debug surfaces."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    attempt_id,
                    started_at,
                    updated_at,
                    completed_at,
                    status,
                    phase,
                    queued_file_count,
                    needed_file_count,
                    succeeded_file_count,
                    failed_file_count,
                    input_bytes,
                    source_payload_read_bytes,
                    cursor_fingerprint_read_bytes,
                    parse_time_s,
                    convergence_time_s,
                    current_source,
                    current_path,
                    error,
                    rss_current_mb,
                    rss_peak_self_mb,
                    rss_peak_children_mb,
                    cgroup_path,
                    cgroup_memory_current_mb,
                    cgroup_memory_peak_mb,
                    cgroup_memory_swap_current_mb,
                    worker_in_flight_count,
                    worker_completed_count,
                    worker_total_count,
                    stale_cursor_write_count,
                    source_paths_json
                FROM live_ingest_attempt
                ORDER BY updated_at DESC, started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            LiveIngestAttempt(
                attempt_id=str(row[0]),
                started_at=str(row[1]),
                updated_at=str(row[2]),
                completed_at=row[3],
                status=str(row[4]),
                phase=str(row[5]),
                queued_file_count=int(row[6] or 0),
                needed_file_count=int(row[7] or 0),
                succeeded_file_count=int(row[8] or 0),
                failed_file_count=int(row[9] or 0),
                input_bytes=int(row[10] or 0),
                source_payload_read_bytes=int(row[11] or 0),
                cursor_fingerprint_read_bytes=int(row[12] or 0),
                parse_time_s=float(row[13] or 0),
                convergence_time_s=float(row[14] or 0),
                current_source=row[15],
                current_path=row[16],
                error=row[17],
                rss_current_mb=float(row[18]) if row[18] is not None else None,
                rss_peak_self_mb=float(row[19]) if row[19] is not None else None,
                rss_peak_children_mb=float(row[20]) if row[20] is not None else None,
                cgroup_path=row[21],
                cgroup_memory_current_mb=float(row[22]) if row[22] is not None else None,
                cgroup_memory_peak_mb=float(row[23]) if row[23] is not None else None,
                cgroup_memory_swap_current_mb=float(row[24]) if row[24] is not None else None,
                worker_in_flight_count=int(row[25]) if row[25] is not None else None,
                worker_completed_count=int(row[26]) if row[26] is not None else None,
                worker_total_count=int(row[27]) if row[27] is not None else None,
                stale_cursor_write_count=int(row[28] or 0),
                source_paths_json=str(row[29] or "[]"),
            )
            for row in rows
        ]

    def get(self, path: Path) -> int:
        record = self.get_record(path)
        return record.byte_offset if record is not None else 0

    def get_record(self, path: Path) -> CursorRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    source_path,
                    byte_size,
                    byte_offset,
                    last_complete_newline,
                    record_count,
                    updated_at,
                    last_record_ts,
                    parser_fingerprint,
                    content_fingerprint,
                    tail_hash,
                    source_name,
                    st_dev,
                    st_ino,
                    mtime_ns,
                    source_generation,
                    failure_count,
                    next_retry_at,
                    excluded
                FROM live_cursor
                WHERE source_path = ?
                """,
                (str(path),),
            ).fetchone()
        if row is None:
            return None
        return _cursor_record_from_row(row)

    def get_records(self, paths: Iterable[Path]) -> dict[Path, CursorRecord]:
        """Return cursor records for many paths using batched SQLite reads."""
        unique_paths = tuple(dict.fromkeys(paths))
        if not unique_paths:
            return {}
        records_by_source_path: dict[str, CursorRecord] = {}
        with self._connect() as conn:
            for offset in range(0, len(unique_paths), 500):
                chunk = unique_paths[offset : offset + 500]
                placeholders = ",".join("?" for _path in chunk)
                rows = conn.execute(
                    f"""
                    SELECT
                        source_path,
                        byte_size,
                        byte_offset,
                        last_complete_newline,
                        record_count,
                        updated_at,
                        last_record_ts,
                        parser_fingerprint,
                        content_fingerprint,
                        tail_hash,
                        source_name,
                        st_dev,
                        st_ino,
                        mtime_ns,
                        source_generation,
                        failure_count,
                        next_retry_at,
                        excluded
                    FROM live_cursor
                    WHERE source_path IN ({placeholders})
                    """,
                    tuple(str(path) for path in chunk),
                ).fetchall()
                for row in rows:
                    record = _cursor_record_from_row(row)
                    records_by_source_path[record.source_path] = record
        return {path: records_by_source_path[str(path)] for path in unique_paths if str(path) in records_by_source_path}

    def set(
        self,
        path: Path,
        byte_size: int,
        *,
        byte_offset: int | None = None,
        last_complete_newline: int | None = None,
        record_count: int = 0,
        last_record_ts: str | None = None,
        parser_fingerprint: str | None = None,
        content_fingerprint: str | None = None,
        tail_hash: str | None = None,
        source_name: str | None = None,
        st_dev: int | None = None,
        st_ino: int | None = None,
        mtime_ns: int | None = None,
        source_generation: int | None = None,
        failure_count: int | None = None,
        next_retry_at: str | None = None,
        excluded: bool | None = None,
        allow_backward: bool = False,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        offset = byte_size if byte_offset is None else byte_offset
        newline_offset = offset if last_complete_newline is None else last_complete_newline
        excluded_int = 1 if excluded else 0
        with self._connect() as conn:
            if not allow_backward:
                existing = conn.execute(
                    """
                    SELECT byte_size, byte_offset, parser_fingerprint
                    FROM live_cursor
                    WHERE source_path = ?
                    """,
                    (str(path),),
                ).fetchone()
                if (
                    existing is not None
                    and existing[2] == parser_fingerprint
                    and int(existing[0] or 0) > byte_size
                    and int(existing[1] or 0) > offset
                ):
                    return False
            conn.execute(
                """
                INSERT INTO live_cursor (
                    source_path,
                    byte_size,
                    byte_offset,
                    last_complete_newline,
                    record_count,
                    last_record_ts,
                    parser_fingerprint,
                    content_fingerprint,
                    tail_hash,
                    source_name,
                    st_dev,
                    st_ino,
                    mtime_ns,
                    source_generation,
                    failure_count,
                    next_retry_at,
                    excluded,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source_path) DO UPDATE SET
                    byte_size = excluded.byte_size,
                    byte_offset = excluded.byte_offset,
                    last_complete_newline = excluded.last_complete_newline,
                    record_count = excluded.record_count,
                    last_record_ts = excluded.last_record_ts,
                    parser_fingerprint = excluded.parser_fingerprint,
                    content_fingerprint = excluded.content_fingerprint,
                    tail_hash = excluded.tail_hash,
                    source_name = excluded.source_name,
                    st_dev = excluded.st_dev,
                    st_ino = excluded.st_ino,
                    mtime_ns = excluded.mtime_ns,
                    source_generation = excluded.source_generation,
                    failure_count = excluded.failure_count,
                    next_retry_at = excluded.next_retry_at,
                    excluded = excluded.excluded,
                    updated_at = excluded.updated_at
                """,
                (
                    str(path),
                    byte_size,
                    offset,
                    newline_offset,
                    record_count,
                    last_record_ts,
                    parser_fingerprint,
                    content_fingerprint,
                    tail_hash,
                    source_name,
                    st_dev,
                    st_ino,
                    mtime_ns,
                    source_generation or 0,
                    failure_count or 0,
                    next_retry_at,
                    excluded_int,
                    now,
                ),
            )
            conn.commit()
        return True

    def mark_failed(self, path: Path) -> None:
        """Increment failure count and set exponential backoff."""
        record = self.get_record(path)
        if record is None:
            try:
                stat = path.stat()
                byte_size = stat.st_size
                st_dev = stat.st_dev
                st_ino = stat.st_ino
                mtime_ns = stat.st_mtime_ns
            except FileNotFoundError:
                byte_size = 0
                st_dev = None
                st_ino = None
                mtime_ns = None
            self.set(
                path,
                byte_size,
                st_dev=st_dev,
                st_ino=st_ino,
                mtime_ns=mtime_ns,
            )
            record = self.get_record(path)
            if record is None:
                return
        failures = record.failure_count + 1
        delay_s = min(60 * (2 ** (failures - 1)), 3600)  # cap at 1 hour
        retry_at = datetime.now(UTC).timestamp() + delay_s
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_cursor SET failure_count = ?, next_retry_at = ? WHERE source_path = ?",
                (failures, datetime.fromtimestamp(retry_at, tz=UTC).isoformat(), str(path)),
            )
            conn.commit()

    def mark_excluded(self, path: Path) -> None:
        """Quarantine a source file (poison pill)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_cursor SET excluded = 1 WHERE source_path = ?",
                (str(path),),
            )
            conn.commit()

    def reset_failures(self, path: Path) -> None:
        """Clear failure count and backoff after a successful parse."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_cursor SET failure_count = 0, next_retry_at = NULL WHERE source_path = ?",
                (str(path),),
            )
            conn.commit()

    def list_excluded(self) -> list[str]:
        """Return quarantined source paths."""
        with self._connect() as conn:
            rows = conn.execute("SELECT source_path FROM live_cursor WHERE excluded = 1").fetchall()
        return [str(row[0]) for row in rows]

    def list_failed_with_retry(self) -> list[str]:
        """Return sources that have failed and are NOT currently in backoff."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_path FROM live_cursor WHERE failure_count > 0 AND excluded = 0 AND (next_retry_at IS NULL OR next_retry_at <= ?)",
                (now,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def list_failed_records(self) -> list[CursorRecord]:
        """Return all failed, non-excluded cursor records."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    source_path,
                    byte_size,
                    byte_offset,
                    last_complete_newline,
                    record_count,
                    updated_at,
                    last_record_ts,
                    parser_fingerprint,
                    content_fingerprint,
                    tail_hash,
                    source_name,
                    st_dev,
                    st_ino,
                    mtime_ns,
                    source_generation,
                    failure_count,
                    next_retry_at,
                    excluded
                FROM live_cursor
                WHERE failure_count > 0 AND excluded = 0
                ORDER BY next_retry_at IS NULL DESC, next_retry_at ASC, source_path ASC
                """
            ).fetchall()
        return [_cursor_record_from_row(row) for row in rows]

    def record_convergence_debt(
        self,
        *,
        stage: str,
        subject_type: str,
        subject_id: str,
        error: str | None = None,
        materializer_version: str | None = None,
    ) -> None:
        """Record derived convergence debt without marking source ingest failed."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            record_convergence_debt_sync(
                conn,
                stage=stage,
                subject_type=subject_type,
                subject_id=subject_id,
                error=error,
                materializer_version=materializer_version,
                now=now,
            )
            conn.commit()

    def clear_convergence_debt_except(
        self,
        *,
        subject_type: str,
        subject_id: str,
        stages: Iterable[str],
    ) -> None:
        """Clear convergence debt for a subject except currently failed stages."""
        with self._connect() as conn:
            clear_convergence_debt_except_sync(
                conn,
                subject_type=subject_type,
                subject_id=subject_id,
                stages=stages,
            )
            conn.commit()

    def clear_convergence_debt(
        self,
        *,
        subject_type: str,
        subject_id: str,
        stage: str | None = None,
    ) -> None:
        """Clear derived convergence debt after successful convergence."""
        with self._connect() as conn:
            if stage is None:
                conn.execute(
                    "DELETE FROM live_convergence_debt WHERE subject_type = ? AND subject_id = ?",
                    (subject_type, subject_id),
                )
            else:
                conn.execute(
                    """
                    DELETE FROM live_convergence_debt
                    WHERE stage = ? AND subject_type = ? AND subject_id = ?
                    """,
                    (stage, subject_type, subject_id),
                )
            conn.commit()

    def list_convergence_debt(self, *, limit: int = 20) -> list[LiveConvergenceDebt]:
        """Return recent derived convergence debt records."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    stage,
                    subject_type,
                    subject_id,
                    status,
                    failure_count,
                    first_failed_at,
                    last_failed_at,
                    next_retry_at,
                    materializer_version,
                    last_error
                FROM live_convergence_debt
                ORDER BY last_failed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            LiveConvergenceDebt(
                stage=str(row[0]),
                subject_type=str(row[1]),
                subject_id=str(row[2]),
                status=str(row[3]),
                failure_count=int(row[4] or 0),
                first_failed_at=str(row[5]),
                last_failed_at=str(row[6]),
                next_retry_at=_optional_str(row[7]),
                materializer_version=_optional_str(row[8]),
                last_error=_optional_str(row[9]),
            )
            for row in rows
        ]


__all__ = [
    "CursorRecord",
    "CursorStore",
    "LiveConvergenceDebt",
    "LiveIngestAttempt",
]
