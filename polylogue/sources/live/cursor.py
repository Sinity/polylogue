"""Per-file processing cursor for live ingestion.

Cursor state enables content-aware skip decisions: same-size rewrites,
truncation, and parser version changes are detected via content fingerprint
comparison rather than relying on file size alone.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from polylogue.core.sources import origin_from_provider
from polylogue.sources.live.convergence_debt_retry import (
    convergence_debt_retry_at,
    retry_is_future,
    same_pending_convergence_debt,
)
from polylogue.sources.live.sqlite_locking import best_effort_cursor_write
from polylogue.storage.sqlite.archive_tiers.archive import _provider_for_origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    add_convergence_debt as add_archive_convergence_debt,
)
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    record_daemon_stage_event as record_archive_daemon_stage_event,
)
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    record_ingest_attempt as record_archive_ingest_attempt,
)
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    upsert_ingest_cursor as upsert_archive_ingest_cursor,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_connection

_INSIGHT_DEFERRED_UNTIL_QUIET = "insights deferred until source quiet"
_MAX_CURSOR_FAILURES_BEFORE_EXCLUDE = 5
_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S = 60

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


def _epoch_ms(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _required_epoch_ms(value: str | None) -> int:
    parsed = _epoch_ms(value)
    if parsed is None:
        return int(datetime.now(UTC).timestamp() * 1000)
    return parsed


def _archive_attempt_status(status: str) -> str:
    if status == "abandoned":
        return "interrupted"
    if status == "completed_with_failures":
        return "completed"
    if status in {"running", "completed", "failed", "interrupted"}:
        return status
    return "failed"


def _storage_route_from_payload(payload: dict[str, object] | None) -> str | None:
    if not payload:
        return None
    route = payload.get("storage_route")
    return route if isinstance(route, str) and route else None


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})"))


def _convergence_debt_status(*, stage: str, error: str | None) -> str:
    if stage == "insights" and error == _INSIGHT_DEFERRED_UNTIL_QUIET:
        return "deferred"
    return "failed"


def _convergence_debt_priority(*, stage: str, subject_type: str, subject_id: str) -> int:
    if stage == "fts" and subject_type == "fts_surface" and subject_id == "messages_fts":
        return 100
    return 0


def _origin_value_for_source_name(source_name: str | None) -> str | None:
    if source_name is None:
        return None
    from polylogue.core.enums import Provider

    return origin_from_provider(Provider.from_string(source_name)).value


def _iso_from_epoch_ms(value: object) -> str:
    return datetime.fromtimestamp(_required_int(value) / 1000, tz=UTC).isoformat()


def _cursor_record_from_ops_row(row: sqlite3.Row | tuple[object, ...]) -> CursorRecord:
    origin = _optional_str(row[14])
    return CursorRecord(
        source_path=str(row[0]),
        byte_size=_required_int(row[1] or 0),
        byte_offset=_required_int(row[2] or 0),
        last_complete_newline=_required_int(row[3] or 0),
        record_count=_required_int(row[4] or 0),
        updated_at=_iso_from_epoch_ms(row[15]),
        last_record_ts=_iso_from_epoch_ms(row[5]) if row[5] is not None else None,
        parser_fingerprint=_optional_str(row[6]),
        content_fingerprint=_optional_str(row[7]),
        tail_hash=_optional_str(row[8]),
        source_name=_provider_for_origin(origin).value if origin else None,
        st_dev=_optional_int(row[9]),
        st_ino=_optional_int(row[10]),
        mtime_ns=_optional_int(row[11]),
        failure_count=_required_int(row[12] or 0),
        next_retry_at=_optional_str(row[13]),
        excluded=bool(row[16]) if row[16] is not None else False,
    )


class CursorStore:
    """SQLite-backed live cursor store keyed by source path."""

    def __init__(self, db_path: Path, *, initialize: bool = True) -> None:
        self._db_path = db_path
        self._ops_db_path = db_path.with_name("ops.db")
        self._initialize_lock = threading.Lock()
        self._initialized = False
        if initialize:
            self.initialize()

    def initialize(self) -> None:
        """Create ops state once; callers may place this under writer ownership."""
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            initialize_archive_database(self._ops_db_path, ArchiveTier.OPS)
            self._initialized = True
            self._mark_interrupted_ops_attempts()

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

    @contextmanager
    def _connect_ops(self) -> Iterator[sqlite3.Connection]:
        conn = open_connection(self._ops_db_path, timeout=10.0)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _mark_interrupted_ops_attempts(self) -> None:
        now_ms = _epoch_ms(datetime.now(UTC).isoformat())

        def write() -> None:
            with self._connect_ops() as conn:
                conn.execute(
                    """
                    UPDATE ingest_attempts
                    SET heartbeat_at_ms = ?,
                        finished_at_ms = ?,
                        status = 'interrupted',
                        phase = 'interrupted',
                        error_message = COALESCE(error_message, 'daemon stopped before completing this ingest attempt')
                    WHERE status = 'running'
                    """,
                    (now_ms, now_ms),
                )
                conn.commit()

        best_effort_cursor_write("archive ops interrupted attempt recovery", write)

    @staticmethod
    def _write_cursor_record_on_conn(conn: sqlite3.Connection, record: CursorRecord) -> None:
        origin = _origin_value_for_source_name(record.source_name)
        upsert_archive_ingest_cursor(
            conn,
            source_path=record.source_path,
            updated_at_ms=_required_epoch_ms(record.updated_at),
            origin=origin,
            stat_size=record.byte_size,
            byte_offset=record.byte_offset,
            last_complete_newline=record.last_complete_newline,
            record_count=record.record_count,
            last_record_ts_ms=_epoch_ms(record.last_record_ts),
            parser_fingerprint=record.parser_fingerprint,
            content_fingerprint=record.content_fingerprint,
            tail_hash=record.tail_hash,
            st_dev=record.st_dev,
            st_ino=record.st_ino,
            mtime_ns=record.mtime_ns,
            failure_count=record.failure_count,
            next_retry_at=record.next_retry_at,
            excluded=bool(record.excluded),
        )

    def _write_cursor_record_to_ops(self, record: CursorRecord) -> None:
        with self._connect_ops() as conn:
            self._write_cursor_record_on_conn(conn, record)

    def _sync_cursor_record_to_ops(self, record: CursorRecord) -> bool:
        def write() -> None:
            self._write_cursor_record_to_ops(record)

        return best_effort_cursor_write("archive ops cursor sync", write)

    def _read_modify_write_cursor_record(
        self, path: Path, mutate: Callable[[CursorRecord | None], CursorRecord | None]
    ) -> None:
        """Read the current record and write the mutated result inside ONE
        connection/transaction, so no other writer can observe or clobber the
        intermediate state (#2467-class lost-update race, polylogue-qug2).

        ``BEGIN IMMEDIATE`` takes the write lock before the read, closing the
        window ``get_record`` + ``set`` used to leave open across two
        connections. ``mutate`` returns ``None`` to skip the write entirely
        (e.g. there is no existing record to mutate).
        """

        def write() -> None:
            with self._connect_ops() as conn:
                conn.execute("BEGIN IMMEDIATE")
                current = self._get_record_on_conn(conn, path)
                updated = mutate(current)
                if updated is not None:
                    self._write_cursor_record_on_conn(conn, updated)

        best_effort_cursor_write("archive ops cursor read-modify-write", write)

    def _sync_convergence_debt_to_ops(
        self,
        *,
        stage: str,
        subject_type: str,
        subject_id: str,
        error: str | None,
        materializer_version: str | None,
        now: str,
    ) -> None:
        """Read the current debt row and write the updated one inside ONE
        connection/transaction (``BEGIN IMMEDIATE`` before the read), so a
        concurrent caller for the same (stage, target_type, target_id) can't
        base its own attempts_delta/same-pending decision on a stale read
        that a second in-flight writer is about to invalidate -- the sibling
        race to polylogue-qug2's cursor lost-update, same root cause.
        """
        now_ms = _required_epoch_ms(now)

        def write() -> None:
            with self._connect_ops() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT attempts, next_retry_at, last_error
                    FROM convergence_debt
                    WHERE stage = ? AND target_type = ? AND target_id = ?
                    """,
                    (stage, subject_type, subject_id),
                ).fetchone()
                existing_attempts = int(row[0]) if row is not None else 0
                retry_at = convergence_debt_retry_at(
                    conn,
                    failure_count=max(existing_attempts, 1),
                    error=error,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    archive_root=self._db_path.parent,
                )
                if row is not None and same_pending_convergence_debt(
                    row[1], row[2], error=error, now=now, retry_at=retry_at
                ):
                    return
                attempts_delta = 0 if row is not None and retry_is_future(row[1], now=now) and row[2] == error else 1
                failure_count = existing_attempts + attempts_delta
                retry_at = convergence_debt_retry_at(
                    conn,
                    failure_count=max(failure_count, 1),
                    error=error,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    archive_root=self._db_path.parent,
                )
                add_archive_convergence_debt(
                    conn,
                    stage=stage,
                    target_type=subject_type,
                    target_id=subject_id,
                    status=_convergence_debt_status(stage=stage, error=error),
                    priority=_convergence_debt_priority(
                        stage=stage,
                        subject_type=subject_type,
                        subject_id=subject_id,
                    ),
                    attempts=attempts_delta,
                    last_error=error,
                    next_retry_at=retry_at.isoformat(),
                    materializer_version=materializer_version,
                    created_at_ms=now_ms,
                    updated_at_ms=now_ms,
                )

        best_effort_cursor_write("archive ops convergence debt sync", write)

    def _clear_convergence_debt_from_ops(
        self,
        *,
        subject_type: str,
        subject_id: str,
        stage: str | None = None,
    ) -> None:
        def write() -> None:
            with self._connect_ops() as conn:
                if stage is None:
                    conn.execute(
                        "DELETE FROM convergence_debt WHERE target_type = ? AND target_id = ?",
                        (subject_type, subject_id),
                    )
                else:
                    conn.execute(
                        """
                        DELETE FROM convergence_debt
                        WHERE stage = ? AND target_type = ? AND target_id = ?
                        """,
                        (stage, subject_type, subject_id),
                    )
                conn.commit()

        best_effort_cursor_write("archive ops convergence debt clear", write)

    def _clear_convergence_debt_except_from_ops(
        self,
        *,
        subject_type: str,
        subject_id: str,
        stages: Iterable[str],
    ) -> None:
        preserved = tuple(stages)

        def write() -> None:
            with self._connect_ops() as conn:
                if preserved:
                    placeholders = ",".join("?" for _ in preserved)
                    conn.execute(
                        f"""
                        DELETE FROM convergence_debt
                        WHERE target_type = ?
                          AND target_id = ?
                          AND stage NOT IN ({placeholders})
                        """,
                        (subject_type, subject_id, *preserved),
                    )
                else:
                    conn.execute(
                        "DELETE FROM convergence_debt WHERE target_type = ? AND target_id = ?",
                        (subject_type, subject_id),
                    )
                conn.commit()

        best_effort_cursor_write("archive ops convergence debt clear-except", write)

    def begin_ingest_attempt(
        self,
        *,
        paths: list[Path],
        input_bytes: int,
        queued_file_count: int,
    ) -> str:
        """Record a durable in-flight live-ingest attempt."""
        now = datetime.now(UTC).isoformat()
        now_ms = _required_epoch_ms(now)
        attempt_id = str(uuid.uuid4())
        with self._connect_ops() as conn:
            record_archive_ingest_attempt(
                conn,
                attempt_id=attempt_id,
                status="running",
                phase="planning",
                started_at_ms=now_ms,
                heartbeat_at_ms=now_ms,
                source_paths_json=json.dumps([str(path) for path in paths], separators=(",", ":")),
            )
            record_archive_daemon_stage_event(
                conn,
                attempt_id=attempt_id,
                stage="planning",
                status="running",
                observed_at_ms=now_ms,
                payload={
                    "phase": "planning",
                    "status": "running",
                    "queued_file_count": queued_file_count,
                    "needed_file_count": queued_file_count,
                    "input_bytes": input_bytes,
                },
            )
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
        stage_payload: dict[str, object] | None = None,
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
        now_ms = _required_epoch_ms(datetime.now(UTC).isoformat())
        parsed_raw_count = succeeded_file_count if succeeded_file_count is not None else None
        if parsed_raw_count is None and failed_file_count is not None:
            parsed_raw_count = failed_file_count
        storage_route = _storage_route_from_payload(stage_payload)

        def write() -> None:
            with self._connect_ops() as conn:
                if _table_has_column(conn, "ingest_attempts", "storage_route"):
                    conn.execute(
                        """
                        UPDATE ingest_attempts
                        SET heartbeat_at_ms = ?,
                            status = ?,
                            phase = ?,
                            storage_route = COALESCE(?, storage_route),
                            source_path = COALESCE(?, source_path),
                            origin = COALESCE(?, origin),
                            parsed_raw_count = COALESCE(?, parsed_raw_count),
                            materialized_count = COALESCE(?, materialized_count),
                            error_message = COALESCE(?, error_message)
                        WHERE attempt_id = ?
                        """,
                        (
                            now_ms,
                            _archive_attempt_status(status),
                            phase,
                            storage_route,
                            str(current_path) if current_path is not None else None,
                            _origin_value_for_source_name(current_source),
                            parsed_raw_count,
                            succeeded_file_count,
                            error,
                            attempt_id,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE ingest_attempts
                        SET heartbeat_at_ms = ?,
                            status = ?,
                            phase = ?,
                            source_path = COALESCE(?, source_path),
                            origin = COALESCE(?, origin),
                            parsed_raw_count = COALESCE(?, parsed_raw_count),
                            materialized_count = COALESCE(?, materialized_count),
                            error_message = COALESCE(?, error_message)
                        WHERE attempt_id = ?
                        """,
                        (
                            now_ms,
                            _archive_attempt_status(status),
                            phase,
                            str(current_path) if current_path is not None else None,
                            _origin_value_for_source_name(current_source),
                            parsed_raw_count,
                            succeeded_file_count,
                            error,
                            attempt_id,
                        ),
                    )
                payload: dict[str, object] = {
                    key: value
                    for key, value in {
                        "phase": phase,
                        "status": status,
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
                    }.items()
                    if value is not None
                }
                record_archive_daemon_stage_event(
                    conn,
                    attempt_id=attempt_id,
                    stage=phase,
                    status=_archive_attempt_status(status),
                    observed_at_ms=now_ms,
                    payload=payload,
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
        stage_payload: dict[str, object] | None = None,
        stage_timings_json: str | None = None,
    ) -> bool:
        """Append one durable progress event for a live-ingest attempt."""
        observed_at_ms = _required_epoch_ms(datetime.now(UTC).isoformat())

        def write() -> None:
            payload: dict[str, object] = {
                key: value
                for key, value in {
                    "queued_file_count": queued_file_count,
                    "needed_file_count": needed_file_count,
                    "skipped_file_count": skipped_file_count,
                    "succeeded_file_count": succeeded_file_count,
                    "failed_file_count": failed_file_count,
                    "input_bytes": input_bytes,
                    "source_payload_read_bytes": source_payload_read_bytes,
                    "cursor_fingerprint_read_bytes": cursor_fingerprint_read_bytes,
                    "archive_write_bytes_delta": archive_write_bytes_delta,
                    "parse_time_s": parse_time_s,
                    "convergence_time_s": convergence_time_s,
                    "total_time_s": total_time_s,
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
                    "stage_timings_json": stage_timings_json,
                }.items()
                if value is not None
            }
            payload.update({"phase": phase, "status": status})
            if stage_payload:
                payload.update(stage_payload)
            with self._connect_ops() as conn:
                record_archive_daemon_stage_event(
                    conn,
                    attempt_id=attempt_id,
                    stage=phase,
                    status=_archive_attempt_status(status),
                    observed_at_ms=observed_at_ms,
                    payload=payload,
                )

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
        now_ms = _required_epoch_ms(datetime.now(UTC).isoformat())

        def write() -> None:
            with self._connect_ops() as conn:
                conn.execute(
                    """
                    UPDATE ingest_attempts
                    SET heartbeat_at_ms = ?,
                        finished_at_ms = ?,
                        status = ?,
                        phase = ?,
                        error_message = ?
                    WHERE attempt_id = ?
                    """,
                    (now_ms, now_ms, _archive_attempt_status(status), phase, error, attempt_id),
                )
                conn.commit()

        return best_effort_cursor_write("live ingest attempt finish", write)

    def recent_ingest_attempts(self, *, limit: int = 5) -> list[LiveIngestAttempt]:
        """Return recent live-ingest attempts for status/debug surfaces."""
        with self._connect_ops() as conn:
            rows = conn.execute(
                """
                SELECT
                    attempt_id,
                    started_at_ms,
                    heartbeat_at_ms,
                    finished_at_ms,
                    status,
                    phase,
                    source_path,
                    origin,
                    parsed_raw_count,
                    materialized_count,
                    error_message,
                    source_paths_json
                FROM ingest_attempts
                ORDER BY COALESCE(heartbeat_at_ms, started_at_ms) DESC, started_at_ms DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            attempt_ids = [str(row[0]) for row in rows]
            events_by_attempt: dict[str, list[dict[str, object]]] = {attempt_id: [] for attempt_id in attempt_ids}
            if attempt_ids:
                placeholders = ",".join("?" for _attempt_id in attempt_ids)
                event_rows = conn.execute(
                    f"""
                    SELECT attempt_id, payload_json
                    FROM daemon_stage_events
                    WHERE attempt_id IN ({placeholders})
                    ORDER BY observed_at_ms ASC, event_id ASC
                    """,
                    tuple(attempt_ids),
                ).fetchall()
                for attempt_id, payload_json in event_rows:
                    if attempt_id is None:
                        continue
                    try:
                        payload = json.loads(str(payload_json or "{}"))
                    except json.JSONDecodeError:
                        payload = {}
                    if isinstance(payload, dict):
                        events_by_attempt.setdefault(str(attempt_id), []).append(payload)

        def metric(attempt_id: str, key: str, default: object) -> object:
            value: object = default
            for payload in events_by_attempt.get(attempt_id, []):
                if key in payload:
                    value = payload[key]
            return value

        def int_metric(attempt_id: str, key: str) -> int:
            value = metric(attempt_id, key, 0)
            return value if isinstance(value, int) else 0

        def float_metric(attempt_id: str, key: str) -> float:
            value = metric(attempt_id, key, 0.0)
            return float(value) if isinstance(value, int | float) else 0.0

        def optional_float_metric(attempt_id: str, key: str) -> float | None:
            value = metric(attempt_id, key, None)
            return float(value) if isinstance(value, int | float) else None

        def optional_int_metric(attempt_id: str, key: str) -> int | None:
            value = metric(attempt_id, key, None)
            return value if isinstance(value, int) else None

        def optional_str_metric(attempt_id: str, key: str, fallback: object = None) -> str | None:
            value = metric(attempt_id, key, fallback)
            return value if isinstance(value, str) else None

        return [
            LiveIngestAttempt(
                attempt_id=str(row[0]),
                started_at=_iso_from_epoch_ms(row[1]),
                updated_at=_iso_from_epoch_ms(row[2] if row[2] is not None else row[1]),
                completed_at=_iso_from_epoch_ms(row[3]) if row[3] is not None else None,
                status=str(row[4]),
                phase=str(row[5]),
                queued_file_count=int_metric(str(row[0]), "queued_file_count"),
                needed_file_count=int_metric(str(row[0]), "needed_file_count"),
                succeeded_file_count=int(row[8] or 0),
                failed_file_count=int_metric(str(row[0]), "failed_file_count"),
                input_bytes=int_metric(str(row[0]), "input_bytes"),
                source_payload_read_bytes=int_metric(str(row[0]), "source_payload_read_bytes"),
                cursor_fingerprint_read_bytes=int_metric(str(row[0]), "cursor_fingerprint_read_bytes"),
                parse_time_s=float_metric(str(row[0]), "parse_time_s"),
                convergence_time_s=float_metric(str(row[0]), "convergence_time_s"),
                current_source=row[7],
                current_path=row[6],
                error=row[10],
                rss_current_mb=optional_float_metric(str(row[0]), "rss_current_mb"),
                rss_peak_self_mb=optional_float_metric(str(row[0]), "rss_peak_self_mb"),
                rss_peak_children_mb=optional_float_metric(str(row[0]), "rss_peak_children_mb"),
                cgroup_path=optional_str_metric(str(row[0]), "cgroup_path"),
                cgroup_memory_current_mb=optional_float_metric(str(row[0]), "cgroup_memory_current_mb"),
                cgroup_memory_peak_mb=optional_float_metric(str(row[0]), "cgroup_memory_peak_mb"),
                cgroup_memory_swap_current_mb=optional_float_metric(str(row[0]), "cgroup_memory_swap_current_mb"),
                cgroup_memory_anon_mb=optional_float_metric(str(row[0]), "cgroup_memory_anon_mb"),
                cgroup_memory_file_mb=optional_float_metric(str(row[0]), "cgroup_memory_file_mb"),
                cgroup_memory_inactive_file_mb=optional_float_metric(str(row[0]), "cgroup_memory_inactive_file_mb"),
                worker_in_flight_count=optional_int_metric(str(row[0]), "worker_in_flight_count"),
                worker_completed_count=optional_int_metric(str(row[0]), "worker_completed_count"),
                worker_total_count=optional_int_metric(str(row[0]), "worker_total_count"),
                stale_cursor_write_count=int_metric(str(row[0]), "stale_cursor_write_count"),
                source_paths_json=str(row[11] or "[]"),
            )
            for row in rows
        ]

    def get(self, path: Path) -> int:
        record = self.get_record(path)
        return record.byte_offset if record is not None else 0

    def get_record(self, path: Path) -> CursorRecord | None:
        with self._connect_ops() as conn:
            return self._get_record_on_conn(conn, path)

    @staticmethod
    def _get_record_on_conn(conn: sqlite3.Connection, path: Path) -> CursorRecord | None:
        row = conn.execute(
            """
            SELECT
                source_path,
                stat_size,
                byte_offset,
                last_complete_newline,
                record_count,
                last_record_ts_ms,
                parser_fingerprint,
                content_fingerprint,
                tail_hash,
                st_dev,
                st_ino,
                mtime_ns,
                failure_count,
                next_retry_at,
                origin,
                updated_at_ms,
                excluded
            FROM ingest_cursor
            WHERE source_path = ?
            """,
            (str(path),),
        ).fetchone()
        if row is None:
            return None
        return _cursor_record_from_ops_row(row)

    def get_records(self, paths: Iterable[Path]) -> dict[Path, CursorRecord]:
        """Return cursor records for many paths using batched SQLite reads."""
        unique_paths = tuple(dict.fromkeys(paths))
        if not unique_paths:
            return {}
        records_by_source_path: dict[str, CursorRecord] = {}
        with self._connect_ops() as conn:
            for offset in range(0, len(unique_paths), 500):
                chunk = unique_paths[offset : offset + 500]
                placeholders = ",".join("?" for _path in chunk)
                rows = conn.execute(
                    f"""
                    SELECT
                        source_path,
                        stat_size,
                        byte_offset,
                        last_complete_newline,
                        record_count,
                        last_record_ts_ms,
                        parser_fingerprint,
                        content_fingerprint,
                        tail_hash,
                        st_dev,
                        st_ino,
                        mtime_ns,
                        failure_count,
                        next_retry_at,
                        origin,
                        updated_at_ms,
                        excluded
                    FROM ingest_cursor
                    WHERE source_path IN ({placeholders})
                    """,
                    tuple(str(path) for path in chunk),
                ).fetchall()
                for row in rows:
                    record = _cursor_record_from_ops_row(row)
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
        del source_generation
        if not allow_backward:
            existing = self.get_record(path)
            if (
                existing is not None
                and existing.parser_fingerprint == parser_fingerprint
                and existing.byte_size > byte_size
                and existing.byte_offset > offset
            ):
                return False
        return self._sync_cursor_record_to_ops(
            CursorRecord(
                source_path=str(path),
                byte_size=byte_size,
                byte_offset=offset,
                last_complete_newline=newline_offset,
                record_count=record_count,
                updated_at=now,
                last_record_ts=last_record_ts,
                parser_fingerprint=parser_fingerprint,
                content_fingerprint=content_fingerprint,
                tail_hash=tail_hash,
                source_name=source_name,
                st_dev=st_dev,
                st_ino=st_ino,
                mtime_ns=mtime_ns,
                failure_count=failure_count or 0,
                next_retry_at=next_retry_at,
                excluded=bool(excluded),
            )
        )

    def mark_failed(self, path: Path, *, failed_stat: os.stat_result | None = None) -> None:
        """Increment failure count and set exponential backoff.

        Read-modify-write against ``failure_count`` happens inside one
        connection/transaction (``_read_modify_write_cursor_record``) so a
        second concurrent caller (e.g. the daemon watcher racing a CLI
        reprocess batch on the same ``source_path``) cannot silently clobber
        this increment with a stale read of its own — see polylogue-qug2.
        """

        def mutate(current: CursorRecord | None) -> CursorRecord | None:
            record = current
            if record is None:
                try:
                    stat = path.stat()
                    byte_size = stat.st_size
                    st_dev: int | None = stat.st_dev
                    st_ino: int | None = stat.st_ino
                    mtime_ns: int | None = stat.st_mtime_ns
                except FileNotFoundError:
                    byte_size = 0
                    st_dev = None
                    st_ino = None
                    mtime_ns = None
                record = CursorRecord(
                    source_path=str(path),
                    byte_size=byte_size,
                    byte_offset=byte_size,
                    last_complete_newline=byte_size,
                    record_count=0,
                    updated_at=datetime.now(UTC).isoformat(),
                    st_dev=st_dev,
                    st_ino=st_ino,
                    mtime_ns=mtime_ns,
                )
            failures = record.failure_count + 1
            now = datetime.now(UTC).isoformat()
            if failures >= _MAX_CURSOR_FAILURES_BEFORE_EXCLUDE:
                # Successful cursors intentionally retain the last accepted
                # observation while an updated file is merely in backoff.  At
                # quarantine, however, bind exclusion to the exact failed
                # observation.  Otherwise the watcher mistakes the unchanged
                # poison file for a replacement and immediately revives it.
                observed = failed_stat
                return replace(
                    record,
                    updated_at=now,
                    byte_size=observed.st_size if observed is not None else record.byte_size,
                    st_dev=observed.st_dev if observed is not None else record.st_dev,
                    st_ino=observed.st_ino if observed is not None else record.st_ino,
                    mtime_ns=observed.st_mtime_ns if observed is not None else record.mtime_ns,
                    failure_count=failures,
                    next_retry_at=None,
                    excluded=True,
                )
            delay_s = min(60 * (2 ** (failures - 1)), 3600)  # cap at 1 hour
            retry_at = datetime.now(UTC).timestamp() + delay_s
            return replace(
                record,
                updated_at=now,
                failure_count=failures,
                next_retry_at=datetime.fromtimestamp(retry_at, tz=UTC).isoformat(),
            )

        self._read_modify_write_cursor_record(path, mutate)

    def defer_full_cursor_reconciliation(self, path: Path) -> None:
        """Retry archive-backed full-cursor handoff without poisoning the source.

        A raw full capture can be durable and parseable while its live JSONL
        source is still appending too quickly to establish a stable handoff.
        This is a scheduling condition, not a parse/persistence failure: it
        must never consume the finite failure budget that quarantines malformed
        files.
        """

        def mutate(current: CursorRecord | None) -> CursorRecord | None:
            if current is None:
                return None
            now = datetime.now(UTC)
            return replace(
                current,
                updated_at=now.isoformat(),
                failure_count=0,
                next_retry_at=(now + timedelta(seconds=_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S)).isoformat(),
                excluded=False,
            )

        self._read_modify_write_cursor_record(path, mutate)

    def mark_excluded(self, path: Path) -> None:
        """Quarantine a source file (poison pill)."""

        def mutate(current: CursorRecord | None) -> CursorRecord | None:
            if current is None:
                return None
            return replace(current, updated_at=datetime.now(UTC).isoformat(), excluded=True)

        self._read_modify_write_cursor_record(path, mutate)

    def revive_replaced_exclusion(
        self,
        path: Path,
        *,
        byte_size: int,
        st_dev: int,
        st_ino: int,
        mtime_ns: int,
    ) -> None:
        """Clear a path quarantine only after the observed file was replaced.

        Exclusion belongs to the exact failed file observation, not forever to
        its pathname.  Keep the prior cursor observation intact so the caller
        still routes the replacement through full acquisition.
        """

        def mutate(current: CursorRecord | None) -> CursorRecord | None:
            if current is None or not current.excluded:
                return None
            if (
                current.byte_size,
                current.st_dev,
                current.st_ino,
                current.mtime_ns,
            ) == (byte_size, st_dev, st_ino, mtime_ns):
                return None
            return replace(
                current,
                updated_at=datetime.now(UTC).isoformat(),
                failure_count=0,
                next_retry_at=None,
                excluded=False,
            )

        self._read_modify_write_cursor_record(path, mutate)

    def reset_failures(self, path: Path) -> None:
        """Clear failure count and backoff after a successful parse."""

        def mutate(current: CursorRecord | None) -> CursorRecord | None:
            if current is None:
                return None
            return replace(
                current,
                updated_at=datetime.now(UTC).isoformat(),
                failure_count=0,
                next_retry_at=None,
            )

        self._read_modify_write_cursor_record(path, mutate)

    def list_excluded(self) -> list[str]:
        """Return quarantined source paths."""
        with self._connect_ops() as conn:
            rows = conn.execute("SELECT source_path FROM ingest_cursor WHERE excluded = 1").fetchall()
        return [str(row[0]) for row in rows]

    def list_failed_with_retry(self) -> list[str]:
        """Return sources that have failed and are NOT currently in backoff."""
        now = datetime.now(UTC).isoformat()
        with self._connect_ops() as conn:
            rows = conn.execute(
                "SELECT source_path FROM ingest_cursor WHERE failure_count > 0 AND excluded = 0 AND (next_retry_at IS NULL OR next_retry_at <= ?)",
                (now,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def list_retry_records(self) -> list[CursorRecord]:
        """Return non-excluded cursor records with a scheduled retry.

        Most records here represent ordinary failed ingestion.  A neutral
        full-cursor handoff is also deliberately included: it has no failure
        count because a source that is still appending is healthy, but it does
        need a timed archive-reconciliation wakeup if filesystem events stop.
        """
        with self._connect_ops() as conn:
            rows = conn.execute(
                """
                SELECT
                    source_path,
                        stat_size,
                        byte_offset,
                        last_complete_newline,
                        record_count,
                        last_record_ts_ms,
                        parser_fingerprint,
                    content_fingerprint,
                    tail_hash,
                    st_dev,
                    st_ino,
                        mtime_ns,
                        failure_count,
                        next_retry_at,
                        origin,
                        updated_at_ms,
                    excluded
                FROM ingest_cursor
                WHERE excluded = 0
                  AND (
                      failure_count > 0
                      OR (content_fingerprint IS NULL AND next_retry_at IS NOT NULL)
                  )
                ORDER BY next_retry_at IS NULL DESC, next_retry_at ASC, source_path ASC
                """
            ).fetchall()
        return [_cursor_record_from_ops_row(row) for row in rows]

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
        self._sync_convergence_debt_to_ops(
            stage=stage,
            subject_type=subject_type,
            subject_id=subject_id,
            error=error,
            materializer_version=materializer_version,
            now=now,
        )

    def clear_convergence_debt_except(
        self,
        *,
        subject_type: str,
        subject_id: str,
        stages: Iterable[str],
    ) -> None:
        """Clear convergence debt for a subject except currently failed stages."""
        preserved_stages = tuple(stages)
        self._clear_convergence_debt_except_from_ops(
            subject_type=subject_type,
            subject_id=subject_id,
            stages=preserved_stages,
        )

    def clear_convergence_debt(
        self,
        *,
        subject_type: str,
        subject_id: str,
        stage: str | None = None,
    ) -> None:
        """Clear derived convergence debt after successful convergence."""
        self._clear_convergence_debt_from_ops(subject_type=subject_type, subject_id=subject_id, stage=stage)

    def list_convergence_debt(self, *, limit: int = 20) -> list[LiveConvergenceDebt]:
        """Return recent derived convergence debt records."""
        with self._connect_ops() as conn:
            rows = conn.execute(
                """
                SELECT
                    stage,
                    target_type,
                    target_id,
                    status,
                    attempts,
                    created_at_ms,
                    updated_at_ms,
                    last_error,
                    next_retry_at,
                    materializer_version
                FROM convergence_debt
                ORDER BY priority DESC, updated_at_ms DESC, debt_id DESC
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
                first_failed_at=_iso_from_epoch_ms(row[5]),
                last_failed_at=_iso_from_epoch_ms(row[6]),
                next_retry_at=_optional_str(row[8]),
                materializer_version=_optional_str(row[9]),
                last_error=_optional_str(row[7]),
            )
            for row in rows
        ]


__all__ = [
    "CursorRecord",
    "CursorStore",
    "LiveConvergenceDebt",
    "LiveIngestAttempt",
]
