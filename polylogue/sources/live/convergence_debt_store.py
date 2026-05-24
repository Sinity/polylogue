"""SQLite operations for live convergence debt."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from polylogue.sources.live.convergence_debt_retry import (
    convergence_debt_retry_at,
    retry_is_future,
    same_pending_convergence_debt,
)


def record_convergence_debt_sync(
    conn: sqlite3.Connection,
    *,
    stage: str,
    subject_type: str,
    subject_id: str,
    error: str | None = None,
    materializer_version: str | None = None,
    now: str,
) -> None:
    row = conn.execute(
        """
        SELECT failure_count, next_retry_at, last_error
        FROM live_convergence_debt
        WHERE stage = ? AND subject_type = ? AND subject_id = ?
        """,
        (stage, subject_type, subject_id),
    ).fetchone()
    existing_failure_count = int(row[0]) if row is not None else 0
    retry_at = convergence_debt_retry_at(
        conn,
        failure_count=max(existing_failure_count, 1),
        error=error,
        subject_type=subject_type,
        subject_id=subject_id,
    )
    if row is not None and same_pending_convergence_debt(row[1], row[2], error=error, now=now, retry_at=retry_at):
        return
    failure_count = (
        existing_failure_count
        if row is not None and retry_is_future(row[1], now=now) and row[2] == error
        else existing_failure_count + 1
    )
    retry_at = convergence_debt_retry_at(
        conn,
        failure_count=failure_count,
        error=error,
        subject_type=subject_type,
        subject_id=subject_id,
    )
    conn.execute(
        """
        INSERT INTO live_convergence_debt (
            stage, subject_type, subject_id, status, failure_count,
            first_failed_at, last_failed_at, next_retry_at,
            materializer_version, last_error
        ) VALUES (?, ?, ?, 'failed', ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stage, subject_type, subject_id) DO UPDATE SET
            status = 'failed',
            failure_count = excluded.failure_count,
            last_failed_at = excluded.last_failed_at,
            next_retry_at = excluded.next_retry_at,
            materializer_version = excluded.materializer_version,
            last_error = excluded.last_error
        """,
        (stage, subject_type, subject_id, failure_count, now, now, retry_at.isoformat(), materializer_version, error),
    )


def clear_convergence_debt_except_sync(
    conn: sqlite3.Connection,
    *,
    subject_type: str,
    subject_id: str,
    stages: Iterable[str],
) -> None:
    failed_stages = tuple(dict.fromkeys(str(stage) for stage in stages if stage))
    if not failed_stages:
        conn.execute(
            "DELETE FROM live_convergence_debt WHERE subject_type = ? AND subject_id = ?",
            (subject_type, subject_id),
        )
        return
    placeholders = ", ".join("?" for _stage in failed_stages)
    conn.execute(
        f"""
        DELETE FROM live_convergence_debt
        WHERE subject_type = ?
          AND subject_id = ?
          AND stage NOT IN ({placeholders})
        """,
        (subject_type, subject_id, *failed_stages),
    )
