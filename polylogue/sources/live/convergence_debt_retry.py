"""Retry scheduling helpers for daemon convergence debt."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_HOT_INSIGHT_DEFERRED = "insights deferred until source quiet"


def convergence_debt_retry_delay_s(failure_count: int, *, error: str | None) -> int:
    if error == _HOT_INSIGHT_DEFERRED:
        return 60
    return int(min(60 * (2 ** (failure_count - 1)), 3600))


def convergence_debt_retry_at(
    conn: sqlite3.Connection,
    *,
    failure_count: int,
    error: str | None,
    subject_type: str,
    subject_id: str,
) -> datetime:
    now = datetime.now(UTC)
    delay_s = convergence_debt_retry_delay_s(failure_count, error=error)
    fallback = datetime.fromtimestamp(now.timestamp() + delay_s, tz=UTC)
    if error != _HOT_INSIGHT_DEFERRED:
        return fallback
    source_path = convergence_debt_source_path(conn, subject_type=subject_type, subject_id=subject_id)
    if source_path is None:
        return fallback
    try:
        stat = source_path.stat()
    except OSError:
        return fallback
    quiet_at = datetime.fromtimestamp(stat.st_mtime + delay_s, tz=UTC)
    return max(now, quiet_at)


def convergence_debt_source_path(
    conn: sqlite3.Connection,
    *,
    subject_type: str,
    subject_id: str,
) -> Path | None:
    if subject_type == "source_path":
        return Path(subject_id)
    if subject_type != "conversation_id":
        return None
    try:
        row = conn.execute(
            """
            SELECT r.source_path
            FROM conversations AS c
            JOIN raw_conversations AS r ON r.raw_id = c.raw_id
            WHERE c.conversation_id = ?
              AND r.source_path IS NOT NULL
              AND r.source_path != ''
            LIMIT 1
            """,
            (subject_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else Path(str(row[0]))


def same_pending_convergence_debt(
    next_retry_at: object,
    last_error: object,
    *,
    error: str | None,
    now: str,
    retry_at: datetime,
) -> bool:
    if last_error != error:
        return False
    existing = parse_retry_datetime(next_retry_at)
    current = parse_retry_datetime(now)
    return existing is not None and current is not None and existing > current and existing >= retry_at


def retry_is_future(next_retry_at: object, *, now: str) -> bool:
    existing = parse_retry_datetime(next_retry_at)
    current = parse_retry_datetime(now)
    return existing is not None and current is not None and existing > current


def parse_retry_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
