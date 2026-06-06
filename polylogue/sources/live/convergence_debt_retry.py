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
    archive_root: Path | None = None,
) -> datetime:
    now = datetime.now(UTC)
    delay_s = convergence_debt_retry_delay_s(failure_count, error=error)
    fallback = datetime.fromtimestamp(now.timestamp() + delay_s, tz=UTC)
    if error != _HOT_INSIGHT_DEFERRED:
        return fallback
    source_path = convergence_debt_source_path(
        conn,
        subject_type=subject_type,
        subject_id=subject_id,
        archive_root=archive_root,
    )
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
    archive_root: Path | None = None,
) -> Path | None:
    if subject_type == "source_path":
        return Path(subject_id)
    if subject_type != "session_id":
        return None
    if archive_root is not None:
        root_path = _archive_convergence_debt_source_path_from_root(archive_root, subject_id)
        if root_path is not None:
            return root_path
    archive_path = _archive_convergence_debt_source_path(conn, subject_id)
    return archive_path


def _archive_convergence_debt_source_path_from_root(archive_root: Path, session_id: str) -> Path | None:
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    if not index_db.exists() or not source_db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            conn.execute("ATTACH DATABASE ? AS source_tier", (str(source_db),))
            row = conn.execute(
                """
                SELECT r.source_path
                FROM sessions AS s
                JOIN source_tier.raw_sessions AS r ON r.raw_id = s.raw_id
                WHERE s.session_id = ?
                  AND r.source_path IS NOT NULL
                  AND r.source_path != ''
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            conn.execute("DETACH DATABASE source_tier")
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return None if row is None else Path(str(row[0]))


def _archive_convergence_debt_source_path(conn: sqlite3.Connection, session_id: str) -> Path | None:
    try:
        source_db = _sibling_source_db(conn)
        if source_db is None or not source_db.exists():
            return None
        if not _table_exists(conn, "sessions"):
            return None
        source_alias = _ensure_source_tier_attached(conn, source_db)
        if not _table_exists(conn, "raw_sessions", schema=source_alias):
            return None
        row = conn.execute(
            f"""
            SELECT r.source_path
            FROM sessions AS s
            JOIN {source_alias}.raw_sessions AS r ON r.raw_id = s.raw_id
            WHERE s.session_id = ?
              AND r.source_path IS NOT NULL
              AND r.source_path != ''
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else Path(str(row[0]))


def _ensure_source_tier_attached(conn: sqlite3.Connection, source_db: Path) -> str:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) == "source_tier":
            return "source_tier"
    conn.execute("ATTACH DATABASE ? AS source_tier", (str(source_db),))
    return "source_tier"


def _sibling_source_db(conn: sqlite3.Connection) -> Path | None:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) != "main":
            continue
        path_text = str(row[2] or "")
        if not path_text:
            return None
        return Path(path_text).with_name("source.db")
    return None


def _table_exists(conn: sqlite3.Connection, table: str, *, schema: str = "main") -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


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
