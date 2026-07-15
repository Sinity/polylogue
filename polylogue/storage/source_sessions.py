"""Source-file to session lookup helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.storage.table_existence import table_exists as _table_exists

logger = get_logger(__name__)


def session_ids_for_source_path(conn: sqlite3.Connection, path: Path) -> list[str]:
    return session_ids_for_source_paths(conn, [path]).get(path, [])


def session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    normalized_paths = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized_paths:
        return {}
    archive_result = _schema_archive_session_ids_for_source_paths(conn, normalized_paths)
    if archive_result is not None:
        return archive_result
    return {path: [] for path in normalized_paths}


def _schema_archive_session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
) -> dict[Path, list[str]] | None:
    try:
        if not _table_exists(conn, "sessions"):
            return None
        source_db = _sibling_source_db(conn)
        if source_db is None or not source_db.exists():
            return None
        result: dict[Path, list[str]] = {path: [] for path in paths}
        paths_by_text = {str(path): path for path in paths}
        placeholders = ", ".join("?" for _ in paths)
        source_alias = _ensure_source_tier_attached(conn, source_db)
        if not _table_exists(conn, "raw_sessions", schema=source_alias):
            return None
        rows = conn.execute(
            f"""
            SELECT DISTINCT r.source_path, s.session_id
            FROM sessions AS s
            JOIN {source_alias}.raw_sessions AS r ON r.raw_id = s.raw_id
            WHERE r.source_path IN ({placeholders})
            ORDER BY r.source_path, s.session_id
            """,
            tuple(paths_by_text),
        ).fetchall()
    except Exception as exc:
        # session_ids_for_source_paths() falls through to an all-empty
        # dict when this returns None, identical to "no sessions reference
        # these paths" — log so a query failure isn't mistaken for that.
        logger.warning("source-sessions lookup failed: %s", exc, exc_info=True)
        return None
    for row in rows:
        path = paths_by_text.get(str(row[0]))
        if path is not None:
            result[path].append(str(row[1]))
    return result


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
