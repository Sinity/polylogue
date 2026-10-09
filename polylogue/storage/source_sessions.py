"""Source-file to session lookup helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.storage.sqlite.connection_profile import attach_database


def session_ids_for_source_path(
    conn: sqlite3.Connection,
    path: Path,
    *,
    source_db: Path | None = None,
) -> list[str]:
    return session_ids_for_source_paths(conn, [path], source_db=source_db).get(path, [])


def session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
    *,
    source_db: Path | None = None,
) -> dict[Path, list[str]]:
    normalized_paths = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized_paths:
        return {}
    check_compute_cancelled()
    return _schema_archive_session_ids_for_source_paths(conn, normalized_paths, source_db=source_db)


def _schema_archive_session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
    *,
    source_db: Path | None,
) -> dict[Path, list[str]]:
    resolved_source_db = source_db if source_db is not None else _sibling_source_db(conn)
    if resolved_source_db is None or not resolved_source_db.exists():
        raise ArchiveTierUnavailableError(
            tier="source",
            path=str(resolved_source_db) if resolved_source_db is not None else "source.db",
            reason="Source lookup requires a readable retained tier",
            guidance="Restore readable Source authority, then retry the lookup.",
        )
    result: dict[Path, list[str]] = {path: [] for path in paths}
    paths_by_text = {str(path): path for path in paths}
    placeholders = ", ".join("?" for _ in paths)
    source_alias = _ensure_source_tier_attached(conn, resolved_source_db)
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
    for row in rows:
        path = paths_by_text.get(str(row[0]))
        if path is not None:
            result[path].append(str(row[1]))
    return result


def _ensure_source_tier_attached(conn: sqlite3.Connection, source_db: Path) -> str:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) == "source_tier":
            return "source_tier"
    attach_database(conn, source_db, alias="source_tier")
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
