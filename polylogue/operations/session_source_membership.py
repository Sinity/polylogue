"""Active-generation source membership and quiet-file eligibility.

Sinex path publication and profile/usage derivations consume the same retained
source-to-session joins while keeping configured durable paths separate from
the active index generation. This owner acquires and closes its own lookup
reader; connection-based profile probes retain their caller's snapshot.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path

from polylogue.core.evidence import Measured, Unavailable
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.logging import WARNING, emit
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.connection_profile import attach_database, open_readonly_connection
from polylogue.storage.tier_access import capture_sqlite_read

HOT_INSIGHT_SOURCE_BYTES = 64 * 1024 * 1024
_HOT_INSIGHT_QUIET_SECONDS = 60.0


def _source_path_is_hot_for_insights(path: Path, *, now: float | None = None) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size < HOT_INSIGHT_SOURCE_BYTES:
        return False
    current = time.time() if now is None else now
    return current - stat.st_mtime < _HOT_INSIGHT_QUIET_SECONDS


def _attached_source_db_path(conn: sqlite3.Connection, *, archive_root: Path | None = None) -> Path:
    if archive_root is not None:
        return archive_root / "source.db"
    for _, name, path in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "main" and path:
            return Path(str(path)).with_name("source.db")
    return Path("source.db")


def _ensure_source_tier_attached(conn: sqlite3.Connection, *, archive_root: Path | None = None) -> bool:
    for _, name, _path in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "source_tier":
            return True
    source_db = _attached_source_db_path(conn, archive_root=archive_root)
    if not source_db.exists():
        return False
    attach_database(conn, source_db, alias="source_tier")
    return True


def active_archive_index_path(db_path: Path) -> Path | None:
    """Resolve the active ``index.db`` for the archive rooted at ``db_path``'s directory.

    ``db_path`` always lives directly in the archive root (whether it names
    ``index.db``, ``source.db``, or another tier file), so ``db_path.parent``
    is the archive root -- this mirrors ``ArchiveLocation``'s own resolution
    instead of blindly renaming ``db_path`` to ``index.db`` in place, so an
    active ``.index-active-pointer`` generation is still followed correctly.
    """

    index_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    if not index_db.exists():
        return None
    try:
        conn = open_readonly_connection(index_db)
        try:
            return index_db if _table_exists(conn, "sessions") else None
        finally:
            conn.close()
    except Exception as exc:
        emit(
            "daemon.archive.index_probe_failed",
            level=WARNING,
            outcome="degraded",
            reason="active_index_unreadable",
            path=index_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None


def session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
    *,
    archive_root: Path | None = None,
) -> dict[Path, list[str]]:
    normalized_paths = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized_paths or not _table_exists(conn, "sessions"):
        return {path: [] for path in normalized_paths}
    raw_table = "raw_sessions"
    if not _table_exists(conn, "raw_sessions"):
        raw_table = "source_tier.raw_sessions"
        if not _ensure_source_tier_attached(conn, archive_root=archive_root):
            return {path: [] for path in normalized_paths}
        # Attachment/query failures remain exceptions. An unavailable source
        # tier is not evidence that these paths have no retained sessions.
    result: dict[Path, list[str]] = {path: [] for path in normalized_paths}
    paths_by_text = {str(path): path for path in normalized_paths}
    placeholders = ", ".join("?" for _ in normalized_paths)
    rows = conn.execute(
        f"""
        SELECT DISTINCT r.source_path, s.session_id
        FROM {raw_table} AS r
        JOIN sessions AS s ON s.raw_id = r.raw_id
        WHERE r.source_path IN ({placeholders})
        ORDER BY r.source_path, s.session_id
        """,
        tuple(paths_by_text),
    ).fetchall()
    for source_path, session_id in rows:
        path = paths_by_text.get(str(source_path))
        if path is not None:
            result[path].append(str(session_id))
    return result


def hot_insight_session_ids(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    now: float | None = None,
    archive_root: Path | None = None,
) -> set[str]:
    unique_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not unique_ids or not _table_exists(conn, "sessions"):
        return set()
    raw_table = "raw_sessions"
    if not _table_exists(conn, "raw_sessions"):
        raw_table = "source_tier.raw_sessions"

        def attach_source() -> bool:
            try:
                return _ensure_source_tier_attached(conn, archive_root=archive_root)
            except sqlite3.Error as exc:
                emit(
                    "daemon.archive.source_tier_attach_failed",
                    level=WARNING,
                    outcome="degraded",
                    reason="hot_insight_probe_unavailable",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
                raise

        evidence = capture_sqlite_read(attach_source)
        if isinstance(evidence, Unavailable):
            return set()
        if not isinstance(evidence, Measured):
            raise AssertionError("source attachment produced unsupported evidence")
        if not evidence.value:
            return set()
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT s.session_id, r.source_path
        FROM sessions AS s
        JOIN {raw_table} AS r ON r.raw_id = s.raw_id
        WHERE s.session_id IN ({placeholders})
          AND r.source_path IS NOT NULL
          AND r.source_path != ''
        ORDER BY s.session_id
        """,
        unique_ids,
    ).fetchall()
    current = time.time() if now is None else now
    return {
        str(session_id)
        for session_id, source_path in rows
        if _source_path_is_hot_for_insights(Path(str(source_path)), now=current)
    }


def session_ids_for_paths(
    db_path: Path,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    normalized = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized:
        return {}
    lookup_db = active_archive_index_path(db_path) or db_path
    if not lookup_db.exists():
        return {path: [] for path in normalized}
    conn = open_readonly_connection(lookup_db)
    try:
        return session_ids_for_source_paths(conn, normalized, archive_root=db_path.parent)
    finally:
        conn.close()
