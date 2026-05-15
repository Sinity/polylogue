"""Source-file to conversation lookup helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path


def conversation_ids_for_source_path(conn: sqlite3.Connection, path: Path) -> list[str]:
    return conversation_ids_for_source_paths(conn, [path]).get(path, [])


def conversation_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    normalized_paths = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized_paths or not _table_exists(conn, "raw_conversations") or not _table_exists(conn, "conversations"):
        return {}
    result: dict[Path, list[str]] = {path: [] for path in normalized_paths}
    paths_by_text = {str(path): path for path in normalized_paths}
    placeholders = ", ".join("?" for _ in normalized_paths)
    rows = conn.execute(
        f"""
        SELECT DISTINCT r.source_path, c.conversation_id
        FROM raw_conversations AS r
        JOIN conversations AS c ON c.raw_id = r.raw_id
        WHERE r.source_path IN ({placeholders})
        ORDER BY r.source_path, c.conversation_id
        """,
        tuple(paths_by_text),
    ).fetchall()
    for row in rows:
        path = paths_by_text.get(str(row[0]))
        if path is not None:
            result[path].append(str(row[1]))
    return result


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None
