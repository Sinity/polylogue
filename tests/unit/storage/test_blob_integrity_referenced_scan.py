"""Integrity denominators are canonical blob owners, never raw observation IDs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.blob_liveness import project_live_blob_hashes
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.schema import _ensure_schema


def _init_db(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "archive.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    # raw_sessions lives in the source durability tier (#1743).
    conn.executescript(SOURCE_DDL)
    conn.commit()
    return conn


def _insert_raw(conn: sqlite3.Connection, raw_id: str, acquired_at: str) -> None:
    # raw_sessions carries a single ``origin`` column, INTEGER-ms timestamps, and
    # a 32-byte ``blob_hash`` (#1743). Ordering is irrelevant for this scan.
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, source_path, source_index, blob_hash, blob_size,
            acquired_at_ms, file_mtime_ms
        ) VALUES (?, 'codex-session', ?, 0, ?, 1, 0, 0)
        """,
        (raw_id, f"/src/{raw_id}.jsonl", raw_id.encode().ljust(32, b"_")[:32]),
    )


def test_returns_all_direct_blob_hashes_not_raw_ids(tmp_path: Path) -> None:
    conn = _init_db(tmp_path)
    try:
        _insert_raw(conn, "r_b", "2026-01-02")
        _insert_raw(conn, "r_a", "2026-01-03")
        _insert_raw(conn, "r_c", "2026-01-01")
        conn.commit()
        result = project_live_blob_hashes(conn).live_hashes
    finally:
        conn.close()
    assert result == {raw_id.encode().ljust(32, b"_")[:32].hex() for raw_id in ("r_a", "r_b", "r_c")}


def test_empty_table(tmp_path: Path) -> None:
    conn = _init_db(tmp_path)
    try:
        assert project_live_blob_hashes(conn).live_hashes == frozenset()
    finally:
        conn.close()


def test_missing_table_returns_empty(tmp_path: Path) -> None:
    db = tmp_path / "bare.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        assert project_live_blob_hashes(conn).live_hashes == frozenset()
    finally:
        conn.close()


__all__: list[str] = []
