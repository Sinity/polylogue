"""Tests for the referenced-raw-id scan (#1750 F4).

``_raw_session_hashes`` dropped a redundant ``ORDER BY acquired_at DESC,
raw_id`` — its result is consumed as an unordered set, so the sort over the full
``raw_sessions`` scan was pure overhead. These tests pin the contract that
actually matters: the full set of non-empty ``raw_id`` values is returned
(order irrelevant).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.blob_integrity import _raw_session_hashes
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
        (raw_id, f"/src/{raw_id}.jsonl", bytes(32)),
    )


def test_returns_all_non_empty_raw_ids(tmp_path: Path) -> None:
    conn = _init_db(tmp_path)
    try:
        _insert_raw(conn, "r_b", "2026-01-02")
        _insert_raw(conn, "r_a", "2026-01-03")
        _insert_raw(conn, "r_c", "2026-01-01")
        conn.commit()
        result = set(_raw_session_hashes(conn))
    finally:
        conn.close()
    assert result == {"r_a", "r_b", "r_c"}


def test_empty_table(tmp_path: Path) -> None:
    conn = _init_db(tmp_path)
    try:
        assert _raw_session_hashes(conn) == []
    finally:
        conn.close()


def test_missing_table_returns_empty(tmp_path: Path) -> None:
    db = tmp_path / "bare.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        assert _raw_session_hashes(conn) == []
    finally:
        conn.close()


__all__: list[str] = []
