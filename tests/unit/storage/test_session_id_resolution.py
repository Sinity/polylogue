"""Session-id resolution should stay indexed on large archives."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.storage.sqlite.queries.sessions_identity import resolve_id, session_id_prefix_bounds


def test_session_id_prefix_bounds_make_indexable_range() -> None:
    assert session_id_prefix_bounds("abc") == ("abc", "abd")
    assert session_id_prefix_bounds("abz") == ("abz", "ab{")
    assert session_id_prefix_bounds("") == ("", None)


def test_session_id_prefix_range_uses_primary_key_index(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        lower_bound, upper_bound = session_id_prefix_bounds("abc")
        assert upper_bound is not None
        rows = conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT session_id
            FROM sessions
            WHERE session_id >= ? AND session_id < ?
            ORDER BY session_id
            LIMIT 2
            """,
            (lower_bound, upper_bound),
        ).fetchall()

    plan = " ".join(str(row) for row in rows)
    assert "SEARCH sessions" in plan
    assert "session_id>?" in plan
    assert "session_id<?" in plan


@pytest.mark.asyncio
async def test_resolve_id_keeps_prefix_semantics(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        await conn.executemany(
            "INSERT INTO sessions (session_id) VALUES (?)",
            [("abc-1",), ("abc-2",), ("abd-1",)],
        )
        await conn.commit()

        assert await resolve_id(conn, "abc-1") == "abc-1"
        assert await resolve_id(conn, "abc") is None
        assert await resolve_id(conn, "abd") == "abd-1"
        assert await resolve_id(conn, "missing") is None
