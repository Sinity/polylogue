"""Fresh index bootstrap must give the planner correct relative selectivities.

A database without ``sqlite_stat1`` makes the query planner prefer
low-cardinality equality indexes (``idx_blocks_type_tool``) over
session-scoped ones for writer-hot maintenance queries: the per-session
``action_pairs`` refresh then scans the archive's entire ``tool_use``
population on every session write — O(N^2) over a bulk rebuild, measured live
at >20x replay slowdown (polylogue-l3tk, 2026-07-19). Bootstrap therefore
seeds representative statistics so plans are correct from the first write.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.maintenance.rebuild_index import _refresh_generation_planner_statistics
from polylogue.storage.sqlite.action_pairs import action_pairs_refresh_sql
from polylogue.storage.sqlite.schema import _ensure_schema, ensure_schema_async


def _block_search_plans(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "EXPLAIN QUERY PLAN " + action_pairs_refresh_sql("?"),
        ("session", "session", "session"),
    ).fetchall()
    return [str(row[3]) for row in rows if "idx_blocks" in str(row[3])]


def test_fresh_bootstrap_seeds_planner_statistics(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _ensure_schema(conn)
        seeded = conn.execute(
            "SELECT count(*) FROM sqlite_stat1 WHERE tbl IN ('blocks', 'messages', 'action_pairs')"
        ).fetchone()[0]
        assert seeded > 0
    finally:
        conn.close()


def test_fresh_bootstrap_plans_session_scoped_action_pairs_refresh(tmp_path: Path) -> None:
    """Without seeded stats this exact query planned three full
    ``idx_blocks_type_tool (block_type=?)`` scans on a fresh database."""
    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        _ensure_schema(conn)
        plans = _block_search_plans(conn)
        assert plans, "expected block search steps in the action-pairs refresh plan"
        assert all("idx_blocks_session_position" in step for step in plans), plans
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_async_bootstrap_seeds_planner_statistics(tmp_path: Path) -> None:
    async with aiosqlite.connect(tmp_path / "index.db") as conn:
        await ensure_schema_async(conn)
        cursor = await conn.execute("SELECT count(*) FROM sqlite_stat1 WHERE tbl = 'blocks'")
        row = await cursor.fetchone()
        assert row is not None and row[0] > 0


def test_refresh_generation_planner_statistics_measures_real_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute("DELETE FROM sqlite_stat1")
        conn.commit()
    finally:
        conn.close()

    _refresh_generation_planner_statistics(db_path)

    conn = sqlite3.connect(db_path)
    try:
        measured = conn.execute("SELECT count(*) FROM sqlite_stat1").fetchone()[0]
        assert measured > 0
    finally:
        conn.close()


def test_refresh_generation_planner_statistics_tolerates_missing_file(tmp_path: Path) -> None:
    _refresh_generation_planner_statistics(tmp_path / "missing" / "index.db")
