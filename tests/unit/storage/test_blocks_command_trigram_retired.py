"""``blocks_command_trigram`` is retired from the index tier (polylogue-nv356).

The surface was kept at index v63 for exactly one consumer,
``devtools/affordance_usage.py``'s ``_cli_action_rows``. That module was
deleted on 2026-08-25 (polylogue-9m6ry) with no product-surface replacement,
leaving a trigger-maintained FTS5 index, a generated ``blocks.tool_detail_text``
projection that existed only to feed it, and archive-wide rebuild/repair
machinery, all with no query consumer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.errors import SchemaSkew
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _index_objects_like(conn: sqlite3.Connection, pattern: str) -> list[str]:
    return [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE ?", (pattern,))]


def test_fresh_index_tier_declares_no_trigram_objects(tmp_path: Path) -> None:
    """AC1. Anti-vacuity: restore the CREATE VIRTUAL TABLE in index.py and this goes red."""

    with ArchiveStore(tmp_path / "archive") as facade:
        assert _index_objects_like(facade._conn, "blocks_command_trigram%") == []


def test_blocks_no_longer_carries_the_trigram_only_projection(tmp_path: Path) -> None:
    """``tool_detail_text`` had no consumer left once the trigram table went.

    Anti-vacuity: re-add the generated column to ``archive_tiers_specs.py``
    and this goes red. ``search_text`` stays -- it backs ``messages_fts``.
    """

    with ArchiveStore(tmp_path / "archive") as facade:
        # ``table_xinfo`` -- ``table_info`` omits VIRTUAL generated columns,
        # which is exactly the kind these are.
        columns = {str(row[1]) for row in facade._conn.execute("PRAGMA table_xinfo(blocks)")}

    assert "tool_detail_text" not in columns
    assert "search_text" in columns


def test_the_rebuild_and_repair_machinery_is_gone() -> None:
    """A dropped surface must not leave its rebuild helpers behind.

    Anti-vacuity: re-export any of these names and this goes red.
    """

    from polylogue.storage.fts import fts_lifecycle, sql

    for name in ("rebuild_command_trigram_index_sync",):
        assert not hasattr(fts_lifecycle, name), f"{name} outlived its surface"
    for name in (
        "TRIGRAM_REBUILD_DELETE_ALL_SQL",
        "trigram_delete_session_rows_sql",
        "trigram_insert_session_rows_sql",
        "insert_all_trigram_rows_sql",
    ):
        assert not hasattr(sql, name), f"{name} outlived its surface"


def test_an_index_tier_stamped_at_the_old_identity_refuses_to_open(tmp_path: Path) -> None:
    """AC3. The derived identity moves, so an existing tier meets a typed SchemaSkew.

    Anti-vacuity: suppress the identity comparison on open and this goes red.
    """

    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        assert facade.index_db_path.exists()

    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("UPDATE schema_identity SET identity = ? WHERE tier = 'index'", ("0" * 64,))
        conn.commit()

    with pytest.raises(SchemaSkew):
        with ArchiveStore(root) as facade:
            facade._conn.execute("SELECT 1").fetchone()
