"""Worker-6 raw-frontier admission and pinned-reader regressions."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.storage.raw_retention import raw_frontier_integrity_projection, raw_frontier_integrity_snapshot_from_connections
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_frontier_projection_refuses_a_source_from_a_newer_schema(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path):
        pass
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        current = int(conn.execute("PRAGMA user_version").fetchone()[0])
        conn.execute(f"PRAGMA user_version = {current + 1}")
        conn.commit()
    projection = raw_frontier_integrity_projection(tmp_path, {})
    assert projection.broken_head_status == "unknown"
    assert "schema" in projection.broken_head_reason.lower()
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == current + 1


def test_pinned_frontier_does_not_substitute_a_live_ops_file_for_missing_reader(tmp_path: Path) -> None:
    for tier in (ArchiveTier.SOURCE, ArchiveTier.INDEX, ArchiveTier.OPS):
        initialize_archive_database(tmp_path / f"{tier.value}.db", tier)
    with (
        closing(sqlite3.connect(tmp_path / "source.db")) as source,
        closing(sqlite3.connect(tmp_path / "index.db")) as index,
        closing(sqlite3.connect(tmp_path / "ops.db")) as ops,
    ):
        known = raw_frontier_integrity_snapshot_from_connections(
            source, index_conn=index, ops_conn=ops, ops_schema="main",
        )
        assert known.cursor_ahead_status == "healthy"
        unavailable = raw_frontier_integrity_snapshot_from_connections(
            source, index_conn=index, ops_conn=None, ops_db_path=tmp_path / "ops.db",
        )
        assert unavailable.broken_head_status == "healthy"
        assert unavailable.cursor_ahead_status == "unknown"
        assert "supplied read snapshot" in unavailable.cursor_ahead_reason
