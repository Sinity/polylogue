"""Worker-6 OPS connection, outcome-filter and fresh-bootstrap regressions."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import bootstrap
from polylogue.storage.sqlite.archive_tiers.archive_plan import assert_archive_format_lineage
from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs, upsert_embedding_catchup_run
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_cached_ops_initialization_enables_connection_foreign_keys(tmp_path: Path) -> None:
    path = tmp_path / "ops.db"
    bootstrap.initialize_archive_database(path, ArchiveTier.OPS)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        bootstrap.initialize_archive_tier(conn, ArchiveTier.OPS)
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


@pytest.mark.parametrize("published_before_failure", [False, True])
def test_fresh_bootstrap_format_publication_failure_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_before_failure: bool,
) -> None:
    from polylogue.storage.sqlite.archive_tiers import archive_plan

    original = archive_plan.record_fresh_archive_format

    def interrupted(root: Path) -> Path:
        if published_before_failure:
            original(root)
        raise OSError("injected format publication failure")

    # Bootstrap imports this publisher within its production entry point.
    monkeypatch.setattr(archive_plan, "record_fresh_archive_format", interrupted)
    with pytest.raises(OSError, match="injected format publication failure"):
        bootstrap.initialize_active_archive_root(tmp_path)
    marker_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    assert (marker_root / ".bootstrap.pending").is_file()
    assert not (marker_root / ".bootstrap").exists()
    monkeypatch.setattr(archive_plan, "record_fresh_archive_format", original)
    bootstrap.initialize_active_archive_root(tmp_path)
    from polylogue.storage.sqlite.durable_change_train import durable_train_manifest_paths

    assert durable_train_manifest_paths(marker_root) == ()
    assert not (marker_root / ".bootstrap.pending").exists()
    assert_archive_format_lineage(tmp_path)


def test_embedding_catchup_filter_accepts_completed_with_failures() -> None:
    with closing(sqlite3.connect(":memory:")) as conn:
        bootstrap.initialize_archive_tier(conn, ArchiveTier.OPS)
        upsert_embedding_catchup_run(
            conn,
            run_id="partial",
            started_at_ms=100,
            finished_at_ms=200,
            status="completed_with_failures",
            scanned_sessions=2,
            embedded_sessions=1,
            error_count=1,
        )
        upsert_embedding_catchup_run(conn, run_id="complete", started_at_ms=90, status="completed")
        rows = list_embedding_catchup_runs(conn, status="completed_with_failures")
        assert [(row.run_id, row.status) for row in rows] == [("partial", "completed_with_failures")]
        assert [row.run_id for row in list_embedding_catchup_runs(conn, status="completed")] == ["complete"]
