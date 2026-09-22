"""Stranded-lineage debt must reach the archive root's ``ops.db``.

``_record_lineage_prefix_debt`` derived ``ops.db`` from the index connection's
own directory. SQLite reports the *physical* file behind ``main``, and a
promoted index is reached through ``.index-generations/<gen>/index.db``, so on
an ordinary daemon archive the helper looked for ``ops.db`` inside the
generation directory, found nothing there, and returned without recording the
retryable debt it exists to record.

Anti-vacuity: restore ``index_path.with_name("ops.db")`` in
``polylogue/storage/sqlite/archive_tiers/write.py`` and
``test_debt_reaches_the_root_ops_tier_...`` goes red with zero debt rows.
``test_no_ops_tier_records_nothing`` pins the opposite direction so a fix that
bootstrapped a disposable tier from the write path cannot pass.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import _record_stranded_branch_point_debt

_GENERATION = ".index-generations/gen-0001"


def _promote_index_generation(root: Path) -> Path:
    """Move ``index.db`` into a generation directory and point the root at it."""
    generation_dir = root / _GENERATION
    generation_dir.mkdir(parents=True)
    promoted = generation_dir / "index.db"
    (root / "index.db").rename(promoted)
    os.symlink(promoted, root / "index.db")
    return promoted


def _debt_rows(ops_db: Path) -> list[tuple[str, str]]:
    if not ops_db.exists():
        return []
    with sqlite3.connect(ops_db) as conn:
        if conn.execute("SELECT 1 FROM sqlite_schema WHERE name = 'convergence_debt'").fetchone() is None:
            return []
        return [
            (str(stage), str(target_id))
            for stage, target_id in conn.execute("SELECT stage, target_id FROM convergence_debt ORDER BY target_id")
        ]


class TestLineageDebtResolvesTheArchiveRoot:
    def test_debt_reaches_the_root_ops_tier_from_a_generation(self, tmp_path: Path) -> None:
        with ArchiveStore(tmp_path):
            pass
        promoted = _promote_index_generation(tmp_path)
        assert (tmp_path / "ops.db").exists()
        assert not (promoted.parent / "ops.db").exists()

        # Open on the generation member exactly as a promoted archive does.
        conn = sqlite3.connect(promoted)
        try:
            _record_stranded_branch_point_debt(conn, {"claude-code-session:ext-child"})
        finally:
            conn.close()

        rows = _debt_rows(tmp_path / "ops.db")
        assert [target for _stage, target in rows] == ["claude-code-session:ext-child"]
        assert not (promoted.parent / "ops.db").exists(), "a sibling ops tier was invented in the generation"

    def test_debt_reaches_the_root_ops_tier_without_a_generation(self, tmp_path: Path) -> None:
        """A plain archive keeps recording debt beside its index."""
        with ArchiveStore(tmp_path):
            pass
        conn = sqlite3.connect(tmp_path / "index.db")
        try:
            _record_stranded_branch_point_debt(conn, {"claude-code-session:ext-plain"})
        finally:
            conn.close()
        assert [target for _stage, target in _debt_rows(tmp_path / "ops.db")] == ["claude-code-session:ext-plain"]

    def test_no_ops_tier_records_nothing(self, tmp_path: Path) -> None:
        """Opposite direction: the write path must not bootstrap a disposable tier."""
        index_db = tmp_path / "index.db"
        conn = sqlite3.connect(index_db)
        try:
            _record_stranded_branch_point_debt(conn, {"claude-code-session:ext-orphan"})
        finally:
            conn.close()
        assert not (tmp_path / "ops.db").exists()
