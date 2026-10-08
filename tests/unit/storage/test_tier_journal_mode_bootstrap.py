"""Bootstrap fixes every tier's journal mode; a writer open never rewrites it.

A reference seal records each tier file's incarnation (device, inode, size,
mtime). ``PRAGMA journal_mode=<mode>`` rewrites the database header even when
nothing changes, so an Index left in rollback-journal mode by bootstrap was
switched to WAL by the first writer open -- after a seal was prepared -- and
the publication refused with "index file incarnation changed after reference
preparation".

Anti-vacuity: create the Index outside WAL at bootstrap, or issue the mode
pragma on every open, and the incarnation recorded before the open differs
from the one after it.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root

_TIERS = ("source.db", "index.db", "user.db", "audit.db", "ops.db")


def _incarnation(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _journal_mode(path: Path) -> str:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def test_bootstrap_creates_every_tier_in_wal(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    assert {name: _journal_mode(tmp_path / name) for name in _TIERS} == dict.fromkeys(_TIERS, "wal")


def test_first_writer_open_leaves_the_index_incarnation_unchanged(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    index = tmp_path / "index.db"
    before = _incarnation(index)

    with ArchiveStore.open_existing(tmp_path, read_only=False):
        during = _incarnation(index)

    assert during == before
