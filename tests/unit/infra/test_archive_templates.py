"""Regression coverage for reusable SQLite archive templates."""

from __future__ import annotations

import contextlib
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

from tests.infra.archive_templates import clone_archive_template, finalize_archive_template


def _leave_crash_recovered_wal(database: Path) -> None:
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
assert conn.execute(\"PRAGMA journal_mode=WAL\").fetchone() == (\"wal\",)
conn.execute(\"CREATE TABLE entries (value TEXT)\")
conn.execute(\"INSERT INTO entries VALUES ('written-through-wal')\")
conn.commit()
print(\"ready\", flush=True)
sys.stdin.read()
""",
            str(database),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert writer.stdout is not None
    assert writer.stdout.readline() == "ready\n"
    writer.terminate()
    assert writer.wait(timeout=5) != 0


def test_finalized_template_checkpoints_real_wal_before_freezing_and_cloning(tmp_path: Path) -> None:
    """A crash-left WAL must become a self-contained, immutable clone source.

    Anti-vacuity: removing the quiescence phase leaves the real ``-wal`` and
    ``-shm`` files plus WAL journal mode behind, so the sidecar and journal
    assertions below fail before the production clone path can read the row.
    """
    template = tmp_path / "template"
    template.mkdir()
    database = template / "source.db"
    _leave_crash_recovered_wal(database)

    assert (template / "source.db-wal").exists()
    assert (template / "source.db-shm").exists()

    finalize_archive_template(template)
    clone = tmp_path / "clone"
    clone_archive_template(template, clone)

    for root in (template, clone):
        assert not (root / "source.db-wal").exists()
        assert not (root / "source.db-shm").exists()
    assert not (database.stat().st_mode & stat.S_IWUSR)
    assert clone.joinpath("source.db").stat().st_mode & stat.S_IWUSR

    with contextlib.closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    with contextlib.closing(sqlite3.connect(clone / "source.db")) as conn:
        assert conn.execute("SELECT value FROM entries").fetchall() == [("written-through-wal",)]

    assert not (template.stat().st_mode & os.W_OK)
