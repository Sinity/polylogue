"""The demo evidence reader uses the shared SQLite read profile."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.demo.receipts import _connect


def test_demo_receipt_reader_reads_and_rejects_sqlite_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.db"
    with sqlite3.connect(db_path) as writer:
        writer.execute("CREATE TABLE evidence (value TEXT NOT NULL)")
        writer.execute("INSERT INTO evidence VALUES ('retained')")

    with closing(_connect(db_path)) as reader:
        assert reader.execute("SELECT value FROM evidence").fetchone()[0] == "retained"
        with pytest.raises(sqlite3.DatabaseError, match="not authorized|readonly"):
            reader.execute("INSERT INTO evidence VALUES ('rejected')")

    with sqlite3.connect(db_path) as writer:
        assert writer.execute("SELECT value FROM evidence").fetchall() == [("retained",)]
