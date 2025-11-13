from __future__ import annotations

import sqlite3

from polylogue.index_health import verify_sqlite_indexes
from polylogue.db import open_connection


def test_verify_sqlite_indexes_rebuilds_missing_table(tmp_path):
    db_path = tmp_path / "polylogue.db"
    with open_connection(db_path):
        pass
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.commit()
    notes = verify_sqlite_indexes(db_path)
    assert any("messages_fts" in note for note in notes)
