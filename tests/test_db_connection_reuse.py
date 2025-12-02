import sqlite3
from pathlib import Path

from polylogue import db


def test_open_connection_reuses_same_handle(tmp_path: Path):
    path = tmp_path / "state.db"
    first_id = None
    with db.open_connection(path) as conn1:
        first_id = id(conn1)
        conn1.execute("CREATE TABLE IF NOT EXISTS t(x INT)")
        conn1.commit()
        with db.open_connection(path) as conn2:
            assert id(conn2) == first_id
            conn2.execute("INSERT INTO t(x) VALUES (1)")
    # connection is closed after outer context
    with db.open_connection(path) as conn3:
        assert id(conn3) != first_id
        row = conn3.execute("SELECT COUNT(*) FROM t").fetchone()
        assert row[0] == 1


def test_open_connection_raises_on_mismatched_path(tmp_path: Path):
    path1 = tmp_path / "a.db"
    path2 = tmp_path / "b.db"
    with db.open_connection(path1):
        try:
            with db.open_connection(path2):  # should refuse mixing paths
                pass
        except RuntimeError as exc:
            assert "Existing connection" in str(exc)
        else:
            assert False, "Expected RuntimeError for mismatched DB paths"
