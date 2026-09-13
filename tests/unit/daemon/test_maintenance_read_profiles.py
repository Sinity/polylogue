"""Census the daemon convergence readers that must remain query-only."""

from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.daemon import convergence_stages, convergence_standing_queries
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _open_seeded_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE entries (value TEXT NOT NULL)")


def test_convergence_reader_census_has_no_direct_sqlite_opens() -> None:
    """Mutation: restoring a hand-built reader makes this census fail."""
    for module in (convergence_stages, convergence_standing_queries):
        assert "sqlite3.connect" not in inspect.getsource(module)


@pytest.mark.parametrize(
    "open_reader",
    (open_readonly_connection,),
)
def test_convergence_one_shot_readers_cannot_mutate(
    tmp_path: Path,
    open_reader: Callable[[Path], sqlite3.Connection],
) -> None:
    """Mutation: replacing the declared profile with an ordinary open permits INSERT."""
    database = tmp_path / "maintenance.db"
    _open_seeded_database(database)

    with open_reader(database) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO entries VALUES ('forbidden')")
