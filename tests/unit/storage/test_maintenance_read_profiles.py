"""Census storage maintenance reader authority and mutation denial."""

from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.storage import blob_gc, raw_authority


def _open_seeded_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE entries (value TEXT NOT NULL)")


def test_storage_reader_census_has_no_direct_sqlite_opens() -> None:
    """Mutation: restoring an unmanaged maintenance reader fails the census."""
    for module in (blob_gc, raw_authority):
        assert "sqlite3.connect" not in inspect.getsource(module)


@pytest.mark.parametrize("open_reader", (blob_gc._readonly, raw_authority._readonly))
def test_storage_one_shot_readers_cannot_mutate(
    tmp_path: Path,
    open_reader: Callable[[Path], sqlite3.Connection],
) -> None:
    """Mutation: changing a storage reader to read-write permits INSERT."""
    database = tmp_path / "maintenance.db"
    _open_seeded_database(database)

    with open_reader(database) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO entries VALUES ('forbidden')")
