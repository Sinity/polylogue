from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite import connection_profile


def test_open_readonly_connection_uses_descriptor_bound_database(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('selected')")

    descriptor_handle = db_path.open("rb")
    try:
        reader = connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor_handle.fileno())
        try:
            assert reader.execute("SELECT value FROM evidence").fetchone() == ("selected",)
        finally:
            reader.close()
    finally:
        descriptor_handle.close()


def test_open_readonly_connection_refuses_without_descriptor_bound_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    descriptor_handle = db_path.open("rb")
    try:
        monkeypatch.setattr(connection_profile, "_descriptor_database_uri", lambda _fd, _suffix: None)
        with pytest.raises(RuntimeError, match="descriptor-bound path"):
            connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor_handle.fileno())
    finally:
        descriptor_handle.close()
