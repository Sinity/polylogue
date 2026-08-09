from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite import connection_profile


def test_open_readonly_connection_uses_path_fallback_with_inode_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('selected')")

    descriptor_handle = db_path.open("rb")
    descriptor = descriptor_handle.fileno()
    try:
        monkeypatch.setattr(connection_profile, "_descriptor_database_uri", lambda _fd, _suffix: None)
        reader = connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor)
        try:
            assert reader.execute("SELECT value FROM evidence").fetchone() == ("selected",)
        finally:
            reader.close()
    finally:
        descriptor_handle.close()


def test_open_readonly_connection_path_fallback_rejects_replaced_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "index.db"
    replacement = tmp_path / "replacement.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    descriptor_handle = db_path.open("rb")
    try:
        monkeypatch.setattr(connection_profile, "_descriptor_database_uri", lambda _fd, _suffix: None)
        real_validate = connection_profile._validate_opened_path
        validation_calls = 0

        def replace_after_initial_validation(path: str | Path, fd: int) -> None:
            nonlocal validation_calls
            validation_calls += 1
            real_validate(path, fd)
            if validation_calls == 1:
                db_path.unlink()
                replacement.write_bytes(b"foreign")
                replacement.rename(db_path)

        monkeypatch.setattr(connection_profile, "_validate_opened_path", replace_after_initial_validation)
        with pytest.raises(RuntimeError, match="selected SQLite path was replaced"):
            connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor_handle.fileno())
    finally:
        descriptor_handle.close()
