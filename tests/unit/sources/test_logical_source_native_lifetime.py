"""Logical reconstruction keeps the actual creator and artifact through failure."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.sources import sqlite_export
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    retained_native_sql_owners_on_current_thread,
)
from tests.infra.sqlite_settlement_handle import SettlementHandle


@pytest.mark.parametrize("phase", ["ddl", "population", "reader"])
@pytest.mark.parametrize("failed_close", [False, True])
def test_logical_reconstruction_failure_settles_or_retains_its_actual_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, failed_close: bool
) -> None:
    source = tmp_path / "source.sqlite"
    connection = sqlite3.connect(source)
    try:
        connection.execute("CREATE TABLE messages (id TEXT, body TEXT)")
        connection.execute("INSERT INTO messages VALUES ('one', 'neutral')")
        connection.commit()
    finally:
        connection.close()
    export = tmp_path / "source.export"
    export.write_bytes(sqlite_export.logical_export_bytes(source))
    actual_connect = sqlite3.connect
    targets: list[SettlementHandle] = []
    files: list[Path] = []

    class ConstructionHandle(SettlementHandle):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            if (phase == "ddl" and sql.startswith("CREATE TABLE")) or (
                phase == "population" and sql.startswith("INSERT INTO")
            ):
                raise LookupError("synthetic logical reconstruction refusal")
            return self.connection.execute(sql, *args, **kwargs)

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        native = actual_connect(*args, **kwargs)
        target = (phase == "reader" and "mode=ro" in str(args[0])) or (phase != "reader" and "mode=rwc" in str(args[0]))
        if not target:
            return native
        handle = ConstructionHandle(native)
        files.append(Path(native.execute("PRAGMA database_list").fetchone()[2]))
        if not failed_close:
            handle.allow_cleanup.set()
        targets.append(handle)
        return cast(sqlite3.Connection, handle)

    monkeypatch.setattr(sqlite_export.sqlite3, "connect", connect)
    if failed_close:
        with pytest.raises(NativeConnectionSettlementError) as failure:
            with sqlite_export.logical_source_context(export) as reader:
                assert reader.execute("SELECT body FROM messages").fetchone()[0] == "neutral"
        owner = failure.value.owner
        assert owner in retained_native_sql_owners_on_current_thread()
        assert files[0].is_file()
        directory = files[0].parent
        assert owner.scratch_directory is not None
        if phase != "reader":
            assert isinstance(failure.value.__cause__, LookupError)
        targets[0].allow_cleanup.set()
        owner.close()
        assert not directory.exists()
    elif phase == "reader":
        with sqlite_export.logical_source_context(export) as reader:
            assert reader.execute("SELECT body FROM messages").fetchone()[0] == "neutral"
        assert not files[0].parent.exists()
    else:
        with pytest.raises(LookupError):
            with sqlite_export.logical_source_context(export):
                pytest.fail("construction must fail before yielding a reader")
        assert not files[0].parent.exists()
    assert len(targets) == 1
    assert retained_native_sql_owners_on_current_thread() == ()
    with pytest.raises(sqlite3.ProgrammingError):
        targets[0].connection.execute("SELECT 1")
