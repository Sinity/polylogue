"""Direct Codex SQLite reads and retained exports share exact bounded text."""

from __future__ import annotations

import sqlite3
import threading
from builtins import BaseExceptionGroup
from pathlib import Path
from typing import Any, Literal

import pytest

from polylogue.sources import sqlite_export
from polylogue.sources.parsers.codex_state import iter_codex_state_parts
from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import _MeasuredConnection, connect_measured
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    retained_native_sql_owners_on_current_thread,
)
from polylogue.storage.sqlite.managed_connection import sqlite_connection
from tests.infra.native_sql_descriptor_probe import selected_file_descriptors
from tests.infra.sqlite_cursor_settlement import ControlledCursor


def _state(path: Path, kind: Literal["goals", "memories"], text: str | None, *, generated: bool = True) -> None:
    with sqlite_connection(path) as connection:
        witness = ", witness INTEGER GENERATED ALWAYS AS (length(thread_id)) VIRTUAL" if generated else ""
        if kind == "goals":
            connection.execute(
                f"CREATE TABLE thread_goals(thread_id TEXT,goal_id TEXT,objective TEXT,status TEXT{witness})"
            ).close()
            connection.execute(
                "INSERT INTO thread_goals(thread_id,goal_id,objective,status) VALUES('synthetic-thread','synthetic-goal',?,'active')",
                (text,),
            ).close()
        else:
            connection.execute(
                f"CREATE TABLE stage1_outputs(thread_id TEXT,raw_memory TEXT,rollout_summary TEXT,source_updated_at INTEGER DEFAULT 0,generated_at INTEGER DEFAULT 0,usage_count INTEGER,selected_for_phase2 INTEGER DEFAULT 0{witness})"
            ).close()
            connection.execute(
                "INSERT INTO stage1_outputs(thread_id,raw_memory,rollout_summary) VALUES('synthetic-thread',?,'')",
                (text,),
            ).close()
        connection.commit()


@pytest.mark.parametrize("kind", ["goals", "memories"])
@pytest.mark.parametrize("text", [None, "", "λ\x00中" * 40000])
def test_generated_codex_live_table_matches_retained_export(
    tmp_path: Path, kind: Literal["goals", "memories"], text: str | None
) -> None:
    path = tmp_path / f"{kind}.sqlite"
    _state(path, kind, text)
    parts = list(iter_codex_state_parts(path, state_kind=kind, text_chars=8192))
    field = "objective" if kind == "goals" else "raw_memory"
    recovered = str(parts[0].payload[field]) + "".join(
        str(part.payload["text"]) for part in parts[1:] if part.payload.get("field") == field
    )
    assert recovered == (text or "")
    assert all(len(str(part.payload.get("text", ""))) <= 8192 for part in parts)
    store = BlobStore(tmp_path / "blob")
    captured = snapshot_sqlite_to_blob(path, store)
    assert list(iter_codex_state_parts(store.blob_path(captured.blob_hash), state_kind=kind, text_chars=8192)) == parts


@pytest.mark.parametrize("generated", [False, True])
def test_codex_text_reader_failed_child_close_keeps_original_native_owner(
    tmp_path: Path,
    generated: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "goals.sqlite"
    _state(path, "goals", "λ" * 40000, generated=generated)
    metadata = path.stat()
    identity = (metadata.st_dev, metadata.st_ino)
    observe_descriptors = Path("/proc/self/fd").is_dir()
    allow_cleanup = False
    selected: list[sqlite3.Connection] = []
    blocked: list[ControlledCursor] = []
    original_connect = connect_measured
    original_blob_close = NativeSQLCustodyOwner.close_incremental_blob
    original_require = NativeSQLCustodyOwner.require_connection
    completed: list[bool] = []
    registered: list[NativeSQLCustodyOwner] = []
    primary = OSError("synthetic Codex child close failure")

    class FaultCursor(ControlledCursor):
        def execute(self, sql: str, parameters: Any = ()) -> FaultCursor:
            result = super().execute(sql, parameters)
            if generated and sql.startswith("SELECT substr(CAST(objective AS BLOB)"):
                self.allow_cleanup.clear()
                blocked.append(self)
            return result

    def capture(database: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(database, *args, **kwargs)
        selected.append(connection)
        original_cursor = connection.cursor
        monkeypatch.setattr(connection, "cursor", lambda: original_cursor(factory=FaultCursor))
        return connection

    def close_blob(owner: NativeSQLCustodyOwner, blob: sqlite3.Blob) -> None:
        if owner.connection in selected and not allow_cleanup:
            raise primary
        original_blob_close(owner, blob)

    def require(owner: NativeSQLCustodyOwner) -> sqlite3.Connection:
        connection = original_require(owner)
        if connection in selected and owner not in registered:
            owner.retain_settlement_callback(lambda: completed.append(True))
            registered.append(owner)
        return connection

    monkeypatch.setattr(sqlite_export, "connect_measured", capture)
    monkeypatch.setattr(NativeSQLCustodyOwner, "require_connection", require)
    if not generated:
        monkeypatch.setattr(NativeSQLCustodyOwner, "close_incremental_blob", close_blob)
    try:
        with pytest.raises((NativeConnectionSettlementError, BaseExceptionGroup)):
            list(iter_codex_state_parts(path, state_kind="goals", text_chars=8192))
        owners = tuple(
            owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection in selected
        )
        assert len(owners) == 1
        owner = owners[0]
        assert registered == [owner]
        assert owner.connection is not None and owner.close_required
        if observe_descriptors:
            assert selected_file_descriptors(identity)
        if generated:
            assert isinstance(owner.connection, _MeasuredConnection)
            assert blocked and all(id(cursor) in owner.connection._unsettled_native_cursors for cursor in blocked)
        else:
            assert owner._incremental_blobs
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert completed == []
        allow_cleanup = True
        for cursor in blocked:
            cursor.allow_cleanup.set()
        owner.close()
        assert owner.connection is None and completed == [True]
        if observe_descriptors:
            assert selected_file_descriptors(identity) == ()
    finally:
        allow_cleanup = True
        for cursor in blocked:
            cursor.allow_cleanup.set()
        for retained in retained_native_sql_owners_on_current_thread():
            if retained.connection in selected:
                retained.close()


@pytest.mark.parametrize("kind", ["goals", "memories"])
@pytest.mark.parametrize("storage", ["VIRTUAL", "STORED"])
@pytest.mark.parametrize("text", [None, "", "λ\x00中" * 40000])
def test_generated_required_codex_field_is_read_directly(
    tmp_path: Path, kind: Literal["goals", "memories"], storage: str, text: str | None
) -> None:
    path = tmp_path / "generated.sqlite"
    field = "objective" if kind == "goals" else "raw_memory"
    table = "thread_goals" if kind == "goals" else "stage1_outputs"
    extra = (
        ", goal_id TEXT,status TEXT"
        if kind == "goals"
        else ",rollout_summary TEXT,source_updated_at INTEGER DEFAULT 0,generated_at INTEGER DEFAULT 0,usage_count INTEGER,selected_for_phase2 INTEGER DEFAULT 0"
    )
    with sqlite_connection(path) as connection:
        connection.execute(
            f"CREATE TABLE {table}(thread_id TEXT, input TEXT, {field} TEXT "
            f"GENERATED ALWAYS AS (input) {storage}{extra})"
        ).close()
        if kind == "goals":
            connection.execute(
                "INSERT INTO thread_goals(thread_id,input,goal_id,status) VALUES('thread',?,'goal','active')", (text,)
            ).close()
        else:
            connection.execute(
                "INSERT INTO stage1_outputs(thread_id,input,rollout_summary) VALUES('thread',?,'')", (text,)
            ).close()
        connection.commit()
    parts = list(iter_codex_state_parts(path, state_kind=kind, text_chars=8192))
    recovered = str(parts[0].payload[field]) + "".join(
        str(part.payload["text"]) for part in parts[1:] if part.payload.get("field") == field
    )
    assert recovered == (text or "")
    assert all(len(str(part.payload.get("text", ""))) <= 8192 for part in parts)


@pytest.mark.parametrize("generated", [False, True])
def test_codex_early_exit_settles_the_original_reader(
    tmp_path: Path, generated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "early.sqlite"
    _state(path, "goals", "λ" * 40000, generated=generated)
    selected: list[sqlite3.Connection] = []
    original_connect = connect_measured

    def capture(database: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(database, *args, **kwargs)
        selected.append(connection)
        return connection

    monkeypatch.setattr(sqlite_export, "connect_measured", capture)
    reader = iter_codex_state_parts(path, state_kind="goals", text_chars=8192)
    try:
        assert next(reader).payload["objective"] == "λ" * 8192
    finally:
        reader.close()
    assert selected
    assert not any(owner.connection in selected for owner in retained_native_sql_owners_on_current_thread())
    if Path("/proc/self/fd").is_dir():
        metadata = path.stat()
        assert selected_file_descriptors((metadata.st_dev, metadata.st_ino)) == ()
    for connection in selected:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("generated", [False, True])
def test_codex_cancellation_settles_original_native_children(
    tmp_path: Path, generated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel

    path = tmp_path / "cancel.sqlite"
    _state(path, "goals", "λ" * 100000, generated=generated)
    selected: list[sqlite3.Connection] = []
    original_connect = connect_measured

    def capture(database: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(database, *args, **kwargs)
        selected.append(connection)
        return connection

    monkeypatch.setattr(sqlite_export, "connect_measured", capture)
    cancelled = threading.Event()
    token = compute_cancel.set(cancelled)
    reader = iter_codex_state_parts(path, state_kind="goals", text_chars=8192)
    try:
        assert next(reader).part_kind == "record"
        cancelled.set()
        with pytest.raises(DaemonOperationCancelled):
            list(reader)
    finally:
        reader.close()
        compute_cancel.reset(token)
    assert selected
    assert not any(owner.connection in selected for owner in retained_native_sql_owners_on_current_thread())
