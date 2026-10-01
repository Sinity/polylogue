from __future__ import annotations

import gc
import sqlite3
import sys
import weakref
from pathlib import Path
from typing import cast

import pytest

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import (
    connect_measured,
    io_phase_process_snapshot,
    io_phase_snapshot,
    live_connection_cursors,
)
from polylogue.storage.sqlite.connection_profile import (
    open_isolated_write_connection,
    open_readonly_connection,
    open_source_tier_write_connection,
)
from polylogue.storage.sqlite.wal_checkpoint import checkpoint_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.native_sql_descriptor_probe import selected_file_descriptors
from tests.infra.sqlite_cursor_settlement import (
    BeforeNativeInitCursor,
    ConstructorStatementCursor,
    ControlledCursor,
    InvalidReturnCursor,
    UnhashableCursor,
)


def _count(tier: str, phase: str, inside: bool) -> int:
    return sum(
        sample.count
        for sample in io_phase_snapshot()
        if sample.tier == tier and sample.phase == phase and sample.inside_writer_lease is inside and sample.succeeded
    )


@pytest.mark.skipif(sys.platform != "linux", reason="physical descriptor observation uses Linux procfs")
@pytest.mark.parametrize("route", ["cursor", "execute", "executemany", "executescript"])
def test_measured_connection_settles_every_cursor_route_before_native_close(tmp_path: Path, route: str) -> None:
    path = tmp_path / "cursor.db"
    connection = connect_measured(path)
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.executemany("INSERT INTO evidence VALUES (?)", ((1,), (2,), (3,)))
    connection.commit()
    if route == "cursor":
        cursor = connection.cursor().execute("SELECT value FROM evidence")
    elif route == "execute":
        cursor = connection.execute("SELECT value FROM evidence")
    elif route == "executemany":
        cursor = connection.executemany("INSERT INTO evidence VALUES (?)", ((4,), (5,)))
    else:
        cursor = connection.executescript("INSERT INTO evidence VALUES (6); SELECT value FROM evidence;")
    if route in {"cursor", "execute"}:
        assert next(cursor) == (1,)
    metadata = path.stat()
    identity = metadata.st_dev, metadata.st_ino
    assert selected_file_descriptors(identity)
    connection.close()
    assert selected_file_descriptors(identity) == ()
    # The cursor remains reachable here: GC did not supply physical cleanup.
    with pytest.raises(sqlite3.ProgrammingError):
        cursor.fetchone()
    connection.close()


def test_custom_unhashable_equal_cursors_keep_distinct_physical_ownership(tmp_path: Path) -> None:
    connection = connect_measured(tmp_path / "custom.db")
    first = connection.cursor(factory=UnhashableCursor)
    second = connection.cursor(factory=UnhashableCursor)
    assert first is not second and first == second
    assert isinstance(first, UnhashableCursor)
    with pytest.raises(TypeError):
        hash(first)
    first.execute("SELECT 1")
    second.execute("SELECT 2")
    connection.close()
    assert first.close_attempts == second.close_attempts == 1
    connection.close()
    assert first.close_attempts == second.close_attempts == 1


def test_discarded_completed_cursors_are_not_pinned_by_the_connection(tmp_path: Path) -> None:
    connection = connect_measured(tmp_path / "completed.db")
    reference = weakref.ref(connection.execute("SELECT 1"))
    gc.collect()
    assert reference() is None
    cursor = connection.cursor(factory=ControlledCursor)
    assert cursor.execute("SELECT 1").fetchall() == [(1,)]
    cursor.close()
    reference = weakref.ref(cursor)
    del cursor
    gc.collect()
    assert reference() is None
    connection.close()


@pytest.mark.skipif(sys.platform != "linux", reason="physical descriptor observation uses Linux procfs")
def test_execute_error_traceback_does_not_keep_native_descriptor_after_connection_close(tmp_path: Path) -> None:
    path = tmp_path / "error.db"
    connection = connect_measured(path)
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.executemany("INSERT INTO evidence VALUES (?)", ((1,), (2,)))
    connection.commit()
    metadata = path.stat()
    identity = metadata.st_dev, metadata.st_ino

    def refuse(value: int) -> int:
        raise ValueError("synthetic source scalar failure")

    connection.create_function("refuse", 1, refuse)
    with pytest.raises(sqlite3.OperationalError) as failure:
        connection.execute("SELECT refuse(value) FROM evidence")
    assert failure.value.__traceback__ is not None
    assert selected_file_descriptors(identity)
    connection.close()
    assert selected_file_descriptors(identity) == ()
    assert failure.value.__traceback__ is not None
    connection.close()


def test_returned_source_handle_records_actual_transaction_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    before = {phase: _count("source", phase, True) for phase in ("connection_create", "begin", "commit", "rollback")}
    with write_lease("io-phase-test", archive_root=tmp_path):
        conn = open_source_tier_write_connection(path, archive_root=tmp_path)
        try:
            conn.execute("CREATE TABLE sample (value INTEGER)")
            conn.execute("INSERT INTO sample VALUES (1)")
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO sample VALUES (2)")
            conn.commit()
            conn.execute("BEGIN")
            conn.execute("INSERT INTO sample VALUES (3)")
            conn.rollback()
            assert conn.execute("SELECT value FROM sample ORDER BY value").fetchall() == [(1,), (2,)]
        finally:
            conn.close()
    assert _count("source", "connection_create", True) - before["connection_create"] == 1
    assert _count("source", "begin", True) - before["begin"] == 2
    assert _count("source", "commit", True) - before["commit"] >= 2
    assert _count("source", "rollback", True) - before["rollback"] == 1
    outside_before = _count("source", "connection_create", False)
    reader = open_readonly_connection(path, validate_schema=False)
    reader.close()
    assert _count("source", "connection_create", False) - outside_before == 1
    payload = io_phase_process_snapshot()
    assert payload["scope"] == "process"
    assert isinstance(payload["pid"], int)
    samples = cast(list[dict[str, object]], payload["samples"])
    assert any(sample["tier"] == "source" and sample["phase"] == "connection_create" for sample in samples)


def test_context_manager_commit_is_counted_once(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    before = _count("source", "commit", True)
    with write_lease("io-context-test", archive_root=tmp_path):
        conn = open_source_tier_write_connection(path, archive_root=tmp_path)
        try:
            with conn:
                conn.execute("CREATE TABLE sample (value INTEGER)")
                conn.execute("INSERT INTO sample VALUES (1)")
        finally:
            conn.close()
    assert _count("source", "commit", True) - before == 1


def test_checkpoint_and_blob_syncs_are_counted_at_the_syscalls(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    before_checkpoint = _count("index", "checkpoint", True)
    with write_lease("io-checkpoint-test", archive_root=tmp_path):
        conn = open_isolated_write_connection(db, purpose="io phase checkpoint test", archive_root=tmp_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE sample (value INTEGER)")
            conn.commit()
            checkpoint_connection(conn, "PASSIVE", boundary="recurring")
        finally:
            conn.close()
    assert _count("index", "checkpoint", True) - before_checkpoint == 1

    before_file = _count("source", "blob_file_fsync", False)
    before_directory = _count("source", "blob_directory_fsync", False)
    BlobStore(tmp_path / "blobs").write_from_bytes(b"synthetic blob")
    assert _count("source", "blob_file_fsync", False) - before_file == 1
    assert _count("source", "blob_directory_fsync", False) - before_directory >= 1


@pytest.mark.skipif(sys.platform != "linux", reason="physical descriptor observation uses Linux procfs")
@pytest.mark.parametrize("factory", [ConstructorStatementCursor, BeforeNativeInitCursor, InvalidReturnCursor])
def test_plain_cursor_constructor_failure_keeps_exact_native_statement_in_creator_census(
    tmp_path: Path, factory: type[sqlite3.Cursor]
) -> None:
    path = tmp_path / "constructor.db"
    connection = connect_measured(path)
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.commit()
    metadata = path.stat()
    identity = metadata.st_dev, metadata.st_ino
    with pytest.raises((ValueError, TypeError)) as failure:
        connection.cursor(factory=factory)
    assert failure.value.__traceback__ is not None
    if factory is BeforeNativeInitCursor:
        assert live_connection_cursors(connection) == ()
    else:
        cursors = live_connection_cursors(connection)
        assert len(cursors) == 1 and type(cursors[0]) is factory
    connection.close()
    assert selected_file_descriptors(identity) == ()
    assert failure.value.__traceback__ is not None
    connection.close()


def test_raising_arbitrary_factory_without_a_cursor_preserves_invocation_without_fake_obligation(
    tmp_path: Path,
) -> None:
    connection = connect_measured(tmp_path / "callback.db")
    calls: list[sqlite3.Connection] = []

    def factory(owner: sqlite3.Connection) -> sqlite3.Cursor:
        calls.append(owner)
        raise ValueError("synthetic callback refused before constructing a cursor")

    with pytest.raises(ValueError):
        connection.cursor(factory=factory)
    assert calls == [connection]
    assert live_connection_cursors(connection) == ()
    connection.close()


@pytest.mark.parametrize("closed", [False, True])
def test_native_cursor_admission_precedes_plain_factory_invocation(tmp_path: Path, closed: bool) -> None:
    import threading

    calls: list[sqlite3.Connection] = []
    failures: list[BaseException] = []

    class ObservedCursor(sqlite3.Cursor):
        def __init__(self, connection: sqlite3.Connection) -> None:
            calls.append(connection)
            super().__init__(connection)

    connection = connect_measured(tmp_path / "admission.db")
    if closed:
        connection.close()

    def create() -> None:
        try:
            connection.cursor(factory=ObservedCursor)
        except BaseException as error:
            failures.append(error)

    if closed:
        create()
    else:
        foreign = threading.Thread(target=create)
        foreign.start()
        foreign.join()
    assert len(failures) == 1 and isinstance(failures[0], sqlite3.ProgrammingError)
    assert calls == []
    assert live_connection_cursors(connection) == ()
    connection.close()
