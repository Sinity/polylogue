from __future__ import annotations

import gc
import sqlite3
import weakref
from builtins import BaseExceptionGroup
from pathlib import Path
from typing import cast

import pytest

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import (
    close_connection_cursor,
    connect_measured,
    connection_cursor,
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
    if Path("/proc/self/fd").is_dir():
        assert selected_file_descriptors(identity)
    connection.close()
    if Path("/proc/self/fd").is_dir():
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
    if Path("/proc/self/fd").is_dir():
        assert selected_file_descriptors(identity)
    connection.close()
    if Path("/proc/self/fd").is_dir():
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
    if Path("/proc/self/fd").is_dir():
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


def test_measured_connection_retains_failed_cursor_without_native_owner(tmp_path: Path) -> None:
    import gc
    import weakref

    from polylogue.storage.io_phase_metrics import connect_measured, live_connection_cursors
    from tests.infra.native_sql_descriptor_probe import selected_file_descriptors

    path = tmp_path / "primitive-control.db"
    connection = connect_measured(path)
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.executemany("INSERT INTO evidence VALUES (?)", [(1,), (2,)])
    connection.commit()
    metadata = path.stat()
    identity = metadata.st_dev, metadata.st_ino
    observe_descriptors = Path("/proc/self/fd").is_dir()
    cursor = connection.cursor(factory=ControlledCursor)
    assert isinstance(cursor, ControlledCursor)
    cursor.execute("SELECT value FROM evidence ORDER BY value")
    cursor.fetchone()
    cursor.allow_cleanup.clear()
    actual = weakref.ref(cursor)

    def discard_close_error() -> None:
        try:
            connection.close()
        except BaseExceptionGroup:
            pass

    try:
        discard_close_error()
        del cursor
        gc.collect()
        retained = actual()
        assert retained is not None and retained.close_attempts == 1
        assert live_connection_cursors(connection) == (retained,)
        if observe_descriptors:
            assert selected_file_descriptors(identity)
        retained.allow_cleanup.set()
        connection.close()
        assert retained.close_attempts == 2
        if observe_descriptors:
            assert not selected_file_descriptors(identity)
        del retained
        gc.collect()
        assert actual() is None
    finally:
        if retained_cursor := actual():
            retained_cursor.allow_cleanup.set()
        connection.close()


@pytest.mark.parametrize("execute_fails", [False, True])
def test_statement_context_retains_failed_native_close_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute_fails: bool
) -> None:
    from polylogue.storage.io_phase_metrics import _MeasuredConnection

    path = tmp_path / "statement-close.db"
    connection = connect_measured(path)
    cursor_factory = _MeasuredConnection.cursor
    selected: list[ControlledCursor] = []
    close_failure = OSError("synthetic actual statement close failure")

    def blocked_cursor(original: _MeasuredConnection) -> sqlite3.Cursor:
        cursor = cursor_factory(original, factory=ControlledCursor)
        assert isinstance(cursor, ControlledCursor)
        cursor.cleanup_failure = close_failure
        cursor.allow_cleanup.clear()
        selected.append(cursor)
        return cursor

    monkeypatch.setattr(_MeasuredConnection, "cursor", blocked_cursor)
    try:
        with pytest.raises(BaseExceptionGroup if execute_fails else OSError) as failure:
            with connection_cursor(
                connection, "SELECT absent_column" if execute_fails else "SELECT 1 UNION ALL SELECT 2"
            ) as cursor:
                assert next(cursor) == (1,)
        if execute_fails:
            assert isinstance(failure.value, BaseExceptionGroup)
            assert isinstance(failure.value.exceptions[0], sqlite3.OperationalError)
            assert failure.value.exceptions[1] is close_failure
        else:
            assert failure.value is close_failure
        retained = weakref.ref(selected.pop())
        if not execute_fails:
            del cursor
        failure.value.__traceback__ = None
        if isinstance(failure.value, BaseExceptionGroup):
            for component in failure.value.exceptions:
                component.__traceback__ = None
        # ExceptionInfo stores its own original traceback independently of
        # exception.__traceback__. Drop that last test-owned frame carrier.
        del failure
        gc.collect()
        assert retained() is not None
        assert retained() in live_connection_cursors(connection)
        actual = retained()
        assert isinstance(actual, ControlledCursor)
        actual.allow_cleanup.set()
        connection.close()

        assert live_connection_cursors(connection) == ()
    finally:
        for child in live_connection_cursors(connection):
            if isinstance(child, ControlledCursor):
                child.allow_cleanup.set()
        connection.close()


@pytest.mark.parametrize("probe", ["table", "relation", "view", "trigger", "index", "column", "attached"])
def test_canonical_sync_introspection_settles_its_actual_native_cursors(probe: str) -> None:
    from polylogue.core import sqlite_introspection

    connection = connect_measured(":memory:")
    try:
        with connection_cursor(connection, "CREATE TABLE evidence(value INTEGER)"):
            pass
        with connection_cursor(connection, "CREATE INDEX evidence_value ON evidence(value)"):
            pass
        with connection_cursor(connection, "CREATE VIEW evidence_view AS SELECT value FROM evidence"):
            pass
        with connection_cursor(
            connection, "CREATE TRIGGER evidence_trigger AFTER INSERT ON evidence BEGIN SELECT 1; END"
        ):
            pass
        with connection_cursor(connection, "ATTACH ':memory:' AS input"):
            pass
        with connection_cursor(connection, "CREATE TABLE input.evidence(value INTEGER)"):
            pass
        if probe == "table":
            result = sqlite_introspection.table_exists(connection, "evidence")
        elif probe == "relation":
            result = sqlite_introspection.relation_exists(connection, "evidence_view")
        elif probe == "view":
            result = sqlite_introspection.view_exists(connection, "evidence_view")
        elif probe == "trigger":
            result = sqlite_introspection.trigger_exists(connection, "evidence_trigger")
        elif probe == "index":
            result = sqlite_introspection.index_exists(connection, "evidence_value")
        elif probe == "column":
            result = sqlite_introspection.column_exists(connection, "evidence", "value")
        else:
            result = sqlite_introspection.table_exists(connection, "evidence", schema="input")
        assert result
        assert live_connection_cursors(connection) == ()
    finally:
        connection.close()


def test_standalone_core_introspection_works_in_a_fresh_native_interpreter() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sqlite3; from polylogue.core.sqlite_introspection import table_exists; "
            "c=sqlite3.connect(':memory:'); c.execute('CREATE TABLE evidence(value INTEGER)').close(); "
            "assert table_exists(c, 'evidence'); c.close(); print('plain introspection settled')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "plain introspection settled"


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_statement_context_physically_settles_declared_connection_types(measured: bool, fails: bool) -> None:
    connection = connect_measured(":memory:") if measured else sqlite3.connect(":memory:")
    original = OSError("neutral statement consumer failure")
    try:
        if fails:
            with pytest.raises(OSError) as failure:
                with connection_cursor(connection, "SELECT 7") as cursor:
                    assert cursor.fetchone() == (7,)
                    raise original
            assert failure.value is original
        else:
            with connection_cursor(connection, "SELECT 7") as cursor:
                assert cursor.fetchone() == (7,)
        with pytest.raises(sqlite3.ProgrammingError):
            cursor.fetchone()
        if measured:
            assert live_connection_cursors(connection) == ()
    finally:
        connection.close()


@pytest.mark.parametrize("measured", [False, True])
def test_canonical_cursor_close_supports_original_raw_and_measured_suppliers(measured: bool) -> None:
    connection = connect_measured(":memory:") if measured else sqlite3.connect(":memory:")
    cursor = connection.cursor(factory=ControlledCursor)
    assert isinstance(cursor, ControlledCursor)
    cursor.execute("SELECT 1 UNION ALL SELECT 2")
    cursor.allow_cleanup.clear()
    try:
        with pytest.raises(OSError) as caught:
            close_connection_cursor(connection, cursor)
        assert caught.value is cursor.cleanup_failure
        if measured:
            assert cursor in live_connection_cursors(connection)
        cursor.allow_cleanup.set()
        close_connection_cursor(connection, cursor)
        with pytest.raises(sqlite3.ProgrammingError):
            cursor.fetchone()
    finally:
        cursor.allow_cleanup.set()
        close_connection_cursor(connection, cursor)
        connection.close()
