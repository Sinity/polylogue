"""Contract tests for the reusable durability fault fixture."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.infra.durability_faults import (
    DurabilityFaultPoint,
    DurabilityFaultRegistry,
    InjectedCrash,
)


def test_registry_reaches_direct_and_pathlib_filesystem_boundaries(
    tmp_path: Path, durability_faults: DurabilityFaultRegistry
) -> None:
    """The fixture observes the calls users actually make, including Path.unlink."""
    target = tmp_path / "payload"
    target.write_bytes(b"payload")

    durability_faults.arm(DurabilityFaultPoint.UNLINK, action="kill")
    with pytest.raises(InjectedCrash):
        with durability_faults.installed():
            target.unlink()

    assert durability_faults.count(DurabilityFaultPoint.UNLINK) == 1
    assert target.exists()


def test_registry_wraps_sqlite_commit_without_replacing_connection_semantics(
    tmp_path: Path, durability_faults: DurabilityFaultRegistry
) -> None:
    """A commit fault is injected at the connection method, not SQL text."""
    database = tmp_path / "tier.db"
    durability_faults.arm(DurabilityFaultPoint.COMMIT)
    with pytest.raises(RuntimeError, match="injected commit fault"):
        with durability_faults.installed():
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE item (value TEXT)")
            connection.commit()

    assert durability_faults.count(DurabilityFaultPoint.COMMIT) == 1


def test_run_requires_reaching_the_selected_point_and_runs_recovery(durability_faults: DurabilityFaultRegistry) -> None:
    recovered: list[str] = []
    checked: list[str] = []

    result = durability_faults.run(
        DurabilityFaultPoint.FSYNC,
        lambda: _fsync_once(),
        recover=lambda: recovered.append("recovered"),
        assert_invariants=lambda: checked.append("sound"),
    )

    assert result.interrupted
    assert recovered == ["recovered"]
    assert checked == ["sound"]
    assert result.events == (result.events[0],)


def test_run_rejects_a_vacuous_operation(durability_faults: DurabilityFaultRegistry) -> None:
    with pytest.raises(AssertionError, match="was not reached"):
        durability_faults.run(
            DurabilityFaultPoint.REPLACE,
            lambda: None,
            recover=lambda: None,
            assert_invariants=lambda: None,
        )


def _fsync_once() -> None:
    import os

    descriptor = os.open("/dev/null", os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("action", ["raise", "kill"])
@pytest.mark.parametrize("context_commit", [False, True])
def test_recovery_starts_after_interrupted_connections_are_closed(
    tmp_path: Path,
    action: str,
    context_commit: bool,
) -> None:
    """Leaving the captured transaction alive locks out recovery's first write."""
    database = tmp_path / "interrupted.db"
    seed = sqlite3.connect(database)
    seed.execute("CREATE TABLE records(value INTEGER)")
    seed.commit()
    seed.close()
    retained: list[sqlite3.Connection] = []
    registry = DurabilityFaultRegistry()

    def operation() -> None:
        conn = sqlite3.connect(database)
        retained.append(conn)
        if context_commit:
            with conn:
                conn.execute("INSERT INTO records VALUES (1)")
        else:
            conn.execute("INSERT INTO records VALUES (1)")
            conn.commit()

    def recovery() -> None:
        assert len(retained) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            retained[0].execute("SELECT 1")
        conn = sqlite3.connect(database, timeout=0)
        try:
            conn.execute("INSERT INTO records VALUES (2)")
            conn.commit()
        finally:
            conn.close()

    def invariants() -> None:
        conn = sqlite3.connect(database)
        try:
            assert conn.execute("SELECT value FROM records").fetchall() == [(2,)]
        finally:
            conn.close()

    result = registry.run(
        DurabilityFaultPoint.COMMIT,
        operation,
        recover=recovery,
        assert_invariants=invariants,
        action=action,
    )
    assert result.interrupted
    assert registry.count(DurabilityFaultPoint.COMMIT) == 1


@pytest.mark.parametrize("custom_factory", [False, True])
@pytest.mark.parametrize("positional_factory", [False, True])
def test_fault_connections_remain_native_backup_endpoints(custom_factory: bool, positional_factory: bool) -> None:
    """Native backup targets and row_factory assignment must work before the selected commit fault."""
    from tests.infra.durability_faults import InjectedCrash

    class CountingConnection(sqlite3.Connection):
        reads = 0

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            self.reads += 1
            return super().execute(sql, *args, **kwargs)

    native_source = sqlite3.connect(":memory:")
    native_target = sqlite3.connect(":memory:")
    native_source.execute("CREATE TABLE records(value INTEGER)")
    native_source.execute("INSERT INTO records VALUES (7)")
    native_source.commit()
    registry = DurabilityFaultRegistry()
    try:
        with pytest.raises(InjectedCrash):
            with registry.installed():
                factory = CountingConnection if custom_factory else sqlite3.Connection
                if positional_factory:
                    connection = sqlite3.connect(":memory:", 5.0, 0, "DEFERRED", True, factory)
                else:
                    connection = sqlite3.connect(":memory:", factory=factory)
                assert isinstance(connection, sqlite3.Connection)
                if custom_factory:
                    assert isinstance(connection, CountingConnection)
                native_source.backup(connection)
                connection.row_factory = sqlite3.Row
                assert connection.execute("SELECT value FROM records").fetchone()["value"] == 7
                if custom_factory:
                    assert connection.reads == 1
                connection.backup(native_target)
                assert native_target.execute("SELECT value FROM records").fetchall() == [(7,)]
                registry.arm(DurabilityFaultPoint.COMMIT, action="kill")
                connection.execute("INSERT INTO records VALUES (8)")
                connection.commit()
        assert registry.count(DurabilityFaultPoint.COMMIT) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    finally:
        native_source.close()
        native_target.close()


def test_statement_crash_uses_the_same_native_connection_and_lifetime() -> None:
    from tests.infra.durability_faults import InjectedCrash, crash_after_mutating_statements

    native = sqlite3.connect(":memory:")
    native.execute("CREATE TABLE records(value INTEGER)")
    native.commit()
    try:
        with pytest.raises(InjectedCrash):
            with crash_after_mutating_statements(2) as budget:
                conn = sqlite3.connect(":memory:")
                native.backup(conn)
                conn.row_factory = sqlite3.Row
                conn.execute("INSERT INTO records VALUES (1)")
                conn.execute("INSERT INTO records VALUES (2)")
        assert budget == [2, 2]
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    finally:
        native.close()


@pytest.mark.parametrize("statement_instrumentation", [False, True])
def test_successful_instrumentation_preserves_caller_connection_ownership(statement_instrumentation: bool) -> None:
    from tests.infra.durability_faults import crash_after_mutating_statements

    registry = DurabilityFaultRegistry()
    instrumentation = crash_after_mutating_statements(1) if statement_instrumentation else registry.installed()
    with instrumentation:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE records(value INTEGER)")
    try:
        registry.arm(DurabilityFaultPoint.COMMIT, action="kill")
        conn.execute("INSERT INTO records VALUES (1)")
        conn.commit()
        assert conn.execute("SELECT value FROM records").fetchall() == [(1,)]
        assert registry.count(DurabilityFaultPoint.COMMIT) == 0
    finally:
        conn.close()
