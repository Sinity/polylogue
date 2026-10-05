"""Backup preflight refuses original read faults before retaining artifacts."""

from __future__ import annotations

import os
import sqlite3
from builtins import BaseExceptionGroup, ExceptionGroup
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.write_lease import write_lease
from polylogue.operations.archive_backup import backup_archive
from polylogue.storage import backup_package as backup
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.io_phase_metrics import _MeasuredConnection
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    open_readonly_connection,
    retained_native_sql_owners_on_current_thread,
)
from tests.infra.durable_tier_fixtures import bootstrap_baseline_archive
from tests.infra.sqlite_cursor_settlement import ControlledCursor

_QUERY = "SELECT 1 FROM sqlite_master LIMIT 1"


def _run(root: Path, destination: Path, *, check_only: bool) -> backup.BackupResult:
    if check_only:
        return backup_archive(output_dir=destination, archive_root_path=root, check_only=True)
    with write_lease("synthetic-backup-preflight", archive_root=root):
        with OwnedArchiveLocation.acquire(ArchiveLocation.resolve(root)) as owner:
            return backup_archive(output_dir=destination, archive_root_path=root, archive_owner=owner)


def _contains(error: BaseException, expected: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is expected:
            return True
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested in (current.__cause__, current.__context__, getattr(current, "failure", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


@pytest.mark.parametrize("check_only", [True, False])
@pytest.mark.parametrize("failure_type", [sqlite3.OperationalError, PermissionError, DaemonOperationCancelled])
def test_public_backup_preflight_preserves_original_read_failure_and_settles_native_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool, failure_type: type[Exception]
) -> None:
    root = tmp_path / "archive"
    bootstrap_baseline_archive(root, monkeypatch)
    destination = tmp_path / "backups"
    before = {path.name: path.read_bytes() for path in root.glob("*.db")}
    original = open_readonly_connection
    failure = failure_type("synthetic original backup read fault")
    selected: list[sqlite3.Connection] = []
    reached: list[ControlledCursor] = []

    class FaultCursor(ControlledCursor):
        def execute(self, sql: str, parameters: Any = ()) -> FaultCursor:
            if sql == _QUERY:
                reached.append(self)
                raise failure
            return super().execute(sql, parameters)

    def capture(path: Path, **kwargs: Any) -> sqlite3.Connection:
        connection = original(path, **kwargs)
        if path.name == "source.db":
            selected.append(connection)
            cursor = connection.cursor
            monkeypatch.setattr(connection, "cursor", lambda: cursor(factory=FaultCursor))
        return connection

    monkeypatch.setattr(backup, "open_readonly_connection", capture)
    with pytest.raises(failure_type) as refused:
        _run(root, destination, check_only=check_only)
    assert refused.value is failure and reached and selected
    assert not any(owner.connection in selected for owner in retained_native_sql_owners_on_current_thread())
    for connection in selected:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    assert not destination.exists() or list(destination.iterdir()) == []
    assert {path.name: path.read_bytes() for path in root.glob("*.db")} == before


@pytest.mark.parametrize("check_only", [True, False])
@pytest.mark.parametrize("fault", ["cursor", "connection", "version-and-connection"])
def test_public_backup_preflight_retains_failed_native_close_until_exact_creator_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool, fault: str
) -> None:
    root = tmp_path / "archive"
    bootstrap_baseline_archive(root, monkeypatch)
    destination = tmp_path / "backups"
    original = open_readonly_connection
    original_close = _MeasuredConnection.close
    selected: list[sqlite3.Connection] = []
    blocked: list[ControlledCursor] = []
    close_attempts: list[sqlite3.Connection] = []
    allow_close = False
    primary = sqlite3.OperationalError("synthetic original version probe fault")
    cleanup = sqlite3.OperationalError("synthetic original physical close fault")

    class FaultCursor(ControlledCursor):
        def execute(self, sql: str, parameters: Any = ()) -> FaultCursor:
            if fault == "version-and-connection" and sql == "PRAGMA user_version":
                raise primary
            result = super().execute(sql, parameters)
            if fault == "cursor" and sql == _QUERY:
                blocked.append(self)
                self.allow_cleanup.clear()
            return result

    def capture(path: Path, **kwargs: Any) -> sqlite3.Connection:
        connection = original(path, **kwargs)
        if path.name == "source.db":
            selected.append(connection)
            cursor = connection.cursor
            monkeypatch.setattr(connection, "cursor", lambda: cursor(factory=FaultCursor))
        return connection

    def close(connection: _MeasuredConnection) -> None:
        if connection in selected and fault != "cursor" and not allow_close:
            close_attempts.append(connection)
            raise cleanup
        original_close(connection)

    monkeypatch.setattr(backup, "open_readonly_connection", capture)
    monkeypatch.setattr(_MeasuredConnection, "close", close)
    try:
        with pytest.raises((NativeConnectionSettlementError, BaseExceptionGroup)) as refused:
            _run(root, destination, check_only=check_only)
        owners = tuple(
            owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection in selected
        )
        assert len(owners) == 1
        owner = owners[0]
        assert owner.connection is selected[0] and owner.close_required
        if fault == "cursor":
            assert isinstance(owner.connection, _MeasuredConnection)
            assert blocked and all(id(cursor) in owner.connection._unsettled_native_cursors for cursor in blocked)
        else:
            assert close_attempts == selected
            assert _contains(refused.value, cleanup)
            if fault == "version-and-connection":
                assert _contains(refused.value, primary)
        assert not destination.exists() or list(destination.iterdir()) == []
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        allow_close = True
        for cursor in blocked:
            cursor.allow_cleanup.set()
        owner.close()
        assert owner.connection is None
    finally:
        allow_close = True
        for cursor in blocked:
            cursor.allow_cleanup.set()
        for owner in retained_native_sql_owners_on_current_thread():
            if owner.connection in selected:
                owner.close()


@pytest.mark.parametrize("check_only", [True, False])
def test_public_backup_preserves_missing_tier_diagnostics_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool
) -> None:
    root = tmp_path / "archive"
    bootstrap_baseline_archive(root, monkeypatch)
    (root / "user.db").unlink()
    destination = tmp_path / "backups"
    result = _run(root, destination, check_only=check_only)
    assert result.ok is False and result.error is not None and result.warnings
    assert result.output_path is None
    assert not destination.exists() or list(destination.iterdir()) == []


@pytest.mark.parametrize("check_only", [True, False])
def test_public_backup_preserves_separate_disk_probe_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool
) -> None:
    root = tmp_path / "archive"
    bootstrap_baseline_archive(root, monkeypatch)
    failure = OSError("synthetic disk observation unavailable")

    def unavailable(path: str) -> None:
        raise failure

    monkeypatch.setattr(os, "statvfs", unavailable)
    result = _run(root, tmp_path / "backups", check_only=check_only)
    assert len(result.warnings) == 1 and str(failure) in result.warnings[0]
    assert result.ok is (not check_only)


@pytest.mark.uses_real_clock("starts the real daemon coordinator and public operation client")
@pytest.mark.parametrize("check_only", [True, False])
@pytest.mark.parametrize("error_code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_ERROR, None])
def test_daemon_backup_preflight_classifies_only_named_io_faults_before_write_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool, error_code: int | None
) -> None:
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from tests.infra.daemon_operations import running_daemon_operations

    failure = (
        PermissionError("synthetic original backup preflight permission refusal")
        if error_code is None
        else sqlite3.OperationalError("synthetic typed backup preflight refusal")
    )
    if isinstance(failure, sqlite3.OperationalError):
        assert error_code is not None
        failure.sqlite_errorcode = error_code
    observed: list[BaseException] = []
    begins: list[object] = []
    original_begin = DaemonOperationRuntime.begin_unbound_write

    def refuse(path: Path) -> None:
        observed.append(failure)
        raise failure

    def begin(self: DaemonOperationRuntime, *args: Any, **kwargs: Any) -> Any:
        begins.append(self)
        return original_begin(self, *args, **kwargs)

    destination = tmp_path / "packages"
    with running_daemon_operations(tmp_path / "archive") as stack:
        monkeypatch.setattr(backup, "_require_readable_sqlite", refuse)
        monkeypatch.setattr(DaemonOperationRuntime, "begin_unbound_write", begin)
        envelope = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(destination), "check_only": check_only},
            archive_root=str(stack.archive_root),
        )
    assert observed == [failure]
    assert begins == []
    assert not destination.exists()
    assert envelope is not None
    retryable = error_code != sqlite3.SQLITE_ERROR
    assert envelope["outcome"] == ("failed" if retryable else "rejected")
    assert envelope["error"]["retryable"] is retryable
    assert envelope["error"]["code"] == ("backup_io_fault" if retryable else "OperationalError")


@pytest.mark.uses_real_clock("starts the real daemon coordinator and public operation client")
@pytest.mark.parametrize("check_only", [True, False])
def test_daemon_backup_preserves_disk_probe_advisory_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool
) -> None:
    from tests.infra.daemon_operations import running_daemon_operations

    def unavailable(path: object) -> Any:
        raise OSError("synthetic backup disk observation unavailable")

    destination = tmp_path / "packages"
    with running_daemon_operations(tmp_path / "archive") as stack:
        monkeypatch.setattr(os, "statvfs", unavailable)
        envelope = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(destination), "check_only": check_only},
            archive_root=str(stack.archive_root),
        )
    assert envelope is not None
    assert envelope["outcome"] == ("rejected" if check_only else "completed")
    if check_only:
        assert not destination.exists()
    else:
        assert Path(envelope["result"]["result"]["output_path"]).is_dir()


@pytest.mark.uses_real_clock("starts the real daemon coordinator and public operation client")
@pytest.mark.parametrize("fault_kind", ["cancel", "grouped_cleanup"])
def test_daemon_backup_preflight_delegates_cancellation_and_grouped_cleanup_to_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_kind: str
) -> None:
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from tests.infra.daemon_operations import running_daemon_operations

    failure: BaseException = (
        DaemonOperationCancelled("synthetic original preflight cancellation")
        if fault_kind == "cancel"
        else ExceptionGroup(
            "synthetic unsettled preflight cleanup", [sqlite3.OperationalError("busy"), RuntimeError("cleanup")]
        )
    )
    observed: list[BaseException] = []
    begins: list[object] = []
    original_begin = DaemonOperationRuntime.begin_unbound_write

    def refuse(path: Path) -> None:
        observed.append(failure)
        raise failure

    def begin(self: DaemonOperationRuntime, *args: Any, **kwargs: Any) -> Any:
        begins.append(self)
        return original_begin(self, *args, **kwargs)

    destination = tmp_path / "packages"
    with running_daemon_operations(tmp_path / "archive") as stack:
        monkeypatch.setattr(backup, "_require_readable_sqlite", refuse)
        monkeypatch.setattr(DaemonOperationRuntime, "begin_unbound_write", begin)
        envelope = stack.client.operation(
            "maintenance.backup", {"output_dir": str(destination)}, archive_root=str(stack.archive_root)
        )
    assert observed == [failure]
    assert begins == []
    assert not destination.exists()
    assert envelope is not None
    assert envelope["outcome"] == ("cancelled" if fault_kind == "cancel" else "failed")
    if fault_kind == "grouped_cleanup":
        assert envelope["error"]["code"] == "ExceptionGroup"
        assert envelope["error"]["retryable"] is False


@pytest.mark.uses_real_clock("starts the real daemon coordinator and public operation client")
@pytest.mark.parametrize("check_only", [True, False])
def test_daemon_backup_missing_required_tier_refuses_before_write_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool
) -> None:
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from tests.infra.daemon_operations import running_daemon_operations

    begins: list[object] = []
    original_begin = DaemonOperationRuntime.begin_unbound_write

    def begin(self: DaemonOperationRuntime, *args: Any, **kwargs: Any) -> Any:
        begins.append(self)
        return original_begin(self, *args, **kwargs)

    destination = tmp_path / "packages"
    with running_daemon_operations(tmp_path / "archive") as stack:
        (stack.archive_root / "user.db").unlink()
        monkeypatch.setattr(DaemonOperationRuntime, "begin_unbound_write", begin)
        envelope = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(destination), "check_only": check_only},
            archive_root=str(stack.archive_root),
        )
    assert begins == []
    assert not destination.exists()
    assert envelope is not None and envelope["outcome"] == "rejected"
    assert envelope["error"]["code"] == "backup_failed"
    assert envelope["error"]["data"]["backup_result"]["check_only"] is check_only


@pytest.mark.uses_real_clock("starts the real daemon coordinator and public operation client")
def test_daemon_backup_does_not_classify_later_snapshot_failure_as_readonly_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.infra.daemon_operations import running_daemon_operations

    failure = sqlite3.OperationalError("synthetic later snapshot storage failure")
    failure.sqlite_errorcode = sqlite3.SQLITE_BUSY
    reached: list[BaseException] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        reached.append(failure)
        raise failure

    with running_daemon_operations(tmp_path / "archive") as stack:
        monkeypatch.setattr(backup, "create_backup_package", refuse)
        envelope = stack.client.operation(
            "maintenance.backup",
            {"output_dir": str(tmp_path / "packages")},
            archive_root=str(stack.archive_root),
        )
    assert reached == [failure]
    assert envelope is not None
    assert envelope["outcome"] == "indeterminate"
    assert envelope["error"]["code"] != "backup_io_fault"
