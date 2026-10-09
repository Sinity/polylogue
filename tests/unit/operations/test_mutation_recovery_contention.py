"""Native SQLite contention preserves accepted mutation recovery custody."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from polylogue.operations import mutation_actuators
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_transaction import OperationExecutor
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.operation_recovery import recover_on_admitted_owner
from tests.unit.operations.test_mutation_actuators import _seed_archive_session
from tests.unit.operations.test_mutation_crash_recovery import _SCENARIOS, _crash_mid_mutation, _principal


@contextmanager
def _native_user_contention(root: Path) -> Iterator[list[sqlite3.Connection]]:
    original = open_connection
    calls: list[sqlite3.Connection] = []
    # Keep the fault owner on this test thread through compute-worker cleanup.
    # Its transaction is used sequentially on the admitted worker, then retired
    # here after the worker has returned.
    blocker = sqlite3.connect(root / "user.db", check_same_thread=False)

    def contended(path: Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original(path, *args, **kwargs)
        if Path(path) == root / "user.db":
            # The actuator receives its real production connection. A real
            # second transaction holds the native lock; no exception is mocked.
            conn.execute("PRAGMA busy_timeout=0")
            blocker.execute("BEGIN IMMEDIATE")
            calls.append(conn)
        return conn

    try:
        with patch.object(mutation_actuators, "open_connection", contended):
            yield calls
    finally:
        blocker.rollback()
        blocker.close()


def _ledger(root: Path, operation_id: str) -> dict[str, Any]:
    with closing(sqlite3.connect(root / "audit.db")) as conn:
        state = conn.execute(
            "SELECT status, terminal_reason, unknown_count, error_summary FROM operation_runs WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        authority = conn.execute(
            "SELECT actor_ref, plan_hash, preview_id, initial_authorization_id FROM operation_runs WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        targets = conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id=? ORDER BY ordinal", (operation_id,)
        ).fetchall()
        barrier = conn.execute(
            "SELECT COUNT(*) FROM operation_runs WHERE operation_id=? AND status IN ('running','interrupted')",
            (operation_id,),
        ).fetchone()[0]
        recovery_events = conn.execute(
            "SELECT COUNT(*) FROM operation_events WHERE operation_id=? AND event_type='recovery_resolved'",
            (operation_id,),
        ).fetchone()[0]
    with closing(sqlite3.connect(root / "user.db")) as conn:
        setting = conn.execute("SELECT value_json FROM user_settings WHERE setting_key='subscription_tier'").fetchone()
    return {
        "state": state,
        "targets": targets,
        "barrier_count": barrier,
        "setting": setting,
        "authority": authority,
        "recovery_events": recovery_events,
    }


def test_native_busy_setting_recovery_vs_normal_indeterminate(tmp_path: Path) -> None:
    """Terminalizing wrapped SQLITE_BUSY drops the barrier and healthy replay."""
    scenario = next(item for item in _SCENARIOS if item.name == "set-user-setting")
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    _seed_archive_session(recovery_root, native_id="bootstrap")
    operation_id = _crash_mid_mutation(recovery_root, scenario, "before-apply")
    original_authority = _ledger(recovery_root, operation_id)["authority"]
    with _native_user_contention(recovery_root) as blockers:
        recover_on_admitted_owner(recovery_root)
        assert len(blockers) == 1
    recovery_fault = _ledger(recovery_root, operation_id)
    recover_on_admitted_owner(recovery_root)
    recovery_healthy = _ledger(recovery_root, operation_id)
    recover_on_admitted_owner(recovery_root)
    assert _ledger(recovery_root, operation_id) == recovery_healthy

    normal_root = tmp_path / "normal"
    normal_root.mkdir()
    _seed_archive_session(normal_root, native_id="bootstrap")
    native_error = None
    with ArchiveStore.open_existing(normal_root, read_only=False) as archive:
        assert scenario.args is not None
        args = scenario.args(normal_root, archive)
        binding = runtime_operation_binding(scenario.actuator)
        executor = OperationExecutor.for_archive_root(normal_root)
        principal = _principal(binding)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=normal_root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        with _native_user_contention(normal_root) as blockers:
            with write_lease("wave8.native-contention.normal", archive_root=normal_root):
                try:
                    executor.execute_bound(binding, preview, authorization, args)
                except RuntimeError as error:
                    cause = error.__cause__
                    assert isinstance(cause, sqlite3.OperationalError)
                    native_error = [cause.sqlite_errorcode, cause.sqlite_errorname]
                else:
                    raise AssertionError("native lock did not refuse the actuator transaction")
            assert len(blockers) == 1
    with closing(sqlite3.connect(normal_root / "audit.db")) as conn:
        normal_id = conn.execute("SELECT operation_id FROM operation_runs").fetchone()[0]
    normal_fault = _ledger(normal_root, normal_id)
    assert native_error == [sqlite3.SQLITE_BUSY, "SQLITE_BUSY"]
    assert recovery_fault["state"][0] == "interrupted"
    assert recovery_fault["targets"] == [("unknown",)]
    assert recovery_fault["barrier_count"] == 1
    assert recovery_fault["setting"] is None
    assert recovery_fault["recovery_events"] == 0
    assert recovery_healthy["state"][:2] == ("completed", "recovered_complete")
    assert recovery_healthy["barrier_count"] == 0
    assert recovery_healthy["setting"] == ('"max_5x"',)
    assert recovery_healthy["recovery_events"] == 1
    assert recovery_fault["authority"] == recovery_healthy["authority"] == original_authority
    assert normal_fault["state"][0] == "interrupted"
    assert normal_fault["targets"] == [("unknown",)]
    assert normal_fault["barrier_count"] == 1


def test_setting_recovery_keeps_deterministic_failures_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying every generic actuator error would leave this barrier unresolved."""
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "set-user-setting")
    operation_id = _crash_mid_mutation(root, scenario, "before-apply")

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("deterministic actuator refusal")

    monkeypatch.setattr(mutation_actuators.SetUserSettingActuator, "apply", refuse)
    recover_on_admitted_owner(root)
    failed = _ledger(root, operation_id)
    assert failed["state"][:2] == ("failed", "recovery_replay_failed")
    assert failed["targets"] == [("failed",)]
    assert failed["barrier_count"] == 0
    assert failed["setting"] is None
