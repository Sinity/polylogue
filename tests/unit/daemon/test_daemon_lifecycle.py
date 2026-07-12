"""Daemon lifecycle forensic contract tests.

These exercise the real disposable OPS-tier DDL and writer functions: replacing
them with an in-memory flag or test-only validator makes the fresh/stale and
signal forensic assertions fail.
"""

from __future__ import annotations

import signal
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.core.json import json_document
from polylogue.daemon import lifecycle as lifecycle_module
from polylogue.daemon.lifecycle import (
    DAEMON_HEARTBEAT_STALE_AFTER_SECONDS,
    DaemonLifecycle,
    install_signal_handlers,
    lifecycle_status,
    process_heartbeat_age_seconds,
    restore_signal_handlers,
)
from polylogue.daemon.status import _check_daemon_liveness, format_daemon_status_lines


def _bind_ops_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    index_db = tmp_path / "index.db"
    monkeypatch.setattr(lifecycle_module, "active_index_db_path", lambda: index_db)
    return tmp_path / "ops.db"


def test_lifecycle_row_records_start_heartbeat_signal_and_clean_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops_db = _bind_ops_db(monkeypatch, tmp_path)
    lifecycle = DaemonLifecycle.start(details={"component": "test"})
    lifecycle.heartbeat()
    lifecycle.record_signal_best_effort(signal.SIGTERM)
    lifecycle.stop(exit_kind="signal")

    with sqlite3.connect(ops_db) as conn:
        row = conn.execute(
            """
            SELECT started_at_ms, stopped_at_ms, last_heartbeat_at_ms,
                   signal, exit_kind, details_json
            FROM daemon_lifecycle WHERE run_id = ?
            """,
            (lifecycle.run_id,),
        ).fetchone()

    assert row is not None
    assert row[0] <= row[2] <= row[1]
    assert row[3] == "SIGTERM"
    assert row[4] == "signal"
    assert '"component":"test"' in row[5]


def test_lifecycle_status_rejects_stale_unstopped_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_ops_db(monkeypatch, tmp_path)
    lifecycle = DaemonLifecycle.start()
    current_ms = 2_000_000_000_000
    stale_ms = current_ms - int((DAEMON_HEARTBEAT_STALE_AFTER_SECONDS + 1) * 1000)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "UPDATE daemon_lifecycle SET last_heartbeat_at_ms = ? WHERE run_id = ?",
            (stale_ms, lifecycle.run_id),
        )
        conn.commit()

    payload = lifecycle_status(now_ms=current_ms)

    assert payload["state"] == "vanished"
    assert payload["running"] is False
    assert payload["heartbeat_age_s"] == DAEMON_HEARTBEAT_STALE_AFTER_SECONDS + 1


def test_cli_status_formats_a_stale_heartbeat_as_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = {"state": "vanished", "running": False, "heartbeat_age_s": 1801.0}
    monkeypatch.setattr("polylogue.daemon.lifecycle.lifecycle_status", lambda: stale)

    assert _check_daemon_liveness() is False
    lines = format_daemon_status_lines(json_document({"daemon_liveness": False, "daemon_lifecycle": stale}))

    assert "  Status: vanished heartbeat" in lines


def test_sigterm_dumps_threads_and_persists_signal_before_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_ops_db(monkeypatch, tmp_path)
    lifecycle = DaemonLifecycle.start()
    previous = install_signal_handlers(lifecycle)
    try:
        with patch("polylogue.daemon.lifecycle.faulthandler.dump_traceback") as dump:
            with pytest.raises(SystemExit) as raised:
                signal.raise_signal(signal.SIGTERM)

        assert raised.value.code == 128 + signal.SIGTERM
        dump.assert_called_once()
        assert dump.call_args.kwargs["all_threads"] is True
        status = lifecycle_status()
        assert status["signal"] == "SIGTERM"
    finally:
        restore_signal_handlers(previous)
        lifecycle.stop(exit_kind="signal")


def test_atexit_sentinel_marks_python_exit_without_claiming_a_clean_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_ops_db(monkeypatch, tmp_path)
    DaemonLifecycle.start()
    lifecycle_module._atexit_sentinel()

    status = lifecycle_status()

    assert status["state"] == "stopped"
    assert status["exit_kind"] == "atexit"
    assert process_heartbeat_age_seconds() is not None
