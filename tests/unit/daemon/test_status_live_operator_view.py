"""What ``polylogued status`` shows an operator during a long daemon run.

Two gaps made the plain status command misleading while a build ran: the live
HTTP probe never authenticated, so a running daemon always refused it and the
CLI silently recomputed status in its own process; and a declared service that
failed was invisible to the ``ok`` verdict and to the text output. The probe
now uses the daemon's peer-verified machine socket, as every CLI verb does.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.config import load_polylogue_config
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon import commands as daemon_commands
from polylogue.daemon.services import ServiceCapability, ServiceState
from polylogue.daemon.status import (
    format_daemon_status_lines,
    supervised_service_snapshot,
)
from polylogue.daemon.supervisor import DaemonSupervisor
from tests.infra.daemon_operations import cli_daemon_archive


def test_live_probe_reads_the_running_daemon_over_its_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe gets the daemon's own ``status`` operation over the machine
    socket. Anti-vacuity: the previous HTTP probe sent no bearer, every
    running daemon refused it, and the probe returned ``None``."""
    with cli_daemon_archive(tmp_path / "archive", monkeypatch, home=tmp_path / "home"):
        payload = daemon_commands._live_daemon_status_payload(load_polylogue_config())
    assert payload is not None
    assert "total_sessions" in payload


def test_live_probe_without_a_daemon_reports_typed_absence(
    workspace_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent resident has no in-process substitute view."""
    payload = daemon_commands._live_daemon_status_payload(load_polylogue_config())
    assert payload["ok"] is False
    assert payload["daemon_liveness"] is False
    snapshot = payload["status_snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["reason"] == "daemon_absent"
    assert capsys.readouterr().err == ""


def test_live_probe_reports_a_daemon_that_fails_the_request(
    workspace_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A resident refusal retains its code without local recomputation."""
    from polylogue.cli.operation_kernel import OperationFailedError

    with patch(
        "polylogue.cli.operation_kernel.dispatch",
        side_effect=OperationFailedError("unauthorized", "machine authentication required"),
    ):
        payload = daemon_commands._live_daemon_status_payload(load_polylogue_config())
    assert payload["ok"] is False
    snapshot = payload["status_snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["reason"] == "unauthorized"
    assert payload["daemon_liveness"] is None
    assert capsys.readouterr().err == ""


def test_a_failed_isolated_service_is_named_with_its_reason() -> None:
    """Anti-vacuity: without the failure projection status carried only the
    bare state string, so the reason an ``isolate`` service stopped was
    visible nowhere outside the supervisor. States and failures come from one
    read, so the failed state and its failure row always agree."""

    async def failing() -> None:
        raise RuntimeError("sweep exploded")

    async def healthy() -> None:
        return None

    async def scenario() -> list[dict[str, object]] | None:
        supervisor = DaemonSupervisor(capabilities=frozenset(ServiceCapability) - {ServiceCapability.SCHEMA_BLOCKED})
        supervisor.start("secret_scan_sweep", failing)
        supervisor.start("health_check", healthy)
        await supervisor.wait()
        assert supervisor.state("secret_scan_sweep") is ServiceState.FAILED
        daemon_cli._set_active_supervisor(supervisor)
        try:
            snapshot = supervised_service_snapshot()
            assert snapshot is not None
            states, failures = snapshot
            assert states["secret_scan_sweep"] == "failed"
            return failures
        finally:
            daemon_cli._set_active_supervisor(None)

    failures = asyncio.run(scenario())
    assert failures is not None
    assert [(row["service"], row["state"], row["reason"]) for row in failures] == [
        ("secret_scan_sweep", "failed", "RuntimeError: sweep exploded")
    ]


def test_text_status_lists_failed_services_and_currently_failing_loops() -> None:
    """Anti-vacuity: the formatter printed neither ``services`` nor
    ``periodic_loops``, so these errors reached only ``--format json``. The
    verdict is the recorded outcome of the latest pass, not an ordering of
    wall-clock stamps: the timestamps here are deliberately inverted, as after
    a clock step, and must not change which loop is listed."""
    lines = format_daemon_status_lines(
        {
            "ok": False,
            "service_failures": [{"service": "secret_scan_sweep", "state": "failed", "reason": "RuntimeError: boom"}],
            "periodic_loops": [
                {
                    "name": "wal_checkpoint",
                    "last_error": "disk I/O error",
                    "last_error_type": "OperationalError",
                    "last_error_at": 200.0,
                    "last_run_completed_at": 300.0,
                    "last_run_failed": True,
                    "failures": 3,
                    "runs": 10,
                },
                {
                    "name": "fts_sweep",
                    "last_error": "old",
                    "last_error_type": "OperationalError",
                    "last_error_at": 500.0,
                    "last_run_completed_at": 100.0,
                    "last_run_failed": False,
                    "failures": 1,
                    "runs": 9,
                },
            ],
        }
    )
    text = "\n".join(lines)
    assert "FAILED SERVICES: 1" in text
    assert "secret_scan_sweep: failed - RuntimeError: boom" in text
    assert "Failing loops: 1" in text
    assert "wal_checkpoint: OperationalError: disk I/O error" in text
    assert "fts_sweep" not in text
