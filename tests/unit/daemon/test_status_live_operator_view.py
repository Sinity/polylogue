"""What ``polylogued status`` shows an operator during a long daemon run.

Two gaps made the plain status command misleading while a build ran: the live
probe never authenticated, so a running daemon always refused it and the CLI
silently recomputed status in its own process; and a declared service that
failed was invisible to the ``ok`` verdict and to the text output.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.services import ServiceCapability, ServiceState
from polylogue.daemon.status import (
    format_daemon_status_lines,
    supervised_service_failures,
)
from polylogue.daemon.supervisor import DaemonSupervisor

_TOKEN = "neutral-test-token"


class _StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.headers.get("Authorization") != f"Bearer {_TOKEN}":
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps({"ok": True, "daemon": "polylogued", "probe": "live"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def status_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", url)
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_live_probe_sends_the_daemons_persisted_token(
    workspace_env: dict[str, Path],
    status_server: str,
    tmp_path: Path,
) -> None:
    """Anti-vacuity: without the ``Authorization`` header the fake daemon
    answers 401 and the probe returns ``None`` -- which is what every running
    daemon did to ``polylogued status`` before."""
    token_file = tmp_path / "api-token"
    token_file.write_text(_TOKEN, encoding="utf-8")
    token_file.chmod(0o600)
    with patch("polylogue.daemon.api_auth.api_auth_token_path", return_value=token_file):
        payload = daemon_cli._live_daemon_status_payload(timeout=5.0)
    assert payload is not None
    assert payload["probe"] == "live"


def test_live_probe_reports_a_refusal_instead_of_recomputing_silently(
    workspace_env: dict[str, Path],
    status_server: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A daemon that answers but refuses is named on stderr; a probe that
    swallowed the 401 printed nothing and the in-process recomputation looked
    like the daemon's view. Anti-vacuity: drop the ``HTTPError`` branch and
    stderr stays empty."""
    token_file = tmp_path / "api-token"
    token_file.write_text("a-stale-token", encoding="utf-8")
    token_file.chmod(0o600)
    with patch("polylogue.daemon.api_auth.api_auth_token_path", return_value=token_file):
        payload = daemon_cli._live_daemon_status_payload(timeout=5.0)
    assert payload is None
    assert "refused the status request (HTTP 401)" in capsys.readouterr().err


def test_live_probe_never_mints_a_token(
    workspace_env: dict[str, Path],
    status_server: str,
    tmp_path: Path,
) -> None:
    """A status read must not write the credential store. Anti-vacuity: using
    ``load_or_mint_api_auth_token`` in the probe creates the file."""
    token_file = tmp_path / "absent-token"
    with patch("polylogue.daemon.api_auth.api_auth_token_path", return_value=token_file):
        daemon_cli._live_daemon_status_payload(timeout=5.0)
    assert not token_file.exists()


def test_a_failed_isolated_service_is_named_with_its_reason() -> None:
    """Anti-vacuity: without ``supervised_service_failures`` status carried only
    the bare state string, so the reason an ``isolate`` service stopped was
    visible nowhere outside the supervisor."""

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
            return supervised_service_failures()
        finally:
            daemon_cli._set_active_supervisor(None)

    failures = asyncio.run(scenario())
    assert failures is not None
    assert [(row["service"], row["state"], row["reason"]) for row in failures] == [
        ("secret_scan_sweep", "failed", "RuntimeError: sweep exploded")
    ]


def test_text_status_lists_failed_services_and_currently_failing_loops() -> None:
    """Anti-vacuity: the formatter printed neither ``services`` nor
    ``periodic_loops``, so these errors reached only ``--format json``. A loop
    whose later run completed is recovered and must not be listed."""
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
                    "last_run_completed_at": 100.0,
                    "failures": 3,
                    "runs": 10,
                },
                {
                    "name": "fts_sweep",
                    "last_error": "old",
                    "last_error_type": "OperationalError",
                    "last_error_at": 50.0,
                    "last_run_completed_at": 100.0,
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
