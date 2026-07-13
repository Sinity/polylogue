from __future__ import annotations

import subprocess
import sys
import threading
from os import getpid
from pathlib import Path

import pytest


def test_daemon_client_import_does_not_load_storage() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import polylogue.cli.daemon_client; assert 'polylogue.storage' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"POLYLOGUE_NO_DAEMON": "1"}, True),
        ({"POLYLOGUE_NO_DAEMON": "off"}, False),
        ({"POLYLOGUE_DAEMON": "off"}, True),
    ],
)
def test_daemon_escape_environment_is_explicit(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], expected: bool
) -> None:
    from polylogue.cli.archive_query import _daemon_disabled

    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    assert _daemon_disabled() is expected


def test_daemon_probe_rejects_the_tmp_archive_config_trap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A socket from a different resolved archive must never answer this CLI."""

    from polylogue.cli.daemon_client import DaemonClient

    client = DaemonClient(tmp_path / "daemon.sock")
    monkeypatch.setattr(
        client,
        "request_json",
        lambda _method, _path: {
            "archive_root": "/tmp",
            "index_schema_version": 24,
            "daemon_version": "0.1.0",
        },
    )

    assert (
        client.probe(
            archive_root="/realm/archive",
            index_schema_version=24,
            daemon_version="0.1.0",
        )
        is None
    )


def test_daemon_client_probes_the_production_uds_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stdlib client reaches the production AF_UNIX server, not a TCP substitute."""

    from http import HTTPStatus

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer

    def health(self: DaemonAPIHandler) -> None:
        self._send_json(
            HTTPStatus.OK,
            {
                "archive_root": "/realm/archive",
                "index_schema_version": 24,
                "daemon_version": "0.1.0",
                "commit": "test",
                "started_at": "2026-07-13T00:00:00+00:00",
            },
        )

    monkeypatch.setattr(DaemonAPIHandler, "_handle_health", health)
    socket_path = Path("/realm/tmp") / f"polylogue-uds-{getpid()}.sock"
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DaemonClient(socket_path)
        assert (
            client.probe(
                archive_root="/realm/archive",
                index_schema_version=24,
                daemon_version="0.1.0",
            )
            is not None
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
