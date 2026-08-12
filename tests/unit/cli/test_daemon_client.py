from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from os import getpid
from pathlib import Path

import pytest


@pytest.fixture
def _short_uds_runtime_dir() -> Iterator[Path]:
    """Keep UDS route tests under the operating system socket-path limit."""
    runtime_dir = Path(tempfile.mkdtemp(prefix="plg-client-uds-"))
    try:
        yield runtime_dir
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


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


def test_daemon_client_probes_the_production_uds_server(
    monkeypatch: pytest.MonkeyPatch, _short_uds_runtime_dir: Path
) -> None:
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
    socket_path = _short_uds_runtime_dir / f"daemon-{getpid()}.sock"
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    server.auth_token = "uds-test-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DaemonClient(socket_path, auth_token="uds-test-token")
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


def test_daemon_client_can_probe_matching_writer_through_degraded_health(
    monkeypatch: pytest.MonkeyPatch,
    _short_uds_runtime_dir: Path,
) -> None:
    """Maintenance discovers the writer without weakening query readiness."""
    from http import HTTPStatus

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer

    def degraded_health(self: DaemonAPIHandler) -> None:
        self._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            {
                "archive_root": "/realm/archive",
                "index_schema_version": 24,
                "daemon_version": "0.1.0",
                "raw_failure_lifecycle_state": "degraded",
            },
        )

    monkeypatch.setattr(DaemonAPIHandler, "_handle_health", degraded_health)
    socket_path = _short_uds_runtime_dir / f"degraded-{getpid()}.sock"
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
            is None
        )
        assert (
            client.probe(
                archive_root="/realm/archive",
                index_schema_version=24,
                daemon_version="0.1.0",
                accept_degraded=True,
            )
            is not None
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_daemon_client_rejects_unrelated_503_when_degraded_probe_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    _short_uds_runtime_dir: Path,
) -> None:
    """The production probe route accepts a maintenance ``degraded`` 503,
    not any matching identity payload. Mutating that lifecycle state to
    ``blocked`` must keep the writer unavailable to the caller."""
    from http import HTTPStatus

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer

    def blocked_health(self: DaemonAPIHandler) -> None:
        self._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            {
                "archive_root": "/realm/archive",
                "index_schema_version": 24,
                "daemon_version": "0.1.0",
                "raw_failure_lifecycle_state": "blocked",
            },
        )

    monkeypatch.setattr(DaemonAPIHandler, "_handle_health", blocked_health)
    socket_path = _short_uds_runtime_dir / f"blocked-{getpid()}.sock"
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
                accept_degraded=True,
            )
            is None
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_daemon_client_preserves_typed_4xx_detail_from_the_production_uds_server(
    _short_uds_runtime_dir: Path,
) -> None:
    """Maintenance clients can surface a daemon validation reason, not only a transport failure."""
    from http import HTTPStatus

    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
    from polylogue.daemon_client import DaemonClient, DaemonResponseError

    class InvalidCanaryReportHandler(DaemonAPIHandler):
        def _handle_consume_canary_report(self) -> None:
            self._send_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "canary_report_invalid",
                "receipt is missing the canonical acceptance profile",
            )

    socket_path = _short_uds_runtime_dir / f"canary-4xx-{getpid()}.sock"
    server = DaemonAPIUnixHTTPServer(socket_path, InvalidCanaryReportHandler)
    server.auth_token = "uds-test-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DaemonClient(socket_path, auth_token="uds-test-token")
        with pytest.raises(DaemonResponseError, match="missing the canonical acceptance profile") as raised:
            client.request_json(
                "POST",
                "/api/maintenance/consume-canary-report",
                {"report_path": "/fixture/report.json"},
                raise_for_status=True,
            )
        assert raised.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
        assert raised.value.code == "canary_report_invalid"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_daemon_mutation_timeout_is_typed_indeterminate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A connected daemon with no receipt is never interchangeable with no daemon."""

    from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError

    socket_path = tmp_path / "daemon.sock"
    socket_path.touch()

    class TimedOutConnection:
        connected = True

        def __init__(self, _socket_path: Path, _timeout: float | None) -> None:
            pass

        def request(self, *_args: object, **_kwargs: object) -> None:
            pass

        def getresponse(self) -> object:
            raise TimeoutError("slow daemon response")

        def close(self) -> None:
            pass

    monkeypatch.setattr("polylogue.daemon_client._UnixHTTPConnection", TimedOutConnection)

    with pytest.raises(DaemonMutationIndeterminateError, match="POST /api/cli/delete"):
        DaemonClient(socket_path, timeout_s=0.01).request_mutation_json(
            "POST", "/api/cli/delete", {"session_ids": ["s1"]}
        )
