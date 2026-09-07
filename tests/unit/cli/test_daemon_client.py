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
    # Managed pytest deliberately puts TMPDIR under its deeply nested lease.
    # AF_UNIX counts the complete encoded path, so choose a short socket-only
    # root independently of fixture storage.
    runtime_dir = Path(tempfile.mkdtemp(prefix="plg-uds-", dir="/tmp"))
    try:
        yield runtime_dir
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def test_uds_server_preserves_bind_error_during_partial_initialization(tmp_path: Path) -> None:
    """A failed AF_UNIX bind is not masked by cleanup of uninitialized state."""
    from http.server import BaseHTTPRequestHandler

    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer

    overlong_socket = tmp_path / ("socket-" + "x" * 160)
    with pytest.raises(OSError, match="AF_UNIX path too long"):
        DaemonAPIUnixHTTPServer(overlong_socket, BaseHTTPRequestHandler)


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


def test_operation_rejects_a_socket_serving_a_different_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A socket from a different resolved archive must never answer this CLI.

    Anti-vacuity: dropping the archive-identity comparison in ``operation``
    makes this envelope acceptable.
    """

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.daemon_client import DaemonOperationProtocolError
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    client = DaemonClient(tmp_path / "daemon.sock")
    captured: dict[str, object] = {}

    def fake_request(
        method: str, path: str, body: dict[str, object], **_kwargs: object
    ) -> tuple[int, dict[str, object]]:
        captured.update(body)
        return 200, {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": "status",
            "request_id": body["request_id"],
            "archive": {"root": "/tmp"},
        }

    monkeypatch.setattr(client, "_request_json_response", fake_request)

    with pytest.raises(DaemonOperationProtocolError, match="different archive identity"):
        client.operation("status", {}, archive_root="/realm/archive")


def test_operation_reaches_the_production_uds_server(
    monkeypatch: pytest.MonkeyPatch, _short_uds_runtime_dir: Path
) -> None:
    """The stdlib client reaches the production AF_UNIX server in one request.

    Anti-vacuity: the handler below fails the test if the client issues a
    health probe before its operation.
    """

    from http import HTTPStatus

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    def refuse_health(self: DaemonAPIHandler) -> None:
        raise AssertionError("the CLI path must not issue a health probe")

    def status(self: DaemonAPIHandler) -> None:
        self._send_json(HTTPStatus.OK, {"daemon": {"running": True}})

    monkeypatch.setattr(DaemonAPIHandler, "_handle_health", refuse_health)
    monkeypatch.setattr(DaemonAPIHandler, "_handle_status", lambda self, _params: status(self))
    socket_path = _short_uds_runtime_dir / f"daemon-{getpid()}.sock"
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    server.auth_token = "uds-test-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DaemonClient(socket_path, auth_token="uds-test-token", timeout_s=2)
        envelope = client.operation("status", {})
        assert envelope is not None
        assert envelope["protocol"] == DAEMON_OPERATION_PROTOCOL
        assert envelope["result"] == {"daemon": {"running": True}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_machine_socket_rejects_legacy_non_operation_routes(
    _short_uds_runtime_dir: Path,
) -> None:
    """The machine socket exposes only the declared operation endpoint."""
    from http import HTTPStatus

    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
    from polylogue.daemon_client import DaemonClient, DaemonResponseError

    class InvalidMaintenanceHandler(DaemonAPIHandler):
        def _handle_rebuild_index(self) -> None:
            self._send_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "canary_report_invalid",
                "receipt is missing the canonical acceptance profile",
            )

    socket_path = _short_uds_runtime_dir / f"canary-4xx-{getpid()}.sock"
    server = DaemonAPIUnixHTTPServer(socket_path, InvalidMaintenanceHandler)
    server.auth_token = "uds-test-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = DaemonClient(socket_path, auth_token="uds-test-token")
        with pytest.raises(DaemonResponseError, match="daemon returned HTTP 404") as raised:
            client.request_json(
                "POST",
                "/api/maintenance/rebuild-index",
                {"promote": False},
                raise_for_status=True,
            )
        assert raised.value.status == HTTPStatus.NOT_FOUND
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

    with pytest.raises(DaemonMutationIndeterminateError, match="POST /api/operation"):
        DaemonClient(socket_path, timeout_s=0.01).operation(
            "mutation.session.delete.execute", {"authorization_tokens": ["t1"]}
        )


def test_daemon_mutation_interrupt_after_connect_is_typed_indeterminate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl-C cannot turn an accepted mutation with no receipt into an ordinary abort."""

    from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError

    socket_path = tmp_path / "daemon.sock"
    socket_path.touch()

    class InterruptedConnection:
        connected = True

        def __init__(self, _socket_path: Path, _timeout: float | None) -> None:
            pass

        def request(self, *_args: object, **_kwargs: object) -> None:
            pass

        def getresponse(self) -> object:
            raise KeyboardInterrupt

        def close(self) -> None:
            pass

    monkeypatch.setattr("polylogue.daemon_client._UnixHTTPConnection", InterruptedConnection)

    with pytest.raises(DaemonMutationIndeterminateError, match="POST /api/operation"):
        DaemonClient(socket_path).operation("mutation.session.delete.execute", {"authorization_tokens": ["t1"]})
