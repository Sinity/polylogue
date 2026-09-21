from __future__ import annotations

import contextlib
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.infra.daemon_operations import running_daemon_operations


def _recorded_exchanges(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every socket exchange the client performs, delegating to the real one.

    The transport itself is untouched, so this observes the production route
    rather than replacing it: any extra request -- a liveness preflight, a
    compatibility status read, a retry -- appears as an extra entry.
    """

    from polylogue.daemon_client import DaemonClient

    seen: list[tuple[str, str]] = []
    original = DaemonClient._request_json_response

    def record(self: DaemonClient, method: str, path: str, body: object = None, **kwargs: object) -> object:
        seen.append((method, path))
        return original(self, method, path, body, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(DaemonClient, "_request_json_response", record)
    return seen


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
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer

    overlong_socket = tmp_path / ("socket-" + "x" * 160)
    with running_daemon_operations(tmp_path / "archive") as stack:
        with pytest.raises(OSError, match="AF_UNIX path too long"):
            DaemonAPIUnixHTTPServer(
                overlong_socket,
                archive_root=stack.archive_root,
                auth_token=None,
                write_bridge=stack.write_bridge,
                execution_kernel=stack.execution_kernel,
                operation_runtime=stack.runtime,
            )


def test_daemon_client_import_does_not_load_storage() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import polylogue.daemon_client; assert 'polylogue.storage' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_client_exposes_no_generic_offline_operation_fallback() -> None:
    """The only local fallback is named and constrained to read operations."""

    from polylogue.daemon_client import DaemonClient

    assert not hasattr(DaemonClient, "operation_with_direct_fallback")
    assert hasattr(DaemonClient, "operation_with_read_fallback")


def test_client_exposes_no_arbitrary_daemon_http_route() -> None:
    """The transport speaks the operation protocol, not the daemon's web routes.

    Anti-vacuity: restore a public ``request_json(method, path, ...)`` -- the
    generic route caller this client used to carry -- and this is red.  While
    it existed, "the CLI never calls a browser HTTP endpoint" was a claim about
    who happened to call what, and the two tests that guarded it patched a
    method the operation path never reached.
    """

    from polylogue.daemon_client import DaemonClient

    assert not hasattr(DaemonClient, "request_json")
    assert not hasattr(DaemonClient, "_raise_response_error")


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

    from polylogue.daemon_client import DaemonClient, DaemonOperationProtocolError
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


def test_operation_reaches_the_production_uds_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The stdlib client reaches the maintained production operation stack.

    Anti-vacuity: add any preflight -- a health GET, a status probe, a second
    exchange of any kind -- to :meth:`DaemonClient.operation` and ``exchanges``
    grows past the single declared POST, which is red.  The predecessor of this
    assertion patched ``DaemonClient.request_json`` instead, and a health
    preflight issued through the real transport left it green.
    """

    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    exchanges = _recorded_exchanges(monkeypatch)
    with running_daemon_operations(tmp_path / "archive") as stack:
        envelope = stack.client.operation("status", {}, archive_root=str(stack.archive_root))
    assert envelope is not None
    assert envelope["protocol"] == DAEMON_OPERATION_PROTOCOL
    assert envelope["result"]["total_sessions"] == 0
    assert exchanges == [("POST", "/api/operation")], exchanges


@contextlib.contextmanager
def _raw_unix_http_responder(
    socket_path: Path, *, status: int, payload: dict[str, object], framing: str | None = None
) -> Iterator[None]:
    """Serve one arbitrary HTTP payload without exercising daemon behavior."""

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    body = json.dumps(payload, separators=(",", ":")).encode()
    framing = framing if framing is not None else f"Content-Length: {len(body)}\r\n"
    response = (
        f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n{framing}Connection: close\r\n\r\n"
    ).encode() + body

    def serve_once() -> None:
        connection, _address = listener.accept()
        with connection:
            connection.recv(4096)
            connection.sendall(response)

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    try:
        yield
    finally:
        listener.close()
        thread.join(timeout=2)


@pytest.mark.parametrize("mutation", [False, True], ids=["read", "mutation"])
@pytest.mark.parametrize(
    "framing",
    [
        "Content-Length: 3\r\n",
        "Content-Length: -1\r\n",
        "Content-Length: 2\r\nContent-Length: 2\r\n",
        "Content-Length: 2\r\nContent-Length: 3\r\n",
        "Content-Length: 2\r\nTransfer-Encoding: identity\r\n",
        "",
    ],
    ids=["partial", "negative", "duplicate", "conflicting", "transfer-encoding", "missing"],
)
def test_operation_transport_refuses_invalid_response_framing(
    _short_uds_runtime_dir: Path, framing: str, mutation: bool
) -> None:
    """Accepting valid JSON from an incomplete or ambiguous frame makes this red."""
    from polylogue.daemon_client import (
        DaemonClient,
        DaemonMutationIndeterminateError,
        DaemonOperationProtocolError,
    )

    socket_path = _short_uds_runtime_dir / "framing.sock"
    error = DaemonMutationIndeterminateError if mutation else DaemonOperationProtocolError
    with _raw_unix_http_responder(socket_path, status=200, payload={}, framing=framing):
        client = DaemonClient(socket_path, timeout_s=1)
        with pytest.raises(error) as raised:
            client._request_json_response(
                "POST", "/api/operation", {"request_id": "framing-request"}, mutation=mutation
            )
        if mutation:
            assert isinstance(raised.value, DaemonMutationIndeterminateError)
            assert raised.value.request_id == "framing-request"


def test_daemon_mutation_timeout_is_typed_indeterminate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A connected daemon with no receipt is never interchangeable with no daemon."""

    from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError

    socket_path = tmp_path / "daemon.sock"
    socket_path.touch()

    class TimedOutConnection:
        connected = True

        def __init__(self, _socket_path: Path, _timeout: float | None) -> None:
            pass

        def connect(self) -> None:
            """PR #5043 connects before resolving credentials; the double must too."""

        def request(self, *_args: object, **_kwargs: object) -> None:
            pass

        def getresponse(self) -> object:
            raise TimeoutError("slow daemon response")

        def close(self) -> None:
            pass

    monkeypatch.setattr("polylogue.daemon_client._UnixHTTPConnection", TimedOutConnection)

    with pytest.raises(DaemonMutationIndeterminateError, match="POST /api/operation"):
        DaemonClient(socket_path, timeout_s=0.01).operation(
            "mutation.session.delete.execute", {"authorization_refs": ["ref-1"]}
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

        def connect(self) -> None:
            """PR #5043 connects before resolving credentials; the double must too."""

        def request(self, *_args: object, **_kwargs: object) -> None:
            pass

        def getresponse(self) -> object:
            raise KeyboardInterrupt

        def close(self) -> None:
            pass

    monkeypatch.setattr("polylogue.daemon_client._UnixHTTPConnection", InterruptedConnection)

    with pytest.raises(DaemonMutationIndeterminateError, match="POST /api/operation"):
        DaemonClient(socket_path).operation("mutation.session.delete.execute", {"authorization_refs": ["ref-1"]})


def test_initial_post_interrupt_signals_the_same_request_without_claiming_no_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError

    client = DaemonClient(tmp_path / "daemon.sock")
    cancelled: list[str] = []

    def interrupted(*args: object, **kwargs: object) -> object:
        raise DaemonMutationIndeterminateError(
            method="POST", path="/api/operation", request_id="interrupt-request"
        ) from KeyboardInterrupt()

    monkeypatch.setattr(client, "operation", interrupted)
    monkeypatch.setattr(client, "cancel", lambda request_id, **kwargs: cancelled.append(request_id))
    with pytest.raises(DaemonMutationIndeterminateError):
        client.operation_to_completion(
            "mutation.session.delete.execute", {"authorization_refs": ["ref-1"]}, archive_root=str(tmp_path)
        )
    assert cancelled == ["interrupt-request"]


def test_write_deadline_does_not_mutate_a_shared_clients_read_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.daemon_client import DaemonClient

    client = DaemonClient(tmp_path / "daemon.sock", timeout_s=0.25)
    captured: list[object] = []

    def request(*args: object, **kwargs: object) -> None:
        captured.append((client.timeout_s, kwargs["timeout_s"]))
        return None

    monkeypatch.setattr(client, "_request_json_response", request)
    assert (
        client.operation("mutation.session.delete.execute", {"authorization_refs": ["ref-1"]}, deadline_ms=1500) is None
    )
    assert captured == [(0.25, 2.5)]
    assert client.timeout_s == 0.25


def test_await_interrupt_cancels_the_original_request_not_the_control_exchange(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError

    client = DaemonClient(tmp_path / "daemon.sock")
    cancelled: list[str] = []
    monkeypatch.setattr(
        client,
        "operation",
        lambda *args, **kwargs: {"request_id": "accepted-mutation", "outcome": "accepted", "result": {"sequence": 1}},
    )

    def interrupted(*args: object, **kwargs: object) -> object:
        raise DaemonMutationIndeterminateError(
            method="POST", path="/api/operation", request_id="await-control"
        ) from KeyboardInterrupt()

    monkeypatch.setattr(client, "await_operation", interrupted)
    monkeypatch.setattr(client, "cancel", lambda request_id, **kwargs: cancelled.append(request_id))
    with pytest.raises(DaemonMutationIndeterminateError):
        client.operation_to_completion(
            "mutation.session.delete.execute", {"authorization_refs": ["ref-1"]}, archive_root=str(tmp_path)
        )
    assert cancelled == ["accepted-mutation"]


def test_an_accepted_write_is_never_called_indeterminate_without_one_receipt_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A durably accepted write consults its lifecycle before being given up on.

    ``operation_to_completion`` budgets the whole exchange at ``spec.deadline_s``,
    while its own submit is allowed to take that deadline plus a second. When the
    submit uses the budget, the receipt wait has none left -- and reporting
    ``indeterminate`` there claims the outcome is unknown without ever asking the
    durable lifecycle that already settled it. Every such report costs the
    operator a manual recovery for a write that had a receipt waiting.

    Anti-vacuity: the guarded ``while perf_counter() < deadline:`` this replaces
    performs zero awaits under this clock, so ``awaited`` stays empty and the
    returned outcome is ``indeterminate`` -- both assertions go red.
    """
    from polylogue.daemon_client import DaemonClient

    # The first read builds the deadline; every later read is past it, which is
    # the exhausted-budget state this law is about.
    readings = iter([0.0])
    monkeypatch.setattr("polylogue.daemon_client.perf_counter", lambda: next(readings, 1_000_000.0))

    reference = {
        "request_id": "accepted-write",
        "archive_identity": "archive-identity",
        "principal_ref": "principal",
        "fingerprint": "fingerprint",
        "operation_name": "mutation.session.tag",
    }
    accepted = {
        "request_id": "accepted-write",
        "outcome": "accepted",
        "result": {"sequence": 1, "outcome": "accepted", "reference": reference},
        "accepted_reference": reference,
    }
    settled: dict[str, object] = {
        key: {}
        for key in (
            "archive",
            "generation",
            "readiness",
            "served_by",
            "timing",
            "schema_versions",
            "authority_snapshot",
        )
    }
    settled["degraded_components"] = []
    settled["result"] = {
        "sequence": 2,
        "outcome": "completed",
        "effect": "committed",
        "affected_count": 1,
        "reference": reference,
    }

    client = DaemonClient(tmp_path / "daemon.sock")
    awaited: list[tuple[str, int]] = []

    def await_operation(request_id: str, **kwargs: object) -> dict[str, object]:
        awaited.append((request_id, int(str(kwargs["timeout_ms"]))))
        return settled

    monkeypatch.setattr(client, "operation", lambda *args, **kwargs: accepted)
    monkeypatch.setattr(client, "await_operation", await_operation)

    envelope = client.operation_to_completion(
        "mutation.session.tag",
        {"session_ids": ["codex-session:one"], "tags": ["t"]},
        archive_root=str(tmp_path),
    )

    assert awaited == [("accepted-write", 1)]
    assert envelope is not None
    assert envelope["outcome"] == "completed"
