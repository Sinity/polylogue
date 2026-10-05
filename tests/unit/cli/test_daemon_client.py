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
from typing import cast

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


def test_client_exposes_no_offline_operation_fallback_at_all() -> None:
    """The transport has no local execution route, named or generic.

    It used to carry ``operation_with_read_fallback``, which ran
    ``execute_operation`` in the caller's process against a locally opened
    ``ArchiveStore`` whenever no socket answered. It had no production caller
    at any point after the CLI kernel took over dispatch, so it was a second
    execution mode reachable only by a new adapter that found it -- exactly
    the shape the sole-writer programme removes (polylogue-3eexy AC3).

    Anti-vacuity: restore either name and this is red. Asserting only the
    absence of ``operation_with_direct_fallback``, as this did before, passes
    with the read fallback still present.
    """

    from polylogue.daemon_client import DaemonClient

    assert not hasattr(DaemonClient, "operation_with_direct_fallback")
    assert not hasattr(DaemonClient, "operation_with_read_fallback")


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
    from polylogue.cli.read_dispatch import daemon_route_disabled

    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    assert daemon_route_disabled() is expected


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
            "mutation.session.delete.execute", {"authorization_ref": "ref-1"}
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
        DaemonClient(socket_path).operation("mutation.session.delete.execute", {"authorization_ref": "ref-1"})


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
            "mutation.session.delete.execute", {"authorization_ref": "ref-1"}, archive_root=str(tmp_path)
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
    assert client.operation("mutation.session.delete.execute", {"authorization_ref": "ref-1"}, deadline_ms=1500) is None
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
    with pytest.raises(DaemonMutationIndeterminateError) as raised:
        client.operation_to_completion(
            "mutation.session.delete.execute", {"authorization_ref": "ref-1"}, archive_root=str(tmp_path)
        )
    assert cancelled == ["accepted-mutation"]
    assert raised.value.request_id == "accepted-mutation"


def test_await_disconnect_reports_the_accepted_mutation_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A dropped ``operation.await`` exchange names the accepted write for recovery.

    Anti-vacuity: re-raising the await's own failure carries the control
    call's id, which has no durable mutation receipt to recover.
    """
    from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError

    client = DaemonClient(tmp_path / "daemon.sock")
    monkeypatch.setattr(
        client,
        "operation",
        lambda *args, **kwargs: {"request_id": "accepted-mutation", "outcome": "accepted", "result": {"sequence": 1}},
    )

    def disconnected(*args: object, **kwargs: object) -> object:
        raise DaemonMutationIndeterminateError(
            method="POST", path="/api/operation", request_id="await-control"
        ) from ConnectionResetError()

    monkeypatch.setattr(client, "await_operation", disconnected)
    with pytest.raises(DaemonMutationIndeterminateError) as raised:
        client.operation_to_completion(
            "mutation.session.delete.execute", {"authorization_ref": "ref-1"}, archive_root=str(tmp_path)
        )
    assert raised.value.request_id == "accepted-mutation"


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
        "fingerprint": "a" * 64,
        "artifact_kind": "fixture-accepted-request",
        "artifact_ref": "fixture-acceptance",
        "accepted_at_ms": 0,
        "part_count": 1,
        "accepted_deadline_unix_ms": None,
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


def _accepted_ingest() -> tuple[dict[str, object], dict[str, object]]:
    reference: dict[str, object] = {
        "request_id": "accepted-ingest",
        "archive_identity": "archive-identity",
        "principal_ref": "principal",
        "fingerprint": "a" * 64,
        "artifact_kind": "fixture-accepted-request",
        "artifact_ref": "fixture-acceptance",
        "accepted_at_ms": 0,
        "part_count": 1,
        "accepted_deadline_unix_ms": None,
        "operation_name": "ingest",
    }
    accepted: dict[str, object] = {
        "request_id": "accepted-ingest",
        "outcome": "accepted",
        "result": {"sequence": 1, "outcome": "accepted", "reference": reference},
        "accepted_reference": reference,
    }
    return accepted, reference


def test_follow_operation_waits_on_the_accepted_request_within_the_callers_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``import --wait`` follows the work it already submitted; it never resubmits.

    Anti-vacuity: budget the waits by the declared ingest deadline (300s)
    instead of ``wait_s`` and the ``timeout_ms`` assertion goes red; route the
    follow through ``operation_to_completion`` and the submit guard raises.
    """
    from polylogue.daemon_client import DaemonClient

    accepted, reference = _accepted_ingest()
    readings = iter([0.0, 0.0, 0.5, 0.5])
    monkeypatch.setattr("polylogue.daemon_client.perf_counter", lambda: next(readings, 0.5))
    states: Iterator[dict[str, object]] = iter(
        [
            {"result": {"sequence": 2, "outcome": "running", "reference": reference}},
            {
                **{key: {} for key in ("archive", "generation", "readiness", "served_by", "timing", "schema_versions")},
                "authority_snapshot": {},
                "degraded_components": [],
                "result": {"sequence": 3, "outcome": "cancelled", "reference": reference},
            },
        ]
    )
    client = DaemonClient(tmp_path / "daemon.sock")
    awaited: list[tuple[str, int, int]] = []

    def await_operation(request_id: str, **kwargs: object) -> dict[str, object]:
        awaited.append((request_id, int(str(kwargs["after_sequence"])), int(str(kwargs["timeout_ms"]))))
        return next(states)

    def no_resubmission(*args: object, **kwargs: object) -> None:
        raise AssertionError("follow_operation resubmitted accepted work")

    monkeypatch.setattr(client, "operation", no_resubmission)
    monkeypatch.setattr(client, "await_operation", await_operation)

    envelope = client.follow_operation("ingest", accepted, archive_root=str(tmp_path), wait_s=2.0)

    assert [(request_id, after) for request_id, after, _ in awaited] == [("accepted-ingest", 1), ("accepted-ingest", 2)]
    assert all(timeout_ms <= 2000 for _, _, timeout_ms in awaited)
    assert envelope["outcome"] == "cancelled"


def test_follow_operation_reports_an_exhausted_budget_as_indeterminate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Running work past the caller's budget is indeterminate, not failed or resubmitted."""
    from polylogue.daemon_client import DaemonClient

    accepted, reference = _accepted_ingest()
    readings = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr("polylogue.daemon_client.perf_counter", lambda: next(readings, 10.0))
    client = DaemonClient(tmp_path / "daemon.sock")
    monkeypatch.setattr(
        client,
        "await_operation",
        lambda request_id, **kwargs: {"result": {"sequence": 2, "outcome": "running", "reference": reference}},
    )

    envelope = client.follow_operation("ingest", accepted, archive_root=str(tmp_path), wait_s=1.0)

    assert envelope["outcome"] == "indeterminate"
    assert envelope["request_id"] == "accepted-ingest"


def test_progress_frames_are_delivered_before_terminal_and_renderer_failures_are_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Long-running operation progress is observational, not a terminal gate."""
    from polylogue.daemon_client import DaemonClient

    reference = {
        "request_id": "embedding-progress",
        "archive_identity": "archive-identity",
        "principal_ref": "principal",
        "fingerprint": "a" * 64,
        "artifact_kind": "fixture-accepted-request",
        "artifact_ref": "fixture-acceptance",
        "accepted_at_ms": 0,
        "part_count": 1,
        "accepted_deadline_unix_ms": None,
        "operation_name": "maintenance.embeddings.backfill",
    }
    accepted = {
        "request_id": "embedding-progress",
        "outcome": "accepted",
        "result": {"sequence": 1, "outcome": "accepted", "reference": reference},
        "accepted_reference": reference,
    }
    states: Iterator[dict[str, object]] = iter(
        [
            {
                "archive": {},
                "generation": {},
                "readiness": {},
                "served_by": {},
                "timing": {},
                "schema_versions": {},
                "authority_snapshot": {},
                "degraded_components": [],
                "result": {
                    "sequence": 1,
                    "outcome": "running",
                    "reference": reference,
                    "progress_sequence": 1,
                    "progress_events": [
                        {"sequence": 1, "session_id": "s1", "state": "started", "estimated_cost_usd": 0.01}
                    ],
                },
            },
            {
                "archive": {},
                "generation": {},
                "readiness": {},
                "served_by": {},
                "timing": {},
                "schema_versions": {},
                "authority_snapshot": {},
                "degraded_components": [],
                "result": {
                    "sequence": 2,
                    "outcome": "completed",
                    "reference": reference,
                    "progress_sequence": 1,
                    "progress_events": [],
                    "result": {
                        "operation": "maintenance.embeddings.backfill",
                        "outcome": "completed",
                        "sequence": 1,
                        "effect": "committed",
                        "affected_count": 1,
                        "stop_reason": None,
                        "progress": {
                            "state": "complete",
                            "computed": 1,
                            "failed": 0,
                            "estimated_cost_usd": 0.01,
                        },
                        "result": {"done": 1, "pending": 0, "failed": 0},
                    },
                },
            },
        ]
    )
    after_sequences: list[tuple[int, int]] = []
    monkeypatch.setattr(DaemonClient, "operation", lambda self, *args, **kwargs: accepted)

    def await_operation(self: DaemonClient, _request_id: str, **kwargs: object) -> dict[str, object]:
        after_sequences.append((cast(int, kwargs["after_sequence"]), cast(int, kwargs["after_progress_sequence"])))
        return next(states)

    monkeypatch.setattr(DaemonClient, "await_operation", await_operation)
    seen: list[object] = []

    def broken_renderer(frame: object) -> None:
        seen.append(frame)
        raise RuntimeError("terminal renderer broke")

    result = DaemonClient(tmp_path / "daemon.sock").operation_to_completion(
        "maintenance.embeddings.backfill",
        {},
        archive_root=str(tmp_path),
        progress_callback=broken_renderer,
    )

    assert after_sequences == [(1, 0), (1, 1)]
    assert len(seen) == 1
    assert result is not None and result["outcome"] == "completed"
    assert result["result"]["result"] == {"done": 1, "pending": 0, "failed": 0}


_MUTATION: tuple[str, dict[str, object]] = ("mutation.session.delete.execute", {"authorization_ref": "ref-1"})


def _captured_operation_envelope(tmp_path: Path, operation: str, request_id: str) -> tuple[dict[str, object], str]:
    """Return a real daemon envelope rewritten into a typed ``failed`` answer for ``operation``.

    The authority fields come from the production stack, so the rewritten
    envelope passes every coherence check the client applies; only the
    operation identity and the terminal outcome change.
    """
    from polylogue.operations.daemon_protocol import daemon_operation_spec

    with running_daemon_operations(tmp_path / "archive") as stack:
        captured = stack.client.operation("status", {}, archive_root=str(stack.archive_root))
        archive_root = str(stack.archive_root)
    assert captured is not None
    spec = daemon_operation_spec(operation)
    assert spec is not None
    authority = cast(dict[str, object], captured["authority"])
    envelope = {
        **captured,
        "operation": operation,
        "request_id": request_id,
        "outcome": "failed",
        "result": None,
        "accepted_reference": None,
        "error": {"code": "internal_error", "detail": "handler raised after dispatch", "retryable": False},
        "authority": {**authority, "class": spec.authority.value, "fallback": spec.fallback.value},
    }
    return envelope, archive_root


@pytest.mark.parametrize(("operation", "payload"), [("status", {}), _MUTATION], ids=["read", "mutation"])
def test_a_typed_5xx_operation_envelope_is_an_explicit_failure_not_absence(
    tmp_path: Path, _short_uds_runtime_dir: Path, operation: str, payload: dict[str, object]
) -> None:
    """A daemon that answers 500 with a valid operation envelope reported a typed failure.

    Anti-vacuity: drop the ``500 <= status <= 599`` admission for protocol
    envelopes in ``DaemonClient._validate_operation_response`` and the read
    raises ``DaemonOperationProtocolError`` while the mutation raises
    ``DaemonMutationIndeterminateError``, sending the operator to recover a
    write whose failure the daemon already stated.
    """
    from polylogue.daemon_client import DaemonClient

    envelope, archive_root = _captured_operation_envelope(tmp_path, operation, "typed-5xx")
    socket_path = _short_uds_runtime_dir / "typed-5xx.sock"
    with _raw_unix_http_responder(socket_path, status=500, payload=envelope):
        answered = DaemonClient(socket_path, timeout_s=5).operation(
            operation, payload, archive_root=archive_root, request_id="typed-5xx"
        )

    assert answered is not None
    assert answered["outcome"] == "failed"
    assert cast(dict[str, object], answered["error"])["code"] == "internal_error"


@pytest.mark.parametrize(("operation", "payload"), [("status", {}), _MUTATION], ids=["read", "mutation"])
def test_an_untyped_5xx_is_never_treated_as_an_operation_answer(
    _short_uds_runtime_dir: Path, operation: str, payload: dict[str, object]
) -> None:
    """Strictness is kept for bodies that do not claim the operation protocol.

    Anti-vacuity: admit every 5xx body regardless of its protocol and neither
    call raises.
    """
    from polylogue.daemon_client import (
        DaemonClient,
        DaemonMutationIndeterminateError,
        DaemonOperationProtocolError,
    )

    socket_path = _short_uds_runtime_dir / "untyped-5xx.sock"
    error = DaemonOperationProtocolError if operation == "status" else DaemonMutationIndeterminateError
    with _raw_unix_http_responder(socket_path, status=500, payload={"error": "Internal Server Error"}):
        with pytest.raises(error):
            DaemonClient(socket_path, timeout_s=5).operation(operation, payload, archive_root="/archive")


@pytest.mark.parametrize("body", [{"error": "Not Found"}, {}], ids=["legacy-envelope", "empty-object"])
def test_an_older_daemon_without_the_operation_route_is_absence_for_reads(
    _short_uds_runtime_dir: Path, body: dict[str, object]
) -> None:
    """A daemon that predates ``/api/operation`` answers an untyped 404.

    A read falls back to the local reader (``None``); a write is never reported
    absent, because nothing proves an older process did not act on it.

    Anti-vacuity: remove the untyped-404 branch in ``DaemonClient.operation``
    and the read raises ``DaemonOperationProtocolError``, so an ordinary query
    crashes behind a stale daemon instead of falling back.
    """
    from polylogue.daemon_client import DaemonClient, DaemonMutationIndeterminateError

    read_socket = _short_uds_runtime_dir / "legacy-404-read.sock"
    with _raw_unix_http_responder(read_socket, status=404, payload=body):
        assert DaemonClient(read_socket, timeout_s=5).operation("status", {}, archive_root="/archive") is None
    write_socket = _short_uds_runtime_dir / "legacy-404-write.sock"
    with _raw_unix_http_responder(write_socket, status=404, payload=body):
        with pytest.raises(DaemonMutationIndeterminateError):
            DaemonClient(write_socket, timeout_s=5).operation(*_MUTATION, archive_root="/archive")


def test_a_404_claiming_the_operation_protocol_is_validated_strictly(_short_uds_runtime_dir: Path) -> None:
    """Only an untyped 404 means an older daemon; a typed one must be coherent.

    Anti-vacuity: treat every 404 as absence and this incoherent protocol
    envelope returns ``None`` instead of raising.
    """
    from polylogue.daemon_client import DaemonClient, DaemonOperationProtocolError
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    socket_path = _short_uds_runtime_dir / "typed-404.sock"
    with _raw_unix_http_responder(socket_path, status=404, payload={"protocol": DAEMON_OPERATION_PROTOCOL}):
        with pytest.raises(DaemonOperationProtocolError):
            DaemonClient(socket_path, timeout_s=5).operation("status", {}, archive_root="/archive")


@pytest.mark.parametrize(
    "operation,payload",
    [
        ("query.aggregate", {"mode": "count"}),
        ("read.chronicle", {"params": {"sort": "messages"}}),
        ("read.chronicle", {"params": {"sort": "date"}}),
    ],
)
def test_reads_have_no_implicit_request_or_transport_deadline(
    operation: str, payload: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Restoring any spec/scan timeout or the client fallback makes this fail."""
    from polylogue.daemon_client import DaemonClient

    client = DaemonClient(tmp_path / "daemon.sock")
    captured: list[tuple[object, object]] = []

    def request(_method: str, _path: str, body: dict[str, object], **kwargs: object) -> None:
        captured.append((body["deadline_ms"], kwargs["timeout_s"]))
        return None

    monkeypatch.setattr(client, "_request_json_response", request)
    assert client.operation(operation, payload, archive_root=str(tmp_path)) is None
    assert captured == [(None, None)]
    assert client.timeout_s == 0.1


def test_unbounded_response_does_not_inherit_the_clients_short_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drive the actual transport constructor; forwarding None alone is insufficient."""
    import errno

    from polylogue.daemon_client import DaemonClient

    observed: list[float | None] = []

    class AbsentConnection:
        connected = False

        def __init__(self, _path: Path, timeout: float | None) -> None:
            observed.append(timeout)

        def connect(self) -> None:
            raise FileNotFoundError(errno.ENOENT, "synthetic absent daemon")

        def close(self) -> None:
            pass

    monkeypatch.setattr("polylogue.daemon_client._UnixHTTPConnection", AbsentConnection)
    assert DaemonClient(tmp_path / "daemon.sock").operation("query.aggregate", {"mode": "count"}) is None
    assert observed == [None]


def test_an_explicit_dispatch_deadline_reaches_the_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A caller's ``deadline_ms`` is the request's deadline, not the derived scan one.

    Anti-vacuity (Codex P1, #5695): let ``_ask_daemon`` omit ``deadline_ms``
    and a one-second scan-shaped read goes out without its explicit deadline.
    """
    from types import SimpleNamespace

    from polylogue.cli import operation_kernel
    from polylogue.daemon_client import DaemonClient

    seen: list[object] = []

    def operation(self: DaemonClient, name: str, payload: dict[str, object], **kwargs: object) -> None:
        seen.append(kwargs.get("deadline_ms"))
        return None

    monkeypatch.setattr(DaemonClient, "operation", operation)
    request = operation_kernel.OperationRequest("read.chronicle", {"params": {"sort": "messages", "limit": 1}})
    with pytest.raises(operation_kernel.OperationUnavailableError):
        operation_kernel.dispatch(SimpleNamespace(), request, archive_root=tmp_path, deadline_ms=1000)

    assert seen == [1000]


def test_invalid_chronicle_payloads_reach_execution_for_their_typed_refusal() -> None:
    """The pre-dispatch classifiers never raise on a request execution will refuse.

    Anti-vacuity (Codex P2, #5695): catch only ``ValueError`` and a bogus
    sort's ``QuerySpecError`` escapes the classifier before execution.
    """
    from polylogue.operations.daemon_reads import read_is_archive_scan, requires_vector_snapshot

    payload = {"params": {"sort": "bogus"}}
    assert read_is_archive_scan("read.chronicle", payload) is False
    assert requires_vector_snapshot("read.chronicle", payload) is False


@pytest.mark.parametrize(
    ("operation", "payload", "bound", "daemon_bound"),
    [
        ("cli.query", {}, True, True),
        ("mutation.session.delete.preview", {"session_ids": ["sample"]}, True, True),
        ("mutation.session.delete.cancel", {"preview_ref": "preview"}, True, True),
        ("mutation.session.delete.execute", {"authorization_ref": "authorization"}, True, True),
        ("maintenance.backup", {"output_dir": "/synthetic/backups"}, False, True),
        (
            "maintenance.restore_verified_backup",
            {"backup_dir": "/synthetic/package", "destination": "/synthetic/new"},
            False,
            True,
        ),
        ("user.settings.get", {"setting_key": "subscription_tier"}, False, True),
        ("user.settings.list", {}, False, True),
        ("insights.hermes_health", {}, False, True),
        ("status", {}, False, False),
        ("operation.status", {"request_id": "original"}, False, False),
        ("operation.cancel", {"request_id": "original"}, False, False),
    ],
)
def test_connected_operation_binds_versions_without_a_discovery_exchange(
    monkeypatch: pytest.MonkeyPatch,
    _short_uds_runtime_dir: Path,
    operation: str,
    payload: dict[str, object],
    bound: bool,
    daemon_bound: bool,
) -> None:
    from polylogue.daemon_client import (
        DaemonClient,
        DaemonMutationIndeterminateError,
        DaemonOperationProtocolError,
        _UnixHTTPConnection,
    )
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
    from polylogue.version import POLYLOGUE_VERSION

    requests: list[dict[str, object]] = []
    original = _UnixHTTPConnection.request

    def record(self: _UnixHTTPConnection, method: str, path: str, **kwargs: object) -> None:
        requests.append(json.loads(cast(bytes, kwargs["body"])))
        original(self, method, path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_UnixHTTPConnection, "request", record)
    socket_path = _short_uds_runtime_dir / "versions.sock"
    # The peer deliberately provides no operation receipt. Observe the actual
    # request bytes without claiming its transport stub accepted any effect.
    with _raw_unix_http_responder(socket_path, status=200, payload={}):
        with pytest.raises((DaemonMutationIndeterminateError, DaemonOperationProtocolError)):
            DaemonClient(socket_path).operation(operation, payload, request_id="original")
    assert len(requests) == 1
    assert requests[0]["request_id"] == "original"
    assert requests[0]["index_schema_version"] == (INDEX_SCHEMA_VERSION if bound else None)
    assert requests[0]["daemon_version"] == (POLYLOGUE_VERSION if daemon_bound else None)


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("cli.query", {}),
        ("user.settings.get", {"setting_key": "subscription_tier"}),
        ("user.settings.list", {}),
        ("insights.hermes_health", {}),
        ("maintenance.backup", {"output_dir": "/synthetic/backups"}),
        ("maintenance.restore_verified_backup", {"backup_dir": "/synthetic/package", "destination": "/synthetic/new"}),
    ],
)
def test_explicit_operation_version_preconditions_are_preserved(
    monkeypatch: pytest.MonkeyPatch, _short_uds_runtime_dir: Path, operation: str, payload: dict[str, object]
) -> None:
    from polylogue.daemon_client import (
        DaemonClient,
        DaemonMutationIndeterminateError,
        DaemonOperationProtocolError,
        _UnixHTTPConnection,
    )

    requests: list[dict[str, object]] = []
    original = _UnixHTTPConnection.request

    def record(self: _UnixHTTPConnection, method: str, path: str, **kwargs: object) -> None:
        requests.append(json.loads(cast(bytes, kwargs["body"])))
        original(self, method, path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_UnixHTTPConnection, "request", record)
    socket_path = _short_uds_runtime_dir / "explicit.sock"
    with _raw_unix_http_responder(socket_path, status=200, payload={}):
        with pytest.raises((DaemonOperationProtocolError, DaemonMutationIndeterminateError)):
            DaemonClient(socket_path).operation(
                operation, payload, index_schema_version=999, daemon_version="selected-build"
            )
    assert len(requests) == 1
    assert requests[0]["index_schema_version"] == 999
    assert requests[0]["daemon_version"] == "selected-build"


def test_absent_daemon_operation_does_not_load_version_or_storage(_short_uds_runtime_dir: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from polylogue.daemon_client import DaemonClient; "
            "assert DaemonClient(Path(sys.argv[1])).operation('cli.query', {}) is None; "
            "assert 'polylogue.storage' not in sys.modules; assert 'polylogue.version' not in sys.modules",
            str(_short_uds_runtime_dir / "absent.sock"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("timeout_ms", [1, 2000, 30000])
def test_receipt_wait_binds_the_exchange_to_its_actual_wait_budget(
    timeout_ms: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.daemon_client import DaemonClient

    client = DaemonClient(tmp_path / "daemon.sock")
    captured: list[tuple[object, object, object]] = []

    def request(_method: str, _path: str, body: dict[str, object], **kwargs: object) -> None:
        payload = body["payload"]
        assert isinstance(payload, dict)
        captured.append((payload["timeout_ms"], body["deadline_ms"], kwargs["timeout_s"]))
        return None

    monkeypatch.setattr(client, "_request_json_response", request)
    assert client.await_operation("original-request", archive_root=str(tmp_path), timeout_ms=timeout_ms) is None
    assert captured == [(timeout_ms, timeout_ms, timeout_ms / 1000 + 1.0)]
    assert client.timeout_s == 0.1
