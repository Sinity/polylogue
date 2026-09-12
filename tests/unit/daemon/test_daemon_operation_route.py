"""Production UDS operation route contracts."""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
from pathlib import Path

import pytest

from polylogue.daemon.uds import MachineOperationHandler
from polylogue.daemon_client import DaemonClient, DaemonOperationRejectedError
from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
from polylogue.operations.mutation_transaction import MAX_MUTATION_PLAN_TARGETS, MutationPlan, MutationReceipt
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.storage_records import SessionBuilder

pytestmark = pytest.mark.uses_real_clock(
    "starts the real UDS listener and coordinator loop; wall-clock events bound socket and writer ownership waits"
)


def _seed_sessions(root: Path, *, count: int, title: str = "Operation route session") -> tuple[str, ...]:
    """Seed one fully bootstrapped synthetic archive before daemon startup."""

    session_ids: list[str] = []
    for number in range(count):
        builder = (
            SessionBuilder(root / "index.db", f"operation-{number}")
            .provider("codex")
            .title(title)
            .add_message(text=f"Synthetic daemon operation session {number}.")
        )
        builder.save()
        session_ids.append(builder.native_session_id())
    return tuple(session_ids)


def test_one_uds_operation_request_returns_canonical_read_without_health_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: add a health preflight and the patched client entrypoint fails."""

    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=2)

    def unexpected_health_probe(*args: object, **kwargs: object) -> object:
        raise AssertionError("operation client must not issue a health probe")

    monkeypatch.setattr(DaemonClient, "request_json", unexpected_health_probe)
    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        envelope = stack.client.operation(
            "cli.query",
            {"params": {"limit": 7}},
            archive_root=str(stack.archive_root),
        )

    assert envelope is not None
    assert envelope["outcome"] == "completed"
    assert envelope["readiness"]["ready"] is True
    assert envelope["authority"]["writes"] == "daemon-owned"
    assert envelope["result"]["total"] == len(session_ids)
    assert {item["id"] for item in envelope["result"]["items"]} == set(session_ids)


def test_repeated_daemon_query_uses_revision_scoped_result_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated production UDS read reuses the canonical result payload.

    Anti-vacuity: the second request still traverses the real operation route,
    but a patched canonical query body must only run once.  The cache is then
    invalidated explicitly, proving that freshness is a write boundary rather
    than a TTL guess.
    """
    from polylogue.operations import daemon_reads
    from polylogue.storage.search.cache import invalidate_search_cache

    invalidate_search_cache()
    calls = 0
    original = daemon_reads._query_payload

    def counted(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(daemon_reads, "_query_payload", counted)
    with running_daemon_operations(tmp_path / "archive") as stack:
        first = stack.client.operation("cli.query", {"params": {"limit": 1}}, archive_root=str(stack.archive_root))
        second = stack.client.operation("cli.query", {"params": {"limit": 1}}, archive_root=str(stack.archive_root))

    assert first is not None and second is not None
    assert first["result"] == second["result"]
    assert calls == 1


def test_machine_listener_uses_the_independent_operation_handler(tmp_path: Path) -> None:
    """Mutation: delegate machine requests through the browser handler and this fails."""

    def seed(root: Path) -> None:
        _seed_sessions(root, count=1)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        assert stack.server.RequestHandlerClass is MachineOperationHandler
        assert MachineOperationHandler.__bases__[0].__name__ == "BaseHTTPRequestHandler"
        envelope = stack.client.operation("status", {}, archive_root=str(stack.archive_root))

    assert envelope is not None
    assert envelope["result"]["total_sessions"] == 1


def test_fresh_runtime_prepares_tier_journals_before_first_snapshot_and_mutation(tmp_path: Path) -> None:
    """Mutation: omit writer startup journal activation and the first audited preview self-locks."""
    import sqlite3

    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)
        # Source/user/audit can be untouched rollback-journal tiers in a fresh
        # archive. Startup, never the operation reader, must activate them.
        for tier in ("source", "user", "audit"):
            with sqlite3.connect(root / f"{tier}.db") as connection:
                connection.execute("PRAGMA journal_mode=DELETE")

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        read = stack.client.operation("cli.query", {"params": {}}, archive_root=str(stack.archive_root))
        assert read is not None and read["outcome"] == "completed"
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview", {"session_ids": list(ids)}, archive_root=str(stack.archive_root)
        )
        assert preview is not None and preview["outcome"] == "completed"
        assert preview["result"]["session_ids"] == list(ids)


def test_authentication_refusal_is_not_an_indeterminate_mutation(tmp_path: Path) -> None:
    """Mutation: treat the ingress 401 as a lost receipt and the typed refusal disappears."""
    with running_daemon_operations(tmp_path / "archive") as stack:
        stack.server.auth_token = "synthetic-test-credential"
        with pytest.raises(DaemonOperationRejectedError, match="unauthorized"):
            stack.client.operation("mutation.session.delete.preview", {"session_ids": ["codex:absent"]})
        assert not stack.runtime._exchanges


@pytest.mark.parametrize("operation", ["status", "mutation.session.delete.preview"])
def test_connection_saturation_refuses_before_acceptance_and_recovers(tmp_path: Path, operation: str) -> None:
    """An empty 503 or bypassed admission loses the explicit no-execution guarantee."""
    with running_daemon_operations(tmp_path / "archive") as stack:
        # Reserve the real ingress budget without a timing-dependent fleet of
        # slow sockets. The next request still traverses the production listener.
        for _ in range(stack.server.request_queue_size):
            assert stack.server._connections.acquire(blocking=False)
        try:
            payload: dict[str, object] = {} if operation == "status" else {"session_ids": ["codex:absent"]}
            with pytest.raises(DaemonOperationRejectedError) as rejected:
                stack.client.operation(operation, payload)
            assert rejected.value.outcome == "connection_backpressure"
            assert not stack.runtime._exchanges
        finally:
            for _ in range(stack.server.request_queue_size):
                stack.server._connections.release()

        recovered = stack.client.operation("status", {}, archive_root=str(stack.archive_root))
        assert recovered is not None and recovered["outcome"] == "completed"


def test_changed_intent_cannot_reuse_a_durable_request_id(tmp_path: Path) -> None:
    """Mutation: ignore the durable fingerprint and a changed selection inherits prior authority."""
    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=2)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        first = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": [ids[0]]},
            archive_root=str(stack.archive_root),
            request_id="stable-preview-intent",
        )
        assert first is not None
        assert first["outcome"] == "completed"
        changed = stack.client.operation(
            "mutation.session.delete.preview",
            {"session_ids": [ids[1]]},
            archive_root=str(stack.archive_root),
            request_id="stable-preview-intent",
        )
        assert changed is not None and changed["outcome"] == "rejected"
        assert changed.get("accepted_reference") is None
        assert all(stack.session_exists(session_id) for session_id in ids)


def test_operation_route_bounds_the_real_canonical_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: bypass the UDS response bound and the oversized canonical rows escape."""

    import json

    import polylogue.daemon.uds as uds

    def seed(root: Path) -> None:
        _seed_sessions(root, count=8)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        unbounded = stack.client.operation("cli.query", {"params": {"limit": 8}}, archive_root=str(stack.archive_root))
        assert unbounded is not None and unbounded["outcome"] == "completed"
        assert len(json.dumps(unbounded, separators=(",", ":")).encode()) > 4096
        monkeypatch.setattr(uds, "MAX_OPERATION_RESULT_BYTES", 4096)
        envelope = stack.client.operation(
            "cli.query",
            {"params": {"limit": 8}},
            archive_root=str(stack.archive_root),
        )

    assert envelope is not None
    assert envelope["outcome"] == "failed"
    assert envelope["result"] is None
    assert envelope["error"]["code"] == "result_too_large"


def test_kernel_authenticated_uid_reference_survives_client_and_daemon_restart(tmp_path: Path) -> None:
    """Mutation: principal contains a PID/random token and durable retry changes authority."""
    root = tmp_path / "archive"
    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)

    with running_daemon_operations(root, seed_archive=seed) as first:
        prepared = first.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(ids)},
            archive_root=str(root),
            request_id="durable-local-preview",
        )
    with running_daemon_operations(root) as restarted:
        recovered = restarted.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(ids)},
            archive_root=str(root),
            request_id="durable-local-preview",
        )

    assert prepared is not None and recovered is not None
    assert prepared["accepted_reference"]["principal_ref"] == f"daemon:unix:uid:{os.getuid()}"
    assert recovered["accepted_reference"] == prepared["accepted_reference"]
    assert recovered["result"] == prepared["result"]


def test_uds_refuses_when_kernel_peer_credentials_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: fall back to a guessed local principal and this read executes."""
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    original = socket.socket.getsockopt

    def unavailable(sock: socket.socket, level: int, name: int, *args: object) -> object:
        if level == socket.SOL_SOCKET and name == socket.SO_PEERCRED:
            raise OSError("synthetic unavailable peer credentials")
        return original(sock, level, name, *args)

    with running_daemon_operations(tmp_path / "archive") as stack:
        monkeypatch.setattr(socket.socket, "getsockopt", unavailable)
        refused = stack.client.request_json(
            "POST",
            "/api/operation",
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "request_id": "unavailable-peer-credentials",
                "operation": "completion",
                "payload": {"kind": "field"},
            },
            accepted_statuses=frozenset({401}),
        )
    assert refused is not None
    assert refused["outcome"] == "rejected"
    assert refused["error"]["code"] == "peer_authentication_unavailable"


def test_disconnected_queued_control_releases_its_compute_admission(tmp_path: Path) -> None:
    """A pre-acceptance control disconnect must not retain a future worker slot.

    Anti-vacuity: omitting the control cancellation handle from the scheduler
    leaves the queued preview exchange and its reservation until the blockers
    release, even though the real UDS peer has already disconnected.
    """
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

    entered = threading.Semaphore(0)
    release = threading.Event()

    def block_worker() -> None:
        entered.release()
        assert release.wait(timeout=5)

    with running_daemon_operations(tmp_path / "archive") as stack:
        blockers = [stack.execution_kernel.submit(block_worker) for _ in range(2)]
        assert all(entered.acquire(timeout=2) for _ in blockers)
        request_id = "disconnected-queued-control"
        body = json.dumps(
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "request_id": request_id,
                "operation": "mutation.session.delete.preview",
                "payload": {"session_ids": ["codex:absent"]},
                "archive_root": str(stack.archive_root),
            },
            separators=(",", ":"),
        ).encode()
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            peer.connect(str(stack.socket_path))
            peer.sendall(
                b"POST /api/operation HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n" + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(lambda: request_id in stack.runtime._exchanges, timeout=2)
            peer.close()
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(lambda: request_id not in stack.runtime._exchanges, timeout=2)
            assert stack.execution_kernel.snapshot().used_units == len(blockers)
        finally:
            release.set()
            peer.close()
            for blocker in blockers:
                blocker.future.result(timeout=2)


def test_cancelled_long_delete_retains_writer_until_blocked_apply_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: release the admitted writer on cancellation and the later preview completes early."""

    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=MAX_MUTATION_PLAN_TARGETS + 1)

    entered_apply = threading.Event()
    release_apply = threading.Event()
    original_apply = SessionDeleteActuator.apply

    def blocked_apply(self: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs) -> MutationReceipt:
        entered_apply.set()
        if not release_apply.wait(timeout=5):
            raise TimeoutError("test did not release the admitted delete actuator")
        return original_apply(self, plan, args)

    monkeypatch.setattr(SessionDeleteActuator, "apply", blocked_apply)
    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(stack.archive_root),
        )
        assert preview is not None
        preview_result = preview["result"]
        authorization = stack.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_refs": preview_result["preview_refs"]},
            archive_root=str(stack.archive_root),
        )
        assert authorization is not None
        execute_request_id = "blocked-delete"
        execute_result: queue.Queue[dict[str, object]] = queue.Queue()

        def execute() -> None:
            client = DaemonClient(stack.socket_path, timeout_s=5)
            response = client.operation(
                "mutation.session.delete.execute",
                {"authorization_refs": authorization["result"]["authorization_refs"]},
                archive_root=str(stack.archive_root),
                request_id=execute_request_id,
            )
            assert response is not None
            execute_result.put(response)

        execute_thread = threading.Thread(target=execute, name="blocked-delete-client")
        execute_thread.start()
        later_result: queue.Queue[dict[str, object]] = queue.Queue()
        later_completed = threading.Event()

        def later_writer() -> None:
            client = DaemonClient(stack.socket_path, timeout_s=5)
            response = client.operation(
                "mutation.session.delete.preview",
                {"session_ids": [session_ids[-1]]},
                archive_root=str(stack.archive_root),
                request_id="later-preview",
            )
            assert response is not None
            later_result.put(response)
            later_completed.set()

        later_thread: threading.Thread | None = None
        try:
            assert entered_apply.wait(timeout=2)
            accepted = execute_result.get(timeout=2)
            assert accepted["outcome"] == "accepted"

            status = stack.client.operation(
                "operation.status",
                {"request_id": execute_request_id},
                archive_root=str(stack.archive_root),
            )
            cancellation = stack.client.cancel(execute_request_id, archive_root=str(stack.archive_root))
            assert status is not None
            assert status["result"]["outcome"] in {"accepted", "running"}
            assert cancellation is not None

            later_thread = threading.Thread(target=later_writer, name="later-delete-preview")
            later_thread.start()
            assert later_result.empty()
            assert not later_completed.wait(timeout=0.15)
        finally:
            release_apply.set()
            execute_thread.join(timeout=3)
            if later_thread is not None:
                later_thread.join(timeout=3)

        assert not execute_thread.is_alive()
        assert later_thread is not None
        assert not later_thread.is_alive()
        final = stack.client.await_operation(execute_request_id, archive_root=str(stack.archive_root), timeout_ms=2_000)
        assert final is not None
        state = final["result"]
        assert state["outcome"] == "cancelled"
        assert state["completed_chunks"] == 1
        assert state["not_attempted"] == [1]
        assert all(not stack.session_exists(session_id) for session_id in session_ids[:-1])
        assert stack.session_exists(session_ids[-1])
