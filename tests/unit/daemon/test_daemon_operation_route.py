"""Production UDS operation route contracts."""

from __future__ import annotations

import json
import os
import queue
import socket
import sqlite3
import threading
from copy import deepcopy
from dataclasses import replace
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


def test_restart_recovers_indeterminate_mutation_without_replaying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable unknown effect is returned after restart, never submitted again."""
    root = tmp_path / "archive"
    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=1)

    apply_calls = 0
    original_apply = SessionDeleteActuator.apply

    def apply_once_as_indeterminate(
        actuator: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs
    ) -> MutationReceipt:
        nonlocal apply_calls
        apply_calls += 1
        # The archive mutation really happens, but the worker loses the
        # outcome at the domain boundary.  Audit therefore persists unknown,
        # which is the restart case this route must recover.
        return replace(original_apply(actuator, plan, args), status="unknown", detail="synthetic lost outcome")

    monkeypatch.setattr(SessionDeleteActuator, "apply", apply_once_as_indeterminate)
    with running_daemon_operations(root, seed_archive=seed) as first:
        preview = first.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(root),
            request_id="indeterminate-preview",
        )
        assert preview is not None
        authorization = first.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_refs": preview["result"]["preview_refs"]},
            archive_root=str(root),
            request_id="indeterminate-authorize",
        )
        assert authorization is not None
        lost = first.client.operation_to_completion(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization["result"]["authorization_refs"]},
            archive_root=str(root),
            request_id="indeterminate-execute",
        )
        assert lost is not None and lost["outcome"] == "indeterminate"
        accepted_reference = lost["accepted_reference"]

    assert apply_calls == 1
    with running_daemon_operations(root) as restarted:
        # Startup recovery conservatively adjudicates this synthetic fixture
        # from the already-deleted target. Re-introduce the persisted unknown
        # outcome after startup so the route is tested against a durable
        # indeterminate record, exactly as a crashed domain writer leaves it.
        with sqlite3.connect(root / "audit.db") as connection:
            connection.execute(
                """
                UPDATE operation_runs
                SET unknown_count = 1, unknown_reason = ?
                WHERE operation_id IN (
                    SELECT operation_id FROM machine_request_parts
                    WHERE request_id = ? AND operation_id IS NOT NULL
                )
                """,
                ("synthetic lost outcome", "indeterminate-execute"),
            )
            connection.commit()
        recovered = restarted.client.operation(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization["result"]["authorization_refs"]},
            archive_root=str(root),
            request_id="indeterminate-execute",
        )
        assert recovered is not None
        assert recovered["outcome"] == "indeterminate"
        assert recovered["accepted_reference"] == accepted_reference
        assert recovered["result"]["reference"] == accepted_reference

    assert apply_calls == 1
    assert all(not restarted.session_exists(session_id) for session_id in session_ids)


def test_disconnected_after_durable_acceptance_recovers_without_replaying_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost post-acceptance UDS response recovers one durable execution.

    The raw peer is closed only after the actuator has entered, which proves
    that the acceptance record exists before transport loss.  A retry with the
    same request identity must observe that record, and a restarted daemon must
    return its terminal receipt without invoking the actuator again.
    """
    root = tmp_path / "archive"
    session_ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal session_ids
        session_ids = _seed_sessions(root, count=1)

    entered_apply = threading.Event()
    release_apply = threading.Event()
    apply_calls = 0
    original_apply = SessionDeleteActuator.apply

    def blocked_apply(self: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs) -> MutationReceipt:
        nonlocal apply_calls
        apply_calls += 1
        entered_apply.set()
        if not release_apply.wait(timeout=5):
            raise TimeoutError("test did not release the accepted delete actuator")
        return original_apply(self, plan, args)

    monkeypatch.setattr(SessionDeleteActuator, "apply", blocked_apply)
    request_id = "disconnect-after-acceptance"
    accepted_reference: dict[str, object]
    authorization_refs: list[str]
    with running_daemon_operations(root, seed_archive=seed) as stack:
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
            archive_root=str(root),
        )
        assert preview is not None
        authorization = stack.client.operation_to_completion(
            "mutation.session.delete.authorize",
            {"preview_refs": preview["result"]["preview_refs"]},
            archive_root=str(root),
        )
        assert authorization is not None
        authorization_refs = list(authorization["result"]["authorization_refs"])

        from polylogue.operations.daemon_protocol import DaemonOperationRequest

        request = DaemonOperationRequest(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization_refs},
            archive_root=str(root),
            request_id=request_id,
            deadline_ms=10_000,
        )
        body = json.dumps(request.to_dict(), separators=(",", ":")).encode()
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            peer.connect(str(stack.socket_path))
            peer.sendall(
                b"POST /api/operation HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n" + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
            # SessionDeleteActuator.apply runs only after the durable
            # accept_execution_batch transition has committed.
            assert entered_apply.wait(timeout=2)
        finally:
            peer.close()

        recovered = stack.client.operation(
            request.operation,
            request.payload,
            archive_root=request.archive_root,
            request_id=request.request_id,
            deadline_ms=request.deadline_ms,
        )
        assert recovered is not None
        assert recovered["outcome"] == "running"
        accepted_reference = recovered["accepted_reference"]
        assert recovered["result"]["reference"] == accepted_reference
        assert recovered["result"]["completed_chunks"] == 0

        try:
            release_apply.set()
            terminal = stack.client.operation_to_completion(
                request.operation,
                request.payload,
                archive_root=str(root),
                request_id=request_id,
            )
            assert terminal is not None
            assert terminal["outcome"] == "completed"
            assert terminal["accepted_reference"] == accepted_reference
            assert terminal["result"]["reference"] == accepted_reference
            assert terminal["result"]["completed_chunks"] == 1
        finally:
            release_apply.set()

        assert apply_calls == 1
        assert all(not stack.session_exists(session_id) for session_id in session_ids)

    with running_daemon_operations(root) as restarted:
        replay = restarted.client.operation(
            request.operation,
            request.payload,
            archive_root=str(root),
            request_id=request.request_id,
            deadline_ms=request.deadline_ms,
        )
        assert replay is not None
        assert replay["outcome"] == "completed"
        assert replay["accepted_reference"] == accepted_reference
        assert replay["result"]["reference"] == accepted_reference
        assert apply_calls == 1


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
            assert status is not None
            assert status["result"]["outcome"] in {"accepted", "running"}
            from time import monotonic

            started = monotonic()
            timed_out = stack.client.operation(
                "operation.await",
                {"request_id": execute_request_id, "after_sequence": status["result"]["sequence"], "timeout_ms": 2_000},
                archive_root=str(stack.archive_root),
                deadline_ms=25,
            )
            assert timed_out is not None and timed_out["outcome"] == "timed-out"
            assert monotonic() - started < 1.0
            assert not release_apply.is_set()

            from polylogue.archive.query.execution_control import QueryExecutionContext
            from polylogue.operations.daemon_protocol import DaemonOperationRequest
            from polylogue.operations.mutation_transaction import MutationPrincipal
            from polylogue.operations.operation_context import OperationControlResult

            waiter_entered, waiter_released = threading.Event(), threading.Event()
            control = stack.runtime.control

            def observe_waiter(
                request: DaemonOperationRequest,
                principal: MutationPrincipal,
                archive_identity: str,
                *,
                execution_context: QueryExecutionContext | None = None,
            ) -> OperationControlResult:
                observing = request.request_id == "disconnected-await"
                if observing:
                    waiter_entered.set()
                try:
                    return control(request, principal, archive_identity, execution_context=execution_context)
                finally:
                    if observing:
                        waiter_released.set()

            monkeypatch.setattr(stack.runtime, "control", observe_waiter)
            body = json.dumps(
                DaemonOperationRequest(
                    "operation.await",
                    {
                        "request_id": execute_request_id,
                        "after_sequence": status["result"]["sequence"],
                        "timeout_ms": 30_000,
                    },
                    request_id="disconnected-await",
                ).to_dict()
            ).encode()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
                peer.connect(str(stack.socket_path))
                peer.sendall(
                    (
                        "POST /api/operation HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(body)}\r\n\r\n"
                    ).encode()
                    + body
                )
                assert waiter_entered.wait(timeout=2)
            assert waiter_released.wait(timeout=2), "disconnected await retained its handler until the 30s deadline"
            assert not release_apply.is_set()
            cancellation = stack.client.cancel(execute_request_id, archive_root=str(stack.archive_root))
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
        assert state["stop_reason"] == "cancelled"
        assert state["reference"] == accepted["accepted_reference"]
        assert all(not stack.session_exists(session_id) for session_id in session_ids[:-1])
        assert stack.session_exists(session_ids[-1])

    with running_daemon_operations(tmp_path / "archive") as restarted:
        replay = restarted.client.operation(
            "mutation.session.delete.execute",
            {"authorization_refs": authorization["result"]["authorization_refs"]},
            archive_root=str(restarted.archive_root),
            request_id=execute_request_id,
        )
        assert replay is not None and replay["outcome"] == "cancelled"
        assert replay["accepted_reference"] == accepted["accepted_reference"]
        assert replay["result"]["not_attempted"] == [1]
        assert restarted.session_exists(session_ids[-1])


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_code"),
    [
        ("partial", 400, "invalid_request"),
        ("malformed", 400, "invalid_request"),
        ("duplicate-json", 400, "invalid_request"),
        ("nested-json", 400, "invalid_request"),
        ("nonfinite", 400, "invalid_request"),
        ("wrong-protocol", 400, "invalid_request"),
        ("operation-bound", 400, "invalid_request"),
        ("whitespace-bound", 413, "request_too_large"),
        ("transport-bound", 413, "request_too_large"),
        ("duplicate-length", 400, "invalid_framing"),
        ("transfer-encoding", 400, "invalid_framing"),
        ("wrong-type", 415, "unsupported_media_type"),
        ("wrong-method", 405, "method_not_allowed"),
        ("unknown-method", 501, "invalid_http_request"),
        ("wrong-version", 505, "unsupported_http_version"),
        ("browser-route", 404, "operation_endpoint_required"),
    ],
)
def test_machine_ingress_faults_refuse_without_dispatch_and_release_connection(
    tmp_path: Path,
    case: str,
    expected_status: int,
    expected_code: str,
) -> None:
    """Removing an ingress guard admits invalid input or leaves no typed refusal."""
    from http.client import HTTPResponse

    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL, MAX_DECLARED_OPERATION_BODY_BYTES

    errors: queue.SimpleQueue[str] = queue.SimpleQueue()
    with running_daemon_operations(tmp_path / "archive", server_error_sink=errors) as stack:
        body = json.dumps(
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "operation": "status",
                "request_id": "ingress-law",
                "payload": {},
            }
        ).encode()
        method, path, version = "POST", "/api/operation", "HTTP/1.1"
        content_type = "application/json"
        extra = ""
        declared_length = len(body)
        if case == "partial":
            declared_length += 1
        elif case == "malformed":
            body = b"{"
        elif case == "duplicate-json":
            body = body[:-1] + b',"operation":"status"}'
        elif case == "nested-json":
            body = b"[" * 2000 + b"0" + b"]" * 2000
        elif case == "nonfinite":
            body = body.replace(b'"payload": {}', b'"payload": {"value": NaN}')
        elif case == "wrong-protocol":
            body = body.replace(DAEMON_OPERATION_PROTOCOL.encode(), b"unknown/v99")
        elif case == "operation-bound":
            body = json.dumps(
                {
                    "protocol": DAEMON_OPERATION_PROTOCOL,
                    "operation": "completion",
                    "request_id": "ingress-law",
                    "payload": {"incomplete": "x" * (65 * 1024)},
                }
            ).encode()
        elif case == "transport-bound":
            declared_length = MAX_DECLARED_OPERATION_BODY_BYTES + 1
        elif case == "whitespace-bound":
            body += b" " * (65 * 1024)
        elif case == "duplicate-length":
            extra = f"Content-Length: {len(body)}\r\n"
        elif case == "transfer-encoding":
            extra = "Transfer-Encoding: chunked\r\n"
        elif case == "wrong-type":
            content_type = "text/plain"
        elif case == "wrong-method":
            method = "GET"
        elif case == "unknown-method":
            method = "TRACE"
        elif case == "wrong-version":
            version = "HTTP/1.0"
        elif case == "browser-route":
            path = "/api/health"
        if case not in {"partial", "transport-bound"}:
            declared_length = len(body)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(2)
            peer.connect(str(stack.socket_path))
            peer.sendall(
                (
                    f"{method} {path} {version}\r\nHost: localhost\r\nContent-Type: {content_type}\r\n"
                    f"Content-Length: {declared_length}\r\n{extra}\r\n"
                ).encode()
                + body
            )
            if case == "partial":
                peer.shutdown(socket.SHUT_WR)
            with HTTPResponse(peer) as response:
                response.begin()
                payload = json.loads(response.read())
                assert response.status == expected_status
                assert payload["outcome"] == "rejected"
                assert payload["error"]["code"] == expected_code
        assert not stack.runtime._exchanges
        assert errors.empty()
        recovered = stack.client.operation("status", {})
        assert recovered is not None and recovered["outcome"] == "completed"


@pytest.mark.parametrize("field", ["generation", "schemas", "archive", "served-by", "timing", "degraded", "fallback"])
def test_client_refuses_incoherent_authority_from_a_real_operation(
    tmp_path: Path,
    field: str,
) -> None:
    """Each mutant changes one copy of the observed authority without its peer."""
    from polylogue.daemon_client import DaemonOperationProtocolError
    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    with running_daemon_operations(tmp_path / "archive") as stack:
        request = DaemonOperationRequest("status", {}, request_id="authority-law")
        response = stack.client.operation("status", {}, request_id="authority-law")
        assert response is not None
        changed = deepcopy(response)
        if field == "generation":
            changed["generation"]["id"] = "stale-generation"
        elif field == "schemas":
            changed["schema_versions"]["source"] += 1
        elif field == "archive":
            changed["archive"]["archive_identity"] = "other-archive"
        elif field == "served-by":
            changed["served_by"]["identity"] = "other-server"
        elif field == "timing":
            changed["authority_snapshot"]["queue_ms"] += 1
        elif field == "degraded":
            changed["readiness"]["degraded_components"] = ["missing-source"]
        elif field == "fallback":
            changed["authority"]["fallback"] = "never"
        with pytest.raises(DaemonOperationProtocolError, match="incoherent"):
            DaemonClient._validate_operation_response(request, 200, changed)


def test_control_result_metadata_comes_from_the_durable_receipt_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using admission-time metadata after a recovery read returns obsolete schema evidence."""
    from dataclasses import replace

    import polylogue.operations.daemon_execution as execution
    from polylogue.operations.operation_context import OperationControlRead, observe_control_authority

    ids: tuple[str, ...] = ()

    def seed(root: Path) -> None:
        nonlocal ids
        ids = _seed_sessions(root, count=1)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        accepted = stack.client.operation_to_completion(
            "mutation.session.delete.preview",
            {"session_ids": list(ids)},
            archive_root=str(stack.archive_root),
            request_id="receipt-authority",
        )
        assert accepted is not None and accepted["outcome"] == "completed"

        def earlier_observation(root: Path) -> OperationControlRead:
            snapshot = observe_control_authority(root)
            return replace(
                snapshot, schema_versions={**snapshot.schema_versions, "source": snapshot.schema_versions["source"] - 1}
            )

        monkeypatch.setattr(execution, "observe_control_authority", earlier_observation)
        recovered = stack.client.operation(
            "operation.await",
            {"request_id": "receipt-authority", "after_sequence": 0, "timeout_ms": 100},
            archive_root=str(stack.archive_root),
        )
        assert recovered is not None and recovered["outcome"] == "completed"
        assert recovered["generation"]["id"] == accepted["generation"]["id"]
        assert recovered["schema_versions"] == {tier: accepted["schema_versions"][tier] for tier in ("source", "audit")}
        assert recovered["result"]["reference"] == accepted["accepted_reference"]
