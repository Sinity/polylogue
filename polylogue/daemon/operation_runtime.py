"""Resident operation ownership and event-driven completion over audit references."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic, time

from polylogue.archive.query.execution_control import QueryExecutionContext
from polylogue.daemon.execution import BoundedComputeAdapter, CancellationHandle, DaemonBackpressureError
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
from polylogue.operations.audit import (
    AuditContinuityPendingError,
    AuditRepository,
    MachineRequestBinding,
    MachineRequestConflictError,
)
from polylogue.operations.daemon_execution import execute_operation, operation_envelope
from polylogue.operations.daemon_protocol import (
    DaemonAuthority,
    DaemonOperationEnvelope,
    DaemonOperationRequest,
    daemon_operation_spec,
)
from polylogue.operations.daemon_reads import DaemonReadDependencies
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.operations.operation_context import OperationContext, PinnedOperationRead


class BeforeAcceptanceCancelledError(RuntimeError):
    """Cancellation won the lock before durable prepare could begin."""


@dataclass(slots=True)
class _Exchange:
    request: DaemonOperationRequest
    context: OperationContext
    deadline: float
    deadline_unix_ms: int
    future: Future[DaemonOperationEnvelope] | None = None
    cancellation: CancellationHandle = field(default_factory=CancellationHandle)
    acceptance_started: bool = False
    snapshot: PinnedOperationRead | None = None
    binding: MachineRequestBinding | None = None
    queue_ms: int = 0
    started_at: float = field(default_factory=monotonic)


class DaemonOperationRuntime:
    def __init__(
        self,
        archive_root: Path,
        *,
        write_bridge: DaemonWriteThreadBridge,
        execution_kernel: BoundedComputeAdapter,
        read_dependencies: DaemonReadDependencies | None = None,
        read_dependencies_factory: Callable[[], DaemonReadDependencies] | None = None,
    ) -> None:
        self.archive_root = archive_root.resolve()
        self._bridge = write_bridge
        self._kernel = execution_kernel
        self._read_dependencies = read_dependencies
        self._read_dependencies_factory = read_dependencies_factory
        self._condition = threading.Condition(threading.RLock())
        self._exchanges: dict[str, _Exchange] = {}

    def publication_guard(self) -> AbstractContextManager[None]:
        return self._bridge.hold("operation.pin-read")

    def run_write(self, name: str, work: Callable[[], DaemonOperationEnvelope]) -> DaemonOperationEnvelope:
        return self._bridge.run_sync_with_timeout(f"operation.{name}", None, work)

    def observe_snapshot(self, request: DaemonOperationRequest, snapshot: PinnedOperationRead) -> None:
        with self._condition:
            exchange = self._exchanges[str(request.request_id)]
            exchange.snapshot = snapshot
            exchange.binding = MachineRequestBinding(
                snapshot.identity.authority_identity_digest,
                str(request.request_id),
                exchange.context.principal.actor_ref,
                request.fingerprint,
                request.operation,
            )
            spec = daemon_operation_spec(request.operation)
            if spec is not None and spec.authority is DaemonAuthority.READ:
                exchange.cancellation.add_listener(snapshot.archive.interrupt_reads)

    def request_deadline_unix_ms(self, request: DaemonOperationRequest) -> int:
        return self._exchanges[str(request.request_id)].deadline_unix_ms

    def stop_reason(self, request: DaemonOperationRequest) -> str | None:
        exchange = self._exchanges[str(request.request_id)]
        if exchange.cancellation.cancelled:
            return "cancelled"
        return "deadline" if monotonic() >= exchange.deadline else None

    def _notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def audit_for_request(self, request: DaemonOperationRequest, context: OperationContext) -> AuditRepository:
        exchange = self._exchanges[str(request.request_id)]

        def before_prepare() -> None:
            with self._condition:
                if exchange.cancellation.cancelled or monotonic() >= exchange.deadline:
                    raise BeforeAcceptanceCancelledError("operation cancelled before durable acceptance")
                exchange.acceptance_started = True

        return AuditRepository(
            context.archive_root / "audit.db",
            attempt_owner_id=AuditRepository.current_process_attempt_owner(),
            before_machine_prepare=before_prepare,
            on_commit=self._notify,
        )

    def _durable(self, exchange: _Exchange) -> dict[str, object] | None:
        if exchange.binding is None:
            return None
        audit = AuditRepository.for_archive_root(self.archive_root)
        try:
            with audit.settled_machine_read():
                return audit.machine_request(exchange.binding)
        except AuditContinuityPendingError:
            return None

    def _recovery_state(self, record: dict[str, object]) -> dict[str, object]:
        audit = AuditRepository.for_archive_root(self.archive_root)
        try:
            with audit.settled_machine_read():
                binding = MachineRequestBinding(
                    **{
                        key: str(record[key])
                        for key in (
                            "archive_identity",
                            "request_id",
                            "principal_ref",
                            "fingerprint",
                            "operation_name",
                        )
                    }
                )
                current = audit.machine_request(binding)
                assert current is not None
                return machine_request_state(audit, current)
        except AuditContinuityPendingError:
            return {"outcome": "indeterminate", "sequence": 0, "reference": record}

    def _pending_envelope(
        self, exchange: _Exchange, *, outcome: str, record: dict[str, object] | None = None
    ) -> dict[str, object]:
        return operation_envelope(
            exchange.request,
            exchange.context,
            snapshot=exchange.snapshot,
            outcome=outcome,
            reference=record,
            queue_ms=exchange.queue_ms,
            started_at=exchange.started_at,
            result=self._recovery_state(record) if record is not None else None,
        ).to_dict()

    def call(
        self,
        request: DaemonOperationRequest,
        principal: MutationPrincipal,
        *,
        started_at: float | None = None,
        client_disconnect: CancellationHandle | None = None,
    ) -> dict[str, object]:
        started = monotonic() if started_at is None else started_at
        spec = daemon_operation_spec(request.operation)
        if spec is None:
            raise ValueError("operation is not declared")
        dependencies = (
            self._read_dependencies_factory()
            if self._read_dependencies_factory is not None
            else self._read_dependencies
        )
        dependencies = replace(dependencies or DaemonReadDependencies(), status_now_ms=int(time() * 1000))
        deadline = started + min(spec.deadline_s, (request.deadline_ms or int(spec.deadline_s * 1000)) / 1000)
        read_control = (
            QueryExecutionContext(
                call_id=str(request.request_id),
                query_ref=request.fingerprint,
                deadline_monotonic=deadline,
                owner_ref=principal.actor_ref,
            )
            if spec.authority is DaemonAuthority.READ
            else None
        )
        context = OperationContext(self.archive_root, principal, "daemon", self, dependencies, read_control)
        if request.operation.startswith("operation."):
            return execute_operation(request, context).to_dict()
        request_id = str(request.request_id)
        peer_closed = False
        with self._condition:
            exchange = self._exchanges.get(request_id)
            if exchange is not None:
                if exchange.request.fingerprint != request.fingerprint or exchange.context.principal != principal:
                    return operation_envelope(
                        request,
                        context,
                        outcome="rejected",
                        error={"code": "request_identity_conflict", "retryable": False},
                    ).to_dict()
            else:
                if len(self._exchanges) >= 64:
                    return operation_envelope(
                        request,
                        context,
                        outcome="rejected",
                        error={
                            "code": "operation_capacity",
                            "retryable": True,
                        },
                    ).to_dict()
                exchange = _Exchange(
                    request,
                    context,
                    deadline,
                    int((time() + max(0, deadline - monotonic())) * 1000),
                    started_at=started,
                )
                self._exchanges[request_id] = exchange
                if read_control is not None:
                    exchange.cancellation.add_listener(read_control.cancel)

                def work() -> DaemonOperationEnvelope:
                    exchange.queue_ms = max(0, int((monotonic() - started) * 1000))
                    if exchange.cancellation.cancelled:
                        raise BeforeAcceptanceCancelledError("operation cancelled before dispatch")
                    return execute_operation(request, context)

                try:
                    scheduled = self._kernel.submit(
                        work,
                        admission_class="interactive-read" if spec.authority is DaemonAuthority.READ else "control",
                        cancellation=exchange.cancellation if spec.authority is DaemonAuthority.READ else None,
                    )
                except DaemonBackpressureError:
                    self._exchanges.pop(request_id)
                    return operation_envelope(
                        request,
                        context,
                        outcome="rejected",
                        error={
                            "code": "compute_backpressure",
                            "retryable": True,
                        },
                    ).to_dict()
                exchange.future = scheduled.future

                def settled(_future: Future[object]) -> None:
                    with self._condition:
                        if self._exchanges.get(request_id) is exchange:
                            self._exchanges.pop(request_id)
                        self._condition.notify_all()

                exchange.future.add_done_callback(settled)
            assert exchange.future is not None
            if client_disconnect is not None:

                def disconnected() -> None:
                    nonlocal peer_closed
                    with self._condition:
                        peer_closed = True
                        if spec.authority is DaemonAuthority.READ or not exchange.acceptance_started:
                            exchange.cancellation.cancel()
                        self._condition.notify_all()

                client_disconnect.add_listener(disconnected)
            while True:
                try:
                    record = self._durable(exchange)
                except MachineRequestConflictError:
                    return operation_envelope(
                        request,
                        context,
                        snapshot=exchange.snapshot,
                        outcome="rejected",
                        error={"code": "request_identity_conflict", "retryable": False},
                    ).to_dict()
                if peer_closed and not exchange.future.done():
                    return self._pending_envelope(
                        exchange,
                        outcome="indeterminate" if exchange.acceptance_started else "disconnected-before-acceptance",
                        record=record,
                    )
                if exchange.future.done():
                    try:
                        envelope = exchange.future.result().to_dict()
                    except BeforeAcceptanceCancelledError:
                        envelope = self._pending_envelope(exchange, outcome="cancelled")
                    except Exception as exc:
                        outcome = "indeterminate" if exchange.acceptance_started else "failed"
                        envelope = self._pending_envelope(exchange, outcome=outcome, record=record)
                        envelope["error"] = {"code": type(exc).__name__, "detail": str(exc), "retryable": False}
                    if exchange.acceptance_started:
                        audit = AuditRepository.for_archive_root(self.archive_root)
                        try:
                            with audit.settled_machine_read():
                                record = (
                                    audit.machine_request(exchange.binding) if exchange.binding is not None else None
                                )
                                if record is None and envelope.get("outcome") == "indeterminate":
                                    # The actual worker settled and continuity
                                    # proves there is no accepted domain work.
                                    envelope["outcome"] = "failed"
                        except AuditContinuityPendingError:
                            envelope["outcome"] = "indeterminate"
                    if record is not None:
                        envelope["accepted_reference"] = record
                        state = self._recovery_state(record)
                        envelope["outcome"] = state["outcome"]
                        envelope["result"] = state.get("result", state)
                        # Durable receipts outrank an exception raised after
                        # publication. A stale handler error is not authority.
                        if state["outcome"] == "completed":
                            envelope.pop("error", None)
                    envelope["timing"] = {
                        "elapsed_ms": max(0, int((monotonic() - started) * 1000)),
                        "queue_ms": exchange.queue_ms,
                    }
                    if isinstance(envelope.get("authority_snapshot"), dict):
                        envelope["authority_snapshot"].update(envelope["timing"])
                    return envelope
                if record is not None:
                    return self._pending_envelope(exchange, outcome="accepted", record=record)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    if not exchange.acceptance_started:
                        exchange.cancellation.cancel()
                        return self._pending_envelope(exchange, outcome="timed-out")
                    return self._pending_envelope(exchange, outcome="indeterminate")
                self._condition.wait(timeout=remaining)

    def control(
        self, request: DaemonOperationRequest, principal: MutationPrincipal, archive_identity: str
    ) -> dict[str, object]:
        target = str(request.payload["request_id"])
        deadline = monotonic() + min(30.0, int(request.payload.get("timeout_ms", 0)) / 1000)
        after = int(request.payload.get("after_sequence", 0))
        audit = AuditRepository.for_archive_root(self.archive_root)
        if request.operation == "operation.cancel":
            with self._condition:
                exchange = self._exchanges.get(target)
                if exchange is not None:
                    if exchange.context.principal != principal:
                        raise PermissionError("operation reference belongs to another principal")
                    exchange.cancellation.cancel()
                    self._condition.notify_all()
            if exchange is None:
                # A settled request has no live worker. Queue the durable fence
                # through the same writer owner, never under the waiter lock.
                def fence() -> None:
                    record = audit.machine_request_for_principal(archive_identity, target, principal.actor_ref)
                    if record is None:
                        raise ValueError("operation_reference_unknown")
                    if record["artifact_kind"] == "execution-batch":
                        binding = MachineRequestBinding(
                            **{
                                key: str(record[key])
                                for key in (
                                    "archive_identity",
                                    "request_id",
                                    "principal_ref",
                                    "fingerprint",
                                    "operation_name",
                                )
                            }
                        )
                        parts = audit.machine_parts(binding)
                        if any(part["operation_id"] is None for part in parts):
                            audit.stop_machine_batch(binding, "cancelled")

                try:
                    self._bridge.run_sync_with_timeout("operation.cancel", 2.0, fence)
                except TimeoutError:
                    return {"outcome": "indeterminate", "sequence": 0, "cancellation_requested": True}
                self._notify()
        with self._condition:
            while True:
                exchange = self._exchanges.get(target)
                if exchange is not None and exchange.context.principal != principal:
                    raise PermissionError("operation reference belongs to another principal")
                pending = False
                try:
                    with audit.settled_machine_read():
                        record = audit.machine_request_for_principal(archive_identity, target, principal.actor_ref)
                        state = machine_request_state(audit, record) if record is not None else None
                except AuditContinuityPendingError:
                    pending = True
                    state = {"outcome": "indeterminate", "sequence": 0}
                if state is None:
                    if exchange is None:
                        raise ValueError("operation_reference_unknown")
                    state = {"outcome": "running", "sequence": 0}
                if request.operation != "operation.await":
                    return state
                if not pending and (int(state["sequence"]) > after or state["outcome"] not in {"running", "accepted"}):
                    return state
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return state
                self._condition.wait(timeout=remaining)
