"""Resident operation ownership and event-driven completion over audit references."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic, time
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext, QueryTimeoutError
from polylogue.daemon.execution import BoundedComputeAdapter, CancellationHandle, DaemonBackpressureError
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
from polylogue.operations.audit import (
    AuditContinuityPendingError,
    AuditRepository,
    MachineRequestBinding,
    MachineRequestConflictError,
)
from polylogue.operations.daemon_execution import execute_operation, operation_envelope, validate_execution_request
from polylogue.operations.daemon_protocol import (
    AcceptedOperationReference,
    DaemonAuthority,
    DaemonOperationEnvelope,
    DaemonOperationRequest,
    daemon_operation_spec,
)
from polylogue.operations.daemon_reads import DaemonReadDependencies
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.operations.operation_context import OperationContext, PinnedOperationRead, observe_control_authority

if TYPE_CHECKING:
    from polylogue.daemon.session_insight_maintenance import SessionInsightMaintenance
    from polylogue.operations.insight_acceptance import AcceptedInsightPart, SessionInsightPartReceipt

_T = TypeVar("_T")


def _operation_int(value: object, *, field: str) -> int:
    """Reject malformed operation payloads instead of coercing control facts."""

    if type(value) is not int:
        raise ValueError(f"operation {field} is not an integer")
    return value


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
        owner_loop: asyncio.AbstractEventLoop | None = None,
        session_maintenance: SessionInsightMaintenance | None = None,
    ) -> None:
        self.archive_root = archive_root.resolve()
        self._bridge = write_bridge
        self._kernel = execution_kernel
        self._read_dependencies = read_dependencies
        self._read_dependencies_factory = read_dependencies_factory
        self._owner_loop = owner_loop
        self._session_maintenance = session_maintenance
        self._condition = threading.Condition(threading.RLock())
        self._exchanges: dict[str, _Exchange] = {}
        self._closing = False

    async def shutdown(self) -> None:
        """Stop admission and settle actual operation workers before owner teardown."""
        with self._condition:
            self._closing = True
            exchanges = tuple(self._exchanges.values())
            for exchange in exchanges:
                exchange.cancellation.cancel()
            self._condition.notify_all()
        pending = asyncio.gather(
            *(asyncio.wrap_future(exchange.future) for exchange in exchanges if exchange.future is not None),
            return_exceptions=True,
        )
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
            raise

    @property
    def shutdown_settled(self) -> bool:
        with self._condition:
            return self._closing and not self._exchanges

    def publication_guard(self) -> AbstractContextManager[None]:
        return self._bridge.hold("operation.pin-read")

    def run_write(self, name: str, work: Callable[[], _T]) -> _T:
        return self._bridge.run_sync_with_timeout(f"operation.{name}", None, work)

    async def compute_phase(self, work: Callable[[], _T]) -> _T:
        """Await shared admission without occupying another kernel worker."""
        submitted = self._kernel.submit(work, admission_class="control")
        pending = asyncio.wrap_future(submitted.future)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not pending.cancelled():
                pending.exception()
            raise

    async def write_phase(self, name: str, work: Callable[[], _T]) -> _T:
        result = await self._bridge.run_async(f"operation.{name}", work)
        self._notify()
        return result

    def require_session_maintenance(self) -> None:
        if self._session_maintenance is None:
            raise ValueError("session derivation runtime is unavailable")

    def session_profile_plan_binding(self, *, opened_index_path: Path) -> tuple[str, str]:
        self.require_session_maintenance()
        assert self._session_maintenance is not None
        return self._session_maintenance.plan_binding(opened_index_path=opened_index_path)

    async def converge_ingest_sessions(
        self,
        request: DaemonOperationRequest,
        session_ids: tuple[str, ...],
        *,
        expected_recipe: str,
        stop_requested: Callable[[], str | None],
    ) -> SessionInsightPartReceipt:
        self.require_session_maintenance()
        assert self._session_maintenance is not None
        return await self._session_maintenance.converge_ingest_sessions(
            session_ids,
            expected_recipe=expected_recipe,
            stop_requested=lambda: self.stop_reason(request) or stop_requested(),
        )

    async def converge_insight_part(
        self, request: DaemonOperationRequest, part: AcceptedInsightPart, *, stop_requested: Callable[[], str | None]
    ) -> SessionInsightPartReceipt:
        self.require_session_maintenance()
        assert self._session_maintenance is not None
        return await self._session_maintenance.converge_part(
            part, stop_requested=lambda: self.stop_reason(request) or stop_requested()
        )

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
                record = audit.machine_request(exchange.binding)
                # Preview-page staging is durable authority preparation, not
                # acceptance of the execution manifest. Only its seal crosses
                # the machine acceptance boundary.
                if record is not None and record["artifact_kind"] == "insight-preview-pages":
                    return None
                return record
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
            return {
                "outcome": "indeterminate",
                "sequence": 0,
                "reference": AcceptedOperationReference.from_record(record).to_dict(),
            }

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
            if spec.authority is DaemonAuthority.READ or request.operation.startswith("operation.")
            else None
        )
        context = OperationContext(self.archive_root, principal, "daemon", self, dependencies, read_control)
        try:
            request = validate_execution_request(request, context)
        except (ValueError, PermissionError) as exc:
            return operation_envelope(
                request,
                context,
                started_at=started,
                outcome="rejected",
                error={"code": str(exc), "detail": str(exc), "retryable": False},
            ).to_dict()
        if request.operation.startswith("operation."):
            assert read_control is not None
            if client_disconnect is not None:

                def disconnect_control() -> None:
                    read_control.cancel()
                    self._notify()

                client_disconnect.add_listener(disconnect_control)
            return execute_operation(request, context).to_dict()
        if spec.accepted_reference:
            try:
                control = observe_control_authority(self.archive_root)
            except ValueError as exc:
                return operation_envelope(
                    request,
                    context,
                    outcome="rejected",
                    error={"code": str(exc), "retryable": False},
                ).to_dict()
            if request.archive_root is not None and Path(request.archive_root).resolve() != self.archive_root.resolve():
                return operation_envelope(
                    request,
                    context,
                    snapshot=control,
                    outcome="rejected",
                    error={"code": "archive_identity_mismatch", "retryable": False},
                ).to_dict()
            binding = MachineRequestBinding(
                control.identity.authority_identity_digest,
                str(request.request_id),
                principal.actor_ref,
                request.fingerprint,
                request.operation,
            )
            audit = AuditRepository.for_archive_root(self.archive_root)
            record: dict[str, object] | None = None
            try:
                with audit.settled_machine_read():
                    record = audit.machine_request(binding)
                    durable = machine_request_state(audit, record) if record is not None else None
            except AuditContinuityPendingError:
                durable = None
            except MachineRequestConflictError:
                return operation_envelope(
                    request,
                    context,
                    snapshot=control,
                    outcome="rejected",
                    error={"code": "request_identity_conflict", "retryable": False},
                ).to_dict()
            if durable is not None and durable["outcome"] in {"completed", "failed", "cancelled"}:
                # Initial generation/recipe preconditions were checked at
                # acceptance. A historical terminal receipt does not reopen
                # index/source or become false after ordinary reconvergence.
                return operation_envelope(
                    request,
                    context,
                    snapshot=control,
                    started_at=started,
                    outcome=str(durable["outcome"]),
                    reference=record,
                    result=durable.get("result", durable),
                ).to_dict()
        request_id = str(request.request_id)
        peer_closed = False
        with self._condition:
            if self._closing:
                return operation_envelope(
                    request,
                    context,
                    outcome="rejected",
                    error={"code": "runtime_stopping", "retryable": True},
                ).to_dict()
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
                    if request.operation in {"ingest", "maintenance.insights.rebuild"}:
                        if self._owner_loop is None:
                            self._exchanges.pop(request_id)
                            return operation_envelope(
                                request,
                                context,
                                outcome="rejected",
                                error={"code": "ingest_runtime_unavailable", "retryable": False},
                            ).to_dict()
                        from polylogue.operations.daemon_ingest import execute_ingest_operation
                        from polylogue.operations.daemon_insights import execute_insights_rebuild_operation

                        staged = (
                            execute_ingest_operation
                            if request.operation == "ingest"
                            else execute_insights_rebuild_operation
                        )
                        exchange.future = asyncio.run_coroutine_threadsafe(staged(request, context), self._owner_loop)
                    else:
                        scheduled = self._kernel.submit(
                            work,
                            admission_class="interactive-read" if spec.authority is DaemonAuthority.READ else "control",
                            # A control exchange keeps its durable authority after
                            # acceptance, but before that boundary a disconnect or
                            # deadline must release a queued reservation just as a
                            # read does.  The operation body still decides any
                            # in-flight post-acceptance cancellation semantics.
                            cancellation=exchange.cancellation,
                        )
                        exchange.future = scheduled.future
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

                def settled(_future: Future[DaemonOperationEnvelope]) -> None:
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
                        outcome=(
                            "disconnected-after-acceptance"
                            if record is not None
                            else "indeterminate"
                            if exchange.acceptance_started
                            else "disconnected-before-acceptance"
                        ),
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
                        envelope["accepted_reference"] = AcceptedOperationReference.from_record(record).to_dict()
                        state = self._recovery_state(record)
                        envelope["outcome"] = state["outcome"]
                        envelope["result"] = state.get("result", state)
                        # Durable receipts outrank an exception raised after
                        # publication. A stale handler error is not authority.
                        if state["outcome"] == "completed":
                            envelope.pop("error", None)
                    timing: dict[str, int] = {
                        "elapsed_ms": max(0, int((monotonic() - started) * 1000)),
                        "queue_ms": exchange.queue_ms,
                    }
                    envelope["timing"] = timing
                    authority_snapshot = envelope.get("authority_snapshot")
                    if isinstance(authority_snapshot, dict):
                        authority_snapshot.update(timing)
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
        self,
        request: DaemonOperationRequest,
        principal: MutationPrincipal,
        archive_identity: str,
        *,
        execution_context: QueryExecutionContext | None = None,
    ) -> dict[str, object]:
        target = str(request.payload["request_id"])
        deadline = monotonic() + min(30.0, _operation_int(request.payload.get("timeout_ms", 0), field="timeout") / 1000)
        if execution_context is not None and execution_context.deadline_monotonic is not None:
            deadline = min(deadline, execution_context.deadline_monotonic)
        after = _operation_int(request.payload.get("after_sequence", 0), field="after sequence")
        audit = AuditRepository.for_archive_root(self.archive_root)
        if execution_context is not None:
            if execution_context.cancelled:
                raise QueryCancelledError("operation control exchange disconnected")
            if execution_context.deadline_exceeded():
                raise QueryTimeoutError("operation control exchange deadline expired")
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
                    if record["artifact_kind"] in {"execution-batch", "source-generation"}:
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
                        if any(part["operation_id"] is None for part in parts) or (
                            record["artifact_kind"] == "source-generation"
                            and machine_request_state(audit, record)["outcome"] not in {"completed", "failed"}
                        ):
                            audit.stop_machine_batch(binding, "cancelled")

                try:
                    self._bridge.run_sync_with_timeout("operation.cancel", 2.0, fence)
                except TimeoutError:
                    return {"outcome": "indeterminate", "sequence": 0, "cancellation_requested": True}
                self._notify()
        with self._condition:
            while True:
                # Once a cancellation fence starts, its actual receipt decides
                # the outcome. A late deadline cannot turn it into no-effect.
                if execution_context is not None and request.operation != "operation.cancel":
                    if execution_context.cancelled:
                        raise QueryCancelledError("operation control exchange disconnected")
                    if execution_context.deadline_exceeded():
                        raise QueryTimeoutError("operation control exchange deadline expired")
                if self._closing:
                    raise QueryCancelledError("operation runtime is stopping")
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
                sequence = _operation_int(state["sequence"], field="state sequence")
                if not pending and (sequence > after or state["outcome"] not in {"running", "accepted"}):
                    return state
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return state
                self._condition.wait(timeout=remaining)
