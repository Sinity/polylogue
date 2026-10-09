"""Resident operation ownership and event-driven completion over audit references."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError, Future
from contextlib import AbstractContextManager, closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic, time
from typing import TYPE_CHECKING, Any, TypeVar, cast

from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext, QueryTimeoutError
from polylogue.core.compute import (
    BoundedComputeAdapter,
    CancellationHandle,
    DaemonBackpressureError,
    DaemonOperationCancelled,
)
from polylogue.core.compute_cancel import compute_cancel
from polylogue.core.digest import stdlib_chunks
from polylogue.core.durable_fs import sync_directory
from polylogue.core.raw_failure_evidence import (
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
    RetainedRawDependencyRefusalError,
)
from polylogue.core.stage_admission import stage_write_admission, stage_write_admission_bound
from polylogue.core.staged_body import StagedBody
from polylogue.core.write_lease import adopt_write_lease
from polylogue.daemon.drive_catchup import DriveCatchupExecution
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge, StagedTask
from polylogue.logging import WARNING, emit, propagate
from polylogue.operations.audit import (
    MACHINE_PAGE_KINDS,
    AuditContinuityError,
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
from polylogue.operations.daemon_reads import DaemonReadDependencies, read_is_archive_scan
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.operations.operation_context import (
    OperationControlRead,
    OperationControlResult,
    PinnedOperationRead,
    observe_control_authority,
    open_operation_control,
)
from polylogue.operations.operation_context_types import OperationContext

if TYPE_CHECKING:
    from polylogue.daemon.session_insight_maintenance import SessionInsightMaintenance
    from polylogue.operations.audit import CanonicalAuditLiteral
    from polylogue.operations.insight_acceptance import AcceptedInsightPart, SessionInsightPartReceipt
    from polylogue.operations.raw_observation_owner import RetainedMaterializationResult

_T = TypeVar("_T")


def _operation_int(value: object, *, field: str) -> int:
    """Reject malformed operation payloads instead of coercing control facts."""

    if type(value) is not int:
        raise ValueError(f"operation {field} is not an integer")
    return value


#: Operations that run as coroutines on the owner loop. Each crosses its
#: acceptance boundary through ``audit_for_request`` or ``begin_unbound_write``,
#: which refuse once the exchange is cancelled.
_STAGED_OPERATIONS = frozenset(
    {
        "ingest",
        "mutation.session.excision",
        "maintenance.raw-authority-frontier",
        "mutation.raw-authority-blocker.resolve",
        "mutation.facade.record_work_event",
        "maintenance.insights.rebuild",
        "maintenance.embeddings.backfill",
        "maintenance.backup",
        "maintenance.restore_verified_backup",
        "mutation.session.delete.preview",
        "mutation.identity-reset.preview",
        "mutation.session.mark",
    }
)


#: Request-body bytes of the staged exchange whose task is running. A staged
#: operation's compute phases reserve them, as the scheduled route does for its
#: one submission, so a queued staged exchange is not admitted as weightless.
_STAGED_REQUEST_BYTES: contextvars.ContextVar[int] = contextvars.ContextVar("polylogue_staged_request_bytes", default=0)


#: Name prefix of the ingest owner's re-drive task on its owner loop.
REDRIVE_TASK_PREFIX = "polylogue-ingest-redrive:"


class BeforeAcceptanceCancelledError(RuntimeError):
    """Cancellation won the lock before durable prepare could begin."""


@dataclass(slots=True)
class _Exchange:
    request: DaemonOperationRequest
    context: OperationContext
    deadline: float | None
    deadline_unix_ms: int | None
    future: Future[DaemonOperationEnvelope] | None = None
    cancellation: CancellationHandle = field(default_factory=CancellationHandle)
    acceptance_started: bool = False
    snapshot: PinnedOperationRead | OperationControlRead | None = None
    binding: MachineRequestBinding | None = None
    accepted_reference: dict[str, object] | None = None
    queue_ms: int = 0
    started_at: float = field(default_factory=monotonic)
    progress_events: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=64))
    progress_sequence: int = 0
    progress_gap_until: int = 0
    settled_at: float | None = None
    terminal_transfer_error: str | None = None
    terminal_envelope: dict[str, object] | None = None
    result_document: dict[str, object] | None = None
    result_summary: dict[str, object] | None = None


class DaemonOperationRuntime:
    def __init__(
        self,
        archive_root: Path,
        *,
        write_bridge: DaemonWriteThreadBridge,
        execution_kernel: BoundedComputeAdapter,
        raw_observation_owner: RawObservationConvergenceOwner,
        read_dependencies: DaemonReadDependencies | None = None,
        read_dependencies_factory: Callable[[], DaemonReadDependencies] | None = None,
        owner_loop: asyncio.AbstractEventLoop | None = None,
        session_maintenance: SessionInsightMaintenance | None = None,
    ) -> None:
        self.archive_root = archive_root.resolve()
        from polylogue.core.staged_body import reap_stale_staging

        reap_stale_staging(self.archive_root / "operation-inputs")
        self._bridge = write_bridge
        self._kernel = execution_kernel
        self.raw_observation_owner = raw_observation_owner
        self._read_dependencies = read_dependencies
        self._read_dependencies_factory = read_dependencies_factory
        self._owner_loop = owner_loop
        self._session_maintenance = session_maintenance
        # The daemon CLI installs its resident embedding convergence owner
        # here so staged backfill operations share the same pass lock.
        self.embedding_convergence: object | None = None
        self._condition = threading.Condition(threading.RLock())
        self._exchanges: dict[str, _Exchange] = {}
        self._closing = False
        from polylogue.operations.assertion_export import AssertionExportImages

        self.assertion_exports = AssertionExportImages()
        self._terminal_scratch: tempfile.TemporaryDirectory[str] | None = None
        self._terminal_epoch = uuid.uuid4().hex
        # The ingest owner's re-drive of accepted generations a dead process
        # left without a terminal checkpoint, and the request ids a cancel
        # has fenced while it runs.
        self._redrive: Future[None] | None = None
        self._redrive_cancelled: set[str] = set()
        # Set on the owner loop once every eligible run is claimed.
        self._redrive_claimed: threading.Event = threading.Event()

    async def materialize_retained_raw_ids(
        self,
        raw_ids: tuple[str, ...],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None],
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None],
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None],
        before_publication: Callable[[], None],
    ) -> RetainedMaterializationResult:
        owner = self.raw_observation_owner
        if (
            owner._archive_root.resolve() != self.archive_root
            or owner._compute_adapter is not self._kernel
            or owner._write_coordinator is not self._bridge.coordinator
        ):
            raise ValueError("operation materialization requires its original supplied resident owner")
        return await owner.materialize_retained_raw_ids(
            raw_ids,
            on_terminal_refusal=on_terminal_refusal,
            on_dependency_refusal=on_dependency_refusal,
            on_membership_refusal=on_membership_refusal,
            before_publication=before_publication,
        )

    def start_accepted_ingest_redrive(self) -> None:
        """Re-drive accepted ingests left without a terminal checkpoint, once per owner start.

        Startup recovery leaves such runs to this owner
        (``IngestRecovery``); it must run after that recovery. The re-drive
        runs on the owner loop beside request exchanges and uses the same
        writer and compute phases as a fresh ingest request.
        """
        if self._owner_loop is None or self._session_maintenance is None:
            return
        with self._condition:
            if self._redrive is not None or self._closing:
                return
            from polylogue.operations.daemon_ingest import redrive_accepted_ingests

            def stop_requested(request_id: str) -> str | None:
                with self._condition:
                    if request_id in self._redrive_cancelled:
                        return "cancelled"
                    return "shutdown" if self._closing else None

            claimed = self._redrive_claimed

            async def redrive() -> None:
                try:
                    await redrive_accepted_ingests(
                        self,
                        self.archive_root,
                        stop_requested=stop_requested,
                        on_commit=self._notify,
                        on_claimed=claimed.set,
                    )
                except Exception as exc:
                    emit(
                        "ingest.redrive.unavailable",
                        level=WARNING,
                        outcome="refused",
                        error_type=type(exc).__name__,
                        error_detail=str(exc)[:512],
                    )

            # Named: the re-drive is this runtime's own declared child on the
            # owner loop, not an anonymous task.
            self._redrive = StagedTask(
                self._owner_loop, redrive, name=f"{REDRIVE_TASK_PREFIX}{self.archive_root}"
            ).future
            redrive_future = self._redrive
        try:
            on_owner_loop = asyncio.get_running_loop() is self._owner_loop
        except RuntimeError:
            on_owner_loop = False
        if on_owner_loop:
            # The claim phase runs on this very loop: blocking here would
            # deadlock it. The caller awaits ``accepted_ingest_redrive_claimed``.
            return
        # Hold the caller -- the server, before it exposes its listeners --
        # until every eligible run is claimed: a resent request then reads
        # its run as ``running`` and follows it to the receipt, instead of
        # reading the still-``interrupted`` run as indeterminate.
        while not claimed.wait(0.05):
            if redrive_future.done():
                break

    async def accepted_ingest_redrive_claimed(self) -> None:
        """Wait, on the owner loop, until the started re-drive claimed every eligible run.

        The owner-loop composition (``polylogued run``) constructs its server
        on the loop the re-drive runs on, so it awaits this before its
        listeners serve instead of blocking in the constructor.
        """
        redrive = self._redrive
        while redrive is not None and not self._redrive_claimed.is_set() and not redrive.done():
            await asyncio.sleep(0.02)

    async def shutdown(self) -> None:
        """Stop admission and settle actual operation workers before owner teardown."""
        with self._condition:
            self._closing = True
            exchanges = tuple(self._exchanges.values())
            for exchange in exchanges:
                exchange.cancellation.cancel()
            redrive = self._redrive
            self._condition.notify_all()
        pending = asyncio.gather(
            *(asyncio.wrap_future(exchange.future) for exchange in exchanges if exchange.future is not None),
            *(() if redrive is None else (asyncio.wrap_future(redrive),)),
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
        finally:
            if pending.done():
                with self._condition:
                    self._exchanges.clear()
                    self.assertion_exports.close()
                    if self._terminal_scratch is not None:
                        self._terminal_scratch.cleanup()
                        self._terminal_scratch = None

    @property
    def shutdown_settled(self) -> bool:
        with self._condition:
            return (
                self._closing
                and not self._exchanges
                and self._terminal_scratch is None
                and (self._redrive is None or self._redrive.done())
            )

    def publication_guard(self) -> AbstractContextManager[object]:
        return self._bridge.hold("operation.pin-read")

    def run_write(self, name: str, work: Callable[[], _T]) -> _T:
        return self._bridge.run_sync_with_timeout(f"operation.{name}", None, work)

    async def compute_phase(self, work: Callable[[], _T]) -> _T:
        """Await shared admission without occupying another kernel worker."""
        # The kernel's pool outlives every bind, so its threads carry no
        # correlation context of their own (verified: a bare submit sees an
        # empty context where a propagate()d one does not).
        submitted = self._kernel.submit(
            propagate(work), admission_class="control", estimated_bytes=_STAGED_REQUEST_BYTES.get()
        )
        # ``SubmittedOperation.wait`` releases queued work, requests creator-
        # owned SQL settlement, and retains this exchange until physical work
        # is done even when the caller is cancelled repeatedly.
        return await submitted.wait()

    async def write_phase(self, name: str, work: Callable[[], _T]) -> _T:
        result = await self._bridge.run_async(f"operation.{name}", work)
        self._notify()
        return result

    def prepared_compute_adapter(self) -> BoundedComputeAdapter:
        """Borrow this phase's actual creator reservation for retained preparation."""
        self._kernel.require_current_creator()
        if not stage_write_admission_bound():
            raise RuntimeError("retained preparation requires its admitted publication phase")
        return self._kernel

    async def prepared_phase(
        self, name: str, work: Callable[[], _T], *, estimated_bytes: int, exclusive_bytes: bool = False
    ) -> _T:
        """Retain preparation, short publication and cleanup on one admitted creator."""
        coordinator = self._bridge.coordinator
        cancelled = threading.Event()

        def admit_write(actor: str, operation: Callable[[], _T]) -> _T:
            with self._bridge.hold(actor) as delegation, adopt_write_lease(delegation):
                return operation()

        def run() -> _T:
            self._kernel.require_current_creator()
            with stage_write_admission(admit_write):
                return work()

        def submit_worker(worker: Callable[[], None]) -> Future[None]:
            return self._kernel.submit(
                propagate(worker),
                admission_class="control",
                estimated_bytes=estimated_bytes,
                exclusive_bytes=exclusive_bytes,
            ).future

        token = compute_cancel.set(cancelled)
        try:
            pending = asyncio.create_task(
                coordinator.run_prepared_sync(
                    f"operation.{name}",
                    run,
                    submit_worker=submit_worker,
                    # Original registered native/seal owners remain discoverable
                    # until physical close; no carrier leaves this phase.
                    settlement_owners=lambda: (),
                ),
                name=f"polylogue-operation-prepared:{name}",
            )
        finally:
            compute_cancel.reset(token)
        result = await DriveCatchupExecution(coordinator, compute_adapter=self._kernel).settle(
            pending, label=name, cancel_requested=cancelled.set
        )
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
        session_ids: tuple[str, ...],
        *,
        expected_recipe: str,
        stop_requested: Callable[[], str | None],
    ) -> SessionInsightPartReceipt:
        """Derive one page of ingested sessions; ``stop_requested`` is the ingest attempt's own stop."""
        self.require_session_maintenance()
        assert self._session_maintenance is not None
        return await self._session_maintenance.converge_ingest_sessions(
            session_ids,
            expected_recipe=expected_recipe,
            stop_requested=stop_requested,
        )

    async def converge_insight_part(
        self, request: DaemonOperationRequest, part: AcceptedInsightPart, *, stop_requested: Callable[[], str | None]
    ) -> SessionInsightPartReceipt:
        self.require_session_maintenance()
        assert self._session_maintenance is not None
        return await self._session_maintenance.converge_part(
            part, stop_requested=lambda: self.stop_reason(request) or stop_requested()
        )

    def observe_snapshot(
        self, request: DaemonOperationRequest, snapshot: PinnedOperationRead | OperationControlRead
    ) -> None:
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
                assert isinstance(snapshot, PinnedOperationRead)
                exchange.cancellation.add_listener(snapshot.archive.interrupt_reads)

    def request_deadline_unix_ms(self, request: DaemonOperationRequest) -> int:
        deadline = self._exchanges[str(request.request_id)].deadline_unix_ms
        assert deadline is not None  # only write owners request durable deadline evidence
        return deadline

    def stop_reason(self, request: DaemonOperationRequest) -> str | None:
        exchange = self._exchanges[str(request.request_id)]
        if exchange.cancellation.cancelled:
            return "cancelled"
        return "deadline" if (exchange.deadline is not None and monotonic() >= exchange.deadline) else None

    def _notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def emit_progress(self, request: DaemonOperationRequest, event: Mapping[str, object]) -> None:
        """Publish one bounded, request-scoped observation for ``operation.await``.

        Progress is deliberately in-memory. Durably bound operations use
        Audit for terminal state; unbound accepted operations use their exact
        retained exchange result. A slow waiter may observe a visible gap and
        must then read terminal state rather than infer work from frames.
        """
        request_id = str(request.request_id)
        with self._condition:
            exchange = self._exchanges.get(request_id)
            if exchange is None or exchange.request.operation != request.operation:
                return
            exchange.progress_sequence += 1
            frame = {**dict(event), "sequence": exchange.progress_sequence}
            if len(exchange.progress_events) == exchange.progress_events.maxlen:
                oldest = exchange.progress_events[0]
                exchange.progress_gap_until = max(
                    exchange.progress_gap_until,
                    _operation_int(oldest["sequence"], field="progress sequence"),
                )
            exchange.progress_events.append(frame)
            self._condition.notify_all()

    @staticmethod
    def _progress_state(exchange: _Exchange, after_sequence: int) -> dict[str, object]:
        frames = [
            frame
            for frame in exchange.progress_events
            if _operation_int(frame["sequence"], field="progress sequence") > after_sequence
        ]
        gap: dict[str, int] | None = None
        if exchange.progress_gap_until > after_sequence:
            gap = {
                "from_sequence": after_sequence + 1,
                "to_sequence": exchange.progress_gap_until,
            }
        return {
            "progress_sequence": exchange.progress_sequence,
            "progress_events": frames,
            "progress_gap": gap,
        }

    def audit_for_request(self, request: DaemonOperationRequest, context: OperationContext) -> AuditRepository:
        exchange = self._exchanges[str(request.request_id)]

        def before_prepare() -> None:
            with self._condition:
                if exchange.cancellation.cancelled or (
                    exchange.deadline is not None and monotonic() >= exchange.deadline
                ):
                    raise BeforeAcceptanceCancelledError("operation cancelled before durable acceptance")
                exchange.acceptance_started = True

        return AuditRepository(
            context.archive_root / "audit.db",
            attempt_owner_id=AuditRepository.current_process_attempt_owner(),
            before_machine_prepare=before_prepare,
            on_commit=self._notify,
        )

    def begin_unbound_write(
        self, request: DaemonOperationRequest, *, snapshot: PinnedOperationRead | OperationControlRead
    ) -> None:
        """Fence a snapshotless filesystem write before its first possible effect."""
        with self._condition:
            exchange = self._exchanges[str(request.request_id)]
            if exchange.cancellation.cancelled or (exchange.deadline is not None and monotonic() >= exchange.deadline):
                raise BeforeAcceptanceCancelledError("operation cancelled before backup admission")
            exchange.snapshot = snapshot
            exchange.acceptance_started = True
            self._condition.notify_all()

    @staticmethod
    def _terminal_principal(principal: MutationPrincipal) -> dict[str, object]:
        return {
            "actor_ref": principal.actor_ref,
            "capabilities": sorted(principal.capabilities),
            "surface": principal.surface,
            "role_label": principal.role_label,
        }

    def _terminal_path(self, request_id: str) -> Path | None:
        if self._terminal_scratch is None:
            return None
        return Path(self._terminal_scratch.name) / (
            hashlib.sha256(request_id.encode("utf-8", "surrogatepass")).hexdigest() + ".json"
        )

    def _read_retained_terminal(
        self, request_id: str, principal: MutationPrincipal, archive_identity: str
    ) -> dict[str, object] | None:
        path = self._terminal_path(request_id)
        if path is None:
            return None
        try:
            with path.open(encoding="utf-8") as stream:
                packet = json.load(stream)
        except FileNotFoundError:
            return None
        if packet["epoch"] != self._terminal_epoch or packet["request_id"] != request_id:
            raise ValueError("operation_result_custody_mismatch")
        if packet["principal"] != self._terminal_principal(principal):
            raise PermissionError("operation reference belongs to another principal")
        if packet["archive_identity"] != archive_identity:
            raise ValueError("archive_identity_stale")
        return cast("dict[str, object]", packet)

    @staticmethod
    def _retained_terminal_envelope(exchange: _Exchange) -> dict[str, object]:
        assert exchange.future is not None and exchange.future.done()
        if exchange.terminal_envelope is not None:
            return exchange.terminal_envelope
        if not exchange.future.cancelled():
            try:
                exchange.terminal_envelope = exchange.future.result().to_dict()
                return exchange.terminal_envelope
            except Exception as exc:
                error = {
                    "code": str(getattr(exc, "code", type(exc).__name__)),
                    "detail": str(exc),
                    "retryable": False,
                    "data": getattr(exc, "data", {}),
                }
        else:
            error = {"code": "accepted_worker_cancelled", "retryable": True}
        # A cancelled/failed worker after possible effect is not proof of
        # no effect. Keep its actual settlement separate from domain success.
        exchange.terminal_envelope = operation_envelope(
            exchange.request, exchange.context, snapshot=exchange.snapshot, outcome="indeterminate", error=error
        ).to_dict()
        return exchange.terminal_envelope

    @staticmethod
    def _retained_control_state(terminal: dict[str, object]) -> dict[str, object]:
        result = terminal.get("result")
        state = dict(result) if isinstance(result, dict) else {"sequence": 0}
        state.setdefault("sequence", 0)
        state["outcome"] = terminal["outcome"]
        if terminal.get("error") is not None:
            state["error"] = terminal["error"]
        return state

    def _unbound_control_snapshot(self, archive_identity: str) -> OperationControlRead:
        snapshot = observe_control_authority(self.archive_root)
        if snapshot.identity.authority_identity_digest != archive_identity:
            raise ValueError("archive_identity_stale")
        return snapshot

    @staticmethod
    def _retains_terminal(exchange: _Exchange) -> bool:
        # A bound mutation keeps its original authority. Its complete product
        # also needs terminal custody; a binding is not a retention exemption.
        return exchange.acceptance_started and (
            exchange.binding is None
            or exchange.result_document is not None
            or exchange.request.operation == "mutation.session.excision"
        )

    def _owned_retained_terminal_state(
        self, request_id: str, principal: MutationPrincipal, archive_identity: str
    ) -> dict[str, object] | None:
        exchange = self._exchanges.get(request_id)
        if (
            exchange is None
            or not self._retains_terminal(exchange)
            or exchange.future is None
            or not exchange.future.done()
        ):
            return None
        if exchange.context.principal != principal:
            raise PermissionError("operation reference belongs to another principal")
        assert exchange.snapshot is not None
        if exchange.snapshot.identity.authority_identity_digest != archive_identity:
            raise ValueError("archive_identity_stale")
        state = self._retained_control_state(self._retained_terminal_envelope(exchange))
        if exchange.terminal_transfer_error is not None:
            state["terminal_custody_error"] = exchange.terminal_transfer_error
        return state

    def result_document_identity(self, request: DaemonOperationRequest) -> dict[str, object]:
        with self._condition:
            exchange = self._exchanges[str(request.request_id)]
            if exchange.request.fingerprint != request.fingerprint or exchange.result_document is None:
                raise ValueError("operation_result_document_missing")
            return dict(exchange.result_document)

    def retain_result_document(
        self,
        request: DaemonOperationRequest,
        context: OperationContext,
        summary: Mapping[str, object],
        literal: CanonicalAuditLiteral,
    ) -> None:
        """Transfer the complete settled product into this request's existing custody."""
        with self._condition:
            exchange = self._exchanges[str(request.request_id)]
            if exchange.context.principal != context.principal or exchange.request.fingerprint != request.fingerprint:
                raise ValueError("operation_result_custody_mismatch")
            if not exchange.acceptance_started or exchange.snapshot is None:
                raise ValueError("operation_result_not_accepted")
            if self._terminal_scratch is None:
                self._terminal_scratch = tempfile.TemporaryDirectory(
                    prefix="polylogue-operation-results-", dir=os.environ.get("TMPDIR")
                )
            terminal_path = self._terminal_path(str(request.request_id))
            assert terminal_path is not None
            document_path = terminal_path.with_suffix(".product.json")
            fd, temporary = tempfile.mkstemp(prefix=".product-", dir=self._terminal_scratch.name)
        digest = hashlib.sha256()
        length = 0
        try:
            with os.fdopen(fd, "wb") as output:
                with closing(literal.verified_chunks()) as chunks:
                    for chunk in chunks:
                        from polylogue.core.compute_cancel import check_compute_cancelled

                        check_compute_cancelled()
                        output.write(chunk)
                        digest.update(chunk)
                        length += len(chunk)
                if length != literal.byte_length or digest.hexdigest() != literal.sha256:
                    raise ValueError("operation_result_document_corrupt")
                output.flush()
                os.fsync(output.fileno())
            with self._condition:
                if exchange.result_document is not None:
                    raise ValueError("operation_result_custody_conflict")
                os.replace(temporary, document_path)
                sync_directory(document_path.parent)
                exchange.result_summary = dict(summary)
                exchange.result_document = {
                    "request_id": str(request.request_id),
                    "byte_length": length,
                    "sha256": literal.sha256,
                }
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _retain_terminal(self, exchange: _Exchange) -> None:
        assert exchange.snapshot is not None
        request_id = str(exchange.request.request_id)
        envelope = self._retained_terminal_envelope(exchange)
        if (
            exchange.request.operation == "mutation.session.excision"
            and envelope.get("outcome") == "completed"
            and exchange.result_document is None
        ):
            raise ValueError("operation_result_document_missing")
        if exchange.result_document is not None and envelope.get("outcome") == "completed":
            result = envelope.get("result")
            if not isinstance(result, dict) or result.get("result") != exchange.result_summary:
                raise ValueError("operation_result_summary_mismatch")
            if result.get("result_document") != exchange.result_document:
                raise ValueError("operation_result_document_identity_mismatch")
            product_path = self._terminal_path(request_id)
            if product_path is None:
                raise ValueError("operation_result_document_missing")
            try:
                with product_path.with_suffix(".product.json").open("rb") as stream:
                    if os.fstat(stream.fileno()).st_size != exchange.result_document["byte_length"]:
                        raise ValueError("operation_result_document_corrupt")
                    # The successful sink already verified the installed bytes.
                    # A retained failed transfer must re-prove those same bytes
                    # before metadata recovery can advertise completed delivery.
                    if exchange.terminal_transfer_error is not None:
                        from polylogue.core.compute_cancel import check_compute_cancelled

                        digest = hashlib.sha256()
                        length = 0
                        while True:
                            check_compute_cancelled()
                            chunk = stream.read(64 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                            length += len(chunk)
                        if (
                            length != exchange.result_document["byte_length"]
                            or digest.hexdigest() != exchange.result_document["sha256"]
                        ):
                            raise ValueError("operation_result_document_corrupt")
            except FileNotFoundError as exc:
                raise ValueError("operation_result_document_missing") from exc
        packet = {
            "epoch": self._terminal_epoch,
            "request_id": request_id,
            "fingerprint": exchange.request.fingerprint,
            "principal": self._terminal_principal(exchange.context.principal),
            "archive_identity": exchange.snapshot.identity.authority_identity_digest,
            "envelope": envelope,
            "result_document": exchange.result_document,
        }
        if self._terminal_scratch is None:
            self._terminal_scratch = tempfile.TemporaryDirectory(
                prefix="polylogue-operation-results-", dir=os.environ.get("TMPDIR")
            )
        path = self._terminal_path(request_id)
        assert path is not None
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                if json.load(stream) != packet:
                    raise ValueError("operation_result_custody_conflict")
            sync_directory(path.parent)
            return
        fd, temporary = tempfile.mkstemp(prefix=".result-", dir=self._terminal_scratch.name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(packet, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            Path(temporary).unlink(missing_ok=True)

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
        except AuditContinuityError:
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
        except AuditContinuityError:
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
        request_body_bytes: int | None = None,
        input_body: StagedBody | None = None,
    ) -> dict[str, object]:
        """Transfer sealed input custody only to a newly admitted worker."""
        try:
            if request.operation == "mutation.annotation.import_batch":
                descriptor = request.payload.get("input")
                if (
                    input_body is None
                    or not isinstance(descriptor, dict)
                    or descriptor
                    != {
                        "sha256": input_body.sha256,
                        "size_bytes": input_body.size_bytes,
                    }
                ):
                    raise ValueError("annotation input custody differs from request identity")
            elif input_body is not None:
                raise ValueError("operation does not declare an input body")
            return self._call(
                request,
                principal,
                input_body=input_body,
                started_at=started_at,
                client_disconnect=client_disconnect,
                request_body_bytes=request_body_bytes,
            )
        finally:
            if input_body is not None and not input_body.adopted:
                input_body.discard()

    def _call(
        self,
        request: DaemonOperationRequest,
        principal: MutationPrincipal,
        *,
        started_at: float | None = None,
        client_disconnect: CancellationHandle | None = None,
        request_body_bytes: int | None = None,
        input_body: StagedBody | None = None,
    ) -> dict[str, object]:
        if request_body_bytes is None:
            # Direct callers have no wire body. Count its canonical encoding
            # incrementally instead of allocating another complete body.
            request_body_bytes = sum(len(part.encode()) for part in stdlib_chunks(request.to_dict(), ensure_ascii=True))
        if request_body_bytes < 0:
            raise ValueError("request body byte count must be nonnegative")
        started = monotonic() if started_at is None else started_at
        spec = daemon_operation_spec(request.operation)
        if spec is None:
            raise ValueError("operation is not declared")
        dependencies = (
            self._read_dependencies_factory()
            if self._read_dependencies_factory is not None
            else self._read_dependencies
        )
        dependencies = replace(
            dependencies or DaemonReadDependencies(),
            status_now_ms=int(time() * 1000),
            assertion_exports=self.assertion_exports,
            assertion_export_principal=principal,
        )
        archive_scan = spec.authority is DaemonAuthority.READ and read_is_archive_scan(
            request.operation, request.payload
        )
        deadline_s = spec.deadline_s
        if request.deadline_ms is not None:
            caller_deadline_s = request.deadline_ms / 1000
            deadline_s = caller_deadline_s if deadline_s is None else min(deadline_s, caller_deadline_s)
        deadline = None if deadline_s is None else started + deadline_s
        read_control = (
            QueryExecutionContext(
                call_id=str(request.request_id),
                query_ref=request.fingerprint,
                deadline_monotonic=deadline,
                owner_ref=principal.actor_ref,
                workload_class="scan" if archive_scan else "interactive",
            )
            if spec.authority is DaemonAuthority.READ or request.operation.startswith("operation.")
            else None
        )
        context = OperationContext(self.archive_root, principal, "daemon", self, dependencies, read_control, input_body)
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
        recovered_custody: tuple[OperationControlRead, MachineRequestBinding, dict[str, object]] | None = None
        if spec.accepted_reference or spec.durable_request:
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
            except AuditContinuityError:
                durable = None
            except MachineRequestConflictError:
                return operation_envelope(
                    request,
                    context,
                    snapshot=control,
                    outcome="rejected",
                    error={"code": "request_identity_conflict", "retryable": False},
                ).to_dict()
            if (
                spec.accepted_reference
                and durable is not None
                and durable["outcome"]
                in {
                    "completed",
                    "degraded",
                    "failed",
                    "cancelled",
                    "interrupted",
                }
            ):
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
                    error=cast("dict[str, object] | None", durable.get("error")),
                ).to_dict()
            # A durable record with a started attempt is the recovery authority
            # after a daemon restart.  Re-enqueuing it would create a second
            # exchange and could replay a mutation whose first effect is merely
            # not yet observable.  ``accepted`` is different: every remaining
            # part is unattempted, so no effect can be in flight.  It falls
            # through, joining this daemon's live exchange when one exists and
            # otherwise re-dispatching the handler, which resumes the durable
            # record's unstarted parts and never replays a consumed one.
            # Returning it here instead left an accepted request whose daemon
            # died before its first part stranded at ``accepted`` for good.
            # Preview-page records are only staging authority; _durable
            # deliberately excludes them from this recovery boundary so their
            # normal sealing exchange may continue.
            # A paged batch still staging is resumed the same way: its handler
            # appends the pages it has not yet accepted.
            if (
                record is not None
                and record["artifact_kind"] != "insight-preview-pages"
                and record["artifact_kind"] not in MACHINE_PAGE_KINDS
                and durable is not None
                and durable["outcome"] in {"running", "indeterminate"}
            ):
                return operation_envelope(
                    request,
                    context,
                    snapshot=control,
                    started_at=started,
                    outcome=str(durable["outcome"]),
                    reference=record,
                    result=durable,
                ).to_dict()
            if record is not None and record["artifact_kind"] != "insight-preview-pages":
                recovered_custody = (control, binding, record)

        def admission_refusal(code: str) -> dict[str, object]:
            # Refusing new execution does not relinquish a durable request.
            # Both descriptor forms use this boundary before an Exchange exists.
            if recovered_custody is not None:
                snapshot, _binding, retained_record = recovered_custody
                return operation_envelope(
                    request,
                    context,
                    snapshot=snapshot,
                    outcome="indeterminate",
                    reference=retained_record,
                    result=self._recovery_state(retained_record),
                    error={"code": code, "retryable": False},
                ).to_dict()
            return operation_envelope(
                request,
                context,
                outcome="rejected",
                error={"code": code, "retryable": True},
            ).to_dict()

        request_id = str(request.request_id)
        peer_closed = False
        with self._condition:
            if self._closing:
                return admission_refusal("runtime_stopping")
            path = self._terminal_path(request_id)
            if path is not None and path.exists():
                identity = observe_control_authority(self.archive_root).identity.authority_identity_digest
                try:
                    retained = self._read_retained_terminal(request_id, principal, identity)
                except (PermissionError, ValueError) as exc:
                    return operation_envelope(
                        request, context, outcome="rejected", error={"code": str(exc), "retryable": False}
                    ).to_dict()
                assert retained is not None
                if retained["fingerprint"] != request.fingerprint:
                    return operation_envelope(
                        request,
                        context,
                        outcome="rejected",
                        error={"code": "request_identity_conflict", "retryable": False},
                    ).to_dict()
                return cast("dict[str, object]", retained["envelope"])
            exchange = self._exchanges.get(request_id)
            if exchange is not None:
                if self._retains_terminal(exchange) and exchange.future is not None and exchange.future.done():
                    identity = observe_control_authority(self.archive_root).identity.authority_identity_digest
                    assert exchange.snapshot is not None
                    if exchange.snapshot.identity.authority_identity_digest != identity:
                        return operation_envelope(
                            request,
                            context,
                            outcome="rejected",
                            error={"code": "archive_identity_stale", "retryable": False},
                        ).to_dict()
                if exchange.request.fingerprint != request.fingerprint or exchange.context.principal != principal:
                    return operation_envelope(
                        request,
                        context,
                        outcome="rejected",
                        error={"code": "request_identity_conflict", "retryable": False},
                    ).to_dict()
            else:
                # A failed terminal transfer retains actual custody. Retry
                # that transfer before admitting more mutating work; a disk
                # fault is an explicit admission hold, not result eviction.
                if spec.authority is not DaemonAuthority.READ:
                    for key, held in tuple(self._exchanges.items()):
                        if held.terminal_transfer_error is None:
                            continue
                        try:
                            self._retain_terminal(held)
                        except Exception:
                            return admission_refusal("operation_result_custody_unavailable")
                        self._exchanges.pop(key)
                # Completed progress exchanges are short-lived replay buffers,
                # not active work. Keep them long enough for a CLI whose first
                # await races a fast completion, while bounding total memory.
                stale = [
                    key
                    for key, item in self._exchanges.items()
                    if item.future is not None
                    and item.future.done()
                    and not self._retains_terminal(item)
                    and item.terminal_transfer_error is None
                    and item.settled_at is not None
                    and monotonic() - item.settled_at > 300.0
                ]
                for key in stale:
                    self._exchanges.pop(key, None)
                if len(self._exchanges) >= 64:
                    settled_progress = sorted(
                        (item.settled_at or item.started_at, key)
                        for key, item in self._exchanges.items()
                        if item.future is not None
                        and item.future.done()
                        and item.request.operation == "maintenance.embeddings.backfill"
                    )
                    for _settled_at, key in settled_progress:
                        self._exchanges.pop(key, None)
                        if len(self._exchanges) < 64:
                            break
                if sum(item.future is None or not item.future.done() for item in self._exchanges.values()) >= 64:
                    return admission_refusal("operation_capacity")
                exchange = _Exchange(
                    request,
                    context,
                    deadline,
                    None if deadline is None else int((time() + max(0, deadline - monotonic())) * 1000),
                    started_at=started,
                )
                if recovered_custody is not None:
                    # Acceptance belongs to the durable request, including a
                    # resumed handler that has not observed its first snapshot.
                    exchange.snapshot, exchange.binding, recovered_record = recovered_custody
                    exchange.acceptance_started = True
                    exchange.accepted_reference = AcceptedOperationReference.from_record(recovered_record).to_dict()
                self._exchanges[request_id] = exchange
                if read_control is not None:
                    exchange.cancellation.add_listener(read_control.cancel)

                def work() -> DaemonOperationEnvelope:
                    exchange.queue_ms = max(0, int((monotonic() - started) * 1000))
                    if exchange.cancellation.cancelled:
                        raise BeforeAcceptanceCancelledError("operation cancelled before dispatch")
                    return execute_operation(request, context)

                try:
                    if request.operation in _STAGED_OPERATIONS:
                        if self._owner_loop is None:
                            self._exchanges.pop(request_id)
                            if recovered_custody is not None:
                                return admission_refusal("ingest_runtime_unavailable")
                            return operation_envelope(
                                request,
                                context,
                                outcome="rejected",
                                error={"code": "ingest_runtime_unavailable", "retryable": False},
                            ).to_dict()
                        from polylogue.daemon.embedding_owner import execute_embedding_backfill_operation
                        from polylogue.operations.archive_backup import (
                            execute_backup_operation,
                            execute_restore_verified_backup_operation,
                        )
                        from polylogue.operations.daemon_excision import execute_session_excision_operation
                        from polylogue.operations.daemon_ingest import execute_ingest_operation
                        from polylogue.operations.daemon_insights import execute_insights_rebuild_operation
                        from polylogue.operations.daemon_mutations import (
                            execute_raw_authority_blocker_resolve_operation,
                            execute_raw_authority_frontier_operation,
                            execute_selected_preview_operation,
                            execute_session_mark_operation,
                        )
                        from polylogue.operations.facade_writers import facade_record_work_event

                        staged = {
                            "ingest": execute_ingest_operation,
                            "mutation.session.excision": execute_session_excision_operation,
                            "mutation.facade.record_work_event": facade_record_work_event,
                            "maintenance.raw-authority-frontier": execute_raw_authority_frontier_operation,
                            "mutation.raw-authority-blocker.resolve": execute_raw_authority_blocker_resolve_operation,
                            "mutation.session.delete.preview": execute_selected_preview_operation,
                            "mutation.identity-reset.preview": execute_selected_preview_operation,
                            "mutation.session.mark": execute_session_mark_operation,
                            "maintenance.insights.rebuild": execute_insights_rebuild_operation,
                            "maintenance.embeddings.backfill": execute_embedding_backfill_operation,
                            "maintenance.backup": execute_backup_operation,
                            "maintenance.restore_verified_backup": execute_restore_verified_backup_operation,
                        }[request.operation]

                        staged_body_bytes = request_body_bytes

                        async def run_staged() -> Any:
                            _STAGED_REQUEST_BYTES.set(staged_body_bytes)
                            return await staged(request, context)

                        staged_task = StagedTask(self._owner_loop, run_staged)
                        exchange.future = staged_task.future

                        def cancel_staged_before_acceptance() -> None:
                            with self._condition:
                                if not exchange.acceptance_started or request.operation in {
                                    "maintenance.raw-authority-frontier",
                                    "mutation.raw-authority-blocker.resolve",
                                }:
                                    staged_task.cancel()

                        exchange.cancellation.add_listener(cancel_staged_before_acceptance)
                    else:
                        scheduled = self._kernel.submit(
                            propagate(work),
                            estimated_bytes=request_body_bytes,
                            admission_class=(
                                "bulk-candidate"
                                if archive_scan
                                else "interactive-read"
                                if spec.authority is DaemonAuthority.READ
                                else "control"
                            ),
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
                    return admission_refusal("compute_backpressure")

                def settled(_future: Future[DaemonOperationEnvelope]) -> None:
                    with self._condition:
                        exchange.settled_at = monotonic()
                        retain_progress = request.operation == "maintenance.embeddings.backfill"
                        retain_terminal = self._retains_terminal(exchange)
                        if retain_terminal:
                            try:
                                self._retain_terminal(exchange)
                            except Exception as exc:
                                exchange.terminal_transfer_error = type(exc).__name__
                                emit("operation.result.transfer_failed", level=WARNING, error_type=type(exc).__name__)
                                retain_progress = True
                        if self._exchanges.get(request_id) is exchange and not retain_progress:
                            self._exchanges.pop(request_id)
                        self._condition.notify_all()

                if input_body is not None:
                    input_body.adopted = True
                    exchange.future.add_done_callback(lambda _future: input_body.discard())
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
                if record is not None:
                    exchange.accepted_reference = AcceptedOperationReference.from_record(record).to_dict()
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
                if spec.progress and record is not None:
                    # Progress-enabled requests always hand the CLI its durable
                    # reference first. Even a very fast owner must leave the
                    # first operation.await exchange available to drain frames.
                    return self._pending_envelope(exchange, outcome="accepted", record=record)
                if exchange.future.done():
                    try:
                        envelope = exchange.future.result().to_dict()
                    except CancelledError:
                        # The staged task cancels its proxy only after its own
                        # cleanup finishes. That proves settlement, not absence
                        # of effects once acceptance may have started.
                        envelope = self._pending_envelope(
                            exchange,
                            outcome="indeterminate" if exchange.acceptance_started else "cancelled",
                            record=record,
                        )
                    except BeforeAcceptanceCancelledError:
                        envelope = self._pending_envelope(exchange, outcome="cancelled")
                    except DaemonOperationCancelled:
                        # The scheduler cancels a queued, pre-acceptance task by
                        # completing its future with this error, having already
                        # released the reservation with no work started. Without
                        # this branch it fell into the generic handler below and
                        # a clean cancellation was reported to the caller as
                        # ``failed`` carrying a DaemonOperationCancelled error.
                        envelope = self._pending_envelope(exchange, outcome="cancelled")
                    except Exception as exc:
                        retryable_admission = (
                            isinstance(exc, DaemonBackpressureError) and not exchange.acceptance_started
                        )
                        outcome = (
                            "indeterminate"
                            if exchange.acceptance_started
                            else "rejected"
                            if retryable_admission
                            else "failed"
                        )
                        envelope = self._pending_envelope(exchange, outcome=outcome, record=record)
                        envelope["error"] = {
                            "code": str(getattr(exc, "code", type(exc).__name__)),
                            "detail": str(exc),
                            "retryable": retryable_admission,
                            "data": getattr(exc, "data", {}),
                        }
                    if exchange.acceptance_started and exchange.binding is not None:
                        audit = AuditRepository.for_archive_root(self.archive_root)
                        try:
                            with audit.settled_machine_read():
                                record = audit.machine_request(exchange.binding)
                                if (
                                    record is None
                                    and envelope.get("outcome") == "indeterminate"
                                    and request.operation != "mutation.session.excision"
                                ):
                                    # The actual worker settled and continuity
                                    # proves there is no accepted domain work.
                                    envelope["outcome"] = "failed"
                        except AuditContinuityError:
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
                        envelope["authority_snapshot"] = {**authority_snapshot, **timing}
                    return envelope
                if record is not None:
                    return self._pending_envelope(exchange, outcome="accepted", record=record)
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    if not exchange.acceptance_started:
                        exchange.cancellation.cancel()
                        return self._pending_envelope(exchange, outcome="timed-out")
                    return self._pending_envelope(exchange, outcome="indeterminate")
                self._condition.wait(timeout=remaining)

    def _result_document_page(
        self,
        request: DaemonOperationRequest,
        principal: MutationPrincipal,
        archive_identity: str,
        snapshot: OperationControlRead,
    ) -> OperationControlResult:
        target = str(request.payload["request_id"])
        with self._condition:
            owned = self._exchanges.get(target)
            if owned is not None:
                if self._owned_retained_terminal_state(target, principal, archive_identity) is None:
                    raise ValueError("operation_result_document_not_settled")
                if owned.future is None or not owned.future.done() or not owned.acceptance_started:
                    raise ValueError("operation_result_document_not_settled")
                held_document = owned.result_document
                if not isinstance(held_document, dict):
                    raise ValueError("operation_result_document_missing")
                if any(request.payload[key] != held_document[key] for key in ("request_id", "byte_length", "sha256")):
                    raise ValueError("operation_result_document_identity_mismatch")
                self._retain_terminal(owned)
                owned.terminal_transfer_error = None
            packet = self._read_retained_terminal(target, principal, archive_identity)
            if packet is None:
                raise ValueError("operation_result_document_expired")
            document = packet.get("result_document")
            if not isinstance(document, dict):
                raise ValueError("operation_result_document_missing")
            if any(request.payload[key] != document[key] for key in ("request_id", "byte_length", "sha256")):
                raise ValueError("operation_result_document_identity_mismatch")
            offset = _operation_int(request.payload.get("offset", 0), field="result offset")
            length = int(document["byte_length"])
            if offset > length:
                raise ValueError("operation_result_document_cursor_invalid")
            path = self._terminal_path(target)
            assert path is not None
            try:
                with path.with_suffix(".product.json").open("rb") as stream:
                    if os.fstat(stream.fileno()).st_size != length:
                        raise ValueError("operation_result_document_corrupt")
                    stream.seek(offset)
                    chunk = stream.read(min(64 * 1024, length - offset))
            except FileNotFoundError as exc:
                raise ValueError("operation_result_document_missing") from exc
            if len(chunk) != min(64 * 1024, length - offset):
                raise ValueError("operation_result_document_corrupt")
            end = offset + len(chunk)
            return OperationControlResult(
                {
                    "document": document,
                    "offset": offset,
                    "data_base64": base64.b64encode(chunk).decode("ascii"),
                    "next_offset": end if end < length else None,
                },
                snapshot,
            )

    def control(
        self,
        request: DaemonOperationRequest,
        principal: MutationPrincipal,
        archive_identity: str,
        *,
        execution_context: QueryExecutionContext | None = None,
    ) -> OperationControlResult:
        if request.operation == "operation.result":
            if execution_context is not None:
                from polylogue.operations.operation_context import abort_checkpoint

                abort_checkpoint(execution_context)()
            document_snapshot = self._unbound_control_snapshot(archive_identity)
            return self._result_document_page(request, principal, archive_identity, document_snapshot)
        target = str(request.payload["request_id"])
        deadline = monotonic() + min(30.0, _operation_int(request.payload.get("timeout_ms", 0), field="timeout") / 1000)
        if execution_context is not None and execution_context.deadline_monotonic is not None:
            deadline = min(deadline, execution_context.deadline_monotonic)
        after = _operation_int(request.payload.get("after_sequence", 0), field="after sequence")
        after_progress = _operation_int(
            request.payload.get("after_progress_sequence", 0), field="after progress sequence"
        )
        audit = AuditRepository.for_archive_root(self.archive_root)
        if execution_context is not None:
            if execution_context.cancelled:
                raise QueryCancelledError("operation control exchange disconnected")
            if request.operation != "operation.await" and execution_context.deadline_exceeded():
                raise QueryTimeoutError("operation control exchange deadline expired")
        # Await's deadline limits waiting, not the one lifecycle read owed to
        # an accepted operation. Even an already-expired poll authenticates
        # its reference and reads the actual state before returning below.
        # Set only when *this* request cancelled a live, pre-acceptance
        # exchange. The scheduler completes a queued task's future
        # synchronously inside ``cancel()`` and the runtime's ``settled``
        # callback removes the exchange in the same call, so by the time the
        # loop below looks there is neither a live exchange nor a durable
        # record. Reading that absence as ``operation_reference_unknown`` told
        # the caller there is no such operation -- about the operation it had
        # just cancelled. Nothing is committed before acceptance, so the
        # cancellation is the whole outcome.
        with self._condition:
            if self._closing:
                raise QueryCancelledError("operation runtime is stopping")
            retained = self._read_retained_terminal(target, principal, archive_identity)
            if retained is not None:
                return OperationControlResult(
                    self._retained_control_state(cast("dict[str, object]", retained["envelope"])),
                    self._unbound_control_snapshot(archive_identity),
                )
            owned_terminal = self._owned_retained_terminal_state(target, principal, archive_identity)
            if owned_terminal is not None:
                return OperationControlResult(owned_terminal, self._unbound_control_snapshot(archive_identity))
        cancelled_before_acceptance = False
        if request.operation == "operation.cancel":
            with self._condition:
                exchange = self._exchanges.get(target)
                if exchange is not None:
                    if exchange.context.principal != principal:
                        raise PermissionError("operation reference belongs to another principal")
                    exchange.cancellation.cancel()
                    self._condition.notify_all()
                    # A retry resend of the original request creates its own
                    # exchange entry, so ``exchange is None`` alone cannot
                    # gate the durable fence: a resumed ingest crossed
                    # ``before_machine_prepare`` on an earlier attempt and
                    # never re-enters this exchange's own acceptance path,
                    # so a live re-drive can still be running this exact
                    # target. Cancelling only the exchange's reporting
                    # coroutine leaves that re-drive unfenced and unnotified,
                    # and it can still materialize and finalize the
                    # generation after the client believes it cancelled.
                    #
                    # Acceptance is therefore read from the durable request,
                    # never from this exchange's own flag: a resend of an
                    # accepted request reads its record and never sets it.
                    live_exchange = True
                    acceptance_in_flight = exchange.acceptance_started
                else:
                    live_exchange = False
                    acceptance_in_flight = False
            # A live exchange with no durable request was cancelled before
            # acceptance: nothing is committed, so no writer fence is queued.
            # Read without the writer, so a busy writer cannot turn that
            # cancellation into an indeterminate answer.
            # Once acceptance has started, its durable write may be in flight
            # (source prepared, audit not yet committed): absence is then
            # decided under the writer, after that write, never by a plain read.
            cancelled_before_acceptance = (
                live_exchange
                and not acceptance_in_flight
                and audit.machine_request_for_principal(archive_identity, target, principal.actor_ref) is None
            )
            if not cancelled_before_acceptance:
                # Queue the durable fence through the same writer owner, never
                # under the waiter lock.
                absent_after_serialization = [False]

                def fence() -> None:
                    record = audit.machine_request_for_principal(archive_identity, target, principal.actor_ref)
                    if record is None:
                        if live_exchange:
                            # Serialized after any in-flight acceptance: the
                            # cancellation won, and nothing was accepted.
                            absent_after_serialization[0] = True
                            return
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
                        parts = audit.iter_machine_parts(binding)
                        if any(part["operation_id"] is None for part in parts) or (
                            record["artifact_kind"] == "source-generation"
                            and machine_request_state(audit, record)["outcome"]
                            not in {"completed", "degraded", "failed"}
                        ):
                            audit.stop_machine_batch(binding, "cancelled")

                try:
                    self._bridge.run_sync_with_timeout("operation.cancel", 2.0, fence)
                    cancelled_before_acceptance = absent_after_serialization[0]
                    with self._condition:
                        # A running re-drive of this request observes the
                        # fence at its next stop check.
                        if self._redrive is not None and not self._redrive.done():
                            self._redrive_cancelled.add(target)
                except TimeoutError:
                    return OperationControlResult(
                        {"outcome": "indeterminate", "sequence": 0, "cancellation_requested": True},
                        None,
                    )
                self._notify()
        with self._condition:
            while True:
                # Once a cancellation fence starts, its actual receipt decides
                # the outcome. A late deadline cannot turn it into no-effect.
                if execution_context is not None and request.operation != "operation.cancel":
                    if execution_context.cancelled:
                        raise QueryCancelledError("operation control exchange disconnected")
                    if request.operation != "operation.await" and execution_context.deadline_exceeded():
                        raise QueryTimeoutError("operation control exchange deadline expired")
                if self._closing:
                    raise QueryCancelledError("operation runtime is stopping")
                exchange = self._exchanges.get(target)
                if exchange is not None and exchange.context.principal != principal:
                    raise PermissionError("operation reference belongs to another principal")
                retained = self._read_retained_terminal(target, principal, archive_identity)
                if retained is not None:
                    return OperationControlResult(
                        self._retained_control_state(cast("dict[str, object]", retained["envelope"])),
                        self._unbound_control_snapshot(archive_identity),
                    )
                pending = False
                snapshot: OperationControlRead | None = None
                try:
                    with open_operation_control(self.archive_root, audit=audit) as snapshot:
                        if snapshot.identity.authority_identity_digest != archive_identity:
                            raise ValueError("archive_identity_stale")
                        record = audit.machine_request_for_principal(archive_identity, target, principal.actor_ref)
                        state = (
                            machine_request_state(
                                audit,
                                record,
                                parts_offset=_operation_int(
                                    request.payload.get("parts_offset", 0), field="parts offset"
                                ),
                                parts_limit=_operation_int(request.payload.get("parts_limit", 40), field="parts limit"),
                            )
                            if record is not None
                            else None
                        )
                except AuditContinuityError:
                    pending = True
                    still_executing = (
                        exchange is not None and exchange.future is not None and not exchange.future.done()
                    )
                    if cancelled_before_acceptance and exchange is None:
                        state = {
                            "outcome": "cancelled",
                            "effect": "no-effect",
                            "cancellation_requested": True,
                            "sequence": 0,
                        }
                    else:
                        state = {"outcome": "running" if still_executing else "indeterminate", "sequence": 0}
                owned_terminal = self._owned_retained_terminal_state(target, principal, archive_identity)
                if owned_terminal is not None:
                    state = owned_terminal
                    pending = False
                if state is None:
                    unwinding = (
                        exchange is not None
                        and cancelled_before_acceptance
                        and exchange.request.operation in _STAGED_OPERATIONS
                    )
                    if exchange is None or unwinding:
                        if cancelled_before_acceptance:
                            # This request released an exchange that had
                            # committed nothing. A staged task still unwinding
                            # its cleanup cannot commit later: its acceptance
                            # fence reads the same cancellation under this lock.
                            return OperationControlResult(
                                {
                                    "outcome": "cancelled",
                                    "sequence": 0,
                                    "effect": "no-effect",
                                    "cancellation_requested": True,
                                },
                                snapshot,
                            )
                        raise ValueError("operation_reference_unknown")
                    state = {"outcome": "running", "sequence": 0}
                if exchange is not None and "reference" not in state and exchange.accepted_reference is not None:
                    # A concurrent audit continuity publication may make the
                    # settled read briefly unavailable while progress remains
                    # observable. Keep the wait bound to the already returned
                    # immutable acceptance reference; terminal state still
                    # comes from the next durable read.
                    state = {**state, "reference": exchange.accepted_reference}
                if request.operation != "operation.await":
                    return OperationControlResult(state, snapshot)
                sequence = _operation_int(state["sequence"], field="state sequence")
                progress_exchange = (
                    exchange is not None
                    and (target_spec := daemon_operation_spec(exchange.request.operation)) is not None
                    and target_spec.progress
                )
                if request.operation == "operation.await" and progress_exchange:
                    assert exchange is not None
                    progress = self._progress_state(exchange, after_progress)
                    if progress["progress_events"] or progress["progress_gap"] is not None:
                        return OperationControlResult({**state, **progress}, snapshot)
                    if not pending and state["outcome"] not in {"running", "accepted"}:
                        return OperationControlResult({**state, **progress}, snapshot)
                elif not pending and (sequence > after or state["outcome"] not in {"running", "accepted"}):
                    return OperationControlResult(state, snapshot)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return OperationControlResult(state, snapshot)
                self._condition.wait(timeout=remaining)

    async def recover_interrupted_operations(self, *, resolver_actor_ref: str) -> None:
        """Recover through this runtime's original creator and short writer admissions."""
        from polylogue.operations.mutation_replay import recover_interrupted_operations

        def recover() -> None:
            recover_interrupted_operations(
                self.archive_root,
                resolver_actor_ref=resolver_actor_ref,
                input_demand=self.prepared_compute_adapter().amend_current_input_demand,
            )

        await self.prepared_phase("recovery", recover, estimated_bytes=0, exclusive_bytes=True)
