"""The daemon's lease-free owner for archive embedding computation.

Embedding is the one derivation whose cost is a network round trip the daemon
does not control. Running it from inside the writer gate meant a slow or hung
provider blocked every unrelated archive publication for as long as the call
took, so this module runs the pass the other way round: the work is admitted to
the process's bounded compute capacity, where it holds neither the writer gate
nor the embedding generation lock, and each short write it needs -- attempt
reservation, one published window, the catch-up receipt -- is admitted back
through the daemon's write coordinator for exactly that write.

Two rules make that safe and are enforced here rather than documented:

* the compute worker is never the event loop thread, because
  :meth:`DaemonWriteThreadBridge.run_sync` blocks on the loop it is scheduling
  onto, and
* no second compute pool is created; the adapter is the one published for the
  process (:func:`polylogue.daemon.execution.daemon_compute_adapter`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeVar, cast

from polylogue.core.enums import OperationStatus
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
from polylogue.operations.audit import MachineRequestBinding
from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
from polylogue.operations.daemon_protocol import DaemonOperationEnvelope, DaemonOperationRequest
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import (
    MutationAuthorization,
    MutationPlan,
    MutationPreview,
    MutationReceipt,
    MutationTarget,
    MutationTargetStatus,
    build_typed_plan,
)
from polylogue.operations.operation_context import OperationContext, PinnedOperationRead, open_operation_read

if TYPE_CHECKING:
    from typing import SupportsFloat, SupportsInt

    from polylogue.daemon.derivation import DerivationReport

T = TypeVar("T")

__all__ = [
    "ComposedEmbeddingConvergence",
    "DaemonEmbeddingAdmission",
    "EmbeddingConvergenceResult",
    "compose_embedding_convergence",
    "execute_embedding_backfill_operation",
]


class _PassReceipt(TypedDict):
    run_id: str | None
    started_at_ms: int
    scanned_sessions: int
    reserved_cost_usd: float


class DaemonEmbeddingAdmission:
    """Admit one short embedding write through the daemon's write coordinator.

    Calling this from the event loop thread would deadlock: the bridge blocks
    the calling thread until the coordinator has run the operation on that same
    loop. The guard below turns that into a typed refusal instead of a hang.
    """

    def __init__(self, bridge: DaemonWriteThreadBridge, loop: asyncio.AbstractEventLoop) -> None:
        self._bridge = bridge
        self._loop_thread_id = threading.get_ident() if loop.is_running() else None

    def __call__(self, actor: str, function: Callable[[], T], /) -> T:
        if self._loop_thread_id is not None and threading.get_ident() == self._loop_thread_id:
            raise RuntimeError(
                f"embedding phase {actor} was admitted from the daemon event loop thread; "
                "lease-free embedding work must run on a compute worker"
            )
        # A ``None`` wait keeps daemon ownership until the phase returns its
        # receipt. Abandoning a reserved attempt or a completed publication on a
        # caller-side timeout would strand exactly the state this route exists
        # to keep consistent.
        return self._bridge.run_sync_with_timeout(actor, None, function)


@dataclass(frozen=True, slots=True)
class EmbeddingConvergenceResult:
    """One embedding pass, with policy deferral distinct from output readiness."""

    report: DerivationReport | None
    deferred_reason: str | None = None

    @property
    def converged(self) -> bool:
        return (
            self.deferred_reason is None
            and self.report is not None
            and self.report.pending == 0
            and self.report.failed == 0
        )


EmbeddingConvergenceCallback = Callable[[Sequence[str] | None], Awaitable[EmbeddingConvergenceResult]]

# Preserve the former daemon catch-up envelope at the message partition grain.
# A report budget counts provider computations, which is the paid unit here.
EMBEDDING_PASS_MAX_MESSAGES = 2_500
EMBEDDING_PASS_DEADLINE_S = 30.0


def _catchup_receipt_status(*, failures: int, pending: int, stopped: bool) -> OperationStatus:
    """Classify a catch-up receipt without treating bounded work as complete."""
    if failures:
        return OperationStatus.FAILED
    if pending or stopped:
        return OperationStatus.INTERRUPTED
    return OperationStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class ComposedEmbeddingConvergence:
    """One retained owner and adapter for the daemon's shared compute capacity."""

    callback: EmbeddingConvergenceCallback

    async def __call__(self, scope: Sequence[str] | None) -> EmbeddingConvergenceResult:
        return await self.callback(scope)


class _EmbeddingBackfillExecution:
    """Accepted-operation lifecycle for one operator embedding pass."""

    def __init__(self, request: DaemonOperationRequest, context: OperationContext) -> None:
        if context.runtime is None:
            raise PermissionError("daemon_required")
        self.request = request
        self.context = context
        self.runtime = context.runtime
        self.audit = self.runtime.audit_for_request(request, context)
        self.snapshot: PinnedOperationRead | None = None
        self.binding: MachineRequestBinding | None = None
        self.plan: MutationPlan | None = None
        self.preview_ref: str | None = None
        self.authorization: MutationAuthorization | None = None
        self.operation_id: str | None = None
        self.record: dict[str, object] | None = None
        self.scope: tuple[str, ...] | None = None
        self.scope_limited = False

    async def accept(self) -> None:
        """Pin authority and commit an auditable accepted reference before work."""
        from polylogue.operations.embedding_derivation import select_embedding_session_window

        def prepare() -> tuple[
            MachineRequestBinding,
            tuple[str, ...] | None,
            bool,
            dict[str, object] | None,
        ]:
            with open_operation_read(
                self.context.archive_root,
                publication_guard=self.runtime.publication_guard,
            ) as pinned:
                _validate_identity(self.request, self.context, pinned)
                self.snapshot = pinned
                self.runtime.observe_snapshot(self.request, pinned)
                binding = MachineRequestBinding(
                    pinned.identity.authority_identity_digest,
                    str(self.request.request_id),
                    self.context.principal.actor_ref,
                    self.request.fingerprint,
                    self.request.operation,
                )
                with self.audit.settled_machine_read():
                    existing = self.audit.machine_request(binding)
                payload = self.request.payload
                scope = None
                if (
                    payload.get("max_sessions") is not None
                    or payload.get("max_messages") is not None
                    or payload.get("min_messages") is not None
                    or bool(payload.get("rebuild"))
                ):
                    scope, scope_limited = select_embedding_session_window(
                        pinned.archive.index_db_path,
                        archive_root=self.context.archive_root,
                        rebuild=bool(payload.get("rebuild")),
                        max_sessions=cast(int | None, payload.get("max_sessions")),
                        max_messages=cast(int | None, payload.get("max_messages")),
                        min_messages=cast(int | None, payload.get("min_messages")),
                    )
                else:
                    scope_limited = False
                return binding, scope, scope_limited, existing

        binding, scope, scope_limited, existing = await self.runtime.compute_phase(prepare)
        self.binding, self.scope, self.scope_limited, self.record = binding, scope, scope_limited, existing
        if existing is not None:
            self.operation_id = str(self.audit.machine_parts(binding)[0]["operation_id"])
            return

        payload = dict(self.request.payload)

        def commit_acceptance() -> None:
            assert self.binding is not None and self.snapshot is not None
            archive_instance_id = self.audit.ensure_archive_authority(now_ms=int(time.time() * 1000))
            target_ref = f"embedding-pass:{self.request.request_id}"
            identity_digest = hashlib.sha256(target_ref.encode()).hexdigest()
            target = MutationTarget(
                kind="embedding-pass",
                ref=target_ref,
                policy_key="embedding-backfill",
                identity_digest=identity_digest,
                effect_identity=f"{self.request.operation}:{target_ref}",
                durability="derived",
                recovery="retry_convergent",
            )
            parameter_digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
            ).hexdigest()
            self.plan = build_typed_plan(
                operation=self.request.operation,
                operation_version=1,
                archive_instance_id=archive_instance_id,
                archive_identity_digest=self.binding.archive_identity,
                targets=(target,),
                affected_tiers=("embeddings", "ops"),
                parameter_digest=parameter_digest,
                required_capabilities=("archive.embeddings.backfill",),
                destructive_class="maintenance",
                required_confirmation="role_only",
                prepared_at_ms=int(time.time() * 1000),
                expires_at_ms=self.runtime.request_deadline_unix_ms(self.request),
                context={
                    **payload,
                    "scope": list(self.scope) if self.scope is not None else None,
                    "scope_limited": self.scope_limited,
                },
            )
            preview = MutationPreview("pending-preview", self.plan)
            self.preview_ref = self.audit.create_preview(self.plan, self.context.principal)
            preview = MutationPreview(self.preview_ref, self.plan)
            token = secrets.token_urlsafe(32)
            authorization = MutationAuthorization(
                plan_hash=self.plan.plan_hash,
                actor=self.context.principal.actor_ref,
                role=self.context.principal.role_label or "",
                capability="archive.embeddings.backfill",
                confirmation_strength="role_only",
                authorized_at=str(int(time.time() * 1000)),
                preview_ref=self.preview_ref,
                token=token,
                expires_at_ms=self.plan.expires_at_ms,
                capabilities=("archive.embeddings.backfill",),
                surface=self.context.principal.surface,
            )
            auth_ref = self.audit.issue_authorization(preview, self.context.principal, authorization)
            self.authorization = replace(authorization, authorization_id=str(auth_ref), token=None)
            with self.audit.bind_machine_request(
                self.binding,
                transition="consume_authorization_and_start",
            ):
                self.operation_id = self.audit.consume_authorization_and_start(preview, self.authorization)
            self.record = self.audit.machine_request(self.binding)

        await self.runtime.write_phase("embedding.accept", commit_acceptance)

    async def stop(self, reason: str) -> None:
        if self.binding is not None and self.record is not None:
            binding = self.binding
            await self.runtime.write_phase("embedding.stop", lambda: self.audit.stop_machine_batch(binding, reason))

    async def finalize(self, payload: dict[str, object], *, status: str = "applied") -> None:
        if self.operation_id is None or self.plan is None:
            return
        receipt = MutationReceipt(
            operation=self.plan.operation,
            plan_hash=self.plan.plan_hash,
            status=cast(MutationTargetStatus, status),
            target_refs=self.plan.target_refs,
            affected_count=1 if status == "applied" else 0,
            detail=None,
            receipt_ref=None,
            applied_at=str(int(time.time() * 1000)),
        )
        summary = "embedding_receipt:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        await self.runtime.write_phase(
            "embedding.finalize",
            lambda: self.audit.finalize_attempt(
                str(self.operation_id), status=status, receipt=receipt, error_summary=summary
            ),
        )

    async def state(self) -> dict[str, object]:
        assert self.binding is not None
        binding = self.binding

        def read() -> dict[str, object]:
            with self.audit.settled_machine_read():
                record = self.audit.machine_request(binding)
                if record is None:
                    raise ValueError("embedding request has no durable binding")
                self.record = record
                return machine_request_state(self.audit, record)

        return await self.runtime.compute_phase(read)


def compose_embedding_convergence(
    index_db_path: Path,
    *,
    compute_adapter: BoundedComputeAdapter,
    write_bridge: DaemonWriteThreadBridge,
    quiet: Callable[[], bool] | None = None,
    max_messages: int | None = None,
    max_cost_usd: float | None = None,
    stop_after_seconds: int | None = None,
    max_errors: int | None = None,
    scope_limited: bool = False,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> ComposedEmbeddingConvergence:
    """Compose the common-kernel embedding owner once for a daemon process.

    The callback keeps the legacy daemon envelope: no more than 2,500 message
    computations and 30 seconds per pass, with the remaining monthly estimate
    converted into a smaller compute budget before a provider call can start.
    Limits produce pending work or a typed policy deferral; they never certify
    output from a receipt or a cursor.
    """

    from polylogue.config import load_polylogue_config
    from polylogue.daemon.convergence import DaemonConverger, DerivationConvergenceOwner
    from polylogue.daemon.derivation import Budget, Outcome
    from polylogue.operations.embedding_derivation import (
        estimated_embedding_message_cost,
        make_embedding_derivation,
        make_embedding_frame,
    )

    archive_root = index_db_path.parent
    loop = asyncio.get_running_loop()
    admission = DaemonEmbeddingAdmission(write_bridge, loop)
    cfg = load_polylogue_config()
    if not bool(cfg.embedding_enabled):

        async def disabled(_scope: Sequence[str] | None) -> EmbeddingConvergenceResult:
            return EmbeddingConvergenceResult(None, "disabled")

        return ComposedEmbeddingConvergence(disabled)
    voyage_key = cfg.get("voyage_api_key")
    if not voyage_key:

        async def no_key(_scope: Sequence[str] | None) -> EmbeddingConvergenceResult:
            return EmbeddingConvergenceResult(None, "provider_unavailable")

        return ComposedEmbeddingConvergence(no_key)
    monthly_cap = float(str(cfg.get("embedding_max_cost_usd", 0.0)))
    estimated_cost_per_message = estimated_embedding_message_cost()
    pass_lock = asyncio.Lock()
    receipt_lock = threading.Lock()
    active_receipt: _PassReceipt | None = None

    def reserve(actor: str, function: Callable[[], T], /) -> T:
        """Create one conservative spend reservation before the first provider call."""

        nonlocal active_receipt
        with receipt_lock:
            receipt = active_receipt
            if receipt is None:
                raise RuntimeError("embedding reservation has no active convergence pass")
            if receipt["run_id"] is None:
                from polylogue.core.enums import OperationStatus
                from polylogue.daemon.embedding_backlog import _upsert_archive_embedding_catchup_run

                receipt["run_id"] = admission(
                    "embedding.catchup_receipt",
                    partial(
                        _upsert_archive_embedding_catchup_run,
                        archive_root / "ops.db",
                        status=OperationStatus.RUNNING,
                        started_at_ms=int(receipt["started_at_ms"]),
                        scanned_sessions=int(receipt["scanned_sessions"]),
                        estimated_cost_usd=float(receipt["reserved_cost_usd"]),
                    ),
                )
        return admission(actor, function)

    progress_count = 0

    def observe_progress(event: Mapping[str, object]) -> None:
        """Forward intermediate work without making it an output receipt."""

        nonlocal progress_count
        progress_count += 1
        if progress_callback is not None:
            progress_callback(
                {
                    **dict(event),
                    "sequence": progress_count,
                    "started_count": progress_count,
                    "estimated_cost_usd": progress_count * estimated_cost_per_message,
                }
            )

    adapter = make_embedding_derivation(
        index_db_path,
        voyage_api_key=str(voyage_key),
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
        archive_root=archive_root,
        reserve=reserve,
        quiet=quiet,
        progress_callback=observe_progress,
    )
    if adapter is None:

        async def unavailable(_scope: Sequence[str] | None) -> EmbeddingConvergenceResult:
            return EmbeddingConvergenceResult(None, "provider_unavailable")

        return ComposedEmbeddingConvergence(unavailable)
    owner = DerivationConvergenceOwner(
        DaemonConverger(stages=(), derivations=(adapter,)),
        compute_adapter=compute_adapter,
        write_bridge=write_bridge,
    )

    async def converge(scope: Sequence[str] | None) -> EmbeddingConvergenceResult:
        nonlocal active_receipt, progress_count
        async with pass_lock:
            progress_count = 0
            compute_budget = EMBEDDING_PASS_MAX_MESSAGES
            if max_messages is not None:
                compute_budget = min(compute_budget, max_messages)
            if max_cost_usd is not None:
                compute_budget = min(compute_budget, max(0, int(max_cost_usd / estimated_cost_per_message)))
            if monthly_cap > 0.0:
                from polylogue.daemon.embedding_backlog import _archive_embedding_catchup_estimated_cost_this_month

                spent = _archive_embedding_catchup_estimated_cost_this_month(archive_root / "ops.db")
                remaining = monthly_cap - spent
                compute_budget = min(compute_budget, max(0, int(remaining / estimated_cost_per_message)))
                if compute_budget <= 0:
                    return EmbeddingConvergenceResult(None, "monthly_cost_cap")
            receipt: _PassReceipt = {
                "run_id": None,
                "started_at_ms": int(time.time() * 1000),
                "scanned_sessions": len(tuple(scope or ())),
                # An interrupted pass keeps this conservative reserve, so a
                # restart cannot spend beyond the configured monthly cap.
                "reserved_cost_usd": compute_budget * estimated_cost_per_message,
            }
            with receipt_lock:
                active_receipt = receipt
            try:
                frame = make_embedding_frame(index_db_path, archive_root=archive_root, adapter=adapter, scope=scope)
                report = await owner.converge(
                    frame,
                    budget=Budget(
                        page=min(128, compute_budget),
                        discovery=compute_budget,
                        inspection=compute_budget,
                        compute=compute_budget,
                        publication=compute_budget,
                        retained_outcomes=compute_budget,
                        deadline_s=min(
                            EMBEDDING_PASS_DEADLINE_S,
                            float(stop_after_seconds) if stop_after_seconds is not None else EMBEDDING_PASS_DEADLINE_S,
                        ),
                    ),
                    domains=(adapter.domain,),
                )
                computed = report.work.computed
                run_id = receipt["run_id"]
                # Hoisted out of the ``run_id is not None`` branch below: the
                # max_errors deferral reads it unconditionally, so leaving it
                # bound only on the receipt path raised NameError whenever a
                # pass ran without a receipt run id.
                failures = report.count(Outcome.FAILED)
                if run_id is not None:
                    # Attempt rows are telemetry only.  This final estimate is
                    # deliberately conservative: a failed provider call can
                    # still be billable, while refs/meta/vector inspection is
                    # the sole readiness authority.
                    from polylogue.daemon.embedding_backlog import _upsert_archive_embedding_catchup_run

                    # A receipt with work left is not a completed catch-up.
                    # Preserve it as retryable interrupted debt even when no
                    # provider call failed (for example a cost or time cap).
                    # Cancellation/deadline can also arrive after the last
                    # derivation observation, so consult the request stop
                    # signal before declaring a clean completion.
                    stopped = scope_limited or bool(quiet and quiet())
                    receipt_status = _catchup_receipt_status(
                        failures=failures,
                        pending=report.pending,
                        stopped=stopped,
                    )

                    await write_bridge.run_async(
                        "embedding.catchup_receipt",
                        partial(
                            _upsert_archive_embedding_catchup_run,
                            archive_root / "ops.db",
                            run_id=str(run_id),
                            status=receipt_status,
                            started_at_ms=int(receipt["started_at_ms"]),
                            finished_at_ms=int(time.time() * 1000),
                            scanned_sessions=int(receipt["scanned_sessions"]),
                            error_count=failures,
                            embedded_messages=computed,
                            estimated_cost_usd=computed * estimated_cost_per_message,
                            error_message="embedding derivation key failures" if failures else None,
                        ),
                    )
                deferred = None
                if max_cost_usd is not None and compute_budget < EMBEDDING_PASS_MAX_MESSAGES and report.pending:
                    deferred = "cost_cap_exceeded"
                elif monthly_cap > 0.0 and compute_budget < EMBEDDING_PASS_MAX_MESSAGES and report.pending:
                    deferred = "monthly_cost_cap"
                elif stop_after_seconds is not None and report.pending:
                    deferred = "stop_after_seconds"
                elif max_errors is not None and failures >= max_errors:
                    deferred = "max_errors"
                elif scope_limited:
                    deferred = "max_sessions"
                return EmbeddingConvergenceResult(report, deferred)
            finally:
                with receipt_lock:
                    active_receipt = None

    return ComposedEmbeddingConvergence(converge)


async def execute_embedding_backfill_operation(
    request: DaemonOperationRequest, context: OperationContext
) -> DaemonOperationEnvelope:
    """Run an operator-requested pass through the resident embedding owner.

    The CLI supplies bounds as intent; provider credentials, tier writes, cost
    reservation, and the actual derivation remain daemon-owned.  The ambient
    backlog owner is deliberately reused so a manual pass cannot create a
    second embedding writer.
    """
    from polylogue.daemon.execution import daemon_compute_adapter
    from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge, daemon_write_coordinator
    from polylogue.operations.embedding_derivation import estimated_embedding_message_cost

    request = validate_execution_request(request, context)
    runtime = context.runtime
    owner_loop = getattr(runtime, "_owner_loop", None) if runtime is not None else None
    if runtime is None or owner_loop is None:
        raise PermissionError("daemon_required")
    root = context.archive_root
    execution = _EmbeddingBackfillExecution(request, context)
    try:
        await execution.accept()
        if execution.plan is None:
            # A replayed request id already has a durable accepted attempt.
            # Return its authoritative state instead of running provider jobs
            # a second time.
            state = await execution.state()
            return operation_envelope(
                request,
                context,
                snapshot=execution.snapshot,
                outcome=str(state["outcome"]),
                reference=execution.record,
                result=state.get("result", state),
            )
        payload = request.payload
        if bool(payload.get("rebuild")) and execution.record is not None:
            from polylogue.operations.embedding_derivation import mark_embedding_sessions_needs_reindex

            await runtime.write_phase(
                "embedding.rebuild-mark",
                lambda: mark_embedding_sessions_needs_reindex(
                    root / "index.db", embeddings_db_path=root / "embeddings.db"
                ),
            )

        def _bound(key: str) -> int | None:
            value = payload.get(key)
            return None if value is None else int(cast("SupportsInt", value))

        max_messages = _bound("max_messages")
        max_cost_usd = (
            None if payload.get("max_cost_usd") is None else float(cast("SupportsFloat", payload["max_cost_usd"]))
        )
        stop_after_seconds = _bound("stop_after_seconds")
        max_errors = _bound("max_errors")
        owner = compose_embedding_convergence(
            root / "index.db",
            compute_adapter=daemon_compute_adapter(),
            write_bridge=DaemonWriteThreadBridge(daemon_write_coordinator(), owner_loop),
            quiet=lambda: runtime.stop_reason(request) is not None,
            max_messages=max_messages,
            max_cost_usd=max_cost_usd,
            stop_after_seconds=stop_after_seconds,
            max_errors=max_errors,
            scope_limited=execution.scope_limited,
            progress_callback=lambda event: runtime.emit_progress(request, event),
        )
        result = await owner(execution.scope)
        report = result.report
        stop_reason = runtime.stop_reason(request) or result.deferred_reason
        terminal: dict[str, object] = {
            "operation": request.operation,
            "outcome": "completed"
            if stop_reason is None
            else ("cancelled" if stop_reason == "cancelled" else "stopped"),
            "sequence": 1,
            "effect": "committed" if report is not None and report.done else "no-effect",
            "affected_count": 0 if report is None else report.done,
            "stop_reason": stop_reason,
            "progress": {
                "state": "stopped" if stop_reason is not None else "complete",
                "computed": 0 if report is None else report.work.computed,
                "failed": 0 if report is None else report.failed,
                "estimated_cost_usd": (
                    0.0 if report is None else report.work.computed * estimated_embedding_message_cost()
                ),
            },
            "result": {
                "done": 0 if report is None else report.done,
                "pending": 0 if report is None else report.pending,
                "failed": 0 if report is None else report.failed,
            },
        }
        if stop_reason is not None:
            audit_reason = (
                "cancelled" if stop_reason == "cancelled" else "deadline" if stop_reason == "deadline" else "refused"
            )
            await execution.stop(audit_reason)
        await execution.finalize(terminal)
    except Exception as exc:
        if execution.operation_id is not None:
            await execution.stop("cancelled" if runtime.stop_reason(request) == "cancelled" else "refused")
            await execution.finalize(
                {
                    "operation": request.operation,
                    "outcome": "failed",
                    "sequence": 1,
                    "effect": "no-effect",
                    "stop_reason": "refused",
                    "error": str(exc)[:512],
                },
                status="failed",
            )
        raise
    state = await execution.state()
    return operation_envelope(
        request,
        context,
        snapshot=execution.snapshot,
        outcome=str(state["outcome"]),
        reference=execution.record,
        result=state.get("result", state),
    )
