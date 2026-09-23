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
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeVar

from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge

if TYPE_CHECKING:
    from polylogue.daemon.derivation import DerivationReport
    from polylogue.operations.daemon_protocol import DaemonOperationEnvelope, DaemonOperationRequest
    from polylogue.operations.operation_context import OperationContext

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


@dataclass(frozen=True, slots=True)
class ComposedEmbeddingConvergence:
    """One retained owner and adapter for the daemon's shared compute capacity."""

    callback: EmbeddingConvergenceCallback

    async def __call__(self, scope: Sequence[str] | None) -> EmbeddingConvergenceResult:
        return await self.callback(scope)


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

    adapter = make_embedding_derivation(
        index_db_path,
        voyage_api_key=str(voyage_key),
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
        archive_root=archive_root,
        reserve=reserve,
        quiet=quiet,
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
        nonlocal active_receipt
        async with pass_lock:
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
                if run_id is not None:
                    # Attempt rows are telemetry only.  This final estimate is
                    # deliberately conservative: a failed provider call can
                    # still be billable, while refs/meta/vector inspection is
                    # the sole readiness authority.
                    from polylogue.core.enums import OperationStatus
                    from polylogue.daemon.embedding_backlog import _upsert_archive_embedding_catchup_run

                    failures = report.count(Outcome.FAILED)
                    await write_bridge.run_async(
                        "embedding.catchup_receipt",
                        partial(
                            _upsert_archive_embedding_catchup_run,
                            archive_root / "ops.db",
                            run_id=str(run_id),
                            status=OperationStatus.FAILED if failures else OperationStatus.COMPLETED,
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
    from polylogue.operations.daemon_execution import operation_envelope

    runtime = context.runtime
    owner_loop = getattr(runtime, "_owner_loop", None) if runtime is not None else None
    if runtime is None or owner_loop is None:
        raise PermissionError("daemon_required")
    root = context.archive_root
    payload = request.payload
    max_sessions = payload.get("max_sessions")
    max_messages = payload.get("max_messages")
    max_cost_usd = payload.get("max_cost_usd")
    min_messages = payload.get("min_messages")
    stop_after_seconds = payload.get("stop_after_seconds")
    max_errors = payload.get("max_errors")

    # Resolve the bounded session window on the daemon's read side.  The
    # resulting ids are only intent; all embedding writes still go through the
    # resident owner and its write coordinator.
    scope: tuple[str, ...] | None = None
    if max_sessions is not None or max_messages is not None or min_messages is not None or bool(payload.get("rebuild")):
        from polylogue.operations.embedding_derivation import (
            select_embedding_session_window,
        )

        scope = select_embedding_session_window(
            root / "index.db",
            archive_root=root,
            rebuild=bool(payload.get("rebuild")),
            max_sessions=int(max_sessions) if max_sessions is not None else None,
            max_messages=int(max_messages) if max_messages is not None else None,
            min_messages=int(min_messages) if min_messages is not None else None,
        )

    owner = compose_embedding_convergence(
        root / "index.db",
        compute_adapter=daemon_compute_adapter(),
        write_bridge=DaemonWriteThreadBridge(daemon_write_coordinator(), owner_loop),
        max_messages=int(max_messages) if max_messages is not None else None,
        max_cost_usd=float(max_cost_usd) if max_cost_usd is not None else None,
        stop_after_seconds=int(stop_after_seconds) if stop_after_seconds is not None else None,
        max_errors=int(max_errors) if max_errors is not None else None,
    )
    result = await owner(scope)
    from polylogue.operations.embedding_derivation import estimated_embedding_message_cost

    report = result.report
    payload = {
        "operation": request.operation,
        "outcome": "completed" if result.deferred_reason is None else "stopped",
        "sequence": 1,
        "effect": "committed" if report is not None and report.done else "no-effect",
        "affected_count": 0 if report is None else report.done,
        "stop_reason": result.deferred_reason,
        "progress": {
            "state": "stopped" if result.deferred_reason is not None else "complete",
            "computed": 0 if report is None else report.work.computed,
            "failed": 0 if report is None else report.failed,
            "cost_usd": 0.0 if report is None else report.work.computed * estimated_embedding_message_cost(),
        },
        "result": {
            "done": 0 if report is None else report.done,
            "pending": 0 if report is None else report.pending,
            "failed": 0 if report is None else report.failed,
        },
    }
    return operation_envelope(request, context, result=payload)
