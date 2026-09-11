"""Canonical typed machine execution, shared by direct and transport adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Protocol, TypeVar

from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext, QueryTimeoutError
from polylogue.operations.audit import AuditRepository, MachineRequestRecoveredError
from polylogue.operations.daemon_protocol import (
    AuthoritySnapshot,
    DaemonAuthority,
    DaemonOperationEnvelope,
    DaemonOperationRequest,
    daemon_operation_spec,
    validate_operation_result,
)
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.operations.operation_context import (
    OperationContext,
    OperationControlRead,
    PinnedOperationRead,
    observe_control_authority,
    open_operation_read,
)
from polylogue.version import POLYLOGUE_VERSION

_T = TypeVar("_T")

if TYPE_CHECKING:
    from polylogue.operations.insight_acceptance import AcceptedInsightPart, SessionInsightPartReceipt


class OperationRuntime(Protocol):
    """Resident authority injected into the product executor by its owner."""

    def publication_guard(self) -> AbstractContextManager[None]: ...

    def run_write(self, name: str, work: Callable[[], _T]) -> _T: ...

    async def compute_phase(self, work: Callable[[], _T]) -> _T: ...

    async def write_phase(self, name: str, work: Callable[[], _T]) -> _T: ...

    def require_session_maintenance(self) -> None: ...

    def session_profile_plan_binding(self, *, opened_index_path: Path) -> tuple[str, str]: ...

    async def converge_ingest_sessions(
        self,
        request: DaemonOperationRequest,
        session_ids: tuple[str, ...],
        *,
        expected_recipe: str,
        stop_requested: Callable[[], str | None],
    ) -> SessionInsightPartReceipt: ...

    async def converge_insight_part(
        self, request: DaemonOperationRequest, part: AcceptedInsightPart, *, stop_requested: Callable[[], str | None]
    ) -> SessionInsightPartReceipt: ...

    def audit_for_request(self, request: DaemonOperationRequest, context: OperationContext) -> AuditRepository: ...

    def control(
        self, request: DaemonOperationRequest, principal: MutationPrincipal, archive_identity: str
    ) -> dict[str, object]: ...

    def observe_snapshot(self, request: DaemonOperationRequest, snapshot: PinnedOperationRead) -> None: ...

    def request_deadline_unix_ms(self, request: DaemonOperationRequest) -> int: ...

    def stop_reason(self, request: DaemonOperationRequest) -> str | None: ...


def operation_envelope(
    request: DaemonOperationRequest,
    context: OperationContext,
    *,
    snapshot: PinnedOperationRead | OperationControlRead | None = None,
    started_at: float | None = None,
    queue_ms: int = 0,
    result: object = None,
    outcome: str = "completed",
    error: dict[str, object] | None = None,
    reference: dict[str, object] | None = None,
) -> DaemonOperationEnvelope:
    spec = daemon_operation_spec(request.operation)
    elapsed = max(0, int((monotonic() - started_at) * 1000)) if started_at is not None else 0
    identity = snapshot.identity if snapshot is not None else None
    versions = snapshot.schema_versions if snapshot is not None else {}
    degraded = snapshot.degraded_components if snapshot is not None else ("authority_unavailable",)
    authority = AuthoritySnapshot(
        archive_identity=identity.authority_identity_digest if identity is not None else "unavailable",
        generation=identity.active_generation if identity is not None else "unavailable",
        schema_versions=versions,
        served_by=context.serving_identity,
        elapsed_ms=elapsed,
        queue_ms=queue_ms,
        degraded_components=degraded,
    )
    return DaemonOperationEnvelope(
        operation=request.operation,
        archive={
            "root": str(context.archive_root),
            "archive_identity": authority.archive_identity,
            "daemon_version": POLYLOGUE_VERSION,
            "index_schema_version": versions.get("index"),
            "tier_schema_versions": versions,
        },
        generation={"id": authority.generation, "tier_schema_versions": versions},
        readiness={
            "ready": snapshot is not None and not degraded,
            "state": "ready" if snapshot is not None and not degraded else "degraded",
            "degraded_components": list(degraded),
        },
        authority={
            "mode": context.serving_identity,
            "class": spec.authority.value if spec is not None else "unavailable",
            "fallback": spec.fallback.value if spec is not None else "never",
            "writes": "daemon-owned",
        },
        progress={"state": "complete" if outcome == "completed" else outcome},
        outcome=outcome,
        served_by={"identity": context.serving_identity, "daemon_version": POLYLOGUE_VERSION},
        timing={"elapsed_ms": elapsed, "queue_ms": queue_ms},
        degraded_components=degraded,
        schema_versions=versions,
        result=result,
        error=error,
        request_id=request.request_id,
        accepted_reference=reference,
        authority_snapshot=authority.to_dict(),
    )


def _validate_identity(
    request: DaemonOperationRequest, context: OperationContext, snapshot: PinnedOperationRead | OperationControlRead
) -> None:
    if request.archive_root is not None and context.archive_root.resolve() != Path(request.archive_root).resolve():
        raise ValueError("archive_identity_mismatch")
    if request.expected_archive_identity not in (None, snapshot.identity.authority_identity_digest):
        raise ValueError("archive_identity_stale")
    if request.expected_generation_id not in (None, snapshot.identity.active_generation):
        raise ValueError("generation_stale")
    if request.index_schema_version not in (None, snapshot.schema_versions.get("index")):
        raise ValueError("schema_version_mismatch")
    if request.daemon_version not in (None, POLYLOGUE_VERSION):
        raise ValueError("daemon_version_mismatch")


def validate_execution_request(request: DaemonOperationRequest, context: OperationContext) -> DaemonOperationRequest:
    """Apply the same admission contract before synchronous or staged execution."""
    request = DaemonOperationRequest.from_dict(request.to_dict())
    spec = daemon_operation_spec(request.operation)
    assert spec is not None
    if len(json.dumps(request.to_dict(), separators=(",", ":")).encode()) > spec.max_body_bytes:
        raise ValueError("request_too_large")
    if spec.capability not in context.principal.capabilities:
        raise PermissionError(f"operation requires capability {spec.capability}")
    if context.runtime is None and not spec.direct_allowed:
        raise PermissionError("daemon_required")
    return request


def execute_operation(request: DaemonOperationRequest, context: OperationContext) -> DaemonOperationEnvelope:
    """Validate and execute the declared operation against explicit authority."""

    started = monotonic()
    snapshot: PinnedOperationRead | None = None
    try:
        request = validate_execution_request(request, context)
        spec = daemon_operation_spec(request.operation)
        assert spec is not None
        if request.operation.startswith("operation."):
            assert context.runtime is not None
            control_snapshot = observe_control_authority(context.archive_root)
            _validate_identity(request, context, control_snapshot)
            result = context.runtime.control(
                request, context.principal, control_snapshot.identity.authority_identity_digest
            )
            validate_operation_result(request.operation, result)
            return operation_envelope(request, context, snapshot=control_snapshot, started_at=started, result=result)

        def execute(*, mutating: bool) -> DaemonOperationEnvelope:
            nonlocal snapshot
            guard = (nullcontext if mutating else context.runtime.publication_guard) if context.runtime else None
            from polylogue.operations.daemon_reads import requires_vector_snapshot

            vector_binding = context.read_dependencies.vector_binding if context.read_dependencies is not None else None
            vector_model = (
                vector_binding.model
                if vector_binding is not None and requires_vector_snapshot(request.operation, request.payload)
                else None
            )
            read_control = (
                None
                if mutating
                else context.read_control
                or QueryExecutionContext(
                    call_id=str(request.request_id),
                    query_ref=request.fingerprint,
                    deadline_monotonic=started + min(spec.deadline_s, (request.deadline_ms or 2000) / 1000),
                    owner_ref=context.principal.actor_ref,
                )
            )
            with open_operation_read(
                context.archive_root,
                publication_guard=guard,
                vector_model=vector_model,
                execution_context=read_control,
            ) as snapshot:
                _validate_identity(request, context, snapshot)
                if context.runtime is not None:
                    context.runtime.observe_snapshot(request, snapshot)
                if mutating:
                    assert context.runtime is not None
                    from polylogue.operations import daemon_mutations

                    audit = context.runtime.audit_for_request(request, context)
                    handler = getattr(daemon_mutations, spec.handler)
                    result = handler(request, context, audit, snapshot)
                else:
                    from polylogue.operations.daemon_reads import execute_read_operation

                    result = execute_read_operation(
                        request.operation,
                        request.payload,
                        archive=snapshot.archive,
                        serving_identity=context.serving_identity,
                        dependencies=replace(
                            context.read_dependencies,
                            vector_connection=snapshot.archive.operation_vector_connection,
                            vector_failure=snapshot.vector_failure or context.read_dependencies.vector_failure,
                        )
                        if context.read_dependencies is not None
                        else None,
                    )
                validate_operation_result(request.operation, result)
                return operation_envelope(request, context, snapshot=snapshot, started_at=started, result=result)

        if spec.authority is DaemonAuthority.READ:
            return execute(mutating=False)
        assert context.runtime is not None
        return context.runtime.run_write(spec.name, lambda: execute(mutating=True))
    except MachineRequestRecoveredError as exc:
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            started_at=started,
            result=exc.record,
            outcome="accepted",
            reference=exc.record,
        )
    except (QueryCancelledError, QueryTimeoutError) as exc:
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            started_at=started,
            outcome="cancelled" if isinstance(exc, QueryCancelledError) else "timed-out",
            error={"code": type(exc).__name__, "detail": str(exc), "retryable": True},
        )
    except (ValueError, PermissionError) as exc:
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            started_at=started,
            outcome="rejected",
            error={"code": str(exc), "detail": str(exc), "retryable": False},
        )
