"""Canonical typed machine execution, shared by direct and transport adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Protocol, TypeVar

from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext, QueryTimeoutError
from polylogue.core.errors import ArchiveTierUnavailableError, SchemaRefusalError
from polylogue.core.raw_failure_evidence import (
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
    RetainedRawDependencyRefusalError,
)
from polylogue.operations.audit import AuditRepository, MachineRequestRecoveredError
from polylogue.operations.daemon_protocol import (
    AcceptedOperationReference,
    AuthoritySnapshot,
    DaemonAuthority,
    DaemonOperationEnvelope,
    DaemonOperationRequest,
    daemon_operation_spec,
    validate_operation_result,
)
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.operations.operation_context import (
    OperationControlRead,
    OperationControlResult,
    PinnedOperationRead,
    observe_control_authority,
    observe_embedding_mutation_authority,
    open_operation_read,
)
from polylogue.operations.operation_context_types import OperationContext
from polylogue.storage.embeddings.generations import EmbeddingGenerationBusyError
from polylogue.version import POLYLOGUE_VERSION

_T = TypeVar("_T")

if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.operations.assertion_export import AssertionExportImages
    from polylogue.operations.audit import CanonicalAuditLiteral
    from polylogue.operations.insight_acceptance import AcceptedInsightPart, SessionInsightPartReceipt
    from polylogue.operations.raw_observation_owner import RetainedMaterializationResult


class OperationRuntime(Protocol):
    """Resident authority injected into the product executor by its owner."""

    assertion_exports: AssertionExportImages

    def publication_guard(self) -> AbstractContextManager[object]: ...

    def run_write(self, name: str, work: Callable[[], _T]) -> _T: ...

    def begin_unbound_write(
        self, request: DaemonOperationRequest, *, snapshot: PinnedOperationRead | OperationControlRead
    ) -> None: ...

    async def materialize_retained_raw_ids(
        self,
        raw_ids: tuple[str, ...],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None],
        on_dependency_refusal: Callable[[RetainedRawDependencyRefusalError], None],
        on_membership_refusal: Callable[[CohortMembershipRefusalError], None],
        before_publication: Callable[[], None],
    ) -> RetainedMaterializationResult: ...

    async def compute_phase(self, work: Callable[[], _T]) -> _T: ...

    async def write_phase(self, name: str, work: Callable[[], _T]) -> _T: ...

    async def recover_interrupted_operations(self, *, resolver_actor_ref: str) -> None: ...

    def prepared_compute_adapter(self) -> BoundedComputeAdapter: ...

    async def prepared_phase(
        self, name: str, work: Callable[[], _T], *, estimated_bytes: int, exclusive_bytes: bool = False
    ) -> _T: ...

    def result_document_identity(self, request: DaemonOperationRequest) -> dict[str, object]: ...

    def retain_result_document(
        self,
        request: DaemonOperationRequest,
        context: OperationContext,
        summary: Mapping[str, object],
        literal: CanonicalAuditLiteral,
    ) -> None: ...

    def require_session_maintenance(self) -> None: ...

    def session_profile_plan_binding(self, *, opened_index_path: Path) -> tuple[str, str]: ...

    async def converge_ingest_sessions(
        self,
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
        self,
        request: DaemonOperationRequest,
        principal: MutationPrincipal,
        archive_identity: str,
        *,
        execution_context: QueryExecutionContext | None = None,
    ) -> OperationControlResult: ...

    def observe_snapshot(
        self, request: DaemonOperationRequest, snapshot: PinnedOperationRead | OperationControlRead
    ) -> None: ...

    def request_deadline_unix_ms(self, request: DaemonOperationRequest) -> int: ...

    def stop_reason(self, request: DaemonOperationRequest) -> str | None: ...

    def emit_progress(self, request: DaemonOperationRequest, event: Mapping[str, object]) -> None: ...

    embedding_convergence: object | None


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
    degraded_components: tuple[str, ...] | None = None,
    progress: dict[str, object] | None = None,
    admitted_snapshot: OperationControlRead | None = None,
) -> DaemonOperationEnvelope:
    spec = daemon_operation_spec(request.operation)
    elapsed = max(0, int((monotonic() - started_at) * 1000)) if started_at is not None else 0
    identity = snapshot.identity if snapshot is not None else None
    versions = snapshot.schema_versions if snapshot is not None else {}
    degraded = (
        snapshot.degraded_components
        if snapshot is not None
        else degraded_components
        if degraded_components is not None
        else ("authority_unavailable",)
    )
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
            **(
                {
                    "admitted_archive_identity": admitted_snapshot.identity.authority_identity_digest,
                    "admitted_generation": admitted_snapshot.identity.active_generation,
                }
                if admitted_snapshot is not None
                else {}
            ),
        },
        progress=progress or {"state": "complete" if outcome == "completed" else outcome},
        outcome=outcome,
        served_by={"identity": context.serving_identity, "daemon_version": POLYLOGUE_VERSION},
        timing={"elapsed_ms": elapsed, "queue_ms": queue_ms},
        degraded_components=degraded,
        schema_versions=versions,
        result=result,
        error=error,
        request_id=request.request_id,
        accepted_reference=AcceptedOperationReference.from_record(reference).to_dict()
        if reference is not None
        else None,
        authority_snapshot=authority.to_dict(),
    )


def _observe_explicit_index_condition(
    request: DaemonOperationRequest,
    snapshot: OperationControlRead,
    *,
    archive_root: Path,
    read_control: QueryExecutionContext,
) -> OperationControlRead:
    """Observe only a caller-supplied Index condition for independent reads."""
    if request.index_schema_version is None:
        return snapshot
    from polylogue.archive.query.execution_control import InterruptibleSQLiteRead
    from polylogue.operations.user_overlay_reads import readable_required_tier
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    location = ArchiveLocation.resolve(archive_root)
    if ArchiveIdentity.resolve_location(location) != snapshot.identity:
        raise ValueError("archive_identity_stale")
    with (
        readable_required_tier(location.active_index_path, ArchiveTier.INDEX) as connection,
        InterruptibleSQLiteRead(read_control).control_connection(connection),
    ):
        connection.execute("BEGIN")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(archive_root)) != snapshot.identity:
        raise ValueError("archive_identity_stale")
    return replace(snapshot, schema_versions={**snapshot.schema_versions, "index": version})


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
    if spec.capability not in context.principal.capabilities:
        raise PermissionError(f"operation requires capability {spec.capability}")
    if context.runtime is None:
        raise PermissionError("daemon_required")
    return request


def execute_operation(request: DaemonOperationRequest, context: OperationContext) -> DaemonOperationEnvelope:
    """Validate and execute the declared operation against explicit authority."""

    started = monotonic()
    snapshot: PinnedOperationRead | OperationControlRead | None = None
    try:
        request = validate_execution_request(request, context)
        spec = daemon_operation_spec(request.operation)
        assert spec is not None
        if request.operation == "user.assertions.export.release":
            # Release owns only private scratch and must work after User disappears.
            from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

            read_control = context.read_control or QueryExecutionContext(
                call_id=str(request.request_id),
                query_ref=request.fingerprint,
                deadline_monotonic=None if request.deadline_ms is None else started + request.deadline_ms / 1000,
                owner_ref=context.principal.actor_ref,
            )
            snapshot = _observe_explicit_index_condition(
                request,
                OperationControlRead(
                    ArchiveIdentity.resolve_location(ArchiveLocation.resolve(context.archive_root)), {}, ()
                ),
                archive_root=context.archive_root,
                read_control=read_control,
            )
            _validate_identity(request, context, snapshot)
            if context.runtime is None:
                raise ValueError("assertion export requires its resident selection owner")
            result: dict[str, object] = {
                "released": context.runtime.assertion_exports.release(
                    str(request.payload["selection_ref"]), context.principal
                )
            }
            read_control.mark_cleanup_complete()
            validate_operation_result(request.operation, result)
            return operation_envelope(request, context, snapshot=snapshot, started_at=started, result=result)
        if request.operation == "mutation.session.excision":
            raise PermissionError("staged_execution_required")
        if request.operation == "insights.hermes_health":
            from polylogue.operations.hermes_health import execute_hermes_health
            from polylogue.operations.operation_context import abort_checkpoint
            from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

            read_control = context.read_control or QueryExecutionContext(
                call_id=str(request.request_id),
                query_ref=request.fingerprint,
                deadline_monotonic=None if request.deadline_ms is None else started + request.deadline_ms / 1000,
                owner_ref=context.principal.actor_ref,
            )
            checkpoint = abort_checkpoint(read_control)
            checkpoint()
            identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(context.archive_root))
            snapshot = _observe_explicit_index_condition(
                request,
                OperationControlRead(identity, {}, ()),
                archive_root=context.archive_root,
                read_control=read_control,
            )
            _validate_identity(request, context, snapshot)
            dependencies = context.read_dependencies
            if dependencies is None or dependencies.hermes_root is None:
                raise ValueError("Hermes health requires the resident configured source root")
            result = execute_hermes_health(
                context.archive_root, hermes_root=dependencies.hermes_root, checkpoint=checkpoint
            )
            read_control.mark_cleanup_complete()
            validate_operation_result(request.operation, result)
            if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(context.archive_root)) != identity:
                raise ValueError("archive changed while reading Hermes health")
            return operation_envelope(request, context, snapshot=snapshot, started_at=started, result=result)
        if request.operation in {"user.settings.get", "user.settings.list"}:
            from polylogue.archive.query.execution_control import InterruptibleSQLiteRead
            from polylogue.operations.operation_context import abort_checkpoint
            from polylogue.operations.user_overlay_reads import read_user_settings, readable_required_tier
            from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
            from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

            read_control = context.read_control or QueryExecutionContext(
                call_id=str(request.request_id),
                query_ref=request.fingerprint,
                deadline_monotonic=None if request.deadline_ms is None else started + request.deadline_ms / 1000,
                owner_ref=context.principal.actor_ref,
            )
            checkpoint = abort_checkpoint(read_control)
            identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(context.archive_root))
            snapshot = _observe_explicit_index_condition(
                request,
                OperationControlRead(identity, {}, ()),
                archive_root=context.archive_root,
                read_control=read_control,
            )
            _validate_identity(request, context, snapshot)
            with (
                readable_required_tier(
                    context.archive_root / "user.db", ArchiveTier.USER, on_settled=read_control.mark_cleanup_complete
                ) as connection,
                InterruptibleSQLiteRead(read_control).control_connection(connection),
            ):
                connection.execute("BEGIN")
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                snapshot = replace(snapshot, schema_versions={**snapshot.schema_versions, "user": version})
                result = read_user_settings(
                    request.operation, request.payload, connection=connection, checkpoint=checkpoint
                )
                validate_operation_result(request.operation, result)
                if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(context.archive_root)) != identity:
                    raise ValueError("archive changed while reading user settings")
                return operation_envelope(request, context, snapshot=snapshot, started_at=started, result=result)
        if request.operation == "session.identity-reset.targets":
            from polylogue.operations.daemon_mutations import identity_reset_targets
            from polylogue.operations.operation_context import abort_checkpoint

            assert context.runtime is not None
            read_control = context.read_control or QueryExecutionContext(
                call_id=str(request.request_id),
                query_ref=request.fingerprint,
                deadline_monotonic=None if request.deadline_ms is None else started + request.deadline_ms / 1000,
                owner_ref=context.principal.actor_ref,
            )
            checkpoint = abort_checkpoint(read_control)
            checkpoint()
            with open_operation_read(
                context.archive_root,
                publication_guard=context.runtime.publication_guard,
                execution_context=read_control,
            ) as snapshot:
                _validate_identity(request, context, snapshot)
                audit = context.runtime.audit_for_request(request, context)
                with audit.settled_machine_read():
                    result = identity_reset_targets(request, context, audit, snapshot)
                checkpoint()
                validate_operation_result(request.operation, result)
                # A reset batch can settle a committed prefix while cancellation
                # leaves its suffix untouched. Its terminal envelope must carry
                # the same durable outcome as the batch, never default success.
                outcome = (
                    str(result["outcome"])
                    if request.operation == "mutation.identity-reset" and isinstance(result, dict)
                    else "completed"
                )
                return operation_envelope(
                    request, context, snapshot=snapshot, started_at=started, result=result, outcome=outcome
                )

        if request.operation.startswith("operation."):
            assert context.runtime is not None
            control_snapshot = observe_control_authority(context.archive_root)
            snapshot = control_snapshot
            _validate_identity(request, context, control_snapshot)
            control_result = context.runtime.control(
                request,
                context.principal,
                control_snapshot.identity.authority_identity_digest,
                execution_context=context.read_control,
            )
            result = control_result.state
            snapshot = control_result.snapshot
            validate_operation_result(request.operation, result)
            return operation_envelope(request, context, snapshot=snapshot, started_at=started, result=result)

        # Lifecycle adoption/checkpoint needs the embedding tier free of our
        # own pinned reader. Observe tier versions through short-lived reads,
        # then enter the handler under the writer lease.
        if request.operation == "maintenance.embeddings.failure.resolve":
            runtime = context.runtime
            assert runtime is not None

            def resolve_failure() -> DaemonOperationEnvelope:
                nonlocal snapshot
                from polylogue.operations.daemon_protocol import resolve_operation_handler

                snapshot = observe_embedding_mutation_authority(context.archive_root)
                _validate_identity(request, context, snapshot)
                runtime.observe_snapshot(request, snapshot)
                audit = runtime.audit_for_request(request, context)
                result = resolve_operation_handler(spec)(request, context, audit, snapshot)
                validate_operation_result(request.operation, result)
                # The handler may adopt a bootstrap-created embeddings.db into a
                # generation, changing the archive identity after admission.
                # Keep the request bound to the pre-write snapshot but report
                # the settled identity in the successful response.
                from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

                settled_identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(context.archive_root))
                return operation_envelope(
                    request,
                    context,
                    snapshot=replace(snapshot, identity=settled_identity),
                    admitted_snapshot=snapshot,
                    started_at=started,
                    result=result,
                )

            return runtime.run_write(spec.name, resolve_failure)

        def execute(*, mutating: bool) -> DaemonOperationEnvelope:
            nonlocal snapshot
            guard = (nullcontext if mutating else context.runtime.publication_guard) if context.runtime else None
            from polylogue.operations.daemon_reads import requires_vector_snapshot

            vector_binding = context.read_dependencies.vector_binding if context.read_dependencies is not None else None
            vector_recipe = (
                vector_binding.recipe
                if vector_binding is not None
                and requires_vector_snapshot(
                    request.operation, request.payload, acquisition_enabled=bool(vector_binding.voyage_key)
                )
                else None
            )
            read_control = (
                None
                if mutating
                else context.read_control
                or QueryExecutionContext(
                    call_id=str(request.request_id),
                    query_ref=request.fingerprint,
                    deadline_monotonic=None if request.deadline_ms is None else started + request.deadline_ms / 1000,
                    owner_ref=context.principal.actor_ref,
                )
            )
            with open_operation_read(
                context.archive_root,
                publication_guard=guard,
                vector_recipe=vector_recipe,
                execution_context=read_control,
            ) as snapshot:
                _validate_identity(request, context, snapshot)
                if context.runtime is not None:
                    context.runtime.observe_snapshot(request, snapshot)
                if mutating:
                    assert context.runtime is not None
                    from polylogue.operations.daemon_protocol import resolve_operation_handler

                    audit = context.runtime.audit_for_request(request, context)
                    handler = resolve_operation_handler(spec)
                    result = handler(request, context, audit, snapshot)
                else:
                    from polylogue.operations.daemon_reads import DaemonReadDependencies, execute_read_operation
                    from polylogue.operations.operation_context import abort_checkpoint

                    dependencies = (
                        replace(
                            context.read_dependencies,
                            vector_connection=snapshot.archive.operation_vector_connection,
                            vector_failure=snapshot.vector_failure or context.read_dependencies.vector_failure,
                        )
                        if context.read_dependencies is not None
                        else DaemonReadDependencies()
                    )
                    assert read_control is not None
                    result = execute_read_operation(
                        request.operation,
                        request.payload,
                        archive=snapshot.archive,
                        serving_identity=context.serving_identity,
                        dependencies=replace(dependencies, raise_if_aborted=abort_checkpoint(read_control)),
                        read_view=snapshot.read_view,
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
    except SchemaRefusalError as exc:
        from polylogue.daemon.derived_degradation import schema_refusal_details

        details = schema_refusal_details(exc)
        tier = str(details["tier"])
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            started_at=started,
            outcome="degraded",
            degraded_components=(f"derived_schema:{tier}",),
            progress={**details, "state": "degraded"},
            error={
                "code": str(details["code"]),
                "detail": str(exc),
                "retryable": True,
                "data": details,
            },
        )
    except ArchiveTierUnavailableError as exc:
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            started_at=started,
            outcome="rejected",
            error={
                "code": exc.code,
                "detail": exc.public_message,
                "retryable": False,
                "data": {"tier": exc.tier, "guidance": exc.guidance},
            },
        )
    except EmbeddingGenerationBusyError as exc:
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            started_at=started,
            outcome="rejected",
            error={"code": "embedding_generation_busy", "detail": str(exc), "retryable": True},
        )
    except (ValueError, PermissionError) as exc:
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            started_at=started,
            outcome="rejected",
            # A refusal that declares its own code keeps it, so a typed
            # refusal reaches a client by the same token over the socket as
            # it does in-process; an undeclared one keeps its message.
            error={"code": str(getattr(exc, "code", "") or exc), "detail": str(exc), "retryable": False},
        )
