"""Session excision on one original admitted preparation and publication phase."""

from __future__ import annotations

from time import monotonic, time

from polylogue.core.stage_admission import admit_stage_write
from polylogue.core.write_lease import require_write_lease
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
from polylogue.operations.daemon_protocol import (
    DaemonAuthorization,
    DaemonOperationEnvelope,
    DaemonOperationRequest,
    daemon_operation_spec,
    validate_operation_result,
)
from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs
from polylogue.operations.mutation_transaction import (
    ConfirmationRequiredError,
    MutationAuthorization,
    MutationPreview,
    OperationExecutor,
    compute_parameter_digest,
)
from polylogue.operations.operation_context import OperationControlRead, open_operation_read
from polylogue.operations.operation_context_types import OperationContext


def validate_session_excision_request(
    request: DaemonOperationRequest, context: OperationContext
) -> DaemonOperationRequest:
    request = validate_execution_request(request, context)
    spec = daemon_operation_spec(request.operation)
    if spec is None or spec.name != "mutation.session.excision":
        raise ValueError("session excision requires its declared operation")
    if spec.authorization is DaemonAuthorization.CONFIRMATION and request.payload.get("confirm") is not True:
        raise ConfirmationRequiredError(f"{request.operation} requires explicit confirmation")
    return request


async def execute_session_excision_operation(
    request: DaemonOperationRequest,
    context: OperationContext,
) -> DaemonOperationEnvelope:
    request = validate_session_excision_request(request, context)
    runtime = context.runtime
    assert runtime is not None
    await runtime.recover_interrupted_operations(resolver_actor_ref=context.principal.actor_ref)
    started_at = monotonic()
    audit = runtime.audit_for_request(request, context)

    def observe_authority(*, admission_held: bool = False) -> OperationControlRead:
        # Detach actual Index-dependent authority and physically close its readers
        # before the original Excision producer retains its own tier witnesses.
        if admission_held:
            # Authorization already owns the same writer exclusion. A second
            # bridge hold would wait behind itself before durable begin.
            require_write_lease("operation.session.excision.authority", archive_root=context.archive_root)
        with open_operation_read(
            context.archive_root,
            publication_guard=None if admission_held else runtime.publication_guard,
            execution_context=context.read_control,
        ) as snapshot:
            _validate_identity(request, context, snapshot)
            return OperationControlRead(snapshot.identity, snapshot.schema_versions, snapshot.degraded_components)

    def execute() -> DaemonOperationEnvelope:
        authority = observe_authority()
        _validate_identity(request, context, authority)
        runtime.observe_snapshot(request, authority)
        payload = request.payload
        args = SessionExcisionArgs(
            archive_root=context.archive_root,
            session_id=str(payload["session_id"]),
            reason=str(payload["reason"]),
            actor=str(payload["actor"]),
            cascade_lineage=bool(payload.get("cascade_lineage", False)),
            input_demand=runtime.prepared_compute_adapter().amend_current_input_demand,
            result_sink=lambda summary, literal: runtime.retain_result_document(request, context, summary, literal),
        )
        binding = runtime_operation_binding(SessionExcisionActuator())
        executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
        raw_plan = binding.actuator.prepare(args)

        def authorize() -> tuple[MutationPreview, MutationAuthorization]:
            current = observe_authority(admission_held=True)
            _validate_identity(request, context, current)
            if current.identity != authority.identity:
                raise ValueError("archive changed before excision authorization")
            runtime.begin_unbound_write(request, snapshot=authority)
            preview = executor.prepare_bound(
                binding,
                args,
                context.principal,
                archive_instance_id=audit.ensure_archive_authority(now_ms=int(time() * 1000)),
                archive_identity_digest=authority.identity.authority_identity_digest,
                parameter_digest=compute_parameter_digest(raw_plan),
                raw_plan=raw_plan,
            )
            authorization = executor.authorize_bound(
                binding,
                preview,
                context.principal,
                confirmation_strength="bound_token",
            )
            return preview, authorization

        preview, authorization = admit_stage_write("operation.session.excision.authorize", authorize)
        receipt = executor.execute_bound(binding, preview, authorization, args)
        if receipt.status in {"blocked", "unknown"}:
            raise ValueError(receipt.detail or "session excision did not apply")
        result = {
            "operation": request.operation,
            "outcome": "completed",
            "sequence": 1,
            "effect": "committed" if receipt.affected_count else "no-effect",
            "affected_count": receipt.affected_count,
            "receipt_ref": receipt.receipt_ref,
            "result": dict(receipt.domain_receipt),
            "result_document": runtime.result_document_identity(request),
        }
        validate_operation_result(request.operation, result)
        settled = observe_authority()
        if settled.identity.authority_identity_digest != authority.identity.authority_identity_digest:
            raise ValueError("archive_identity_stale")
        return operation_envelope(
            request,
            context,
            snapshot=settled,
            admitted_snapshot=authority,
            started_at=started_at,
            result=result,
        )

    return await runtime.prepared_phase(
        "session.excision",
        execute,
        estimated_bytes=0,
        exclusive_bytes=True,
    )
