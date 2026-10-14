"""Read view for one session's orchestration evidence."""

from __future__ import annotations

import json

from polylogue.cli.operation_kernel import OperationKernelError, OperationRequest
from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read
from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content, materialized_read_outcome
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def run_read_orchestration(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the evidence MCP ``get(projection="orchestration")`` returns."""

    session_id = invocation.session_id
    assert session_id is not None
    try:
        result, _ = dispatch_read(
            env.config,
            OperationRequest("read.orchestration", {"session_id": session_id}),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
            selection_epoch=request.selection_epoch,
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    payload = result.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("read.orchestration returned no payload")
    from polylogue.analysis.orchestration_evidence import SessionOrchestrationEvidence
    from polylogue.surfaces.outcome import OutcomeEnvelope

    evidence = SessionOrchestrationEvidence.model_validate(payload)
    outcome = materialized_read_outcome(
        result,
        OutcomeEnvelope(
            state=evidence.outcome,
            reason="orchestration_evidence_gaps" if evidence.gaps else None,
            detail={"gaps": evidence.gaps},
        ),
    )
    content = json.dumps(payload, indent=2) + "\n"
    deliver_content(
        env, content, destination=invocation.destination, out_path=invocation.out_path, output_format="json"
    )

    from polylogue.cli.render.outcome import finish_supplied_outcome

    finish_supplied_outcome(outcome)


__all__ = ["run_read_orchestration"]
