"""Executable declarations for daemon maintenance and intake HTTP adapters."""

from __future__ import annotations

from polylogue.daemon.route_types import RouteMethod, RouteSpec, RouteStability
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
)


def _consumer_edges(producer: str) -> tuple[CompletenessEdge, ...]:
    return (
        CompletenessEdge(producer, "daemon-http", "route", "polylogue/daemon/http.py"),
        CompletenessEdge(producer, "openapi-schema", "generated-document", "docs/openapi/search.yaml"),
    )


def _maintenance_route(
    *,
    declaration_id: str,
    path: str,
    public_name: str,
    handler: str,
    request_contract: str,
    response_contract: str,
    discovery_text: str,
    example: ExampleSpec,
    domain_operation: str | None = None,
    migration_reason: str = "",
    stability: RouteStability = "stable",
    write_gate: bool = False,
) -> RouteSpec:
    method: RouteMethod = "POST"
    binding_key = f"{method} {path}"
    producer = f"polylogue.daemon.http.DaemonAPIHandler.{handler}"
    return RouteSpec(
        kernel=DeclarationSpec(
            declaration_id=declaration_id,
            family_id="daemon.maintenance",
            public_name=public_name,
            owner_path="polylogue/daemon/http.py",
            compatibility=CompatibilityKey(
                identity="daemon-route",
                lifecycle="stable",
                authority="daemon-maintenance",
                access_result_shape="maintenance-result",
                durability="durable-or-operational",
            ),
            producer=producer,
            role_gate="bearer_if_configured_and_same_origin",
            schema_ref=response_contract,
            discovery_text=discovery_text,
            repair_command="devtools render openapi",
            handlers=(
                HandlerBinding(
                    surface="daemon-http",
                    owner_path="polylogue/daemon/http.py",
                    symbol=handler,
                    binding_key=binding_key,
                ),
            ),
            outputs=(OutputSpec("response", "json", response_contract, path),),
            examples=(example,),
            completeness_edges=_consumer_edges(producer),
        ),
        method=method,
        path=path,
        request_contract=request_contract,
        response_contract=response_contract,
        auth_policy="bearer_if_configured_and_same_origin",
        domain_operation=domain_operation,
        passes_params=False,
        auth_scope="read",
        write_gate=write_gate,
        migration_reason=migration_reason,
        kind="maintenance",
        stability=stability,
    )


ROUTES: tuple[RouteSpec, ...] = (
    _maintenance_route(
        declaration_id="daemon.maintenance.reset",
        path="/api/reset",
        public_name="reset",
        handler="_handle_reset",
        request_contract="ResetSessionRequest",
        response_contract="MutationResultPayload",
        discovery_text="Reset one selected session through the daemon write owner.",
        example=ExampleSpec(
            "one-session", "Reset one selected session", (("scope", "session"), ("session_id", "session-a"))
        ),
        migration_reason="This compatibility route still calls the safe session-delete API directly; maintenance.reset owns archive-file reset.",
        write_gate=True,
    ),
    _maintenance_route(
        declaration_id="daemon.maintenance.ingest",
        path="/api/ingest",
        public_name="ingest",
        handler="_handle_ingest",
        request_contract="IngestRequest",
        response_contract="DaemonOperationResult",
        discovery_text="Stage and submit one import source to the daemon ingest operation.",
        example=ExampleSpec("staged-source", "Ingest a staged source", (("path", "capture.jsonl"),)),
        domain_operation="ingest",
    ),
    _maintenance_route(
        declaration_id="daemon.maintenance.demo.augment",
        path="/api/demo/augment",
        public_name="demo-augment",
        handler="_handle_demo_augment",
        request_contract="DemoAugmentRequest",
        response_contract="DemoAugmentResult",
        discovery_text="Apply deterministic demo augmentation through the daemon write bridge.",
        example=ExampleSpec("default", "Apply demo augmentation", ()),
        migration_reason="This compatibility route still applies demo writes through the bridge; the declared maintenance.demo.augment operation is not its executor.",
        stability="operational",
    ),
)


__all__ = ["ROUTES"]
