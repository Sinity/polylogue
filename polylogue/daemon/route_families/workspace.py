"""Executable declarations for bounded workspace HTTP projections."""

from __future__ import annotations

from polylogue.daemon.route_types import RouteMethod, RouteSpec
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


def _workspace_route(
    *,
    declaration_id: str,
    path: str,
    public_name: str,
    handler: str,
    request_contract: str,
    response_contract: str,
    discovery_text: str,
    example: ExampleSpec,
) -> RouteSpec:
    method: RouteMethod = "GET"
    binding_key = f"{method} {path}"
    producer = f"polylogue.daemon.http.DaemonAPIHandler.{handler}"
    return RouteSpec(
        kernel=DeclarationSpec(
            declaration_id=declaration_id,
            family_id="daemon.workspace",
            public_name=public_name,
            owner_path="polylogue/daemon/http.py",
            compatibility=CompatibilityKey(
                identity="daemon-route",
                lifecycle="shell-supported",
                authority="daemon-read",
                access_result_shape="bounded-workspace-envelope",
                durability="read-only",
            ),
            producer=producer,
            role_gate="credential_if_configured",
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
        auth_policy="credential_if_configured",
        domain_operation=None,
        passes_params=True,
        auth_scope="read",
        migration_reason="Workspace route still uses the established direct archive projection; no canonical daemon operation owns it.",
        kind="workspace",
        stability="shell_supported",
    )


ROUTES: tuple[RouteSpec, ...] = (
    _workspace_route(
        declaration_id="daemon.workspace.stack",
        path="/api/stack",
        public_name="stack",
        handler="_handle_stack",
        request_contract="StackWorkspaceQuery",
        response_contract="StackWorkspaceEnvelope",
        discovery_text="Read a bounded workspace view over selected sessions.",
        example=ExampleSpec(
            "two-sessions",
            "Read two selected sessions in a bounded window",
            (("ids", "session-a,session-b"), ("limit", 20)),
        ),
    ),
    _workspace_route(
        declaration_id="daemon.workspace.compare",
        path="/api/compare",
        public_name="compare",
        handler="_handle_compare",
        request_contract="CompareWorkspaceQuery",
        response_contract="CompareWorkspaceEnvelope",
        discovery_text="Compare two sessions with a bounded message window.",
        example=ExampleSpec(
            "prompt-alignment",
            "Compare two sessions by prompt alignment",
            (("left", "session-a"), ("right", "session-b"), ("align", "prompt"), ("limit", 20)),
        ),
    ),
)


__all__ = ["ROUTES"]
