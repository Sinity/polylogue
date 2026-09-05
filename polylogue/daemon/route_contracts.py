"""Stable daemon HTTP route contract metadata.

This module is intentionally descriptive: dispatch still lives in
``polylogue.daemon.http``, but route classes, auth posture, and stability are
owned here so docs, tests, OpenAPI generation, and future web-workbench code do
not infer security semantics from handler names. Tests compare this metadata
against route patterns exposed by the live dispatcher tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
)

RouteKind = Literal[
    "browser_shell",
    "operational",
    "read_query",
    "read_detail",
    "user_overlay",
    "workspace",
    "maintenance",
    "capture",
    "observability",
]
RouteStability = Literal["stable", "shell_supported", "operational", "private"]
AuthPolicy = Literal[
    "unauthenticated_loopback",
    "credential_if_configured",
    "credential_and_same_origin",
    "bearer_if_configured_and_same_origin",
    "first_party_same_origin",
    "observability_flag_then_loopback_or_bearer",
]


@dataclass(frozen=True)
class RouteContract:
    """Machine-readable contract for one daemon HTTP route pattern."""

    method: Literal["GET", "POST", "DELETE"]
    pattern: str
    kind: RouteKind
    stability: RouteStability
    auth_policy: AuthPolicy
    response_contract: str
    notes: str = ""
    domain_operation: str | None = None


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """HTTP projection of one shared declaration-kernel record.

    The kernel owns identity, handler ownership, and output/schema edges. The
    HTTP projection adds the transport vocabulary that the route adapter and
    OpenAPI renderer need. Keeping this projection beside the kernel record
    makes a route declaration executable without teaching the shared kernel
    about HTTP.
    """

    kernel: DeclarationSpec
    method: Literal["GET", "POST", "DELETE"]
    path: str
    request_contract: str
    response_contract: str
    auth_policy: AuthPolicy
    domain_operation: str


# Compatibility name for callers that adopted the first kernel projection.
DaemonRouteDeclaration = RouteSpec


_FIND_DECLARATION = RouteSpec(
    kernel=DeclarationSpec(
        declaration_id="daemon.find.sessions",
        family_id="daemon.read-query",
        public_name="find",
        owner_path="polylogue/daemon/http.py",
        compatibility=CompatibilityKey(
            identity="daemon-route",
            lifecycle="stable",
            authority="daemon-read",
            access_result_shape="search-envelope-or-session-list",
            durability="read-only",
        ),
        producer="polylogue.daemon.http.DaemonAPIHandler._handle_list_sessions",
        role_gate="credential_if_configured",
        schema_ref="polylogue.surfaces.payloads.SearchEnvelope|SessionListResponse",
        discovery_text="Find sessions with the shared archive query semantics.",
        repair_command="devtools render openapi",
        handlers=(
            HandlerBinding(
                surface="daemon-http",
                owner_path="polylogue/daemon/http.py",
                symbol="_handle_list_sessions",
                binding_key="GET /api/sessions",
            ),
        ),
        outputs=(
            OutputSpec(
                name="response",
                kind="json",
                schema_ref="SearchEnvelope|SessionListResponse",
                target_path="/api/sessions",
            ),
        ),
        examples=(),
        completeness_edges=(),
    ),
    method="GET",
    path="/api/sessions",
    request_contract="SessionSearchQuery",
    response_contract="SearchEnvelope | SessionListResponse",
    auth_policy="credential_if_configured",
    domain_operation="sessions.find",
)

_STATUS_DECLARATION = RouteSpec(
    kernel=DeclarationSpec(
        declaration_id="daemon.status",
        family_id="daemon.read-status",
        public_name="status",
        owner_path="polylogue/daemon/http.py",
        compatibility=CompatibilityKey("daemon-route", "stable", "daemon-read", "status-envelope", "read-only"),
        producer="polylogue.daemon.http.DaemonAPIHandler._handle_status",
        role_gate="credential_if_configured",
        schema_ref="DaemonStatusPayload",
        discovery_text="Report daemon status and archive readiness.",
        repair_command="devtools render openapi",
        handlers=(HandlerBinding("daemon-http", "polylogue/daemon/http.py", "_handle_status", "GET /api/status"),),
        outputs=(OutputSpec("response", "json", "DaemonStatusPayload", "/api/status"),),
        examples=(ExampleSpec("default", "Read daemon status"),),
        completeness_edges=(
            CompletenessEdge(
                "polylogue.daemon.http.DaemonAPIHandler._handle_status",
                "daemon-http",
                "route",
                "polylogue/daemon/http.py",
            ),
        ),
    ),
    method="GET",
    path="/api/status",
    request_contract="StatusQuery",
    response_contract="DaemonStatusPayload",
    auth_policy="credential_if_configured",
    domain_operation="daemon.status",
)

_QUERY_UNITS_DECLARATION = RouteSpec(
    kernel=DeclarationSpec(
        declaration_id="daemon.query.units",
        family_id="daemon.read-query-units",
        public_name="query-units",
        owner_path="polylogue/daemon/http.py",
        compatibility=CompatibilityKey("daemon-route", "stable", "daemon-read", "query-unit-envelope", "read-only"),
        producer="polylogue.daemon.http.DaemonAPIHandler._handle_query_units",
        role_gate="credential_if_configured",
        schema_ref="QueryUnitResultEnvelope",
        discovery_text="Execute a bounded terminal query-unit page.",
        repair_command="devtools render openapi",
        handlers=(
            HandlerBinding("daemon-http", "polylogue/daemon/http.py", "_handle_query_units", "GET /api/query-units"),
        ),
        outputs=(OutputSpec("response", "json", "QueryUnitResultEnvelope", "/api/query-units"),),
        examples=(ExampleSpec("default", "Read a bounded query-unit page"),),
        completeness_edges=(
            CompletenessEdge(
                "polylogue.daemon.http.DaemonAPIHandler._handle_query_units",
                "daemon-http",
                "route",
                "polylogue/daemon/http.py",
            ),
        ),
    ),
    method="GET",
    path="/api/query-units",
    request_contract="QueryUnitQuery",
    response_contract="QueryUnitResultEnvelope",
    auth_policy="credential_if_configured",
    domain_operation="query.units",
)

_READ_DECLARATION = RouteSpec(
    kernel=DeclarationSpec(
        declaration_id="daemon.read.session",
        family_id="daemon.read-detail",
        public_name="read-session",
        owner_path="polylogue/daemon/http.py",
        compatibility=CompatibilityKey("daemon-route", "stable", "daemon-read", "session-read-envelope", "read-only"),
        producer="polylogue.daemon.http.DaemonAPIHandler._handle_get_session_read",
        role_gate="credential_if_configured",
        schema_ref="SessionReadViewEnvelope",
        discovery_text="Read one bounded session view.",
        repair_command="devtools render openapi",
        handlers=(
            HandlerBinding(
                "daemon-http", "polylogue/daemon/http.py", "_handle_get_session_read", "GET /api/sessions/:id/read"
            ),
        ),
        outputs=(OutputSpec("response", "json", "SessionReadViewEnvelope", "/api/sessions/:id/read"),),
        examples=(ExampleSpec("messages", "Read a bounded session message view"),),
        completeness_edges=(
            CompletenessEdge(
                "polylogue.daemon.http.DaemonAPIHandler._handle_get_session_read",
                "daemon-http",
                "route",
                "polylogue/daemon/http.py",
            ),
        ),
    ),
    method="GET",
    path="/api/sessions/:id/read",
    request_contract="SessionReadQuery",
    response_contract="SessionReadViewEnvelope",
    auth_policy="credential_if_configured",
    domain_operation="sessions.read",
)

DAEMON_ROUTE_REGISTRY = DeclarationRegistry()
DAEMON_ROUTE_DECLARATIONS: tuple[RouteSpec, ...] = (
    _FIND_DECLARATION,
    _STATUS_DECLARATION,
    _QUERY_UNITS_DECLARATION,
    _READ_DECLARATION,
)
for _declaration in DAEMON_ROUTE_DECLARATIONS:
    DAEMON_ROUTE_REGISTRY.register(_declaration.kernel)


def daemon_route_declaration(method: str, path: str) -> RouteSpec:
    """Return the executable declaration for an exact daemon route."""

    for declaration in DAEMON_ROUTE_DECLARATIONS:
        if declaration.method == method.upper() and declaration.path == path:
            return declaration
    raise KeyError(f"no daemon route declaration for {method.upper()} {path}")


def route_contract_from_declaration(declaration: DaemonRouteDeclaration) -> RouteContract:
    """Lower a kernel-backed route declaration to legacy public metadata."""

    kind: RouteKind = "read_query"
    if declaration.domain_operation == "sessions.read":
        kind = "read_detail"
    elif declaration.domain_operation == "daemon.status":
        kind = "operational"
    return RouteContract(
        declaration.method,
        declaration.path,
        kind,
        "stable",
        declaration.auth_policy,
        declaration.response_contract,
        f"declaration={declaration.kernel.declaration_id}; request={declaration.request_contract}",
        declaration.domain_operation,
    )


ROUTE_CONTRACTS: tuple[RouteContract, ...] = (
    RouteContract(
        "POST", "/api/user/marks", "user_overlay", "stable", "credential_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "POST", "/api/user/annotations", "user_overlay", "stable", "credential_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "POST", "/api/user/saved-views", "user_overlay", "stable", "credential_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "POST", "/api/user/recall-packs", "user_overlay", "stable", "credential_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "POST", "/api/user/workspaces", "user_overlay", "stable", "credential_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "DELETE", "/api/user/marks", "user_overlay", "stable", "credential_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "DELETE",
        "/api/user/annotations/:id",
        "user_overlay",
        "stable",
        "credential_and_same_origin",
        "mutation envelope",
    ),
    RouteContract(
        "DELETE",
        "/api/user/saved-views/:id",
        "user_overlay",
        "stable",
        "credential_and_same_origin",
        "mutation envelope",
    ),
    RouteContract(
        "DELETE",
        "/api/user/recall-packs/:id",
        "user_overlay",
        "stable",
        "credential_and_same_origin",
        "mutation envelope",
    ),
    RouteContract(
        "DELETE",
        "/api/user/workspaces/:id",
        "user_overlay",
        "stable",
        "credential_and_same_origin",
        "mutation envelope",
    ),
)


def stable_route_contracts() -> tuple[RouteContract, ...]:
    """Return stable public daemon route contracts."""

    return tuple(route for route in ROUTE_CONTRACTS if route.stability == "stable")


def route_contract_for(method: str, path: str) -> RouteContract | None:
    """Return the contract matching ``method path``, if any."""

    normalized_method = method.upper()
    normalized_path = "/" + path.strip("/")
    if normalized_path == "/":
        normalized_path = "/"
    for route in ROUTE_CONTRACTS:
        if route.method != normalized_method:
            continue
        if _pattern_matches(route.pattern, normalized_path):
            return route
    return None


def route_contract_for_pattern(method: str, pattern: str) -> RouteContract:
    """Return the exact contract for a declared ``method pattern`` pair."""

    normalized_method = method.upper()
    for route in ROUTE_CONTRACTS:
        if route.method == normalized_method and route.pattern == pattern:
            return route
    raise KeyError(f"no daemon route contract for {normalized_method} {pattern}")


def _pattern_matches(pattern: str, path: str) -> bool:
    if pattern == path:
        return True
    pattern_parts = _split_path(pattern)
    path_parts = _split_path(path)
    if len(pattern_parts) != len(path_parts):
        return False
    return all(
        pattern_part.startswith(":") or pattern_part == path_part
        for pattern_part, path_part in zip(pattern_parts, path_parts, strict=True)
    )


def _split_path(path: str) -> tuple[str, ...]:
    if path == "/":
        return ()
    return tuple(part for part in path.strip("/").split("/") if part)


__all__ = [
    "DAEMON_ROUTE_DECLARATIONS",
    "DAEMON_ROUTE_REGISTRY",
    "DaemonRouteDeclaration",
    "ROUTE_CONTRACTS",
    "RouteContract",
    "RouteSpec",
    "daemon_route_declaration",
    "route_contract_for",
    "route_contract_for_pattern",
    "route_contract_from_declaration",
    "stable_route_contracts",
]
