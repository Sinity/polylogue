"""Daemon HTTP route declarations and compatibility inventory.

API declarations own dispatch and OpenAPI route identity. A declared handler
may still be a compatibility adapter; ``domain_operation`` and
``migration_reason`` distinguish product operation adoption from routing.
"""

from __future__ import annotations

from polylogue.daemon.route_types import (
    AuthPolicy as AuthPolicy,
)
from polylogue.daemon.route_types import (
    RouteContract,
    RouteKind,
    RouteSpec,
)
from polylogue.daemon.route_types import (
    RouteStability as RouteStability,
)
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
)

# Compatibility name for callers that adopted the first kernel projection.
DaemonRouteDeclaration = RouteSpec


def _consumer_edges(producer: str) -> tuple[CompletenessEdge, ...]:
    """Consumers that go blind when this route declaration is absent.

    Both edges are hard failures, not documentation. ``daemon-http``
    generates the route's dispatch adapter from the declaration
    (``_declared_get_routes``), and ``validate_declared_route_reachability``
    refuses daemon startup when generation and installation disagree.
    ``openapi-schema`` raises ``RuntimeError`` from
    ``devtools/render_openapi.py`` when the rendered document carries no
    operation for a declared method/path.
    """

    return (
        CompletenessEdge(producer, "daemon-http", "route", "polylogue/daemon/http.py"),
        CompletenessEdge(producer, "openapi-schema", "generated-document", "docs/openapi/search.yaml"),
    )


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
        examples=(
            # Derived from the live request contract: ``_handle_list_sessions``
            # reads ``limit``/``offset``/``cursor`` directly and routes every
            # other key through ``_build_query_spec_params``; ``query`` is
            # compiled by the shared expression parser, so an origin clause is
            # a structured filter rather than an FTS term.
            ExampleSpec("default", "Read the first bounded session page", (("limit", 20),)),
            ExampleSpec(
                "origin-filter",
                "Find sessions from one origin with the shared query grammar",
                (("query", "origin:claude-code-session"), ("limit", 5)),
            ),
        ),
        completeness_edges=_consumer_edges("polylogue.daemon.http.DaemonAPIHandler._handle_list_sessions"),
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
        # ``_handle_status`` accepts ``params: ... | None`` and reads none of
        # them: the status snapshot takes no request input, so the only
        # faithful example is the parameterless one.
        examples=(ExampleSpec("default", "Read daemon status", ()),),
        completeness_edges=_consumer_edges("polylogue.daemon.http.DaemonAPIHandler._handle_status"),
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
        examples=(
            # ``_handle_query_units`` reads ``expression``/``limit``/``offset``
            # on an initial request. The follow-up shape takes only the opaque
            # ``continuation`` the previous page emitted and rejects any other
            # key, so no static literal can stand for it; it is deliberately
            # not declared as an example.
            ExampleSpec(
                "default",
                "Read a bounded query-unit page",
                (("expression", "messages where role:user"), ("limit", 2)),
            ),
        ),
        completeness_edges=_consumer_edges("polylogue.daemon.http.DaemonAPIHandler._handle_query_units"),
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
        examples=(
            # ``view`` must already exist in ``READ_VIEW_HTTP_CAPABILITIES`` and
            # ``format`` must be one that capability declares; the messages view
            # additionally reads ``limit``/``offset``.
            ExampleSpec(
                "messages",
                "Read a bounded session message view",
                (("view", "messages"), ("format", "json"), ("limit", 50), ("offset", 0)),
            ),
        ),
        completeness_edges=_consumer_edges("polylogue.daemon.http.DaemonAPIHandler._handle_get_session_read"),
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
        declaration.kind or kind,
        declaration.stability or "stable",
        declaration.auth_policy,
        declaration.response_contract,
        f"declaration={declaration.kernel.declaration_id}; request={declaration.request_contract}"
        + (f"; operation-migration={declaration.migration_reason}" if declaration.migration_reason else ""),
        declaration.domain_operation,
        True,
    )


def declared_route_keys() -> frozenset[tuple[str, str]]:
    """Return the route identities that are required to remain executable.

    The compatibility inventory is a separate reachability witness for routes
    with a named product operation. A route without one still has a bound
    handler and a migration reason in its declaration.
    """

    return frozenset((route.method, route.pattern) for route in ROUTE_CONTRACTS if route.domain_operation is not None)


def metadata_only_api_routes() -> tuple[RouteContract, ...]:
    """Return API contracts that remain metadata-only, with named reasons.

    This is deliberately a read-only inventory.  It is used by the focused
    route-contract gate to ensure a newly added ``/api`` route cannot bypass
    either a declaration or an explicit migration classification.
    """

    return tuple(
        route
        for route in ROUTE_CONTRACTS
        if route.pattern.startswith("/api/") and route.metadata_only_reason is not None
    )


ROUTE_CONTRACTS: tuple[RouteContract, ...] = (
    RouteContract(
        "GET",
        "/",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "semantic archive overview HTML",
        "Canonical typed WebUI overview; browser data access remains behind the authenticated /api boundary.",
    ),
    RouteContract(
        "GET",
        "/observability",
        "browser_shell",
        "shell_supported",
        "credential_if_configured",
        "semantic observability HTML",
        "SSR-first registry and status projection; credentials protect embedded insight evidence when configured.",
    ),
    RouteContract(
        "GET",
        "/cost",
        "browser_shell",
        "shell_supported",
        "credential_if_configured",
        "semantic cost/usage HTML",
        "SSR-first registry-driven cost rollup, usage timeline, and session drill-down; credentials protect embedded spend evidence when configured.",
    ),
    RouteContract(
        "GET",
        "/sessions",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "semantic session list HTML",
        "SSR-first origin/date/repo faceted session list; Preact enhances only bounded pagination.",
    ),
    RouteContract(
        "GET",
        "/sessions/:session_id",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "semantic session read HTML",
        "SSR-first session shell: header, lineage banner, and a simple message-flow placeholder Preact enhances with paging.",
    ),
    RouteContract(
        "GET",
        "/search",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "semantic search results HTML",
        "SSR-first ranked search over the shared SearchEnvelope; Preact enhances only cursor-based pagination.",
    ),
    RouteContract(
        "GET",
        "/assets/:asset",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "manifest-governed immutable Vite asset",
        "Only content-hashed files named by the packaged Vite manifest are served.",
    ),
    RouteContract(
        "GET",
        "/s/:session_id",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "semantic session read HTML",
        "Typed WebUI session deep-link; equivalent to /sessions/:session_id.",
    ),
    RouteContract(
        "GET",
        "/w/:mode",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "text/html web shell",
        "Workspace shell bootstrap for registered workspace modes.",
    ),
    RouteContract(
        "GET",
        "/p",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "text/html paste browser",
        "Standalone reader page; archive API calls remain authenticated.",
    ),
    RouteContract(
        "GET",
        "/a",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "text/html attachment library",
        "Standalone reader page; archive API calls remain authenticated.",
    ),
    RouteContract(
        "GET",
        "/healthz/live",
        "operational",
        "operational",
        "unauthenticated_loopback",
        "health liveness JSON",
        "Unauthenticated for systemd/docker/kubernetes probes.",
    ),
    RouteContract(
        "GET",
        "/healthz/ready",
        "operational",
        "operational",
        "unauthenticated_loopback",
        "health readiness JSON",
        "Unauthenticated for systemd/docker/kubernetes probes.",
    ),
    RouteContract(
        "GET",
        "/metrics",
        "operational",
        "operational",
        "unauthenticated_loopback",
        "Prometheus text exposition",
        "Unauthenticated for Prometheus scrapers; no raw archive content.",
    ),
    *(route_contract_from_declaration(route) for route in DAEMON_ROUTE_DECLARATIONS),
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


def _family_declarations() -> tuple[RouteSpec, ...]:
    # Shared route types are separate, so family modules import directly.
    from polylogue.daemon.route_families.maintenance import ROUTES as MAINTENANCE_ROUTES
    from polylogue.daemon.route_families.operational import ROUTES as OPERATIONAL_ROUTES
    from polylogue.daemon.route_families.read_detail import ROUTES as READ_DETAIL_ROUTES
    from polylogue.daemon.route_families.read_query import ROUTES as READ_QUERY_ROUTES
    from polylogue.daemon.route_families.user_overlay import ROUTES as USER_OVERLAY_ROUTES
    from polylogue.daemon.route_families.workspace import ROUTES as WORKSPACE_ROUTES

    return (
        READ_DETAIL_ROUTES
        + READ_QUERY_ROUTES
        + WORKSPACE_ROUTES
        + MAINTENANCE_ROUTES
        + OPERATIONAL_ROUTES
        + USER_OVERLAY_ROUTES
    )


_FAMILY_DECLARATIONS = _family_declarations()
_family_keys = {(route.method, route.path) for route in _FAMILY_DECLARATIONS}
if len(_family_keys) != len(_FAMILY_DECLARATIONS):
    raise RuntimeError("duplicate daemon family declaration method/path")
DAEMON_ROUTE_DECLARATIONS += _FAMILY_DECLARATIONS
for _declaration in _FAMILY_DECLARATIONS:
    DAEMON_ROUTE_REGISTRY.register(_declaration.kernel)
ROUTE_CONTRACTS += tuple(route_contract_from_declaration(declaration) for declaration in _FAMILY_DECLARATIONS)


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
    "declared_route_keys",
    "metadata_only_api_routes",
    "stable_route_contracts",
]
