"""Executable HTTP bindings for operational, observability, and web-auth routes."""

from __future__ import annotations

from typing import Literal

from polylogue.daemon.route_types import AuthPolicy, RouteKind, RouteMethod, RouteSpec, RouteStability
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
)

_ROUTE_METADATA: dict[tuple[RouteMethod, str], tuple[RouteKind, RouteStability, AuthPolicy, str]] = {
    ("POST", "/api/web-auth/session"): (
        "browser_shell",
        "shell_supported",
        "first_party_same_origin",
        "WebCredentialBootstrapPayload",
    ),
    ("DELETE", "/api/web-auth/session"): (
        "browser_shell",
        "shell_supported",
        "first_party_same_origin",
        "WebCredentialRevocationPayload",
    ),
    ("GET", "/api/health/check"): ("operational", "stable", "credential_if_configured", "JSON"),
    ("GET", "/api/health"): ("operational", "stable", "credential_if_configured", "JSON"),
    ("GET", "/api/events"): ("operational", "stable", "credential_if_configured", "SSE or JSON event poll"),
    ("GET", "/api/agents/coordination"): (
        "operational",
        "stable",
        "credential_if_configured",
        "AgentCoordinationPayload",
    ),
    ("GET", "/api/provider-usage"): ("operational", "stable", "credential_if_configured", "ProviderUsageReport"),
    ("GET", "/api/archive-debt"): ("operational", "stable", "credential_if_configured", "ArchiveDebtListPayload"),
    ("GET", "/api/import/explain"): (
        "operational",
        "shell_supported",
        "credential_if_configured",
        "ImportExplainPayload",
    ),
    ("POST", "/api/operation"): ("operational", "private", "credential_if_configured", "DaemonOperationEnvelope"),
    ("POST", "/api/cli/query"): (
        "read_query",
        "private",
        "credential_if_configured",
        "SearchEnvelope / SessionListResponse with route_state",
    ),
    ("POST", "/api/telemetry/mcp-calls"): (
        "operational",
        "private",
        "bearer_if_configured_and_same_origin",
        "MCP call-log receipt",
    ),
    ("GET", "/api/webui/observability"): (
        "observability",
        "shell_supported",
        "credential_if_configured",
        "WebUI observability projection",
    ),
    ("GET", "/api/webui/freshness"): (
        "observability",
        "shell_supported",
        "credential_if_configured",
        "NamedSourceFreshness projection",
    ),
    ("GET", "/api/webui/insights/:name"): (
        "observability",
        "shell_supported",
        "credential_if_configured",
        "Single WebUI insight descriptor projection",
    ),
}


def _route(
    method: RouteMethod,
    path: str,
    handler: str,
    *,
    passes_params: bool = False,
    auth_scope: Literal["read", "events", "user_state"] = "read",
    write_gate: bool = False,
    migration_reason: str,
) -> RouteSpec:
    kind, stability, auth_policy, response_contract = _ROUTE_METADATA[(method, path)]
    declaration_id = "daemon.http." + method.lower() + "." + path.strip("/").replace("/", ".").replace(":", "")
    producer = f"polylogue.daemon.http.DaemonAPIHandler.{handler}"
    return RouteSpec(
        kernel=DeclarationSpec(
            declaration_id=declaration_id,
            family_id=declaration_id,
            public_name=f"{method.lower()}-{path.strip('/').replace('/', '-').replace(':', '')}",
            owner_path="polylogue/daemon/http.py",
            compatibility=CompatibilityKey(
                identity="daemon-route",
                lifecycle=stability.replace("_", "-"),
                authority="daemon-http",
                access_result_shape=response_contract,
                durability="route-dependent",
            ),
            producer=producer,
            role_gate=auth_policy,
            schema_ref=response_contract,
            discovery_text=f"Execute {method} {path} through the daemon HTTP adapter.",
            repair_command="devtools render openapi",
            handlers=(HandlerBinding("daemon-http", "polylogue/daemon/http.py", handler, f"{method} {path}"),),
            outputs=(OutputSpec("response", "stream" if path == "/api/events" else "json", response_contract, path),),
            examples=(ExampleSpec("default", f"Call {method} {path}", ()),),
            completeness_edges=(
                CompletenessEdge(producer, "daemon-http", "route", "polylogue/daemon/http.py"),
                CompletenessEdge(producer, "openapi-schema", "generated-document", "docs/openapi/search.yaml"),
            ),
        ),
        method=method,
        path=path,
        request_contract="HTTP route request",
        response_contract=response_contract,
        auth_policy=auth_policy,
        domain_operation=None,
        passes_params=passes_params,
        auth_scope=auth_scope,
        write_gate=write_gate,
        migration_reason=migration_reason,
        kind=kind,
        stability=stability,
    )


_OPERATIONAL_REASON = (
    "The handler remains the established HTTP compatibility adapter; it does not call one fixed archive operation."
)

ROUTES: tuple[RouteSpec, ...] = (
    _route(
        "POST",
        "/api/web-auth/session",
        "_handle_web_auth_bootstrap",
        migration_reason="First-party credential lifecycle is owned by the daemon credential registry.",
    ),
    _route(
        "DELETE",
        "/api/web-auth/session",
        "_handle_web_auth_revoke",
        migration_reason="First-party credential lifecycle is owned by the daemon credential registry.",
    ),
    _route("GET", "/api/health/check", "_handle_health_check", migration_reason=_OPERATIONAL_REASON),
    _route("GET", "/api/health", "_handle_health", migration_reason=_OPERATIONAL_REASON),
    _route(
        "GET",
        "/api/events",
        "_handle_events",
        passes_params=True,
        auth_scope="events",
        migration_reason="The daemon event bus owns polling and SSE streaming.",
    ),
    _route(
        "GET",
        "/api/agents/coordination",
        "_handle_agent_coordination",
        passes_params=True,
        migration_reason=_OPERATIONAL_REASON,
    ),
    _route(
        "GET", "/api/provider-usage", "_handle_provider_usage", passes_params=True, migration_reason=_OPERATIONAL_REASON
    ),
    _route(
        "GET", "/api/archive-debt", "_handle_archive_debt", passes_params=True, migration_reason=_OPERATIONAL_REASON
    ),
    _route(
        "GET", "/api/import/explain", "_handle_import_explain", passes_params=True, migration_reason=_OPERATIONAL_REASON
    ),
    _route(
        "POST",
        "/api/operation",
        "_handle_daemon_operation",
        migration_reason="The payload selects its declared operation dynamically; the HTTP endpoint is the shared dispatcher.",
    ),
    _route(
        "POST",
        "/api/cli/query",
        "_handle_cli_query",
        migration_reason="The CLI query compatibility body is lowered by the daemon query adapter.",
    ),
    _route(
        "POST",
        "/api/telemetry/mcp-calls",
        "_handle_mcp_call_log",
        write_gate=True,
        migration_reason="Bounded telemetry receipt is written by the daemon writer.",
    ),
    _route("GET", "/api/webui/observability", "_handle_webui_observability", migration_reason=_OPERATIONAL_REASON),
    _route(
        "GET",
        "/api/webui/freshness",
        "_handle_webui_source_freshness",
        passes_params=True,
        migration_reason=_OPERATIONAL_REASON,
    ),
    _route(
        "GET",
        "/api/webui/insights/:name",
        "_handle_webui_insight",
        passes_params=True,
        migration_reason=_OPERATIONAL_REASON,
    ),
)

__all__ = ["ROUTES"]
