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
    "bearer_if_configured",
    "bearer_and_same_origin",
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


ROUTE_CONTRACTS: tuple[RouteContract, ...] = (
    RouteContract(
        "GET",
        "/",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "text/html web shell",
        "HTML bootstrap only; shell JavaScript calls authenticated /api routes.",
    ),
    RouteContract(
        "GET",
        "/s/:session_id",
        "browser_shell",
        "shell_supported",
        "unauthenticated_loopback",
        "text/html web shell",
        "Session deep-link bootstrap.",
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
    RouteContract(
        "POST",
        "/v1/traces",
        "observability",
        "private",
        "observability_flag_then_loopback_or_bearer",
        "OTLP protobuf response",
    ),
    RouteContract(
        "POST",
        "/v1/metrics",
        "observability",
        "private",
        "observability_flag_then_loopback_or_bearer",
        "OTLP protobuf response",
    ),
    RouteContract(
        "POST",
        "/v1/logs",
        "observability",
        "private",
        "observability_flag_then_loopback_or_bearer",
        "OTLP protobuf response",
    ),
    RouteContract("GET", "/api/health/check", "operational", "stable", "bearer_if_configured", "JSON"),
    RouteContract("GET", "/api/health", "operational", "stable", "bearer_if_configured", "JSON"),
    RouteContract("GET", "/api/status", "operational", "stable", "bearer_if_configured", "Daemon status JSON"),
    RouteContract("GET", "/api/events", "operational", "stable", "bearer_if_configured", "SSE or JSON event poll"),
    RouteContract("GET", "/api/sessions", "read_query", "stable", "bearer_if_configured", "SearchEnvelope"),
    RouteContract("GET", "/api/facets", "read_query", "stable", "bearer_if_configured", "Facets envelope"),
    RouteContract("GET", "/api/query-units", "read_query", "stable", "bearer_if_configured", "QueryUnitResultEnvelope"),
    RouteContract(
        "GET",
        "/api/archive-debt",
        "operational",
        "stable",
        "bearer_if_configured",
        "ArchiveDebtListPayload",
        "Unified archive debt rows shared by CLI, Python API, MCP, and daemon clients.",
    ),
    RouteContract(
        "GET",
        "/api/import/explain",
        "operational",
        "shell_supported",
        "bearer_if_configured",
        "ImportExplainPayload",
        "Local import/source evidence explanation; paths are redacted unless explicitly requested.",
    ),
    RouteContract(
        "GET", "/api/refs/resolve", "read_query", "stable", "bearer_if_configured", "PublicRefResolutionPayload"
    ),
    RouteContract(
        "GET",
        "/api/query-completions",
        "read_query",
        "stable",
        "bearer_if_configured",
        "query completion metadata",
    ),
    RouteContract(
        "GET",
        "/api/read-view-profiles",
        "read_query",
        "stable",
        "bearer_if_configured",
        "read-view profile metadata",
    ),
    RouteContract(
        "GET",
        "/api/assertions",
        "user_overlay",
        "stable",
        "bearer_if_configured",
        "AssertionClaimListPayload",
        "Read-only assertion-backed overlay claims shared by the web workbench and API clients.",
    ),
    RouteContract("GET", "/api/sources", "read_detail", "shell_supported", "bearer_if_configured", "source list JSON"),
    RouteContract("GET", "/api/sessions/:id", "read_detail", "stable", "bearer_if_configured", "Session detail JSON"),
    RouteContract(
        "GET",
        "/api/sessions/:id/messages",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "session messages JSON",
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/recovery",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "RecoveryReadPayload",
        "Consumes shared recovery digest/work-packet DTOs for the local workbench and API clients.",
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/read",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "SessionReadViewEnvelope",
        "Executes supported single-session read profiles over shared DTO/facade payload helpers.",
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/raw",
        "read_detail",
        "shell_supported",
        "bearer_if_configured",
        "raw session payload JSON",
        "Raw preview is opt-in and authenticated.",
    ),
    RouteContract(
        "GET", "/api/sessions/:id/cost", "read_detail", "shell_supported", "bearer_if_configured", "cost JSON"
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/provenance",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "provenance envelope",
        "Raw bytes require the include_raw query parameter.",
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/topology",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "topology envelope",
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/topology/parent-chain",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "parent-chain topology envelope",
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/similar",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "similar-session envelope",
    ),
    RouteContract(
        "GET",
        "/api/sessions/:id/attachments",
        "read_detail",
        "shell_supported",
        "bearer_if_configured",
        "session attachment envelope",
    ),
    RouteContract(
        "GET",
        "/api/insights/sessions/:id",
        "read_detail",
        "stable",
        "bearer_if_configured",
        "session insights envelope",
    ),
    RouteContract(
        "GET",
        "/api/raw_artifacts/:id",
        "read_detail",
        "shell_supported",
        "bearer_if_configured",
        "raw artifact preview",
        "Authenticated raw preview helper for the local shell.",
    ),
    RouteContract(
        "GET",
        "/api/thread-continue-templates",
        "read_detail",
        "shell_supported",
        "bearer_if_configured",
        "thread continuation templates",
    ),
    RouteContract(
        "GET", "/api/paste-browser", "read_query", "shell_supported", "bearer_if_configured", "paste browser JSON"
    ),
    RouteContract(
        "GET", "/api/attachments", "read_query", "shell_supported", "bearer_if_configured", "attachment library JSON"
    ),
    RouteContract("GET", "/api/stack", "workspace", "shell_supported", "bearer_if_configured", "stack workspace JSON"),
    RouteContract(
        "GET", "/api/compare", "workspace", "shell_supported", "bearer_if_configured", "compare workspace JSON"
    ),
    RouteContract("GET", "/api/user/marks", "user_overlay", "stable", "bearer_if_configured", "marks JSON"),
    RouteContract("GET", "/api/user/annotations", "user_overlay", "stable", "bearer_if_configured", "annotations JSON"),
    RouteContract(
        "GET", "/api/user/annotations/:id", "user_overlay", "stable", "bearer_if_configured", "annotation JSON"
    ),
    RouteContract("GET", "/api/user/saved-views", "user_overlay", "stable", "bearer_if_configured", "saved views JSON"),
    RouteContract(
        "GET", "/api/user/saved-views/:id", "user_overlay", "stable", "bearer_if_configured", "saved view JSON"
    ),
    RouteContract(
        "GET", "/api/user/recall-packs", "user_overlay", "stable", "bearer_if_configured", "recall packs JSON"
    ),
    RouteContract(
        "GET", "/api/user/recall-packs/:id", "user_overlay", "stable", "bearer_if_configured", "recall pack JSON"
    ),
    RouteContract("GET", "/api/user/workspaces", "user_overlay", "stable", "bearer_if_configured", "workspaces JSON"),
    RouteContract(
        "GET", "/api/user/workspaces/:id", "user_overlay", "stable", "bearer_if_configured", "workspace JSON"
    ),
    RouteContract(
        "GET",
        "/api/maintenance/operations",
        "maintenance",
        "stable",
        "bearer_if_configured",
        "maintenance operations JSON",
    ),
    RouteContract(
        "GET",
        "/api/maintenance/status/:id",
        "maintenance",
        "stable",
        "bearer_if_configured",
        "maintenance operation status JSON",
    ),
    RouteContract("POST", "/api/reset", "maintenance", "stable", "bearer_and_same_origin", "reset result JSON"),
    RouteContract("POST", "/api/ingest", "maintenance", "stable", "bearer_and_same_origin", "ingest result JSON"),
    RouteContract(
        "POST",
        "/api/maintenance/plan",
        "maintenance",
        "stable",
        "bearer_and_same_origin",
        "maintenance operation preview",
    ),
    RouteContract(
        "POST",
        "/api/maintenance/run",
        "maintenance",
        "stable",
        "bearer_and_same_origin",
        "maintenance operation result",
    ),
    RouteContract("POST", "/api/user/marks", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"),
    RouteContract(
        "POST", "/api/user/annotations", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "POST", "/api/user/saved-views", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "POST", "/api/user/recall-packs", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "POST", "/api/user/workspaces", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
    ),
    RouteContract("DELETE", "/api/user/marks", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"),
    RouteContract(
        "DELETE", "/api/user/annotations/:id", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "DELETE", "/api/user/saved-views/:id", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "DELETE", "/api/user/recall-packs/:id", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
    ),
    RouteContract(
        "DELETE", "/api/user/workspaces/:id", "user_overlay", "stable", "bearer_and_same_origin", "mutation envelope"
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
    "ROUTE_CONTRACTS",
    "RouteContract",
    "route_contract_for",
    "route_contract_for_pattern",
    "stable_route_contracts",
]
