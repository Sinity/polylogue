"""Daemon route-contract metadata tests (#1847)."""

from __future__ import annotations

from http import HTTPStatus

import pytest

from polylogue.daemon.http import _parameterized_get_routes, _static_get_routes, implemented_daemon_route_patterns
from polylogue.daemon.route_contracts import (
    ROUTE_CONTRACTS,
    route_contract_for,
    route_contract_for_pattern,
    stable_route_contracts,
)
from tests.unit.daemon.test_daemon_http_security import (
    ENDPOINTS_DELETE,
    ENDPOINTS_GET,
    ENDPOINTS_POST,
    _capture_responses,
    _make_handler,
)


def _implemented_route_set() -> set[tuple[str, str]]:
    routes = implemented_daemon_route_patterns()
    assert len(routes) == len(set(routes))
    return set(routes)


def test_route_contract_patterns_are_unique_per_method() -> None:
    """One method/pattern pair has exactly one contract owner."""

    pairs = [(route.method, route.pattern) for route in ROUTE_CONTRACTS]
    assert len(pairs) == len(set(pairs))


def test_implemented_daemon_routes_have_contract_metadata() -> None:
    """Every implemented daemon route must declare kind, auth, and payload posture."""

    declared: set[tuple[str, str]] = {(route.method, route.pattern) for route in ROUTE_CONTRACTS}
    implemented = _implemented_route_set()
    assert implemented - declared == set()


def test_route_contract_metadata_targets_live_daemon_routes() -> None:
    """Route contracts should describe actual dispatcher routes, not stale docs."""

    declared: set[tuple[str, str]] = {(route.method, route.pattern) for route in ROUTE_CONTRACTS}
    implemented = _implemented_route_set()
    assert declared - implemented == set()


def test_get_dispatch_tables_are_bound_to_route_contracts() -> None:
    """GET dispatcher entries consume route contracts instead of restating them."""

    for static_route in _static_get_routes():
        assert static_route.contract is route_contract_for_pattern("GET", static_route.pattern)
        assert static_route.pattern == static_route.contract.pattern
    for parameterized_route in _parameterized_get_routes():
        assert parameterized_route.contract is route_contract_for_pattern("GET", parameterized_route.pattern)
        assert parameterized_route.pattern == parameterized_route.contract.pattern


def test_stable_routes_have_explicit_auth_and_response_contracts() -> None:
    """Stable routes must declare the security posture and response shape."""

    stable_routes = stable_route_contracts()
    assert stable_routes
    for route in stable_routes:
        assert route.auth_policy in {
            "bearer_if_configured",
            "bearer_and_same_origin",
            "unauthenticated_loopback",
            "observability_flag_then_loopback_or_bearer",
        }
        assert route.response_contract


def test_stable_user_overlay_mutations_use_shared_envelope_contract() -> None:
    """Browser write routes expose the common mutation result payload (#1847)."""

    for route in stable_route_contracts():
        if route.kind == "user_overlay" and route.method in {"POST", "DELETE"}:
            assert route.auth_policy == "bearer_and_same_origin"
            assert route.response_contract == "mutation envelope"


def test_evidence_routes_are_published_as_stable_api() -> None:
    """Typed assertion and read-view routes are stable API contracts (#1847)."""

    assertion_route = route_contract_for("GET", "/api/assertions")
    read_route = route_contract_for("GET", "/api/sessions/codex-session:abc/read")

    assert assertion_route is not None
    assert assertion_route.stability == "stable"
    assert assertion_route.auth_policy == "bearer_if_configured"
    assert assertion_route.response_contract == "AssertionClaimListPayload"

    assert read_route is not None
    assert read_route.stability == "stable"
    assert read_route.auth_policy == "bearer_if_configured"
    assert read_route.response_contract == "SessionReadViewEnvelope"


def test_read_view_execution_route_is_published_as_stable_api() -> None:
    """The read-view route has a typed envelope and stable route contract."""

    stable_patterns = {route.pattern for route in stable_route_contracts()}
    assert "/api/sessions/:id/read" in stable_patterns
    route = route_contract_for("GET", "/api/sessions/codex-session:abc/read")
    assert route is not None
    assert route.kind == "read_detail"
    assert route.response_contract == "SessionReadViewEnvelope"


@pytest.mark.parametrize(
    ("method", "path", "expected_pattern", "expected_kind", "expected_auth"),
    [
        ("GET", "/", "/", "browser_shell", "unauthenticated_loopback"),
        ("GET", "/healthz/live", "/healthz/live", "operational", "unauthenticated_loopback"),
        ("GET", "/metrics", "/metrics", "operational", "unauthenticated_loopback"),
        ("GET", "/api/sessions", "/api/sessions", "read_query", "bearer_if_configured"),
        ("GET", "/api/query-units", "/api/query-units", "read_query", "bearer_if_configured"),
        ("GET", "/api/archive-debt", "/api/archive-debt", "operational", "bearer_if_configured"),
        ("GET", "/api/dev-loop", "/api/dev-loop", "operational", "bearer_if_configured"),
        ("GET", "/api/agents/coordination", "/api/agents/coordination", "operational", "bearer_if_configured"),
        ("GET", "/api/import/explain", "/api/import/explain", "operational", "bearer_if_configured"),
        ("GET", "/api/assertions", "/api/assertions", "user_overlay", "bearer_if_configured"),
        (
            "GET",
            "/api/sessions/codex-session:abc/messages",
            "/api/sessions/:id/messages",
            "read_detail",
            "bearer_if_configured",
        ),
        (
            "GET",
            "/api/sessions/codex-session:abc/read",
            "/api/sessions/:id/read",
            "read_detail",
            "bearer_if_configured",
        ),
        (
            "GET",
            "/api/sessions/codex-session:abc/provenance",
            "/api/sessions/:id/provenance",
            "read_detail",
            "bearer_if_configured",
        ),
        ("POST", "/api/user/marks", "/api/user/marks", "user_overlay", "bearer_and_same_origin"),
        (
            "DELETE",
            "/api/user/saved-views/view-1",
            "/api/user/saved-views/:id",
            "user_overlay",
            "bearer_and_same_origin",
        ),
    ],
)
def test_route_contract_lookup_matches_static_and_parameterized_routes(
    method: str,
    path: str,
    expected_pattern: str,
    expected_kind: str,
    expected_auth: str,
) -> None:
    route = route_contract_for(method, path)
    assert route is not None
    assert route.pattern == expected_pattern
    assert route.kind == expected_kind
    assert route.auth_policy == expected_auth


@pytest.mark.parametrize("path", ENDPOINTS_GET)
def test_authenticated_get_security_matrix_routes_are_classified(path: str) -> None:
    """The existing GET auth matrix should be a subset of route contracts."""

    route = route_contract_for("GET", path)
    assert route is not None
    assert route.auth_policy == "bearer_if_configured"


@pytest.mark.parametrize("path", ENDPOINTS_POST)
def test_post_security_matrix_routes_are_mutation_classified(path: str) -> None:
    """Every POST endpoint in the security matrix should require origin checks."""

    route = route_contract_for("POST", path)
    assert route is not None
    assert route.auth_policy == "bearer_and_same_origin"


@pytest.mark.parametrize("path", ENDPOINTS_DELETE)
def test_delete_security_matrix_routes_are_mutation_classified(path: str) -> None:
    """Every DELETE endpoint in the security matrix should require origin checks."""

    route = route_contract_for("DELETE", path)
    assert route is not None
    assert route.auth_policy == "bearer_and_same_origin"


def test_unknown_route_has_no_contract() -> None:
    assert route_contract_for("GET", "/api/not-a-real-route") is None


@pytest.mark.parametrize("path", ["/", "/s/codex-session:abc", "/p", "/a"])
def test_shell_bootstrap_is_unauthenticated_on_loopback(path: str) -> None:
    """Local shell bootstrap remains frictionless on loopback."""

    handler = _make_handler("GET", path)
    send_error, _ = _capture_responses(handler)
    handler._serve_web_shell = lambda: None  # type: ignore[method-assign]
    handler._serve_paste_browser_page = lambda: None  # type: ignore[method-assign]
    handler._serve_attachment_library_page = lambda: None  # type: ignore[method-assign]

    handler.do_GET()

    for call in send_error.call_args_list:
        assert call.args[0] != HTTPStatus.UNAUTHORIZED


@pytest.mark.parametrize("path", ["/", "/s/codex-session:abc", "/p", "/a"])
def test_shell_bootstrap_requires_token_on_non_loopback(path: str) -> None:
    """Remote-bound shell bootstrap must not bypass the daemon token."""

    class RemoteMockServer:
        auth_token = "secret"
        api_host = "0.0.0.0"

    handler = _make_handler("GET", path)
    handler.server = RemoteMockServer()  # type: ignore[assignment]
    handler.client_address = ("192.0.2.10", 12345)
    send_error, _ = _capture_responses(handler)

    handler.do_GET()

    send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")
