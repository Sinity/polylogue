"""Daemon route-contract metadata tests (#1847)."""

from __future__ import annotations

from http import HTTPStatus

import pytest

from polylogue.daemon.route_contracts import ROUTE_CONTRACTS, route_contract_for, stable_route_contracts
from tests.unit.daemon.test_daemon_http_security import (
    ENDPOINTS_DELETE,
    ENDPOINTS_GET,
    ENDPOINTS_POST,
    _capture_responses,
    _make_handler,
)


def test_route_contract_patterns_are_unique_per_method() -> None:
    """One method/pattern pair has exactly one contract owner."""

    pairs = [(route.method, route.pattern) for route in ROUTE_CONTRACTS]
    assert len(pairs) == len(set(pairs))


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


@pytest.mark.parametrize(
    ("method", "path", "expected_pattern", "expected_kind", "expected_auth"),
    [
        ("GET", "/", "/", "browser_shell", "unauthenticated_loopback"),
        ("GET", "/healthz/live", "/healthz/live", "operational", "unauthenticated_loopback"),
        ("GET", "/metrics", "/metrics", "operational", "unauthenticated_loopback"),
        ("GET", "/api/sessions", "/api/sessions", "read_query", "bearer_if_configured"),
        ("GET", "/api/query-units", "/api/query-units", "read_query", "bearer_if_configured"),
        (
            "GET",
            "/api/sessions/codex-session:abc/messages",
            "/api/sessions/:id/messages",
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
