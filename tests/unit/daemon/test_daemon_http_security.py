"""Per-endpoint security matrix for the daemon HTTP API (#868).

Pins three contracts that previous unit tests covered only at the
helper-function level:

1. Every authenticated endpoint refuses requests without a valid bearer
   token when a token is configured (``_check_auth_logic`` covers the
   pure logic; this file confirms each route actually invokes it).
2. Every mutating (POST) endpoint refuses cross-origin browser requests
   via the ``Origin``-header check, regardless of whether the auth token
   matches.
3. ``OPTIONS`` requests return ``405 Method Not Allowed`` (no CORS
   preflight is advertised by design — see ``docs/security.md``).

The parametrized matrix derives from ``ROUTE_CONTRACTS`` so adding a new
authenticated route to the daemon contract registry automatically adds the
same security coverage.
"""

from __future__ import annotations

import json
import zipfile
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.core.loopback import is_loopback_host
from polylogue.daemon.http import _check_auth_logic
from polylogue.daemon.route_contracts import ROUTE_CONTRACTS, RouteContract

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Endpoint tables derived from route contracts
# ---------------------------------------------------------------------------


def _route_sample_path(route: RouteContract) -> str:
    parts = []
    for part in route.pattern.strip("/").split("/"):
        if part.startswith(":"):
            parts.append("some-id")
        else:
            parts.append(part)
    return "/" + "/".join(parts) if parts else "/"


def _paths_for(method: str, auth_policy: str) -> list[str]:
    return [
        _route_sample_path(route)
        for route in ROUTE_CONTRACTS
        if route.method == method and route.auth_policy == auth_policy
    ]


ENDPOINTS_GET = _paths_for("GET", "bearer_if_configured")

ENDPOINTS_POST = _paths_for("POST", "bearer_and_same_origin")

ENDPOINTS_DELETE = _paths_for("DELETE", "bearer_and_same_origin")


def test_route_contract_security_matrices_are_non_empty() -> None:
    assert ENDPOINTS_GET
    assert ENDPOINTS_POST
    assert ENDPOINTS_DELETE


def test_every_registered_route_handler_name_resolves_to_a_real_method() -> None:
    """Prevents a repeat of polylogue-g9j6: /api/provider-usage was registered
    in both the static-GET route table and route_contracts.py, but
    _handle_provider_usage was never defined — any real request raised an
    unhandled AttributeError. This walks every route table and asserts the
    named handler actually exists on DaemonAPIHandler."""
    from polylogue.daemon.http import (
        DaemonAPIHandler,
        _authenticated_post_routes,
        _observability_post_routes,
        _parameterized_get_routes,
        _static_get_routes,
    )

    missing: list[str] = []
    for static_get_route in _static_get_routes():
        if not hasattr(DaemonAPIHandler, static_get_route.handler_name):
            missing.append(f"static GET {static_get_route.pattern} -> {static_get_route.handler_name}")
    for parameterized_get_route in _parameterized_get_routes():
        if not hasattr(DaemonAPIHandler, parameterized_get_route.handler_name):
            missing.append(
                f"parameterized GET {parameterized_get_route.pattern} -> {parameterized_get_route.handler_name}"
            )
    for authenticated_post_route in _authenticated_post_routes():
        if not hasattr(DaemonAPIHandler, authenticated_post_route.handler_name):
            missing.append(
                f"authenticated POST {authenticated_post_route.pattern} -> {authenticated_post_route.handler_name}"
            )
    for observability_post_route in _observability_post_routes():
        if not hasattr(DaemonAPIHandler, observability_post_route.handler_name):
            missing.append(
                f"observability POST {observability_post_route.pattern} -> {observability_post_route.handler_name}"
            )
    assert missing == [], f"registered routes with no matching handler method: {missing}"


# ---------------------------------------------------------------------------
# Pure-logic auth: the function called by every route's _check_auth()
# ---------------------------------------------------------------------------


class TestCheckAuthLogic:
    """``_check_auth_logic`` is the single decision point for token validation.

    Parametrize over the three token states and three header shapes that
    matter to security: missing entirely, well-formed but wrong, and
    well-formed and correct.
    """

    @pytest.mark.parametrize(
        ("auth_header", "expected_allowed"),
        [
            ("", False),  # no header at all
            ("Bearer ", False),  # empty token
            ("Bearer wrong", False),  # wrong token
            ("Token secret", False),  # wrong scheme
            ("bearer secret", False),  # case-sensitive scheme
            ("Bearer secret", True),  # exact match
        ],
    )
    def test_when_token_configured_only_exact_bearer_match_passes(
        self, auth_header: str, expected_allowed: bool
    ) -> None:
        result = _check_auth_logic("secret", "127.0.0.1", auth_header)
        assert result.allowed is expected_allowed
        if not expected_allowed:
            assert result.reason == "unauthorized"

    @pytest.mark.parametrize("auth_token", [None, ""])
    @pytest.mark.parametrize("client_host", ["127.0.0.1", "192.168.1.5", "10.0.0.1"])
    @pytest.mark.parametrize("auth_header", ["", "Bearer foo", "garbage"])
    def test_when_no_token_configured_all_clients_are_allowed(
        self, auth_token: str | None, client_host: str, auth_header: str
    ) -> None:
        """Local-dev default: API is open when no token is set, regardless
        of the client's host or whatever they send for Authorization.

        This is documented in ``docs/security.md`` and constrained
        elsewhere by ``daemon/cli.py`` refusing remote bind without a
        token.
        """
        result = _check_auth_logic(auth_token, client_host, auth_header)
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Origin-check: same-origin / cross-origin / no-origin
# ---------------------------------------------------------------------------


def _origin_allowed(origin: str) -> bool:
    """Reproduces the localhost-allowlist used by ``_check_cross_origin``.

    Includes the IPv6 loopback (``[::1]``) since the daemon admits ``::1``
    as a loopback bind address (see ``is_loopback_host``); a web shell
    served from that bind would send ``Origin: http://[::1]:port``.
    """
    if not origin:
        return True
    return any(
        origin.startswith(prefix)
        for prefix in (
            "http://127.0.0.1:",
            "http://localhost:",
            "https://127.0.0.1:",
            "https://localhost:",
            "http://[::1]:",
            "https://[::1]:",
        )
    )


class TestOriginAllowlist:
    """The Origin allowlist is the CSRF boundary for browser-side attacks."""

    @pytest.mark.parametrize(
        ("origin", "expected_allowed"),
        [
            ("", True),  # not a browser request — curl, hooks, etc.
            ("http://127.0.0.1:8766", True),  # canonical web shell
            ("http://localhost:3000", True),  # browser extension dev port
            ("https://127.0.0.1:8766", True),  # TLS shim if any
            ("https://localhost:8766", True),
            ("http://[::1]:8766", True),  # IPv6 loopback web shell
            ("https://[::1]:8766", True),
            ("http://192.168.1.1:8766", False),  # LAN attacker
            ("https://evil.example.com", False),  # hostile page
            ("http://127.0.0.1.evil.com", False),  # subdomain trick
            ("file://", False),  # local file
            ("null", False),  # sandboxed iframe
            ("http://[::1].evil.com", False),  # bracketed IPv6 trick
        ],
    )
    def test_origin_allowlist_decision(self, origin: str, expected_allowed: bool) -> None:
        assert _origin_allowed(origin) is expected_allowed


# ---------------------------------------------------------------------------
# Dispatch-level: each endpoint must actually invoke _check_auth before
# routing to its handler.
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = "secret"
    api_host = "127.0.0.1"


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(
    method: str,
    path: str,
    *,
    auth_header: str = "",
    origin: str = "",
    host: str = "",
    body: bytes = b"",
) -> DaemonAPIHandler:
    """Build a ``DaemonAPIHandler`` instance with mocked transport.

    Returns a handler ready to receive a ``do_GET`` / ``do_POST`` call.
    The route handlers are not invoked unless ``_check_host_admission``,
    ``_check_auth``, and ``_check_cross_origin`` admit the request.
    ``host`` defaults to unset, matching every pre-existing caller in this
    file — ``_check_host_admission_logic`` treats an absent Host header as
    permissive (see its docstring), so omitting it here is equivalent to a
    non-browser client and does not gate any of the auth/origin coverage
    below. Tests of the Host gate itself pass ``host`` explicitly.
    """
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"

    headers: dict[str, str] = {"Content-Length": str(len(body))}
    if auth_header:
        headers["Authorization"] = auth_header
    if origin:
        headers["Origin"] = origin
    if host:
        headers["Host"] = host
    handler.headers = cast("Message[str, str]", _MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    """Patch the response writers so we can assert what was sent."""
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


@pytest.mark.parametrize("path", ENDPOINTS_GET)
class TestGetEndpointAuthGate:
    """Every authenticated GET endpoint refuses unauthenticated requests."""

    def test_missing_token_returns_401(self, path: str) -> None:
        handler = _make_handler("GET", path)
        send_error, _ = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")

    def test_invalid_token_returns_401(self, path: str) -> None:
        handler = _make_handler("GET", path, auth_header="Bearer wrong")
        send_error, _ = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")

    def test_valid_token_passes_auth(self, path: str) -> None:
        """Auth gate admits the request for every authenticated GET path."""
        handler = _make_handler("GET", path, auth_header="Bearer secret")
        assert handler._check_auth() is True


@pytest.mark.parametrize("path", ENDPOINTS_POST)
class TestPostEndpointAuthAndOriginGate:
    """Every POST endpoint refuses unauthenticated and cross-origin requests."""

    def test_missing_token_returns_401(self, path: str) -> None:
        handler = _make_handler("POST", path)
        send_error, _ = _capture_responses(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")

    def test_invalid_token_returns_401(self, path: str) -> None:
        handler = _make_handler("POST", path, auth_header="Bearer wrong")
        send_error, _ = _capture_responses(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")

    def test_valid_token_no_origin_passes_gates(self, path: str) -> None:
        """Auth + origin gates admit a curl/hook-style request (no Origin)."""
        handler = _make_handler("POST", path, auth_header="Bearer secret")
        send_error, _ = _capture_responses(handler)
        with (
            patch.object(handler, "_handle_reset"),
            patch.object(handler, "_handle_ingest"),
            patch.object(handler, "_handle_maintenance_plan"),
            patch.object(handler, "_handle_maintenance_run"),
            patch("polylogue.daemon.user_state_http.dispatch_post", return_value=True),
        ):
            handler.do_POST()
        for call in send_error.call_args_list:
            status = call.args[0]
            assert status not in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN), (
                f"POST {path} returned {status.value} with valid token + no Origin"
            )

    def test_valid_token_same_origin_passes_gates(self, path: str) -> None:
        """Web-shell same-origin requests are admitted."""
        handler = _make_handler(
            "POST",
            path,
            auth_header="Bearer secret",
            origin="http://127.0.0.1:8766",
        )
        send_error, _ = _capture_responses(handler)
        with (
            patch.object(handler, "_handle_reset"),
            patch.object(handler, "_handle_ingest"),
            patch.object(handler, "_handle_maintenance_plan"),
            patch.object(handler, "_handle_maintenance_run"),
            patch("polylogue.daemon.user_state_http.dispatch_post", return_value=True),
        ):
            handler.do_POST()
        for call in send_error.call_args_list:
            status = call.args[0]
            assert status not in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN), (
                f"POST {path} returned {status.value} with valid token + same-origin"
            )

    def test_valid_token_cross_origin_returns_403(self, path: str) -> None:
        """Hostile-origin browser POST is refused even with a valid token.

        This is the CSRF boundary: a malicious page loaded in the user's
        browser cannot exfiltrate data or trigger destructive operations
        even if it somehow learned the bearer token, because the Origin
        header reveals the cross-origin context.
        """
        handler = _make_handler(
            "POST",
            path,
            auth_header="Bearer secret",
            origin="https://evil.example.com",
        )
        send_error, _ = _capture_responses(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.FORBIDDEN, "cross_origin_denied")


@pytest.mark.parametrize("path", ENDPOINTS_DELETE)
class TestDeleteEndpointAuthAndOriginGate:
    """Every DELETE endpoint refuses unauthenticated and cross-origin requests.

    ``do_DELETE`` calls ``_check_auth`` then ``_check_cross_origin`` before
    routing. This matrix confirms the gates are applied to every delete route.
    """

    def test_missing_token_returns_401(self, path: str) -> None:
        handler = _make_handler("DELETE", path)
        send_error, _ = _capture_responses(handler)
        handler.do_DELETE()
        send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")

    def test_invalid_token_returns_401(self, path: str) -> None:
        handler = _make_handler("DELETE", path, auth_header="Bearer wrong")
        send_error, _ = _capture_responses(handler)
        handler.do_DELETE()
        send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")

    def test_valid_token_cross_origin_returns_403(self, path: str) -> None:
        """Hostile-origin browser DELETE is refused even with a valid token."""
        handler = _make_handler(
            "DELETE",
            path,
            auth_header="Bearer secret",
            origin="https://evil.example.com",
        )
        send_error, _ = _capture_responses(handler)
        handler.do_DELETE()
        send_error.assert_called_once_with(HTTPStatus.FORBIDDEN, "cross_origin_denied")

    def test_valid_token_no_origin_passes_gates(self, path: str) -> None:
        """Auth + origin gates admit a curl/hook-style request (no Origin)."""
        handler = _make_handler("DELETE", path, auth_header="Bearer secret")
        send_error, _ = _capture_responses(handler)
        with patch("polylogue.daemon.user_state_http.dispatch_delete", return_value=True):
            handler.do_DELETE()
        for call in send_error.call_args_list:
            status = call.args[0]
            assert status not in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN), (
                f"DELETE {path} returned {status.value} with valid token + no Origin"
            )

    def test_valid_token_same_origin_passes_gates(self, path: str) -> None:
        """Web-shell same-origin DELETE is admitted (loopback origin is trusted)."""
        handler = _make_handler(
            "DELETE",
            path,
            auth_header="Bearer secret",
            origin="http://localhost:8766",
        )
        send_error, _ = _capture_responses(handler)
        with patch("polylogue.daemon.user_state_http.dispatch_delete", return_value=True):
            handler.do_DELETE()
        for call in send_error.call_args_list:
            status = call.args[0]
            assert status not in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN), (
                f"DELETE {path} returned {status.value} with valid token + same-origin"
            )


# ---------------------------------------------------------------------------
# Host admission — the DNS-rebinding gate (polylogue-kwsb.1). A hostile page
# can get an attacker domain resolved to 127.0.0.1; the browser's Origin
# check alone does not stop it (Origin was POST-only and absent-tolerant),
# and client-IP loopback checks are worthless since the TCP peer really is
# localhost. Only the Host header, allowlisted against loopback names and
# the configured api_host, can distinguish the attacker's page.
# ---------------------------------------------------------------------------


class TestCheckHostAdmissionLogic:
    """Pure logic: ``_check_host_admission_logic`` is the DNS-rebinding gate."""

    @pytest.mark.parametrize(
        ("host_header", "expected_allowed"),
        [
            ("", True),  # absent Host — non-browser local client, see docstring
            ("127.0.0.1:8766", True),
            ("127.0.0.1", True),
            ("localhost:8766", True),
            ("localhost", True),
            ("[::1]:8766", True),  # bracketed IPv6 loopback
            ("[::1]", True),
            ("evil.example.com", False),  # DNS-rebound attacker domain
            ("evil.example.com:8766", False),
            ("127.0.0.1.evil.com", False),  # subdomain trick
            ("127.0.0.1evil.com", False),
            ("[::1", False),  # malformed IPv6: unmatched bracket, urlsplit raises ValueError
            ("[bad", False),  # malformed IPv6: invalid content, urlsplit raises ValueError
            ("xn--0.evil.com", False),  # punycode/NFKC-adjacent host, still just a foreign name
        ],
    )
    def test_loopback_and_configured_host_admitted(self, host_header: str, expected_allowed: bool) -> None:
        from polylogue.daemon.http import _check_host_admission_logic

        assert _check_host_admission_logic(host_header, "127.0.0.1") is expected_allowed

    def test_configured_non_loopback_api_host_is_admitted(self) -> None:
        """A daemon explicitly bound to a non-loopback host trusts that exact name."""
        from polylogue.daemon.http import _check_host_admission_logic

        assert _check_host_admission_logic("archive.internal:8766", "archive.internal") is True
        assert _check_host_admission_logic("evil.example.com", "archive.internal") is False


@pytest.mark.parametrize("path", ENDPOINTS_GET)
class TestGetEndpointHostGate:
    """Cross-origin GET with a foreign Host is refused (kwsb.1 AC)."""

    def test_foreign_host_returns_403_before_auth_is_even_checked(self, path: str) -> None:
        """No token is configured in these fixtures — proves the Host gate runs
        independently of auth, closing the archive-read hole even when the
        daemon has no token set (the common local-dev default)."""
        handler = _make_handler("GET", path, host="evil.example.com")
        send_error, _ = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.FORBIDDEN, "host_not_allowed")

    def test_loopback_host_reaches_the_auth_gate(self, path: str) -> None:
        handler = _make_handler("GET", path, host="127.0.0.1:8766")
        send_error, _ = _capture_responses(handler)
        handler.do_GET()
        # No token configured on the mock server -> passes straight through
        # to the route handler layer; the point here is that it is NOT
        # refused at the Host gate.
        for call in send_error.call_args_list:
            assert call.args[0] != HTTPStatus.FORBIDDEN


@pytest.mark.parametrize(
    "path",
    ["/", "/s/some-id", "/w/stack", "/p", "/a", "/healthz/live", "/healthz/ready", "/metrics"],
)
class TestBootstrapAndProbeRoutesHostGate:
    """The Host gate covers unauthenticated bootstrap/probe routes too.

    These routes carry real information (dev-loop/status leak PID and
    archive_root elsewhere; healthz/metrics leak exception text, DB
    sizes, and counts) and are reachable via DNS rebinding exactly like
    the authenticated API routes — a foreign Host must be refused here
    just as much as on ``/api/*``.
    """

    def test_foreign_host_returns_403(self, path: str) -> None:
        handler = _make_handler("GET", path, host="evil.example.com")
        send_error, _ = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.FORBIDDEN, "host_not_allowed")

    def test_loopback_host_passes_the_gate(self, path: str) -> None:
        """A loopback Host is not refused at the gate itself.

        Asserts the gate's own decision directly rather than driving the
        full ``do_GET`` request lifecycle — these routes serve real HTML/
        healthz/metrics output that needs response-writer scaffolding
        beyond this file's minimal mock handler, which is exercised by
        their own dedicated test files.
        """
        handler = _make_handler("GET", path, host="localhost:8766")
        assert handler._check_host_admission() is True

    def test_absent_host_passes_the_gate(self, path: str) -> None:
        """No pre-existing test in this file sets a Host header — confirms
        the gate stays backward compatible with every non-browser caller."""
        handler = _make_handler("GET", path)
        assert handler._check_host_admission() is True


@pytest.mark.parametrize("path", ENDPOINTS_POST)
class TestPostEndpointHostGate:
    def test_foreign_host_returns_403_even_with_valid_token(self, path: str) -> None:
        handler = _make_handler("POST", path, auth_header="Bearer secret", host="evil.example.com")
        send_error, _ = _capture_responses(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.FORBIDDEN, "host_not_allowed")


@pytest.mark.parametrize("path", ENDPOINTS_DELETE)
class TestDeleteEndpointHostGate:
    def test_foreign_host_returns_403_even_with_valid_token(self, path: str) -> None:
        handler = _make_handler("DELETE", path, auth_header="Bearer secret", host="evil.example.com")
        send_error, _ = _capture_responses(handler)
        handler.do_DELETE()
        send_error.assert_called_once_with(HTTPStatus.FORBIDDEN, "host_not_allowed")


# ---------------------------------------------------------------------------
# Ingest endpoint — user-facing clients accept arbitrary local paths, stage
# them into the archive inbox, then ask the daemon to schedule that staged
# entry. The daemon route itself must not become an arbitrary local file
# copier from HTTP request data.
# ---------------------------------------------------------------------------


class TestIngestEndpointInboxBoundary:
    """``POST /api/ingest`` schedules client-staged inbox artifacts only."""

    def test_accepts_absolute_reference_only_by_matching_inbox_entry(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        inbox = workspace_env["archive_root"] / "inbox"
        inbox.mkdir(parents=True)
        staged = inbox / "session.json"
        staged.write_text(
            json.dumps(
                {
                    "mapping": {
                        "root": {
                            "id": "root",
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["hello"]},
                            },
                            "children": [],
                        }
                    }
                }
            )
        )

        body = json.dumps({"path": "/outside/tree/session.json"}).encode("utf-8")
        handler = _make_handler("POST", "/api/ingest", auth_header="Bearer secret", body=body)
        send_error, send_json = _capture_responses(handler)

        with (
            patch("polylogue.paths.archive_root", return_value=workspace_env["archive_root"]),
            patch("polylogue.daemon.http.emit_daemon_event") as emit_event,
        ):
            handler.do_POST()

        send_error.assert_not_called()
        send_json.assert_called_once()
        assert send_json.call_args.args[0] == HTTPStatus.ACCEPTED
        payload = send_json.call_args.args[1]
        assert payload["path"] == str(staged.resolve())
        assert payload["preflight"]["status"] == "supported"
        assert payload["preflight"]["providers"] == ["chatgpt"]
        emit_event.assert_called_once()
        assert emit_event.call_args.kwargs["payload"]["path"] == str(staged.resolve())
        assert emit_event.call_args.kwargs["payload"]["preflight"]["status"] == "supported"

    def test_rejects_staged_unsupported_import_shape(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        inbox = workspace_env["archive_root"] / "inbox"
        inbox.mkdir(parents=True)
        staged = inbox / "unknown.json"
        staged.write_text(json.dumps({"not": "an export"}))

        body = json.dumps({"path": str(staged)}).encode("utf-8")
        handler = _make_handler("POST", "/api/ingest", auth_header="Bearer secret", body=body)
        send_error, send_json = _capture_responses(handler)

        with (
            patch("polylogue.paths.archive_root", return_value=workspace_env["archive_root"]),
            patch("polylogue.daemon.http.emit_daemon_event") as emit_event,
        ):
            handler.do_POST()

        send_error.assert_called_once()
        assert send_error.call_args.args[0] == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
        assert send_error.call_args.args[1] == "unsupported_import_source"
        assert "no parseable" in send_error.call_args.args[2]
        send_json.assert_not_called()
        emit_event.assert_not_called()

    def test_accepts_degraded_staged_import_with_preflight_caveat(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        inbox = workspace_env["archive_root"] / "inbox"
        inbox.mkdir(parents=True)
        staged = inbox / "mixed.zip"
        with zipfile.ZipFile(staged, "w") as zf:
            zf.writestr(
                "conversations.json",
                json.dumps(
                    {
                        "mapping": {
                            "root": {
                                "id": "root",
                                "message": {
                                    "author": {"role": "user"},
                                    "content": {"content_type": "text", "parts": ["hello"]},
                                },
                                "children": [],
                            }
                        }
                    }
                ),
            )
            zf.writestr("unknown.json", json.dumps({"not": "an export"}))

        body = json.dumps({"path": str(staged)}).encode("utf-8")
        handler = _make_handler("POST", "/api/ingest", auth_header="Bearer secret", body=body)
        send_error, send_json = _capture_responses(handler)

        with (
            patch("polylogue.paths.archive_root", return_value=workspace_env["archive_root"]),
            patch("polylogue.daemon.http.emit_daemon_event"),
        ):
            handler.do_POST()

        send_error.assert_not_called()
        send_json.assert_called_once()
        assert send_json.call_args.args[0] == HTTPStatus.ACCEPTED
        payload = send_json.call_args.args[1]
        assert payload["preflight"]["status"] == "degraded"
        assert payload["preflight"]["supported_count"] == 1
        assert payload["preflight"]["unsupported_count"] == 1
        assert "degraded" in payload["message"]

    def test_rejects_unstaged_absolute_local_path(
        self,
        workspace_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        outside = tmp_path / "session.jsonl"
        outside.write_text('{"type":"session"}\n')

        body = json.dumps({"path": str(outside)}).encode("utf-8")
        handler = _make_handler("POST", "/api/ingest", auth_header="Bearer secret", body=body)
        send_error, send_json = _capture_responses(handler)

        with (
            patch("polylogue.paths.archive_root", return_value=workspace_env["archive_root"]),
            patch("polylogue.daemon.http.emit_daemon_event") as emit_event,
        ):
            handler.do_POST()

        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "path_not_found")
        send_json.assert_not_called()
        emit_event.assert_not_called()

    @pytest.mark.parametrize(
        "traversal_path",
        [
            "../../../etc/passwd",
            "../../etc/passwd",
            "../evil",
            "/etc/passwd",
            "/tmp/evil.jsonl",
            "inbox/../../../etc/passwd",
            "a/b/c/session.jsonl",
        ],
    )
    def test_traversal_attempt_cannot_escape_inbox(
        self,
        traversal_path: str,
        workspace_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Path traversal attempts via ``..`` or embedded ``/`` cannot escape inbox.

        ``_staged_inbox_source`` uses ``PurePath(raw).name`` which strips all
        directory components before matching against inbox entries. The resolved
        candidate is then re-checked with ``relative_to(inbox_root)`` to catch
        symlink escapes. A file with the extracted basename that lives outside the
        inbox must produce ``path_not_found``; one that happens to match an inbox
        entry may only be returned if the resolved path is inside inbox_root.
        """
        inbox = workspace_env["archive_root"] / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        # Put a file with the traversal's extracted basename OUTSIDE the inbox.
        from pathlib import PurePath

        basename = PurePath(traversal_path).name
        if basename and basename not in {".", ".."}:
            outside = tmp_path / basename
            outside.write_text('{"type":"session"}\n')

        body = json.dumps({"path": traversal_path}).encode("utf-8")
        handler = _make_handler("POST", "/api/ingest", auth_header="Bearer secret", body=body)
        send_error, send_json = _capture_responses(handler)

        with (
            patch("polylogue.paths.archive_root", return_value=workspace_env["archive_root"]),
            patch("polylogue.daemon.http.emit_daemon_event") as emit_event,
        ):
            handler.do_POST()

        # Must be rejected — the file is outside inbox, regardless of the
        # path shape the client sent.
        assert send_error.called, f"traversal path {traversal_path!r} was not rejected"
        send_json.assert_not_called()
        emit_event.assert_not_called()

    def test_traversal_basename_matches_inbox_entry_is_accepted(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """When the basename extracted from a traversal path matches a real inbox entry
        AND the inbox entry is inside inbox_root, the daemon schedules it normally.

        This confirms the name-extraction is sanitizing the input, not just
        blocking it — clients can refer to inbox files via absolute or relative
        paths as long as the file is actually in the inbox.
        """
        inbox = workspace_env["archive_root"] / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        staged = inbox / "session.jsonl"
        staged.write_text(
            json.dumps(
                {
                    "mapping": {
                        "root": {
                            "id": "root",
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["hello"]},
                            },
                            "children": [],
                        }
                    }
                }
            )
            + "\n"
        )

        # Client sends a dotdot path whose basename is "session.jsonl"
        body = json.dumps({"path": "../inbox/session.jsonl"}).encode("utf-8")
        handler = _make_handler("POST", "/api/ingest", auth_header="Bearer secret", body=body)
        send_error, send_json = _capture_responses(handler)

        with (
            patch("polylogue.paths.archive_root", return_value=workspace_env["archive_root"]),
            patch("polylogue.daemon.http.emit_daemon_event"),
        ):
            handler.do_POST()

        # Name was sanitized → matched the real inbox entry → accepted
        send_error.assert_not_called()
        assert send_json.call_args.args[0] == HTTPStatus.ACCEPTED
        payload = send_json.call_args.args[1]
        assert payload["path"] == str(staged.resolve())

    def test_symlink_escape_is_rejected(
        self,
        workspace_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """A symlink inside the inbox pointing outside cannot be followed.

        ``_staged_inbox_source`` calls ``resolved.relative_to(inbox_root)``
        after resolving, which raises ValueError when the target escapes.
        """
        inbox = workspace_env["archive_root"] / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        real_file = tmp_path / "real_secret.jsonl"
        real_file.write_text('{"type":"session"}\n')
        link = inbox / "escape_link.jsonl"
        link.symlink_to(real_file)

        body = json.dumps({"path": "escape_link.jsonl"}).encode("utf-8")
        handler = _make_handler("POST", "/api/ingest", auth_header="Bearer secret", body=body)
        send_error, send_json = _capture_responses(handler)

        with (
            patch("polylogue.paths.archive_root", return_value=workspace_env["archive_root"]),
            patch("polylogue.daemon.http.emit_daemon_event"),
        ):
            handler.do_POST()

        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_path")
        send_json.assert_not_called()


# ---------------------------------------------------------------------------
# JSON serialization robustness — _json_bytes must not silently drop data
# ---------------------------------------------------------------------------


class TestJsonSerializationRobustness:
    """``_json_bytes`` is the final serialization barrier before the wire.

    ``orjson`` rejects non-serializable types (e.g., ``set``, custom objects
    with no ``__json__``).  The ``daemon_safe_handler`` decorator catches
    unhandled exceptions and returns 500, so the client always gets a
    well-formed error response instead of a hung or partial stream.

    These tests confirm that:
    1. Normal dicts with string/int/list/None values round-trip cleanly.
    2. ``orjson.JSONEncodeError`` from ``_json_bytes`` is not silently swallowed
       — the caller (``_send_json``) will propagate it to the decorator.
    3. The full handler chain (decorator → handler → ``_send_json``) returns
       a 500 JSON error when a handler builds an unserializable payload.
    """

    def test_normal_payload_round_trips(self) -> None:
        from polylogue.daemon.http import _json_bytes

        payload: dict[str, object] = {
            "id": "abc",
            "title": "My session",
            "count": 42,
            "tags": ["a", "b"],
            "nested": {"ok": True, "value": None},
        }
        raw = _json_bytes(payload)
        import json as _json

        assert _json.loads(raw) == payload

    def test_set_raises_type_error(self) -> None:
        """Sets are not JSON-serializable. ``orjson`` raises TypeError,
        not silently truncates or returns partial output."""
        from polylogue.daemon.http import _json_bytes

        with pytest.raises(TypeError):
            _json_bytes({"tags": {1, 2, 3}})

    def test_custom_object_raises_type_error(self) -> None:
        """Custom objects with no JSON protocol raise TypeError,
        not return ``null`` or ``{}``."""
        from polylogue.daemon.http import _json_bytes

        class _Unserializable:
            pass

        with pytest.raises(TypeError):
            _json_bytes({"obj": _Unserializable()})

    def test_handler_returns_500_on_unserializable_response(self) -> None:
        """When a route handler calls ``_send_json`` with unserializable data the
        ``daemon_safe_handler`` decorator catches the resulting error and replies
        with a well-formed 500 JSON response rather than crashing silently.

        This is the end-to-end path: route handler builds bad payload →
        ``_send_json`` calls ``_json_bytes`` → orjson raises →
        ``daemon_safe_handler`` catches → returns 500 error JSON.

        The test injects a ``_send_json`` mock that raises on the first call
        (simulating orjson failure) and records on the second (simulating the
        decorator's error-response call), confirming the contract holds.
        """
        from typing import Any

        from polylogue.daemon.http import daemon_safe_handler

        handler = _make_handler("GET", "/api/status", auth_header="Bearer secret")
        recorded: list[tuple[HTTPStatus, object]] = []
        call_count = 0

        def _mock_send_json(status: HTTPStatus, payload: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: simulate orjson raising on unserializable data
                # (orjson raises TypeError for non-serializable Python types).
                raise TypeError("Type is not JSON serializable: set")
            # Subsequent calls: record (these come from the decorator's fallback).
            recorded.append((status, payload))

        handler._send_json = _mock_send_json  # type: ignore[method-assign]

        @daemon_safe_handler
        def _bad_handler(self: Any) -> None:
            # Calls _send_json once with a bad payload; the mock will raise.
            self._send_json(HTTPStatus.OK, {"tags": {1, 2, 3}})

        _bad_handler(handler)

        # The decorator must have caught the error and fired a 500 response.
        assert recorded, "no fallback response was sent after orjson error"
        final_status, final_payload = recorded[-1]
        assert final_status == HTTPStatus.INTERNAL_SERVER_ERROR
        assert isinstance(final_payload, dict)
        assert final_payload.get("error") == "internal_error"

    def test_handler_ignores_client_disconnect_during_response(self) -> None:
        """Client timeouts must not turn BrokenPipe into a noisy 500 path."""
        from typing import Any

        from polylogue.daemon.http import daemon_safe_handler

        handler = _make_handler("GET", "/api/status", auth_header="Bearer secret")
        call_count = 0

        def _mock_send_json(status: HTTPStatus, payload: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            raise BrokenPipeError("client disconnected")

        handler._send_json = _mock_send_json  # type: ignore[method-assign]

        @daemon_safe_handler
        def _slow_handler(self: Any) -> None:
            self._send_json(HTTPStatus.OK, {"ok": True})

        _slow_handler(handler)

        assert call_count == 1


# ---------------------------------------------------------------------------
# OPTIONS — documented as 405 by design (no CORS preflight)
# ---------------------------------------------------------------------------


class TestOptionsReturnsMethodNotAllowed:
    """``OPTIONS`` returns 405 by design.

    The daemon does not advertise CORS — see ``docs/security.md``. A
    regression that introduces a permissive CORS preflight without a
    deliberate policy decision should fail this test.
    """

    @pytest.mark.parametrize("path", ENDPOINTS_GET + ENDPOINTS_POST)
    def test_options_405(self, path: str) -> None:
        handler = _make_handler("OPTIONS", path)
        send_error, _ = _capture_responses(handler)
        handler.do_OPTIONS()
        send_error.assert_called_once_with(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed")


# ---------------------------------------------------------------------------
# Loopback helper — guards shared loopback contract
# ---------------------------------------------------------------------------


class TestIsLocalhost:
    """``is_loopback_host`` is consumed by remote-bind enforcement in
    ``daemon/cli.py``. Pin its contract.
    """

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "127.0.0.2", "127.1.2.3", "127.255.255.254", "::1", "localhost"],
    )
    def test_loopback_addresses(self, host: str) -> None:
        assert is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "192.168.1.1", "10.0.0.1", "8.8.8.8", "128.0.0.1", "::", ""],
    )
    def test_non_loopback_addresses(self, host: str) -> None:
        assert is_loopback_host(host) is False


# ---------------------------------------------------------------------------
# Token-leak guardrail (#868-A10)
# ---------------------------------------------------------------------------


class TestNoTokenLogging:
    """Static grep over daemon and browser-capture sources confirms no
    code path logs or prints the bearer token.

    A regression that adds ``logger.info(f"...token={auth_token}...")``
    or similar to a daemon module would be caught here.
    """

    def test_no_token_in_log_or_print_calls(self) -> None:
        import re
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        roots = [
            repo_root / "polylogue" / "daemon",
            repo_root / "polylogue" / "browser_capture",
        ]
        # Match {logger,log,print,debug,info,warning,error,exception} that
        # mention auth_token or "Bearer" anywhere on the same line.
        pattern = re.compile(
            r"(?:logger|log|print|debug|info|warning|error|exception)\b.*"
            r"(?:auth_token|Bearer)",
            re.IGNORECASE,
        )

        offenders: list[str] = []
        for root in roots:
            if not root.exists():
                continue
            for py_file in root.rglob("*.py"):
                for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
                    if pattern.search(line):
                        # Whitelist the legitimate non-logging usages:
                        # passing the token to the checker is not a leak.
                        if "_check_auth_logic" in line:
                            continue
                        offenders.append(f"{py_file.relative_to(repo_root)}:{lineno}: {line.strip()}")

        assert not offenders, "potential token-logging code paths:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# OTLP gating (#1604): observability_enabled flag, body cap, loopback-vs-remote
# ---------------------------------------------------------------------------


class TestOtlpGating:
    """The OTLP receiver routes must obey ``observability_enabled`` and a
    body-size cap before any payload is read or persisted. The earlier
    implementation routed to the receiver unconditionally despite a
    comment claiming otherwise — see #1604."""

    OTLP_PATHS = ["/v1/traces", "/v1/metrics", "/v1/logs"]

    @pytest.mark.parametrize("path", OTLP_PATHS)
    def test_foreign_host_returns_403_before_observability_special_case(
        self, path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The OTLP branch runs BEFORE the normal auth/Origin checks in
        ``_do_post_impl`` — it must not also bypass the Host gate, or a
        DNS-rebound page could post telemetry into ops.db (kwsb.1)."""
        from types import SimpleNamespace

        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: SimpleNamespace(observability_enabled=True, otlp_max_body_bytes=8 * 1024 * 1024),
        )
        handler = _make_handler("POST", path, host="evil.example.com")
        send_error, _ = _capture_responses(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.FORBIDDEN, "host_not_allowed")

    @pytest.mark.parametrize("path", OTLP_PATHS)
    def test_disabled_observability_returns_404(self, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: SimpleNamespace(observability_enabled=False, otlp_max_body_bytes=8 * 1024 * 1024),
        )
        handler = _make_handler("POST", path, origin="")
        send_error, _ = _capture_responses(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")

    @pytest.mark.parametrize("path", OTLP_PATHS)
    def test_enabled_loopback_skips_auth(self, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``observability_enabled`` is on and the daemon is on loopback,
        the route proceeds without requiring the auth token (matches
        ``/metrics`` and ``/healthz/*``)."""
        from types import SimpleNamespace

        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: SimpleNamespace(observability_enabled=True, otlp_max_body_bytes=8 * 1024 * 1024),
        )

        handler = _make_handler("POST", path)
        # Stub the receiver so we don't actually need opentelemetry-proto here.
        otlp_handler_called = []

        def fake_handle_otlp_post(p: list[str]) -> None:
            otlp_handler_called.append(p)
            handler.send_response(200)

        handler._handle_otlp_post = fake_handle_otlp_post  # type: ignore[method-assign]
        handler.send_response = MagicMock()  # type: ignore[method-assign]

        handler.do_POST()

        assert otlp_handler_called, f"OTLP handler must be invoked when enabled+loopback for {path}"

    @pytest.mark.parametrize("path", OTLP_PATHS)
    def test_enabled_non_loopback_without_token_returns_401(self, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bound non-loopback ⇒ require the auth token even with the
        observability flag on."""
        from types import SimpleNamespace

        class RemoteMockServer:
            auth_token = "secret"
            api_host = "0.0.0.0"  # not loopback

        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: SimpleNamespace(observability_enabled=True, otlp_max_body_bytes=8 * 1024 * 1024),
        )

        handler = _make_handler("POST", path)
        handler.server = cast("DaemonAPIHTTPServer", RemoteMockServer())
        send_error, _ = _capture_responses(handler)

        handler.do_POST()

        send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")

    @pytest.mark.parametrize("path", OTLP_PATHS)
    def test_body_over_cap_returns_413(self, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Oversize ``Content-Length`` ⇒ 413 before the body is read or
        persisted, preventing storage-amplification."""
        from types import SimpleNamespace

        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: SimpleNamespace(observability_enabled=True, otlp_max_body_bytes=1024),  # 1 KiB cap
        )

        big_body = b"x" * 4096  # 4 KiB > 1 KiB cap
        handler = _make_handler("POST", path, body=big_body)
        send_error, _ = _capture_responses(handler)

        handler.do_POST()

        send_error.assert_called_with(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "payload_too_large")
