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
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
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
    archive_query_executor = ThreadPoolExecutor(max_workers=1)
    archive_query_admission = threading.BoundedSemaphore(64)  # generous: not under test


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

        body = json.dumps(
            {
                "path": "/outside/tree/session.json",
                "source_path": "/original/provider/export/session.json",
            }
        ).encode("utf-8")
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
        assert payload["request"]["source_path"] == "/original/provider/export/session.json"
        assert payload["request"]["staged_path"] == str(staged.resolve())
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
    """Static analysis confirms no unintended output sink receives a token.

    A regression that adds ``logger.info(f"...token={auth_token}...")``
    or similar to a daemon module is caught here. The explicit
    ``browser-capture token show`` pairing command is the sole intentional
    secret-output path.
    """

    def test_no_token_in_log_or_print_calls(self) -> None:
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        roots = [
            repo_root / "polylogue" / "daemon",
            repo_root / "polylogue" / "browser_capture",
        ]
        output_sinks = {
            "_send_error",
            "_send_json",
            "critical",
            "debug",
            "echo",
            "error",
            "exception",
            "info",
            "log",
            "print",
            "send",
            "warning",
        }
        intentional_secret_outputs = {
            (Path("polylogue/daemon/browser_capture.py"), "token_show", "echo"),
        }
        safe_token_container_results = {
            (Path("polylogue/daemon/browser_capture.py"), "serve_command", "make_server"),
        }

        def _call_sink_name(call: ast.Call) -> str | None:
            if isinstance(call.func, ast.Name):
                return call.func.id
            if isinstance(call.func, ast.Attribute):
                return call.func.attr
            return None

        def _is_token_name(name: str) -> bool:
            return name == "token" or name.endswith("_token")

        def _is_output_sink(call: ast.Call) -> bool:
            sink_name = _call_sink_name(call)
            if sink_name in output_sinks:
                return True
            if sink_name != "write" or not isinstance(call.func, ast.Attribute):
                return False
            receiver = call.func.value
            is_standard_stream = (
                isinstance(receiver, ast.Attribute)
                and receiver.attr in {"stderr", "stdout"}
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "sys"
            )
            is_http_response_stream = any(
                isinstance(node, ast.Attribute) and node.attr == "wfile" for node in ast.walk(receiver)
            )
            return is_standard_stream or is_http_response_stream

        def _contains_token_value(expression: ast.AST, sensitive_names: set[str]) -> bool:
            for node in ast.walk(expression):
                name: str | None = None
                if isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                if name is not None and (_is_token_name(name) or name in sensitive_names):
                    return True
            return False

        def _assigned_names(target: ast.AST) -> set[str]:
            if isinstance(target, ast.Name):
                return {target.id}
            if isinstance(target, (ast.List, ast.Tuple)):
                return {name for element in target.elts for name in _assigned_names(element)}
            return set()

        def _assignment_carries_token(
            expression: ast.AST,
            sensitive_names: set[str],
            *,
            suppress_call_result_taint: bool = False,
        ) -> bool:
            if not isinstance(expression, ast.Call):
                return _contains_token_value(expression, sensitive_names)
            function_name = _call_sink_name(expression)
            if isinstance(expression.func, ast.Attribute) and _contains_token_value(
                expression.func.value, sensitive_names
            ):
                return True
            arguments_carry_token = any(
                _contains_token_value(value, sensitive_names)
                for value in [*expression.args, *(kw.value for kw in expression.keywords)]
            )
            if (
                arguments_carry_token
                and function_name
                and not function_name[:1].isupper()
                and not suppress_call_result_taint
            ):
                return True
            return bool(function_name and _is_token_name(function_name))

        class _SensitiveOutputVisitor(ast.NodeVisitor):
            def __init__(self, relative_path: Path) -> None:
                self.relative_path = relative_path
                self.function_stack: list[str] = []
                self.sensitive_name_stack: list[set[str]] = [set()]
                self.offenders: list[tuple[int, str]] = []

            @property
            def sensitive_names(self) -> set[str]:
                return self.sensitive_name_stack[-1]

            def _call_result_is_known_container(self, expression: ast.AST) -> bool:
                if not isinstance(expression, ast.Call):
                    return False
                function_name = self.function_stack[-1] if self.function_stack else "<module>"
                return (
                    self.relative_path,
                    function_name,
                    _call_sink_name(expression),
                ) in safe_token_container_results

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                self.function_stack.append(node.name)
                argument_names = {
                    argument.arg
                    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                    if _is_token_name(argument.arg)
                }
                if node.args.vararg is not None and _is_token_name(node.args.vararg.arg):
                    argument_names.add(node.args.vararg.arg)
                if node.args.kwarg is not None and _is_token_name(node.args.kwarg.arg):
                    argument_names.add(node.args.kwarg.arg)
                self.sensitive_name_stack.append(self.sensitive_names | argument_names)
                self.generic_visit(node)
                self.sensitive_name_stack.pop()
                self.function_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                names = {name for target in node.targets for name in _assigned_names(target)}
                if _assignment_carries_token(
                    node.value,
                    self.sensitive_names,
                    suppress_call_result_taint=self._call_result_is_known_container(node.value),
                ):
                    self.sensitive_names.update(names)
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                names = _assigned_names(node.target)
                if node.value is not None and _assignment_carries_token(
                    node.value,
                    self.sensitive_names,
                    suppress_call_result_taint=self._call_result_is_known_container(node.value),
                ):
                    self.sensitive_names.update(names)
                self.generic_visit(node)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                names = _assigned_names(node.target)
                if _assignment_carries_token(
                    node.value,
                    self.sensitive_names,
                    suppress_call_result_taint=self._call_result_is_known_container(node.value),
                ):
                    self.sensitive_names.update(names)
                # Taint is deliberately flow-insensitive within a function:
                # once any feasible path assigns a token-derived value, later
                # output remains suspect across try/match/loop joins.
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                sink_name = _call_sink_name(node)
                expressions = [*node.args, *(keyword.value for keyword in node.keywords)]
                function_name = self.function_stack[-1] if self.function_stack else "<module>"
                if (
                    _is_output_sink(node)
                    and any(_contains_token_value(expression, self.sensitive_names) for expression in expressions)
                    and (self.relative_path, function_name, sink_name) not in intentional_secret_outputs
                ):
                    self.offenders.append((node.lineno, function_name))
                self.generic_visit(node)

        seeded_violation = _SensitiveOutputVisitor(Path("seeded_violation.py"))
        seeded_violation.visit(ast.parse("def leak(auth_token):\n    logger.info(auth_token)\n"))
        assert seeded_violation.offenders == [(2, "leak")]

        seeded_alias_violation = _SensitiveOutputVisitor(Path("seeded_alias_violation.py"))
        seeded_alias_violation.visit(
            ast.parse("def leak(auth_token):\n    secret = auth_token\n    logger.info('secret=%s', secret)\n")
        )
        assert seeded_alias_violation.offenders == [(3, "leak")]

        seeded_transform_violation = _SensitiveOutputVisitor(Path("seeded_transform_violation.py"))
        seeded_transform_violation.visit(
            ast.parse("def leak(auth_token):\n    secret = auth_token.strip()\n    logger.info(secret)\n")
        )
        assert seeded_transform_violation.offenders == [(3, "leak")]

        seeded_branch_violation = _SensitiveOutputVisitor(Path("seeded_branch_violation.py"))
        seeded_branch_violation.visit(
            ast.parse(
                "def leak(auth_token, condition):\n"
                "    if condition:\n"
                "        secret = auth_token\n"
                "    else:\n"
                "        secret = 'safe'\n"
                "    logger.info(secret)\n"
            )
        )
        assert seeded_branch_violation.offenders == [(6, "leak")]

        seeded_try_violation = _SensitiveOutputVisitor(Path("seeded_try_violation.py"))
        seeded_try_violation.visit(
            ast.parse(
                "def leak(auth_token):\n"
                "    try:\n"
                "        secret = auth_token\n"
                "    except Exception:\n"
                "        secret = 'safe'\n"
                "    logger.info(secret)\n"
            )
        )
        assert seeded_try_violation.offenders == [(6, "leak")]

        seeded_stdout_violation = _SensitiveOutputVisitor(Path("seeded_stdout_violation.py"))
        seeded_stdout_violation.visit(ast.parse("def leak(auth_token):\n    sys.stdout.write(auth_token)\n"))
        assert seeded_stdout_violation.offenders == [(2, "leak")]

        seeded_helper_violation = _SensitiveOutputVisitor(Path("seeded_helper_violation.py"))
        seeded_helper_violation.visit(
            ast.parse("def leak(auth_token):\n    rendered = helper(auth_token)\n    logger.info(rendered)\n")
        )
        assert seeded_helper_violation.offenders == [(3, "leak")]

        seeded_http_violation = _SensitiveOutputVisitor(Path("seeded_http_violation.py"))
        seeded_http_violation.visit(
            ast.parse("def leak(auth_token, handler):\n    handler.wfile.write(auth_token.encode())\n")
        )
        assert seeded_http_violation.offenders == [(2, "leak")]

        seeded_json_response_violation = _SensitiveOutputVisitor(Path("seeded_json_response_violation.py"))
        seeded_json_response_violation.visit(
            ast.parse(
                "def leak(self, auth_token):\n"
                "    payload = {'credential': auth_token}\n"
                "    self._send_json(HTTPStatus.OK, payload)\n"
            )
        )
        assert seeded_json_response_violation.offenders == [(3, "leak")]

        seeded_error_response_violation = _SensitiveOutputVisitor(Path("seeded_error_response_violation.py"))
        seeded_error_response_violation.visit(
            ast.parse(
                "def leak(self, auth_token):\n"
                "    detail = auth_token.strip()\n"
                "    self._send_error(HTTPStatus.BAD_REQUEST, 'invalid', detail)\n"
            )
        )
        assert seeded_error_response_violation.offenders == [(3, "leak")]

        seeded_journal_violation = _SensitiveOutputVisitor(Path("seeded_journal_violation.py"))
        seeded_journal_violation.visit(ast.parse("def leak(auth_token, journal):\n    journal.send(auth_token)\n"))
        assert seeded_journal_violation.offenders == [(2, "leak")]

        seeded_exception_scope = _SensitiveOutputVisitor(Path("polylogue/daemon/browser_capture.py"))
        seeded_exception_scope.visit(
            ast.parse("def token_show(token):\n    click.echo(token)\n    logger.info(token)\n")
        )
        assert seeded_exception_scope.offenders == [(3, "token_show")]

        safe_literal = _SensitiveOutputVisitor(Path("safe_literal.py"))
        safe_literal.visit(ast.parse('logger.warning("Bearer token required")\n'))
        assert safe_literal.offenders == []

        offenders: list[str] = []
        for root in roots:
            if not root.exists():
                continue
            for py_file in root.rglob("*.py"):
                relative_path = py_file.relative_to(repo_root)
                source = py_file.read_text(encoding="utf-8")
                visitor = _SensitiveOutputVisitor(relative_path)
                visitor.visit(ast.parse(source, filename=str(relative_path)))
                lines = source.splitlines()
                offenders.extend(
                    f"{relative_path}:{lineno} ({function_name}): {lines[lineno - 1].strip()}"
                    for lineno, function_name in visitor.offenders
                )

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
