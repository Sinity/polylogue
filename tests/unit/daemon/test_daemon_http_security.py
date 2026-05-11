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

The parametrized matrix replaces ad-hoc per-endpoint tests so the cost
of adding a new endpoint includes adding it to ``ENDPOINTS_GET`` /
``ENDPOINTS_POST`` and getting full security coverage automatically.
"""

from __future__ import annotations

from email.message import Message
from http import HTTPStatus
from io import BytesIO
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.daemon.http import _check_auth_logic, _is_localhost

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Endpoint tables — extend here when new routes are added
# ---------------------------------------------------------------------------

ENDPOINTS_GET = [
    "/api/health/check",
    "/api/health",
    "/api/status",
    "/api/conversations",
    "/api/facets",
    "/api/sources",
    "/api/conversations/some-id",
    "/api/conversations/some-id/raw",
    "/api/conversations/some-id/messages",
    "/api/raw_artifacts/some-hash",
]

ENDPOINTS_POST = [
    "/api/reset",
    "/api/ingest",
    "/api/maintenance/plan",
    "/api/maintenance/run",
]


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
    as a loopback bind address (see ``_is_localhost``); a web shell
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
    body: bytes = b"",
) -> DaemonAPIHandler:
    """Build a ``DaemonAPIHandler`` instance with mocked transport.

    Returns a handler ready to receive a ``do_GET`` / ``do_POST`` call.
    The route handlers are not invoked unless ``_check_auth`` and
    ``_check_cross_origin`` admit the request.
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
        """Auth gate admits the request — handler runs (or the route is
        not found, in which case 404, but never 401)."""
        handler = _make_handler("GET", path, auth_header="Bearer secret")
        send_error, send_json = _capture_responses(handler)
        # Patch the actual handlers to avoid touching the DB / archive.
        with (
            patch.object(handler, "_handle_health_check"),
            patch.object(handler, "_handle_health"),
            patch.object(handler, "_handle_status"),
            patch.object(handler, "_handle_list_conversations"),
            patch.object(handler, "_handle_facets"),
            patch.object(handler, "_handle_sources"),
            patch.object(handler, "_handle_get_conversation"),
            patch.object(handler, "_handle_get_messages"),
            patch.object(handler, "_handle_get_conversation_raw"),
            patch.object(handler, "_handle_get_raw_artifact"),
        ):
            handler.do_GET()
        # If 401 was sent, auth gate failed. We assert the negative.
        for call in send_error.call_args_list:
            assert call.args[0] != HTTPStatus.UNAUTHORIZED, f"Endpoint {path} returned 401 with valid token"


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
# Loopback helper — guards _is_localhost contract
# ---------------------------------------------------------------------------


class TestIsLocalhost:
    """``_is_localhost`` is consumed by remote-bind enforcement in
    ``daemon/cli.py``. Pin its contract.
    """

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "127.0.0.2", "127.1.2.3", "127.255.255.254", "::1", "localhost"],
    )
    def test_loopback_addresses(self, host: str) -> None:
        assert _is_localhost(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "192.168.1.1", "10.0.0.1", "8.8.8.8", "128.0.0.1", "::", ""],
    )
    def test_non_loopback_addresses(self, host: str) -> None:
        assert _is_localhost(host) is False


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
