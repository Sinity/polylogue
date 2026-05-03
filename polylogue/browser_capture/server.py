"""HTTP surface for the local browser-capture receiver."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    existing_capture_state,
    receiver_status_payload,
    write_capture_envelope,
)
from polylogue.core.json import dumps_bytes

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})


def _json_bytes(payload: object) -> bytes:
    return dumps_bytes(payload, option=None)


def _origin_allowed(origin: str | None, config: BrowserCaptureReceiverConfig) -> bool:
    if origin is None:
        return True
    if origin.startswith("chrome-extension://"):
        return True
    return origin in config.allowed_origins


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.") or host == "::1"


def _check_token(headers: dict[str, str], config: BrowserCaptureReceiverConfig) -> bool:
    """Validate Authorization: Bearer <token> when auth is configured."""
    if config.auth_token is None:
        return True
    auth = headers.get("Authorization", "")
    return bool(auth.startswith("Bearer ") and auth[7:] == config.auth_token)


class BrowserCaptureHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying receiver configuration."""

    config: BrowserCaptureReceiverConfig

    def __init__(self, server_address: tuple[str, int], config: BrowserCaptureReceiverConfig) -> None:
        self.config = config
        super().__init__(server_address, BrowserCaptureHandler)


class BrowserCaptureHandler(BaseHTTPRequestHandler):
    """Local JSON API used by the browser extension.

    Trust boundary: role inference is the parser's responsibility, not the
    receiver's. The receiver accepts the role field from the extension payload
    as-is and writes it to the spool without reinterpretation. The parser
    (``polylogue.sources.parsers``) may apply heuristics or DOM-based inference
    to produce a canonical role. No positional-index role fallback is performed
    here — only explicit attributes from the extension payload are preserved.
    """

    server: BrowserCaptureHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        raw = _json_bytes(payload)
        origin = self.headers.get("Origin")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if _origin_allowed(origin, self.server.config):
            self.send_header("Access-Control-Allow-Origin", origin or "null")
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(raw)

    def _safe_error(self, status: HTTPStatus, message: str) -> None:
        """Send a safe error response — no absolute paths or stack traces."""
        self._send_json(status, {"ok": False, "error": message})

    def _reject_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if _origin_allowed(origin, self.server.config):
            return False
        self._safe_error(HTTPStatus.FORBIDDEN, "origin_not_allowed")
        return True

    def _reject_token(self) -> bool:
        """Reject if auth token is configured and not present."""
        config = self.server.config
        if config.auth_token is None:
            return False
        if _check_token(dict(self.headers), config):
            return False
        self._safe_error(HTTPStatus.UNAUTHORIZED, "unauthorized")
        return True

    def do_OPTIONS(self) -> None:
        if self._reject_origin() or self._reject_token():
            return
        origin = self.headers.get("Origin")
        self.send_response(HTTPStatus.NO_CONTENT.value)
        self.send_header("Access-Control-Allow-Origin", origin or "null")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        if self._reject_origin() or self._reject_token():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/v1/status":
            self._send_json(HTTPStatus.OK, receiver_status_payload(self.server.config))
            return
        if parsed.path == "/v1/archive-state":
            params = parse_qs(parsed.query)
            provider = params.get("provider", [""])[0]
            session_id = params.get("provider_session_id", [""])[0]
            if not provider or not session_id:
                self._safe_error(HTTPStatus.BAD_REQUEST, "missing_provider_or_session")
                return
            self._send_json(
                HTTPStatus.OK,
                existing_capture_state(provider, session_id, spool_path=self.server.config.spool_path),
            )
            return
        self._safe_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_POST(self) -> None:
        if self._reject_origin() or self._reject_token():
            return
        if urlparse(self.path).path != "/v1/browser-captures":
            self._safe_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_content_length")
            return
        if length <= 0 or length > 10 * 1024 * 1024:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_body_size")
            return
        try:
            payload = json.loads(self.rfile.read(length))
            envelope = BrowserCaptureEnvelope.model_validate(payload)
            result = write_capture_envelope(envelope, spool_path=self.server.config.spool_path)
        except (json.JSONDecodeError, ValidationError, OSError):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_payload")
            return
        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "ok": True,
                "provider": result.provider,
                "provider_session_id": result.provider_session_id,
                "artifact_path": str(result.path),
                "bytes_written": result.bytes_written,
                "replaced": result.replaced,
            },
        )


def make_server(
    host: str,
    port: int,
    *,
    spool_path: Path | None = None,
    allow_remote: bool = False,
    auth_token: str | None = None,
    extra_origins: tuple[str, ...] = (),
) -> BrowserCaptureHTTPServer:
    """Create a configured browser-capture receiver server."""
    if not allow_remote and not _is_loopback(host):
        raise ValueError(f"Host {host!r} is not a loopback address. Use --insecure-allow-remote to bind non-loopback.")

    cfg = BrowserCaptureReceiverConfig.default()
    allowed_origins = cfg.allowed_origins | set(extra_origins)
    config = BrowserCaptureReceiverConfig(
        spool_path=spool_path or cfg.spool_path,
        allowed_origins=frozenset(allowed_origins),
        allow_remote=allow_remote,
        auth_token=auth_token,
    )
    config.validate()
    return BrowserCaptureHTTPServer((host, port), config)


__all__ = ["BrowserCaptureHTTPServer", "BrowserCaptureHandler", "make_server"]
