"""HTTP surface for the local browser-capture receiver."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from pydantic import ValidationError

from polylogue.browser_capture.models import (
    BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD,
    BrowserCaptureAcceptedPayload,
    BrowserCaptureEnvelope,
    BrowserCaptureErrorPayload,
)
from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    existing_capture_state,
    receiver_status_payload,
    write_capture_envelope,
)
from polylogue.core.json import dumps_bytes
from polylogue.core.loopback import is_loopback_host
from polylogue.logging import get_logger

logger = get_logger(__name__)

MAX_BROWSER_CAPTURE_BODY_BYTES = 128 * 1024 * 1024


def _json_bytes(payload: object) -> bytes:
    return dumps_bytes(payload, option=None)


def _origin_allowed(origin: str | None, config: BrowserCaptureReceiverConfig) -> bool:
    if origin is None:
        return True
    if origin in config.allowed_origins:
        return True
    return (
        origin.startswith("chrome-extension://") and BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD in config.allowed_origins
    )


def _is_loopback(host: str) -> bool:
    return is_loopback_host(host)


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
    _polylogue_request_id: str
    _polylogue_status: int | None

    def log_message(self, format: str, *args: object) -> None:
        return

    def _request_id(self) -> str:
        existing = getattr(self, "_polylogue_request_id", None)
        if isinstance(existing, str):
            return existing
        header = self.headers.get("X-Request-ID", "").strip()
        request_id = "".join(ch for ch in header if ch.isalnum() or ch in "-_")[:80] if header else uuid4().hex[:16]
        if not request_id:
            request_id = uuid4().hex[:16]
        self._polylogue_request_id = request_id
        return request_id

    def send_response(self, code: int, message: str | None = None) -> None:
        self._polylogue_status = code
        super().send_response(code, message)

    def _finish_observed_request(self, method: str, started_at: float) -> None:
        logger.info(
            "browser_capture.request",
            request_id=self._request_id(),
            method=method,
            path=urlparse(self.path).path,
            status=getattr(self, "_polylogue_status", None),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            origin=self.headers.get("Origin"),
        )

    def _observe_request(self, method: str, fn: Callable[[], None]) -> None:
        started_at = time.perf_counter()
        try:
            fn()
        finally:
            self._finish_observed_request(method, started_at)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        raw = _json_bytes(payload)
        origin = self.headers.get("Origin")
        self.send_response(status.value)
        self.send_header("X-Request-ID", self._request_id())
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if _origin_allowed(origin, self.server.config):
            self.send_header("Access-Control-Allow-Origin", origin or "null")
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(raw)

    def _safe_error(self, status: HTTPStatus, message: str) -> None:
        """Send a safe error response — no absolute paths or stack traces."""
        self._send_json(status, BrowserCaptureErrorPayload(error=message).model_dump(mode="json"))

    def _reject_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if _origin_allowed(origin, self.server.config):
            return False
        logger.warning("browser_capture.origin_rejected", request_id=self._request_id(), origin=origin)
        self._safe_error(HTTPStatus.FORBIDDEN, "origin_not_allowed")
        return True

    def _reject_token(self) -> bool:
        """Reject if auth token is configured and not present."""
        config = self.server.config
        if config.auth_token is None:
            return False
        if _check_token(dict(self.headers), config):
            return False
        logger.warning(
            "browser_capture.token_rejected", request_id=self._request_id(), origin=self.headers.get("Origin")
        )
        self._safe_error(HTTPStatus.UNAUTHORIZED, "unauthorized")
        return True

    def do_OPTIONS(self) -> None:
        self._observe_request("OPTIONS", self._do_options)

    def _do_options(self) -> None:
        # Browser CORS preflights cannot carry Authorization.  Guard origin here
        # and enforce the bearer token on the actual data request.
        if self._reject_origin():
            return
        origin = self.headers.get("Origin")
        self.send_response(HTTPStatus.NO_CONTENT.value)
        self.send_header("X-Request-ID", self._request_id())
        self.send_header("Access-Control-Allow-Origin", origin or "null")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        self._observe_request("GET", self._do_get)

    def _do_get(self) -> None:
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
        self._observe_request("POST", self._do_post)

    def _do_post(self) -> None:
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
        if length <= 0 or length > MAX_BROWSER_CAPTURE_BODY_BYTES:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_body_size")
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            logger.warning("browser_capture.invalid_json", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_json")
            return
        try:
            envelope = BrowserCaptureEnvelope.model_validate(payload)
        except ValidationError:
            logger.warning("browser_capture.invalid_payload", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_payload")
            return
        try:
            result = write_capture_envelope(envelope, spool_path=self.server.config.spool_path)
        except OSError as exc:
            logger.warning("browser_capture.write_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        logger.info(
            "browser_capture.capture_accepted",
            request_id=self._request_id(),
            provider=result.provider,
            provider_session_id=result.provider_session_id,
            artifact_ref=result.artifact_ref,
            bytes_written=result.bytes_written,
            replaced=result.replaced,
        )
        self._send_json(
            HTTPStatus.ACCEPTED,
            BrowserCaptureAcceptedPayload(
                capture_id=envelope.capture_id or f"{result.provider}:{result.provider_session_id}",
                provider=result.provider,
                provider_session_id=result.provider_session_id,
                artifact_ref=result.artifact_ref,
                bytes_written=result.bytes_written,
                replaced=result.replaced,
            ).model_dump(mode="json"),
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
