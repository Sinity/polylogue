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
from polylogue.lib.json import dumps_bytes


def _json_bytes(payload: object) -> bytes:
    return dumps_bytes(payload, option=None)


def _origin_allowed(origin: str | None, config: BrowserCaptureReceiverConfig) -> bool:
    if origin is None:
        return True
    if origin.startswith("chrome-extension://"):
        return True
    return origin in config.allowed_origins


class BrowserCaptureHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying receiver configuration."""

    config: BrowserCaptureReceiverConfig

    def __init__(self, server_address: tuple[str, int], config: BrowserCaptureReceiverConfig) -> None:
        self.config = config
        super().__init__(server_address, BrowserCaptureHandler)


class BrowserCaptureHandler(BaseHTTPRequestHandler):
    """Local JSON API used by the browser extension."""

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

    def _reject_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if _origin_allowed(origin, self.server.config):
            return False
        self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "origin_not_allowed"})
        return True

    def do_OPTIONS(self) -> None:
        if self._reject_origin():
            return
        origin = self.headers.get("Origin")
        self.send_response(HTTPStatus.NO_CONTENT.value)
        self.send_header("Access-Control-Allow-Origin", origin or "null")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        if self._reject_origin():
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
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "missing_provider_or_session"},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                existing_capture_state(provider, session_id, inbox_path=self.server.config.inbox_path),
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        if self._reject_origin():
            return
        if urlparse(self.path).path != "/v1/browser-captures":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_content_length"})
            return
        if length <= 0 or length > 10 * 1024 * 1024:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_body_size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            envelope = BrowserCaptureEnvelope.model_validate(payload)
            result = write_capture_envelope(envelope, inbox_path=self.server.config.inbox_path)
        except (json.JSONDecodeError, ValidationError, OSError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": type(exc).__name__, "detail": str(exc)})
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


def make_server(host: str, port: int, *, inbox_path: Path | None = None) -> BrowserCaptureHTTPServer:
    """Create a configured browser-capture receiver server."""
    config = BrowserCaptureReceiverConfig.default()
    if inbox_path is not None:
        config = BrowserCaptureReceiverConfig(inbox_path=inbox_path, allowed_origins=config.allowed_origins)
    return BrowserCaptureHTTPServer((host, port), config)


__all__ = ["BrowserCaptureHTTPServer", "BrowserCaptureHandler", "make_server"]
