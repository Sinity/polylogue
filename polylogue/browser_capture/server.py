"""HTTP surface for the local browser-capture receiver."""

from __future__ import annotations

import hashlib
import hmac
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
    BrowserCaptureCapabilitiesPayload,
    BrowserCaptureEnvelope,
    BrowserCaptureErrorPayload,
    BrowserPostAckPayload,
    BrowserPostCommandAckRequest,
    BrowserPostCommandListPayload,
    BrowserPostCommandRequest,
    BrowserPostEnqueuedPayload,
)
from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    BrowserPostCommandConflictError,
    BrowserPostCommandStateError,
    BrowserPostDisabledError,
    SpoolQuotaExceededError,
    ack_post_command,
    browser_post_enabled,
    capture_response_id,
    enqueue_post_command,
    existing_capture_state,
    poll_post_commands,
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
    return bool(auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], config.auth_token))


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
        logger.debug(
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
        if parsed.path == "/v1/browser-captures/capabilities":
            self._send_json(HTTPStatus.OK, BrowserCaptureCapabilitiesPayload().model_dump(mode="json"))
            return
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
                existing_capture_state(
                    provider,
                    session_id,
                    spool_path=self.server.config.spool_path,
                    archive_root=self.server.config.archive_root,
                ),
            )
            return
        if parsed.path == "/v1/post-commands":
            params = parse_qs(parsed.query)
            post_provider = params.get("provider", [""])[0] or None
            try:
                commands = poll_post_commands(provider=post_provider, spool_path=self.server.config.spool_path)
            except OSError as exc:
                logger.warning("browser_capture.post_poll_failed", request_id=self._request_id(), error=repr(exc))
                self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
                return
            self._send_json(
                HTTPStatus.OK,
                BrowserPostCommandListPayload(
                    post_enabled=browser_post_enabled(),
                    commands=commands,
                ).model_dump(mode="json", exclude_none=True),
            )
            return
        self._safe_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_POST(self) -> None:
        self._observe_request("POST", self._do_post)

    def _read_json_body(self) -> object | None:
        """Read and parse a JSON request body, sending an error and returning None on failure."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_content_length")
            return None
        if length <= 0 or length > MAX_BROWSER_CAPTURE_BODY_BYTES:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_body_size")
            return None
        raw = self.rfile.read(length)
        try:
            parsed: object = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("browser_capture.invalid_json", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_json")
            return None
        self._request_content_hash = hashlib.sha256(raw).hexdigest()
        return parsed

    def _do_post(self) -> None:
        if self._reject_origin() or self._reject_token():
            return
        path = urlparse(self.path).path
        if path == "/v1/post-commands":
            self._post_command_enqueue()
            return
        if path.startswith("/v1/post-commands/") and path.endswith("/ack"):
            command_id = path[len("/v1/post-commands/") : -len("/ack")]
            self._post_command_ack(command_id)
            return
        if path != "/v1/browser-captures":
            self._safe_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            envelope = BrowserCaptureEnvelope.model_validate(payload)
        except ValidationError:
            logger.warning("browser_capture.invalid_payload", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_payload")
            return
        if envelope.provenance.extension_instance_id is None:
            logger.warning("browser_capture.missing_instance_id", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "missing_extension_instance_id")
            return
        try:
            result = write_capture_envelope(envelope, spool_path=self.server.config.spool_path)
        except SpoolQuotaExceededError as exc:
            logger.warning("browser_capture.spool_quota_exceeded", request_id=self._request_id(), error=str(exc))
            self._safe_error(HTTPStatus.TOO_MANY_REQUESTS, "spool_quota_exceeded")
            return
        except OSError as exc:
            logger.warning("browser_capture.write_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        logger.debug(
            "browser_capture.capture_accepted",
            request_id=self._request_id(),
            provider=result.provider,
            provider_session_id=result.provider_session_id,
            artifact_ref=result.artifact_ref,
            bytes_written=result.bytes_written,
            replaced=result.replaced,
            deduplicated=result.deduplicated,
            capture_instance_id=result.capture_instance_id,
        )
        self._send_json(
            HTTPStatus.ACCEPTED,
            BrowserCaptureAcceptedPayload(
                capture_id=capture_response_id(result.provider, result.provider_session_id, envelope.capture_id),
                provider=result.provider,
                provider_session_id=result.provider_session_id,
                artifact_ref=result.artifact_ref,
                content_hash=self._request_content_hash,
                dedup_content_hash=result.dedup_content_hash,
                bytes_written=result.bytes_written,
                replaced=result.replaced,
                deduplicated=result.deduplicated,
                capture_instance_id=result.capture_instance_id,
            ).model_dump(mode="json"),
        )

    def _post_command_enqueue(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserPostCommandRequest.model_validate(payload)
        except ValidationError:
            logger.warning("browser_capture.invalid_post_command", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_post_command")
            return
        try:
            command = enqueue_post_command(request, spool_path=self.server.config.spool_path)
        except BrowserPostDisabledError:
            logger.warning("browser_capture.post_disabled", request_id=self._request_id())
            self._safe_error(HTTPStatus.FORBIDDEN, "post_disabled")
            return
        except BrowserPostCommandConflictError:
            logger.warning("browser_capture.post_command_conflict", request_id=self._request_id())
            self._safe_error(HTTPStatus.CONFLICT, "duplicate_post_command")
            return
        except SpoolQuotaExceededError as exc:
            logger.warning("browser_capture.post_command_quota_exceeded", request_id=self._request_id(), error=str(exc))
            self._safe_error(HTTPStatus.TOO_MANY_REQUESTS, "post_command_quota_exceeded")
            return
        except OSError as exc:
            logger.warning("browser_capture.post_enqueue_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        logger.debug(
            "browser_capture.post_command_enqueued",
            request_id=self._request_id(),
            command_id=command.command_id,
            provider=command.provider,
            submit=command.submit,
        )
        self._send_json(
            HTTPStatus.ACCEPTED,
            BrowserPostEnqueuedPayload(
                command_id=command.command_id,
                provider=command.provider,
                status=command.status,
                submit=command.submit,
            ).model_dump(mode="json"),
        )

    def _post_command_ack(self, command_id: str) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            ack = BrowserPostCommandAckRequest.model_validate(payload)
        except ValidationError:
            logger.warning("browser_capture.invalid_post_ack", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_post_ack")
            return
        try:
            command = ack_post_command(command_id, ack, spool_path=self.server.config.spool_path)
        except BrowserPostCommandStateError:
            logger.warning("browser_capture.post_ack_invalid_state", request_id=self._request_id())
            self._safe_error(HTTPStatus.CONFLICT, "invalid_post_command_state")
            return
        except OSError as exc:
            logger.warning("browser_capture.post_ack_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        if command is None:
            self._safe_error(HTTPStatus.NOT_FOUND, "unknown_command")
            return
        logger.debug(
            "browser_capture.post_command_acked",
            request_id=self._request_id(),
            command_id=command.command_id,
            status=command.status,
        )
        self._send_json(
            HTTPStatus.OK,
            BrowserPostAckPayload(command_id=command.command_id, status=command.status).model_dump(mode="json"),
        )


def make_server(
    host: str,
    port: int,
    *,
    spool_path: Path | None = None,
    archive_root: Path | None = None,
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
        archive_root=archive_root,
        allowed_origins=frozenset(allowed_origins),
        allow_remote=allow_remote,
        auth_token=auth_token,
    )
    config.validate()
    return BrowserCaptureHTTPServer((host, port), config)


__all__ = ["BrowserCaptureHTTPServer", "BrowserCaptureHandler", "make_server"]
