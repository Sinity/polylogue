"""HTTP surface for the local browser-capture receiver."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from uuid import uuid4

from pydantic import ValidationError

from polylogue.browser_capture.actions import (
    BrowserActionConflictError,
    BrowserActionLeaseError,
    BrowserActionQuotaError,
    BrowserActionStateError,
    browser_action_capabilities,
    claim_action,
    enqueue_action,
    get_action,
    list_actions,
    read_action_attachment,
    reconcile_action,
    update_action,
)
from polylogue.browser_capture.capture_jobs import CaptureJobError, registry_for_receiver
from polylogue.browser_capture.models import (
    BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD,
    BrowserActionCapabilitiesPayload,
    BrowserActionListPayload,
    BrowserActionPayload,
    BrowserActionReconcileRequest,
    BrowserActionRequest,
    BrowserActionUpdateRequest,
    BrowserBackfillCheckpointAcceptedPayload,
    BrowserBackfillCheckpointPayload,
    BrowserBackfillCheckpointRequest,
    BrowserCaptureAcceptedPayload,
    BrowserCaptureCapabilitiesPayload,
    BrowserCaptureEnvelope,
    BrowserCaptureErrorPayload,
)
from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    SpoolQuotaExceededError,
    capture_response_id,
    existing_capture_state,
    read_backfill_checkpoint,
    receiver_identity,
    receiver_status_payload,
    write_backfill_checkpoint,
    write_capture_envelope,
)
from polylogue.core.json import dumps_bytes
from polylogue.core.loopback import is_loopback_host
from polylogue.logging import get_logger

logger = get_logger(__name__)

MAX_BROWSER_CAPTURE_BODY_BYTES = 128 * 1024 * 1024
_SAFE_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")


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

    def _send_bytes(self, payload: bytes, *, content_type: str, filename: str) -> None:
        if any(ord(character) < 32 or ord(character) == 127 for character in filename):
            raise ValueError("download filename contains control characters")
        safe_content_type = content_type if _SAFE_MEDIA_TYPE.fullmatch(content_type) else "application/octet-stream"
        ascii_filename = filename.encode("ascii", "ignore").decode("ascii").replace('"', "").replace("\\", "")
        ascii_filename = ascii_filename or "download"
        origin = self.headers.get("Origin")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("X-Request-ID", self._request_id())
        self.send_header("Content-Type", safe_content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{quote(filename, safe='')}",
        )
        if _origin_allowed(origin, self.server.config):
            self.send_header("Access-Control-Allow-Origin", origin or "null")
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(payload)

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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Request-ID, X-Polylogue-Client-Protocol",
        )
        self.send_header("Access-Control-Max-Age", "600")
        # Chrome's Private Network Access policy blocks an already-origin-approved
        # extension fetch to this loopback receiver unless the preflight explicitly
        # grants it (https://developer.chrome.com/blog/private-network-access-preflight).
        # Without this the browser reports a bare "Failed to fetch" with no other
        # signal, so the extension popup's health check hangs forever.
        if self.headers.get("Access-Control-Request-Private-Network", "").lower() == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
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
        if parsed.path == "/v1/browser-actions/capabilities":
            self._send_json(
                HTTPStatus.OK,
                BrowserActionCapabilitiesPayload(providers=browser_action_capabilities()).model_dump(mode="json"),
            )
            return
        if parsed.path == "/v1/browser-actions":
            params = parse_qs(parsed.query)
            claim_by = params.get("claim_by", [""])[0]
            try:
                actions = (
                    [claimed]
                    if claim_by
                    and (claimed := claim_action(claim_by, spool_path=self.server.config.spool_path)) is not None
                    else ([] if claim_by else list_actions(spool_path=self.server.config.spool_path))
                )
            except ValueError:
                self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_id")
                return
            except (OSError, BrowserActionStateError) as exc:
                logger.warning("browser_capture.action_list_failed", request_id=self._request_id(), error=repr(exc))
                self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
                return
            self._send_json(HTTPStatus.OK, BrowserActionListPayload(actions=actions).model_dump(mode="json"))
            return
        if parsed.path.startswith("/v1/browser-actions/") and "/attachments/" in parsed.path:
            prefix = "/v1/browser-actions/"
            action_id, attachment_id = parsed.path[len(prefix) :].split("/attachments/", maxsplit=1)
            try:
                result = read_action_attachment(action_id, attachment_id, spool_path=self.server.config.spool_path)
            except BrowserActionConflictError:
                self._safe_error(HTTPStatus.CONFLICT, "browser_action_attachment_integrity_mismatch")
                return
            except ValueError:
                self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_id")
                return
            except (OSError, BrowserActionStateError) as exc:
                logger.warning(
                    "browser_capture.action_attachment_failed", request_id=self._request_id(), error=repr(exc)
                )
                self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
                return
            if result is None:
                self._safe_error(HTTPStatus.NOT_FOUND, "unknown_browser_action_attachment")
                return
            attachment, content = result
            self._send_bytes(content, content_type=attachment.mime_type, filename=attachment.name)
            return
        if parsed.path.startswith("/v1/browser-actions/"):
            action_id = parsed.path[len("/v1/browser-actions/") :]
            try:
                action = get_action(action_id, spool_path=self.server.config.spool_path)
            except ValueError:
                self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_id")
                return
            except (OSError, BrowserActionStateError) as exc:
                logger.warning("browser_capture.action_read_failed", request_id=self._request_id(), error=repr(exc))
                self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
                return
            if action is None:
                self._safe_error(HTTPStatus.NOT_FOUND, "unknown_browser_action")
                return
            self._send_json(HTTPStatus.OK, BrowserActionPayload(action=action).model_dump(mode="json"))
            return
        if parsed.path.startswith("/v1/capture-jobs/"):
            job_id = parsed.path.removeprefix("/v1/capture-jobs/")
            if not job_id or "/" in job_id:
                self._safe_error(HTTPStatus.NOT_FOUND, "not_found")
                return
            params = parse_qs(parsed.query)
            try:
                protocol = int(params.get("client_protocol", ["-1"])[0])
            except ValueError:
                protocol = -1
            try:
                registry = registry_for_receiver(
                    self.server.config.spool_path,
                    receiver_identity(self.server.config),
                )
                if job_id == "capabilities":
                    capture_job_payload = registry.capabilities()
                elif job_id == "orphans":
                    capture_job_payload = registry.list_orphans(protocol)
                else:
                    capture_job_payload = registry.get(
                        job_id,
                        {
                            "provider": params.get("provider", [""])[0],
                            "account_scope": params.get("account_scope", [""])[0],
                            "client_protocol": protocol,
                        },
                    )
            except CaptureJobError as exc:
                self._capture_job_error(exc)
                return
            except (sqlite3.Error, OSError) as exc:
                self._capture_job_storage_error(exc)
                return
            self._send_json(HTTPStatus.OK, capture_job_payload)
            return
        if parsed.path == "/v1/backfill-checkpoint":
            params = parse_qs(parsed.query)
            instance_id = params.get("extension_instance_id", [""])[0]
            if not instance_id:
                self._safe_error(HTTPStatus.BAD_REQUEST, "missing_extension_instance_id")
                return
            record = read_backfill_checkpoint(instance_id, spool_path=self.server.config.spool_path)
            if record is None:
                self._safe_error(HTTPStatus.NOT_FOUND, "checkpoint_not_found")
                return
            self._send_json(
                HTTPStatus.OK,
                BrowserBackfillCheckpointPayload(
                    extension_instance_id=record.extension_instance_id,
                    checkpoint=record.checkpoint,
                    stored_at=record.stored_at,
                ).model_dump(mode="json"),
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
        if path == "/v1/browser-actions":
            self._browser_action_enqueue()
            return
        if path.startswith("/v1/browser-actions/") and path.endswith("/events"):
            action_id = path[len("/v1/browser-actions/") : -len("/events")]
            self._browser_action_update(action_id)
            return
        if path.startswith("/v1/browser-actions/") and path.endswith("/reconcile"):
            action_id = path[len("/v1/browser-actions/") : -len("/reconcile")]
            self._browser_action_reconcile(action_id)
            return
        if path in {"/v1/capture-jobs", "/v1/capture-jobs/discover"} or (
            path.startswith("/v1/capture-jobs/") and path.endswith(("/adopt", "/update"))
        ):
            self._capture_job_post(path)
            return
        if path == "/v1/backfill-checkpoint":
            self._backfill_checkpoint_store()
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

    def do_PUT(self) -> None:
        self._observe_request("PUT", self._do_put)

    def _do_put(self) -> None:
        if self._reject_origin() or self._reject_token():
            return
        path = urlparse(self.path).path
        if path.startswith("/v1/capture-jobs/") and path.endswith("/checkpoint"):
            self._capture_job_checkpoint(path)
            return
        self._safe_error(HTTPStatus.NOT_FOUND, "not_found")

    def _capture_job_body(self) -> dict[str, object] | None:
        payload = self._read_json_body()
        if not isinstance(payload, dict):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_capture_job")
            return None
        if "client_protocol" not in payload:
            raw_protocol = self.headers.get("X-Polylogue-Client-Protocol", "-1")
            try:
                payload["client_protocol"] = int(raw_protocol)
            except ValueError:
                payload["client_protocol"] = -1
        return payload

    def _capture_job_error(self, exc: CaptureJobError) -> None:
        try:
            status = HTTPStatus(exc.status)
        except ValueError:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._send_json(status, {"error": {"code": exc.code, "details": exc.details}})

    def _capture_job_storage_error(self, exc: sqlite3.Error | OSError) -> None:
        logger.warning("browser_capture.capture_job_registry_unavailable", error=repr(exc))
        self._capture_job_error(CaptureJobError(500, "registry_unavailable"))

    def _capture_job_post(self, path: str) -> None:
        payload = self._capture_job_body()
        if payload is None:
            return
        try:
            registry = registry_for_receiver(
                self.server.config.spool_path,
                receiver_identity(self.server.config),
            )
            if path == "/v1/capture-jobs":
                status, result = registry.create(payload)
            elif path == "/v1/capture-jobs/discover":
                status, result = HTTPStatus.OK, registry.discover(payload)
            elif path.endswith("/update"):
                job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/update")
                status, result = HTTPStatus.OK, registry.update(job_id, payload)
            else:
                job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/adopt")
                status, result = HTTPStatus.OK, registry.adopt(job_id, payload)
        except CaptureJobError as exc:
            self._capture_job_error(exc)
            return
        except (sqlite3.Error, OSError) as exc:
            self._capture_job_storage_error(exc)
            return
        self._send_json(HTTPStatus(status), result)

    def _capture_job_checkpoint(self, path: str) -> None:
        payload = self._capture_job_body()
        if payload is None:
            return
        job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/checkpoint")
        try:
            result = registry_for_receiver(
                self.server.config.spool_path,
                receiver_identity(self.server.config),
            ).checkpoint(job_id, payload)
        except CaptureJobError as exc:
            self._capture_job_error(exc)
            return
        except (sqlite3.Error, OSError) as exc:
            self._capture_job_storage_error(exc)
            return
        self._send_json(HTTPStatus.OK, result)

    def _browser_action_enqueue(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserActionRequest.model_validate(payload)
            action = enqueue_action(
                request,
                receiver_id=receiver_identity(self.server.config),
                spool_path=self.server.config.spool_path,
            )
        except ValidationError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action")
            return
        except BrowserActionConflictError:
            self._safe_error(HTTPStatus.CONFLICT, "browser_action_conflict")
            return
        except BrowserActionQuotaError:
            self._safe_error(HTTPStatus.TOO_MANY_REQUESTS, "browser_action_quota_exceeded")
            return
        except ValueError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_attachment")
            return
        except (OSError, BrowserActionStateError) as exc:
            logger.warning("browser_capture.action_enqueue_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        self._send_json(HTTPStatus.ACCEPTED, BrowserActionPayload(action=action).model_dump(mode="json"))

    def _browser_action_update(self, action_id: str) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserActionUpdateRequest.model_validate(payload)
            action = update_action(action_id, request, spool_path=self.server.config.spool_path)
        except ValidationError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_update")
            return
        except BrowserActionLeaseError:
            self._safe_error(HTTPStatus.CONFLICT, "browser_action_lease_owner_mismatch")
            return
        except BrowserActionConflictError:
            self._safe_error(HTTPStatus.CONFLICT, "browser_action_receipt_conflict")
            return
        except ValueError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_id")
            return
        except (OSError, BrowserActionStateError) as exc:
            logger.warning("browser_capture.action_update_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        if action is None:
            self._safe_error(HTTPStatus.NOT_FOUND, "unknown_browser_action")
            return
        self._send_json(HTTPStatus.OK, BrowserActionPayload(action=action).model_dump(mode="json"))

    def _browser_action_reconcile(self, action_id: str) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserActionReconcileRequest.model_validate(payload)
            action = reconcile_action(action_id, request, spool_path=self.server.config.spool_path)
        except ValidationError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_reconciliation")
            return
        except BrowserActionConflictError:
            self._safe_error(HTTPStatus.CONFLICT, "browser_action_reconciliation_conflict")
            return
        except ValueError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_id")
            return
        except (OSError, BrowserActionStateError) as exc:
            logger.warning("browser_capture.action_reconcile_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        if action is None:
            self._safe_error(HTTPStatus.NOT_FOUND, "unknown_browser_action")
            return
        self._send_json(HTTPStatus.OK, BrowserActionPayload(action=action).model_dump(mode="json"))

    def _backfill_checkpoint_store(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserBackfillCheckpointRequest.model_validate(payload)
        except ValidationError:
            logger.warning("browser_capture.invalid_backfill_checkpoint", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_backfill_checkpoint")
            return
        try:
            record = write_backfill_checkpoint(request, spool_path=self.server.config.spool_path)
        except SpoolQuotaExceededError as exc:
            logger.warning(
                "browser_capture.backfill_checkpoint_quota_exceeded", request_id=self._request_id(), error=str(exc)
            )
            self._safe_error(HTTPStatus.TOO_MANY_REQUESTS, "backfill_checkpoint_quota_exceeded")
            return
        except OSError as exc:
            logger.warning(
                "browser_capture.backfill_checkpoint_write_failed", request_id=self._request_id(), error=repr(exc)
            )
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        accepted = BrowserBackfillCheckpointAcceptedPayload(
            extension_instance_id=record.extension_instance_id,
            stored_at=record.stored_at,
            bytes_written=len(_json_bytes(record.model_dump(mode="json"))),
        )
        logger.debug(
            "browser_capture.backfill_checkpoint_stored",
            request_id=self._request_id(),
            extension_instance_id=record.extension_instance_id,
            bytes_written=accepted.bytes_written,
        )
        self._send_json(HTTPStatus.ACCEPTED, accepted.model_dump(mode="json"))


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
