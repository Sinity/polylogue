"""HTTP surface for the local browser-capture receiver."""

from __future__ import annotations

import hmac
import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO, NotRequired, TypedDict, cast
from urllib.parse import parse_qs, quote, urlparse
from uuid import uuid4

import ijson
from pydantic import ValidationError

from polylogue.browser_capture.actions import (
    ACTION_ATTACHMENT_CHUNK_BYTES,
    BrowserActionConflictError,
    BrowserActionLeaseError,
    BrowserActionQuotaError,
    BrowserActionStateError,
    browser_action_capabilities,
    claim_action,
    decide_action_approval,
    enqueue_action,
    get_action,
    list_actions,
    open_action_attachment,
    reconcile_action,
    store_action_attachment,
    update_action,
)
from polylogue.browser_capture.capture_jobs import CaptureJobError, CaptureJobRegistry, registry_for_receiver
from polylogue.browser_capture.capture_stream import (
    CaptureBodyIncompleteError,
    CaptureEnvelopeError,
    CaptureSummary,
    SpoolStorageExhaustedError,
    StagedCapture,
    is_storage_exhausted,
    reap_stale_staging,
    stage_capture_body,
    stage_capture_chunks,
    summarize_capture_file,
)
from polylogue.browser_capture.models import (
    BROWSER_CAPTURE_API_SCHEMA,
    BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD,
    BrowserActionApprovalDecisionRequest,
    BrowserActionCapabilitiesPayload,
    BrowserActionListPayload,
    BrowserActionPayload,
    BrowserActionReconcileRequest,
    BrowserActionRequest,
    BrowserActionUpdateRequest,
    BrowserCaptureAcceptedPayload,
    BrowserCaptureCapabilitiesPayload,
    BrowserCaptureErrorPayload,
    BrowserCaptureHealthEventAcceptedPayload,
    BrowserCaptureHealthEventRequest,
    BrowserCapturePairingRedeemPayload,
    BrowserCapturePairingRedeemRequest,
    BrowserCaptureReceiverAttestationPayload,
    BrowserCaptureReceiverAttestationRequest,
)
from polylogue.browser_capture.pairing import (
    PairingCodeAlreadyUsedError,
    PairingCodeError,
    PairingCodeExpiredError,
    PairingCodeInvalidError,
    PairingCodeRateLimitedError,
    redeem_pairing_code,
)
from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    BrowserCaptureSpoolConflictError,
    admit_staged_capture,
    attest_receiver,
    capture_response_id,
    existing_capture_state,
    receiver_identity,
    receiver_status_payload,
)
from polylogue.core.loopback import is_loopback_host
from polylogue.logging import INFO, WARNING, emit, get_logger
from polylogue.paths import archive_root as default_archive_root
from polylogue.schemas.observation_spill import StreamedJSONDocument

# Import the daemon event ledger inside capture-health route handlers so HTTP
# server bootstrap does not load its storage dependencies before a health request.

logger = get_logger(__name__)

#: Bound on the ordinary in-memory control routes (browser actions, pairing,
#: health reports, assertion candidates). Captures and CaptureJob bodies use
#: spool staging and the disk-backed JSON view instead.
MAX_CONTROL_BODY_BYTES = 128 * 1024 * 1024
_CONTENT_LENGTH = re.compile(r"[0-9]+")
_SAFE_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_JSON_SPOOL_REFUSAL = b'{"error":{"code":"spool_storage_exhausted","details":{}}}'
_JSON_STORAGE_FAILURE = b'{"error":{"code":"registry_unavailable","details":{}}}'
_CAPTURE_JOB_BODY_FIELDS = frozenset(
    {
        "acquisition_id",
        "after",
        "before_revision",
        "binding",
        "checkpoint",
        "client_protocol",
        "cursor",
        "descriptor_digest",
        "expected_lease_generation",
        "expected_revision",
        "generation",
        "intent",
        "intent_key",
        "kind",
        "lease_id",
        "lease_ttl_seconds",
        "limit",
        "member_name",
        "member_names",
        "metadata",
        "ordinal",
        "outcome",
        "payload",
        "plan_digest",
        "proof",
        "provenance",
        "provider",
        "provider_meta",
        "refs",
        "request_id",
        "retention",
        "retry",
        "scope",
        "session_id",
        "sha256",
        "size_bytes",
    }
)


class _MissionControlArchivePayload(TypedDict):
    status: str
    session_id: str | None
    ref: str | None


class _MissionControlCostPayload(TypedDict):
    status: str
    total_usd: float | None
    provenance: list[str]


class _MissionControlAssertionsPayload(TypedDict):
    status: str
    items: list[dict[str, object]]


class _MissionControlPayload(TypedDict):
    status: str
    archive: _MissionControlArchivePayload
    cost: _MissionControlCostPayload
    assertions: _MissionControlAssertionsPayload
    reason: NotRequired[str]


def mission_control_archive_facts(
    archive_root: Path,
    indexed_session_id: str,
) -> tuple[_MissionControlCostPayload, _MissionControlAssertionsPayload] | None:
    """Read cost and judged assertions for one canonical session.

    Returns ``None`` when the archive cannot answer, so the caller degrades to
    an explicit unknown instead of reporting a fabricated zero-cost success.
    """
    try:
        from polylogue import Polylogue
        from polylogue.analysis.archive import SessionCostInsightQuery
        from polylogue.api.sync.bridge import run_coroutine_sync

        poly = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
        cost: _MissionControlCostPayload = {"status": "unknown", "total_usd": None, "provenance": []}
        costs = run_coroutine_sync(
            poly.list_session_cost_insights(SessionCostInsightQuery(session_id=indexed_session_id))
        )
        if costs:
            estimate = costs[0].estimate
            cost = {
                "status": estimate.status if estimate.status != "unavailable" else "unknown",
                "total_usd": None if estimate.total_usd is None else float(estimate.total_usd),
                "provenance": list(estimate.provenance),
            }
        # Membership comes from the canonical composed transcript, not claim
        # scope metadata or a guessed message-ID prefix. Candidates stay private.
        claims = run_coroutine_sync(
            poly.list_assertion_claim_payloads(session_id=indexed_session_id, statuses=("active",), limit=5)
        )
        assertions: _MissionControlAssertionsPayload = {
            "status": "available",
            "items": [claim.model_dump(mode="json") for claim in claims],
        }
    except Exception as exc:  # a read projection must degrade, never become success
        logger.warning("browser_capture.mission_control_degraded", error=repr(exc))
        return None
    return cost, assertions


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
        # Staging files left by a receiver that died mid-upload are invisible
        # to the spool quota; reclaim them before accepting new uploads.
        reaped = reap_stale_staging(config.spool_path)
        if reaped:
            emit("browser_capture.stale_staging_reaped", level=INFO, reclaimed=reaped)
        super().__init__(server_address, BrowserCaptureHandler)

    def service_actions(self) -> None:
        """Autonomous bounded receiver retirement, owned by serve_forever."""
        try:
            registry_for_receiver(self.config.spool_path, receiver_identity(self.config)).maintenance_step()
        except (sqlite3.Error, OSError, CaptureJobError) as exc:
            emit("browser_capture.capture_job_registry_unavailable", level=WARNING, error=repr(exc))

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            registry_for_receiver(self.config.spool_path, receiver_identity(self.config)).close_maintenance()


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
        encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)

        def encoded_chunks() -> Iterator[bytes]:
            for piece in encoder.iterencode(payload):
                if piece:
                    yield piece.encode("utf-8")

        try:
            staged = stage_capture_chunks(
                iter(encoded_chunks()), spool_root=self.server.config.spool_path, durable=False
            )
        except SpoolStorageExhaustedError:
            self._send_json_staging_refusal(HTTPStatus.INSUFFICIENT_STORAGE, _JSON_SPOOL_REFUSAL)
            return
        except OSError as exc:
            if is_storage_exhausted(exc):
                self._send_json_staging_refusal(HTTPStatus.INSUFFICIENT_STORAGE, _JSON_SPOOL_REFUSAL)
            else:
                self._send_json_staging_refusal(HTTPStatus.INTERNAL_SERVER_ERROR, _JSON_STORAGE_FAILURE)
            return
        try:
            origin = self.headers.get("Origin")
            self.send_response(status.value)
            self.send_header("X-Request-ID", self._request_id())
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(staged.size_bytes))
            if _origin_allowed(origin, self.server.config):
                self.send_header("Access-Control-Allow-Origin", origin or "null")
                self.send_header("Vary", "Origin")
            self.end_headers()
            with staged.path.open("rb") as stream:
                while chunk := stream.read(64 * 1024):
                    self.wfile.write(chunk)
        finally:
            staged.discard()

    def _send_json_staging_refusal(self, status: HTTPStatus, body: bytes) -> None:
        """Send one fixed storage error without retrying the failed staging path."""
        if getattr(self, "_polylogue_status", None) is not None:
            self.close_connection = True
            return
        origin = self.headers.get("Origin")
        self.send_response(status.value)
        self.send_header("X-Request-ID", self._request_id())
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if _origin_allowed(origin, self.server.config):
            self.send_header("Access-Control-Allow-Origin", origin or "null")
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _send_attachment(self, stream: BinaryIO, size: int, *, content_type: str, filename: str) -> None:
        if any(ord(character) < 32 or ord(character) == 127 for character in filename):
            raise ValueError("download filename contains control characters")
        safe_content_type = content_type if _SAFE_MEDIA_TYPE.fullmatch(content_type) else "application/octet-stream"
        ascii_filename = filename.encode("ascii", "ignore").decode("ascii").replace('"', "").replace("\\", "")
        ascii_filename = ascii_filename or "download"
        origin = self.headers.get("Origin")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("X-Request-ID", self._request_id())
        self.send_header("Content-Type", safe_content_type)
        self.send_header("Content-Length", str(size))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{quote(filename, safe='')}",
        )
        if _origin_allowed(origin, self.server.config):
            self.send_header("Access-Control-Allow-Origin", origin or "null")
            self.send_header("Vary", "Origin")
        self.end_headers()
        while chunk := stream.read(ACTION_ATTACHMENT_CHUNK_BYTES):
            self.wfile.write(chunk)

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
        # The daemon normally uses the lazy stdlib logger, whose plain handler
        # does not render structured kwargs.  Keep this rejection forensic
        # without emitting credentials or arbitrary request headers.
        logger.warning(
            "browser_capture.token_rejected request_id=%s path=%s origin=%s",
            self._request_id(),
            urlparse(self.path).path,
            self.headers.get("Origin") or "-",
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
            "Content-Type, Authorization, X-Request-ID, X-Polylogue-Client-Protocol, X-Polylogue-Extension-Contract, X-Polylogue-Checkpoint, X-Polylogue-Native",
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
        if parsed.path == "/v1/mission-control":
            params = parse_qs(parsed.query)
            provider = params.get("provider", [""])[0]
            session_id = params.get("provider_session_id", [""])[0]
            if not provider or not session_id:
                self._safe_error(HTTPStatus.BAD_REQUEST, "missing_provider_or_session")
                return
            self._mission_control(provider, session_id)
            return
        if parsed.path == "/v1/capture-health":
            self._capture_health_list()
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
            body_started = False
            try:
                with open_action_attachment(
                    action_id, attachment_id, spool_path=self.server.config.spool_path
                ) as result:
                    if result is None:
                        self._safe_error(HTTPStatus.NOT_FOUND, "unknown_browser_action_attachment")
                        return
                    attachment, stream = result
                    body_started = True
                    self._send_attachment(
                        stream, attachment.size_bytes, content_type=attachment.mime_type, filename=attachment.name
                    )
            except BrowserActionConflictError:
                self._safe_error(HTTPStatus.CONFLICT, "browser_action_attachment_integrity_mismatch")
            except ValueError:
                self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_id")
            except (OSError, BrowserActionStateError) as exc:
                logger.warning(
                    "browser_capture.action_attachment_failed",
                    request_id=self._request_id(),
                    error_type=type(exc).__name__,
                )
                if body_started:
                    self.close_connection = True
                else:
                    self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "attachment_storage_unavailable")
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
            suffix = parsed.path.removeprefix("/v1/capture-jobs/")
            if "/checkpoint-artifacts/" in suffix:
                if self.server.config.auth_token is None:
                    self._safe_error(HTTPStatus.UNAUTHORIZED, "checkpoint_artifact_auth_required")
                    return
                job_id, digest = suffix.split("/checkpoint-artifacts/", 1)
                descriptor = self._checkpoint_descriptor()
                if descriptor is None:
                    return
                sent = False
                try:
                    registry = registry_for_receiver(
                        self.server.config.spool_path, receiver_identity(self.server.config)
                    )
                    with (
                        registry.result_scope(),
                        registry.checkpoint_artifact(job_id, digest, descriptor) as (stream, size),
                    ):
                        self.send_response(HTTPStatus.OK.value)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(size))
                        self.send_header("X-Request-ID", self._request_id())
                        origin = self.headers.get("Origin")
                        if _origin_allowed(origin, self.server.config):
                            self.send_header("Access-Control-Allow-Origin", origin or "null")
                            self.send_header("Vary", "Origin")
                        self.end_headers()
                        sent = True
                        while chunk := stream.read(64 * 1024):
                            self.wfile.write(chunk)
                except CaptureJobError as exc:
                    if not sent:
                        self._capture_job_error(exc)
                except (sqlite3.Error, OSError) as exc:
                    if not sent:
                        self._capture_job_storage_error(exc)
                return
            if suffix.endswith("/events"):
                job_id = suffix.removesuffix("/events")
                params = parse_qs(parsed.query)
                try:
                    protocol = int(params.get("client_protocol", ["-1"])[0])
                    limit = min(max(int(params.get("limit", ["100"])[0]), 1), 500)
                    raw_cursor = params.get("before_revision", [""])[0]
                    before_revision = int(raw_cursor) if raw_cursor else None
                    if before_revision is not None and before_revision > (1 << 63) - 1:
                        raise ValueError("cursor outside SQLite integer range")
                except ValueError:
                    self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_capture_job_events_query")
                    return
                try:
                    registry = registry_for_receiver(
                        self.server.config.spool_path, receiver_identity(self.server.config)
                    )
                    with registry.result_scope():
                        capture_job_events_payload = registry.events(
                            job_id,
                            {
                                "provider": params.get("provider", [""])[0],
                                "scope": self._capture_scope_query(params),
                                "client_protocol": protocol,
                                "limit": limit,
                                "before_revision": before_revision,
                            },
                        )
                        self._send_json(HTTPStatus.OK, capture_job_events_payload)
                except CaptureJobError as exc:
                    if getattr(self, "_polylogue_status", None) is None:
                        self._capture_job_error(exc)
                    else:
                        self.close_connection = True
                    return
                except (sqlite3.Error, OSError) as exc:
                    if getattr(self, "_polylogue_status", None) is None:
                        self._capture_job_storage_error(exc)
                    else:
                        self.close_connection = True
                return
            if suffix.startswith("orphans/") and suffix.endswith("/payload"):
                if self.server.config.auth_token is None:
                    self._safe_error(HTTPStatus.UNAUTHORIZED, "orphan_inspection_auth_required")
                    return
                source_digest = suffix[len("orphans/") : -len("/payload")]
                params = parse_qs(parsed.query)
                try:
                    protocol = int(params.get("client_protocol", ["-1"])[0])
                except ValueError:
                    protocol = -1
                try:
                    registry = registry_for_receiver(
                        self.server.config.spool_path, receiver_identity(self.server.config)
                    )
                    with registry.result_scope(), registry.inspect_orphan(source_digest, protocol) as (stream, size):
                        self.send_response(HTTPStatus.OK.value)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(size))
                        self.send_header("X-Request-ID", self._request_id())
                        origin = self.headers.get("Origin")
                        if _origin_allowed(origin, self.server.config):
                            self.send_header("Access-Control-Allow-Origin", origin or "null")
                            self.send_header("Vary", "Origin")
                        self.end_headers()
                        while chunk := stream.read(64 * 1024):
                            self.wfile.write(chunk)
                except CaptureJobError as exc:
                    self._capture_job_error(exc)
                except (sqlite3.Error, OSError) as exc:
                    # After response headers, a disconnected inspector simply
                    # ends the stream; retained evidence is never consumed.
                    if getattr(self, "_polylogue_status", None) != HTTPStatus.OK.value:
                        self._capture_job_storage_error(exc)
                return
            job_id = suffix
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
                with registry.result_scope():
                    if job_id == "capabilities":
                        capture_job_payload = registry.capabilities()
                    elif job_id == "orphans":
                        capture_job_payload = registry.list_orphans(protocol, params.get("cursor", [None])[0])
                    else:
                        capture_job_payload = registry.get(
                            job_id,
                            {
                                "provider": params.get("provider", [""])[0],
                                "scope": self._capture_scope_query(params),
                                "client_protocol": protocol,
                            },
                        )
                    self._send_json(HTTPStatus.OK, capture_job_payload)
            except CaptureJobError as exc:
                if getattr(self, "_polylogue_status", None) is None:
                    self._capture_job_error(exc)
                else:
                    self.close_connection = True
                return
            except (sqlite3.Error, OSError) as exc:
                if getattr(self, "_polylogue_status", None) is None:
                    self._capture_job_storage_error(exc)
                else:
                    self.close_connection = True
            return
        self._safe_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_POST(self) -> None:
        self._observe_request("POST", self._do_post)

    def _content_length(self, *, allow_empty: bool = False) -> int | None:
        """Return a declared length (zero only for file uploads), or send the error."""
        if allow_empty and self.headers.get("Content-Length") is None:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_content_length")
            return None
        declared = self.headers.get("Content-Length", "0").strip()
        # ``Content-Length = 1*DIGIT``; ``int`` would also take a sign,
        # ``_`` separators and non-ASCII digits.
        if not _CONTENT_LENGTH.fullmatch(declared):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_content_length")
            return None
        length = int(declared)
        if length < 0 or (length == 0 and not allow_empty):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_body_size")
            return None
        return length

    def _read_json_body(self, *, max_bytes: int | None = MAX_CONTROL_BODY_BYTES) -> object | None:
        """Read and parse a JSON control message, sending an error and returning None on failure."""
        length = self._content_length()
        if length is None:
            return None
        if max_bytes is not None and length > max_bytes:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_body_size")
            return None
        raw = self.rfile.read(length)
        try:
            parsed: object = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("browser_capture.invalid_json", request_id=self._request_id())
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_json")
            return None
        return parsed

    def _stage_capture_body(self, *, progress: Callable[[], None] | None = None) -> StagedCapture | None:
        """Stream the capture body into the spool's staging area.

        No size refusal: the body is copied chunk by chunk while hashed, so
        the receiver never holds it. Its declared length is reserved on disk
        first; a spool filesystem that cannot hold it is retryable pressure
        (507), not a refusal of the capture.
        """
        length = self._content_length()
        if length is None:
            return None

        def read_chunk(size: int) -> bytes:
            # read1 returns available bytes, rather than waiting for a whole
            # requested chunk. Renew only this admitted operation's authority
            # when actual producer progress resumes, including after a pause.
            chunk = self.rfile.read1(size)
            if chunk and progress is not None:
                progress()
            return chunk

        try:
            return stage_capture_body(read_chunk, length, spool_root=self.server.config.spool_path)
        except CaptureBodyIncompleteError:
            emit(
                "browser_capture.incomplete_body",
                level=WARNING,
                reason="incomplete_body",
                request_id=self._request_id(),
            )
            self._safe_error(HTTPStatus.BAD_REQUEST, "incomplete_body")
            return None
        except SpoolStorageExhaustedError as exc:
            emit(
                "browser_capture.spool_storage_exhausted",
                level=WARNING,
                reason="spool_storage_exhausted",
                request_id=self._request_id(),
                bytes=exc.requested_bytes,
            )
            self._safe_error(HTTPStatus.INSUFFICIENT_STORAGE, "spool_storage_exhausted")
            return None
        except OSError as exc:
            if is_storage_exhausted(exc):
                emit(
                    "browser_capture.spool_storage_exhausted",
                    level=WARNING,
                    reason="spool_storage_exhausted",
                    request_id=self._request_id(),
                )
                self._safe_error(HTTPStatus.INSUFFICIENT_STORAGE, "spool_storage_exhausted")
                return None
            emit(
                "browser_capture.write_failed",
                level=WARNING,
                reason="write_failed",
                request_id=self._request_id(),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return None

    def _do_post(self) -> None:
        if self._reject_origin():
            return
        path = urlparse(self.path).path
        # Pairing redemption is deliberately unauthenticated -- its whole
        # purpose is bootstrapping the bearer token an unpaired extension
        # does not have yet (polylogue-gnie). Origin is still enforced above;
        # the pairing code itself (short-lived, single-use, rate-limited) is
        # the credential for this one route.
        if path == "/v1/pairing/redeem":
            self._pairing_redeem()
            return
        # Attestation is unauthenticated for the opposite reason: a client
        # must authenticate this receiver before it sends the bearer here.
        # The answer is a MAC over the client's fresh challenge, never the
        # bearer itself.
        if path == "/v1/receiver/attest":
            self._receiver_attest()
            return
        if self._reject_token():
            return
        if path == "/v1/capture-health":
            self._capture_health_report()
            return
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
        if path.startswith("/v1/browser-actions/") and path.endswith("/approval"):
            action_id = path[len("/v1/browser-actions/") : -len("/approval")]
            self._browser_action_approval(action_id)
            return
        if path in {"/v1/capture-jobs", "/v1/capture-jobs/discover"} or (
            path.startswith("/v1/capture-jobs/")
            and path.endswith(
                (
                    "/adopt",
                    "/update",
                    "/events",
                    "/native/begin",
                    "/native/cancel",
                    "/native/prepare",
                    "/native/plan",
                    "/native/asset",
                    "/native/finalize",
                    "/native/publish",
                )
            )
        ):
            self._capture_job_post(path)
            return
        if path == "/v1/assertion-candidates":
            self._assertion_candidate_capture()
            return
        if path != "/v1/browser-captures":
            self._safe_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        staged = self._stage_capture_body()
        if staged is None:
            return
        try:
            self._admit_capture(staged)
        finally:
            staged.discard()

    def _admit_capture(
        self,
        staged: StagedCapture,
        *,
        native_authority: bool = False,
        prepared_summary: CaptureSummary | None = None,
        respond: bool = True,
    ) -> dict[str, object] | None:
        def error(status: HTTPStatus, reason: str) -> None:
            if not respond:
                raise CaptureJobError(int(status), reason)
            self._safe_error(status, reason)

        try:
            summary = prepared_summary if prepared_summary is not None else summarize_capture_file(staged.path)
        except CaptureEnvelopeError as exc:
            emit("browser_capture.invalid_envelope", level=WARNING, reason=exc.reason, request_id=self._request_id())
            error(HTTPStatus.BAD_REQUEST, exc.reason)
            return None
        except OSError as exc:
            emit(
                "browser_capture.write_failed",
                level=WARNING,
                reason="write_failed",
                request_id=self._request_id(),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return None
        if summary.head.provenance.extension_instance_id is None and not native_authority:
            logger.warning("browser_capture.missing_instance_id", request_id=self._request_id())
            error(HTTPStatus.BAD_REQUEST, "missing_extension_instance_id")
            return None
        try:
            result = admit_staged_capture(staged, summary, spool_path=self.server.config.spool_path)
        except BrowserCaptureSpoolConflictError as exc:
            logger.warning("browser_capture.spool_conflict", request_id=self._request_id(), error=str(exc))
            error(HTTPStatus.CONFLICT, "spool_conflict")
            return None
        except OSError as exc:
            logger.warning("browser_capture.write_failed", request_id=self._request_id(), error=repr(exc))
            if is_storage_exhausted(exc):
                error(HTTPStatus.INSUFFICIENT_STORAGE, "spool_storage_exhausted")
                return None
            error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return None
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
        payload = BrowserCaptureAcceptedPayload(
            capture_id=capture_response_id(result.provider, result.provider_session_id, result.capture_id),
            provider=result.provider,
            provider_session_id=result.provider_session_id,
            artifact_ref=result.artifact_ref,
            outcome=result.outcome,
            submitted_content_hash=staged.sha256,
            content_hash=result.content_hash,
            dedup_content_hash=result.dedup_content_hash,
            bytes_written=result.bytes_written,
            replaced=result.replaced,
            deduplicated=result.deduplicated,
            capture_instance_id=result.capture_instance_id,
            accepted_identities=list(result.accepted_identities),
        ).model_dump(mode="json")
        if respond:
            self._send_json(HTTPStatus.ACCEPTED, payload)
        return payload

    def _assertion_candidate_capture(self) -> None:
        payload = self._read_json_body()
        policy = payload.get("context_policy") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or not isinstance(policy, dict) or policy.get("inject") is not False:
            self._safe_error(HTTPStatus.BAD_REQUEST, "candidate_policy_required")
            return
        evidence_refs = payload.get("evidence_refs")
        observation = payload.get("source_observation")
        if (
            not isinstance(payload.get("body_text"), str)
            or not isinstance(payload.get("kind"), str)
            or not isinstance(evidence_refs, list)
            or len(evidence_refs) != 1
            or not isinstance(evidence_refs[0], str)
            or not isinstance(observation, dict)
            or observation.get("fidelity") != "native"
            or not observation.get("provider_message_id")
            or not isinstance(observation.get("origin"), str)
            or not str(observation.get("origin")).strip()
            or not isinstance(observation.get("provider_conversation_id"), str)
            or not str(observation.get("provider_conversation_id")).strip()
            or not payload.get("target_ref")
        ):
            self._safe_error(HTTPStatus.BAD_REQUEST, "exact_message_evidence_required")
            return
        try:
            from polylogue.api.archive import candidate_capture_kind
            from polylogue.daemon.api_auth import resolve_api_auth_token
            from polylogue.daemon.socket_path import daemon_socket_path
            from polylogue.daemon_client import DaemonClient

            root = self.server.config.archive_root or default_archive_root()
            provider_message_id = str(observation["provider_message_id"])
            origin = str(observation["origin"])
            conversation_id = str(observation["provider_conversation_id"])
            expected_message_ref = f"{origin}:{conversation_id}:n:{provider_message_id}"
            if payload["target_ref"] != expected_message_ref:
                raise ValueError("selected message target does not match its native observation")
            config = self.server.config
            # The client holds the credential; the receipt it returns does not.
            client = DaemonClient(
                daemon_socket_path(root),
                auth_token=lambda: resolve_api_auth_token(
                    config.api_auth_token,
                    allow_no_auth=config.api_allow_no_auth,
                ),
            )
            response = client.operation_to_completion(
                # A mutation is accepted before it completes; follow it to the
                # terminal receipt rather than reading "accepted" as failure.
                "mutation.assertion.candidate.capture",
                {
                    "body_text": payload["body_text"],
                    "kind": candidate_capture_kind(payload["kind"]).value,
                    "refs": [f"message:{expected_message_ref}"],
                    # An assertion's scope is an object ref, so the candidate is
                    # scoped to the selected message's session; the capture
                    # artifact travels as source evidence.
                    "scope_refs": [f"session:{origin}:{conversation_id}"],
                    "evidence_refs": [evidence_refs[0]],
                    "author_ref": str(payload.get("author_ref") or "user:browser-extension"),
                    "author_kind": str(payload.get("author_kind") or "user"),
                    "idempotency_key": payload.get("idempotency_key"),
                },
                archive_root=str(root),
            )
            if response is not None and response.get("outcome") == "rejected":
                # The daemon refused this request's content; that is the
                # caller's error to fix, not an unavailable daemon.
                error = response.get("error")
                code = error.get("code") if isinstance(error, dict) else None
                self._safe_error(HTTPStatus.BAD_REQUEST, str(code or "daemon_candidate_capture_rejected"))
                return
            if response is None or response.get("outcome") not in {"completed", "no-effect"}:
                self._safe_error(HTTPStatus.SERVICE_UNAVAILABLE, "daemon_candidate_capture_unavailable")
                return
            candidate = response.get("result")
            if not isinstance(candidate, dict):
                self._safe_error(HTTPStatus.BAD_GATEWAY, "daemon_candidate_capture_invalid_result")
                return
        except (ValueError, KeyError) as exc:
            self._safe_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except (OSError, sqlite3.Error, RuntimeError) as exc:
            logger.warning("browser_capture.assertion_candidate_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "assertion_candidate_write_failed")
            return
        self._send_json(HTTPStatus.ACCEPTED, {"ok": True, "status": "applied", "candidate": candidate})

    def do_PUT(self) -> None:
        self._observe_request("PUT", self._do_put)

    def _do_put(self) -> None:
        if self._reject_origin() or self._reject_token():
            return
        path = urlparse(self.path).path
        self.close_connection = True
        if path == "/v1/browser-action-attachments":
            length = self._content_length(allow_empty=True)
            if length is None:
                return
            try:
                reference = store_action_attachment(self.rfile.read, length, spool_path=self.server.config.spool_path)
            except ValueError:
                self._safe_error(HTTPStatus.BAD_REQUEST, "incomplete_browser_action_attachment")
                return
            except BrowserActionConflictError:
                self._safe_error(HTTPStatus.CONFLICT, "browser_action_attachment_integrity_mismatch")
                return
            except OSError as exc:
                self._safe_error(
                    HTTPStatus.INSUFFICIENT_STORAGE if is_storage_exhausted(exc) else HTTPStatus.INTERNAL_SERVER_ERROR,
                    "attachment_storage_unavailable",
                )
                return
            self._send_json(HTTPStatus.CREATED, {"attachment_ref": reference, "size_bytes": length})
            return
        # A body can be refused before admission consumes it. Close this
        # request's connection so unread bytes cannot become another request.
        self.close_connection = True
        try:
            if path.startswith("/v1/capture-jobs/") and path.endswith(("/native/member", "/native/asset")):
                self._capture_job_native_member(path)
            elif path.startswith("/v1/capture-jobs/") and path.endswith("/checkpoint"):
                self._capture_job_checkpoint(path)
            else:
                self._safe_error(HTTPStatus.NOT_FOUND, "not_found")
        except CaptureJobError as exc:
            self._capture_job_error(exc)
        except (sqlite3.Error, OSError) as exc:
            self._capture_job_storage_error(exc)

    def _capture_job_error(self, exc: CaptureJobError) -> None:
        try:
            status = HTTPStatus(exc.status)
        except ValueError:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._send_json(status, {"error": {"code": exc.code, "details": exc.details}})

    def _capture_job_storage_error(self, exc: sqlite3.Error | OSError) -> None:
        logger.warning("browser_capture.capture_job_registry_unavailable", error=repr(exc))
        exhausted = isinstance(exc, OSError) and is_storage_exhausted(exc)
        exhausted = exhausted or (
            isinstance(exc, sqlite3.Error) and getattr(exc, "sqlite_errorcode", 0) & 0xFF == sqlite3.SQLITE_FULL
        )
        self._capture_job_error(
            CaptureJobError(507, "spool_storage_exhausted")
            if exhausted
            else CaptureJobError(500, "registry_unavailable")
        )

    def _capture_job_post(self, path: str) -> None:
        staged = self._stage_capture_body()
        if staged is None:
            return
        try:
            with StreamedJSONDocument(staged.path) as root:
                if not isinstance(root, dict):
                    self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_capture_job")
                    return
                # SpilledObject is a read-only lazy mapping. The registry
                # expects a mutable top-level dict for the protocol header;
                # retain every owned operation field and all nested lazy values.
                payload = cast(dict[str, object], {key: root[key] for key in root if key in _CAPTURE_JOB_BODY_FIELDS})
                if "client_protocol" not in payload:
                    try:
                        payload["client_protocol"] = int(self.headers.get("X-Polylogue-Client-Protocol", "-1"))
                    except ValueError:
                        payload["client_protocol"] = -1
                registry = registry_for_receiver(
                    self.server.config.spool_path,
                    receiver_identity(self.server.config),
                )
                try:
                    with registry.result_scope():
                        if path == "/v1/capture-jobs":
                            status, result = registry.create(payload)
                        elif path == "/v1/capture-jobs/discover":
                            status, result = HTTPStatus.OK, registry.discover(payload)
                        elif path.endswith("/native/begin"):
                            if self.server.config.auth_token is None:
                                raise CaptureJobError(401, "native_preparation_auth_required")
                            job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/native/begin")
                            status, result = HTTPStatus.OK, registry.native_begin(job_id, payload)
                        elif "/native/" in path:
                            if self.server.config.auth_token is None:
                                raise CaptureJobError(401, "native_preparation_auth_required")
                            job_id, operation = path.removeprefix("/v1/capture-jobs/").rsplit("/native/", 1)
                            operations = {
                                "cancel": registry.native_cancel,
                                "prepare": registry.native_prepare,
                                "plan": registry.native_plan,
                                "asset": registry.native_asset,
                                "finalize": registry.native_finalize,
                            }
                            if operation == "publish":
                                result = self._publish_native_capture(registry, job_id, payload)
                                status = HTTPStatus.ACCEPTED
                            elif operation in operations:
                                status, result = HTTPStatus.OK, operations[operation](job_id, payload)
                            else:
                                raise CaptureJobError(404, "not_found")
                        elif path.endswith("/update"):
                            job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/update")
                            status, result = HTTPStatus.OK, registry.update(job_id, payload)
                        elif path.endswith("/events"):
                            job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/events")
                            status, result = HTTPStatus.OK, registry.event(job_id, payload)
                        else:
                            job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/adopt")
                            status, result = HTTPStatus.OK, registry.adopt(job_id, payload)
                        self._send_json(HTTPStatus(status), result)
                except CaptureJobError as exc:
                    if getattr(self, "_polylogue_status", None) is None:
                        self._capture_job_error(exc)
                    else:
                        self.close_connection = True
                except (ValidationError, ValueError, ijson.JSONError):
                    if getattr(self, "_polylogue_status", None) is None:
                        self._capture_job_error(CaptureJobError(400, "invalid_native_preparation"))
                    else:
                        self.close_connection = True
                except (sqlite3.Error, OSError) as exc:
                    if getattr(self, "_polylogue_status", None) is None:
                        self._capture_job_storage_error(exc)
                    else:
                        self.close_connection = True
        except (ijson.JSONError, ValueError):
            emit("browser_capture.invalid_json", level=WARNING, request_id=self._request_id())
            if getattr(self, "_polylogue_status", None) is None:
                self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_json")
            else:
                self.close_connection = True
        except (sqlite3.Error, OSError) as exc:
            if getattr(self, "_polylogue_status", None) is None:
                self._capture_job_storage_error(exc)
            else:
                self.close_connection = True
        finally:
            staged.discard()

    def _publish_native_capture(
        self, registry: CaptureJobRegistry, job_id: str, body: dict[str, object]
    ) -> dict[str, object]:
        def admit(staged: StagedCapture, summary: CaptureSummary) -> dict[str, object]:
            payload = self._admit_capture(staged, native_authority=True, prepared_summary=summary, respond=False)
            if payload is None:
                raise CaptureJobError(500, "native_admission_missing_receipt")
            return payload

        return registry.native_publish(job_id, body, admit)

    def _capture_job_native_member(self, path: str) -> None:
        if self.server.config.auth_token is None:
            self._safe_error(HTTPStatus.UNAUTHORIZED, "native_preparation_auth_required")
            return
        try:
            descriptor = json.loads(self.headers.get("X-Polylogue-Native", ""))
            if not isinstance(descriptor, dict):
                raise ValueError("native descriptor must be an object")
        except (ValueError, TypeError):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_native_descriptor")
            return
        job_id = path.removeprefix("/v1/capture-jobs/").rsplit("/native/", 1)[0]
        registry = registry_for_receiver(self.server.config.spool_path, receiver_identity(self.server.config))
        # Admission precedes disk reservation and every body read. The
        # context remains owned until staging and publication settle.
        with registry.result_scope(), registry.artifact_progress(job_id, descriptor, native=True) as progress:
            staged = self._stage_capture_body(progress=progress)
            if staged is None:
                return
            try:
                progress()
                result = (
                    registry.native_asset(job_id, descriptor, staged)
                    if path.endswith("/native/asset")
                    else registry.native_member(job_id, descriptor, staged)
                )
                self._send_json(HTTPStatus.OK, result)
            finally:
                staged.discard()

    def _capture_job_checkpoint(self, path: str) -> None:
        if self.server.config.auth_token is None:
            self._safe_error(HTTPStatus.UNAUTHORIZED, "checkpoint_artifact_auth_required")
            return
        payload = self._checkpoint_descriptor()
        if payload is None:
            return
        job_id = path.removeprefix("/v1/capture-jobs/").removesuffix("/checkpoint")
        registry = registry_for_receiver(self.server.config.spool_path, receiver_identity(self.server.config))
        with registry.result_scope(), registry.artifact_progress(job_id, payload) as progress:
            staged = self._stage_capture_body(progress=progress)
            if staged is None:
                return
            try:
                progress()
                result = registry.checkpoint(job_id, payload, staged)
                self._send_json(HTTPStatus.OK, result)
            finally:
                staged.discard()

    @staticmethod
    def _capture_scope_query(params: dict[str, list[str]]) -> object:
        try:
            return json.loads(params.get("scope", [""])[0])
        except (TypeError, ValueError) as exc:
            raise CaptureJobError(400, "invalid_capture_scope") from exc

    def _checkpoint_descriptor(self) -> dict[str, object] | None:
        """Read scalar scope/fencing metadata, never checkpoint records."""
        try:
            payload = json.loads(self.headers.get("X-Polylogue-Checkpoint", ""))
        except (ValueError, TypeError):
            payload = None
        fields = {
            "provider": str,
            "scope": dict,
            "client_protocol": int,
            "request_id": str,
            "expected_revision": int,
            "lease_id": str,
            "generation": int,
            "proof": str,
            "sequence": int,
            "digest": str,
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != set(fields)
            or any(type(payload[key]) is not expected for key, expected in fields.items())
        ):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_checkpoint_descriptor")
            return None
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        if (
            not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", payload["provider"])
            or not re.fullmatch(uuid_pattern, payload["request_id"])
            or not re.fullmatch(uuid_pattern, payload["lease_id"])
            or not re.fullmatch(r"[A-Za-z0-9_-]{43}", payload["proof"])
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", payload["digest"])
            or any(
                not 0 <= payload[key] <= (1 << 53) - 1
                for key in ("client_protocol", "expected_revision", "generation", "sequence")
            )
        ):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_checkpoint_descriptor")
            return None
        payload["checkpoint"] = {"sequence": payload.pop("sequence"), "digest": payload.pop("digest")}
        return payload

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

    def _browser_action_approval(self, action_id: str) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserActionApprovalDecisionRequest.model_validate(payload)
            action = decide_action_approval(action_id, request, spool_path=self.server.config.spool_path)
        except ValidationError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_approval")
            return
        except BrowserActionConflictError:
            self._safe_error(HTTPStatus.CONFLICT, "browser_action_approval_conflict")
            return
        except ValueError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_browser_action_id")
            return
        except (OSError, BrowserActionStateError) as exc:
            logger.warning("browser_capture.action_approval_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "write_failed")
            return
        if action is None:
            self._safe_error(HTTPStatus.NOT_FOUND, "unknown_browser_action")
            return
        self._send_json(HTTPStatus.OK, BrowserActionPayload(action=action).model_dump(mode="json"))

    def _pairing_redeem(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserCapturePairingRedeemRequest.model_validate(payload)
        except ValidationError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_pairing_request")
            return
        try:
            token = redeem_pairing_code(request.code)
        except (PairingCodeExpiredError, PairingCodeAlreadyUsedError, PairingCodeInvalidError) as exc:
            logger.warning(
                "browser_capture.pairing_redeem_rejected", request_id=self._request_id(), reason=type(exc).__name__
            )
            self._safe_error(HTTPStatus.UNAUTHORIZED, "pairing_code_rejected")
            return
        except PairingCodeRateLimitedError:
            logger.warning("browser_capture.pairing_redeem_rate_limited", request_id=self._request_id())
            self._safe_error(HTTPStatus.TOO_MANY_REQUESTS, "pairing_code_rate_limited")
            return
        except PairingCodeError as exc:
            logger.warning("browser_capture.pairing_redeem_failed", request_id=self._request_id(), error=repr(exc))
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "pairing_failed")
            return
        logger.debug("browser_capture.pairing_redeemed", request_id=self._request_id())
        self._send_json(
            HTTPStatus.OK,
            BrowserCapturePairingRedeemPayload(
                auth_token=token,
                receiver_id=receiver_identity(self.server.config),
            ).model_dump(mode="json"),
        )

    def _receiver_attest(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserCaptureReceiverAttestationRequest.model_validate(payload)
        except ValidationError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_attestation_request")
            return
        proof = attest_receiver(self.server.config, request.challenge)
        if proof is None:
            # With auth disabled there is no secret to prove possession of.
            self._safe_error(HTTPStatus.CONFLICT, "receiver_auth_disabled")
            return
        self._send_json(
            HTTPStatus.OK,
            BrowserCaptureReceiverAttestationPayload(
                api_schema=BROWSER_CAPTURE_API_SCHEMA,
                receiver_id=receiver_identity(self.server.config),
                proof=proof,
            ).model_dump(mode="json"),
        )

    def _mission_control(self, provider: str, provider_session_id: str) -> None:
        """Return the receiver-authoritative, read-only Layer 2 projection."""
        state = existing_capture_state(
            provider,
            provider_session_id,
            spool_path=self.server.config.spool_path,
            archive_root=self.server.config.archive_root,
        )
        indexed_value = state.get("indexed_session_id")
        indexed = indexed_value if isinstance(indexed_value, str) else None
        # `run_daemon_services` constructs the receiver without an explicit
        # archive root, so the configured value is None on the production
        # route. `existing_capture_state` above already resolves the same
        # default to find the indexed session; refusing to resolve it here
        # made cost and assertions permanently unavailable in production while
        # every test that passed a temporary root saw them. The assertion
        # candidate handler resolves it the same way.
        archive_root = self.server.config.archive_root or default_archive_root()
        projection: _MissionControlPayload = {
            "status": "available" if indexed else "uncaptured",
            "archive": {
                "status": "available" if indexed else "uncaptured",
                "session_id": indexed,
                "ref": f"session:{indexed}" if indexed else None,
            },
            "cost": {"status": "unknown", "total_usd": None, "provenance": []},
            "assertions": {"status": "unknown", "items": []},
        }
        if indexed:
            facts = mission_control_archive_facts(archive_root, indexed)
            if facts is None:
                logger.warning(
                    "browser_capture.mission_control_degraded",
                    error="archive projection is unavailable",
                    archive_root=str(archive_root),
                )
                projection["status"] = "unknown"
                projection["reason"] = "archive_projection_unavailable"
            else:
                projection["cost"], projection["assertions"] = facts
        self._send_json(HTTPStatus.OK, projection)

    def _capture_health_report(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            request = BrowserCaptureHealthEventRequest.model_validate(payload)
        except ValidationError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_capture_health_event")
            return
        from polylogue.core.errors import SchemaSkew
        from polylogue.daemon.events import CAPTURE_HEALTH_EVENT_KIND, CaptureHistoryStorageError, emit_daemon_event

        try:
            event_id = emit_daemon_event(
                CAPTURE_HEALTH_EVENT_KIND,
                operation_id=request.extension_instance_id,
                payload={
                    "event": request.event,
                    "provider": request.provider,
                    "provider_session_id": request.provider_session_id,
                    "visible_count": request.visible_count,
                    "captured_count": request.captured_count,
                    "reason": request.reason,
                    "detail": request.detail,
                },
            )
        except SchemaSkew:
            self._safe_error(HTTPStatus.CONFLICT, "schema_skew")
            return
        except CaptureHistoryStorageError as exc:
            self._safe_error(HTTPStatus(exc.http_status_code), exc.code)
            return
        logger.debug(
            "browser_capture.capture_health_reported",
            request_id=self._request_id(),
            event=request.event,
            provider=request.provider,
            provider_session_id=request.provider_session_id,
        )
        self._send_json(
            HTTPStatus.ACCEPTED,
            BrowserCaptureHealthEventAcceptedPayload(event_id=event_id).model_dump(mode="json"),
        )

    def _capture_health_list(self) -> None:
        from polylogue.core.errors import SchemaSkew
        from polylogue.daemon.events import CaptureHistoryCursorError, CaptureHistoryStorageError, capture_health_page

        params = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        if set(params) - {"page_size", "cursor"} or any(len(values) != 1 for values in params.values()):
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_history_parameters")
            return
        try:
            page_size = int(params.get("page_size", ["100"])[0])
        except ValueError:
            self._safe_error(HTTPStatus.BAD_REQUEST, "invalid_history_page_size")
            return
        try:
            page = capture_health_page(page_size=page_size, cursor=params.get("cursor", [None])[0])
        except CaptureHistoryCursorError as exc:
            reason = str(exc)
            self._safe_error(
                HTTPStatus.CONFLICT if reason == "history_cursor_reset" else HTTPStatus.BAD_REQUEST, reason
            )
            return
        except SchemaSkew:
            self._safe_error(HTTPStatus.CONFLICT, "schema_skew")
            return
        except CaptureHistoryStorageError as exc:
            self._safe_error(HTTPStatus(exc.http_status_code), exc.code)
            return
        self._send_json(HTTPStatus.OK, {"ok": True, **page})


def make_server(
    host: str,
    port: int,
    *,
    spool_path: Path | None = None,
    archive_root: Path | None = None,
    allow_remote: bool = False,
    auth_token: str | None = None,
    extra_origins: tuple[str, ...] = (),
    api_auth_token: str | None = None,
    api_allow_no_auth: bool = False,
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
        api_auth_token=api_auth_token,
        api_allow_no_auth=api_allow_no_auth,
    )
    config.validate()
    return BrowserCaptureHTTPServer((host, port), config)


__all__ = ["BrowserCaptureHTTPServer", "BrowserCaptureHandler", "make_server"]
