"""Daemon HTTP API server for the Polylogue local daemon."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from polylogue.core.loopback import is_loopback_origin
from polylogue.daemon import user_state_http, workspace_routes
from polylogue.daemon.events import (
    emit_daemon_event,
    get_latest_event_id,
)
from polylogue.daemon.status import daemon_status_payload
from polylogue.errors import PolylogueError
from polylogue.logging import get_logger
from polylogue.paths import db_path
from polylogue.surfaces.payloads import (
    MutationResultPayload,
    QueryErrorPayload,
    QueryMissDiagnosticsPayload,
    ReaderActionAvailabilityPayload,
    TargetRefPayload,
    _build_flags_from_conversation,
    _extract_cwd,
    _extract_repo,
    reader_anchor,
    reader_conversation_actions,
    reader_message_actions,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.archive.query.spec import ConversationQuerySpec

logger = get_logger(__name__)


def _json_bytes(payload: object) -> bytes:
    import orjson

    return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)


def _dump_target_ref(target_ref: TargetRefPayload) -> dict[str, object]:
    return target_ref.model_dump(mode="json", exclude_none=True)


def _dump_actions(actions: Mapping[str, ReaderActionAvailabilityPayload]) -> dict[str, object]:
    return {name: availability.model_dump(mode="json", exclude_none=True) for name, availability in actions.items()}


def _staged_inbox_source(raw_path: object, inbox: Path) -> tuple[Path | None, str | None]:
    """Resolve an ingest request to an existing file already staged in inbox.

    The HTTP API must not copy arbitrary local paths supplied by clients.
    Clients with local filesystem access stage first; the daemon then
    schedules the matching inbox entry for its normal watcher/convergence path.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "missing_path"

    source_name = PurePath(raw_path).name
    if not source_name or source_name in {".", ".."}:
        return None, "invalid_path"

    try:
        inbox_root = inbox.resolve()
        candidates = list(inbox.iterdir())
    except FileNotFoundError:
        return None, "path_not_found"
    except OSError:
        return None, "path_not_found"

    for candidate in candidates:
        if candidate.name != source_name:
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(inbox_root)
        except ValueError:
            return None, "invalid_path"
        return resolved, None

    return None, "path_not_found"


def _confidence_tag(status: str) -> str:
    """Map a cost-estimate status to the MK3 data-quality chip vocabulary (#1122).

    The reader cost panel renders one chip per surfaced number; the chip
    classname comes from this mapping. ``q-canonical`` is reserved for
    provider-reported exact totals; ``q-estimated`` for catalog-priced
    estimates; ``q-heuristic`` for partial coverage; ``q-unavailable``
    for the unpriced state.
    """
    if status == "exact":
        return "q-canonical"
    if status == "priced":
        return "q-estimated"
    if status == "partial":
        return "q-heuristic"
    return "q-unavailable"


def _basis_dict(basis: Any) -> dict[str, float]:
    return {
        "provider_reported_usd": float(basis.provider_reported_usd),
        "api_equivalent_usd": float(basis.api_equivalent_usd),
        "subscription_equivalent_usd": float(basis.subscription_equivalent_usd),
        "catalog_priced_usd": float(basis.catalog_priced_usd),
        "tool_surcharge_usd": float(basis.tool_surcharge_usd),
    }


def _usage_dict(usage: Any) -> dict[str, int]:
    return {
        "input_tokens": int(usage.input_tokens),
        "output_tokens": int(usage.output_tokens),
        "cache_read_tokens": int(usage.cache_read_tokens),
        "cache_write_tokens": int(usage.cache_write_tokens),
        "total_tokens": int(usage.total_tokens),
    }


def _cost_panel_payload(insight: Any) -> dict[str, object]:
    """Render a typed ``SessionCostInsight`` as a cost-panel JSON payload (#1122)."""
    estimate = insight.estimate
    return {
        "conversation_id": insight.conversation_id,
        "provider": insight.provider_name,
        "model_name": estimate.model_name,
        "normalized_model": estimate.normalized_model,
        "status": estimate.status,
        "confidence": float(estimate.confidence),
        "confidence_tag": _confidence_tag(estimate.status),
        "currency": estimate.currency,
        "total_usd": float(estimate.total_usd),
        "basis": _basis_dict(estimate.basis),
        "usage": _usage_dict(estimate.usage),
        "per_model_breakdown": [
            {
                "model_name": entry.model_name,
                "normalized_model": entry.normalized_model,
                "total_usd": float(entry.total_usd),
                "basis": _basis_dict(entry.basis),
                "usage": _usage_dict(entry.usage),
            }
            for entry in estimate.per_model_breakdown
        ],
        "missing_reasons": list(estimate.missing_reasons),
        "unavailable_reason": estimate.unavailable_reason,
        "provenance": list(estimate.provenance),
    }


def _empty_cost_payload(conversation_id: str, provider: str | None) -> dict[str, object]:
    """Explicit ``unavailable`` payload when no cost insight is materialized (#1122)."""
    return {
        "conversation_id": conversation_id,
        "provider": provider,
        "model_name": None,
        "normalized_model": None,
        "status": "unavailable",
        "confidence": 0.0,
        "confidence_tag": "q-unavailable",
        "currency": "USD",
        "total_usd": 0.0,
        "basis": {
            "provider_reported_usd": 0.0,
            "api_equivalent_usd": 0.0,
            "subscription_equivalent_usd": 0.0,
            "catalog_priced_usd": 0.0,
            "tool_surcharge_usd": 0.0,
        },
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
        },
        "per_model_breakdown": [],
        "missing_reasons": ["no_session_cost_insight"],
        "unavailable_reason": "no_messages",
        "provenance": [],
    }


def _message_type_value(message: object) -> str:
    message_type = getattr(message, "message_type", "")
    if hasattr(message_type, "value"):
        return str(message_type.value)
    return str(message_type)


def daemon_safe_handler(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that discriminates PolylogueError types to HTTP status codes.

    PolylogueError subclasses carry ``http_status_code`` — use it.
    Unexpected exceptions map to 500 and are logged.
    """

    @functools.wraps(fn)
    def wrapper(self: DaemonAPIHandler, *args: object, **kwargs: object) -> None:
        try:
            fn(self, *args, **kwargs)
        except PolylogueError as exc:
            status = (
                HTTPStatus(exc.http_status_code)
                if 100 <= exc.http_status_code <= 599
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            field = getattr(exc, "field", None)
            self._send_json(
                status,
                QueryErrorPayload(
                    error=type(exc).__name__,
                    detail=str(exc),
                    field=field,
                ).model_dump(mode="json"),
            )
        except Exception:
            logger.exception("unhandled error in %s", fn.__name__)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                QueryErrorPayload(error="internal_error").model_dump(mode="json"),
            )

    return wrapper


def _build_query_spec_params(
    params: dict[str, list[str]],
    handler: DaemonAPIHandler,
) -> dict[str, object]:
    """Build ConversationQuerySpec-compatible params from HTTP query string."""
    spec_params: dict[str, object] = {}

    for key in (
        "query",
        "contains",
        "exclude_text",
        "retrieval_lane",
        "cwd_prefix",
        "action_text",
        "title",
        "conv_id",
        "since",
        "until",
        "sort",
        "similar_text",
        "since_session",
        "since_session_id",
        "message_type",
    ):
        val = handler._get_param(params, key)
        if val is not None:
            spec_params[key] = val

    for key in (
        "provider",
        "exclude_provider",
        "tag",
        "exclude_tag",
        "repo",
        "has_type",
        "referenced_path",
        "action",
        "exclude_action",
        "action_sequence",
        "tool",
        "exclude_tool",
    ):
        val = handler._get_param(params, key)
        if val is not None:
            spec_params[key] = val

    for key in (
        "latest",
        "reverse",
        "filter_has_tool_use",
        "filter_has_thinking",
        "filter_has_paste",
        "typed_only",
    ):
        if handler._get_bool(params, key):
            spec_params[key] = True

    for key in ("min_messages", "max_messages", "min_words", "sample"):
        val = handler._get_param(params, key)
        if val is not None:
            with contextlib.suppress(ValueError, TypeError):
                spec_params[key] = int(val)

    return spec_params


def _check_auth_logic(
    auth_token: str | None,
    client_host: str,
    auth_header: str,
) -> _AuthResult:
    """Pure logic for auth checks — testable without HTTP handler setup."""
    if not auth_token:
        return _AuthResult(allowed=True, reason=None)
    if not auth_header.startswith("Bearer "):
        return _AuthResult(allowed=False, reason="unauthorized")
    if auth_header[7:] != auth_token:
        return _AuthResult(allowed=False, reason="unauthorized")
    return _AuthResult(allowed=True, reason=None)


class _AuthResult:
    def __init__(self, *, allowed: bool, reason: str | None) -> None:
        self.allowed = allowed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.allowed


class DaemonAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the daemon API server.

    Runs async archive operations via ``asyncio.run()`` in a thread pool
    worker. This is safe because each request runs in its own thread.
    """

    server: DaemonAPIHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    @property
    def _auth_token(self) -> str | None:
        return getattr(self.server, "auth_token", None)

    @property
    def _api_host(self) -> str:
        return getattr(self.server, "api_host", "127.0.0.1")

    @property
    def _client_host(self) -> str:
        """Extract client IP from the request."""
        # The client_address is (host, port) from the underlying socket.
        return str(self.client_address[0])

    def _check_auth(self) -> bool:
        """Validate the Authorization header against the daemon token.

        When no token is configured the API is open (local dev default).
        When a token IS configured, all clients — including localhost —
        must present it. Loopback is not a security boundary when a
        browser on the same host can reach the daemon.

        ``access_token`` query parameter is accepted as a fallback for
        clients that cannot set custom headers (``EventSource``).
        """
        auth_header = self.headers.get("Authorization", "")
        if not auth_header and self._auth_token:
            parsed = urlparse(self.path)
            qs_params = parse_qs(parsed.query)
            qs_token = qs_params.get("access_token", [None])[0]
            if qs_token:
                auth_header = f"Bearer {qs_token}"
        result = _check_auth_logic(self._auth_token, self._client_host, auth_header)
        if not result.allowed:
            self._send_error(HTTPStatus.UNAUTHORIZED, result.reason or "unauthorized")
        return result.allowed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(
        self,
        status: HTTPStatus,
        payload: object,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        raw = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, status: HTTPStatus, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: HTTPStatus, code: str) -> None:
        self._send_json(status, {"ok": False, "error": code})

    def _parse_path(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        path = parsed.path.strip("/").split("/")
        params = parse_qs(parsed.query)
        return path, params

    def _get_param(self, params: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
        values = params.get(key)
        if values:
            return values[0]
        return default

    def _get_int(self, params: dict[str, list[str]], key: str, default: int = 0) -> int:
        val = self._get_param(params, key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        return default

    def _get_bool(self, params: dict[str, list[str]], key: str) -> bool:
        val = self._get_param(params, key)
        if val is None:
            return False
        return val.lower() in ("1", "true", "yes", "on")

    # ------------------------------------------------------------------
    # Async operation runner
    # ------------------------------------------------------------------

    async def _run_archive_query(self, handler: Callable) -> object:  # type: ignore[type-arg]
        from polylogue.api import Polylogue

        async with Polylogue() as polylogue:
            return await handler(polylogue)

    def _sync_run(self, handler: Callable) -> object:  # type: ignore[type-arg]
        return asyncio.run(self._run_archive_query(handler))

    def do_OPTIONS(self) -> None:
        self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed")

    # ------------------------------------------------------------------
    # Route dispatch
    # ------------------------------------------------------------------

    def _dispatch_get(self, path: list[str], params: dict[str, list[str]]) -> None:
        """Dispatch GET requests via route table."""
        # Web shell is the only unauthenticated endpoint (localhost only).
        if (
            path == [""]
            or (len(path) == 2 and path[0] == "c" and bool(path[1]))
            or (len(path) == 2 and path[0] == "w" and path[1] in workspace_routes.WORKSPACE_SHELL_MODES)
        ):
            self._serve_web_shell()
            return

        if not self._check_auth():
            return

        if path == ["api", "health", "check"]:
            self._handle_health_check()
        elif path == ["api", "health"]:
            self._handle_health()
        elif path == ["api", "status"]:
            self._handle_status(params)
        elif path == ["api", "events"]:
            self._handle_events(params)
        elif path == ["api", "conversations"]:
            self._handle_list_conversations(params)
        elif path == ["api", "facets"]:
            self._handle_facets(params)
        elif workspace_routes.dispatch_get(self, path, params) or (
            path[:2] == ["api", "user"] and user_state_http.dispatch_get(self, path[2:], params)
        ):
            return
        elif path == ["api", "sources"]:
            self._handle_sources()
        elif len(path) == 3 and path[:2] == ["api", "conversations"] and path[2]:
            self._handle_get_conversation(path[2])
        elif len(path) == 4 and path[:2] == ["api", "conversations"] and path[3] == "messages":
            self._handle_get_messages(path[2], params)
        elif len(path) == 4 and path[:2] == ["api", "conversations"] and path[3] == "raw":
            self._handle_get_conversation_raw(path[2])
        elif len(path) == 4 and path[:2] == ["api", "conversations"] and path[3] == "cost":
            self._handle_get_conversation_cost(path[2])
        elif len(path) == 4 and path[:3] == ["api", "raw_artifacts"]:
            self._handle_get_raw_artifact(path[3])
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_GET(self) -> None:
        path, params = self._parse_path()
        self._dispatch_get(path, params)

    def _check_cross_origin(self) -> bool:
        """Reject browser cross-origin POSTs to mutating endpoints.

        Returns True if the request is allowed, sends 403 and returns
        False if the Origin header indicates a cross-origin browser request.
        """
        origin = self.headers.get("Origin", "")
        if not origin:
            return True  # Not a browser request
        if is_loopback_origin(origin):
            return True
        self._send_error(HTTPStatus.FORBIDDEN, "cross_origin_denied")
        return False

    def do_POST(self) -> None:
        path, params = self._parse_path()

        if not self._check_auth():
            return
        if not self._check_cross_origin():
            return

        if path == ["api", "reset"]:
            self._handle_reset()
            return
        if path == ["api", "ingest"]:
            self._handle_ingest()
            return
        if path == ["api", "maintenance", "plan"]:
            self._handle_maintenance_plan()
            return
        if path == ["api", "maintenance", "run"]:
            self._handle_maintenance_run()
            return
        if path[:2] == ["api", "user"] and user_state_http.dispatch_post(self, path[2:]):
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_DELETE(self) -> None:
        path, params = self._parse_path()

        if not self._check_auth():
            return
        if not self._check_cross_origin():
            return

        if path[:2] == ["api", "user"] and user_state_http.dispatch_delete(self, path[2:], params):
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    # ------------------------------------------------------------------
    # Web shell
    # ------------------------------------------------------------------

    def _serve_web_shell(self) -> None:
        from polylogue.daemon.web_shell import WEB_SHELL_HTML

        self._send_html(HTTPStatus.OK, WEB_SHELL_HTML)

    # ------------------------------------------------------------------
    # Handlers: health
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_health_check(self) -> None:
        """CI-facing health check with deterministic exit semantics.

        Returns 200 when all FAST health checks pass, 503 when any
        non-OK health alert is present.  Suitable for health check
        endpoints in Docker, systemd, and CI pipelines.
        """
        try:
            from polylogue.daemon.health import HealthTier, check_health

            health = check_health(tiers={HealthTier.FAST, HealthTier.MEDIUM})
            if health.overall_status == "ok":
                self._send_json(HTTPStatus.OK, {"ok": True, "status": "healthy"})
            else:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "status": health.overall_status, "alerts": len(health.alerts)},
                )
        except Exception:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "status": "error", "detail": "health check failed"},
            )

    def _handle_health(self) -> None:
        dbp = db_path()
        db_size = dbp.stat().st_size if dbp.exists() else 0
        wal_size = 0
        wal = dbp.with_suffix(".db-wal")
        if wal.exists():
            wal_size = wal.stat().st_size
        disk_free = 0
        try:
            st = os.statvfs(str(dbp.parent))
            disk_free = st.f_frsize * st.f_bavail
        except OSError:
            pass

        quick_check_ok = True
        try:
            from polylogue.config import Config
            from polylogue.paths import archive_root, render_root
            from polylogue.readiness import get_readiness

            cfg = Config(archive_root=archive_root(), render_root=render_root(), sources=[])
            report = get_readiness(cfg, deep=False, probe_only=False)
            quick_check_ok = report.counts().ok > 0
        except Exception:
            quick_check_ok = False

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": quick_check_ok,
                "db_size_bytes": db_size,
                "wal_size_bytes": wal_size,
                "disk_free_bytes": disk_free,
                "blob_dir_size_bytes": 0,
                "quick_check": "pass" if quick_check_ok else "error",
                "quick_check_age_s": None,
            },
        )

    # ------------------------------------------------------------------
    # Handlers: status
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_status(self, params: dict[str, list[str]] | None = None) -> None:
        latest_event_id = get_latest_event_id()
        etag = f'W/"events-{latest_event_id}"'
        if_none_match = self.headers.get("If-None-Match", "")
        if if_none_match and if_none_match == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED.value)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        status = daemon_status_payload()
        if isinstance(status, dict):
            status["last_event_id"] = latest_event_id
        self._send_json(HTTPStatus.OK, status, extra_headers={"ETag": etag})

    # ------------------------------------------------------------------
    # Handlers: events (SSE + JSON poll) — implementation in events_http
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_events(self, params: dict[str, list[str]]) -> None:
        """Dispatch ``GET /api/events`` to the realtime channel handler."""
        from polylogue.daemon.events_http import handle_events

        handle_events(self, params)

    # ------------------------------------------------------------------
    # Handlers: list conversations
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_list_conversations(self, params: dict[str, list[str]]) -> None:
        query_params = _build_query_spec_params(params, self)
        limit = self._get_int(params, "limit", 50)
        offset = self._get_int(params, "offset", 0)

        async def _list(poly: Polylogue) -> object:
            return await self._do_list(poly, query_params, limit, offset)

        result = self._sync_run(_list)
        self._send_json(HTTPStatus.OK, result)

    async def _do_list(
        self,
        poly: Polylogue,
        query_params: dict[str, object],
        limit: int,
        offset: int,
    ) -> object:
        from polylogue.archive.query.spec import ConversationQuerySpec

        spec = ConversationQuerySpec.from_params(
            {**query_params, "limit": limit, "offset": offset},
        )
        query_text = query_params.get("query") or query_params.get("contains")

        # When search terms are present, return ranked result envelope with
        # per-hit match evidence instead of plain row dicts.
        if query_text and not query_params.get("similar_text"):
            return await self._do_search_list(poly, spec, limit, offset)

        filter_obj = spec.build_filter(poly.repository)
        summaries = await filter_obj.list_summaries()
        total = await spec.count(poly.repository)

        diagnostics = None
        if not summaries and spec.has_filters():
            with contextlib.suppress(ImportError):
                from polylogue.config import ConfigError

                try:
                    raw_diag = await poly.operations.diagnose_query_miss(spec)
                    diagnostics = QueryMissDiagnosticsPayload.from_diagnostics(raw_diag)
                except ConfigError:
                    pass

        items: list[dict[str, object]] = []
        for summary in summaries:
            flags = _build_flags_from_conversation(summary)
            conversation_id = str(summary.id)
            target_ref = TargetRefPayload.conversation(conversation_id)
            row: dict[str, object] = {
                "id": conversation_id,
                "title": summary.display_title,
                "provider": str(summary.provider) if summary.provider else None,
                "target_ref": _dump_target_ref(target_ref),
                "anchor": reader_anchor("conversation", conversation_id),
                "actions": _dump_actions(reader_conversation_actions()),
                "date": summary.display_date.isoformat() if summary.display_date else None,
                "created_at": summary.created_at.isoformat() if summary.created_at else None,
                "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
                "message_count": getattr(summary, "message_count", 0) or 0,
                "word_count": getattr(summary, "word_count", None),
                "repo": _extract_repo(summary.provider_meta),
                "cwd_display": _extract_cwd(summary.provider_meta),
                "tags": summary.tags,
                "flags": flags.model_dump(mode="json") if flags else None,
                "summary": summary.summary,
            }
            items.append(row)

        result: dict[str, object] = {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
        if diagnostics is not None:
            result["diagnostics"] = diagnostics.model_dump(mode="json")
        return result

    async def _do_search_list(
        self,
        poly: Polylogue,
        spec: ConversationQuerySpec,
        limit: int,
        offset: int,
    ) -> object:
        """Return ranked search hits with match evidence for queries with search terms."""
        from polylogue.archive.query.search_hits import search_hits_for_plan

        plan = spec.to_plan()
        hits = await search_hits_for_plan(plan, poly.repository)
        total = await spec.count(poly.repository)

        diagnostics = None
        if not hits and spec.has_filters():
            with contextlib.suppress(Exception):
                try:
                    raw_diag = await poly.operations.diagnose_query_miss(spec)
                    diagnostics = QueryMissDiagnosticsPayload.from_diagnostics(raw_diag)
                except Exception:
                    pass

        hit_dicts: list[dict[str, object]] = []
        for hit in hits:
            flags = _build_flags_from_conversation(hit.summary)
            conversation_id = str(hit.conversation_id)
            target_ref = TargetRefPayload.conversation(conversation_id)
            if hit.message_id is not None:
                match_target_ref = TargetRefPayload.message(conversation_id=conversation_id, message_id=hit.message_id)
                match_anchor = reader_anchor("message", hit.message_id)
                match_actions = reader_message_actions()
            else:
                match_target_ref = target_ref
                match_anchor = reader_anchor("conversation", conversation_id)
                match_actions = reader_conversation_actions()
            hit_dicts.append(
                {
                    "id": conversation_id,
                    "title": hit.summary.display_title,
                    "provider": str(hit.summary.provider) if hit.summary.provider else None,
                    "target_ref": _dump_target_ref(target_ref),
                    "anchor": reader_anchor("conversation", conversation_id),
                    "actions": _dump_actions(reader_conversation_actions()),
                    "date": hit.summary.display_date.isoformat() if hit.summary.display_date else None,
                    "created_at": hit.summary.created_at.isoformat() if hit.summary.created_at else None,
                    "updated_at": hit.summary.updated_at.isoformat() if hit.summary.updated_at else None,
                    "message_count": hit.summary.message_count,
                    "word_count": getattr(hit.summary, "word_count", None),
                    "repo": _extract_repo(hit.summary.provider_meta),
                    "cwd_display": _extract_cwd(hit.summary.provider_meta),
                    "tags": hit.summary.tags,
                    "flags": flags.model_dump(mode="json") if flags else None,
                    "summary": hit.summary.summary,
                    "match": {
                        "rank": hit.rank,
                        "retrieval_lane": hit.retrieval_lane,
                        "match_surface": hit.match_surface,
                        "target_ref": _dump_target_ref(match_target_ref),
                        "anchor": match_anchor,
                        "actions": _dump_actions(match_actions),
                        "message_id": hit.message_id,
                        "snippet": hit.snippet,
                        "score": hit.score,
                        "matched_terms": list(hit.matched_terms),
                        "score_components": hit.score_components,
                    },
                }
            )

        result: dict[str, object] = {
            "hits": hit_dicts,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
        if diagnostics is not None:
            result["diagnostics"] = diagnostics.model_dump(mode="json")
        return result

    # ------------------------------------------------------------------
    # Handlers: get conversation
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_conversation(self, conv_id: str) -> None:
        async def _get(poly: Polylogue) -> object:
            return await self._do_get_conversation(poly, conv_id)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_conversation(self, poly: Polylogue, conv_id: str) -> object:
        conv = await poly.get_conversation(conv_id)
        if conv is None:
            return None
        flags = _build_flags_from_conversation(conv)
        conversation_id = str(conv.id)
        target_ref = TargetRefPayload.conversation(conversation_id)
        return {
            "id": conversation_id,
            "title": conv.title,
            "display_title": conv.display_title,
            "provider": str(conv.provider) if conv.provider else None,
            "target_ref": _dump_target_ref(target_ref),
            "anchor": reader_anchor("conversation", conversation_id),
            "actions": _dump_actions(reader_conversation_actions()),
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "message_count": len(conv.messages),
            "word_count": conv.word_count,
            "messages": [
                {
                    "id": str(msg.id),
                    "role": str(msg.role),
                    "text": msg.text,
                    "target_ref": _dump_target_ref(
                        TargetRefPayload.message(conversation_id=conversation_id, message_id=msg.id)
                    ),
                    "anchor": reader_anchor("message", msg.id),
                    "actions": _dump_actions(reader_message_actions()),
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "message_type": _message_type_value(msg),
                    "word_count": msg.word_count,
                    "has_tool_use": bool(msg.has_tool_use) if hasattr(msg, "has_tool_use") else False,
                    "has_thinking": bool(msg.has_thinking) if hasattr(msg, "has_thinking") else False,
                    "has_paste": bool(msg.has_paste) if hasattr(msg, "has_paste") else False,
                }
                for msg in conv.messages
            ],
            "tags": conv.tags,
            "branch_type": str(conv.branch_type) if conv.branch_type else None,
            "parent_id": str(conv.parent_id) if conv.parent_id else None,
            "session_id": getattr(conv, "session_id", None),
            "repo": _extract_repo(conv.provider_meta),
            "cwd_display": _extract_cwd(conv.provider_meta),
            "model": conv.provider_meta.get("model") if conv.provider_meta else None,
            "flags": flags.model_dump(mode="json") if flags else None,
            "summary": conv.summary,
            "total": len(conv.messages),
        }

    # ------------------------------------------------------------------
    # Handlers: get conversation raw
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_conversation_raw(self, conv_id: str) -> None:
        async def _get(poly: Polylogue) -> object:
            return await self._do_get_conversation_raw(poly, conv_id)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_conversation_raw(self, poly: Polylogue, conv_id: str) -> object:
        conv = await poly.get_conversation(conv_id)
        if conv is None:
            return None
        raw_items = await poly.get_raw_artifacts_for_conversation(conv_id)
        provider_meta_serializable: dict[str, object] = {}
        if conv.provider_meta:
            for k, v in conv.provider_meta.items():
                try:
                    json.dumps({k: v})
                    provider_meta_serializable[k] = v
                except (TypeError, ValueError):
                    provider_meta_serializable[k] = str(v)
        return {
            "id": str(conv.id),
            "provider": str(conv.provider) if conv.provider else None,
            "title": conv.display_title,
            "provider_meta": provider_meta_serializable,
            "branch_type": str(conv.branch_type) if conv.branch_type else None,
            "parent_id": str(conv.parent_id) if conv.parent_id else None,
            "session_id": getattr(conv, "session_id", None),
            "raw_artifacts": raw_items,
        }

    # ------------------------------------------------------------------
    # Handlers: get conversation cost (#1122)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_conversation_cost(self, conv_id: str) -> None:
        async def _get(poly: Polylogue) -> object:
            return await self._do_get_conversation_cost(poly, conv_id)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_conversation_cost(self, poly: Polylogue, conv_id: str) -> object:
        from polylogue.insights.archive import SessionCostInsightQuery

        insights = await poly.list_session_cost_insights(SessionCostInsightQuery(conversation_id=conv_id))
        if not insights:
            # No matching session-cost insight: confirm the conversation exists
            # so we can distinguish "unknown conversation" (404) from "cost
            # surface unavailable" (200 with explicit unavailable shape).
            conv = await poly.get_conversation(conv_id)
            if conv is None:
                return None
            return _empty_cost_payload(conv_id, str(conv.provider) if conv.provider else None)
        return _cost_panel_payload(insights[0])

    # ------------------------------------------------------------------
    # Handlers: get messages
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_messages(self, conv_id: str, params: dict[str, list[str]]) -> None:
        limit = self._get_int(params, "limit", 50)
        offset = self._get_int(params, "offset", 0)

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_messages(poly, conv_id, limit, offset)

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_messages(self, poly: Polylogue, conv_id: str, limit: int, offset: int) -> object:
        messages, total = await poly.get_messages_paginated(conv_id, limit=limit, offset=offset)
        conversation_id = str(conv_id)
        return {
            "messages": [
                {
                    "id": str(msg.id),
                    "role": str(msg.role),
                    "text": msg.text,
                    "target_ref": _dump_target_ref(
                        TargetRefPayload.message(conversation_id=conversation_id, message_id=msg.id)
                    ),
                    "anchor": reader_anchor("message", msg.id),
                    "actions": _dump_actions(reader_message_actions()),
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "message_type": _message_type_value(msg),
                    "word_count": msg.word_count,
                }
                for msg in messages
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # Handlers: get raw artifact
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_raw_artifact(self, artifact_id: str) -> None:
        async def _get(poly: Polylogue) -> object:
            return await self._do_get_raw_artifacts(poly, artifact_id)

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_raw_artifacts(self, poly: Polylogue, artifact_id: str) -> object:
        raw_items = await poly.get_raw_artifacts_for_conversation(artifact_id)
        return {"raw_artifacts": raw_items}

    # ------------------------------------------------------------------
    # Handlers: facets
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_facets(self, params: dict[str, list[str]]) -> None:
        query_params = _build_query_spec_params(params, self)

        async def _get(poly: Polylogue) -> object:
            return await self._do_facets(poly, query_params)

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_facets(
        self,
        poly: Polylogue,
        query_params: dict[str, object],
    ) -> object:
        if query_params:
            from polylogue.archive.query.spec import ConversationQuerySpec

            spec = ConversationQuerySpec.from_params(query_params)
            filter_obj = spec.build_filter(poly.repository)
            summaries = await filter_obj.list_summaries()
            providers: dict[str, int] = {}
            tags: dict[str, int] = {}
            total_messages = 0
            for s in summaries:
                providers[str(s.provider)] = providers.get(str(s.provider), 0) + 1
                total_messages += s.message_count or 0
                for t in s.tags:
                    tags[t] = tags.get(t, 0) + 1
            return {
                "scoped_to_query": True,
                "providers": providers,
                "tags": tags,
                "repos": {},
                "cwd_prefixes": {},
                "message_types": {},
                "action_types": {},
                "has_flags": {},
                "time_range": None,
                "total_conversations": len(summaries),
                "total_messages": total_messages,
            }
        stats = await poly.stats()
        tags = await poly.list_tags()

        return {
            "scoped_to_query": False,
            "providers": stats.providers,
            "tags": tags,
            "repos": {},
            "cwd_prefixes": {},
            "message_types": {},
            "action_types": {},
            "has_flags": {},
            "time_range": None,
            "total_conversations": stats.conversation_count,
            "total_messages": stats.message_count,
        }

    @daemon_safe_handler
    def _handle_user_state(self, handler: Callable[..., None], *args: object) -> None:
        handler(self, *args)

    # ------------------------------------------------------------------
    # Handlers: sources
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_sources(self) -> None:
        from polylogue.sources.live.watcher import default_sources

        sources = default_sources()
        self._send_json(
            HTTPStatus.OK,
            {"sources": [{"name": s.name, "root": str(s.root), "exists": s.exists()} for s in sources]},
        )

    # ------------------------------------------------------------------
    # Handlers: reset
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_reset(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        body_text = body_raw.decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        scope = body.get("scope", "all")
        conv_id = body.get("conversation_id")

        if scope == "conversation" and not conv_id:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        op_id = f"reset-{scope}-{conv_id[:16] if conv_id else 'all'}"

        async def _do_reset(poly: Polylogue) -> dict[str, object]:
            if scope == "conversation" and conv_id:
                # Route through the typed delete contract so resolution and
                # idempotency live in ArchiveMutationsMixin (#862). The prior
                # implementation invoked the async ``delete_conversation`` from
                # a sync callback, sending the resulting coroutine into the
                # JSON encoder unchanged.
                result = await poly.delete_conversation_safe(conv_id)
                return {"deleted": result.outcome == "deleted", "conversation_id": conv_id}
            return {"ok": True}

        result = self._sync_run(_do_reset)

        emit_daemon_event("reset", operation_id=op_id, payload=result if isinstance(result, dict) else None)

        deleted = result.get("deleted", False) if isinstance(result, dict) else False
        response = MutationResultPayload(
            status="deleted" if deleted else "ok",
            detail=f"reset {scope}" if deleted else f"reset {scope} — no conversations matched",
        )
        self._send_json(HTTPStatus.OK, response.model_dump())

    # ------------------------------------------------------------------
    # Handlers: ingest
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_ingest(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        body_text = body_raw.decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        from polylogue.paths import archive_root

        inbox = archive_root() / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        source, error = _staged_inbox_source(body.get("path"), inbox)
        if error is not None:
            self._send_error(HTTPStatus.BAD_REQUEST, error)
            return
        assert source is not None

        from polylogue.operations.import_contracts import ImportOperation

        op_id = f"ingest-{source.name}"
        emit_daemon_event("ingest", operation_id=op_id, payload={"path": str(source), "inbox": str(inbox)})

        operation = ImportOperation.pending(
            operation_id=op_id,
            path=str(source),
            message="Ingestion scheduled. Check status for progress.",
        )
        self._send_json(HTTPStatus.ACCEPTED, operation.to_dict())

    @daemon_safe_handler
    def _handle_maintenance_plan(self) -> None:
        """POST /api/maintenance/plan — dry-run summary for maintenance targets."""
        content_length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        body_text = body_raw.decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        raw_targets: list[str] = body.get("targets", [])
        targets: tuple[str, ...] = tuple(str(t) for t in raw_targets)

        from polylogue.config import Config
        from polylogue.maintenance.envelope import envelope_from_operation
        from polylogue.maintenance.planner import preview_backfill
        from polylogue.paths import archive_root, render_root

        config = Config(
            archive_root=archive_root(),
            render_root=render_root(),
            sources=[],
        )
        result = preview_backfill(config, targets=targets)
        envelope = envelope_from_operation(result, origin="daemon", mode="preview")
        self._send_json(HTTPStatus.OK, envelope.to_dict())

    @daemon_safe_handler
    def _handle_maintenance_run(self) -> None:
        """POST /api/maintenance/run — execute (or dry-run) maintenance."""
        content_length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        body_text = body_raw.decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        raw_targets: list[str] = body.get("targets", [])
        targets: tuple[str, ...] = tuple(str(t) for t in raw_targets)
        dry_run: bool = bool(body.get("dry_run", False))

        from polylogue.config import Config
        from polylogue.maintenance.envelope import envelope_from_operation
        from polylogue.maintenance.planner import execute_backfill
        from polylogue.paths import archive_root, render_root

        config = Config(
            archive_root=archive_root(),
            render_root=render_root(),
            sources=[],
        )
        result = execute_backfill(config, targets=targets, dry_run=dry_run)
        envelope = envelope_from_operation(result, origin="daemon", mode="execute")
        self._send_json(HTTPStatus.OK, envelope.to_dict())


class DaemonAPIHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server for the daemon API."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        auth_token: str | None = None,
        api_host: str = "127.0.0.1",
    ) -> None:
        super().__init__(server_address, handler_class)
        self.auth_token = auth_token
        self.api_host = api_host
