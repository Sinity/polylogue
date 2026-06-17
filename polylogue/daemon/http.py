"""Daemon HTTP API server for the Polylogue local daemon."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs, urlparse

from polylogue.core.loopback import is_loopback_origin
from polylogue.core.sources import source_name_to_origin
from polylogue.daemon import user_state_http, workspace_routes
from polylogue.daemon.events import (
    emit_daemon_event,
    get_latest_event_id,
)
from polylogue.daemon.status_snapshot import get_status_snapshot_payload
from polylogue.daemon.web_shell_attachments import (
    LibraryEntry,
    attachment_to_envelope,
    build_library_payload,
    render_attachment_library_page,
)
from polylogue.daemon.web_shell_paste import (
    PasteBrowserEntry,
    build_paste_browser_payload,
    envelope_paste_spans,
    render_paste_browser_page,
    snippet_for_paste,
)
from polylogue.errors import PolylogueError
from polylogue.logging import get_logger
from polylogue.surfaces.payloads import (
    MutationResultPayload,
    QueryErrorPayload,
    QueryMissDiagnosticsPayload,
    ReaderActionAvailabilityPayload,
    TargetRefPayload,
    _build_flags_from_session,
    reader_anchor,
    reader_message_actions,
    reader_session_actions,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow
    from polylogue.surfaces.payloads import FacetBucketsPayload

logger = get_logger(__name__)


def _json_bytes(payload: object) -> bytes:
    import orjson

    return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)


def _web_reader_archive_root() -> Path | None:
    """Return the archive root when index.db reader routes should use it."""
    from polylogue.paths import archive_root

    root = archive_root()
    index_db = root / "index.db"
    return root if index_db.exists() else None


def _archive_origin_filter(params: dict[str, list[str]]) -> tuple[str | None, tuple[str, ...]]:
    origin_values = params.get("origin") or []
    if origin_values and origin_values[0]:
        origins = tuple(
            dict.fromkeys(token.strip() for value in origin_values for token in value.split(",") if token.strip())
        )
        return (origins[0] if len(origins) == 1 else None), origins
    return None, ()


def _archive_tag_filter(params: dict[str, list[str]]) -> tuple[str, ...]:
    """Collect ``?tag=`` filters (repeated and/or comma-separated)."""
    tags: list[str] = []
    for value in params.get("tag") or []:
        tags.extend(token.strip() for token in value.split(",") if token.strip())
    return tuple(dict.fromkeys(tags))


def _csv_values(params: dict[str, list[str]], key: str) -> tuple[str, ...]:
    """Collect repeated and/or comma-separated query-string values."""
    values: list[str] = []
    for value in params.get(key) or []:
        values.extend(token.strip() for token in value.split(",") if token.strip())
    return tuple(dict.fromkeys(values))


def _content_projection_from_params(params: dict[str, list[str]]) -> ContentProjectionSpec:
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec

    return ContentProjectionSpec.from_params(
        {
            "no_code_blocks": bool(params.get("no_code_blocks")),
            "no_tool_calls": bool(params.get("no_tool_calls")),
            "no_tool_outputs": bool(params.get("no_tool_outputs")),
            "no_file_reads": bool(params.get("no_file_reads")),
            "prose_only": bool(params.get("prose_only")),
        }
    )


def _archive_datetime_to_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp() * 1000)


_SCOPE_FILTER_KEYS = frozenset(
    {
        "session_ids",
        "provider",
        "source_family",
        "source_root",
        "raw_artifact_id",
        "time_range",
        "failure_kind",
        "parser_version",
    }
)


def _parse_scope_filter_body(body: dict[str, Any]) -> dict[str, Any]:
    """Extract scope-filter fields from a maintenance POST body.

    Accepts both a nested ``{"scope": {"filter": {...}}}`` envelope and
    a flat top-level shape (``session_ids`` etc. directly on the
    body). The flat form is the one the CLI's ``--output-format json``
    plan reuses when an operator pipes it back to the daemon, so
    parity with the CLI is what pins the daemon-side parser.
    """

    scope = body.get("scope")
    if isinstance(scope, dict):
        scope_filter = scope.get("filter")
        if isinstance(scope_filter, dict):
            return dict(scope_filter)
    # Fall back to flat keys on the body itself.
    return {key: body[key] for key in _SCOPE_FILTER_KEYS if key in body}


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
        "session_id": insight.session_id,
        "origin": source_name_to_origin(insight.source_name),
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


def _empty_cost_payload(session_id: str, origin: str | None) -> dict[str, object]:
    """Explicit ``unavailable`` payload when no cost insight is materialized (#1122)."""
    return {
        "session_id": session_id,
        "origin": origin,
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


# ---------------------------------------------------------------------------
# Insights browser helpers (#1120)
#
# The insights browser endpoint surfaces the four per-session insight kinds
# (profile, work-event timeline, phases, work threads) as a single JSON
# envelope so the reader inspector can render them inline. Each kind carries
# a readiness chip from the closed vocabulary ``q-ready`` / ``q-partial`` /
# ``q-missing`` driven by:
#
# - ``q-ready``    — the insight is materialized for this session.
# - ``q-missing``  — the insight has no materialized row for this session
#   (the substrate is empty for this scope, not the whole archive).
# - ``q-partial``  — the insight is materialized but the row count is zero
#   (e.g. a session profile exists but has no work events recorded).
#
# The endpoint never imports insight storage modules directly — it routes
# through the same public ``Polylogue`` facade adapters that CLI and MCP use
# (AC#1120, AC#1018).
# ---------------------------------------------------------------------------

INSIGHT_KINDS: tuple[str, ...] = ("profile", "timeline", "phases", "threads")


def _readiness_tag(*, materialized: bool, row_count: int | None = None) -> str:
    """Map a materialized/row-count pair to the readiness chip vocabulary.

    The chip vocabulary is closed (``q-ready`` / ``q-partial`` / ``q-missing``).
    Unknown / unmaterialized rows are ``q-missing``; materialized rows with
    zero downstream rows are ``q-partial`` (the rebuild ran but produced
    nothing); everything else is ``q-ready``.
    """
    if not materialized:
        return "q-missing"
    if row_count is not None and row_count <= 0:
        return "q-partial"
    return "q-ready"


def _parse_insight_includes(raw: str | None) -> tuple[str, ...]:
    """Resolve the ``?include=`` query param into a stable tuple.

    Returns the canonical insight-kind tuple in :data:`INSIGHT_KINDS` order
    when *raw* is None or empty (default = include everything). Unknown
    tokens are dropped; ordering is normalized to :data:`INSIGHT_KINDS`.
    """
    if raw is None or not raw.strip():
        return INSIGHT_KINDS
    requested = {token.strip().lower() for token in raw.split(",") if token.strip()}
    if not requested:
        return INSIGHT_KINDS
    return tuple(kind for kind in INSIGHT_KINDS if kind in requested)


def _provenance_dict(prov: Any) -> dict[str, object]:
    return {
        "materializer_version": int(getattr(prov, "materializer_version", 0)),
        "materialized_at": getattr(prov, "materialized_at", None),
        "source_updated_at": getattr(prov, "source_updated_at", None),
        "source_sort_key": getattr(prov, "source_sort_key", None),
    }


def _profile_staleness(record: Any, session_updated_at: str | None) -> dict[str, object] | None:
    """Compare a session-profile record's provenance against its session.

    Routes through :func:`polylogue.insights.provenance.is_stale` so the
    daemon insights browser (#1018/#1120) consumes the typed staleness
    helper rather than re-deriving the high-water-mark comparison inline.
    Returns ``None`` when the record lacks the provenance fields the
    helper expects.
    """
    from polylogue.insights.provenance import HasProvenance, is_stale

    if record is None:
        return None
    if not all(
        hasattr(record, field)
        for field in ("materialized_at", "materializer_version", "input_high_water_mark", "input_row_count")
    ):
        return None
    verdict = is_stale(
        cast(HasProvenance, record),
        source_high_water_mark=session_updated_at,
    )
    return {
        "stale": verdict.stale,
        "reason": verdict.reason,
        "insight_high_water_mark": verdict.insight_high_water_mark,
        "source_high_water_mark": verdict.source_high_water_mark,
    }


def _profile_panel_payload(profile: Any) -> dict[str, object]:
    """Project a ``SessionProfile`` into the JSON shape served by the reader.

    Uses :meth:`SessionProfile.to_dict` for fidelity to the substrate shape
    and adds a readiness chip + provenance summary on top.
    """
    body = dict(profile.to_dict())
    return {
        "readiness_tag": _readiness_tag(materialized=True, row_count=int(body.get("message_count", 0) or 0)),
        "materialized": True,
        "profile": body,
    }


def _empty_profile_panel_payload() -> dict[str, object]:
    return {
        "readiness_tag": "q-missing",
        "materialized": False,
        "profile": None,
    }


def _work_event_panel_payload(events: list[Any]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for ev in events:
        items.append(
            {
                "event_id": ev.event_id,
                "event_index": int(ev.event_index),
                "session_id": ev.session_id,
                "origin": source_name_to_origin(ev.source_name),
                "evidence": ev.evidence.model_dump(mode="json"),
                "inference": ev.inference.model_dump(mode="json"),
                "provenance": _provenance_dict(ev.provenance),
            }
        )
    return {
        "readiness_tag": _readiness_tag(materialized=bool(events), row_count=len(events)),
        "materialized": bool(events),
        "count": len(items),
        "events": items,
    }


def _phase_panel_payload(phases: list[Any]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for ph in phases:
        items.append(
            {
                "phase_id": ph.phase_id,
                "phase_index": int(ph.phase_index),
                "session_id": ph.session_id,
                "origin": source_name_to_origin(ph.source_name),
                "evidence": ph.evidence.model_dump(mode="json"),
                "inference": ph.inference.model_dump(mode="json"),
                "provenance": _provenance_dict(ph.provenance),
            }
        )
    return {
        "readiness_tag": _readiness_tag(materialized=bool(phases), row_count=len(phases)),
        "materialized": bool(phases),
        "count": len(items),
        "phases": items,
    }


def _thread_panel_payload(threads: list[Any]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for th in threads:
        items.append(
            {
                "thread_id": th.thread_id,
                "root_id": th.root_id,
                "dominant_repo": th.dominant_repo,
                "thread": th.thread.model_dump(mode="json"),
                "provenance": _provenance_dict(th.provenance),
            }
        )
    return {
        "readiness_tag": _readiness_tag(materialized=bool(threads), row_count=len(threads)),
        "materialized": bool(threads),
        "count": len(items),
        "threads": items,
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
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("daemon http client disconnected in %s", fn.__name__)
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
    """Build SessionQuerySpec-compatible params from HTTP query string."""
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

    def _send_text(
        self,
        status: HTTPStatus,
        body: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        """Send a plain-text body with a caller-chosen ``Content-Type``.

        Used by ``/metrics`` (#1321) to emit Prometheus exposition format
        without piggy-backing on JSON or HTML helpers.
        """
        raw = body.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: HTTPStatus, code: str, detail: str | None = None) -> None:
        """Emit the canonical daemon error envelope.

        Every daemon error response shares one machine-output contract
        (#1818): ``{"ok": false, "error": <code>, "detail": <str|null>,
        "field": <str|null>}``, produced by ``QueryErrorPayload`` so the
        decorator path, the cursor-rejection path, and ad-hoc 4xx sites all
        serialize identically. Health/status payloads use a different,
        deliberately separate shape and do not route through here.
        """
        self._send_json(
            status,
            QueryErrorPayload(error=code, detail=detail).model_dump(mode="json"),
        )

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
            or (len(path) == 2 and path[0] == "s" and bool(path[1]))
            or (len(path) == 2 and path[0] == "w" and path[1] in workspace_routes.WORKSPACE_SHELL_MODES)
        ):
            self._serve_web_shell()
            return

        # Paste browser is a standalone reader page (#1201). Served
        # alongside the main web shell, unauthenticated like the main
        # shell because the daemon binds to loopback by default and the
        # page only embeds JS that calls the authenticated archive API.
        if path == ["p"]:
            self._serve_paste_browser_page()
            return

        # Attachment library is a standalone reader page (#1199). Same
        # auth posture as ``/p`` — the page only embeds JS that calls
        # the authenticated archive API.
        if path == ["a"]:
            self._serve_attachment_library_page()
            return

        # Kubernetes-style probes. Unauthenticated by convention — k8s,
        # docker, and systemd healthchecks don't carry credentials, and the
        # probes leak only liveness/readiness booleans plus structured reason
        # codes (no archive data, no environment). Implementation lives in
        # daemon/healthz.py so http.py stays under its file-size budget.
        if path == ["healthz", "live"]:
            from polylogue.daemon.healthz import handle_healthz_live

            handle_healthz_live(self)
            return
        if path == ["healthz", "ready"]:
            from polylogue.daemon.healthz import handle_healthz_ready

            handle_healthz_ready(self)
            return

        # Prometheus scrape endpoint (#1321). Unauthenticated for the same
        # reasons as /healthz/* — scrapers don't carry credentials and the
        # daemon binds to loopback. Series are derived from the archive
        # SQLite database via read-only connections; no archive content
        # is exposed.
        if path == ["metrics"]:
            from polylogue.daemon.metrics import handle_metrics
            from polylogue.paths import active_index_db_path

            handle_metrics(self, active_index_db_path())
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
        elif path == ["api", "sessions"]:
            self._handle_list_sessions(params)
        elif path == ["api", "facets"]:
            self._handle_facets(params)
        elif path == ["api", "query-completions"]:
            self._handle_query_completions(params)
        elif path == ["api", "read-view-profiles"]:
            self._handle_read_view_profiles()
        elif path == ["api", "paste-browser"]:
            self._handle_paste_browser(params)
        elif path == ["api", "attachments"]:
            self._handle_attachment_library(params)
        elif path == ["api", "stack"]:
            self._handle_stack(params)
        elif path == ["api", "compare"]:
            self._handle_compare(params)
        elif workspace_routes.dispatch_get(self, path, params) or (
            path[:2] == ["api", "user"] and user_state_http.dispatch_get(self, path[2:], params)
        ):
            return
        elif path == ["api", "sources"]:
            self._handle_sources()
        elif len(path) == 3 and path[:2] == ["api", "sessions"] and path[2]:
            self._handle_get_session(path[2])
        elif len(path) == 4 and path[:2] == ["api", "sessions"] and path[3] == "messages":
            self._handle_get_messages(path[2], params)
        elif len(path) == 4 and path[:2] == ["api", "sessions"] and path[3] == "raw":
            self._handle_get_session_raw(path[2])
        elif len(path) == 4 and path[:2] == ["api", "sessions"] and path[3] == "cost":
            self._handle_get_session_cost(path[2])
        elif len(path) == 4 and path[:2] == ["api", "sessions"] and path[3] == "provenance":
            self._handle_get_session_provenance(path[2], params)
        elif len(path) == 4 and path[:2] == ["api", "sessions"] and path[3] == "topology":
            self._handle_get_session_topology(path[2], params)
        elif len(path) == 5 and path[:2] == ["api", "sessions"] and path[3] == "topology" and path[4] == "parent-chain":
            self._handle_get_session_parent_chain(path[2], params)
        elif path == ["api", "thread-continue-templates"]:
            self._handle_get_thread_continue_templates()
        elif len(path) == 4 and path[:3] == ["api", "insights", "sessions"]:
            self._handle_get_session_insights(path[3], params)
        elif len(path) == 4 and path[:2] == ["api", "sessions"] and path[3] == "similar":
            self._handle_get_session_similar(path[2], params)
        elif len(path) == 4 and path[:2] == ["api", "sessions"] and path[3] == "attachments":
            self._handle_get_session_attachments(path[2])
        elif len(path) == 4 and path[:3] == ["api", "raw_artifacts"]:
            self._handle_get_raw_artifact(path[3])
        elif path == ["api", "maintenance", "operations"]:
            self._handle_maintenance_operations()
        elif len(path) == 4 and path[:3] == ["api", "maintenance", "status"]:
            self._handle_maintenance_status(path[3])
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    # #1677: client disconnects (refresh, navigate, tab close) surface as
    # BrokenPipeError from wfile.write(). Stdlib lets that escape as a
    # traceback to journal; we demote it to debug — the client has already
    # given up by the time we know.
    _CLIENT_DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

    def do_GET(self) -> None:
        try:
            path, params = self._parse_path()
            self._dispatch_get(path, params)
        except self._CLIENT_DISCONNECT as exc:
            logger.debug("daemon.http.client_disconnected", method="GET", path=self.path, error=repr(exc))

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
        try:
            self._do_post_impl()
        except self._CLIENT_DISCONNECT as exc:
            logger.debug("daemon.http.client_disconnected", method="POST", path=self.path, error=repr(exc))

    def _do_post_impl(self) -> None:
        path, params = self._parse_path()

        # OTLP receiver endpoints (#1321) gated on the explicit
        # ``observability_enabled`` config flag (closes #1604). When
        # the flag is off the routes return 404 so the receiver does
        # not advertise itself to opportunistic scanners. When on AND
        # the daemon is bound non-loopback we still require the
        # configured auth token — OTLP exporters that DO carry
        # credentials are the only safe non-loopback case.
        if path == ["v1", "traces"] or path == ["v1", "metrics"] or path == ["v1", "logs"]:
            from polylogue.config import load_polylogue_config
            from polylogue.core.loopback import is_loopback_host

            if not load_polylogue_config().observability_enabled:
                self._send_error(HTTPStatus.NOT_FOUND, "not_found")
                return
            if not is_loopback_host(self._api_host) and not self._check_auth():
                return
            self._handle_otlp_post(path)
            return

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
        try:
            self._do_delete_impl()
        except self._CLIENT_DISCONNECT as exc:
            logger.debug("daemon.http.client_disconnected", method="DELETE", path=self.path, error=repr(exc))

    def _do_delete_impl(self) -> None:
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

    def _serve_paste_browser_page(self) -> None:
        self._send_html(HTTPStatus.OK, render_paste_browser_page())

    def _serve_attachment_library_page(self) -> None:
        self._send_html(HTTPStatus.OK, render_attachment_library_page())

    @daemon_safe_handler
    def _handle_paste_browser(self, params: dict[str, list[str]]) -> None:
        limit = self._get_int(params, "limit", 200)
        offset = self._get_int(params, "offset", 0)

        async def _run(poly: Polylogue) -> object:
            return await self._do_paste_browser(poly, limit=limit, offset=offset)

        result = self._sync_run(_run)
        self._send_json(HTTPStatus.OK, result)

    async def _do_paste_browser(
        self,
        poly: Polylogue,
        *,
        limit: int,
        offset: int,
    ) -> object:
        # Walk all session summaries and emit one entry per
        # paste-flagged message. The message-level ``has_paste`` flag
        # is the load-bearing signal — we deliberately do not pre-
        # filter on the session-level flag; direct message writes in
        # fixtures or replay paths may be visible before aggregate
        # columns are refreshed.
        convs = await poly.filter().list_summaries()
        entries: list[PasteBrowserEntry] = []
        total_messages_seen = 0
        for summary in convs:
            conv = await poly.get_session(str(summary.id))
            if conv is None:
                continue
            for msg in conv.messages:
                if not bool(getattr(msg, "has_paste", False)):
                    continue
                total_messages_seen += 1
                if total_messages_seen <= offset:
                    continue
                if len(entries) >= limit:
                    break
                text = msg.text or ""
                spans = envelope_paste_spans(text, has_paste=True)
                snippet = snippet_for_paste(text, spans)
                anchor = reader_anchor("message", msg.id)
                entries.append(
                    PasteBrowserEntry(
                        session_id=str(summary.id),
                        session_title=summary.display_title or str(summary.id),
                        origin=summary.origin,
                        message_id=str(msg.id),
                        message_anchor=anchor,
                        role=str(msg.role) if msg.role else "",
                        timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
                        word_count=int(getattr(msg, "word_count", 0) or 0),
                        snippet=snippet,
                        paste_spans=spans,
                        has_diff=any(span.get("kind") == "diff" for span in spans),
                    )
                )
            if len(entries) >= limit:
                break
        return build_paste_browser_payload(entries, total=total_messages_seen)

    # ------------------------------------------------------------------
    # Handlers: attachment library + per-session attachments (#1199)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_attachment_library(self, params: dict[str, list[str]]) -> None:
        limit = self._get_int(params, "limit", 500)
        offset = self._get_int(params, "offset", 0)
        mime_filter = (params.get("mime") or [""])[0]
        state_filter = (params.get("state") or [""])[0]
        session_filter = (params.get("session") or [""])[0]

        async def _run(poly: Polylogue) -> object:
            return await self._do_attachment_library(
                poly,
                limit=limit,
                offset=offset,
                mime_filter=mime_filter,
                state_filter=state_filter,
                session_filter=session_filter,
            )

        result = self._sync_run(_run)
        self._send_json(HTTPStatus.OK, result)

    async def _do_attachment_library(
        self,
        poly: Polylogue,
        *,
        limit: int,
        offset: int,
        mime_filter: str,
        state_filter: str,
        session_filter: str,
    ) -> object:
        # Walk all session summaries and emit one library entry per
        # attachment. We pre-filter by session id when supplied so
        # the walk stops early; mime/state filters apply after
        # envelope construction (state is derived, not stored).
        summaries = await poly.filter().list_summaries()
        entries: list[LibraryEntry] = []
        total_seen = 0
        for summary in summaries:
            sid = str(summary.id)
            if session_filter and session_filter != sid:
                continue
            conv = await poly.get_session(sid)
            if conv is None:
                continue
            origin = summary.origin
            title = summary.display_title or sid
            for msg in conv.messages:
                for att in msg.attachments or []:
                    envelope = attachment_to_envelope(att, session_id=sid, message_id=msg.id)
                    mime_value = envelope.get("mime_type")
                    if mime_filter and mime_filter not in (mime_value if isinstance(mime_value, str) else ""):
                        continue
                    if state_filter and envelope.get("state") != state_filter:
                        continue
                    total_seen += 1
                    if total_seen <= offset:
                        continue
                    if len(entries) >= limit:
                        break
                    entries.append(
                        LibraryEntry(
                            envelope=envelope,
                            session_title=title,
                            origin=origin,
                            message_anchor=reader_anchor("message", msg.id) if msg.id else None,
                        )
                    )
                if len(entries) >= limit:
                    break
            if len(entries) >= limit:
                break
        return build_library_payload(entries, total=total_seen)

    @daemon_safe_handler
    def _handle_get_session_attachments(self, conv_id: str) -> None:
        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_attachments(poly, conv_id)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_session_attachments(self, poly: Polylogue, conv_id: str) -> object:
        conv = await poly.get_session(conv_id)
        if conv is None:
            return None
        items: list[dict[str, object]] = []
        for msg in conv.messages:
            for att in msg.attachments or []:
                items.append(attachment_to_envelope(att, session_id=str(conv.id), message_id=msg.id))
        return {"items": items, "total": len(items)}

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
        from polylogue.paths import active_index_db_path

        dbp = active_index_db_path()
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
        status = get_status_snapshot_payload()
        if isinstance(status, dict):
            status["last_event_id"] = latest_event_id
            with contextlib.suppress(Exception):
                from polylogue.daemon.status import _check_daemon_liveness

                status["daemon_liveness"] = _check_daemon_liveness()
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
    # Handlers: list sessions
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_list_sessions(self, params: dict[str, list[str]]) -> None:
        from polylogue.archive.query.spec import clamp_query_limit

        query_params = _build_query_spec_params(params, self)
        # Clamp to the shared MAX_QUERY_LIMIT ceiling so the daemon honors the
        # same page-size cap as MCP instead of an arbitrary ?limit=99999999
        # (#1749). The default stays 50; clamp_query_limit only caps the top.
        limit = clamp_query_limit(self._get_int(params, "limit", 50), default=50)
        offset = max(0, self._get_int(params, "offset", 0))
        cursor_values = params.get("cursor") or []
        cursor = cursor_values[0] if cursor_values else None
        if cursor:
            query_params["cursor"] = cursor

        archive_root = _web_reader_archive_root()
        if archive_root is not None:
            self._send_json(HTTPStatus.OK, self._do_archive_list_sessions(archive_root, params, limit, offset))
            return

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
        from polylogue.archive.query.expression import compile_expression_into
        from polylogue.archive.query.spec import SessionQuerySpec

        # Build the flag-derived base spec (all params except the free-text
        # query string), then route the query string through the shared
        # expression parser/lowerer so structured clauses like
        # ``origin:codex has:paste since:7d`` resolve to the correct spec
        # fields rather than being passed as literal FTS terms (#1860).
        query_str = str(query_params.get("query") or "").strip()
        base_params = {k: v for k, v in query_params.items() if k != "query"}
        base = SessionQuerySpec.from_params({**base_params, "limit": limit, "offset": offset})
        spec = compile_expression_into(query_str, base) if query_str else base

        # Route to the ranked search path when the compiled spec carries FTS
        # or vector terms; use the plain list path otherwise (including for
        # pure-DSL queries whose clauses only set structured fields with no
        # FTS text, e.g. ``origin:codex has:paste``).
        if spec.query_terms or spec.contains_terms:
            return await self._do_search_list(poly, spec, limit, offset)

        # A pure vector-only request (similar_text, no FTS term) must surface
        # the same typed EmbeddingRetrievalNotReadyError the CLI and MCP give,
        # not an opaque 500. Routing through the operations layer resolves the
        # vector provider (raising the typed readiness error when embeddings
        # are not ready), which daemon_safe_handler maps to its 409 status
        # instead of falling through to a generic ValueError (#1749).
        if spec.similar_text:
            return await self._do_search_list(poly, spec, limit, offset)

        filter_obj = spec.build_filter(poly.config)
        summaries = await filter_obj.list_summaries()
        total = await spec.count(poly.config)

        diagnostics = None
        if not summaries and spec.has_filters():
            with contextlib.suppress(ImportError):
                from polylogue.config import ConfigError

                try:
                    raw_diag = await poly.diagnose_query_miss(spec)
                    diagnostics = QueryMissDiagnosticsPayload.from_diagnostics(raw_diag)
                except ConfigError:
                    pass

        items: list[dict[str, object]] = []
        for summary in summaries:
            flags = _build_flags_from_session(summary)
            session_id = str(summary.id)
            target_ref = TargetRefPayload.session(session_id)
            row: dict[str, object] = {
                "id": session_id,
                "title": summary.display_title,
                "origin": summary.origin,
                "target_ref": _dump_target_ref(target_ref),
                "anchor": reader_anchor("session", session_id),
                "actions": _dump_actions(reader_session_actions()),
                "date": summary.display_date.isoformat() if summary.display_date else None,
                "created_at": summary.created_at.isoformat() if summary.created_at else None,
                "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
                "message_count": getattr(summary, "message_count", 0) or 0,
                "word_count": getattr(summary, "word_count", None),
                "repo": getattr(summary, "git_repository_url", None),
                "cwd_display": next(iter(getattr(summary, "working_directories", ()) or ()), None),
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
        spec: SessionQuerySpec,
        limit: int,
        offset: int,
    ) -> object:
        """Return the canonical :class:`SearchEnvelope` for ranked queries.

        The same envelope ships across CLI JSON, MCP, Python API, and daemon
        HTTP (#1266). Construction goes through the shared spec builder so
        the cursor / next_offset / ranking-policy fields stay aligned with
        the other surfaces. When ``cursor`` is supplied the response page
        starts strictly after the anchor (#1268).
        """
        from polylogue.api.search_envelope_builder import build_search_envelope_for_spec
        from polylogue.surfaces.payloads import InvalidSearchCursorError

        try:
            envelope = await build_search_envelope_for_spec(
                poly,
                spec,
                limit=limit,
                offset=offset,
            )
        except InvalidSearchCursorError as exc:
            return QueryErrorPayload(error="invalid_cursor", detail=str(exc)).model_dump(mode="json")
        return envelope.model_dump(mode="json")

    def _do_archive_list_sessions(
        self,
        archive_root: Path,
        params: dict[str, list[str]],
        limit: int,
        offset: int,
    ) -> object:
        from polylogue.archive.query.expression import compile_expression
        from polylogue.archive.query.search_hits import search_query_text
        from polylogue.archive.query.spec import QuerySpecError, parse_query_date
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        # Parse/lower only the ``query`` param through the shared expression path
        # so DSL clauses like ``origin:codex has:paste since:7d`` map to the
        # correct ArchiveStore filter arguments rather than being passed as
        # literal FTS text (#1860).  The ``contains`` param is a legacy
        # content-substring filter that must NOT be parsed as DSL — routing it through
        # compile_expression() causes ExpressionCompileError when the value
        # contains spaces or field-like tokens (#1873 Bug 7).
        query_str = self._get_param(params, "query") or ""
        contains_param = self._get_param(params, "contains")
        fts_terms: tuple[str, ...]
        if query_str:
            spec = compile_expression(query_str)
            # Bare words / quoted phrases become the FTS query; field clauses
            # (origin:, tag:, since:, etc.) become structured filter args.
            fts_terms = spec.query_terms + spec.contains_terms
        else:
            spec = None
            fts_terms = ()
        # ``?contains=`` is the legacy literal content filter. It must still
        # filter results, but must NOT be compiled as DSL (a value with spaces or
        # field-like tokens would raise ExpressionCompileError, #1873 Bug 7). Wire
        # it as a literal FTS term so ``GET /api/sessions?contains=foo`` with no
        # ``query`` filters instead of returning the unfiltered first page.
        if contains_param:
            fts_terms = fts_terms + (contains_param,)
        fts_query = search_query_text(fts_terms)

        # HTTP-param-based origin/tag filters — still honoured for backwards
        # compatibility with callers passing ``?origin=...`` or ``?tag=...``
        # as separate query-string parameters.
        _, http_origins = _archive_origin_filter(params)
        http_tags = _archive_tag_filter(params)

        # Merge DSL-compiled spec filters with HTTP-param-based filters.
        origins = tuple(dict.fromkeys((spec.origins if spec else ()) + http_origins))
        excluded_origins = tuple(
            dict.fromkeys(
                (spec.excluded_origins if spec else ())
                + _csv_values(params, "exclude_origin")
                + _csv_values(params, "exclude_provider")
            )
        )
        tags = tuple(dict.fromkeys((spec.tags if spec else ()) + http_tags))
        excluded_tags = tuple(dict.fromkeys((spec.excluded_tags if spec else ()) + _csv_values(params, "exclude_tag")))
        origin = origins[0] if len(origins) == 1 else None

        # Date filters: DSL spec takes precedence over bare HTTP params.
        since_str = (spec.since if spec else None) or self._get_param(params, "since")
        until_str = (spec.until if spec else None) or self._get_param(params, "until")
        # parse_query_date raises QuerySpecError (a PolylogueError carrying
        # http_status_code=400) for unparseable dates; daemon_safe_handler maps
        # it to the QueryErrorPayload-shaped 400 the other surfaces return.
        since_dt = parse_query_date("since", since_str)
        until_dt = parse_query_date("until", until_str)
        since_ms = _archive_datetime_to_ms(since_dt)
        until_ms = _archive_datetime_to_ms(until_dt)

        # Remaining structured filters come from the compiled spec plus the
        # stable HTTP query parameters accepted by the shared query builder.
        # The split-archive fast path does not construct a SessionQuerySpec
        # directly, so it must mirror those public params here.
        has_paste = (spec.filter_has_paste if spec else False) or self._get_bool(params, "has_paste")
        has_tool_use = (spec.filter_has_tool_use if spec else False) or self._get_bool(params, "has_tool_use")
        has_thinking = (spec.filter_has_thinking if spec else False) or self._get_bool(params, "has_thinking")
        repo_names = tuple(dict.fromkeys((spec.repo_names if spec else ()) + _csv_values(params, "repo")))
        has_types = tuple(dict.fromkeys((spec.has_types if spec else ()) + _csv_values(params, "has_type")))
        tool_terms = tuple(dict.fromkeys((spec.tool_terms if spec else ()) + _csv_values(params, "tool")))
        excluded_tool_terms = tuple(
            dict.fromkeys((spec.excluded_tool_terms if spec else ()) + _csv_values(params, "exclude_tool"))
        )
        action_terms = tuple(dict.fromkeys((spec.action_terms if spec else ()) + _csv_values(params, "action")))
        excluded_action_terms = tuple(
            dict.fromkeys((spec.excluded_action_terms if spec else ()) + _csv_values(params, "exclude_action"))
        )
        action_sequence = tuple(
            dict.fromkeys((spec.action_sequence if spec else ()) + _csv_values(params, "action_sequence"))
        )
        action_text_terms = tuple(
            dict.fromkeys((spec.action_text_terms if spec else ()) + _csv_values(params, "action_text"))
        )
        referenced_paths = tuple(
            dict.fromkeys((spec.referenced_path if spec else ()) + _csv_values(params, "referenced_path"))
        )
        cwd_prefix = (spec.cwd_prefix if spec else None) or self._get_param(params, "cwd_prefix")
        title = (spec.title if spec else None) or self._get_param(params, "title")
        min_messages = (spec.min_messages if spec else None) or self._get_int(params, "min_messages", 0) or None
        max_messages = (spec.max_messages if spec else None) or self._get_int(params, "max_messages", 0) or None
        min_words = (spec.min_words if spec else None) or self._get_int(params, "min_words", 0) or None
        max_words = None
        # session_id is passed separately to list_summaries/search_summaries
        # (count_sessions does not accept this param, so it cannot go in _filter_kw).
        spec_session_id = spec.session_id if spec else None

        # Shared filter kwargs forwarded to every ArchiveStore call.
        _filter_kw: dict[str, object] = {
            "origin": origin,
            "origins": origins,
            "excluded_origins": excluded_origins,
            "tags": tags,
            "excluded_tags": excluded_tags,
            "repo_names": repo_names,
            "has_types": has_types,
            "has_tool_use": has_tool_use,
            "has_thinking": has_thinking,
            "has_paste": has_paste,
            "tool_terms": tool_terms,
            "excluded_tool_terms": excluded_tool_terms,
            "action_terms": action_terms,
            "excluded_action_terms": excluded_action_terms,
            "action_sequence": action_sequence,
            "action_text_terms": action_text_terms,
            "referenced_paths": referenced_paths,
            "cwd_prefix": cwd_prefix,
            "title": title,
            "min_messages": min_messages,
            "max_messages": max_messages,
            "min_words": min_words,
            "max_words": max_words,
            "since_ms": since_ms,
            "until_ms": until_ms,
        }

        with ArchiveStore.open_existing(archive_root) as archive:
            # Resolve an ``id:`` clause once, up front, so both branches share one
            # miss/ambiguous policy. list_summaries/search_summaries resolve the
            # token internally and raise KeyError (miss) / ValueError (ambiguous);
            # left unhandled those reach the safe handler as a 500. A miss is a
            # typed-empty page and an ambiguous prefix is a 400 query-spec error,
            # matching the other query surfaces.
            resolved_session_id = spec_session_id
            if spec_session_id is not None:
                try:
                    resolved_session_id = archive.resolve_session_id(spec_session_id)
                except KeyError:
                    if fts_query:
                        return {
                            "query": fts_query,
                            "retrieval_lane": "dialogue",
                            "ranking_policy": "mixed-bm25-rrf-vector",
                            "ranking_policy_version": "1",
                            "hits": [],
                            "total": 0,
                            "limit": limit,
                            "offset": offset,
                        }
                    return {"items": [], "total": 0, "limit": limit, "offset": offset}
                except ValueError as exc:
                    raise QuerySpecError("id", spec_session_id) from exc
            if fts_query:
                hits = archive.search_summaries(
                    fts_query,
                    limit=limit,
                    offset=offset,
                    session_id=resolved_session_id,
                    **_filter_kw,  # type: ignore[arg-type]
                )
                payload: dict[str, object] = {
                    "query": fts_query,
                    "retrieval_lane": "dialogue",
                    "ranking_policy": "mixed-bm25-rrf-vector",
                    "ranking_policy_version": "1",
                    "hits": [self._archive_search_hit_payload(hit) for hit in hits],
                    "total": len(hits),
                    "limit": limit,
                    "offset": offset,
                }
                if not hits:
                    # Zero-result query: attach a diagnostics envelope (matching
                    # the archive reader contract) so the surface can explain the
                    # miss instead of rendering a bare empty list.
                    from polylogue.surfaces.payloads import QueryMissDiagnosticsPayload

                    archive_count = archive.count_sessions(**_filter_kw)  # type: ignore[arg-type]
                    filters = tuple(
                        label
                        for label in (
                            f"query={fts_query!r}",
                            f"origin={origin}" if origin else None,
                            f"tags={list(tags)}" if tags else None,
                        )
                        if label is not None
                    )
                    payload["diagnostics"] = QueryMissDiagnosticsPayload(
                        message=f"No sessions matched {fts_query!r}.",
                        filters=filters,
                        reasons=(),
                        archive_session_count=archive_count,
                    ).model_dump(mode="json", by_alias=True)
                return payload
            summaries = archive.list_summaries(
                limit=limit,
                offset=offset,
                session_id=resolved_session_id,
                **_filter_kw,  # type: ignore[arg-type]
            )
            # count_sessions has no session_id param, so when the page is scoped to
            # a single resolved id an archive-wide total would be reported for a
            # one-session match. An id matches at most one session, so the scoped
            # total is the page length.
            total = (
                len(summaries) if resolved_session_id is not None else archive.count_sessions(**_filter_kw)  # type: ignore[arg-type]
            )
            return {
                "items": [self._archive_summary_payload(summary) for summary in summaries],
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    def _archive_summary_payload(self, summary: ArchiveSessionSummary) -> dict[str, object]:
        session_id = str(summary.session_id)
        target_ref = TargetRefPayload.session(session_id)
        return {
            "id": session_id,
            "session_id": session_id,
            "title": summary.title or session_id,
            "origin": summary.origin,
            "target_ref": _dump_target_ref(target_ref),
            "anchor": reader_anchor("session", session_id),
            "actions": _dump_actions(reader_session_actions()),
            "date": summary.updated_at or summary.created_at,
            "created_at": summary.created_at,
            "updated_at": summary.updated_at,
            "message_count": summary.message_count,
            "word_count": summary.word_count,
            "repo": None,
            "cwd_display": None,
            "tags": list(summary.tags),
            "flags": None,
            "summary": None,
        }

    def _archive_search_hit_payload(self, hit: ArchiveSessionSearchHit) -> dict[str, object]:
        session_id = str(hit.session_id)
        message_id = str(hit.message_id)
        session_ref = TargetRefPayload.session(session_id)
        message_ref = TargetRefPayload.message(session_id=session_id, message_id=message_id)
        return {
            "rank": hit.rank,
            "session": {
                "id": session_id,
                "title": hit.title or session_id,
                "origin": hit.origin,
                "target_ref": _dump_target_ref(session_ref),
                "anchor": reader_anchor("session", session_id),
                "actions": _dump_actions(reader_session_actions()),
            },
            "match": {
                "message_id": message_id,
                "block_id": hit.block_id,
                "snippet": hit.snippet,
                "target_ref": _dump_target_ref(message_ref),
                "anchor": reader_anchor("message", message_id),
                "actions": _dump_actions(reader_message_actions()),
            },
        }

    # ------------------------------------------------------------------
    # Handlers: get session
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session(self, conv_id: str) -> None:
        archive_root = _web_reader_archive_root()
        if archive_root is not None:
            result = self._do_archive_get_session(archive_root, conv_id)
            if result is None:
                self._send_error(HTTPStatus.NOT_FOUND, "not_found")
                return
            self._send_json(HTTPStatus.OK, result)
            return

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session(poly, conv_id)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_session(self, poly: Polylogue, conv_id: str) -> object:
        conv = await poly.get_session(conv_id)
        if conv is None:
            return None
        flags = _build_flags_from_session(conv)
        session_id = str(conv.id)
        target_ref = TargetRefPayload.session(session_id)
        # Flatten attachments across all messages so the inspector
        # tab and the session envelope share one source of truth
        # (#1199). Per-message attachments stay embedded in each
        # message envelope so the inline card renderer doesn't need
        # to cross-reference the session-level list.
        session_attachments: list[dict[str, object]] = []
        for msg in conv.messages:
            for att in msg.attachments or []:
                session_attachments.append(attachment_to_envelope(att, session_id=session_id, message_id=msg.id))
        return {
            "id": session_id,
            "title": conv.title,
            "display_title": conv.display_title,
            "origin": conv.origin,
            "target_ref": _dump_target_ref(target_ref),
            "anchor": reader_anchor("session", session_id),
            "actions": _dump_actions(reader_session_actions()),
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "message_count": len(conv.messages),
            "word_count": conv.word_count,
            "messages": [
                {
                    "id": str(msg.id),
                    "role": str(msg.role),
                    "text": msg.text,
                    "target_ref": _dump_target_ref(TargetRefPayload.message(session_id=session_id, message_id=msg.id)),
                    "anchor": reader_anchor("message", msg.id),
                    "actions": _dump_actions(reader_message_actions()),
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "message_type": _message_type_value(msg),
                    "word_count": msg.word_count,
                    "has_tool_use": bool(msg.has_tool_use) if hasattr(msg, "has_tool_use") else False,
                    "has_thinking": bool(msg.has_thinking) if hasattr(msg, "has_thinking") else False,
                    "has_paste": bool(msg.has_paste) if hasattr(msg, "has_paste") else False,
                    "paste_spans": envelope_paste_spans(
                        msg.text,
                        has_paste=bool(msg.has_paste) if hasattr(msg, "has_paste") else False,
                    ),
                    "attachments": [
                        attachment_to_envelope(att, session_id=session_id, message_id=msg.id)
                        for att in (msg.attachments or [])
                    ],
                }
                for msg in conv.messages
            ],
            "attachments": session_attachments,
            "tags": conv.tags,
            "branch_type": str(conv.branch_type) if conv.branch_type else None,
            "parent_id": str(conv.parent_id) if conv.parent_id else None,
            "session_id": getattr(conv, "session_id", None),
            "repo": getattr(conv, "git_repository_url", None),
            "cwd_display": next(iter(getattr(conv, "working_directories", ()) or ()), None),
            "model": None,
            "flags": flags.model_dump(mode="json") if flags else None,
            "summary": conv.summary,
            "total": len(conv.messages),
        }

    def _do_archive_get_session(self, archive_root: Path, conv_id: str) -> object | None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(archive_root) as archive:
            try:
                session_id = archive.resolve_session_id(conv_id)
                envelope = archive.read_session(session_id)
                summary = archive.read_summary(session_id)
            except KeyError:
                return None

        session_id = envelope.session_id
        created_at = summary.created_at
        updated_at = summary.updated_at
        word_count = summary.word_count
        target_ref = TargetRefPayload.session(session_id)
        messages = [self._archive_message_payload(session_id, message) for message in envelope.messages]
        # Flatten per-message attachments plus session-level orphan attachments
        # into one session-level list, mirroring the archive detail handler
        # so the inspector tab and the session envelope share one source of
        # truth (#1199).
        from polylogue.api.archive import _archive_attachment_to_domain

        session_attachments: list[dict[str, object]] = []
        for message_payload in messages:
            session_attachments.extend(cast("list[dict[str, object]]", message_payload["attachments"]))
        session_attachments.extend(
            attachment_to_envelope(
                _archive_attachment_to_domain(att),
                session_id=session_id,
                message_id=att.message_id,
            )
            for att in envelope.orphan_attachments
        )
        return {
            "id": session_id,
            "session_id": session_id,
            "title": envelope.title,
            "display_title": envelope.title or session_id,
            "origin": envelope.origin,
            "target_ref": _dump_target_ref(target_ref),
            "anchor": reader_anchor("session", session_id),
            "actions": _dump_actions(reader_session_actions()),
            "created_at": created_at,
            "updated_at": updated_at,
            "message_count": len(messages),
            "word_count": word_count,
            "messages": messages,
            "attachments": session_attachments,
            "tags": list(summary.tags),
            "branch_type": None,
            "parent_id": None,
            "repo": None,
            "cwd_display": None,
            "model": None,
            "flags": None,
            "summary": None,
            "total": len(messages),
        }

    def _archive_message_attachments(self, session_id: str, message: ArchiveMessageRow) -> list[dict[str, object]]:
        from polylogue.api.archive import _archive_attachment_to_domain

        return [
            attachment_to_envelope(
                _archive_attachment_to_domain(att),
                session_id=session_id,
                message_id=str(message.message_id),
            )
            for att in message.attachments
        ]

    def _project_text_block(self, text: str | None, projection: ContentProjectionSpec) -> str | None:
        if text is None:
            return None
        if (
            projection.include_prose
            and projection.include_code
            and projection.include_reasoning
            and projection.include_system_noise
        ):
            return text

        from polylogue.archive.message.models import Message
        from polylogue.archive.message.roles import Role
        from polylogue.archive.semantic.content_projection import project_message_content

        projected = project_message_content(
            [Message(id="archive-block", role=Role.ASSISTANT, text=text, blocks=[])],
            projection,
        )
        if not projected:
            return None
        return projected[0].text

    def _project_archive_message(
        self,
        message: ArchiveMessageRow,
        projection: ContentProjectionSpec,
    ) -> ArchiveMessageRow | None:
        if projection.is_default():
            return message
        tool_semantics = {
            block.tool_id: block.semantic_type
            for block in message.blocks
            if block.block_type == "tool_use" and block.tool_id and block.semantic_type
        }
        blocks: list[ArchiveBlockRow] = []
        for block in message.blocks:
            if block.block_type == "thinking" and not projection.include_reasoning:
                continue
            if block.block_type == "code" and not projection.include_code:
                continue
            if block.block_type == "tool_use" and not projection.include_tool_calls:
                continue
            if block.block_type == "tool_result":
                semantic_type = tool_semantics.get(block.tool_id or "", block.semantic_type or "")
                if semantic_type == "file_read" and not (
                    projection.include_file_reads and projection.include_tool_outputs
                ):
                    continue
                if semantic_type != "file_read" and not projection.include_tool_outputs:
                    continue
            if block.block_type in {"image", "document", "file"} and not projection.include_attachments:
                continue
            if block.block_type == "text":
                text = self._project_text_block(block.text, projection)
                if text is None:
                    continue
                blocks.append(replace(block, text=text))
                continue
            if (
                block.block_type not in {"thinking", "code", "tool_use", "tool_result", "image", "document", "file"}
                and not projection.include_prose
            ):
                continue
            blocks.append(block)
        if not blocks and projection.filters_content():
            return None
        return replace(message, blocks=tuple(blocks))

    def _archive_message_payload(self, session_id: str, message: ArchiveMessageRow) -> dict[str, object]:
        message_id = str(message.message_id)
        text = "\n\n".join(str(block.text) for block in message.blocks if block.text)
        has_paste = bool(message.has_paste)
        return {
            "id": message_id,
            "role": str(message.role),
            "text": text,
            "target_ref": _dump_target_ref(TargetRefPayload.message(session_id=session_id, message_id=message_id)),
            "anchor": reader_anchor("message", message_id),
            "actions": _dump_actions(reader_message_actions()),
            "timestamp": message.occurred_at,
            "message_type": message.message_type,
            "word_count": message.word_count,
            "has_tool_use": bool(message.has_tool_use),
            "has_thinking": bool(message.has_thinking),
            "has_paste": has_paste,
            "paste_spans": envelope_paste_spans(text, has_paste=has_paste),
            "attachments": self._archive_message_attachments(session_id, message),
        }

    # ------------------------------------------------------------------
    # Handlers: get session raw
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_raw(self, conv_id: str) -> None:
        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_raw(poly, conv_id)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_session_raw(self, poly: Polylogue, conv_id: str) -> object:
        conv = await poly.get_session(conv_id)
        if conv is None:
            return None
        raw_items = await poly.get_raw_artifacts_for_session(conv_id)
        return {
            "id": str(conv.id),
            "origin": conv.origin,
            "title": conv.display_title,
            "working_directories": list(getattr(conv, "working_directories", ()) or ()),
            "git_branch": getattr(conv, "git_branch", None),
            "git_repository_url": getattr(conv, "git_repository_url", None),
            "branch_type": str(conv.branch_type) if conv.branch_type else None,
            "parent_id": str(conv.parent_id) if conv.parent_id else None,
            "session_id": getattr(conv, "session_id", None),
            "raw_artifacts": raw_items,
        }

    # ------------------------------------------------------------------
    # Handlers: get session cost (#1122)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_cost(self, conv_id: str) -> None:
        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_cost(poly, conv_id)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_session_cost(self, poly: Polylogue, conv_id: str) -> object:
        from polylogue.insights.archive import SessionCostInsightQuery

        insights = await poly.list_session_cost_insights(SessionCostInsightQuery(session_id=conv_id))
        if not insights:
            # No matching session-cost insight: confirm the session exists
            # so we can distinguish "unknown session" (404) from "cost
            # surface unavailable" (200 with explicit unavailable shape).
            conv = await poly.get_session(conv_id)
            if conv is None:
                return None
            return _empty_cost_payload(conv_id, conv.origin)
        return _cost_panel_payload(insights[0])

    # ------------------------------------------------------------------
    # Handlers: insights browser (#1120)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_insights(self, conv_id: str, params: dict[str, list[str]]) -> None:
        """``GET /api/insights/sessions/{id}?include=profile,timeline,phases,threads``.

        Returns a single typed envelope joining the four per-session insight
        kinds for *conv_id* — session profile (#1018), work-event timeline
        (#1133/#1135), phases, and work threads. Each section carries a
        readiness chip drawn from the closed vocabulary
        (``q-ready`` / ``q-partial`` / ``q-missing``).

        Unknown sessions return ``404 not_found``. Existing sessions
        without materialized insights return ``200`` with explicit ``q-missing``
        per-kind shapes so the panel is never blank (AC#1120).
        """
        include_raw = self._get_param(params, "include")
        includes = _parse_insight_includes(include_raw)

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_insights(poly, conv_id, includes)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_session_insights(
        self,
        poly: Polylogue,
        conv_id: str,
        includes: tuple[str, ...],
    ) -> object:
        from polylogue.insights.archive import (
            ArchiveInsightUnavailableError,
            SessionPhaseInsightQuery,
            SessionWorkEventInsightQuery,
            ThreadInsightQuery,
        )

        # Confirm the session exists first: distinguishes "unknown
        # session" (404) from "insights surface unavailable" (200 with
        # explicit q-missing shapes).
        conv = await poly.get_session(conv_id)
        if conv is None:
            return None

        envelope: dict[str, object] = {
            "session_id": conv_id,
            "origin": conv.origin,
            "include": list(includes),
            "kinds": {},
        }
        kinds = envelope["kinds"]
        assert isinstance(kinds, dict)

        if "profile" in includes:
            from polylogue.storage.insights.session.profiles import hydrate_session_profile

            try:
                # Archive read returns the full record directly; hydrate it into
                # the domain ``SessionProfile`` for the panel projection. Native
                # returns ``None`` (rather than raising) when the profile is not
                # materialized, so the except below is defensive only.
                profile_record = await poly.get_session_profile_record(conv_id)
            except ArchiveInsightUnavailableError:
                # The substrate hasn't materialized this insight kind yet;
                # surface q-missing rather than 503 the whole envelope.
                profile_record = None
            profile = hydrate_session_profile(profile_record) if profile_record is not None else None
            panel = _profile_panel_payload(profile) if profile is not None else _empty_profile_panel_payload()
            # Compare the materialized record's provenance against the
            # session's current ``updated_at`` via the typed
            # :func:`polylogue.insights.provenance.is_stale` helper so the
            # reader sees explicit staleness, not just q-ready/q-missing
            # presence chips.
            if profile is not None:
                conv_updated_at = conv.updated_at.isoformat() if conv.updated_at else None
                staleness = _profile_staleness(profile_record, conv_updated_at)
                if staleness is not None:
                    panel["staleness"] = staleness
            kinds["profile"] = panel

        if "timeline" in includes:
            try:
                # Per-session list adapter — same path as MCP `session_work_events`.
                events = await poly.list_session_work_event_insights(
                    SessionWorkEventInsightQuery(session_id=conv_id, limit=None)
                )
            except ArchiveInsightUnavailableError:
                events = []
            kinds["timeline"] = _work_event_panel_payload(events)

        if "phases" in includes:
            try:
                phases = await poly.list_session_phase_insights(
                    SessionPhaseInsightQuery(session_id=conv_id, limit=None)
                )
            except ArchiveInsightUnavailableError:
                phases = []
            kinds["phases"] = _phase_panel_payload(phases)

        if "threads" in includes:
            try:
                # Work threads are not keyed per-session in the substrate;
                # the reader filters the materialized rows by membership.
                all_threads = await poly.list_thread_insights(ThreadInsightQuery(limit=None))
            except ArchiveInsightUnavailableError:
                all_threads = []
            member_threads = [th for th in all_threads if conv_id in (th.thread.session_ids or ())]
            kinds["threads"] = _thread_panel_payload(member_threads)

        return envelope

    # ------------------------------------------------------------------
    # Handlers: per-session provenance (#1125)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_provenance(
        self,
        conv_id: str,
        params: dict[str, list[str]],
    ) -> None:
        """``GET /api/sessions/{id}/provenance[?include_raw=1[&bytes=N]]``.

        Returns the source artifact metadata that produced *conv_id*.
        The raw payload preview is opt-in (``include_raw=1``) and is
        bounded server-side by
        :data:`polylogue.daemon.provenance.RAW_PREVIEW_MAX_BYTES` —
        client-supplied ``bytes`` only narrows the window, never widens
        it.
        """
        from polylogue.daemon.provenance import build_provenance_payload

        include_raw = self._get_bool(params, "include_raw")
        requested_bytes: int | None = None
        raw_bytes_param = self._get_param(params, "bytes")
        if raw_bytes_param is not None:
            try:
                requested_bytes = int(raw_bytes_param)
            except (TypeError, ValueError):
                requested_bytes = None

        payload = build_provenance_payload(
            conv_id,
            include_raw=include_raw,
            requested_bytes=requested_bytes,
        )
        if payload is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, payload)

    # ------------------------------------------------------------------
    # Handlers: per-session topology (#1121)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_topology(
        self,
        conv_id: str,
        params: dict[str, list[str]],
    ) -> None:
        """``GET /api/sessions/{id}/topology[?limit=N]``.

        Returns a bounded :class:`polylogue.insights.topology.SessionTopology`
        envelope rooted at *conv_id*'s lineage root. ``?limit=`` is the
        operator-visible knob; the daemon enforces the hard cap from
        :data:`polylogue.daemon.topology_http.MAX_NODE_LIMIT` regardless of
        client input (#1121 AC: lineage rendering is bounded).
        """
        from polylogue.daemon.topology_http import (
            build_topology_envelope,
            coerce_node_limit,
        )

        node_limit = coerce_node_limit(self._get_param(params, "limit"))
        if node_limit is None:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_limit")
            return

        async def _get(poly: Polylogue) -> object:
            topology = await poly.get_session_topology(conv_id)
            if topology is None:
                return None
            return build_topology_envelope(topology, node_limit=node_limit)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    # ------------------------------------------------------------------
    # Handlers: parent-chain stack envelope + thread-continue templates (#1203)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_parent_chain(
        self,
        conv_id: str,
        params: dict[str, list[str]],
    ) -> None:
        """``GET /api/sessions/{id}/topology/parent-chain``.

        Returns the stack-ready chain envelope shaped by
        :func:`polylogue.daemon.topology_http.build_parent_chain_envelope`.
        The envelope's ``chain_ids`` seed the stack workspace route
        (``/w/stack?ids=...``); ``focus_id`` keeps the operator anchored
        at the session they invoked the action from.

        Query parameters:
        - ``descendants=0`` — omit descendant sessions and return
          only the ancestor chain (root → target).
        """
        from polylogue.daemon.topology_http import build_parent_chain_envelope

        include_descendants_raw = self._get_param(params, "descendants", "1") or "1"
        include_descendants = include_descendants_raw not in ("0", "false", "no")

        async def _get(poly: Polylogue) -> object:
            topology = await poly.get_session_topology(conv_id)
            if topology is None:
                return None
            return build_parent_chain_envelope(topology, include_descendants=include_descendants)

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    @daemon_safe_handler
    def _handle_get_thread_continue_templates(self) -> None:
        """``GET /api/thread-continue-templates``.

        Returns the active agent URL-template registry. Templates are
        substituted client-side so the daemon never sees the messages
        the operator is "continuing" in another agent.
        """
        from polylogue.daemon.thread_continue import build_templates_envelope

        self._send_json(HTTPStatus.OK, build_templates_envelope())

    @daemon_safe_handler
    def _handle_query_completions(self, params: dict[str, list[str]]) -> None:
        """``GET /api/query-completions`` exposes shared query metadata."""

        from polylogue.archive.query.completions import QueryCompletionError, query_completion_payload

        kind = self._get_param(params, "kind") or "field"
        incomplete = self._get_param(params, "incomplete") or ""
        unit = self._get_param(params, "unit")
        field = self._get_param(params, "field")
        try:
            payload = query_completion_payload(kind, incomplete=incomplete, unit=unit, field=field)
        except QueryCompletionError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_query_completion", "message": str(exc)})
            return
        self._send_json(HTTPStatus.OK, {"query_completions": payload})

    @daemon_safe_handler
    def _handle_read_view_profiles(self) -> None:
        """``GET /api/read-view-profiles`` exposes shared read-view metadata."""

        from polylogue.archive.viewport import read_view_profile_payloads

        profiles = read_view_profile_payloads()
        self._send_json(HTTPStatus.OK, {"read_views": profiles, "total": len(profiles)})

    # ------------------------------------------------------------------
    # Handlers: per-session embedding similarity (#1123)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_similar(
        self,
        conv_id: str,
        params: dict[str, list[str]],
    ) -> None:
        """``GET /api/sessions/{id}/similar[?limit=N]``.

        Returns ranked similar sessions through the embedding read
        surface from #828. The endpoint is honest about the embedding
        pipeline's state: when embeddings are disabled, unavailable, or
        the source session has not been embedded yet, the response
        carries an explicit ``status`` rather than an empty success.
        ``limit`` is clamped server-side to
        :data:`polylogue.daemon.similarity.SIMILAR_RESULTS_MAX`.
        """
        from polylogue.daemon.similarity import build_similar_payload

        requested_limit: int | None = None
        raw_limit = self._get_param(params, "limit")
        if raw_limit is not None:
            try:
                requested_limit = int(raw_limit)
            except (TypeError, ValueError):
                requested_limit = None

        payload = build_similar_payload(conv_id, limit=requested_limit)
        if payload is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, payload)

    # ------------------------------------------------------------------
    # Handlers: get messages
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_messages(self, conv_id: str, params: dict[str, list[str]]) -> None:
        limit = self._get_int(params, "limit", 50)
        offset = self._get_int(params, "offset", 0)
        projection = _content_projection_from_params(params)

        archive_root = _web_reader_archive_root()
        if archive_root is not None:
            self._send_json(
                HTTPStatus.OK, self._do_archive_get_messages(archive_root, conv_id, limit, offset, projection)
            )
            return

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_messages(poly, conv_id, limit, offset, projection)

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_messages(
        self,
        poly: Polylogue,
        conv_id: str,
        limit: int,
        offset: int,
        projection: ContentProjectionSpec,
    ) -> object:
        messages, total = await poly.get_messages_paginated(
            conv_id,
            limit=limit,
            offset=offset,
            content_projection=projection,
        )
        session_id = str(conv_id)
        return {
            "messages": [
                {
                    "id": str(msg.id),
                    "role": str(msg.role),
                    "text": msg.text,
                    "target_ref": _dump_target_ref(TargetRefPayload.message(session_id=session_id, message_id=msg.id)),
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

    def _do_archive_get_messages(
        self,
        archive_root: Path,
        conv_id: str,
        limit: int,
        offset: int,
        projection: ContentProjectionSpec,
    ) -> object:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(archive_root) as archive:
            try:
                session_id = archive.resolve_session_id(conv_id)
                envelope = archive.read_session(session_id)
            except KeyError:
                return {"messages": [], "total": 0, "limit": limit, "offset": offset}
        messages = [
            message
            for message in (self._project_archive_message(message, projection) for message in envelope.messages)
            if message
        ]
        page = messages[offset : offset + limit]
        return {
            "messages": [self._archive_message_payload(envelope.session_id, message) for message in page],
            "total": len(messages),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # Handlers: workspace stack/compare
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_stack(self, params: dict[str, list[str]]) -> None:
        ids = workspace_routes.parse_id_list(params)
        focus = self._get_param(params, "focus")
        if not ids:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        archive_root = _web_reader_archive_root()
        if archive_root is not None:
            self._send_json(HTTPStatus.OK, self._do_archive_stack(archive_root, ids, focus))
            return
        workspace_routes.handle_stack(self, params)

    @daemon_safe_handler
    def _handle_compare(self, params: dict[str, list[str]]) -> None:
        left = self._get_param(params, "left")
        right = self._get_param(params, "right")
        align = self._get_param(params, "align", "prompt")
        if not left or not right or align not in workspace_routes.COMPARE_ALIGN_MODES:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        archive_root = _web_reader_archive_root()
        if archive_root is not None:
            self._send_json(HTTPStatus.OK, self._do_archive_compare(archive_root, left, right, align or "prompt"))
            return
        workspace_routes.handle_compare(self, params)

    def _do_archive_stack(self, archive_root: Path, ids: list[str], focus: str | None) -> dict[str, object]:
        items: list[dict[str, object]] = []
        for conv_id in ids:
            payload = self._do_archive_get_session(archive_root, conv_id)
            if not isinstance(payload, dict):
                items.append(workspace_routes.missing_session_target(conv_id))
                continue
            items.append(
                {
                    "target_type": "session",
                    "target_id": str(payload["id"]),
                    "session_id": str(payload["id"]),
                    "status": "resolved",
                    "identity_key": f"session:{payload['id']}",
                    "target_ref": workspace_routes.target_ref_from_session_payload(payload),
                    "session": payload,
                }
            )
        return {
            "mode": "stack",
            "ids": ids,
            "focus": focus,
            "items": items,
            "total": len(items),
            "resolved_count": sum(1 for item in items if item["status"] == "resolved"),
            "degraded_count": sum(1 for item in items if item["status"] != "resolved"),
        }

    def _do_archive_compare(self, archive_root: Path, left: str, right: str, align: str) -> object:
        from polylogue.daemon.compare import build_compare_envelope

        left_payload = self._do_archive_get_session(archive_root, left)
        right_payload = self._do_archive_get_session(archive_root, right)
        return build_compare_envelope(left_payload, right_payload, left, right, align)

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
        raw_items = await poly.get_raw_artifacts_for_session(artifact_id)
        return {"raw_artifacts": raw_items}

    # ------------------------------------------------------------------
    # Handlers: facets
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_facets(self, params: dict[str, list[str]]) -> None:
        query_params = _build_query_spec_params(params, self)

        archive_root = _web_reader_archive_root()
        if archive_root is not None:
            self._send_json(HTTPStatus.OK, self._do_archive_facets(archive_root, params))
            return

        async def _get(poly: Polylogue) -> object:
            return await self._do_facets(poly, query_params)

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_facets(
        self,
        poly: Polylogue,
        query_params: dict[str, object],
    ) -> object:
        """Compute scoped + global facets via the shared archive contract.

        Delegates to :meth:`polylogue.api.archive.PolylogueArchiveMixin.facets`
        so daemon HTTP, MCP, CLI, and the Python API all share one
        scope vocabulary (#1269 / slice D of #873).
        """
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params(query_params) if query_params else None
        response = await poly.facets(spec)
        return response.model_dump(mode="json", by_alias=True)

    def _do_archive_facets(self, archive_root: Path, params: dict[str, list[str]]) -> object:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import FacetBucketsPayload, FacetsResponse

        query = self._get_param(params, "query") or self._get_param(params, "contains") or ""
        origin, origins = _archive_origin_filter(params)
        scoped_to_query = bool(query or origin or origins)
        with ArchiveStore.open_existing(archive_root) as archive:
            global_payload = self._archive_facet_bucket(archive)
            if scoped_to_query:
                session_ids: tuple[str, ...] = ()
                if query:
                    hits = archive.search_summaries(query, limit=10_000, origin=origin, origins=origins)
                    session_ids = tuple(dict.fromkeys(hit.session_id for hit in hits))
                    if not session_ids:
                        scoped_payload = FacetBucketsPayload()
                    else:
                        scoped_payload = self._archive_facet_bucket(
                            archive,
                            origin=origin,
                            origins=origins,
                            session_ids=session_ids,
                        )
                else:
                    scoped_payload = self._archive_facet_bucket(archive, origin=origin, origins=origins)
            else:
                scoped_payload = global_payload
        active = scoped_payload if scoped_to_query else global_payload
        return FacetsResponse.model_validate(
            {
                "scoped_to_query": scoped_to_query,
                "origins": active.origins,
                "tags": active.tags,
                "repos": active.repos,
                "message_types": active.message_types,
                "action_types": active.action_types,
                "has_flags": active.has_flags,
                "total_sessions": active.total_sessions,
                "total_messages": active.total_messages,
                "scoped": scoped_payload,
                "global": global_payload,
            }
        ).model_dump(mode="json", by_alias=True)

    def _archive_facet_bucket(
        self,
        archive: ArchiveStore,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        session_ids: tuple[str, ...] = (),
    ) -> FacetBucketsPayload:
        from polylogue.surfaces.payloads import FacetBucketsPayload

        stats = archive.stats(origin=origin, origins=origins, session_ids=session_ids)
        return FacetBucketsPayload(
            origins=archive.stats_by("origin", origin=origin, origins=origins, session_ids=session_ids),
            tags=archive.stats_by("tag", origin=origin, origins=origins, session_ids=session_ids),
            repos=archive.stats_by("repo", origin=origin, origins=origins, session_ids=session_ids),
            action_types=archive.stats_by("action", origin=origin, origins=origins, session_ids=session_ids),
            total_sessions=stats.total_sessions,
            total_messages=stats.total_messages,
        )

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
        conv_id = body.get("session_id")

        if scope == "session" and not conv_id:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        op_id = f"reset-{scope}-{conv_id[:16] if conv_id else 'all'}"

        async def _do_reset(poly: Polylogue) -> dict[str, object]:
            if scope == "session" and conv_id:
                # Route through the typed delete contract so resolution and
                # idempotency live in ArchiveMutationsMixin (#862). The prior
                # implementation invoked the async ``delete_session`` from
                # a sync callback, sending the resulting coroutine into the
                # JSON encoder unchanged.
                result = await poly.delete_session_safe(conv_id)
                return {"deleted": result.outcome == "deleted", "session_id": conv_id}
            return {"ok": True}

        result = self._sync_run(_do_reset)

        emit_daemon_event("reset", operation_id=op_id, payload=result if isinstance(result, dict) else None)

        deleted = result.get("deleted", False) if isinstance(result, dict) else False
        response = MutationResultPayload(
            status="deleted" if deleted else "ok",
            detail=f"reset {scope}" if deleted else f"reset {scope} — no sessions matched",
        )
        self._send_json(HTTPStatus.OK, response.model_dump())

    # ------------------------------------------------------------------
    # Handlers: ingest
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_otlp_post(self, path: list[str]) -> None:
        """Accept OTLP telemetry and respond with the matching protobuf/JSON envelope.

        Routes to ``handle_traces``, ``handle_metrics``, or ``handle_logs``
        in ``polylogue/daemon/otlp_receiver.py`` based on *path*.
        """
        from polylogue.config import load_polylogue_config
        from polylogue.daemon.otlp_receiver import handle_logs, handle_metrics, handle_traces

        signal = path[1]  # 'traces', 'metrics', or 'logs'
        content_type = self.headers.get("Content-Type", "application/x-protobuf")
        content_length = int(self.headers.get("Content-Length", 0))
        max_body = load_polylogue_config().otlp_max_body_bytes
        if content_length > max_body:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "payload_too_large")
            return
        body = self.rfile.read(content_length) if content_length > 0 else b""
        from polylogue.paths import active_index_db_path

        ops_db = active_index_db_path().with_name("ops.db")

        if signal == "traces":
            result = handle_traces(body, content_type, db_path=str(ops_db))
        elif signal == "metrics":
            result = handle_metrics(body, content_type, db_path=str(ops_db))
        else:
            result = handle_logs(body, content_type, db_path=str(ops_db))

        self.send_response(result.status)
        self.send_header("Content-Type", result.content_type)
        self.send_header("Content-Length", str(len(result.body)))
        self.end_headers()
        self.wfile.write(result.body)

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

        from polylogue.sources.import_preflight import preflight_import_source

        preflight = preflight_import_source(source)
        if not preflight.admissible:
            self._send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, preflight.error_code, preflight.summary())
            return

        # Typed Operation scheduling contract (#1247/#1248): the daemon
        # accepts an ``ImportRequest`` and emits an ``ImportAck`` carrying
        # the shared ``OperationFollowUp`` envelope. The existing wire keys
        # (``path``, ``status``, ``ok``) are preserved for the CLI ingest
        # adapter and HTTP clients that pre-date the typed contract.
        from pydantic import ValidationError

        from polylogue.operations.import_operations import ImportAck, ImportRequest
        from polylogue.operations.operation_contract import OperationFollowUp

        op_id = f"ingest-{source.name}"

        try:
            request = ImportRequest.model_validate(
                {
                    "source_path": body.get("path"),
                    "source_name": source.name,
                    "staged_path": str(source),
                    "idempotency_key": body.get("idempotency_key"),
                }
            )
        except ValidationError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        emit_daemon_event(
            "ingest",
            operation_id=op_id,
            payload={"path": str(source), "inbox": str(inbox), "preflight": preflight.to_dict()},
        )

        message = "Ingestion scheduled. Check status for progress."
        if preflight.status.value == "degraded":
            message = f"{message} {preflight.summary()}"
        ack = ImportAck.pending_import(
            operation_id=op_id,
            follow_up=OperationFollowUp(
                status_endpoint=f"/api/operations/{op_id}",
                poll_after_ms=500,
            ),
            message=message,
        )
        response: dict[str, object] = dict(ack.to_dict())
        response["ok"] = True
        response["path"] = str(source)
        response["preflight"] = preflight.to_dict()
        # Surface the validated, typed request fields so clients can confirm
        # the contract used; reading the request back closes the loop for
        # adapter parity tests.
        response["request"] = request.to_dict()
        self._send_json(HTTPStatus.ACCEPTED, response)

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
        from polylogue.maintenance.scope import MaintenanceScopeFilter
        from polylogue.paths import archive_root, render_root

        try:
            scope_filter = MaintenanceScopeFilter.from_dict(_parse_scope_filter_body(body))
        except (TypeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        config = Config(
            archive_root=archive_root(),
            render_root=render_root(),
            sources=[],
        )
        result = preview_backfill(config, targets=targets, scope_filter=scope_filter)
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
        from polylogue.maintenance.scope import MaintenanceScopeFilter
        from polylogue.paths import archive_root, render_root

        try:
            scope_filter = MaintenanceScopeFilter.from_dict(_parse_scope_filter_body(body))
        except (TypeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        config = Config(
            archive_root=archive_root(),
            render_root=render_root(),
            sources=[],
        )
        result = execute_backfill(config, targets=targets, dry_run=dry_run, scope_filter=scope_filter)
        envelope = envelope_from_operation(result, origin="daemon", mode="execute")
        self._send_json(HTTPStatus.OK, envelope.to_dict())

    @daemon_safe_handler
    def _handle_maintenance_status(self, operation_id: str) -> None:
        """GET /api/maintenance/status/<op_id> — delegate to maintenance_registry_http."""
        from polylogue.daemon.maintenance_registry_http import handle_status

        handle_status(self, operation_id)

    @daemon_safe_handler
    def _handle_maintenance_operations(self) -> None:
        """GET /api/maintenance/operations — delegate to maintenance_registry_http."""
        from polylogue.daemon.maintenance_registry_http import handle_operations

        handle_operations(self)


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
