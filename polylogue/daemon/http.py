"""Daemon HTTP API server for the Polylogue local daemon."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import hmac
import json
import select
import socket
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic, time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, cast
from urllib.parse import parse_qs, parse_qsl, unquote, urlparse, urlsplit

from polylogue.archive.query.transaction import (
    QueryArchiveEpochUnreadableError,
    QueryContinuationInvalidError,
    QueryContinuationStaleError,
    archive_read_context,
)
from polylogue.archive.viewport import READ_VIEW_HTTP_CAPABILITIES
from polylogue.core.compute import (
    BoundedComputeAdapter,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    current_cancellation,
)
from polylogue.core.errors import (
    ArchiveTierUnavailableError,
    DatabaseError,
    PolylogueError,
    SearchIndexUnavailableError,
)
from polylogue.core.json import JSONDocument
from polylogue.core.loopback import is_loopback_host
from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.daemon import workspace_routes
from polylogue.daemon.events import (
    emit_daemon_event,
    get_latest_event_id,
)
from polylogue.daemon.peer_identity import peer_socket_owned_by_current_uid
from polylogue.daemon.route_contracts import (
    DAEMON_ROUTE_DECLARATIONS,
    RouteContract,
    RouteSpec,
    declared_route_keys,
    metadata_only_api_routes,
    route_contract_for_pattern,
    route_contract_from_declaration,
)
from polylogue.daemon.route_families import read_detail, read_query, user_overlay
from polylogue.daemon.route_types import RouteMethod
from polylogue.daemon.status_snapshot import get_status_snapshot_payload
from polylogue.daemon.web_auth import (
    WEB_CREDENTIAL_SCOPES,
    WEB_SIGN_IN_HTML,
    WEB_SIGN_IN_SCRIPT,
    WebCredentialBootstrapPayload,
    WebCredentialDecision,
    WebCredentialRegistry,
    WebCredentialRevocationPayload,
    WebCredentialRevokedPayload,
    WebCredentialScope,
    WebSignInTicketPayload,
    credential_cookie,
    exact_origin_allowed,
    expired_credential_cookie,
    read_web_credential_cookie,
    same_origin_from_headers,
)
from polylogue.daemon.webui_data import (
    LibraryEntry,
    PasteBrowserEntry,
    attachment_to_envelope,
    build_library_payload,
    build_paste_browser_payload,
    envelope_paste_spans,
    snippet_for_paste,
)
from polylogue.daemon.write_coordinator import (
    DaemonWriteCoordinator,
    DaemonWriterSettlementError,
    DaemonWriteThreadBridge,
    register_write_coordinator,
)
from polylogue.declarations import HandlerBinding
from polylogue.logging import DEBUG, ERROR, WARNING, emit, propagate
from polylogue.logging import span as log_span
from polylogue.operations.authority import authority_for_config
from polylogue.operations.http_session_reads import (
    HttpSessionProjectionAdapters,
    execute_http_session_detail,
    execute_http_session_messages,
)
from polylogue.operations.message_locator import (
    MessageNotInSessionError,
)
from polylogue.operations.origin_filters import unknown_origin_filter_tokens
from polylogue.operations.quick_check import HEALTH_RESULT_KEY, observe_quick_check
from polylogue.rendering.semantic_card_placement import (
    semantic_card_placement_for_messages,
)
from polylogue.rendering.semantic_cards import (
    lineage_descriptor_from_session,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import one_shot_diagnostic_read
from polylogue.surfaces.authority import serialize_authority
from polylogue.surfaces.outcome import OutcomeEnvelope, combine_outcomes, decide_outcome, lineage_page_outcome
from polylogue.surfaces.payloads import (
    MutationResultPayload,
    QueryErrorPayload,
    QueryFailurePayload,
    QueryMissDiagnosticsPayload,
    QueryMissReasonPayload,
    ReaderActionAvailabilityPayload,
    RouteReadinessPayload,
    RouteReadinessState,
    SessionReadViewEnvelope,
    TargetRefPayload,
    _build_flags_from_session,
    message_render_envelope_from_domain,
    message_topology_from_domain,
    model_json_document,
    reader_anchor,
    reader_message_actions,
    reader_session_actions,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.daemon.webui import WebUIAsset
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.operations.mutation_transaction import MutationPrincipal
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSummary,
        ArchiveStore,
    )


class _AttachmentRow(Protocol):
    """What the attachment library page needs from an attachment record.

    Declared structurally rather than imported: gate layering disallows
    ``polylogue/daemon`` importing ``polylogue/storage``, and the facade
    cannot name the record type either for the same reason.
    """

    session_id: object
    message_id: str | None


_ARCHIVE_READER_BUSY_TIMEOUT_S = 0.25
_COORDINATION_CACHE_TTL_S = 2.0
_CLI_DELETE_SELECTION_MAX_BYTES = 64 * 1024 * 1024
_CLI_DELETE_AUTHORIZATION_MAX_BYTES = 8_192

_ArchiveQueryResult = TypeVar("_ArchiveQueryResult")


@dataclass(frozen=True)
class _StaticGetRoute:
    contract: RouteContract
    segments: tuple[str, ...]
    handler_name: str
    passes_params: bool = False
    passes_path: bool = False

    @property
    def pattern(self) -> str:
        return self.contract.pattern


@dataclass(frozen=True)
class _ParameterizedGetRoute:
    contract: RouteContract
    prefix: tuple[str, ...]
    suffix: tuple[str, ...]
    handler_name: str
    passes_params: bool = False
    passes_path: bool = False

    @property
    def pattern(self) -> str:
        return self.contract.pattern


@dataclass(frozen=True)
class _DeclaredMutationRoute:
    declaration: RouteSpec
    segments: tuple[str, ...]
    handler_name: str

    def matches(self, path: list[str]) -> bool:
        return len(path) == len(self.segments) and all(
            expected.startswith(":") and bool(actual) or expected == actual
            for expected, actual in zip(self.segments, path, strict=True)
        )


def _declared_mutation_routes(method: str) -> tuple[_DeclaredMutationRoute, ...]:
    return tuple(
        _DeclaredMutationRoute(declaration, _route_segments(declaration.path), _daemon_http_binding(declaration).symbol)
        for declaration in DAEMON_ROUTE_DECLARATIONS
        if declaration.method == method
    )


@dataclass(frozen=True)
class _CoordinationCacheEntry:
    """One short-lived coordination response held only by the daemon."""

    payload: JSONDocument
    expires_at: float


_FACET_EXPENSIVE_FAMILIES = ("repos", "action_types")
_FACET_CANCELLED_REASON = "client_disconnected"
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class _ClientDisconnectedDuringComputeError(ConnectionAbortedError):
    """Raised when a long-running route sees its peer close the socket."""


def _socket_peer_disconnected(connection: object | None) -> bool:
    """Best-effort loopback HTTP peer-close probe without consuming request bytes.

    ``BaseHTTPRequestHandler`` only observes many browser aborts when writing the
    response, which is too late for archive-wide SQLite work. A readable socket
    whose one-byte peek returns ``b""`` means the peer has closed; readable
    bytes mean HTTP pipelining/keepalive data, so the current request should keep
    running. Unsupported socket-like objects return ``False`` so tests and custom
    servers do not get false cancellation.
    """

    if connection is None or not hasattr(connection, "recv"):
        return False
    flags = getattr(socket, "MSG_PEEK", None)
    if flags is None:
        return False
    try:
        readable, _, _ = select.select([cast(socket.socket, connection)], [], [], 0)
    except (OSError, TypeError, ValueError):
        return False
    if not readable:
        return False
    try:
        data = connection.recv(1, flags)
    except BlockingIOError:
        return False
    except _CLIENT_DISCONNECT_ERRORS:
        return True
    except OSError:
        return True
    return bool(data == b"")


def _web_privacy_safe_projection(payload: object, *archive_roots: Path | None) -> object:
    """Drop archive identity and redact configured-root text from web payloads.

    CLI/operator DTOs intentionally retain diagnostic paths. The browser
    projection is a separate public boundary, including free-form caveats and
    serialized error details where a path could otherwise reappear.
    """
    roots = tuple(
        {
            root_text
            for archive_root in archive_roots
            if archive_root is not None
            for root_text in (str(archive_root), str(archive_root.resolve()))
        }
    )

    def project(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): project(child) for key, child in value.items() if key != "archive_root"}
        if isinstance(value, tuple | list):
            return [project(child) for child in value]
        if isinstance(value, str):
            for root in roots:
                if root:
                    value = value.replace(root, "[archive]")
            return value
        return value

    return project(payload)


def _route_segments(pattern: str) -> tuple[str, ...]:
    if pattern == "/":
        return ()
    return tuple(part for part in pattern.strip("/").split("/") if part)


def _read_view_payload_field_values(payload: object, field_name: str, *, max_depth: int = 4) -> tuple[str, ...]:
    """Collect common read-view metadata fields from a profile-specific payload."""

    seen: set[str] = set()
    values: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value not in seen:
            seen.add(value)
            values.append(value)

    def walk(value: object, depth: int) -> None:
        if depth < 0:
            return
        if hasattr(value, "model_dump"):
            value = cast(Any, value).model_dump(mode="python", exclude_none=True)
        if isinstance(value, Mapping):
            field_value = value.get(field_name)
            if isinstance(field_value, str):
                add(field_value)
            elif isinstance(field_value, Sequence) and not isinstance(field_value, str):
                for item in field_value:
                    add(item)
            for child in value.values():
                walk(child, depth - 1)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for child in value:
                walk(child, depth - 1)

    walk(payload, max_depth)
    return tuple(values)


def _static_get_routes() -> tuple[_StaticGetRoute, ...]:
    return () + tuple(route for route in _declared_get_routes() if isinstance(route, _StaticGetRoute))


def _declared_static_get_route(method: str, path: str) -> _StaticGetRoute:
    """Build a production GET adapter from the shared route declaration."""

    declaration = _declaration_for_route(method, path)
    binding = _daemon_http_binding(declaration)
    try:
        contract = route_contract_for_pattern(method, path)
    except KeyError:
        contract = route_contract_from_declaration(declaration)
    return _StaticGetRoute(
        contract=contract,
        segments=_route_segments(declaration.path),
        handler_name=binding.symbol,
        passes_params=declaration.passes_params,
        passes_path=declaration.passes_path,
    )


def _declared_parameterized_get_route(method: str, path: str) -> _ParameterizedGetRoute:
    """Build a parameterized adapter entirely from a declared route spec."""

    declaration = _declaration_for_route(method, path)
    binding = _daemon_http_binding(declaration)
    parts = _route_segments(declaration.path)
    parameter_index = next(index for index, part in enumerate(parts) if part.startswith(":"))
    try:
        contract = route_contract_for_pattern(method, path)
    except KeyError:
        contract = route_contract_from_declaration(declaration)
    return _ParameterizedGetRoute(
        contract=contract,
        prefix=parts[:parameter_index],
        suffix=parts[parameter_index + 1 :],
        handler_name=binding.symbol,
        passes_params=declaration.passes_params,
        passes_path=declaration.passes_path,
    )


def _declaration_for_route(method: str, path: str) -> RouteSpec:
    """Resolve one declaration or raise an actionable generation error."""

    normalized_method = method.upper()
    for declaration in DAEMON_ROUTE_DECLARATIONS:
        if declaration.method == normalized_method and declaration.path == path:
            return declaration
    raise RuntimeError(f"daemon route declaration missing: {normalized_method} {path}")


def _daemon_http_binding(declaration: RouteSpec) -> HandlerBinding:
    """Return the sole daemon-http binding for a route declaration."""

    bindings = tuple(binding for binding in declaration.kernel.handlers if binding.surface == "daemon-http")
    if len(bindings) != 1:
        declaration_id = declaration.kernel.declaration_id
        raise RuntimeError(f"daemon route declaration must have one daemon-http binding: {declaration_id}")
    return bindings[0]


def _declared_get_routes() -> tuple[_StaticGetRoute | _ParameterizedGetRoute, ...]:
    """Build every migrated GET adapter from the declaration registry."""

    routes: list[_StaticGetRoute | _ParameterizedGetRoute] = []
    for declaration in DAEMON_ROUTE_DECLARATIONS:
        if declaration.method != "GET":
            continue
        parts = _route_segments(declaration.path)
        if any(part.startswith(":") for part in parts):
            routes.append(_declared_parameterized_get_route("GET", declaration.path))
        else:
            routes.append(_declared_static_get_route("GET", declaration.path))
    return tuple(routes)


def validate_declared_route_reachability(handler_class: type[BaseHTTPRequestHandler]) -> None:
    """Fail startup when a migrated declaration cannot reach the installed router."""

    # Keep the migration boundary explicit: routes outside the declaration
    # kernel are still allowed during the staged cutover, but every one must
    # carry a named metadata-only reason.  A newly added API contract without
    # that classification fails before the daemon can start serving it.
    unclassified = [
        f"{route.method} {route.pattern}" for route in metadata_only_api_routes() if not route.metadata_only_reason
    ]
    if unclassified:
        raise RuntimeError(f"metadata-only API routes lack migration reasons: {sorted(unclassified)}")

    declared = tuple((item.method, item.path) for item in DAEMON_ROUTE_DECLARATIONS)
    if len(declared) != len(set(declared)):
        raise RuntimeError("duplicate daemon route declaration method/path")
    expected = declared_route_keys()
    if not expected.issubset(set(declared)):
        missing = sorted(expected - set(declared))
        extra = sorted(set(declared) - expected)
        raise RuntimeError(f"daemon declaration generation mismatch: missing={missing}, extra={extra}")
    generated_get_routes = _declared_get_routes()
    generated_mutation_routes = _declared_mutation_routes("POST") + _declared_mutation_routes("DELETE")
    generated = tuple((route.contract.method, route.pattern) for route in generated_get_routes) + tuple(
        (route.declaration.method, route.declaration.path) for route in generated_mutation_routes
    )
    installed_get_routes: tuple[_StaticGetRoute | _ParameterizedGetRoute, ...] = (
        _static_get_routes() + _parameterized_get_routes()
    )
    installed = tuple(
        (route.contract.method, route.pattern)
        for route in installed_get_routes
        if (route.contract.method, route.pattern) in set(declared)
    ) + tuple((route.declaration.method, route.declaration.path) for route in generated_mutation_routes)
    for route in generated_get_routes:
        if not callable(getattr(handler_class, route.handler_name, None)):
            raise RuntimeError(f"daemon route adapter is unreachable: {route.handler_name}")
    for mutation_route in generated_mutation_routes:
        if not callable(getattr(handler_class, mutation_route.handler_name, None)):
            raise RuntimeError(f"daemon route adapter is unreachable: {mutation_route.handler_name}")
    if (
        len(generated) != len(set(generated))
        or len(installed) != len(set(installed))
        or not expected.issubset(set(generated))
        or not expected.issubset(set(installed))
    ):
        missing = sorted(set(declared) - set(installed))
        extra = sorted(set(installed) - set(declared))
        raise RuntimeError(f"daemon declaration generation mismatch: missing={missing}, extra={extra}")


def _parameterized_get_routes() -> tuple[_ParameterizedGetRoute, ...]:
    return () + tuple(route for route in _declared_get_routes() if isinstance(route, _ParameterizedGetRoute))


def _normalize_session_route_id(identifier: str) -> str:
    """Accept session target-ref identity keys in session route ids."""

    return identifier.removeprefix("session:")


def implemented_daemon_route_patterns() -> tuple[tuple[RouteMethod, str], ...]:
    """Return route patterns implemented by daemon HTTP dispatch."""

    routes: list[tuple[RouteMethod, str]] = [
        ("GET", "/"),
        ("GET", "/observability"),
        ("GET", "/cost"),
        ("GET", "/sessions"),
        ("GET", "/sessions/:session_id"),
        ("GET", "/search"),
        ("GET", "/assets/:asset"),
        ("GET", "/s/:session_id"),
        ("GET", "/w/:mode"),
        ("GET", "/p"),
        ("GET", "/a"),
        ("GET", "/healthz/live"),
        ("GET", "/healthz/ready"),
        ("GET", "/metrics"),
        ("GET", "/web-auth/sign-in.js"),
        ("GET", "/web-auth/sign-in"),
    ]
    routes.extend(("GET", route.pattern) for route in _static_get_routes())
    routes.extend(("GET", route.pattern) for route in _parameterized_get_routes())
    routes.extend((route.declaration.method, route.declaration.path) for route in _declared_mutation_routes("POST"))
    routes.extend((route.declaration.method, route.declaration.path) for route in _declared_mutation_routes("DELETE"))
    return tuple(routes)


def _json_bytes(payload: object) -> bytes:
    from polylogue.core.json import dumps_bytes

    return dumps_bytes(payload, append_newline=True)


def _stable_status_identity(value: object) -> object:
    """Remove clock-only status diagnostics from the conditional identity."""
    if isinstance(value, Mapping):
        return {
            str(key): _stable_status_identity(item)
            for key, item in value.items()
            if str(key) not in {"age_s", "evaluated_at", "quick_check_age_s"}
        }
    if isinstance(value, list):
        return [_stable_status_identity(item) for item in value]
    return value


def _web_reader_archive_root() -> Path | None:
    """Return the archive root when archive reader routes should use it."""
    from polylogue.paths import archive_root

    root = archive_root()
    required = ((ArchiveTier.SOURCE, root / "source.db"), (ArchiveTier.INDEX, root / "index.db"))
    for tier, path in required:
        if not path.exists():
            return None
        try:
            with one_shot_diagnostic_read(path, tier=tier) as conn:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        except sqlite3.Error as exc:
            emit(
                "daemon.http.archive_root_probe_failed",
                level=WARNING,
                outcome="error",
                reason="tier_version_probe_failed",
                tier=tier.value if hasattr(tier, "value") else str(tier),
                path=path,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            return None
        if version != ARCHIVE_VERSION_BY_TIER[tier]:
            return None
    return root


def _utc_timestamp_json() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


_CREDENTIAL_QUERY_PARAMETERS = frozenset(
    {
        "access_token",
        "api_key",
        "api_token",
        "auth_token",
        "bearer_token",
        "client_secret",
        "credential",
        "id_token",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)


def _public_route_from_request_path(raw_path: str) -> str:
    """Project a request to route identity without reflecting query values."""

    return urlparse(raw_path).path or "/"


def _request_path_for_log(raw_path: str) -> str:
    """Keep all query values, including misnamed secrets, out of daemon logs."""

    return urlparse(raw_path).path or "/"


def _route_readiness_payload(
    state: RouteReadinessState,
    route: str,
    *,
    reason: str | None = None,
    component: str | None = None,
    stale_available: bool = False,
) -> dict[str, object]:
    return RouteReadinessPayload(
        state=state,
        route=route,
        reason=reason,
        component=component,
        generated_at=_utc_timestamp_json(),
        stale_available=stale_available,
    ).model_dump(mode="json")


def _session_list_state(outcome: OutcomeEnvelope, *, filtered: bool) -> tuple[RouteReadinessState, str | None]:
    """Project the canonical terminal outcome onto the reader's readiness chip.

    The chip is presentation over the one decision the operation already made;
    it never re-derives readiness from the row count.
    """
    if not outcome.rows_are_authoritative:
        return "degraded", outcome.reason
    if outcome.state == "ok":
        return "ready", None
    if filtered:
        return "no_results", "No sessions matched the active query or filters."
    return "empty", "Archive contains no sessions."


def _truthy_query_param(params: dict[str, list[str]], key: str) -> bool:
    values = params.get(key) or []
    if not values:
        return False
    return values[-1].strip().lower() not in {"", "0", "false", "no", "off"}


def _facet_requested_optional_families(params: dict[str, list[str]]) -> set[str]:
    requested: set[str] = set()
    for key in ("family", "families", "include"):
        for value in params.get(key) or []:
            requested.update(token.strip().lower() for token in value.split(",") if token.strip())

    if _truthy_query_param(params, "include_deferred") or _truthy_query_param(params, "include_expensive"):
        requested.update(_FACET_EXPENSIVE_FAMILIES)
    if "all" in requested or "*" in requested:
        requested.update(_FACET_EXPENSIVE_FAMILIES)
    if "repo" in requested:
        requested.add("repos")
    if "actions" in requested or "action" in requested:
        requested.add("action_types")
    return {family for family in requested if family in _FACET_EXPENSIVE_FAMILIES}


def _csv_values(params: dict[str, list[str]], key: str) -> tuple[str, ...]:
    """Collect repeated and/or comma-separated query-string values."""
    values: list[str] = []
    for value in params.get(key) or []:
        values.extend(token.strip() for token in value.split(",") if token.strip())
    return tuple(dict.fromkeys(values))


def _dump_target_ref(target_ref: TargetRefPayload) -> dict[str, object]:
    return target_ref.model_dump(mode="json", exclude_none=True)


def _dump_actions(actions: Mapping[str, ReaderActionAvailabilityPayload]) -> dict[str, object]:
    return {name: availability.model_dump(mode="json", exclude_none=True) for name, availability in actions.items()}


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
        "origin": insight.origin,
        "model_name": estimate.model_name,
        "normalized_model": estimate.normalized_model,
        "status": estimate.status,
        "confidence": float(estimate.confidence),
        "confidence_tag": _confidence_tag(estimate.status),
        "currency": estimate.currency,
        "total_usd": None if estimate.total_usd is None else float(estimate.total_usd),
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
# The insights browser endpoint surfaces the per-session insight kinds
# (profile, work threads) as a single JSON
# envelope so the reader inspector can render them inline. Each kind carries
# a readiness chip from the closed vocabulary ``q-ready`` / ``q-partial`` /
# ``q-missing`` driven by:
#
# - ``q-ready``    — the insight is materialized for this session.
# - ``q-missing``  — the insight has no materialized row for this session
#   (the substrate is empty for this scope, not the whole archive).
# - ``q-partial``  — the insight is materialized but the row count is zero
#   (e.g. a session profile exists but has no threads recorded).
#
# The endpoint never imports insight storage modules directly — it routes
# through the same public ``Polylogue`` facade adapters that CLI and MCP use
# (AC#1120, AC#1018).
# ---------------------------------------------------------------------------

INSIGHT_KINDS: tuple[str, ...] = ("profile", "threads")


def _readiness_tag(outcome: OutcomeEnvelope, *, materialized: bool, row_count: int | None = None) -> str:
    """Map one panel's terminal outcome and row count to the readiness chip.

    The chip vocabulary is closed (``q-error`` / ``q-missing`` / ``q-partial``
    / ``q-ready``). ``q-error`` is what a panel whose insight surface could not
    answer reports; without it an unavailable surface and a session that
    genuinely has no rows both render as zero rows. Unmaterialized rows are
    ``q-missing``; materialized rows with zero downstream rows are
    ``q-partial`` (the rebuild ran but produced nothing).
    """
    if not outcome.rows_are_authoritative:
        return "q-error"
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
    # An unmaterialized insight has no recorded materializer version; keep it unknown.
    version = getattr(prov, "materializer_version", None)
    return {
        "materializer_version": None if version is None else int(version),
        "materialized_at": getattr(prov, "materialized_at", None),
        "source_updated_at": getattr(prov, "source_updated_at", None),
        "source_sort_key": getattr(prov, "source_sort_key", None),
        "input_high_water_mark": getattr(prov, "input_high_water_mark", None),
        "input_high_water_mark_source": getattr(prov, "input_high_water_mark_source", None),
        "time_confidence": getattr(prov, "time_confidence", "unknown"),
    }


def _profile_staleness(record: Any, session_updated_at: str | None) -> dict[str, object] | None:
    """Compare a session-profile record's provenance against its session.

    Routes through :func:`polylogue.analysis.provenance.is_stale` so the
    daemon insights browser (#1018/#1120) consumes the typed staleness
    helper rather than re-deriving the high-water-mark comparison inline.
    Returns ``None`` when the record lacks the provenance fields the
    helper expects.
    """
    from polylogue.analysis.provenance import HasProvenance, is_stale

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


def _profile_panel_payload(profile: Any, provenance: Any) -> dict[str, object]:
    """Project a ``SessionProfile`` into the JSON shape served by the reader.

    Uses :meth:`SessionProfile.to_dict` for fidelity to the substrate shape
    and adds a readiness chip + provenance summary on top.
    """
    body = dict(profile.to_dict())
    outcome = decide_outcome(matched=True)
    return {
        "outcome": outcome.to_dict(),
        "readiness_tag": _readiness_tag(outcome, materialized=True, row_count=1),
        "materialized": True,
        "profile": body,
        "provenance": _provenance_dict(provenance),
    }


def _empty_profile_panel_payload(outcome: OutcomeEnvelope) -> dict[str, object]:
    return {
        "outcome": outcome.to_dict(),
        "readiness_tag": _readiness_tag(outcome, materialized=False),
        "materialized": False,
        "profile": None,
        "provenance": None,
    }


def _thread_panel_payload(threads: list[Any], outcome: OutcomeEnvelope) -> dict[str, object]:
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
        "outcome": outcome.to_dict(),
        "readiness_tag": _readiness_tag(outcome, materialized=bool(threads), row_count=len(threads)),
        "materialized": bool(threads),
        "count": len(items),
        "threads": items,
    }


def _message_type_value(message: object) -> str:
    message_type = getattr(message, "message_type", "")
    if hasattr(message_type, "value"):
        return str(message_type.value)
    return str(message_type)


def _material_origin_value(message: object) -> str:
    material_origin = getattr(message, "material_origin", "")
    if hasattr(material_origin, "value"):
        return str(material_origin.value)
    return str(material_origin)


class DaemonMutationIndeterminate(RuntimeError):  # noqa: N818 - public typed outcome name
    """A mutating route's wait hit its deadline with the write still in flight.

    Distinct from a cancellation: nothing was withdrawn, so the caller must
    re-read the archive rather than assume the mutation did not happen.
    """

    code = "mutation_indeterminate"


def daemon_safe_handler(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that answers a route's escaped exception with a typed envelope.

    The mapping itself is ``_answer_route_exception``; the request boundary in
    ``do_GET``/``do_POST``/``do_DELETE`` applies the same mapping to routes
    that are not decorated.
    """

    @functools.wraps(fn)
    def wrapper(self: DaemonAPIHandler, *args: object, **kwargs: object) -> None:
        try:
            fn(self, *args, **kwargs)
        except Exception as exc:
            _answer_route_exception(self, exc, route=fn.__name__)

    return wrapper


def _answer_route_exception(
    handler: DaemonAPIHandler, exc: Exception, *, route: str, method: str | None = None
) -> None:
    """Map one escaped route exception to its HTTP answer.

    PolylogueError subclasses carry ``http_status_code`` — use it. Unexpected
    exceptions map to a 500 ``internal_error`` envelope whose ``outcome`` is
    an error ``OutcomeEnvelope``, logged once. When the response has already
    started, a second status line would corrupt the stream, so the failure is
    logged and the connection is closed instead. A client that disconnects,
    whether in the route or while this answer is written, is logged at debug.
    """

    context: dict[str, object] = {"route": route}
    if method is not None:
        context["method"] = method
    try:
        if not isinstance(exc, _CLIENT_DISCONNECT_ERRORS):
            _write_route_exception_answer(handler, exc, context=context)
            return
    except _CLIENT_DISCONNECT_ERRORS as write_exc:
        exc = write_exc
    emit(
        "daemon.http.client_disconnected",
        level=DEBUG,
        outcome="skipped",
        reason="client_disconnected",
        error_type=type(exc).__name__,
        **context,
    )


def _write_route_exception_answer(handler: DaemonAPIHandler, exc: Exception, *, context: dict[str, object]) -> None:
    if handler._response_started:
        emit(
            "daemon.http.route_failed",
            level=ERROR,
            outcome="error",
            reason="failed_after_response_started",
            error_type=type(exc).__name__,
            error_detail=str(exc),
            **context,
        )
        handler.close_connection = True
        return
    if isinstance(exc, PolylogueError):
        status = (
            HTTPStatus(exc.http_status_code) if 100 <= exc.http_status_code <= 599 else HTTPStatus.INTERNAL_SERVER_ERROR
        )
        diagnostic = getattr(exc, "diagnostic", None)
        if isinstance(diagnostic, dict):
            handler._send_json(status, diagnostic)
            return
        field = getattr(exc, "field", None)
        handler._send_json(
            status,
            QueryErrorPayload(
                error=type(exc).__name__,
                detail=(exc.public_message if isinstance(exc, ArchiveTierUnavailableError) else str(exc)),
                field=field,
            ).model_dump(mode="json"),
        )
        return
    if isinstance(exc, sqlite3.OperationalError) and _is_sqlite_busy_error(exc):
        handler._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            QueryErrorPayload(
                error="archive_busy",
                detail="Archive read route is temporarily unavailable while the daemon is writing catch-up data.",
            ).model_dump(mode="json"),
            extra_headers={"Retry-After": "1"},
        )
        return
    if isinstance(exc, TimeoutError):
        emit(
            "daemon.http.route_timeout",
            level=WARNING,
            outcome="unmeasured",
            reason="archive_query_timeout",
            status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
            error_type=type(exc).__name__,
            error_detail=str(exc),
            **context,
        )
        handler._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            QueryErrorPayload(error="archive_query_timeout", detail=str(exc)).model_dump(mode="json"),
            extra_headers={"Retry-After": "2"},
        )
        return
    if isinstance(exc, (DaemonBackpressureError, DaemonWriterSettlementError)):
        handler._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            QueryErrorPayload(error=exc.code, detail=str(exc)).model_dump(mode="json"),
            extra_headers={"Retry-After": "1"},
        )
        return
    if isinstance(exc, DaemonMutationIndeterminate):
        emit(
            "daemon.http.mutation_indeterminate",
            level=WARNING,
            outcome="unmeasured",
            reason="mutation_indeterminate",
            status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
            error_type=type(exc).__name__,
            error_detail=str(exc),
            **context,
        )
        handler._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            QueryErrorPayload(error=exc.code, detail=str(exc)).model_dump(mode="json"),
            extra_headers={"Retry-After": "5"},
        )
        return
    if isinstance(exc, QueryArchiveEpochUnreadableError):
        # A missing or unreadable archive tier is retryable unavailability with
        # its own code and guidance, as the query-unit route answers it.
        handler._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            QueryErrorPayload(error=exc.code, detail=str(exc)).model_dump(mode="json"),
        )
        return
    if isinstance(exc, DaemonOperationCancelled):
        handler._send_json(
            HTTPStatus.REQUEST_TIMEOUT,
            QueryErrorPayload(error=exc.code, detail=str(exc)).model_dump(mode="json"),
        )
        return
    error_code = "sqlite_error" if isinstance(exc, sqlite3.OperationalError) else "internal_error"
    emit(
        "daemon.http.route_failed",
        level=ERROR,
        outcome="error",
        reason="sqlite_error" if error_code == "sqlite_error" else "unhandled_error",
        status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
        error_type=type(exc).__name__,
        error_detail=str(exc),
        **context,
    )
    handler._send_json(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        QueryFailurePayload(error=error_code, outcome=OutcomeEnvelope(state="error", reason=error_code)).model_dump(
            mode="json"
        ),
    )


def _is_sqlite_busy_error(exc: sqlite3.OperationalError) -> bool:
    """Defer to SQLite's result code so extended LOCKED codes still get 503."""
    return is_transient_sqlite_lock(exc)


def _build_query_spec_params(
    params: dict[str, list[str]],
    handler: DaemonAPIHandler,
) -> dict[str, object]:
    """Build SessionQuerySpec-compatible params from HTTP query string.

    API and server-rendered session lists pass these operands to the canonical
    query executor, so filter, ordering and retrieval semantics have one owner.
    """
    from polylogue.archive.query.spec import split_repo_names

    spec_params: dict[str, object] = {}

    origins = _csv_values(params, "origin")
    excluded_origins = _csv_values(params, "exclude_origin")
    # polylogue-01fe: an unrecognized ``?origin=`` used to reach the lenient
    # wire-token normalizer and answer HTTP 200 with ``total: 0``, so a
    # mistyped or near-miss origin was reported as "no data" while the CLI
    # rejected the same token and MCP returned the unfiltered aggregate.
    # ``QuerySpecError`` carries http_status_code=400 and ``daemon_safe_handler``
    # renders it as the QueryErrorPayload-shaped 400 the other surfaces return,
    # so all three surfaces now answer this input class the same way. The gate
    # runs before anything is parsed: a request naming an origin that does not
    # exist has no valid interpretation to build a spec from.
    unknown = unknown_origin_filter_tokens([*origins, *excluded_origins])
    if unknown:
        from polylogue.archive.query.spec import QuerySpecError

        raise QuerySpecError("origin", ", ".join(unknown))

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
        "similar_session_id",
        "since_session_id",
        "message_type",
    ):
        val = handler._get_param(params, key)
        if val is not None:
            spec_params[key] = val

    # CSV/repeated-value fields: collect every occurrence of ``?key=a&key=b``
    # as well as comma-joined values (``?key=a,b``) rather than only the
    # first query-string occurrence, matching the archive route's historical
    # ``_csv_values``/``_archive_origin_filter`` behavior.
    if origins:
        spec_params["origin"] = origins
    if excluded_origins:
        spec_params["exclude_origin"] = excluded_origins

    repo_names = tuple(dict.fromkeys(name for value in params.get("repo", ()) for name in split_repo_names(value)))
    if repo_names:
        spec_params["repo"] = repo_names

    for key in (
        "tag",
        "exclude_tag",
        "has_type",
        "referenced_path",
        "action",
        "exclude_action",
        "action_sequence",
        "tool",
        "exclude_tool",
    ):
        values = _csv_values(params, key)
        if values:
            spec_params[key] = values

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

    # Public HTTP aliases for the same booleans (``has_tool_use`` /
    # ``has_thinking`` / ``has_paste_evidence`` are the names the split-archive
    # route and its tests use; ``filter_has_*`` are the SessionQuerySpec field
    # names). Accept both so either param name compiles to the same filter.
    if "filter_has_tool_use" not in spec_params and handler._get_bool(params, "has_tool_use"):
        spec_params["filter_has_tool_use"] = True
    if "filter_has_thinking" not in spec_params and handler._get_bool(params, "has_thinking"):
        spec_params["filter_has_thinking"] = True
    if "filter_has_paste" not in spec_params and handler._get_bool(params, "has_paste_evidence"):
        spec_params["filter_has_paste"] = True

    for key in ("min_messages", "max_messages", "min_words", "max_words", "sample"):
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
    if not hmac.compare_digest(auth_header[7:], auth_token):
        return _AuthResult(allowed=False, reason="unauthorized")
    return _AuthResult(allowed=True, reason=None)


def _check_host_admission_logic(host_header: str, api_host: str) -> bool:
    """Pure logic: is *host_header* an allowed Host for this daemon?

    DNS rebinding: an attacker-controlled domain name can be made to
    resolve to 127.0.0.1, after which a hostile page's "same-origin"
    requests genuinely originate from the local machine — client-IP
    loopback checks cannot detect this, because the TCP peer really is
    localhost. The Host header is the one signal a browser cannot forge
    to match ours: it always names the domain the page believes it is
    talking to, which can never equal our loopback names or the
    configured ``api_host`` unless the attacker already controls DNS for
    that literal string.

    An ABSENT Host header is allowed through: every real HTTP/1.1 browser
    request carries one (RFC 7230 §5.4), so DNS rebinding — which relies
    on the browser itself constructing the request — can never omit it.
    Only a raw non-browser local client could send no Host at all, and
    that threat class is already accepted by this daemon's trust model
    (``_check_auth`` grants full access to any local caller when no
    token is configured). Rejecting on presence-and-mismatch, not on
    absence, closes the real hole without reshaping that model.

    A malformed Host (e.g. unmatched IPv6 brackets) makes ``urlsplit``
    raise ``ValueError`` — treated as a refusal, not left to propagate as
    an unhandled exception.
    """
    if not host_header:
        return True
    try:
        hostname = urlsplit(f"//{host_header}").hostname or ""
    except ValueError:
        return False
    return is_loopback_host(hostname) or hostname == api_host


class _AuthResult:
    def __init__(self, *, allowed: bool, reason: str | None) -> None:
        self.allowed = allowed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.allowed


def _http_session_projection_adapters() -> HttpSessionProjectionAdapters:
    return HttpSessionProjectionAdapters(attachment=attachment_to_envelope, paste_spans=envelope_paste_spans)


class DaemonAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the daemon API server.

    Runs async archive operations via ``asyncio.run()`` in a thread pool
    worker. This is safe because each request runs in its own thread.
    """

    server: DaemonAPIHTTPServer
    # Whether this request's status line has been written; once it has, an
    # escaped exception can no longer be answered with its own status.
    _response_started = False

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_response_only(self, code: int, message: str | None = None) -> None:
        # An interim ``100 Continue`` precedes the real answer; it does not start it.
        if code >= 200:
            self._response_started = True
        super().send_response_only(code, message)

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
    def _web_credentials(self) -> WebCredentialRegistry:
        return self.server.web_credentials

    @property
    def _client_host(self) -> str:
        """Extract client IP from the request."""
        client_address: object = self.client_address
        return str(client_address[0]) if isinstance(client_address, tuple) else "127.0.0.1"

    def _web_credential_token(self) -> str | None:
        token = read_web_credential_cookie(self.headers.get("Cookie", ""))
        if token and not self._peer_is_owner():
            # The cookie's bytes may have leaked to another local uid's process
            # (no port scoping; see ``polylogue.daemon.peer_identity``); honor it only
            # from a peer the kernel itself attributes to this process's uid.
            return None
        return token

    def _peer_is_owner(self) -> bool:
        """Whether the TCP peer of this request is owned by this process's own uid.

        Real for a live socket. ``tests/infra/daemon_http_harness.py`` fakes the
        transport for handler-logic tests and sets this to ``True`` by default,
        since there is no kernel connection-table entry for a fabricated
        ``client_address``.
        """
        # One connection has one peer: the kernel lookup (an ``lsof`` run on
        # macOS) is made once per connection, not once per credential read.
        cached = self.__dict__.get("_peer_owner_decision")
        if isinstance(cached, bool):
            return cached
        client_address: object = self.client_address
        server_address = getattr(self.server, "server_address", None)
        decision = (
            isinstance(client_address, tuple)
            and len(client_address) >= 2
            and isinstance(server_address, tuple)
            and len(server_address) >= 2
            and peer_socket_owned_by_current_uid(
                local_ip=str(server_address[0]),
                local_port=int(server_address[1]),
                remote_ip=str(client_address[0]),
                remote_port=int(client_address[1]),
            )
        )
        self.__dict__["_peer_owner_decision"] = decision
        return decision

    def _web_credential_decision(self, required_scope: WebCredentialScope) -> WebCredentialDecision:
        return self._web_credentials.validate(
            self._web_credential_token(),
            required_scope=required_scope,
            host_header=self.headers.get("Host", ""),
            origin_header=self.headers.get("Origin", ""),
            referer_header=self.headers.get("Referer", ""),
            fetch_site=self.headers.get("Sec-Fetch-Site", ""),
        )

    def _is_web_client_request(self) -> bool:
        return self.headers.get("X-Polylogue-Web-Client", "") == "1" or bool(self.headers.get("Sec-Fetch-Site", ""))

    def _send_web_credential_error(self, decision: WebCredentialDecision) -> None:
        status = (
            HTTPStatus.FORBIDDEN
            if decision.state in {"web_credential_wrong_origin", "web_credential_insufficient_scope"}
            else HTTPStatus.UNAUTHORIZED
        )
        self._send_error(status, decision.state, extra_headers=decision.response_headers())

    def _check_auth(
        self,
        required_scope: WebCredentialScope = "read",
        *,
        allow_web: bool = True,
        refuse: Callable[[HTTPStatus, str], None] | None = None,
    ) -> bool:
        """Validate a machine bearer or a scoped first-party web credential.

        When no token is configured the API is open. This server object only
        ever receives ``auth_token=None`` when the operator explicitly opted
        out via ``--api-allow-no-auth`` (``polylogue.daemon.api_auth``
        auto-mints/loads a persisted token by default, mirroring the
        browser-capture receiver's contract; polylogue-rzve). When a token
        IS configured, all clients — including localhost — must present it.
        Loopback is not a security boundary when a browser on the same host
        can reach the daemon.

        Native ``EventSource`` receives the same HttpOnly cookie as fetch, so
        no credential is ever accepted from a query parameter.
        """
        deny = refuse if refuse is not None else self._send_error
        auth_header = self.headers.get("Authorization", "")
        if not self._auth_token:
            return True
        if auth_header:
            result = _check_auth_logic(self._auth_token, self._client_host, auth_header)
            if not result.allowed:
                deny(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return result.allowed
        if allow_web and self._web_credential_token():
            decision = self._web_credential_decision(required_scope)
            if decision.allowed:
                return True
            self._send_web_credential_error(decision)
            return False
        if allow_web and self._is_web_client_request():
            decision = self._web_credentials.validate(
                None,
                required_scope=required_scope,
                host_header=self.headers.get("Host", ""),
                origin_header=self.headers.get("Origin", ""),
                referer_header=self.headers.get("Referer", ""),
                fetch_site=self.headers.get("Sec-Fetch-Site", ""),
            )
            self._send_web_credential_error(decision)
            return False
        deny(HTTPStatus.UNAUTHORIZED, "unauthorized")
        return False

    def _cli_mutation_principal(self, capability: str) -> MutationPrincipal:
        """Derive, never accept, the audit principal for a CLI mutation request."""

        from polylogue.operations.mutation_transaction import MutationPrincipal

        auth_header = self.headers.get("Authorization", "")
        if not self._auth_token:
            actor_ref = "daemon:unauthenticated-loopback"
            role_label = "daemon-loopback-no-auth"
        elif auth_header.startswith("Bearer "):
            actor_ref = f"daemon:bearer:{hashlib.sha256(auth_header[7:].encode()).hexdigest()}"
            role_label = "daemon-authenticated"
        else:
            raise RuntimeError("authenticated CLI mutation request has no bearer principal")
        return MutationPrincipal(
            actor_ref=actor_ref,
            capabilities=frozenset({capability}),
            surface="cli",
            role_label=role_label,
        )

    def _check_host_admission(self, *, credential_request: bool = False) -> bool:
        """Reject requests whose Host header does not name this daemon.

        See :func:`_check_host_admission_logic` for the rationale. Applied
        before EVERY GET dispatch branch (web shell bootstrap, paste/
        attachment pages, healthz/metrics probes, and the authenticated
        API route tables) and to all POST/DELETE requests.
        The web shell's own bootstrap still works: its Host is loopback
        (or the configured ``api_host``), which this check admits — only
        a foreign Host (the DNS-rebinding signature) is refused. Health/
        metrics probes leak real information (PID, archive_root, DB
        sizes, exception text) and are not merely inert booleans, so they
        are gated too; the documented deployment pattern
        (``docs/docker-compose.yaml``) already targets ``127.0.0.1``/
        ``localhost`` and is unaffected.
        """
        host_header = self.headers.get("Host", "")
        if _check_host_admission_logic(host_header, self._api_host):
            return True
        if credential_request:
            self._send_web_credential_error(WebCredentialDecision(False, "web_credential_wrong_origin"))
        else:
            self._send_error(HTTPStatus.FORBIDDEN, "host_not_allowed")
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _client_disconnected(self) -> bool:
        return _socket_peer_disconnected(getattr(self, "connection", None))

    def _raise_if_client_disconnected(self) -> None:
        if self._client_disconnected():
            raise _ClientDisconnectedDuringComputeError(_FACET_CANCELLED_REASON)

    def _request_id_header(self) -> str | None:
        request_id = self.headers.get("X-Request-ID", "")
        if not request_id or len(request_id) > 128 or "\r" in request_id or "\n" in request_id:
            return None
        return request_id

    def _send_request_id_header(self) -> None:
        request_id = self._request_id_header()
        if request_id:
            self.send_header("X-Request-ID", request_id)

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
        self._send_request_id_header()
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def _send_webui_html(self, status: HTTPStatus, body: str) -> None:
        """Send the no-inline-code WebUI document with a restrictive CSP."""

        raw = body.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        self._send_request_id_header()
        self.end_headers()
        self.wfile.write(raw)

    def _send_webui_asset(self, asset: WebUIAsset) -> None:
        """Send one content-hashed Vite asset with immutable caching."""

        if self.headers.get("If-None-Match", "") == asset.etag:
            self.send_response(HTTPStatus.NOT_MODIFIED.value)
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.send_header("ETag", asset.etag)
            self.send_header("X-Content-Type-Options", "nosniff")
            self._send_request_id_header()
            self.end_headers()
            return
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", asset.content_type)
        self.send_header("Content-Length", str(len(asset.body)))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.send_header("ETag", asset.etag)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self._send_request_id_header()
        self.end_headers()
        self.wfile.write(asset.body)

    def _send_json_with_cookie(
        self,
        status: HTTPStatus,
        payload: object,
        *,
        set_cookie: str,
        credential_state: str,
    ) -> None:
        """Send public lifecycle JSON while isolating protected cookie transport."""

        raw = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Set-Cookie", set_cookie)
        self.send_header("X-Polylogue-Web-Credential-State", credential_state)
        self._send_request_id_header()
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
        self._send_request_id_header()
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(
        self,
        status: HTTPStatus,
        code: str,
        detail: str | None = None,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_payload: Mapping[str, object] | None = None,
    ) -> None:
        """Emit the canonical daemon error envelope.

        Every daemon error response shares one machine-output contract
        (#1818): ``{"ok": false, "error": <code>, "detail": <str|null>,
        "field": <str|null>}``, produced by ``QueryErrorPayload`` so the
        decorator path, the cursor-rejection path, and ad-hoc 4xx sites all
        serialize identically. Health/status payloads use a different,
        deliberately separate shape and do not route through here.
        """
        payload = QueryErrorPayload(error=code, detail=detail).model_dump(mode="json")
        if extra_payload:
            payload.update(extra_payload)
        self._send_json(status, payload, extra_headers=extra_headers)

    def _reject_operation(self, status: HTTPStatus, code: str, detail: str | None = None) -> None:
        """Refuse a machine operation before dispatch, marked so no client calls it indeterminate.

        A pre-dispatch refusal on ``/api/operation`` proves nothing ran, so it
        must never be read as "the mutation may have happened". The UDS
        transport marks its refusals with ``pre_dispatch`` and clients key off
        that marker (polylogue-ji49p); the TCP route's plain ``_send_error``
        envelope carried no protocol/outcome fields, so every one of its
        refusals degraded into an indeterminate mutation at the client.
        """
        from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL

        self._send_error(
            status,
            code,
            detail,
            extra_payload={
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "outcome": "rejected",
                "pre_dispatch": True,
                "error": {"code": code, "detail": detail, "retryable": False},
            },
        )

    def _parse_path(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        path = [unquote(segment) for segment in parsed.path.strip("/").split("/")]
        # Blank values are kept so a route can tell "sent empty" from "absent";
        # _get_param still reads a blank as absent for ordinary parameters.
        params = parse_qs(parsed.query, keep_blank_values=True)
        return path, params

    def _get_param(self, params: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
        values = params.get(key)
        if values and values[0] != "":
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

    def _mutation_wait_budget_s(self) -> float:
        """The bound this request's mutating wait carries.

        A client may declare a shorter deadline with ``X-Polylogue-Deadline-Ms``;
        anything absent, unparseable, or outside the accepted band falls back to
        the route default so a header can never remove the bound.
        """

        headers = getattr(self, "headers", None)
        raw = headers.get("X-Polylogue-Deadline-Ms", "") if headers is not None else ""
        try:
            declared_s = float(raw) / 1000.0
        except (TypeError, ValueError):
            return _MUTATION_WAIT_TIMEOUT_S
        if declared_s <= 0 or declared_s > _MUTATION_WAIT_TIMEOUT_MAX_S:
            return _MUTATION_WAIT_TIMEOUT_S
        return declared_s

    @contextlib.contextmanager
    def _observe_read_peer(self, cancel: Callable[[], None]) -> Iterator[None]:
        from polylogue.daemon.operation_disconnect import observe_peer_disconnect

        with observe_peer_disconnect(self.connection) as disconnected:
            remove = disconnected.add_listener(cancel)
            try:
                yield
            finally:
                remove()

    def _sync_run(self, handler: Callable) -> object:  # type: ignore[type-arg]
        """Run reads on compute workers and admitted writes on writer workers."""
        mutating = getattr(self, "_write_gate_depth", 0) > 0
        if mutating:
            from polylogue.core.write_lease import WriteLeaseDelegation

            bridge = getattr(self.server, "write_bridge", None)
            delegation = getattr(self, "_write_delegation", None)
            if not isinstance(bridge, DaemonWriteThreadBridge) or delegation is None:
                raise RuntimeError("a mutating route needs its admitted writer delegation")
            budget_s = self._mutation_wait_budget_s()
            with log_span(
                "daemon.http.scheduled_route",
                route=_request_path_for_log(getattr(self, "path", "") or ""),
                method=getattr(self, "command", "") or "",
                domain="control",
            ) as route_span:
                self._last_queue_delay_ms = 0
                try:
                    result = bridge.run_admitted_async(
                        cast(WriteLeaseDelegation, delegation),
                        lambda: self._run_archive_query(handler),
                        timeout=budget_s,
                    )
                except FutureTimeoutError as error:
                    route_span.set(reason="mutation_indeterminate", timeout_ms=round(budget_s * 1000, 3))
                    raise DaemonMutationIndeterminate(
                        f"mutation did not complete within {budget_s:.0f}s; "
                        "it may still be in flight -- re-read before retrying"
                    ) from error
                route_span.ok()
                return result

        kernel = getattr(self.server, "execution_kernel", None)
        if isinstance(kernel, BoundedComputeAdapter):
            from polylogue.core.compute import CancellationHandle

            cancellation = CancellationHandle()
            with log_span(
                "daemon.http.scheduled_route",
                route=_request_path_for_log(getattr(self, "path", "") or ""),
                method=getattr(self, "command", "") or "",
                domain="interactive-read",
            ) as route_span:
                submitted = kernel.submit(
                    propagate(lambda: asyncio.run(self._run_archive_query(handler))),
                    admission_class="interactive-read",
                    estimated_bytes=1024 * 1024,
                    cancellation=cancellation,
                )
                try:
                    with self._observe_read_peer(cancellation.cancel):
                        result = submitted.future.result(timeout=_ARCHIVE_QUERY_TIMEOUT_S)
                except FutureTimeoutError as error:
                    cancellation.cancel()
                    submitted.future.cancel()
                    route_span.set(reason="archive_query_timeout", timeout_ms=round(_ARCHIVE_QUERY_TIMEOUT_S * 1000, 3))
                    raise TimeoutError(
                        f"archive query did not complete within {_ARCHIVE_QUERY_TIMEOUT_S:.0f}s; "
                        "the daemon may be busy with catch-up ingestion/embedding"
                    ) from error
                else:
                    route_span.ok()
                    return result
                finally:
                    self._last_queue_delay_ms = int(submitted.queue_delay_s * 1000)
                    route_span.set(elapsed_ms=round(submitted.queue_delay_s * 1000, 3))
        return asyncio.run(self._run_archive_query(handler))

    @contextlib.contextmanager
    def _write_gate(self, actor: str) -> Iterator[None]:
        """Hold the daemon's main-loop writer lease across this sync request."""
        bridge = getattr(self.server, "write_bridge", None)
        if bridge is None:
            # Direct handler unit doubles predate the server-owned bridge. A
            # real DaemonAPIHTTPServer always installs one in ``__init__``.
            yield
            return
        depth = getattr(self, "_write_gate_depth", 0)
        previous = getattr(self, "_write_delegation", None)
        with cast(DaemonWriteThreadBridge, bridge).hold(actor) as delegation:
            self._write_gate_depth = depth + 1
            # The gate admits this request; the delegation is what authorizes
            # its body, which runs on the admitted writer worker in its own event loop.
            self._write_delegation = delegation
            try:
                yield
            finally:
                self._write_gate_depth = depth
                self._write_delegation = previous

    def do_OPTIONS(self) -> None:
        self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed")

    # ------------------------------------------------------------------
    # Route dispatch
    # ------------------------------------------------------------------

    def _dispatch_get(self, path: list[str], params: dict[str, list[str]]) -> None:
        """Dispatch GET requests via route table."""
        if not self._check_host_admission():
            return
        if self._reject_credential_query():
            return

        # The typed WebUI is the canonical browser surface. Every browser
        # route enters the same SSR handlers, asset-manifest boundary, and
        # authentication checks.
        if path == ["web-auth", "sign-in.js"]:
            self._serve_web_sign_in_script()
            return
        if path == ["web-auth", "sign-in"]:
            # The ticket exchange page is served whether or not the browser is
            # already credentialed, so a ticket fragment is always consumed and
            # cleared rather than left in the address bar of an archive page.
            self._send_webui_html(HTTPStatus.OK, WEB_SIGN_IN_HTML)
            return
        if path == [""]:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_archive_overview()
            return
        if path == ["observability"]:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_observability()
            return
        if path == ["cost"]:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_cost()
            return
        if path == ["sessions"]:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_session_list(params)
            return
        if len(path) == 2 and path[0] == "sessions" and bool(path[1]):
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_session_read(path[-1])
            return
        if path == ["search"]:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_search(params)
            return
        if len(path) == 2 and path[0] == "assets" and bool(path[1]):
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_asset(path[-1])
            return

        # The short session deep link is part of the typed reader contract.
        # Keep its stable URL while routing it through the same SSR handler as
        # the long form above.
        if len(path) == 2 and path[0] == "s" and bool(path[1]):
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_session_read(path[1])
            return

        if len(path) == 2 and path[0] == "w" and path[1] in workspace_routes.WORKSPACE_SHELL_MODES:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_workspace(path[1], params)
            return

        if path == ["p"]:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_pastes(params)
            return

        if path == ["a"]:
            if not self._check_shell_bootstrap_access():
                return
            self._serve_webui_attachments(params)
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
            from polylogue.paths import archive_root

            handle_metrics(self, archive_root() / "index.db")
            return

        required_scope: WebCredentialScope = "events" if path == ["api", "events"] else "read"
        if not self._check_auth(required_scope):
            return

        static_route = next((route for route in _static_get_routes() if tuple(path) == route.segments), None)
        if static_route is not None:
            handler = cast(Callable[..., None], getattr(self, static_route.handler_name))
            if static_route.passes_path:
                handler(path, params)
            elif static_route.passes_params:
                handler(params)
            else:
                handler()
            return

        for route in _parameterized_get_routes():
            expected_len = len(route.prefix) + 1 + len(route.suffix)
            if (
                len(path) == expected_len
                and tuple(path[: len(route.prefix)]) == route.prefix
                and path[len(route.prefix)]
                and tuple(path[len(route.prefix) + 1 :]) == route.suffix
            ):
                handler = cast(Callable[..., None], getattr(self, route.handler_name))
                identifier = path[len(route.prefix)]
                if route.pattern.startswith(("/api/sessions/:id", "/api/insights/sessions/:id")):
                    identifier = _normalize_session_route_id(identifier)
                if route.passes_path:
                    handler(path, params)
                elif route.passes_params:
                    handler(identifier, params)
                else:
                    handler(identifier)
                return

        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    # Request boundary: an exception escaping any route is answered through
    # ``_answer_route_exception`` rather than left to socketserver, which
    # would print a traceback and drop the connection with no status. Client
    # disconnects (#1677) are demoted to debug there.

    def do_GET(self) -> None:
        self._response_started = False
        try:
            path, params = self._parse_path()
            self._dispatch_get(path, params)
        except Exception as exc:
            _answer_route_exception(self, exc, route=_request_path_for_log(self.path), method="GET")

    def _reject_credential_query(self) -> bool:
        names = {
            key.lower()
            for key, _value in parse_qsl(urlparse(self.path).query, keep_blank_values=True)
            if key.lower() in _CREDENTIAL_QUERY_PARAMETERS
        }
        if not names:
            return False
        self._send_error(
            HTTPStatus.BAD_REQUEST,
            "credential_in_query",
            "credentials must use Authorization or the protected first-party cookie",
        )
        return True

    def _check_cross_origin(self, *, refuse: Callable[[HTTPStatus, str], None] | None = None) -> bool:
        """Reject browser cross-origin POSTs to mutating endpoints.

        Returns True if the request is allowed, sends 403 and returns
        False if the Origin header indicates a cross-origin browser request.
        """
        origin = self.headers.get("Origin", "")
        if exact_origin_allowed(origin, self.headers.get("Host", "")):
            return True
        (refuse if refuse is not None else self._send_error)(HTTPStatus.FORBIDDEN, "cross_origin_denied")
        return False

    def _shell_request_authorized(self) -> bool:
        """Is this browser-shell request backed by the owner's credential?

        Loopback is not identity: any local uid can open a TCP connection to
        the daemon. When a bearer token is configured, shell HTML (which
        embeds archive content) requires either that bearer or a valid
        first-party web credential cookie, exactly like the ``/api`` routes.
        """

        if not self._auth_token:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header:
            return bool(_check_auth_logic(self._auth_token, self._client_host, auth_header))
        token = self._web_credential_token()
        if not token:
            return False
        fetch_site = self.headers.get("Sec-Fetch-Site", "")
        # A typed URL or bookmark sends the SameSite=Strict cookie with
        # ``Sec-Fetch-Site: none`` and no Origin/Referer; the cookie is only
        # sent on same-site or user-initiated navigations, so that is admitted
        # for a top-level shell GET once the Host and record have matched.
        if fetch_site == "none" and self.headers.get("Sec-Fetch-Mode", "") in {"", "navigate"}:
            fetch_site = "same-origin"
        return self._web_credentials.validate(
            token,
            required_scope="read",
            host_header=self.headers.get("Host", ""),
            origin_header=self.headers.get("Origin", ""),
            referer_header=self.headers.get("Referer", ""),
            fetch_site=fetch_site,
        ).allowed

    def _check_shell_bootstrap_access(self) -> bool:
        """Serve shell HTML only to the credentialed owner.

        A browser navigation without a credential receives the sign-in page;
        any other client receives the ordinary typed 401.
        """

        if self._shell_request_authorized():
            return True
        if "text/html" in self.headers.get("Accept", "") or self.headers.get("Sec-Fetch-Mode", "") == "navigate":
            self._send_webui_html(HTTPStatus.UNAUTHORIZED, WEB_SIGN_IN_HTML)
        else:
            self._send_error(HTTPStatus.UNAUTHORIZED, "unauthorized")
        return False

    def _dispatch_declared_mutation(self, method: str, path: list[str], params: dict[str, list[str]]) -> bool:
        route = next((item for item in _declared_mutation_routes(method) if item.matches(path)), None)
        if route is None:
            return False
        declaration = route.declaration
        if declaration.auth_policy == "first_party_same_origin":
            pass  # The bound credential lifecycle handler validates its own origin.
        elif declaration.auth_policy == "bearer_if_configured_and_same_origin":
            if not self._check_auth(allow_web=False):
                return True
        elif not self._check_auth(declaration.auth_scope):
            return True
        if (
            declaration.auth_policy in {"credential_and_same_origin", "bearer_if_configured_and_same_origin"}
            and not self._check_cross_origin()
        ):
            return True
        handler = cast(Callable[..., None], getattr(self, route.handler_name))
        actor = {
            "_handle_reset": "http.reset",
            "_handle_mcp_call_log": "http.telemetry.mcp-call",
        }.get(route.handler_name, f"http.{method.lower()}.{declaration.kernel.declaration_id}")
        if path[:2] == ["api", "user"] and len(path) > 2:
            actor = f"http.user.{path[2]}.{method.lower()}"
        gate = self._write_gate(actor) if declaration.write_gate else contextlib.nullcontext()
        with gate:
            if declaration.passes_path:
                handler(path, params)
            elif declaration.passes_params:
                handler(params)
            else:
                handler()
        return True

    def do_POST(self) -> None:
        self._response_started = False
        try:
            self._do_post_impl()
        except Exception as exc:
            _answer_route_exception(self, exc, route=_request_path_for_log(self.path), method="POST")

    def _do_post_impl(self) -> None:
        path, params = self._parse_path()
        web_auth_request = path == ["api", "web-auth", "session"]

        if not self._check_host_admission(credential_request=web_auth_request):
            return
        if self._reject_credential_query():
            return

        if self._dispatch_declared_mutation("POST", path, params):
            return

        if not self._check_auth("user_state"):
            return
        if not self._check_cross_origin():
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_DELETE(self) -> None:
        self._response_started = False
        try:
            self._do_delete_impl()
        except Exception as exc:
            _answer_route_exception(self, exc, route=_request_path_for_log(self.path), method="DELETE")

    def _do_delete_impl(self) -> None:
        path, params = self._parse_path()
        web_auth_request = path == ["api", "web-auth", "session"]

        if not self._check_host_admission(credential_request=web_auth_request):
            return
        if self._reject_credential_query():
            return

        if self._dispatch_declared_mutation("DELETE", path, params):
            return

        if not self._check_auth("user_state"):
            return
        if not self._check_cross_origin():
            return

        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    # ------------------------------------------------------------------
    # Typed WebUI
    # ------------------------------------------------------------------

    def _handle_web_auth_bootstrap(self) -> None:
        """Rotate a short-lived first-party credential into an HttpOnly cookie."""

        if not (is_loopback_host(self._api_host) and is_loopback_host(self._client_host)):
            self._send_web_credential_error(WebCredentialDecision(False, "web_credential_wrong_origin"))
            return
        origin = same_origin_from_headers(
            self.headers.get("Origin", ""),
            self.headers.get("Host", ""),
        )
        if origin is None:
            self._send_error(
                HTTPStatus.FORBIDDEN,
                "web_credential_wrong_origin",
                extra_headers={
                    "Cache-Control": "no-store",
                    "X-Polylogue-Web-Credential-State": "web_credential_wrong_origin",
                },
            )
            return
        if self._auth_token and not self._web_bootstrap_proof():
            self._send_web_credential_error(WebCredentialDecision(False, "web_credential_missing"))
            return
        issued = self._web_credentials.issue(
            origin,
            previous_token=self._web_credential_token(),
            scopes=WEB_CREDENTIAL_SCOPES,
        )
        self._send_json_with_cookie(
            HTTPStatus.CREATED,
            WebCredentialBootstrapPayload(credential=issued.public_payload()).model_dump(mode="json"),
            set_cookie=credential_cookie(
                issued.token,
                ttl_s=self._web_credentials.ttl_s,
                secure=issued.origin.startswith("https://"),
            ),
            credential_state="ready",
        )

    def _web_bootstrap_proof(self) -> bool:
        """Does this bootstrap request prove it acts for the archive owner?

        Accepted proofs: the daemon bearer token, a one-time sign-in ticket
        minted against that bearer (``POST /api/web-auth/ticket``), or a
        still-valid web credential being rotated. A bare loopback request is
        not proof: every local uid can reach loopback.
        """

        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            presented = auth_header[7:]
            if self._auth_token and hmac.compare_digest(presented, self._auth_token):
                return True
            return self._web_credentials.redeem_sign_in_ticket(presented)
        if auth_header:
            return False
        return self._web_credential_decision("read").allowed

    def _handle_web_auth_ticket(self) -> None:
        """Mint a one-time browser sign-in ticket for a bearer-authenticated caller."""

        ticket, expires_at = self._web_credentials.issue_sign_in_ticket()
        self._send_json(
            HTTPStatus.CREATED,
            WebSignInTicketPayload(
                ticket=ticket,
                expires_at=datetime.fromtimestamp(expires_at, tz=UTC),
            ).model_dump(mode="json"),
            extra_headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        )

    def _serve_web_sign_in_script(self) -> None:
        """Serve the static sign-in page script; it carries no archive data."""

        raw = WEB_SIGN_IN_SCRIPT.encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self._send_request_id_header()
        self.end_headers()
        self.wfile.write(raw)

    def _handle_web_auth_revoke(self) -> None:
        """Revoke the current first-party credential and clear its cookie."""

        decision = self._web_credential_decision("user_state")
        if not decision.allowed:
            self._send_web_credential_error(decision)
            return
        self._web_credentials.revoke(self._web_credential_token())
        origin = self.headers.get("Origin", "")
        self._send_json_with_cookie(
            HTTPStatus.OK,
            WebCredentialRevocationPayload(
                credential=WebCredentialRevokedPayload(),
            ).model_dump(mode="json"),
            set_cookie=expired_credential_cookie(secure=origin.startswith("https://")),
            credential_state="web_credential_revoked",
        )

    def _serve_webui_secondary(
        self, *, title: str, heading: str, description: str, payload: Mapping[str, object] | None, empty: str
    ) -> None:
        from polylogue.daemon.webui import (
            WebUIAssetBundle,
            WebUIAssetError,
            render_typed_data_page,
            render_webui_asset_error,
        )

        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
            body = render_typed_data_page(
                bundle, title=title, heading=heading, description=description, payload=payload, empty=empty
            )
        except WebUIAssetError as exc:
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return
        self._send_webui_html(HTTPStatus.OK, body)

    def _serve_webui_compare(self, payload: Mapping[str, object] | None) -> None:
        from polylogue.daemon.webui import (
            WebUIAssetBundle,
            WebUIAssetError,
            render_compare_page,
            render_webui_asset_error,
        )

        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
            body = render_compare_page(
                bundle,
                payload=payload,
                empty="Choose two valid session targets to compare.",
            )
        except WebUIAssetError as exc:
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return
        self._send_webui_html(HTTPStatus.OK, body)

    def _serve_webui_workspace(self, mode: str, params: dict[str, list[str]]) -> None:
        archive_root = _web_reader_archive_root()
        payload: Mapping[str, object] | None = None
        window = workspace_routes.parse_message_window(self, params)
        if archive_root is not None and mode == "stack":
            ids = workspace_routes.parse_id_list(params)
            if ids:
                payload = self._do_archive_stack(archive_root, ids, self._get_param(params, "focus"), window)
        elif mode == "compare":
            if archive_root is not None:
                left = self._get_param(params, "left")
                right = self._get_param(params, "right")
                if left and right:
                    result = self._do_archive_compare(
                        archive_root, left, right, self._get_param(params, "align", "prompt") or "prompt", window
                    )
                    payload = result if isinstance(result, Mapping) else None
            # Compare's envelope nests whole session payloads under four keys.
            # The generic typed-data projection renders those through ``str``,
            # which is an escaped Python repr of every message, repeated once
            # per key that holds it. Compare owns a structure-aware renderer.
            self._serve_webui_compare(payload)
            return
        self._serve_webui_secondary(
            title=f"Workspace · {mode}",
            heading=f"Workspace {mode}",
            description="A typed, bounded workspace projection with explicit missing and degraded targets.",
            payload=payload,
            empty="Choose valid session targets to load this workspace view.",
        )

    def _serve_webui_pastes(self, params: dict[str, list[str]]) -> None:
        limit = max(1, min(self._get_int(params, "limit", 200), 500))
        offset = max(0, self._get_int(params, "offset", 0))
        try:
            payload = self._sync_run(lambda poly: self._do_paste_browser(poly, limit=limit, offset=offset))
        except (DaemonBackpressureError, TimeoutError):
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "archive_read_unavailable", "Retry this request shortly.")
            return
        self._serve_webui_secondary(
            title="Pastes",
            heading="Paste evidence",
            description="Messages with structured paste evidence and bounded source links.",
            payload=payload if isinstance(payload, Mapping) else None,
            empty="No paste evidence is available in this archive.",
        )

    def _serve_webui_attachments(self, params: dict[str, list[str]]) -> None:
        limit = max(1, min(self._get_int(params, "limit", 200), 500))
        offset = max(0, self._get_int(params, "offset", 0))
        try:
            payload = self._sync_run(
                lambda poly: self._do_attachment_library(
                    poly,
                    limit=limit,
                    offset=offset,
                    mime_filter=self._get_param(params, "mime") or "",
                    state_filter=self._get_param(params, "state") or "",
                    session_filter=self._get_param(params, "session") or "",
                )
            )
        except (DaemonBackpressureError, TimeoutError):
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "archive_read_unavailable", "Retry this request shortly.")
            return
        self._serve_webui_secondary(
            title="Attachments",
            heading="Attachment library",
            description="Attachment metadata with honest availability and preview states.",
            payload=payload if isinstance(payload, Mapping) else None,
            empty="No attachments are available in this archive.",
        )

    def _serve_webui_archive_overview(self) -> None:
        from polylogue.archive.query.execution_control import (
            QueryCancelledError,
            QueryTimeoutError,
            QueryWorkBudgetExceededError,
        )
        from polylogue.daemon.webui import (
            WebUIAssetBundle,
            WebUIAssetError,
            load_archive_overview_page,
            render_archive_overview_page,
            render_webui_asset_error,
        )

        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
        except WebUIAssetError as exc:
            emit(
                "daemon.webui.assets_unavailable",
                level=ERROR,
                outcome="degraded",
                reason="asset_discovery_failed",
                route=_request_path_for_log(self.path),
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return

        archive_root = _web_reader_archive_root()
        if archive_root is None:
            body = render_archive_overview_page(
                bundle,
                None,
                notice="The archive is unavailable or requires a schema rebuild.",
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, body)
            return
        try:
            page = load_archive_overview_page(archive_root)
        except (QueryCancelledError, QueryTimeoutError, QueryWorkBudgetExceededError):
            body = render_archive_overview_page(
                bundle,
                None,
                notice="The bounded archive query could not complete within its execution budget.",
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, body)
            return
        except sqlite3.OperationalError as exc:
            emit(
                "daemon.webui.page_read_failed",
                level=ERROR,
                outcome="error",
                reason="archive_busy" if _is_sqlite_busy_error(exc) else "sqlite_error",
                route="archive_overview",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            status = HTTPStatus.SERVICE_UNAVAILABLE if _is_sqlite_busy_error(exc) else HTTPStatus.INTERNAL_SERVER_ERROR
            notice = (
                "The archive is temporarily busy; retry the overview shortly."
                if status is HTTPStatus.SERVICE_UNAVAILABLE
                else "The archive overview could not be rendered."
            )
            self._send_webui_html(status, render_archive_overview_page(bundle, None, notice=notice))
            return
        except Exception as exc:
            emit(
                "daemon.webui.page_render_failed",
                level=ERROR,
                outcome="error",
                reason="render_failed",
                route="archive_overview",
                status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                render_archive_overview_page(bundle, None, notice="The archive overview could not be rendered."),
            )
            return
        self._send_webui_html(HTTPStatus.OK, render_archive_overview_page(bundle, page))

    def _serve_webui_session_list(self, params: dict[str, list[str]]) -> None:
        from polylogue.archive.query.spec import (
            DEFAULT_SESSION_LIST_LIMIT,
            QuerySpecError,
            clamp_query_limit,
        )
        from polylogue.daemon.webui import (
            WebUIAssetBundle,
            WebUIAssetError,
            render_session_list_page,
            render_webui_asset_error,
        )

        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
        except WebUIAssetError as exc:
            emit(
                "daemon.webui.assets_unavailable",
                level=ERROR,
                outcome="degraded",
                reason="asset_discovery_failed",
                route=_request_path_for_log(self.path),
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return

        filters = {
            "origin": self._get_param(params, "origin") or "",
            "since": self._get_param(params, "since") or "",
            "repo": self._get_param(params, "repo") or "",
        }
        archive_root = _web_reader_archive_root()
        if archive_root is None:
            body = render_session_list_page(
                bundle,
                None,
                filters,
                notice="The archive is unavailable or requires a schema rebuild.",
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, body)
            return
        limit = clamp_query_limit(
            self._get_int(params, "limit", DEFAULT_SESSION_LIST_LIMIT),
            default=DEFAULT_SESSION_LIST_LIMIT,
        )
        offset = max(0, self._get_int(params, "offset", 0))

        async def _list(poly: Polylogue) -> object:
            return await self._do_list(poly, query_params, limit, offset, route="/sessions")

        try:
            query_params = _build_query_spec_params(params, self)
            cursor = self._get_param(params, "cursor")
            if cursor:
                query_params["cursor"] = cursor
            page = self._sync_run(_list)
        except QuerySpecError as exc:
            self._send_webui_html(
                HTTPStatus.BAD_REQUEST,
                render_session_list_page(bundle, None, filters, notice=str(exc)),
            )
            return
        except sqlite3.OperationalError as exc:
            emit(
                "daemon.webui.page_read_failed",
                level=ERROR,
                outcome="error",
                reason="archive_busy" if _is_sqlite_busy_error(exc) else "sqlite_error",
                route="session_list",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            status = HTTPStatus.SERVICE_UNAVAILABLE if _is_sqlite_busy_error(exc) else HTTPStatus.INTERNAL_SERVER_ERROR
            notice = (
                "The archive is temporarily busy; retry the session list shortly."
                if status is HTTPStatus.SERVICE_UNAVAILABLE
                else "The session list could not be rendered."
            )
            self._send_webui_html(status, render_session_list_page(bundle, None, filters, notice=notice))
            return
        except Exception as exc:
            emit(
                "daemon.webui.page_render_failed",
                level=ERROR,
                outcome="error",
                reason="render_failed",
                route="session_list",
                status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                render_session_list_page(bundle, None, filters, notice="The session list could not be rendered."),
            )
            return
        if not isinstance(page, Mapping):
            self._send_webui_html(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                render_session_list_page(bundle, None, filters, notice="The session list could not be rendered."),
            )
            return
        self._send_webui_html(HTTPStatus.OK, render_session_list_page(bundle, page, filters))

    def _serve_webui_session_read(self, session_id: str) -> None:
        from polylogue.archive.query.spec import DEFAULT_MESSAGE_PAGE_LIMIT
        from polylogue.daemon.webui import (
            WebUIAssetBundle,
            WebUIAssetError,
            render_session_read_page,
            render_webui_asset_error,
        )

        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
        except WebUIAssetError as exc:
            emit(
                "daemon.webui.assets_unavailable",
                level=ERROR,
                outcome="degraded",
                reason="asset_discovery_failed",
                route=_request_path_for_log(self.path),
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return

        archive_root = _web_reader_archive_root()
        if archive_root is None:
            body = render_session_read_page(
                bundle,
                session_id,
                None,
                notice="The archive is unavailable or requires a schema rebuild.",
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, body)
            return
        try:
            session = self._do_archive_get_session(archive_root, session_id, limit=DEFAULT_MESSAGE_PAGE_LIMIT, offset=0)
        except sqlite3.OperationalError as exc:
            emit(
                "daemon.webui.page_read_failed",
                level=ERROR,
                outcome="error",
                reason="archive_busy" if _is_sqlite_busy_error(exc) else "sqlite_error",
                route="session_read",
                session_id=session_id,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            status = HTTPStatus.SERVICE_UNAVAILABLE if _is_sqlite_busy_error(exc) else HTTPStatus.INTERNAL_SERVER_ERROR
            notice = (
                "The archive is temporarily busy; retry this session shortly."
                if status is HTTPStatus.SERVICE_UNAVAILABLE
                else "This session could not be rendered."
            )
            self._send_webui_html(status, render_session_read_page(bundle, session_id, None, notice=notice))
            return
        except Exception as exc:
            emit(
                "daemon.webui.page_render_failed",
                level=ERROR,
                outcome="error",
                reason="render_failed",
                route="session_read",
                session_id=session_id,
                status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                render_session_read_page(bundle, session_id, None, notice="This session could not be rendered."),
            )
            return
        if session is None:
            self._send_webui_html(
                HTTPStatus.NOT_FOUND,
                render_session_read_page(bundle, session_id, None, notice="This session could not be found."),
            )
            return
        if not isinstance(session, Mapping):
            self._send_webui_html(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                render_session_read_page(bundle, session_id, None, notice="This session could not be rendered."),
            )
            return
        self._send_webui_html(HTTPStatus.OK, render_session_read_page(bundle, session_id, session))

    def _serve_webui_search(self, params: dict[str, list[str]]) -> None:
        """Serve the DSL search vertical over the shared ``SearchEnvelope``.

        Semantics stay server-side: this delegates to the same
        ``compile_expression_into``/``_do_search_list`` path used by
        ``GET /api/sessions?query=...`` so the browser never re-filters or
        re-ranks a hit list itself.
        """
        from polylogue.archive.query.expression import ExpressionCompileError, compile_expression_into
        from polylogue.archive.query.spec import SessionQuerySpec, clamp_query_limit
        from polylogue.daemon.webui import (
            SEARCH_RESULT_LIMIT,
            WebUIAssetBundle,
            WebUIAssetError,
            render_search_page,
            render_webui_asset_error,
        )

        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
        except WebUIAssetError as exc:
            emit(
                "daemon.webui.assets_unavailable",
                level=ERROR,
                outcome="degraded",
                reason="asset_discovery_failed",
                route=_request_path_for_log(self.path),
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return

        query = self._get_param(params, "q") or ""
        cursor = self._get_param(params, "cursor")
        limit = clamp_query_limit(self._get_int(params, "limit", SEARCH_RESULT_LIMIT), default=SEARCH_RESULT_LIMIT)

        if not query:
            self._send_webui_html(HTTPStatus.OK, render_search_page(bundle, None, query))
            return

        try:
            base = SessionQuerySpec.from_params(
                {"cursor": cursor, "limit": limit, "offset": 0} if cursor else {"limit": limit, "offset": 0}
            )
            spec = compile_expression_into(query, base)
        except ExpressionCompileError as exc:
            self._send_webui_html(HTTPStatus.OK, render_search_page(bundle, None, query, parse_error=str(exc)))
            return

        async def _search(poly: Polylogue) -> object:
            return await self._do_search_list(poly, spec, limit, 0, route="/search")

        try:
            result = self._sync_run(_search)
        except sqlite3.OperationalError as exc:
            emit(
                "daemon.webui.page_read_failed",
                level=ERROR,
                outcome="error",
                reason="archive_busy" if _is_sqlite_busy_error(exc) else "sqlite_error",
                route="search",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            status = HTTPStatus.SERVICE_UNAVAILABLE if _is_sqlite_busy_error(exc) else HTTPStatus.INTERNAL_SERVER_ERROR
            notice = (
                "The archive is temporarily busy; retry this search shortly."
                if status is HTTPStatus.SERVICE_UNAVAILABLE
                else "This search could not be completed."
            )
            self._send_webui_html(status, render_search_page(bundle, None, query, notice=notice))
            return
        except Exception as exc:
            emit(
                "daemon.webui.page_render_failed",
                level=ERROR,
                outcome="error",
                reason="render_failed",
                route="search",
                status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                render_search_page(bundle, None, query, notice="This search could not be completed."),
            )
            return
        if not isinstance(result, Mapping):
            self._send_webui_html(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                render_search_page(bundle, None, query, notice="This search could not be completed."),
            )
            return
        self._send_webui_html(HTTPStatus.OK, render_search_page(bundle, result, query))

    def _serve_webui_observability(self) -> None:
        """Serve the registry-driven observability SSR page."""
        from polylogue.daemon.webui import (
            WebUIAssetBundle,
            WebUIAssetError,
            build_observability_status_payload,
            render_observability_page,
            render_webui_asset_error,
        )

        bundle: WebUIAssetBundle | None = None
        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
            payload = build_observability_status_payload(get_status_snapshot_payload())
            if not isinstance(payload, Mapping):
                raise RuntimeError("observability projection returned an invalid payload")
        except WebUIAssetError as exc:
            emit(
                "daemon.webui.assets_unavailable",
                level=ERROR,
                outcome="degraded",
                reason="asset_discovery_failed",
                route=_request_path_for_log(self.path),
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return
        except Exception as exc:
            emit(
                "daemon.webui.page_render_failed",
                level=ERROR,
                outcome="error",
                reason="render_failed",
                route="observability",
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            if bundle is None:
                self._send_webui_html(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    render_webui_asset_error("Observability assets are temporarily unavailable."),
                )
                return
            self._send_webui_html(
                HTTPStatus.SERVICE_UNAVAILABLE,
                render_observability_page(
                    bundle,
                    {
                        "contract_version": 1,
                        "status": {"adapter": "unavailable", "components": []},
                        "insights": [],
                        "insights_loaded": False,
                    },
                    notice="Observability data is temporarily unavailable.",
                ),
            )
            return
        assert bundle is not None
        self._send_webui_html(HTTPStatus.OK, render_observability_page(bundle, payload))

    def _serve_webui_cost(self) -> None:
        """Serve the registry-driven cost/usage explorer SSR page."""
        from polylogue.daemon.webui import (
            WebUIAssetBundle,
            WebUIAssetError,
            build_cost_payload,
            render_cost_page,
            render_webui_asset_error,
        )

        bundle: WebUIAssetBundle | None = None
        try:
            bundle = WebUIAssetBundle.discover(self.server.webui_dist_root)
            payload = self._sync_run(lambda poly: build_cost_payload(poly))
            if not isinstance(payload, Mapping):
                raise RuntimeError("cost projection returned an invalid payload")
        except WebUIAssetError as exc:
            emit(
                "daemon.webui.assets_unavailable",
                level=ERROR,
                outcome="degraded",
                reason="asset_discovery_failed",
                route=_request_path_for_log(self.path),
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_webui_html(HTTPStatus.SERVICE_UNAVAILABLE, render_webui_asset_error(str(exc)))
            return
        except Exception as exc:
            emit(
                "daemon.webui.page_render_failed",
                level=ERROR,
                outcome="error",
                reason="render_failed",
                route="cost",
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            if bundle is None:
                self._send_webui_html(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    render_webui_asset_error("Cost and usage assets are temporarily unavailable."),
                )
                return
            self._send_webui_html(
                HTTPStatus.SERVICE_UNAVAILABLE,
                render_cost_page(bundle, None, notice="Cost and usage data is temporarily unavailable."),
            )
            return
        assert bundle is not None
        self._send_webui_html(HTTPStatus.OK, render_cost_page(bundle, payload))

    def _serve_webui_asset(self, name: str) -> None:
        from polylogue.daemon.webui import WebUIAssetBundle, WebUIAssetError

        try:
            asset = WebUIAssetBundle.discover(self.server.webui_dist_root).read_asset(name)
        except FileNotFoundError:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        except WebUIAssetError as exc:
            emit(
                "daemon.webui.asset_read_failed",
                level=ERROR,
                outcome="error",
                reason="asset_read_failed",
                route=name,
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "webui_assets_unavailable", str(exc))
            return
        self._send_webui_asset(asset)

    def _handle_agent_coordination(self, params: dict[str, list[str]]) -> None:
        from polylogue.coordination import build_coordination_envelope

        raw_view = self._get_param(params, "view", "status") or "status"
        view = raw_view if raw_view in {"status", "self", "work-item", "conflicts", "handoff"} else "status"
        limit = self._get_int(params, "limit", 10)
        bounded_limit = max(1, min(limit, 50))
        fresh = self._get_bool(params, "fresh")
        cache_key = (view, bounded_limit)
        server = cast(Any, self.server)
        started_at = monotonic()
        entry: _CoordinationCacheEntry | None = None
        cache_owner = False
        if not fresh:
            with server.coordination_cache_lock:
                while True:
                    candidate = server.coordination_cache.get(cache_key)
                    if candidate is not None and candidate.expires_at > monotonic():
                        entry = candidate
                        break
                    if candidate is not None:
                        del server.coordination_cache[cache_key]
                    if cache_key not in server.coordination_cache_building:
                        server.coordination_cache_building.add(cache_key)
                        cache_owner = True
                        break
                    server.coordination_cache_condition.wait()
        if entry is None:
            try:
                payload = model_json_document(
                    build_coordination_envelope(view=cast(Any, view), limit=bounded_limit), exclude_none=True
                )
                if cache_owner:
                    with server.coordination_cache_lock:
                        server.coordination_cache[cache_key] = _CoordinationCacheEntry(
                            payload=payload,
                            expires_at=monotonic() + _COORDINATION_CACHE_TTL_S,
                        )
            finally:
                if cache_owner:
                    with server.coordination_cache_lock:
                        server.coordination_cache_building.discard(cache_key)
                        server.coordination_cache_condition.notify_all()
            cache_state = "bypass" if fresh else "miss"
        else:
            payload = entry.payload
            cache_state = "hit"
        elapsed_ms = (monotonic() - started_at) * 1000
        self._send_json(
            HTTPStatus.OK,
            payload,
            extra_headers={
                "Cache-Control": "no-store",
                "Server-Timing": f"coordination;dur={elapsed_ms:.1f}",
                "X-Polylogue-Coordination-Cache": cache_state,
                "X-Polylogue-Coordination-Freshness": f"ttl={_COORDINATION_CACHE_TTL_S:g}s; fresh=1 bypasses",
            },
        )

    @daemon_safe_handler
    def _handle_paste_browser(self, params: dict[str, list[str]]) -> None:
        return read_query._handle_paste_browser(self, params)

    #: Declared terminal unit expression behind ``/api/paste-browser``.
    #:
    #: Message grain on purpose. The session-scoped ``has_paste`` filter lowers
    #: to ``sessions.paste_count > 0``, a materialized aggregate that can lag a
    #: direct message write; the paste browser needs the paste-bearing MESSAGES
    #: themselves, which is why polylogue-q54dt declared the message-grain
    #: sibling instead of leaving this route hand-rolling a full-archive walk.
    PASTE_BROWSER_UNIT_QUERY = "messages where has_paste:true"

    async def _do_paste_browser(
        self,
        poly: Polylogue,
        *,
        limit: int,
        offset: int,
    ) -> object:
        # One bounded declared read. ``offset``/``limit`` are pushed into the
        # query-unit route and lowered to SQL LIMIT/OFFSET, so serving page N
        # costs one page, not one full-archive session walk plus a per-session
        # ``get_session`` hydration (polylogue-q54dt).
        envelope = await poly.query_units(
            self.PASTE_BROWSER_UNIT_QUERY,
            limit=limit,
            offset=offset,
        )
        rows = tuple(getattr(envelope, "items", ()))
        # ``next_offset`` is set from a real limit+1 probe by the query-unit
        # executor, so it distinguishes "the page filled" from "this is the
        # end" without a second scan.
        page_truncated = getattr(envelope, "next_offset", None) is not None
        entries: list[PasteBrowserEntry] = []
        # One archive read for the page's sessions, not one per session.
        summaries = await poly.get_session_summaries([str(row.session_id) for row in rows])
        display_titles: dict[str, str] = {}
        for row in rows:
            text = str(getattr(row, "text", "") or "")
            spans = envelope_paste_spans(text, has_paste=True)
            message_id = str(row.message_id)
            occurred_at_ms = getattr(row, "occurred_at_ms", None)
            session_id = str(row.session_id)
            if session_id not in display_titles:
                summary = summaries.get(session_id)
                display_titles[session_id] = str(
                    getattr(summary, "display_title", None) or getattr(row, "title", None) or session_id
                )
            entries.append(
                PasteBrowserEntry(
                    session_id=session_id,
                    session_title=display_titles[session_id],
                    origin=str(row.origin) if row.origin else None,
                    message_id=message_id,
                    message_anchor=reader_anchor("message", message_id),
                    role=str(getattr(row, "role", "") or ""),
                    timestamp=(
                        datetime.fromtimestamp(occurred_at_ms / 1000, tz=UTC).isoformat()
                        if isinstance(occurred_at_ms, int)
                        else None
                    ),
                    word_count=int(getattr(row, "word_count", 0) or 0),
                    snippet=snippet_for_paste(text, spans),
                    paste_spans=spans,
                    has_diff=any(span.get("kind") == "diff" for span in spans),
                )
            )
        return build_paste_browser_payload(entries, offset=offset, page_truncated=page_truncated)

    # ------------------------------------------------------------------
    # Handlers: attachment library + per-session attachments (#1199)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_attachment_library(self, params: dict[str, list[str]]) -> None:
        return read_query._handle_attachment_library(self, params)

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
        # One bounded archive read. Session/message hydration is deliberately
        # absent here: attachment_refs already carries the owning session and
        # message ids, so the SQL page can skip the old archive-wide walk.
        rows = await poly._get_attachment_library_page(
            limit=limit + 1,
            offset=offset,
            mime_filter=mime_filter,
            session_filter=session_filter,
            state_filter=state_filter,
        )
        entries: list[LibraryEntry] = []
        canonical_titles: dict[str, str] = {}
        # One archive read for the page's sessions, not one per session.
        summaries = await poly.get_session_summaries([str(cast(_AttachmentRow, row[0]).session_id) for row in rows])
        for raw_att, _title, origin in rows:
            # The facade returns the attachment opaquely: polylogue/api may not
            # import polylogue/storage (gate layering), so the record type cannot
            # be named in its signature. Bind it structurally here instead.
            att = cast(_AttachmentRow, raw_att)
            sid = str(att.session_id)
            # The SQL page's sessions.title is the parser title, which can be
            # an echoed user prompt for heuristic titles. Resolve the same
            # canonical display label used by session summaries before it
            # reaches the attachment library.
            if sid not in canonical_titles:
                summary = summaries.get(sid)
                canonical_titles[sid] = (
                    str(getattr(summary, "display_label", None) or getattr(summary, "title", None) or sid)
                    if summary is not None
                    else sid
                )
            envelope = attachment_to_envelope(att, session_id=sid, message_id=att.message_id)
            entries.append(
                LibraryEntry(
                    envelope=envelope,
                    session_title=canonical_titles[sid],
                    origin=origin,
                    message_anchor=reader_anchor("message", att.message_id) if att.message_id else None,
                )
            )
        page_truncated = len(entries) > limit
        if page_truncated:
            entries = entries[:limit]
        return build_library_payload(entries, offset=offset, page_truncated=page_truncated)

    @daemon_safe_handler
    def _handle_get_session_attachments(self, conv_id: str) -> None:
        return read_detail._handle_get_session_attachments(self, conv_id)

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

        Returns 200 when the configured health-check tiers pass, 503 when any
        non-OK health alert is present. Suitable for health check endpoints in
        Docker, systemd, and CI pipelines. The default config runs FAST and
        MEDIUM; operators can set ``health.check_tiers = "fast"`` to opt out
        of MEDIUM probes explicitly.
        """
        try:
            from polylogue.config import load_polylogue_config
            from polylogue.daemon.health import check_health, resolve_health_tiers

            cfg = load_polylogue_config()
            health = check_health(tiers=resolve_health_tiers(cfg.health_check_tiers))
            if health.overall_status == "ok":
                self._send_json(HTTPStatus.OK, {"ok": True, "status": "healthy"})
            else:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "status": health.overall_status, "alerts": len(health.alerts)},
                )
        except Exception as exc:
            # Response detail stays generic; the reason goes to the daemon log.
            emit(
                "daemon.http.health_check_failed",
                level=WARNING,
                outcome="error",
                route="health",
                status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "status": "error", "detail": "health check failed"},
            )

    def _handle_health(self) -> None:
        from polylogue.config import load_polylogue_config
        from polylogue.daemon.health import disk_free_bytes, wal_size_bytes
        from polylogue.paths import archive_root
        from polylogue.storage.archive_identity import resolve_active_index_path
        from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
        from polylogue.version import POLYLOGUE_VERSION, VERSION_INFO

        dbp = resolve_active_index_path(archive_root())
        db_size = dbp.stat().st_size if dbp.exists() else None
        wal_size = wal_size_bytes(dbp)
        # polylogue-xvwpi: a failed statvfs is not "zero bytes free" -- that
        # reading is an emergency, and publishing it for an unperformed
        # measurement is the failure-as-zero pattern this bead names. An
        # unmeasured figure is published as null.
        disk_free: int | None = None
        with contextlib.suppress(OSError):
            disk_free = disk_free_bytes(dbp.parent)

        # One producer for this fact across every status surface
        # (polylogue-20d.17.2). The probe itself is unchanged; what moved is
        # that ``quick_check_age_s`` is now rendered from the same observation
        # instead of being a literal null beside a measured result.
        quick_check = observe_quick_check(dbp)

        from polylogue.daemon.status import raw_failure_lifecycle_for_root

        raw_lifecycle = raw_failure_lifecycle_for_root(archive_root())
        raw_lifecycle_payload = {
            "available": raw_lifecycle.available,
            "state": raw_lifecycle.state,
            "reason": raw_lifecycle.reason,
            "parse_failures": raw_lifecycle.parse_failures,
            "validation_failures": raw_lifecycle.validation_failures,
            "deferred": raw_lifecycle.deferred,
            "terminal": raw_lifecycle.terminal,
            "unexplained": raw_lifecycle.unexplained,
        }
        archive_health_ok = quick_check.ok and raw_lifecycle.healthy

        overview: dict[str, object] = {
            "ok": archive_health_ok,
            "db_size_bytes": db_size,
            "wal_size_bytes": wal_size,
            "disk_free_bytes": disk_free,
            # Never measured by this route. A literal 0 rendered as a figure
            # in the operator surface; null says "not measured here".
            "blob_dir_size_bytes": None,
            **quick_check.payload(result_key=HEALTH_RESULT_KEY),
            "raw_failure_lifecycle_available": raw_lifecycle.available,
            "raw_failure_lifecycle_state": raw_lifecycle.state,
            "raw_failure_lifecycle_reason": raw_lifecycle.reason,
            "raw_failure_lifecycle": raw_lifecycle_payload,
            "archive_root": str(load_polylogue_config().archive_root),
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "daemon_version": POLYLOGUE_VERSION,
            "commit": VERSION_INFO.commit,
            "started_at": getattr(self.server, "started_at", None),
        }
        self._send_json(HTTPStatus.OK if archive_health_ok else HTTPStatus.SERVICE_UNAVAILABLE, overview)

    # ------------------------------------------------------------------
    # Handlers: status
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_status(self, params: dict[str, list[str]] | None = None) -> None:
        latest_event_id = get_latest_event_id()
        status = get_status_snapshot_payload()
        status["last_event_id"] = latest_event_id
        # polylogue-xvwpi: suppressing the liveness probe dropped the key
        # entirely and folded the resulting ``None`` into the ETag, so a
        # client that had seen a live value got a 304 over an unmeasured one.
        # The probe's failure is now a published state and part of the ETag.
        try:
            from polylogue.daemon.status import _check_daemon_liveness

            status["daemon_liveness"] = _check_daemon_liveness()
        except Exception as exc:
            status["daemon_liveness"] = None
            status["daemon_liveness_state"] = "unmeasured"
            emit(
                "daemon.http.status_probe_failed",
                level=WARNING,
                outcome="unmeasured",
                reason="daemon_liveness_unreadable",
                route="status",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
        else:
            status["daemon_liveness_state"] = "measured"
        # The frame and live discovery overlay can change without a daemon event.
        raw = _json_bytes(_stable_status_identity(status))
        etag_digest = hashlib.sha256(raw).hexdigest()[:24]
        etag = f'W/"status-{etag_digest}"'
        if_none_match = self.headers.get("If-None-Match", "")
        if if_none_match and if_none_match == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED.value)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self._send_json(HTTPStatus.OK, status, extra_headers={"ETag": etag})

    @daemon_safe_handler
    def _handle_webui_observability(self) -> None:
        """Return daemon-projected status and descriptor-backed insight panels."""
        from polylogue.daemon.webui import build_observability_payload

        payload = self._sync_run(lambda poly: build_observability_payload(poly, get_status_snapshot_payload()))
        self._send_json(HTTPStatus.OK, payload)

    @daemon_safe_handler
    def _handle_webui_insight(self, name: str, params: dict[str, list[str]]) -> None:
        """Return one bounded descriptor panel without a browser query model."""
        from polylogue.analysis.registry import INSIGHT_REGISTRY
        from polylogue.daemon.webui import build_observability_payload

        descriptor = INSIGHT_REGISTRY.get(name)
        if descriptor is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        payload = self._sync_run(
            lambda poly: build_observability_payload(
                poly,
                get_status_snapshot_payload(),
                registry={name: descriptor},
            )
        )
        self._send_json(HTTPStatus.OK, payload)

    @daemon_safe_handler
    def _handle_webui_source_freshness(self, params: dict[str, list[str]]) -> None:
        """Project one operator-named source with the bounded freshness reader."""
        source = self._get_param(params, "source")
        archive_root = _web_reader_archive_root()
        if not source:
            self._send_error(HTTPStatus.BAD_REQUEST, "missing_source", "source is required")
            return
        if archive_root is None:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "archive_unavailable")
            return
        from polylogue.archive.query.source_freshness import project_named_source_freshness
        from polylogue.archive.query.source_freshness_surfaces import source_freshness_status_payload

        freshness = project_named_source_freshness(archive_root, Path(source))
        payload: dict[str, object] = dict(source_freshness_status_payload(freshness))
        payload.update(
            {
                "observed_at": freshness.observed_at,
                "cursor_age_ms": freshness.cursor.age_ms,
                "fts_checked_at": freshness.fts.checked_at,
            }
        )
        self._send_json(HTTPStatus.OK, payload)

    @daemon_safe_handler
    def _handle_overview(self) -> None:
        return read_query._handle_overview(self)

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
        from polylogue.archive.query.spec import DEFAULT_SESSION_LIST_LIMIT, clamp_query_limit

        query_params = _build_query_spec_params(params, self)
        route = _public_route_from_request_path(self.path)
        # Clamp to the shared MAX_QUERY_LIMIT ceiling so the daemon honors the
        # same page-size cap as MCP instead of an arbitrary ?limit=99999999
        # (#1749).
        limit = clamp_query_limit(
            self._get_int(params, "limit", DEFAULT_SESSION_LIST_LIMIT),
            default=DEFAULT_SESSION_LIST_LIMIT,
        )
        offset = max(0, self._get_int(params, "offset", 0))
        cursor_values = params.get("cursor") or []
        cursor = cursor_values[0] if cursor_values else None
        if cursor:
            query_params["cursor"] = cursor

        async def _list(poly: Polylogue) -> object:
            return await self._do_list(poly, query_params, limit, offset, route=route)

        result = self._sync_run(_list)
        self._send_json(HTTPStatus.OK, result)

    async def _do_list(
        self,
        poly: Polylogue,
        query_params: dict[str, object],
        limit: int,
        offset: int,
        *,
        route: str = "/api/sessions",
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
            return await self._do_search_list(poly, spec, limit, offset, route=route)

        # A pure vector-only request (similar_text, no FTS term) must surface
        # the same typed EmbeddingRetrievalNotReadyError the CLI and MCP give,
        # not an opaque 500. Routing through the operations layer resolves the
        # vector provider (raising the typed readiness error when embeddings
        # are not ready), which daemon_safe_handler maps to its 409 status
        # instead of falling through to a generic ValueError (#1749).
        if spec.similar_text or spec.similar_session_id:
            return await self._do_search_list(poly, spec, limit, offset, route=route)

        try:
            summaries, total = await poly.list_session_summaries_with_count(spec)
        except ValueError as exc:
            if spec.session_id is None:
                raise
            from polylogue.archive.query.spec import QuerySpecError

            raise QuerySpecError("id", spec.session_id) from exc
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

        list_outcome = decide_outcome(matched=len(items))
        route_state_name, route_state_reason = _session_list_state(list_outcome, filtered=spec.has_filters())
        from polylogue.archive.query.spec import resolve_default_root_filter, session_count_unit_label

        result: dict[str, object] = {
            "outcome": list_outcome.to_dict(),
            "items": items,
            "total": total,
            "total_unit": session_count_unit_label(
                resolve_default_root_filter(spec.root, boolean_predicate=spec.boolean_predicate)
            ),
            "limit": limit,
            "offset": offset,
            "route_state": _route_readiness_payload(route_state_name, route, reason=route_state_reason),
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
        *,
        route: str = "/api/sessions",
    ) -> object:
        """Return the canonical :class:`SearchEnvelope` for ranked queries.

        The same envelope ships across CLI JSON, MCP, Python API, and daemon
        HTTP (#1266). Construction goes through the shared spec builder so
        the cursor / next_offset / ranking-policy fields stay aligned with
        the other surfaces. When ``cursor`` is supplied the response page
        starts strictly after the anchor (#1268).
        """
        from polylogue.api.search_envelope_builder import build_search_envelope_for_spec
        from polylogue.archive.query.search_cursor import InvalidSearchCursorError

        try:
            envelope = await build_search_envelope_for_spec(
                poly,
                spec,
                limit=limit,
                offset=offset,
                serving_identity="daemon",
            )
        except InvalidSearchCursorError as exc:
            return QueryErrorPayload(error="invalid_cursor", detail=str(exc)).model_dump(mode="json")
        except ValueError as exc:
            if spec.session_id is None:
                raise
            from polylogue.archive.query.spec import QuerySpecError

            raise QuerySpecError("id", spec.session_id) from exc
        except SearchIndexUnavailableError as exc:
            degraded_reason = str(exc)
            from polylogue.archive.query.search_hits import search_query_text
            from polylogue.surfaces.payloads import build_search_envelope

            query = search_query_text(spec.query_terms + spec.contains_terms)
            diagnostics = QueryMissDiagnosticsPayload(
                message=degraded_reason,
                filters=(f"query={query!r}",),
                reasons=(
                    QueryMissReasonPayload(
                        code="search_index_degraded",
                        severity="warning",
                        summary=degraded_reason,
                        detail="The canonical search could not read the archive search index.",
                    ),
                ),
                archive_session_count=None,
            )
            envelope = build_search_envelope(
                (),
                total=None,
                limit=limit,
                offset=offset,
                query=query,
                retrieval_lane=spec.retrieval_lane,
                sort=spec.sort,
                diagnostics=diagnostics,
            ).model_copy(update={"outcome": decide_outcome(matched=0, degraded=("search_index_degraded",))})
            payload = envelope.model_dump(mode="json")
            payload["route_state"] = _route_readiness_payload(
                "degraded",
                route,
                reason=degraded_reason,
                component="message_fts",
            )
            return payload
        payload = envelope.model_dump(mode="json")
        state, reason = _session_list_state(envelope.outcome, filtered=spec.has_filters())
        payload["route_state"] = _route_readiness_payload(state, route, reason=reason)
        return payload

    def _archive_summary_payload(self, summary: ArchiveSessionSummary) -> dict[str, object]:
        """Project one archive summary row into the web reader's list shape.

        The row is hydrated once by the canonical owner
        (``polylogue.archive.hydration``) and this method only applies the web
        reader's mask to the resulting domain summary, so it cannot populate a
        different subset than the API/CLI/MCP summary routes.
        """
        from polylogue.archive.hydration import archive_summary_to_domain
        from polylogue.surfaces.query_rows import session_row

        domain = archive_summary_to_domain(summary)
        session_id = str(domain.id)
        row = session_row(domain)
        target_ref = TargetRefPayload.session(session_id)
        return {
            "id": session_id,
            "session_id": session_id,
            "title": domain.display_title,
            "origin": str(domain.origin),
            "target_ref": _dump_target_ref(target_ref),
            "anchor": reader_anchor("session", session_id),
            "actions": _dump_actions(reader_session_actions()),
            "date": summary.updated_at or summary.created_at,
            "created_at": summary.created_at,
            "updated_at": summary.updated_at,
            "message_count": domain.message_count,
            # Stored session word counter; the domain summary deliberately
            # delegates word totals to the query-row projection.
            "word_count": summary.word_count,
            "terminal_state": domain.terminal_state,
            "total_cost_usd": row.cost_usd,
            "relative_time": row.relative_time,
            "repo": domain.git_repository_url,
            "cwd_display": next(iter(domain.working_directories), None),
            "tags": list(domain.tags),
            "flags": None,
            "summary": None,
        }

    # ------------------------------------------------------------------
    # Handlers: get session
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session(self, conv_id: str, params: dict[str, list[str]]) -> None:
        return read_detail._handle_get_session(self, conv_id, params)

    def _do_archive_get_session_summary(self, archive_root: Path, conv_id: str) -> object | None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            return execute_http_session_detail(
                {"session_id": conv_id, "shape": "summary", "limit": None, "offset": 0},
                archive=archive,
                adapters=_http_session_projection_adapters(),
            )

    async def _do_get_session_window(
        self, poly: Polylogue, conv_id: str, window: workspace_routes.MessageWindow
    ) -> object:
        """Load one session payload bounded to a workspace's declared window.

        The workspace routes render a narrow reading window over each
        referenced session; serving every message of every session made the
        response grow with total session length instead (polylogue-o0zju).
        ``message_count``/``total`` below still report the TRUE length.
        """

        return await self._do_get_session(poly, conv_id, limit=window.limit, offset=window.offset)

    async def _do_get_session(
        self,
        poly: Polylogue,
        conv_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> object:
        """Read one session payload, bounded to a declared message window.

        ``limit=None`` is the whole transcript, for callers that need it
        whole (the JSON session detail route). A declared window composes
        only ``[offset, offset + limit)`` at the storage layer
        (``Polylogue.get_session_page``) rather than composing the transcript
        and slicing it, which is the same bound the archive-backed twin
        ``_do_archive_get_session`` already holds (polylogue-2go3o). The
        attachment flattening and semantic card placement below are
        projections OF the served rows, so they stay bounded with it.

        ``message_count``/``total`` report the TRUE composed length and
        ``word_count`` the session's stored total either way: the window
        never becomes the reported session size.
        """
        page = await poly.get_session_page(conv_id, limit=limit, offset=offset)
        if page is None:
            return None
        conv = page.session
        flags = _build_flags_from_session(conv)
        session_id = str(conv.id)
        target_ref = TargetRefPayload.session(session_id)
        total_message_count = page.total_message_count
        # ``limit=None`` composed every message, so an offset alongside it is
        # still the caller's declared start -- applied here because the
        # storage page read serves a sized window, not an open-ended tail.
        window_messages = conv.messages.to_list()[offset:] if limit is None else conv.messages.to_list()
        # Flatten attachments across all messages so the inspector
        # tab and the session envelope share one source of truth
        # (#1199). Per-message attachments stay embedded in each
        # message envelope so the inline card renderer doesn't need
        # to cross-reference the session-level list.
        session_attachments: list[dict[str, object]] = []
        for msg in window_messages:
            for att in msg.attachments or []:
                session_attachments.append(attachment_to_envelope(att, session_id=session_id, message_id=msg.id))
        # Semantic transcript cards (#ap7): the same provider-neutral shell /
        # file-edit / task / attachment registry the CLI renders to Markdown
        # (``cli/messages.py``), projected per-message for the web reader.
        card_placement = semantic_card_placement_for_messages(
            window_messages,
            session_id=session_id,
            provider_family=conv.origin,
            lineage=lineage_descriptor_from_session(conv),
        )
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
            "message_count": total_message_count,
            "word_count": page.word_count,
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
                    "material_origin": _material_origin_value(msg),
                    "duration_ms": msg.duration_ms,
                    **message_topology_from_domain(msg),
                    "word_count": msg.word_count,
                    "has_tool_use": bool(msg.has_tool_use) if hasattr(msg, "has_tool_use") else False,
                    "has_thinking": bool(msg.has_thinking) if hasattr(msg, "has_thinking") else False,
                    "has_paste_evidence": bool(msg.has_paste) if hasattr(msg, "has_paste") else False,
                    "paste_spans": envelope_paste_spans(
                        msg.text,
                        has_paste=bool(msg.has_paste) if hasattr(msg, "has_paste") else False,
                    ),
                    "semantic_entries": card_placement.entries_for(str(msg.id)),
                    "semantic_cards": card_placement.cards_for(str(msg.id)),
                    "semantic_card_suppressed": card_placement.is_suppressed(str(msg.id)),
                    "attachments": [
                        attachment_to_envelope(att, session_id=session_id, message_id=msg.id)
                        for att in (msg.attachments or [])
                    ],
                }
                for msg in window_messages
            ],
            "attachments": session_attachments,
            "semantic_entries": list(card_placement.session_entries),
            "tags": conv.tags,
            "branch_type": str(conv.branch_type) if conv.branch_type else None,
            "parent_id": str(conv.parent_id) if conv.parent_id else None,
            "session_id": getattr(conv, "session_id", None),
            "repo": getattr(conv, "git_repository_url", None),
            "cwd_display": next(iter(getattr(conv, "working_directories", ()) or ()), None),
            "model": None,
            "flags": flags.model_dump(mode="json") if flags else None,
            "summary": conv.summary,
            "total": total_message_count,
        }

    def _do_archive_get_session(
        self,
        archive_root: Path,
        conv_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> object | None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            return self._run_archive_bounded_query(
                archive,
                deadline_s=None,
                compute=lambda: execute_http_session_detail(
                    {"session_id": conv_id, "shape": "full", "limit": limit, "offset": offset},
                    archive=archive,
                    adapters=_http_session_projection_adapters(),
                ),
            )

    # ------------------------------------------------------------------
    # Handlers: get session raw
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_raw(self, conv_id: str) -> None:
        return read_detail._handle_get_session_raw(self, conv_id)

    async def _do_get_session_raw(self, poly: Polylogue, conv_id: str) -> object:
        conv = await poly.get_session(conv_id)
        if conv is None:
            return None
        raw_artifacts, raw_artifacts_total = await poly.get_raw_artifacts_for_session(conv_id)
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
            "raw_artifacts": raw_artifacts,
            "raw_artifacts_total": raw_artifacts_total,
        }

    # ------------------------------------------------------------------
    # Handlers: get session cost (#1122)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_cost(self, conv_id: str) -> None:
        return read_detail._handle_get_session_cost(self, conv_id)

    @daemon_safe_handler
    def _handle_get_session_evidence_summary(self, conv_id: str) -> None:
        return read_detail._handle_get_session_evidence_summary(self, conv_id)

    async def _do_get_session_cost(self, poly: Polylogue, conv_id: str) -> object:
        from polylogue.analysis.archive import SessionCostInsightQuery

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
        return read_detail._handle_get_session_insights(self, conv_id, params)

    async def _do_get_session_insights(
        self,
        poly: Polylogue,
        conv_id: str,
        includes: tuple[str, ...],
    ) -> object:
        from polylogue.analysis.archive import (
            ArchiveInsightUnavailableError,
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
        panel_outcomes: list[OutcomeEnvelope] = []

        def _unavailable(kind: str, exc: BaseException) -> OutcomeEnvelope:
            emit(
                "daemon.http.session_insight_unavailable",
                level=WARNING,
                outcome="degraded",
                reason="insight_unavailable",
                kind=kind,
                session_id=conv_id,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            return decide_outcome(matched=0, error=f"insight_unavailable:{kind}")

        if "profile" in includes:
            from polylogue.analysis.archive import SessionProfileInsight
            from polylogue.config import active_archive_root
            from polylogue.operations.session_profile_convergence import session_profile_partition_status
            from polylogue.storage.derived.session.profiles import hydrate_session_profile

            profile_outcome: OutcomeEnvelope | None = None
            partition_status: str | None = None
            profile_row_matches = False
            try:
                # Archive read returns the full record directly; hydrate it into
                # the domain ``SessionProfile`` for the panel projection. Native
                # returns ``None`` (rather than raising) when the profile is not
                # materialized, so the except below is the unavailable-surface
                # path, not the unmaterialized one.
                profile_record = await poly.get_session_profile_record(conv_id)
                if profile_record is not None:
                    partition_status, stored_binding, stored_version = await asyncio.to_thread(
                        session_profile_partition_status, active_archive_root(poly.config), profile_record.session_id
                    )
                    profile_row_matches = (
                        profile_record.input_content_hash is not None
                        and profile_record.input_content_hash == stored_binding
                        and profile_record.materializer_version == stored_version
                    )
            except ArchiveInsightUnavailableError as exc:
                # The insight surface could not answer; the panel reports
                # q-error rather than 503-ing the whole envelope.
                profile_record = None
                profile_outcome = _unavailable("profile", exc)
            profile = hydrate_session_profile(profile_record) if profile_record is not None else None
            profile_insight = (
                SessionProfileInsight.from_record(profile_record, tier="evidence")
                if profile_record is not None
                else None
            )
            panel = (
                _profile_panel_payload(profile, profile_insight.provenance)
                if profile is not None and profile_insight is not None
                else _empty_profile_panel_payload(profile_outcome or decide_outcome(matched=0))
            )
            if profile is not None and (partition_status != "valid" or not profile_row_matches):
                # The retained row remains readable as evidence, but it is
                # not a current materialization of this session's inputs.
                row_count = int(profile.message_count or 0)
                panel["outcome"] = decide_outcome(matched=row_count, degraded=("session_profile_stale",)).to_dict()
                panel["readiness_tag"] = "q-partial"
                panel["materialized"] = False
            panel_outcomes.append(OutcomeEnvelope.model_validate(panel["outcome"]))
            # Retain provenance time diagnostics alongside the exact profile
            # partition verdict used for readiness above.
            if profile is not None:
                conv_updated_at = conv.updated_at.isoformat() if conv.updated_at else None
                staleness = _profile_staleness(profile_record, conv_updated_at)
                if staleness is not None:
                    if partition_status != "valid" or not profile_row_matches:
                        staleness["stale"] = True
                        staleness["reason"] = "session_profile_stale"
                    panel["staleness"] = staleness
            kinds["profile"] = panel

        if "threads" in includes:
            try:
                # Work threads are not keyed per-session in the substrate;
                # the reader filters the materialized rows by membership.
                all_threads = await poly.list_thread_insights(ThreadInsightQuery(limit=None))
                threads_error: OutcomeEnvelope | None = None
            except ArchiveInsightUnavailableError as exc:
                all_threads = []
                threads_error = _unavailable("threads", exc)
            member_threads = [th for th in all_threads if conv_id in (th.thread.session_ids or ())]
            threads_outcome = threads_error or decide_outcome(matched=len(member_threads))
            panel_outcomes.append(threads_outcome)
            kinds["threads"] = _thread_panel_payload(member_threads, threads_outcome)

        envelope["outcome"] = combine_outcomes(panel_outcomes).to_dict()
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
        return read_detail._handle_get_session_provenance(self, conv_id, params)

    # ------------------------------------------------------------------
    # Handlers: per-session topology (#1121)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_topology(
        self,
        conv_id: str,
        params: dict[str, list[str]],
    ) -> None:
        return read_detail._handle_get_session_topology(self, conv_id, params)

    # ------------------------------------------------------------------
    # Handlers: parent-chain stack envelope + thread-continue templates (#1203)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_parent_chain(
        self,
        conv_id: str,
        params: dict[str, list[str]],
    ) -> None:
        return read_detail._handle_get_session_parent_chain(self, conv_id, params)

    @daemon_safe_handler
    def _handle_get_thread_continue_templates(self) -> None:
        return read_detail._handle_get_thread_continue_templates(self)

    @daemon_safe_handler
    def _handle_provider_usage(self, params: dict[str, list[str]]) -> None:
        """``GET /api/provider-usage`` returns usage-accounting diagnostics.

        Registered in ``_static_get_routes`` and ``route_contracts.py``
        since #2469 but never implemented — any request raised an
        unhandled ``AttributeError`` (polylogue-g9j6). Mirrors the MCP
        ``provider_usage`` tool, which is a thin wrapper over the same
        ``Polylogue.origin_usage_report`` API method.

        Defaults to ``detail=headline``, NOT ``full`` (unlike the MCP
        tool): ``full`` walks ``session_provider_usage_events`` with a
        Python-side scan-and-compare with no row cap, measured to exceed
        90s on the live archive — a synchronous HTTP request thread is a
        much worse place to block on that than a CLI call a human can
        interrupt (polylogue-dlmv). Callers that want the full
        diagnostics can still request ``?detail=full`` explicitly.

        A missing/unopenable ``index.db`` used to propagate a raw
        ``sqlite3.OperationalError`` as an unhandled 500 (polylogue-d07y).
        Sibling read endpoints (``/api/archive-debt``, the split-archive
        ``/api/sessions`` fast path) degrade gracefully instead — this route
        now returns the same typed ``route_state: degraded`` envelope shape,
        privacy-projected like every other archive-path-bearing payload here.
        """
        origin = self._get_param(params, "origin")
        limit = self._get_int(params, "limit", 25)
        detail = self._get_param(params, "detail", "headline") or "headline"
        from polylogue.paths import archive_root as configured_archive_root

        archive_root = configured_archive_root()

        async def _get(poly: Polylogue) -> object:
            report = await poly.origin_usage_report(origin=origin, limit=limit, detail=detail)
            return report.to_dict()

        try:
            result = self._sync_run(_get)
        except (DatabaseError, sqlite3.Error) as exc:
            emit(
                "daemon.http.provider_usage_degraded",
                level=WARNING,
                outcome="degraded",
                reason="index_unreadable",
                route="provider-usage",
                origin=origin,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            reason = f"Provider usage accounting is unavailable: archive index could not be read ({exc})."
            degraded: dict[str, object] = {
                "archive_root": str(archive_root),
                "origin": origin,
                "detail_level": detail,
                "origins": [],
                "caveats": [reason],
                "route_state": _route_readiness_payload(
                    "degraded",
                    "/api/provider-usage",
                    reason=reason,
                    component="index_db",
                    stale_available=False,
                ),
            }
            self._send_json(HTTPStatus.OK, _web_privacy_safe_projection(degraded, archive_root))
            return

        self._send_json(HTTPStatus.OK, _web_privacy_safe_projection(result, archive_root))

    @daemon_safe_handler
    def _handle_query_units(self, params: dict[str, list[str]]) -> None:
        """``GET /api/query-units`` returns one bounded terminal-unit page.

        An initial request carries expression/filter fields. Follow-up requests
        carry only the opaque ``continuation`` emitted by the previous page;
        the daemon replays the canonical request rather than trusting a browser
        to reconstruct its filters or offset.
        """

        from polylogue.archive.query.expression import ExpressionCompileError
        from polylogue.archive.query.spec import clamp_query_limit
        from polylogue.archive.query.transaction import (
            QueryContinuationInvalidError,
            QueryTransactionRequest,
            decode_query_units_continuation,
        )
        from polylogue.archive.query.unit_results import query_unit_request
        from polylogue.operations.daemon_reads import _query_units_payload

        continuation_token = self._get_param(params, "continuation")
        session_filters: Mapping[str, object] | None = None
        continuation_request: QueryTransactionRequest | None = None
        if continuation_token is not None:
            if set(params) != {"continuation"}:
                self._send_error(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_continuation",
                    "continuation requests must not override the original query parameters",
                )
                return
            try:
                continuation = decode_query_units_continuation(continuation_token)
                continuation_request = continuation.request
                arguments = continuation_request.arguments
                expression_value = arguments.get("expression")
                filters_value = arguments.get("session_filters", {})
                if not isinstance(expression_value, str) or not isinstance(filters_value, Mapping):
                    raise QueryContinuationInvalidError("continuation does not identify a query-unit result")
                expression = expression_value
                session_filters = {str(key): value for key, value in filters_value.items()}
                limit = clamp_query_limit(continuation_request.page_size, default=50)
                offset = continuation_request.offset
            except QueryContinuationInvalidError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, exc.code, str(exc))
                return
            except (TypeError, ValueError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_continuation", str(exc))
                return
        else:
            expression = self._get_param(params, "expression") or ""
            limit = clamp_query_limit(self._get_int(params, "limit", 50), default=50)
            offset = max(0, self._get_int(params, "offset", 0))
        try:
            if session_filters is not None:
                request = query_unit_request(
                    expression=expression,
                    limit=limit,
                    offset=offset,
                    session_filters=session_filters,
                )
            else:
                request = query_unit_request(
                    expression=expression,
                    limit=limit,
                    offset=offset,
                    origin=self._get_param(params, "origin"),
                    origins=_csv_values(params, "origins"),
                    exclude_origin=self._get_param(params, "exclude_origin"),
                    tag=self._get_param(params, "tag"),
                    exclude_tag=self._get_param(params, "exclude_tag"),
                    repo=self._get_param(params, "repo"),
                    has_type=self._get_param(params, "has_type"),
                    referenced_path=self._get_param(params, "referenced_path"),
                    cwd_prefix=self._get_param(params, "cwd_prefix"),
                    action=self._get_param(params, "action"),
                    exclude_action=self._get_param(params, "exclude_action"),
                    action_sequence=self._get_param(params, "action_sequence"),
                    action_text=self._get_param(params, "action_text"),
                    tool=self._get_param(params, "tool"),
                    exclude_tool=self._get_param(params, "exclude_tool"),
                    title=self._get_param(params, "title"),
                    since=self._get_param(params, "since"),
                    until=self._get_param(params, "until"),
                    has_tool_use=self._get_bool(params, "has_tool_use"),
                    has_thinking=self._get_bool(params, "has_thinking"),
                    has_paste=self._get_bool(params, "has_paste_evidence"),
                    typed_only=self._get_bool(params, "typed_only"),
                    min_messages=self._get_param(params, "min_messages"),
                    max_messages=self._get_param(params, "max_messages"),
                    min_words=self._get_param(params, "min_words"),
                    max_words=self._get_param(params, "max_words"),
                    message_type=self._get_param(params, "message_type"),
                )
        except ExpressionCompileError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_query", str(exc))
            return
        archive_root = _web_reader_archive_root()
        if archive_root is None:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "archive_unavailable")
            return

        from polylogue.archive.query.execution_control import QueryTimeoutError, classify_unit_expression_workload
        from polylogue.archive.query.transaction import (
            QueryArchiveEpochUnreadableError,
            QueryContinuationStaleError,
            QueryTransaction,
            query_units_transaction_request,
        )

        # The handler thread is dedicated, so the read runs in place; the
        # execution context supplies the deadline and shares the process-wide
        # admission controller with the async surfaces (polylogue-z9gh.1).
        # A continuation is checked in query_unit_envelope before its result
        # query, from this reader's same SQLite snapshot.
        transaction = QueryTransaction(
            archive_root,
            continuation_request
            if continuation_request is not None
            else query_units_transaction_request(
                expression=expression,
                session_filters=request.session_filters or {},
                page_size=limit,
                offset=offset,
            ),
            workload_class=classify_unit_expression_workload(expression),
            read_timeout=_ARCHIVE_READER_BUSY_TIMEOUT_S,
        )
        try:
            operation_params: dict[str, object] = (
                {"continuation": continuation_token}
                if continuation_token is not None
                else {
                    "expression": expression,
                    "limit": request.limit,
                    "offset": request.offset,
                    "session_filters": request.session_filters or {},
                }
            )
            with self._observe_read_peer(transaction.context.cancel):
                payload = transaction.run_sync(
                    lambda archive: _query_units_payload(
                        operation_params,
                        archive=archive,
                        serving_identity="daemon",
                        execution_context=transaction.context,
                    )
                )
        except QueryTimeoutError:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "query_deadline_exceeded")
            return
        except QueryContinuationStaleError as exc:
            self._send_error(HTTPStatus.CONFLICT, exc.code, str(exc))
            return
        except QueryArchiveEpochUnreadableError as exc:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, exc.code, str(exc))
            return

        self._send_json(HTTPStatus.OK, payload)

    @daemon_safe_handler
    def _handle_archive_debt(self, params: dict[str, list[str]]) -> None:
        """``GET /api/archive-debt`` exposes the shared archive debt payload."""

        from polylogue import Polylogue
        from polylogue.api.sync.bridge import run_coroutine_sync
        from polylogue.paths import archive_root as configured_archive_root

        archive_root = configured_archive_root()
        kinds = _csv_values(params, "kind")
        limit = self._get_int(params, "limit", 50)
        payload = run_coroutine_sync(
            Polylogue(archive_root=archive_root).archive_debt(
                kinds=kinds or None,
                only_actionable=self._get_bool(params, "only_actionable"),
                limit=limit,
                exact_fts=self._get_bool(params, "exact_fts"),
            )
        )
        self._send_json(
            HTTPStatus.OK,
            _web_privacy_safe_projection(payload.model_dump(mode="json", exclude_none=True), archive_root),
        )

    @daemon_safe_handler
    def _handle_import_explain(self, params: dict[str, list[str]]) -> None:
        """``GET /api/import/explain`` explains archived import/source evidence."""

        from polylogue import Polylogue
        from polylogue.api.sync.bridge import run_coroutine_sync

        path = self._get_param(params, "path")
        raw_ref = self._get_param(params, "raw_ref")
        source_path = self._get_param(params, "source_path")
        if not path and not raw_ref and not source_path:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "missing_import_explain_target",
                    "message": "path, raw_ref, or source_path query parameter is required",
                },
            )
            return
        archive_root = _web_reader_archive_root()
        if archive_root is None:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "archive_unavailable")
            return
        payload = run_coroutine_sync(
            Polylogue(archive_root=archive_root, db_path=archive_root / "index.db").explain_import(
                path,
                raw_ref=raw_ref,
                source_path=source_path,
                limit=self._get_int(params, "limit", 100),
                redact_paths=not self._get_bool(params, "no_redact"),
            )
        )
        self._send_json(HTTPStatus.OK, payload.model_dump(mode="json", exclude_none=True))

    @daemon_safe_handler
    def _handle_ref_resolve(self, params: dict[str, list[str]]) -> None:
        return read_query._handle_ref_resolve(self, params)

    @daemon_safe_handler
    def _handle_query_completions(self, params: dict[str, list[str]]) -> None:
        return read_query._handle_query_completions(self, params)

    @daemon_safe_handler
    def _handle_action_affordances(self) -> None:
        return read_query._handle_action_affordances(self)

    @daemon_safe_handler
    def _handle_read_view_profiles(self) -> None:
        return read_query._handle_read_view_profiles(self)

    @daemon_safe_handler
    def _handle_assertions(self, params: dict[str, list[str]]) -> None:
        return user_overlay.handle_assertions(self, params)

    @daemon_safe_handler
    def _handle_user_overlay_get(self, path: list[str], params: dict[str, list[str]]) -> None:
        return user_overlay.handle_get(self, path, params)

    @daemon_safe_handler
    def _handle_user_overlay_post(self, path: list[str], params: dict[str, list[str]]) -> None:
        return user_overlay.handle_post(self, path, params)

    @daemon_safe_handler
    def _handle_user_overlay_delete(self, path: list[str], params: dict[str, list[str]]) -> None:
        return user_overlay.handle_delete(self, path, params)

    # ------------------------------------------------------------------
    # Handlers: shared single-session read-view execution (#1846)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_read(self, conv_id: str, params: dict[str, list[str]]) -> None:
        """``GET /api/sessions/{id}/read`` executes supported read profiles.

        This is a thin workbench adapter over existing read handlers. Accepted
        ``view`` values must already exist in ``/api/read-view-profiles``; the
        route does not introduce a second read-view registry.
        """

        view = (self._get_param(params, "view", "messages") or "messages").strip().lower()
        output_format = (self._get_param(params, "format", "json") or "json").strip().lower()
        capability = READ_VIEW_HTTP_CAPABILITIES.get(view)
        if capability is None:
            self._send_error(HTTPStatus.BAD_REQUEST, "unsupported_read_view")
            return
        if output_format not in capability.formats:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_format")
            return
        if capability.route != "/api/sessions/{session_id}/read":
            self._send_error(HTTPStatus.BAD_REQUEST, "read_view_requires_dedicated_route")
            return

        if view == "messages":
            limit = self._get_int(params, "limit", 50)
            offset = self._get_int(params, "offset", 0)
            window_continuation = self._get_param(params, "continuation")
            # An explicitly blank ``around`` is still an anchor request, so it
            # conflicts with an offset or continuation rather than vanishing.
            around = params["around"][0] if "around" in params else None
            if not self._accept_message_window_anchor(around, window_continuation, offset):
                return
            archive_root = _web_reader_archive_root()
            try:
                if archive_root is not None:
                    payload: object | None = self._do_archive_get_messages(
                        archive_root, conv_id, limit, offset, window_continuation, around
                    )
                else:

                    async def _get(poly: Polylogue) -> object:
                        return await self._do_get_messages(poly, conv_id, limit, offset, window_continuation, around)

                    payload = self._sync_run(_get)
            except MessageNotInSessionError as exc:
                self._send_error(HTTPStatus.NOT_FOUND, exc.code, str(exc))
                return
            except QueryContinuationStaleError as exc:
                self._send_error(HTTPStatus.CONFLICT, exc.code, str(exc))
                return
            except QueryContinuationInvalidError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, exc.code, str(exc))
                return
        elif view == "context":
            if output_format != "json":
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_format")
                return
            boundary = (self._get_param(params, "boundary", "session_start") or "session_start").strip().lower()
            if boundary not in {"session_start", "precompact"}:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_context_boundary")
                return

            async def _get_context(poly: Polylogue) -> dict[str, object] | None:
                context_payload = await poly.context_preamble_payload(
                    conv_id,
                    related_limit=self._get_int(params, "related_limit", 5),
                    boundary=boundary,
                    token_budget=self._get_int(params, "max_tokens", 0) or None,
                )
                if context_payload is None:
                    return None
                return dict(context_payload.model_dump(mode="json", exclude_none=True))

            payload = self._sync_run(_get_context)
        elif view == "context-image":
            if output_format != "json":
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_format")
                return

            async def _get(poly: Polylogue) -> object:
                include_messages = True
                if self._get_param(params, "include_messages") is not None:
                    include_messages = self._get_bool(params, "include_messages")
                max_tokens = self._get_int(params, "max_tokens", 0) or None
                context_payload = await poly.context_image_payload(
                    seed_session_id=conv_id,
                    max_sessions=1,
                    max_tokens=max_tokens,
                    include_messages=include_messages,
                    redact_paths=not self._get_bool(params, "no_redact"),
                )
                return context_payload.model_dump(mode="json", exclude_none=True)

            payload = self._sync_run(_get)
        elif view == "neighbors":
            if output_format != "json":
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_format")
                return

            async def _get(poly: Polylogue) -> object:
                return {
                    "neighbors": await poly.neighbor_candidate_payloads(
                        session_id=conv_id,
                        limit=max(1, self._get_int(params, "limit", 10)),
                        window_hours=max(1, self._get_int(params, "window_hours", 24)),
                    )
                }

            payload = self._sync_run(_get)
        elif view == "correlation":
            if output_format != "json":
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_format")
                return

            confidence = 0.3
            raw_confidence = self._get_param(params, "confidence_threshold")
            if raw_confidence is not None:
                with contextlib.suppress(ValueError, TypeError):
                    confidence = float(raw_confidence)

            async def _get(poly: Polylogue) -> object | None:
                return await poly.session_correlation_payload(
                    conv_id,
                    repo_path=self._get_param(params, "repo_path"),
                    since_hours=max(1, self._get_int(params, "since_hours", 2)),
                    confidence_threshold=max(0.0, min(confidence, 1.0)),
                )

            payload = self._sync_run(_get)
        elif view == "effective_context":
            if output_format != "json":
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_format")
                return

            async def _get_effective(poly: Polylogue) -> object | None:
                messages = await poly.get_effective_context(
                    conv_id,
                    at_position=(
                        self._get_int(params, "at_position", 0)
                        if self._get_param(params, "at_position") is not None
                        else None
                    ),
                )
                if messages is None:
                    return None
                return {"session_id": conv_id, "messages": messages}

            payload = self._sync_run(_get_effective)
        elif view == "lineage":
            if output_format != "json":
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_format")
                return
            from polylogue.analysis.lineage_graph import DEFAULT_LINEAGE_PAGE_LIMIT

            node_limit = self._get_int(params, "node_limit", DEFAULT_LINEAGE_PAGE_LIMIT)
            edge_limit = self._get_int(params, "edge_limit", DEFAULT_LINEAGE_PAGE_LIMIT)
            if node_limit < 0 or edge_limit < 0:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_page_limit")
                return

            async def _get_lineage(poly: Polylogue) -> object | None:
                graph = await poly.compact_lineage(
                    conv_id,
                    node_offset=max(0, self._get_int(params, "node_offset", 0)),
                    node_limit=node_limit,
                    edge_offset=max(0, self._get_int(params, "edge_offset", 0)),
                    edge_limit=edge_limit,
                )
                return None if graph is None else graph.model_dump(mode="json")

            payload = self._sync_run(_get_lineage)
        else:

            async def _get(poly: Polylogue) -> object | None:
                return await self._do_get_session_raw(poly, conv_id)

            payload = self._sync_run(_get)

        if payload is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        from polylogue.archive.viewport import get_read_view_profile

        profile = get_read_view_profile(view)
        envelope = SessionReadViewEnvelope(
            session_id=conv_id,
            view=view,
            format=output_format,
            target_refs=(f"session:{conv_id}",),
            object_refs=_read_view_payload_field_values(payload, "object_refs"),
            evidence_refs=_read_view_payload_field_values(payload, "evidence_refs"),
            caveats=_read_view_payload_field_values(payload, "caveats"),
            lossiness=profile.lossiness,
            evidence_policy=profile.evidence_policy,
            privacy_policy=profile.privacy_policy,
            payload=payload,
        )
        self._send_json(HTTPStatus.OK, model_json_document(envelope, exclude_none=True))

    # ------------------------------------------------------------------
    # Handlers: per-session embedding similarity (#1123)
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_session_similar(
        self,
        conv_id: str,
        params: dict[str, list[str]],
    ) -> None:
        return read_detail._handle_get_session_similar(self, conv_id, params)

    # ------------------------------------------------------------------
    # Handlers: get messages
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_messages(self, conv_id: str, params: dict[str, list[str]]) -> None:
        return read_detail._handle_get_messages(self, conv_id, params)

    def _accept_message_window_anchor(
        self, around: str | None, continuation: str | None, offset: int | None = None
    ) -> bool:
        """Refuse a request that names its window twice, and say so.

        ``around`` asks the route to *decide* the offset; a continuation
        already carries one. Honouring either silently would answer a window
        the caller did not ask for, so the disagreement is the caller's error.
        """

        if around is not None and (continuation or (offset is not None and offset != 0)):
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "around and continuation name two different windows",
            )
            return False
        return True

    async def _do_get_messages(
        self,
        poly: Polylogue,
        conv_id: str,
        limit: int,
        offset: int,
        continuation: str | None = None,
        around: str | None = None,
    ) -> object:
        started_at = monotonic()
        session_id = str(conv_id)
        # polylogue-2go3o: this projection needs the session ROW -- origin
        # plus topology -- so it reads a summary. Reaching those fields
        # through ``get_session`` composed the whole transcript to serve one
        # window, and composed it twice when the window was deep-linked.
        summary = await poly.get_session_summary(conv_id)
        if summary is None:
            return None
        session_id = str(getattr(summary, "session_id", None) or getattr(summary, "id", None) or conv_id)
        # polylogue-i5vqc: a deep link names a message, so the shared read
        # route resolves it to the offset of the window that holds it instead
        # of letting the caller walk pages looking for it. The locate is
        # answered from indexed counts at the storage layer, so it costs no
        # composition at all.
        # polylogue-ijbwq: window arithmetic, snapshot binding and the
        # continuation token come from that same shared execution route; this
        # handler owns only the web-reader projection below.
        window = await poly.read_transcript_window(
            conv_id,
            limit=limit,
            offset=0 if around else offset,
            continuation=continuation,
            around=around,
        )
        messages, total = window.rows, window.total
        completeness = SimpleNamespace(
            complete=window.lineage_complete,
            truncation_reason=window.lineage_truncation_reason,
        )
        # polylogue-ppkj: lineage_descriptor_from_session hard-codes
        # lineage_complete=None (the DB-backed session row carries no such
        # field). Overlay the real read-time signal from the window read so
        # this JSON response -- and the semantic card placement built from it
        # -- can flag a truncated composed transcript instead of serving a
        # partial one with no indication.
        lineage = dataclasses_replace(
            lineage_descriptor_from_session(summary),
            lineage_complete=completeness.complete,
            lineage_truncation_reason=completeness.truncation_reason,
        )
        # Placed over the served window, exactly as the archive-backed twin
        # ``_do_archive_get_messages`` and the CLI's paginated messages view
        # already place theirs: a projection of the page cannot be built from
        # the whole transcript without costing the whole transcript.
        placement = semantic_card_placement_for_messages(
            messages,
            session_id=session_id,
            provider_family=summary.origin,
            lineage=lineage,
        )
        return {
            "session_id": session_id,
            "messages": [
                {
                    **cast(
                        "dict[str, object]",
                        model_json_document(
                            message_render_envelope_from_domain(msg, session_id=session_id),
                            exclude_none=False,
                        ),
                    ),
                    "word_count": msg.word_count,
                    "paste_spans": envelope_paste_spans(msg.text, has_paste=bool(msg.has_paste)),
                    "semantic_entries": placement.entries_for(str(msg.id)),
                    "semantic_cards": placement.cards_for(str(msg.id)),
                    "semantic_card_suppressed": placement.is_suppressed(str(msg.id)),
                    "attachments": [
                        attachment_to_envelope(att, session_id=session_id, message_id=msg.id)
                        for att in (msg.attachments or [])
                    ],
                }
                for msg in messages
            ],
            "semantic_entries": list(placement.session_entries),
            "total": total,
            "limit": window.limit,
            "offset": window.offset,
            "next_offset": window.next_offset,
            "continuation": window.continuation,
            "lineage_complete": completeness.complete,
            "lineage_truncation_reason": completeness.truncation_reason,
            "outcome": lineage_page_outcome(
                matched=len(messages),
                complete=completeness.complete,
                truncation_reason=completeness.truncation_reason,
            ).to_dict(),
            "authority": serialize_authority(
                authority_for_config(poly.config, server_identity="daemon", started_at=started_at)
            ),
        }

    def _do_archive_get_messages(
        self,
        archive_root: Path,
        conv_id: str,
        limit: int,
        offset: int,
        continuation: str | None = None,
        around: str | None = None,
    ) -> object | None:
        from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

        transaction = QueryTransaction(
            archive_root,
            QueryTransactionRequest(
                operation="http.archive.read",
                arguments={"path": getattr(self, "path", "")},
                projection="http-read",
                page_size=limit,
                offset=offset,
            ),
        )
        with self._observe_read_peer(transaction.context.cancel):
            return transaction.run_sync(
                lambda archive: execute_http_session_messages(
                    {
                        "session_id": conv_id,
                        "limit": limit,
                        "offset": offset,
                        "continuation": continuation,
                        "around": around,
                    },
                    archive=archive,
                    adapters=_http_session_projection_adapters(),
                    server_identity="daemon",
                )
            )

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
            window = workspace_routes.parse_message_window(self, params)
            self._send_json(HTTPStatus.OK, self._do_archive_stack(archive_root, ids, focus, window))
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
            window = workspace_routes.parse_message_window(self, params)
            self._send_json(
                HTTPStatus.OK, self._do_archive_compare(archive_root, left, right, align or "prompt", window)
            )
            return
        workspace_routes.handle_compare(self, params)

    def _do_archive_stack(
        self,
        archive_root: Path,
        ids: list[str],
        focus: str | None,
        window: workspace_routes.MessageWindow,
    ) -> dict[str, object]:
        items: list[dict[str, object]] = []
        for conv_id in ids:
            payload = self._do_archive_get_session(archive_root, conv_id, limit=window.limit, offset=window.offset)
            if not isinstance(payload, dict):
                items.append(workspace_routes.missing_session_target(conv_id))
                continue
            items.append(workspace_routes.stack_item(payload))
        return workspace_routes.stack_payload(ids, focus, items, window)

    def _do_archive_compare(
        self,
        archive_root: Path,
        left: str,
        right: str,
        align: str,
        window: workspace_routes.MessageWindow,
    ) -> object:
        from polylogue.daemon.compare import build_compare_envelope

        left_payload = self._do_archive_get_session(archive_root, left, limit=window.limit, offset=window.offset)
        right_payload = self._do_archive_get_session(archive_root, right, limit=window.limit, offset=window.offset)
        return build_compare_envelope(left_payload, right_payload, left, right, align, window=window)

    # ------------------------------------------------------------------
    # Handlers: get raw artifact
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_raw_artifact(self, artifact_id: str) -> None:
        return read_detail._handle_get_raw_artifact(self, artifact_id)

    async def _do_get_raw_artifacts(self, poly: Polylogue, artifact_id: str) -> object:
        raw_items = await poly.get_raw_artifacts_for_session(artifact_id)
        return {"raw_artifacts": raw_items}

    # ------------------------------------------------------------------
    # Handlers: facets
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_facets(self, params: dict[str, list[str]]) -> None:
        return read_query._handle_facets(self, params)

    async def _do_facets(
        self,
        poly: Polylogue,
        query_params: dict[str, object],
        *,
        include_deferred: bool,
    ) -> object:
        """Compute scoped + global facets via the shared archive contract.

        Delegates to :meth:`polylogue.api.archive.PolylogueArchiveMixin.facets`
        so daemon HTTP, MCP, CLI, and the Python API all share one
        scope vocabulary (#1269 / slice D of #873).
        """
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params(query_params) if query_params else None
        response = await poly.facets(spec, include_deferred=include_deferred)
        return response.model_dump(mode="json", by_alias=True)

    def _run_archive_bounded_query(
        self,
        archive: ArchiveStore,
        *,
        deadline_s: float | None,
        compute: Callable[[], _ArchiveQueryResult],
    ) -> _ArchiveQueryResult:
        """Run one archive SQL step with client-abort interruption.

        Browser-side ``AbortController`` cancellation only helps the server if
        the request thread periodically observes the dead socket. SQLite's
        progress handler gives long scans/joins that observation point for the
        split-archive read paths.
        """

        conn = getattr(archive, "_conn", None)
        cancellation = current_cancellation()
        if cancellation is not None and conn is not None:
            cancellation.register_connection(conn)

        def _deadline_expired() -> bool:
            return deadline_s is not None and monotonic() >= deadline_s

        def _raise_if_interrupted() -> None:
            self._raise_if_client_disconnected()
            if cancellation is not None and cancellation.cancelled:
                raise DaemonOperationCancelled("archive read cancelled")
            if _deadline_expired():
                raise TimeoutError("archive query deadline exceeded")

        _raise_if_interrupted()

        def _sqlite_progress() -> int:
            if self._client_disconnected():
                return 1
            if cancellation is not None and cancellation.cancelled:
                return 1
            return 1 if _deadline_expired() else 0

        if conn is not None:
            conn.set_progress_handler(_sqlite_progress, 1000)
        try:
            result = compute()
            _raise_if_interrupted()
            return result
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                if self._client_disconnected():
                    raise _ClientDisconnectedDuringComputeError(_FACET_CANCELLED_REASON) from exc
                if cancellation is not None and cancellation.cancelled:
                    raise DaemonOperationCancelled("archive read cancelled") from exc
                if _deadline_expired():
                    raise TimeoutError("archive query deadline exceeded") from exc
            raise
        finally:
            if conn is not None:
                conn.set_progress_handler(None, 0)
                if cancellation is not None:
                    cancellation.unregister_connection(conn)

    # ------------------------------------------------------------------
    # Handlers: sources
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_sources(self) -> None:
        return read_detail._handle_sources(self)

    # ------------------------------------------------------------------
    # Handlers: reset
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_cli_query(self) -> None:
        """Serve a root-request parameter map through the daemon query compiler.

        The CLI sends exactly :meth:`RootModeRequest.query_params` rather than
        reconstructing a query string itself.  This preserves the daemon as
        the owner of structured-query lowering while keeping the transport
        payload deliberately small and stdlib-friendly.
        """

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0 or content_length > 65_536:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        try:
            body = json.loads(self.rfile.read(content_length))
            raw_params = body["params"]
            if not isinstance(raw_params, dict):
                raise TypeError("params must be an object")
            from polylogue.cli.root_request import RootModeRequest
            from polylogue.operations.query_lowering import expression_from_query_terms

            request = RootModeRequest.from_params(raw_params)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        params: dict[str, list[str]] = {}
        for key, value in request.params.items():
            if value is None or value is False:
                continue
            if isinstance(value, tuple | list):
                params[str(key)] = [str(item) for item in value]
            elif value is True:
                params[str(key)] = ["1"]
            else:
                params[str(key)] = [str(value)]
        expression = expression_from_query_terms(request.query_terms)
        if expression:
            params["query"] = [expression]
        self._handle_list_sessions(params)

    @daemon_safe_handler
    def _handle_daemon_operation(self) -> None:
        """Authenticate transport, then invoke the same canonical machine runtime."""
        from polylogue.operations.daemon_protocol import (
            MAX_DECLARED_OPERATION_BODY_BYTES,
            DaemonOperationRequest,
            daemon_operation_spec,
        )

        # Every refusal below happens before the runtime is entered, so each
        # one is marked pre-dispatch (polylogue-ji49p): an unmarked refusal is
        # converted by the client into an indeterminate mutation.
        if not self._check_auth(allow_web=False, refuse=self._reject_operation) or not self._check_cross_origin(
            refuse=self._reject_operation
        ):
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._reject_operation(HTTPStatus.BAD_REQUEST, "unsupported_transfer_encoding")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
            self._reject_operation(HTTPStatus.BAD_REQUEST, "invalid_content_length")
            return
        length = int(lengths[0])
        if length <= 0 or length > MAX_DECLARED_OPERATION_BODY_BYTES:
            self._reject_operation(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._reject_operation(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type")
            return
        self.connection.settimeout(5.0)
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("partial body")
            request = DaemonOperationRequest.from_dict(json.loads(body))
        except (ValueError, TypeError, TimeoutError, OSError):
            self._reject_operation(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        spec = daemon_operation_spec(request.operation)
        assert spec is not None
        if length > spec.max_body_bytes:
            self._reject_operation(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large")
            return
        self._send_daemon_operation(self._execute_daemon_operation(request))

    def _execute_daemon_operation(self, request: DaemonOperationRequest) -> dict[str, object]:
        from polylogue.operations.daemon_protocol import DAEMON_PRINCIPAL_CAPABILITIES
        from polylogue.operations.mutation_transaction import MutationPrincipal

        web_token = self._web_credential_token()
        if self._auth_token and not self.headers.get("Authorization", "") and web_token:
            required_scope: WebCredentialScope = "read" if self.command == "GET" else "user_state"
            if not self._web_credential_decision(required_scope).allowed:
                raise RuntimeError("daemon operation web credential is no longer valid")
            base = MutationPrincipal(
                actor_ref=f"daemon:web:{hashlib.sha256(web_token.encode()).hexdigest()}",
                capabilities=frozenset({"read"}),
                surface="api",
                role_label="daemon-web-credential",
            )
        else:
            base = self._cli_mutation_principal("read")
        principal = MutationPrincipal(
            actor_ref=base.actor_ref,
            capabilities=DAEMON_PRINCIPAL_CAPABILITIES,
            surface=base.surface,
            role_label=base.role_label,
        )
        runtime = self.server.operation_runtime
        from polylogue.daemon.operation_disconnect import observe_peer_disconnect

        with observe_peer_disconnect(self.connection) as disconnected:
            return runtime.call(request, principal, client_disconnect=disconnected)

    def _send_daemon_operation(self, payload: dict[str, object]) -> None:
        status = (
            HTTPStatus.ACCEPTED if payload["outcome"] in {"accepted", "running", "indeterminate"} else HTTPStatus.OK
        )
        if payload["outcome"] in {"failed", "rejected", "timed-out", "cancelled"}:
            status = HTTPStatus.CONFLICT
        from polylogue.operations.read_result_transport import TRANSFER_BYTES, staged_json_response

        with staged_json_response(payload, append_newline=True) as staged:
            size = staged.seek(0, 2)
            staged.seek(0)
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(size))
            self._send_request_id_header()
            self.end_headers()
            while chunk := staged.read(TRANSFER_BYTES):
                self.wfile.write(chunk)

    def _read_bounded_json_body(self, max_bytes: int) -> dict[str, object] | None:
        raw_content_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_content_length) if raw_content_length is not None else 0
        except (TypeError, ValueError):
            content_length = 0
        if content_length <= 0 or content_length > max_bytes:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return None
        try:
            raw = self.rfile.read(content_length)
            if len(raw) != content_length:
                raise ValueError("interrupted request body")
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            return cast(dict[str, object], body)
        except (json.JSONDecodeError, TypeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return None

    #: The only reset scope this route implements. Every other scope is
    #: refused rather than silently accepted: polylogue-peo7o found the
    #: default scope ("all") answering 200 {"ok": true} and emitting a
    #: reset event with operation_id ``reset-all-all`` having touched
    #: nothing at all. A destructive control route that reports success
    #: for an unperformed mutation is worse than one that refuses.
    RESET_SUPPORTED_SCOPES: ClassVar[frozenset[str]] = frozenset({"session"})

    @daemon_safe_handler
    def _handle_reset(self) -> None:
        raw_content_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_content_length)
        except (TypeError, ValueError):
            # A malformed Content-Length is a client framing error, not a
            # daemon fault; without this it raised through into a 500.
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", "malformed Content-Length")
            return
        if content_length < 0:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", "negative Content-Length")
            return
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(body_raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        if not isinstance(body, dict):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", "request body must be an object")
            return

        # The wire default stays "all" -- what clients actually send -- so an
        # unchanged caller now gets the refusal it always deserved instead of
        # a 200 that reports success for an unperformed mutation.
        scope = body.get("scope", "all")
        conv_id = body.get("session_id")

        if not isinstance(scope, str) or scope not in self.RESET_SUPPORTED_SCOPES:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "unsupported_scope",
                f"reset scope {scope!r} is not implemented by this route",
                extra_payload={"supported_scopes": sorted(self.RESET_SUPPORTED_SCOPES)},
            )
            return
        if not conv_id or not isinstance(conv_id, str):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", "session_id is required")
            return
        # The caller's own ``mutation.session.delete.preview`` reference. The
        # route never prepares a preview for the caller: deleting on a bare
        # session id would record a bound-token authorization for a plan the
        # caller never saw.
        preview_ref = body.get("preview_ref")
        if not preview_ref or not isinstance(preview_ref, str):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", "preview_ref is required")
            return

        op_id = f"reset-{scope}-{conv_id[:16]}"

        async def _do_reset(poly: Polylogue) -> dict[str, object]:
            from polylogue.operations.daemon_errors import DaemonOperationRejectedError

            # Route through the typed delete contract so resolution and
            # idempotency live in ArchiveMutationsMixin (#862).
            try:
                result = await poly.delete_session_safe(conv_id, preview_ref=preview_ref)
            except DaemonOperationRejectedError as exc:
                return {"refused": exc.outcome, "detail": exc.detail, "session_id": conv_id}
            return {"deleted": result.outcome == "deleted", "session_id": conv_id}

        result = self._sync_run(_do_reset)
        if isinstance(result, dict) and isinstance(result.get("refused"), str):
            self._send_error(HTTPStatus.CONFLICT, str(result["refused"]), str(result.get("detail") or ""))
            return

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
    def _handle_mcp_call_log(self) -> None:
        """Persist one MCP call event through the daemon's writer gate."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0 or content_length > 16_384:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        try:
            body = json.loads(self.rfile.read(content_length))
            call_id = str(body["call_id"])
            tool_name = str(body["tool_name"])
            session_id_raw = body.get("session_id")
            session_id = None if session_id_raw is None else str(session_id_raw)
            session_ids_raw = body.get("session_ids", [])
            if not isinstance(session_ids_raw, list):
                raise TypeError("session_ids must be a list")
            session_ids = tuple(str(value) for value in session_ids_raw)
            started_at_ms = int(body["started_at_ms"])
            finished_at_ms = int(body["finished_at_ms"])
            success = body["success"]
            error_detail_raw = body.get("error_detail")
            error_detail = None if error_detail_raw is None else str(error_detail_raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        if (
            not isinstance(body, dict)
            or not call_id
            or len(call_id) > 128
            or not tool_name
            or len(tool_name) > 256
            or (session_id is not None and len(session_id) > 2048)
            or len(session_ids) > 256
            or any(not value or len(value) > 2048 for value in session_ids)
            or not isinstance(success, bool)
            or finished_at_ms < started_at_ms
            or (error_detail is not None and len(error_detail) > 512)
        ):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        from polylogue.paths import archive_root
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_mcp_call
        from polylogue.storage.sqlite.connection_profile import owned_daemon_connection

        ops_db = archive_root() / "ops.db"

        async def record_call(_archive: object) -> None:
            if not ops_db.exists():
                initialize_archive_database(ops_db, ArchiveTier.OPS)
            with owned_daemon_connection(ops_db, archive_root=ops_db.parent) as conn:
                table_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM sqlite_master
                        WHERE type = 'table'
                          AND name IN ('mcp_call_log', 'mcp_call_session_refs')
                        """
                    ).fetchone()[0]
                )
            if table_count != 2:
                initialize_archive_database(ops_db, ArchiveTier.OPS)
            with owned_daemon_connection(ops_db, archive_root=ops_db.parent) as conn, conn:
                record_mcp_call(
                    conn,
                    call_id=call_id,
                    tool_name=tool_name,
                    session_id=session_id,
                    session_ids=session_ids,
                    started_at_ms=started_at_ms,
                    finished_at_ms=finished_at_ms,
                    success=success,
                    error_detail=error_detail,
                )

        try:
            self._sync_run(record_call)
        except ValueError:
            self._send_error(HTTPStatus.CONFLICT, "call_id_conflict")
            return
        self._send_json(HTTPStatus.OK, {"ok": True, "call_id": call_id})

    @daemon_safe_handler
    def _handle_ingest(self) -> None:
        body = self._read_bounded_json_body(65_536)
        if body is None:
            return

        from polylogue.operations.import_staging import import_staging_root, resolve_staged_import

        # Staged imports wait outside the watched inbox, so the ingest
        # operation is the only route that acquires them.
        import_staging_root(self.server.archive_root).mkdir(parents=True, exist_ok=True)
        source, error = resolve_staged_import(body.get("path"), self.server.archive_root)
        if error is not None:
            self._send_error(HTTPStatus.BAD_REQUEST, error)
            return
        assert source is not None

        from polylogue.operations.import_operations import prepare_import_source_admission

        try:
            admission = prepare_import_source_admission(source)
        except (OSError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_source_proof", str(exc))
            return
        preflight = admission.preflight
        if not preflight.admissible:
            self._send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, preflight.error_code, preflight.summary())
            return

        from uuid import uuid4

        from pydantic import ValidationError

        from polylogue.operations.daemon_protocol import DaemonOperationRequest
        from polylogue.operations.import_operations import ImportRequest

        if (
            body.get("source_path", admission.request.source_path) != admission.request.source_path
            or body.get("source_name", admission.request.source_name) != admission.request.source_name
        ):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_source_declaration")
            return
        try:
            request = ImportRequest.model_validate(
                {
                    "source_path": admission.request.source_path,
                    "source_name": admission.request.source_name,
                    "staged_path": str(source),
                    "idempotency_key": body.get("idempotency_key"),
                }
            )
        except ValidationError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        operation = DaemonOperationRequest.from_dict(
            DaemonOperationRequest(
                operation="ingest",
                request_id=request.idempotency_key or uuid4().hex,
                archive_root=str(self.server.archive_root),
                payload={
                    "path": str(source),
                    "source_path": request.source_path,
                    "source_name": request.source_name,
                    "idempotency_key": request.idempotency_key,
                },
            ).to_dict()
        )
        response = self._execute_daemon_operation(operation)
        # The upload UI still consumes these display fields. Only the
        # canonical durable reference can establish that ingestion was accepted.
        from polylogue.operations.daemon_protocol import AcceptedOperationReference

        reference = response.get("accepted_reference")
        if reference is not None:
            AcceptedOperationReference.model_validate(reference)
        response["status"] = (
            "accepted"
            if reference is not None and response["outcome"] in {"accepted", "running", "completed", "indeterminate"}
            else "failed"
        )
        response["operation_id"] = operation.request_id
        response["path"] = str(source)
        response["preflight"] = preflight.to_dict()
        response["request"] = request.to_dict()
        self._send_daemon_operation(response)

    @daemon_safe_handler
    def _handle_demo_augment(self) -> None:
        """POST /api/demo/augment — submit the declared ``maintenance.demo.augment`` operation.

        The operation handler is the one executor for demo augmentation; this
        route only validates the body, submits it, and keeps the route's
        historical success shape.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(body_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        if not isinstance(body, dict) or not isinstance(body.get("with_overlays", False), bool):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        from uuid import uuid4

        from polylogue.operations.daemon_protocol import DaemonOperationRequest

        with_overlays = bool(body.get("with_overlays", False))
        operation = DaemonOperationRequest.from_dict(
            DaemonOperationRequest(
                operation="maintenance.demo.augment",
                request_id=uuid4().hex,
                archive_root=str(self.server.archive_root),
                payload={"with_overlays": with_overlays},
            ).to_dict()
        )
        response = self._execute_daemon_operation(operation)
        if response.get("outcome") != "completed":
            self._send_daemon_operation(response)
            return
        self._send_json(HTTPStatus.OK, {"ok": True, "augmented": True, "overlays": with_overlays})


# Bound for concurrent archive-query execution. ThreadingHTTPServer spawns one
# raw OS thread per accepted connection with no cap, so the archive work itself
# is what must be bounded: the compute adapter caps concurrent DB work
# regardless of connection volume, and _ARCHIVE_QUERY_TIMEOUT_S bounds each
# request's wait so a stalled query returns an honest error instead of
# occupying a connection thread forever.
_ARCHIVE_QUERY_MAX_WORKERS = 8
_ARCHIVE_QUERY_TIMEOUT_S = 30.0
#: Default bound on a mutating route's wait for its submitted writer future
#: (polylogue-8r4zq). Longer than the read timeout because a control mutation
#: legitimately queues behind a convergence-pass lease hold, but finite: an
#: unbounded wait is indistinguishable from a wedged daemon.
_MUTATION_WAIT_TIMEOUT_S = 60.0
#: The longest deadline a client-declared header may ask for.
_MUTATION_WAIT_TIMEOUT_MAX_S = 300.0
# Queue depth beyond the worker count. The adapter's admission is finite in
# both work units and estimated bytes, so an exhausted queue rejects with typed
# backpressure instead of accumulating behind an unbounded executor queue.
_ARCHIVE_QUERY_MAX_QUEUED = 16


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
        write_bridge: DaemonWriteThreadBridge | None = None,
        web_credentials: WebCredentialRegistry | None = None,
        webui_dist_root: Path | None = None,
        archive_root: Path | None = None,
        watch_sources: Sequence[Any] | None = None,
        execution_kernel: BoundedComputeAdapter | None = None,
    ) -> None:
        super().__init__(server_address, handler_class)
        validate_declared_route_reachability(handler_class)
        self.auth_token = auth_token
        self.api_host = api_host
        self.webui_dist_root = webui_dist_root
        self.watch_sources = None if watch_sources is None else tuple(watch_sources)
        self.started_at = datetime.now(UTC).isoformat()
        self.web_credentials = web_credentials or WebCredentialRegistry()
        from polylogue.config import load_polylogue_config

        operation_settings = load_polylogue_config()
        from polylogue.paths import hermes_sessions_path

        hermes_root = next(
            (source.root for source in self.watch_sources or () if source.name == "hermes"),
            hermes_sessions_path(),
        )
        if archive_root is None:
            configured_archive_root = operation_settings.archive_root
            if configured_archive_root is None:
                raise RuntimeError("HTTP daemon requires an archive root")
            archive_root = Path(configured_archive_root)
        self.archive_root = archive_root.resolve()
        self.execution_kernel = execution_kernel or BoundedComputeAdapter(
            max_workers=_ARCHIVE_QUERY_MAX_WORKERS,
            queue_units=_ARCHIVE_QUERY_MAX_QUEUED,
            thread_name_prefix="polylogue-compute",
        )
        self._owned_write_runtime: _StandaloneWriteRuntime | None = None
        if write_bridge is None:
            self._owned_write_runtime = _StandaloneWriteRuntime(
                self.archive_root, compute_adapter=self.execution_kernel
            )
            write_bridge = self._owned_write_runtime.bridge
        self.write_bridge: DaemonWriteThreadBridge = write_bridge
        self._compute_close_lock = threading.Lock()
        self._compute_closed = False
        # Diagnostic alias; every submission goes through the adapter above.
        self.archive_query_executor = self.execution_kernel.executor
        self.coordination_cache: dict[tuple[str, int], _CoordinationCacheEntry] = {}
        self.coordination_cache_lock = threading.Lock()
        self.coordination_cache_condition = threading.Condition(self.coordination_cache_lock)
        self.coordination_cache_building: set[tuple[str, int]] = set()
        from polylogue.daemon.operation_runtime import DaemonOperationRuntime
        from polylogue.operations.daemon_reads import DaemonReadDependencies, VectorReadBinding

        vector_binding = VectorReadBinding(
            operation_settings.voyage_api_key,
            operation_settings.embedding_model,
            operation_settings.embedding_dimension,
        )

        from polylogue.daemon.session_profile_composition import compose_session_profile_callback

        self.session_profile_callback = compose_session_profile_callback(
            self.archive_root,
            compute_adapter=self.execution_kernel,
            write_bridge=self.write_bridge,
            now=time,
        )
        from polylogue.core.enums import ValidationMode
        from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner

        self.operation_runtime = DaemonOperationRuntime(
            self.archive_root,
            write_bridge=self.write_bridge,
            execution_kernel=self.execution_kernel,
            raw_observation_owner=RawObservationConvergenceOwner(
                self.archive_root,
                compute_adapter=self.execution_kernel,
                write_bridge=self.write_bridge,
                write_coordinator=self.write_bridge.coordinator,
                validation_mode=ValidationMode.from_string(operation_settings.schema_validation),
            ),
            owner_loop=self.write_bridge.owner_loop,
            session_maintenance=self.session_profile_callback.maintenance,
            read_dependencies_factory=lambda: DaemonReadDependencies(
                vector_binding=vector_binding,
                runtime_status=get_status_snapshot_payload(),
                status_config=operation_settings,
                hermes_root=hermes_root,
            ),
        )
        # Startup recovery has run under this writer (``polylogued`` before
        # constructing the server, a standalone server in its owned runtime),
        # so accepted ingests it left to their owner are re-driven now.
        self.operation_runtime.start_accepted_ingest_redrive()
        # A server that owns its writer still originates no derivation of its
        # own. Session-profile convergence is the daemon's service
        # (``_periodic_convergence_check`` and the watcher drive this same
        # callback under ``polylogued run``); a standalone server only writes
        # on explicit request through ``operation_runtime``. A self-started
        # sweep here advanced the index-tier query-unit frame under a reader's
        # own freshly issued continuation and turned the next page into a
        # ``query_continuation_stale`` conflict with no external mutation.

    def server_close(self) -> None:
        # Staged workers settle before their shared compute owner closes.
        # Standalone composition also drains its writer; polylogued drains
        # those services before calling this method.
        def close_compute() -> None:
            with self._compute_close_lock:
                if self._compute_closed:
                    return
                self._compute_closed = True
            kernel = getattr(self, "execution_kernel", None)
            if isinstance(kernel, BoundedComputeAdapter):
                kernel.shutdown(wait=False, cancel_futures=True)

        owned_write_runtime = getattr(self, "_owned_write_runtime", None)
        self._owned_write_runtime = None
        if owned_write_runtime is not None:
            owned_write_runtime.close(before_drain=self.operation_runtime.shutdown, after_drain=close_compute)
        else:
            runtime = getattr(self, "operation_runtime", None)
            if runtime is None or runtime.shutdown_settled:
                close_compute()
            else:
                settled = asyncio.run_coroutine_threadsafe(runtime.shutdown(), self.write_bridge.owner_loop)
                settled.add_done_callback(lambda future: close_compute() if future.exception() is None else None)
        super().server_close()


async def _recover_startup_with_compute(
    bridge: DaemonWriteThreadBridge, kernel: BoundedComputeAdapter, archive_root: Path
) -> None:
    """Use the eventual daemon creator for original preparation and short publication."""
    from polylogue.core.stage_admission import stage_write_admission
    from polylogue.core.write_lease import adopt_write_lease
    from polylogue.operations.mutation_replay import RECOVERY_SERVICE_ACTOR_REF, recover_interrupted_operations
    from polylogue.storage.sqlite.connection_profile import retained_native_settlement_owners_on_current_thread

    def admit_write(actor: str, work: Callable[[], Any]) -> Any:
        with bridge.hold(actor) as delegation, adopt_write_lease(delegation):
            return work()

    def recover() -> None:
        kernel.require_current_creator()
        with stage_write_admission(admit_write):
            recover_interrupted_operations(
                archive_root,
                resolver_actor_ref=RECOVERY_SERVICE_ACTOR_REF,
                input_demand=kernel.amend_current_input_demand,
            )

    await bridge.coordinator.run_prepared_sync(
        "daemon.operation_recovery.startup",
        recover,
        submit_worker=lambda worker: (
            kernel.submit(propagate(worker), admission_class="control", estimated_bytes=0, exclusive_bytes=True).future
        ),
        settlement_owners=retained_native_settlement_owners_on_current_thread,
    )


class _StandaloneWriteRuntime:
    """Coordinator loop for HTTP-server use outside ``polylogued``.

    The loop serves request-driven writes only; it starts no periodic
    derivation task of its own.
    """

    def __init__(self, archive_root: Path, *, compute_adapter: BoundedComputeAdapter) -> None:
        ready = threading.Event()
        self.loop = asyncio.new_event_loop()
        self.coordinator: DaemonWriteCoordinator | None = None

        def run() -> None:
            asyncio.set_event_loop(self.loop)
            self.coordinator = DaemonWriteCoordinator(archive_root=archive_root)
            register_write_coordinator(self.loop, self.coordinator)
            # Signal readiness from inside the running loop, so a caller that
            # passes the wait can never see is_running() still false.
            self.loop.call_soon(ready.set)
            self.loop.run_forever()
            self.loop.close()

        self.thread = threading.Thread(target=run, name="daemon-http-writer", daemon=True)
        self.thread.start()
        if not ready.wait(5.0):
            raise RuntimeError("standalone daemon HTTP writer loop failed to start")
        assert self.coordinator is not None
        self.bridge: DaemonWriteThreadBridge = DaemonWriteThreadBridge(self.coordinator, self.loop)
        from polylogue.operations.operation_context import prepare_operation_journals

        try:
            self.bridge.run_sync("daemon.operation_journals.startup", prepare_operation_journals, archive_root)
            recovery = asyncio.run_coroutine_threadsafe(
                _recover_startup_with_compute(self.bridge, compute_adapter, archive_root), self.loop
            )
            self.bridge._await_owner_settlement("daemon.operation_recovery.startup", recovery)
        except BaseException:
            self.close()
            raise

    def close(
        self,
        *,
        before_drain: Callable[[], Awaitable[None]] | None = None,
        after_drain: Callable[[], None] | None = None,
    ) -> None:
        assert self.coordinator is not None
        future = asyncio.run_coroutine_threadsafe(self._stop_when_idle(before_drain, after_drain), self.loop)
        future.add_done_callback(
            lambda completed: self.loop.call_soon_threadsafe(self.loop.stop) if completed.exception() is None else None
        )
        try:
            future.result(timeout=5.5)
        except TimeoutError:
            # The task retains the loop, writer and compute owner until actual
            # publication settles. A bounded close does not release ownership.
            emit(
                "daemon.http.runtime.drain_incomplete",
                level=WARNING,
                outcome="unmeasured",
                reason="still_draining_at_server_close",
                phase="shutdown",
                timeout_ms=5500,
            )
            return
        self.thread.join(timeout=1.0)

    async def _stop_when_idle(
        self,
        before_drain: Callable[[], Awaitable[None]] | None = None,
        after_drain: Callable[[], None] | None = None,
    ) -> None:
        assert self.coordinator is not None
        if before_drain is not None:
            await before_drain()
        while not await self.coordinator.shutdown(timeout=5.0):
            emit(
                "daemon.http.writer.drain_incomplete",
                level=WARNING,
                outcome="unmeasured",
                reason="writer_still_draining_after_server_close",
                phase="shutdown",
                timeout_ms=5000,
            )
        if after_drain is not None:
            after_drain()
