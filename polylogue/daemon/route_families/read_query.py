"""HTTP read adapters for the read query route family.

Handlers retain the original request validation, response shaping, and errors.
Each is installed by its RouteSpec binding in the daemon router.
"""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from polylogue.daemon.route_types import RouteSpec, RouteStability
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue


def _handle_overview(self: Any) -> None:
    """Return one bounded, privacy-safe cockpit landing projection."""
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_privacy_safe_projection, _web_reader_archive_root
    from polylogue.daemon.status_snapshot import get_status_snapshot_payload
    from polylogue.operations.http_read_models import read_archive_overview

    archive_root = _web_reader_archive_root()
    if archive_root is None:
        self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "archive_unavailable")
        return
    with archive_read_context(
        archive_root,
        operation="http.archive.read",
        arguments={"path": getattr(self, "path", "")},
        projection="http-read",
    ) as archive:
        archive_overview = read_archive_overview(archive)
    status = get_status_snapshot_payload()
    snapshot = status.get("status_snapshot")
    components = status.get("component_readiness")
    readiness: dict[str, dict[str, object]] = (
        {
            str(name): {"state": value.get("state", "unknown")}
            for name, value in components.items()
            if isinstance(value, Mapping)
        }
        if isinstance(components, Mapping)
        else {}
    )
    snapshot_payload: dict[str, object] = (
        {
            "state": snapshot.get("state", "unknown"),
            "captured_at": snapshot.get("captured_at"),
            "age_s": snapshot.get("age_s"),
            "refresh_error": snapshot.get("refresh_error"),
        }
        if isinstance(snapshot, Mapping)
        else {"state": "unknown", "captured_at": None, "age_s": None, "refresh_error": None}
    )
    overview: dict[str, object] = {
        "mode": "cockpit-overview",
        "totals": {
            "sessions": archive_overview.total_sessions,
            "messages": archive_overview.total_messages,
            "origins": archive_overview.origins,
        },
        "readiness": readiness,
        "status_snapshot": snapshot_payload,
        "recent": [self._archive_summary_payload(summary) for summary in archive_overview.recent],
        "recent_limit": 6,
    }
    from polylogue.paths import archive_root as configured_archive_root

    self._send_json(
        HTTPStatus.OK,
        _web_privacy_safe_projection(overview, archive_root, configured_archive_root()),
    )


def _handle_facets(self: Any, params: dict[str, list[str]]) -> None:
    from polylogue.daemon.http import _build_query_spec_params, _facet_requested_optional_families

    query_params = _build_query_spec_params(params, self)
    include_deferred = bool(_facet_requested_optional_families(params))

    async def _get(poly: Polylogue) -> object:
        return await self._do_facets(poly, query_params, include_deferred=include_deferred)

    result = self._sync_run(_get)
    self._send_json(HTTPStatus.OK, result)


def _handle_ref_resolve(self: Any, params: dict[str, list[str]]) -> None:
    """``GET /api/refs/resolve`` resolves public object/evidence refs.

    The resolution is the shared operation (``operations/ref_resolution``),
    run against the reader this handler pins.  It deliberately no longer
    constructs a ``Polylogue`` facade inside the daemon's own process: that
    opened a second archive generation per request and made the daemon's
    answer a *different execution* of ref resolution from the one the
    annotation importer admits durable rows on (polylogue-j5u2b).
    """
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.operations.ref_resolution import plan_ref_resolution

    ref = self._get_param(params, "ref")
    if not ref:
        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing_ref", "message": "ref query parameter is required"})
        return
    archive_root = _web_reader_archive_root()
    if archive_root is None:
        self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, "archive_unavailable")
        return
    plan = plan_ref_resolution(ref, archive_root=archive_root)
    if plan.payload is not None:
        payload = plan.payload
    else:
        assert plan.read is not None
        with archive_read_context(
            archive_root,
            operation=plan.operation,
            arguments=plan.arguments,
            projection=plan.projection,
            stable_order=plan.stable_order,
        ) as archive:
            payload = plan.read(archive)
    self._send_json(HTTPStatus.OK, payload.model_dump(mode="json", exclude_none=True))


def _handle_query_completions(self: Any, params: dict[str, list[str]]) -> None:
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


def _handle_action_affordances(self: Any) -> None:
    """``GET /api/action-affordances`` exposes shared query-action metadata."""

    from polylogue.operations.action_contracts import action_affordance_list_payload

    payload = action_affordance_list_payload()
    self._send_json(HTTPStatus.OK, payload.model_dump(mode="json"))


def _handle_read_view_profiles(self: Any) -> None:
    """``GET /api/read-view-profiles`` exposes shared read-view metadata."""

    from polylogue.archive.viewport import read_view_http_capability_payloads, read_view_profile_payloads

    capabilities = read_view_http_capability_payloads()
    profiles = []
    for profile in read_view_profile_payloads():
        view_id = profile.get("view_id")
        capability = capabilities.get(view_id) if isinstance(view_id, str) else None
        if capability is not None:
            profile = {**profile, "http": capability}
        profiles.append(profile)
    self._send_json(HTTPStatus.OK, {"read_views": profiles, "total": len(profiles)})


def _handle_paste_browser(self: Any, params: dict[str, list[str]]) -> None:
    limit = self._get_int(params, "limit", 200)
    offset = self._get_int(params, "offset", 0)

    async def _run(poly: Polylogue) -> object:
        return await self._do_paste_browser(poly, limit=limit, offset=offset)

    result = self._sync_run(_run)
    self._send_json(HTTPStatus.OK, result)


def _handle_attachment_library(self: Any, params: dict[str, list[str]]) -> None:
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.daemon.webui_data import LibraryEntry, attachment_to_envelope, build_library_payload
    from polylogue.operations.http_read_models import read_attachment_library_page
    from polylogue.surfaces.payloads import reader_anchor

    limit = self._get_int(params, "limit", 500)
    offset = self._get_int(params, "offset", 0)
    mime_filter = (params.get("mime") or [""])[0]
    state_filter = (params.get("state") or [""])[0]
    session_filter = (params.get("session") or [""])[0]

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            rows = read_attachment_library_page(
                archive,
                limit=limit + 1,
                offset=offset,
                mime_filter=mime_filter,
                state_filter=state_filter,
                session_filter=session_filter,
            )
        entries = [
            LibraryEntry(
                envelope=attachment_to_envelope(att, session_id=str(att.session_id), message_id=att.message_id),
                session_title=title,
                origin=origin,
                message_anchor=reader_anchor("message", att.message_id) if att.message_id else None,
            )
            for att, title, origin in rows
        ]
        page_truncated = len(entries) > limit
        if page_truncated:
            entries = entries[:limit]
        matched_so_far = offset + len(entries)
        result = build_library_payload(
            entries,
            total=None if page_truncated else matched_so_far,
            total_is_exact=not page_truncated,
            matched_so_far=matched_so_far,
        )
    else:

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


def _route(
    path: str,
    handler: str,
    response_contract: str,
    *,
    request_contract: str,
    stability: RouteStability = "stable",
    operation: str | None = None,
    passes_params: bool = True,
    example: ExampleSpec | None = None,
) -> RouteSpec:
    producer = f"polylogue.daemon.http.DaemonAPIHandler.{handler}"
    return RouteSpec(
        kernel=DeclarationSpec(
            declaration_id="daemon.read-query." + path.removeprefix("/api/").replace("/", ".").replace(":", ""),
            family_id="daemon.read-query." + path.removeprefix("/api/").replace("/", "."),
            public_name="read-query-" + path.removeprefix("/api/").replace("/", "-").replace(":", ""),
            owner_path="polylogue/daemon/http.py",
            compatibility=CompatibilityKey("daemon-route", stability, "daemon-read", response_contract, "read-only"),
            producer=producer,
            role_gate="credential_if_configured",
            schema_ref=response_contract,
            discovery_text=f"Read {path} through the daemon query surface.",
            repair_command="devtools render openapi",
            handlers=(HandlerBinding("daemon-http", "polylogue/daemon/http.py", handler, f"GET {path}"),),
            outputs=(OutputSpec("response", "json", response_contract, path),),
            examples=(example or ExampleSpec("default", f"Read {path}", ()),),
            completeness_edges=(
                CompletenessEdge(producer, "daemon-http", "route", "polylogue/daemon/http.py"),
                CompletenessEdge(producer, "openapi-schema", "generated-document", "docs/openapi/search.yaml"),
            ),
        ),
        method="GET",
        path=path,
        request_contract=request_contract,
        response_contract=response_contract,
        auth_policy="credential_if_configured",
        domain_operation=operation,
        passes_params=passes_params,
        kind="read_query",
        stability=stability,
        migration_reason="legacy direct archive/facade read pending product operation" if operation is None else "",
    )


ROUTES: tuple[RouteSpec, ...] = (
    _route(
        "/api/overview",
        "_handle_overview",
        "bounded cockpit overview",
        request_contract="OverviewQuery",
        stability="shell_supported",
        operation="read_archive_overview",
        passes_params=False,
    ),
    _route(
        "/api/facets",
        "_handle_facets",
        "FacetsResponse with route-state metadata",
        request_contract="SessionFacetQuery",
        operation="facets",
        example=ExampleSpec("default", "Read scoped facets", (("limit", 20),)),
    ),
    _route(
        "/api/refs/resolve",
        "_handle_ref_resolve",
        "PublicRefResolutionPayload",
        request_contract="PublicRefQuery",
        operation="ref_resolution.plan_ref_resolution",
        example=ExampleSpec("session", "Resolve a public session reference", (("ref", "session:example"),)),
    ),
    _route(
        "/api/query-completions",
        "_handle_query_completions",
        "query completion metadata",
        request_contract="QueryCompletionQuery",
        operation="query_completion_payload",
        example=ExampleSpec("field", "Complete a query field", (("kind", "field"),)),
    ),
    _route(
        "/api/action-affordances",
        "_handle_action_affordances",
        "ActionAffordanceListPayload",
        request_contract="EmptyQuery",
        operation="action_affordance_list_payload",
        passes_params=False,
    ),
    _route(
        "/api/read-view-profiles",
        "_handle_read_view_profiles",
        "read-view profile metadata",
        request_contract="EmptyQuery",
        operation="read_view_profile_payloads",
        passes_params=False,
    ),
    _route(
        "/api/paste-browser",
        "_handle_paste_browser",
        "paste browser JSON",
        request_contract="PasteBrowserQuery",
        stability="shell_supported",
        operation="query.units",
        example=ExampleSpec("first-page", "Read a bounded paste page", (("limit", 20),)),
    ),
    _route(
        "/api/attachments",
        "_handle_attachment_library",
        "attachment library JSON",
        request_contract="AttachmentLibraryQuery",
        stability="shell_supported",
        operation="read_attachment_library_page",
        example=ExampleSpec("first-page", "Read a bounded attachment page", (("limit", 20),)),
    ),
)


__all__ = ["ROUTES"]
