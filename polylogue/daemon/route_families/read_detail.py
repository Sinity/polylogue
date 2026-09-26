"""HTTP read adapters for the read detail route family.

Handlers retain the original request validation, response shaping, and errors.
Each is installed by its RouteSpec binding in the daemon router.
"""

from __future__ import annotations

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


def _handle_sources(self: Any) -> None:
    from polylogue.operations.http_read_models import read_configured_sources

    self._send_json(HTTPStatus.OK, read_configured_sources())


def _handle_get_session(self: Any, conv_id: str, params: dict[str, list[str]]) -> None:
    from polylogue.daemon.http import _web_reader_archive_root

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        result = (
            self._do_archive_get_session_summary(archive_root, conv_id)
            if self._get_param(params, "shape") == "summary"
            else self._do_archive_get_session(archive_root, conv_id)
        )
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


def _handle_get_messages(self: Any, conv_id: str, params: dict[str, list[str]]) -> None:
    from polylogue.archive.query.spec import clamp_query_limit
    from polylogue.archive.query.transaction import QueryContinuationInvalidError, QueryContinuationStaleError
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.operations.message_locator import MessageNotInSessionError

    limit = clamp_query_limit(self._get_int(params, "limit", 50), default=50)
    offset = max(0, self._get_int(params, "offset", 0))
    continuation = self._get_param(params, "continuation")
    around = self._get_param(params, "around")
    if not self._accept_message_window_anchor(around, continuation):
        return

    archive_root = _web_reader_archive_root()
    try:
        if archive_root is not None:
            payload = self._do_archive_get_messages(archive_root, conv_id, limit, offset, continuation, around)
        else:

            async def _get(poly: Polylogue) -> object:
                return await self._do_get_messages(poly, conv_id, limit, offset, continuation, around)

            payload = self._sync_run(_get)
    except MessageNotInSessionError as exc:
        self._send_error(HTTPStatus.NOT_FOUND, exc.code, str(exc))
        return
    except QueryContinuationStaleError as exc:
        # A write landed since the token was issued; resuming it would page
        # into shifted rows, so the route refuses rather than answers.
        self._send_error(HTTPStatus.CONFLICT, exc.code, str(exc))
        return
    except QueryContinuationInvalidError as exc:
        self._send_error(HTTPStatus.BAD_REQUEST, exc.code, str(exc))
        return
    self._send_json(HTTPStatus.OK, payload)


def _handle_get_session_raw(self: Any, conv_id: str) -> None:
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.operations.http_read_models import read_session_raw

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            result = read_session_raw(archive, conv_id)
    else:

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_raw(poly, conv_id)

        result = self._sync_run(_get)
    if result is None:
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    self._send_json(HTTPStatus.OK, result)


def _handle_get_session_cost(self: Any, conv_id: str) -> None:
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _cost_panel_payload, _empty_cost_payload, _web_reader_archive_root
    from polylogue.operations.http_read_models import read_session_cost

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            cost_read = read_session_cost(archive, conv_id)
        result = (
            None
            if cost_read is None
            else _cost_panel_payload(cost_read.insight)
            if cost_read.insight is not None
            else _empty_cost_payload(conv_id, cost_read.origin)
        )
    else:

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_cost(poly, conv_id)

        result = self._sync_run(_get)
    if result is None:
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    self._send_json(HTTPStatus.OK, result)


def _handle_get_session_evidence_summary(self: Any, conv_id: str) -> None:
    """Return bounded structural counts for the reader evidence strip."""
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _cost_panel_payload, _empty_cost_payload, _web_reader_archive_root
    from polylogue.logging import WARNING, emit
    from polylogue.operations.http_read_models import read_session_evidence
    from polylogue.surfaces.outcome import decide_outcome

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
        evidence = read_session_evidence(archive, conv_id)
    if evidence is None:
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    gaps = ("lineage_refs_unreadable",) if evidence.lineage_unreadable else ()
    if evidence.lineage_error is not None:
        exc = evidence.lineage_error
        emit(
            "daemon.http.session_evidence_degraded",
            level=WARNING,
            outcome="degraded",
            reason="lineage_refs_unreadable",
            session_id=conv_id,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
    cost_read = evidence.cost
    cost = (
        _cost_panel_payload(cost_read.insight)
        if cost_read.insight is not None
        else _empty_cost_payload(conv_id, cost_read.origin)
    )
    self._send_json(
        HTTPStatus.OK,
        {
            "mode": "session-evidence-summary",
            "session_id": evidence.session_id,
            "origin": evidence.origin,
            "tool_calls": evidence.tool_calls,
            "outcomes": dict(zip(("ok", "failed", "unknown"), evidence.outcomes, strict=True)),
            "cost": {"total_usd": cost.get("total_usd"), "confidence_tag": cost.get("confidence_tag", "q-missing")},
            "lineage_refs": [
                {"session_id": session_id, "kind": kind, "status": status}
                for session_id, kind, status in evidence.lineage_refs
            ],
            "lineage_refs_authoritative": not gaps,
            "lineage_limit": 20,
            "outcome": decide_outcome(
                matched=evidence.tool_calls + len(evidence.lineage_refs), degraded=gaps
            ).to_dict(),
        },
    )


def _handle_get_session_provenance(
    self: Any,
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


def _handle_get_session_topology(
    self: Any,
    conv_id: str,
    params: dict[str, list[str]],
) -> None:
    """``GET /api/sessions/{id}/topology[?limit=N]``.

    Returns a bounded :class:`polylogue.analysis.topology.SessionTopology`
    envelope rooted at *conv_id*'s lineage root. ``?limit=`` is the
    operator-visible knob; the daemon enforces the hard cap from
    :data:`polylogue.daemon.topology_http.MAX_NODE_LIMIT` regardless of
    client input (#1121 AC: lineage rendering is bounded).
    """
    import asyncio

    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.daemon.topology_http import (
        build_topology_envelope,
        coerce_node_limit,
        coerce_node_offset,
    )
    from polylogue.operations.http_read_models import read_session_topology

    node_limit = coerce_node_limit(self._get_param(params, "limit"))
    if node_limit is None:
        self._send_error(HTTPStatus.BAD_REQUEST, "invalid_limit")
        return
    node_offset = coerce_node_offset(self._get_param(params, "continuation"))
    if node_offset is None:
        self._send_error(HTTPStatus.BAD_REQUEST, "invalid_continuation")
        return

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            topology = asyncio.run(
                read_session_topology(archive, conv_id, node_offset=node_offset, node_limit=node_limit)
            )
        result = build_topology_envelope(topology, node_limit=node_limit) if topology is not None else None
    else:

        async def _get(poly: Polylogue) -> object:
            topology = await poly.get_session_topology(conv_id, node_offset=node_offset, node_limit=node_limit)
            if topology is None:
                return None
            return build_topology_envelope(topology, node_limit=node_limit)

        result = self._sync_run(_get)
    if result is None:
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    self._send_json(HTTPStatus.OK, result)


def _handle_get_session_parent_chain(
    self: Any,
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
    import asyncio

    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.daemon.topology_http import build_parent_chain_envelope
    from polylogue.operations.http_read_models import read_session_topology

    include_descendants_raw = self._get_param(params, "descendants", "1") or "1"
    include_descendants = include_descendants_raw not in ("0", "false", "no")

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            topology = asyncio.run(read_session_topology(archive, conv_id))
        result = (
            build_parent_chain_envelope(topology, include_descendants=include_descendants)
            if topology is not None
            else None
        )
    else:

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


def _handle_get_session_similar(
    self: Any,
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


def _handle_get_session_attachments(self: Any, conv_id: str) -> None:
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.daemon.webui_data import attachment_to_envelope
    from polylogue.operations.http_read_models import read_session_attachments

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            attachments_read = read_session_attachments(archive, conv_id)
        result = (
            None
            if attachments_read is None
            else {
                "items": [
                    attachment_to_envelope(attachment, session_id=attachments_read.session_id, message_id=message_id)
                    for attachment, message_id in attachments_read.attachments
                ],
                "total": len(attachments_read.attachments),
            }
        )
    else:

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_attachments(poly, conv_id)

        result = self._sync_run(_get)
    if result is None:
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    self._send_json(HTTPStatus.OK, result)


def _handle_get_session_insights(self: Any, conv_id: str, params: dict[str, list[str]]) -> None:
    """Return the bounded profile and thread panels for one session."""
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _parse_insight_includes, _web_reader_archive_root
    from polylogue.operations.http_read_models import read_session_insights

    include_raw = self._get_param(params, "include")
    includes = _parse_insight_includes(include_raw)
    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            read = read_session_insights(archive, conv_id, includes)
        result = _session_insight_panel_payload(read, includes) if read is not None else None
    else:

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_session_insights(poly, conv_id, includes)

        result = self._sync_run(_get)
    if result is None:
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    self._send_json(HTTPStatus.OK, result)


def _session_insight_panel_payload(read: object, includes: tuple[str, ...]) -> dict[str, object]:
    from polylogue.analysis.archive import SessionProfileInsight
    from polylogue.daemon.http import (
        _empty_profile_panel_payload,
        _profile_panel_payload,
        _profile_staleness,
        _thread_panel_payload,
    )
    from polylogue.logging import WARNING, emit
    from polylogue.operations.http_read_models import SessionInsightsRead
    from polylogue.surfaces.outcome import OutcomeEnvelope, combine_outcomes, decide_outcome

    assert isinstance(read, SessionInsightsRead)
    envelope: dict[str, object] = {
        "session_id": read.session_id,
        "origin": read.origin,
        "include": list(includes),
        "kinds": {},
    }
    kinds = envelope["kinds"]
    assert isinstance(kinds, dict)
    panel_outcomes: list[OutcomeEnvelope] = []

    def unavailable(kind: str, exc: BaseException) -> OutcomeEnvelope:
        emit(
            "daemon.http.session_insight_unavailable",
            level=WARNING,
            outcome="degraded",
            reason="insight_unavailable",
            kind=kind,
            session_id=read.session_id,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return decide_outcome(matched=0, error=f"insight_unavailable:{kind}")

    if "profile" in includes:
        profile_outcome = unavailable("profile", read.profile_error) if read.profile_error else None
        record = read.profile_record
        profile = read.profile
        profile_insight = SessionProfileInsight.from_record(record, tier="evidence") if record is not None else None
        panel = (
            _profile_panel_payload(profile, profile_insight.provenance)
            if profile is not None and profile_insight is not None
            else _empty_profile_panel_payload(profile_outcome or decide_outcome(matched=0))
        )
        if profile is not None and (read.partition_status != "valid" or not read.profile_row_matches):
            row_count = int(profile.message_count or 0)
            panel["outcome"] = decide_outcome(matched=row_count, degraded=("session_profile_stale",)).to_dict()
            panel["readiness_tag"] = "q-partial"
            panel["materialized"] = False
        panel_outcomes.append(OutcomeEnvelope.model_validate(panel["outcome"]))
        if profile is not None:
            staleness = _profile_staleness(record, read.updated_at)
            if staleness is not None:
                if read.partition_status != "valid" or not read.profile_row_matches:
                    staleness["stale"] = True
                    staleness["reason"] = "session_profile_stale"
                panel["staleness"] = staleness
        kinds["profile"] = panel

    if "threads" in includes:
        threads_error = unavailable("threads", read.threads_error) if read.threads_error else None
        threads_outcome = threads_error or decide_outcome(matched=len(read.threads))
        panel_outcomes.append(threads_outcome)
        kinds["threads"] = _thread_panel_payload(list(read.threads), threads_outcome)

    envelope["outcome"] = combine_outcomes(panel_outcomes).to_dict()
    return envelope


def _handle_get_raw_artifact(self: Any, artifact_id: str) -> None:
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.operations.http_read_models import read_raw_artifacts

    archive_root = _web_reader_archive_root()
    if archive_root is not None:
        with archive_read_context(
            archive_root,
            operation="http.archive.read",
            arguments={"path": getattr(self, "path", "")},
            projection="http-read",
        ) as archive:
            result = read_raw_artifacts(archive, artifact_id)
    else:

        async def _get(poly: Polylogue) -> object:
            return await self._do_get_raw_artifacts(poly, artifact_id)

        result = self._sync_run(_get)
    self._send_json(HTTPStatus.OK, result)


def _handle_get_thread_continue_templates(self: Any) -> None:
    """``GET /api/thread-continue-templates``.

    Returns the active agent URL-template registry. Templates are
    substituted client-side so the daemon never sees the messages
    the operator is "continuing" in another agent.
    """
    from polylogue.daemon.thread_continue import build_templates_envelope

    self._send_json(HTTPStatus.OK, build_templates_envelope())


def _route(
    path: str,
    handler: str,
    response_contract: str,
    *,
    request_contract: str,
    stability: RouteStability = "stable",
    operation: str | None = None,
    passes_params: bool = False,
    example: ExampleSpec | None = None,
) -> RouteSpec:
    producer = f"polylogue.daemon.http.DaemonAPIHandler.{handler}"
    return RouteSpec(
        kernel=DeclarationSpec(
            declaration_id="daemon.read-detail." + path.removeprefix("/api/").replace("/", ".").replace(":", ""),
            family_id="daemon.read-detail." + path.removeprefix("/api/").replace("/", ".").replace(":", ""),
            public_name="read-detail-" + path.removeprefix("/api/").replace("/", "-").replace(":", ""),
            owner_path="polylogue/daemon/http.py",
            compatibility=CompatibilityKey("daemon-route", stability, "daemon-read", response_contract, "read-only"),
            producer=producer,
            role_gate="credential_if_configured",
            schema_ref=response_contract,
            discovery_text=f"Read {path} through the daemon detail surface.",
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
        kind="read_detail",
        stability=stability,
        migration_reason="legacy direct archive/facade read pending product operation" if operation is None else "",
    )


ROUTES: tuple[RouteSpec, ...] = (
    _route(
        "/api/sources",
        "_handle_sources",
        "source list JSON",
        request_contract="EmptyQuery",
        stability="shell_supported",
        operation="read_configured_sources",
    ),
    _route(
        "/api/sessions/:id",
        "_handle_get_session",
        "Session detail JSON",
        request_contract="SessionDetailQuery",
        operation="execute_http_session_detail",
        passes_params=True,
    ),
    _route(
        "/api/sessions/:id/messages",
        "_handle_get_messages",
        "session messages JSON",
        request_contract="SessionMessagesQuery",
        operation="execute_http_session_messages",
        passes_params=True,
        example=ExampleSpec("first-page", "Read a session message page", (("limit", 50),)),
    ),
    _route(
        "/api/sessions/:id/raw",
        "_handle_get_session_raw",
        "raw session payload JSON",
        request_contract="SessionRawQuery",
        stability="shell_supported",
        operation="read_session_raw",
    ),
    _route(
        "/api/sessions/:id/cost",
        "_handle_get_session_cost",
        "cost JSON",
        request_contract="SessionCostQuery",
        stability="shell_supported",
        operation="read_session_cost",
    ),
    _route(
        "/api/sessions/:id/evidence-summary",
        "_handle_get_session_evidence_summary",
        "bounded session evidence summary",
        request_contract="SessionEvidenceQuery",
        stability="shell_supported",
        operation="read_session_evidence",
    ),
    _route(
        "/api/sessions/:id/provenance",
        "_handle_get_session_provenance",
        "provenance envelope",
        request_contract="SessionProvenanceQuery",
        operation="build_provenance_payload",
        passes_params=True,
        example=ExampleSpec("metadata", "Read source provenance", (("include_raw", 0),)),
    ),
    _route(
        "/api/sessions/:id/topology",
        "_handle_get_session_topology",
        "topology envelope",
        request_contract="SessionTopologyQuery",
        operation="read_session_topology",
        passes_params=True,
    ),
    _route(
        "/api/sessions/:id/topology/parent-chain",
        "_handle_get_session_parent_chain",
        "parent-chain topology envelope",
        request_contract="ParentChainQuery",
        operation="read_session_topology",
        passes_params=True,
    ),
    _route(
        "/api/sessions/:id/similar",
        "_handle_get_session_similar",
        "similar-session envelope",
        request_contract="SimilarSessionQuery",
        operation="build_similar_payload",
        passes_params=True,
    ),
    _route(
        "/api/sessions/:id/attachments",
        "_handle_get_session_attachments",
        "session attachment envelope",
        request_contract="SessionAttachmentsQuery",
        stability="shell_supported",
        operation="read_session_attachments",
    ),
    _route(
        "/api/insights/sessions/:id",
        "_handle_get_session_insights",
        "session insights envelope",
        request_contract="SessionInsightsQuery",
        operation="read_session_insights",
        passes_params=True,
    ),
    _route(
        "/api/raw_artifacts/:id",
        "_handle_get_raw_artifact",
        "raw artifact preview",
        request_contract="RawArtifactQuery",
        stability="shell_supported",
        operation="read_raw_artifacts",
    ),
    _route(
        "/api/thread-continue-templates",
        "_handle_get_thread_continue_templates",
        "thread continuation templates",
        request_contract="EmptyQuery",
        stability="shell_supported",
        operation="build_templates_envelope",
    ),
)


__all__ = ["ROUTES"]
