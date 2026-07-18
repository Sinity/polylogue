"""Protocol-native MCP read and privileged algebra.

The compatibility registrars remain internal implementation substrate. This
module is the public cutover registration surface: the six default read
transactions (query/read/get/explain/context/status), plus the role-gated
privileged transactions (write/judge/run/maintenance) registered only for
roles whose ladder includes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from polylogue.mcp.declarations.adapter import register_declared_handler
from polylogue.mcp.declarations.models import mcp_role_allows
from polylogue.mcp.payloads import MCPArchiveStatsPayload, MCPRootPayload, session_topology_payload

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.coordination import CoordinationEnvelopeCache
    from polylogue.maintenance.scope import MaintenanceScopeFilter
    from polylogue.mcp.declarations.adapter import ToolRegistrar
    from polylogue.mcp.server_support import ServerCallbacks
    from polylogue.surfaces.payloads import ContextPreambleProjectState


@dataclass(frozen=True)
class _EmbeddingStatusEnv:
    """Adapts ``ServerCallbacks`` config access to ``embedding_status_payload``'s ``_HasConfig`` protocol."""

    config: Config


def _object_ref(ref: str) -> str:
    """Lower a stable Polylogue URI to the public ref accepted by the API."""
    if not ref.startswith("polylogue://"):
        return ref
    prefix, separator, object_id = ref.removeprefix("polylogue://").partition("/")
    if not separator or not object_id:
        raise ValueError("stable Polylogue URIs require an object kind and id")
    return f"{prefix}:{object_id}"


async def _query_sessions(
    hooks: ServerCallbacks,
    *,
    expression: str | None,
    limit: int | None,
    origin: str | None,
    tag: str | None,
    repo: str | None,
    since: str | None,
    until: str | None,
    sort: str | None,
    min_messages: int | None,
    max_messages: int | None,
    min_words: int | None,
) -> str:
    """Session-level rows for ``query(projection="sessions", ...)``.

    Ranked (top-k) search when ``expression`` is given as free text;
    otherwise an exhaustive session listing filtered by the other
    parameters. Reuses the same ``archive_search_payload`` /
    ``archive_session_list_payload`` machinery the retired ``search`` /
    ``list_sessions`` tools used, since ``query_units`` (the DSL path)
    explicitly rejects ``sessions`` as a terminal unit source.
    """
    from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest
    from polylogue.mcp.archive_support import archive_search_payload, archive_session_list_payload, mcp_archive_root
    from polylogue.mcp.query_contracts import build_session_query_request

    request = build_session_query_request(
        query=expression,
        origin=origin,
        tag=tag,
        repo=repo,
        since=since,
        until=until,
        sort=sort,
        limit=limit,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    clamped_limit = hooks.clamp_limit(request.limit)
    spec = request.build_spec(hooks.clamp_limit)
    effective_offset = max(0, spec.offset)
    config = hooks.get_config()
    archive_root = mcp_archive_root(config)

    if request.query:
        from polylogue.surfaces.payloads import search_cursor_request_identity

        transaction = QueryTransaction(
            archive_root,
            QueryTransactionRequest(
                operation="query",
                arguments=request.response_arguments(),
                page_size=clamped_limit,
                offset=effective_offset,
                projection="search-envelope",
                stable_order=request.sort or "date",
            ),
        )
        with hooks.response_context("query", request.response_arguments()):
            return hooks.json_payload(
                await transaction.run(
                    lambda archive: archive_search_payload(
                        archive,
                        spec,
                        query=request.query or "",
                        limit=clamped_limit,
                        offset=effective_offset,
                        retrieval_lane=request.retrieval_lane or "dialogue",
                        sort=request.sort,
                        config=config,
                        archive_root=archive_root,
                        include_affordances=False,
                        cursor=None,
                        request_identity=search_cursor_request_identity(request.response_arguments()),
                    )
                )
            )

    transaction = QueryTransaction(
        archive_root,
        QueryTransactionRequest(
            operation="query",
            arguments=request.response_arguments(),
            page_size=clamped_limit,
            offset=effective_offset,
            projection="session-summary",
            stable_order=request.sort or "date",
        ),
    )
    with hooks.response_context("query", request.response_arguments()):
        return hooks.json_payload(
            await transaction.run(
                lambda archive: archive_session_list_payload(archive, spec, config=config, archive_root=archive_root)
            )
        )


#: ``query(projection=..., ...)`` values that list a durable personal-state
#: record kind rather than an archive content unit. Restores read access the
#: retired per-tool registrars (``server_mutation_tools.py``,
#: ``server_personal_state_tools.py``, ``server_tools.py``'s
#: ``blackboard_list``) used to provide -- the underlying facade calls never
#: moved.
_PERSONAL_STATE_PROJECTIONS = frozenset(
    {"marks", "annotations", "saved_views", "recall_packs", "workspaces", "corrections", "blackboard"}
)

#: ``query(projection=..., ...)`` values that compile a distilled insight
#: report over a matched session scope, rather than listing content units.
_INSIGHT_PROJECTIONS = frozenset({"postmortem", "pathologies", "abandoned_sessions", "stuck_sessions"})


def _saved_view_payload(row: dict[str, str]) -> Any:
    from polylogue.mcp.payloads import MCPSavedViewPayload

    try:
        query = json.loads(row["query_json"])
    except (json.JSONDecodeError, TypeError):
        query = {}
    if not isinstance(query, dict):
        query = {}
    return MCPSavedViewPayload(view_id=row["view_id"], name=row["name"], query=query, created_at=row["created_at"])


def _recall_pack_payload(row: dict[str, str]) -> Any:
    from polylogue.mcp.payloads import MCPRecallPackPayload

    try:
        session_ids = json.loads(row["session_ids_json"])
    except (json.JSONDecodeError, TypeError):
        session_ids = []
    if not isinstance(session_ids, list):
        session_ids = []
    try:
        payload = json.loads(row["payload_json"])
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return MCPRecallPackPayload(
        pack_id=row["pack_id"],
        label=row["label"],
        session_ids=tuple(str(item) for item in session_ids),
        payload=payload,
        created_at=row["created_at"],
    )


def _workspace_payload(row: dict[str, str]) -> Any:
    from polylogue.mcp.payloads import MCPReaderWorkspacePayload

    try:
        open_targets = json.loads(row["open_targets_json"])
    except (json.JSONDecodeError, TypeError):
        open_targets = []
    if not isinstance(open_targets, list):
        open_targets = []
    try:
        layout = json.loads(row["layout_json"])
    except (json.JSONDecodeError, TypeError):
        layout = {}
    if not isinstance(layout, dict):
        layout = {}
    try:
        active_target = json.loads(row["active_target_json"])
    except (json.JSONDecodeError, TypeError):
        active_target = {}
    if not isinstance(active_target, dict):
        active_target = {}
    return MCPReaderWorkspacePayload(
        workspace_id=row["workspace_id"],
        name=row["name"],
        mode=row["mode"],
        open_targets=tuple(item for item in open_targets if isinstance(item, dict)),
        layout=layout,
        active_target=active_target,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def _query_personal_state(hooks: ServerCallbacks, projection: str, *, limit: int | None) -> str:
    """``query(projection=<personal-state kind>, ...)`` -- list durable personal-state records.

    Thin dispatch onto the already-live, already-tested ``Polylogue`` list
    facade methods. Offset pagination is not yet exposed here, mirroring the
    ``projection="sessions"`` precedent (also fixed at offset 0).
    """
    from polylogue.mcp.archive_support import blackboard_note_payload
    from polylogue.mcp.mutation_support import page_items
    from polylogue.mcp.payloads import (
        MCPBlackboardNoteListPayload,
        MCPReaderWorkspaceListPayload,
        MCPRecallPackListPayload,
        MCPSavedViewListPayload,
        MCPUserAnnotationListPayload,
        MCPUserAnnotationPayload,
        MCPUserMarkListPayload,
        MCPUserMarkPayload,
    )

    poly = hooks.get_polylogue()
    clamped_limit = hooks.clamp_limit(limit)

    with hooks.response_context("query", {"projection": projection, "limit": clamped_limit}):
        if projection == "marks":
            rows = await poly.list_marks()
            mark_items = tuple(
                MCPUserMarkPayload(
                    target_type=row["target_type"],
                    target_id=row["target_id"],
                    session_id=row["session_id"],
                    message_id=row.get("message_id") or None,
                    mark_type=row["mark_type"],
                    created_at=row["created_at"],
                )
                for row in rows
            )
            mark_page, mark_total, mark_offset, mark_next_offset = page_items(mark_items, limit=clamped_limit, offset=0)
            return hooks.json_payload(
                MCPUserMarkListPayload(
                    items=mark_page,
                    total=mark_total,
                    limit=clamped_limit,
                    offset=mark_offset,
                    next_offset=mark_next_offset,
                )
            )

        if projection == "annotations":
            rows = await poly.list_annotations()
            annotation_items = tuple(
                MCPUserAnnotationPayload(
                    annotation_id=row["annotation_id"],
                    target_type=row["target_type"],
                    target_id=row["target_id"],
                    session_id=row["session_id"],
                    message_id=row.get("message_id") or None,
                    note_text=row["note_text"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            )
            annotation_page, annotation_total, annotation_offset, annotation_next_offset = page_items(
                annotation_items, limit=clamped_limit, offset=0
            )
            return hooks.json_payload(
                MCPUserAnnotationListPayload(
                    items=annotation_page,
                    total=annotation_total,
                    limit=clamped_limit,
                    offset=annotation_offset,
                    next_offset=annotation_next_offset,
                )
            )

        if projection == "saved_views":
            rows = await poly.list_views()
            view_items = tuple(_saved_view_payload(row) for row in rows)
            view_page, view_total, view_offset, view_next_offset = page_items(view_items, limit=clamped_limit, offset=0)
            return hooks.json_payload(
                MCPSavedViewListPayload(
                    items=view_page,
                    total=view_total,
                    limit=clamped_limit,
                    offset=view_offset,
                    next_offset=view_next_offset,
                )
            )

        if projection == "recall_packs":
            rows = await poly.list_recall_packs()
            pack_items = tuple(_recall_pack_payload(row) for row in rows)
            pack_page, pack_total, pack_offset, pack_next_offset = page_items(pack_items, limit=clamped_limit, offset=0)
            return hooks.json_payload(
                MCPRecallPackListPayload(
                    items=pack_page,
                    total=pack_total,
                    limit=clamped_limit,
                    offset=pack_offset,
                    next_offset=pack_next_offset,
                )
            )

        if projection == "workspaces":
            rows = await poly.list_workspaces()
            workspace_items = tuple(_workspace_payload(row) for row in rows)
            workspace_page, workspace_total, workspace_offset, workspace_next_offset = page_items(
                workspace_items, limit=clamped_limit, offset=0
            )
            return hooks.json_payload(
                MCPReaderWorkspaceListPayload(
                    items=workspace_page,
                    total=workspace_total,
                    limit=clamped_limit,
                    offset=workspace_offset,
                    next_offset=workspace_next_offset,
                )
            )

        if projection == "corrections":
            corrections = await poly.list_corrections()
            all_correction_items = [
                {
                    "session_id": correction.session_id,
                    "kind": correction.kind.value,
                    "payload": dict(correction.payload),
                    "note": correction.note,
                    "created_at": correction.created_at.isoformat(),
                }
                for correction in corrections
            ]
            correction_page = all_correction_items[:clamped_limit]
            correction_next_offset = len(correction_page) if len(correction_page) < len(all_correction_items) else None
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "corrections": correction_page,
                        "total": len(all_correction_items),
                        "limit": clamped_limit,
                        "offset": 0,
                        "next_offset": correction_next_offset,
                    }
                )
            )

        assert projection == "blackboard", f"unhandled personal-state projection: {projection}"
        notes = await poly.list_blackboard_notes(limit=clamped_limit)
        note_items = tuple(blackboard_note_payload(note) for note in notes)
        return hooks.json_payload(MCPBlackboardNoteListPayload(items=note_items, total=len(note_items)))


async def _query_insight_projection(
    hooks: ServerCallbacks,
    projection: str,
    *,
    limit: int | None,
    origin: str | None,
    tag: str | None,
    repo: str | None,
    since: str | None,
    until: str | None,
) -> str:
    """``query(projection="postmortem"|"pathologies"|"abandoned_sessions"|"stuck_sessions", ...)``.

    ``postmortem``/``pathologies`` reuse the same filter-building scaffolding
    ``_query_sessions`` does (``build_session_query_request`` ->
    ``.build_spec(...)``), then drop the MCP default page limit before handing
    the spec to the analysis-capped facade call -- otherwise the shared
    ``build_spec`` page-size default (10) would silently cap the matched scope
    to 10 sessions and defeat the facade's own 200-session analysis cap.
    """
    from dataclasses import replace

    from polylogue.mcp.query_contracts import build_session_query_request

    poly = hooks.get_polylogue()

    with hooks.response_context(
        "query",
        {"projection": projection, "origin": origin, "tag": tag, "repo": repo, "since": since, "until": until},
    ):
        if projection in ("postmortem", "pathologies"):
            request = build_session_query_request(origin=origin, tag=tag, repo=repo, since=since, until=until)
            spec = replace(request.build_spec(hooks.clamp_limit), limit=None)
            if projection == "postmortem":
                bundle = await poly.postmortem_bundle(spec, limit=limit)
                return hooks.json_payload(bundle, exclude_none=True)
            report = await poly.pathology_report(spec, limit=limit)
            return hooks.json_payload(report, exclude_none=True)

        if projection == "abandoned_sessions":
            abandoned = await poly.find_abandoned_sessions(since=since, repo_path=repo, limit=hooks.clamp_limit(limit))
            return hooks.json_payload(MCPRootPayload(root=abandoned), exclude_none=True)

        assert projection == "stuck_sessions", f"unhandled insight projection: {projection}"
        from polylogue.insights.archive import SessionLatencyProfileInsightQuery

        stuck = await poly.find_stuck_session_latency_profile_insights(
            SessionLatencyProfileInsightQuery(origin=origin, since=since, until=until, limit=hooks.clamp_limit(limit))
        )
        return hooks.json_payload(
            MCPRootPayload(root={"items": [insight.model_dump(mode="json") for insight in stuck], "total": len(stuck)}),
            exclude_none=True,
        )


def _git_project_state(cwd: str | None) -> ContextPreambleProjectState | None:
    """Read branch + recent commits from a local git checkout, best-effort.

    Never raises: a missing/non-git ``cwd`` must not break SessionStart
    context injection.
    """
    import subprocess

    from polylogue.surfaces.payloads import ContextPreambleProjectState

    try:
        branch: str | None = None
        commits: list[str] = []
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd or ".",
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
        result2 = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd or ".",
        )
        if result2.returncode == 0:
            commits = [line.strip() for line in result2.stdout.strip().split("\n") if line]
        if branch or commits:
            return ContextPreambleProjectState(branch=branch, recent_commits=commits)
    except Exception:
        pass
    return None


async def _resume_preamble(
    hooks: ServerCallbacks,
    *,
    session_id: str | None,
    repo_path: str | None,
    cwd: str | None,
    recent_files: tuple[str, ...],
    related_limit: int,
) -> str:
    """Build the SessionStart preamble: lineage, resume candidates, project git state, assertion guidance."""
    from polylogue.context.preamble import build_context_preamble_payload
    from polylogue.surfaces.payloads import ContextPreamble

    preamble = await build_context_preamble_payload(
        hooks.get_polylogue(),
        session_id=session_id or "",
        related_limit=related_limit,
        repo_path=repo_path,
        cwd=cwd,
        recent_files=recent_files,
        source_tool_calls={"context": "polylogue-mcp"},
        require_session=False,
    )
    if preamble is None:
        preamble = ContextPreamble(preamble_version="1.0", source_tool_calls={"context": "polylogue-mcp"})

    project = _git_project_state(cwd)
    if project is not None:
        payload = preamble.model_dump(mode="json", exclude_none=True)
        payload["project_state"] = project.model_dump(mode="json", exclude_none=True)
        preamble = ContextPreamble.model_validate(payload)
    return hooks.json_payload(preamble, exclude_none=True)


def register_cutover_read_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    """Register exactly the six default, role-read transactions."""

    # Deferred import + lazy singleton: every other branch in this module
    # imports its dependencies inside the closure that uses them, to keep
    # server construction itself cheap; a status(scope="coordination") call
    # never needs to happen in a given session, so this avoids paying
    # coordination.envelope's import cost for sessions that never make one.
    _coordination_cache_holder: list[CoordinationEnvelopeCache] = []

    def _coordination_cache() -> CoordinationEnvelopeCache:
        if not _coordination_cache_holder:
            from polylogue.coordination import CoordinationEnvelopeCache

            _coordination_cache_holder.append(CoordinationEnvelopeCache())
        return _coordination_cache_holder[0]

    async def query(
        expression: str | None = None,
        limit: int | None = None,
        projection: str = "default",
        continuation: str | None = None,
        origin: str | None = None,
        tag: str | None = None,
        repo: str | None = None,
        since: str | None = None,
        until: str | None = None,
        sort: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> str:
        """Execute a terminal DSL page, or resume it using only its q2 token.

        ``projection="sessions"`` switches to session-level rows instead of
        unit-source rows: ``expression`` becomes a free-text ranked search
        (top-k) when given, or an exhaustive listing filtered by
        origin/tag/repo/since/until/sort/min_messages/max_messages/min_words
        when omitted.

        ``projection`` also accepts durable personal-state kinds --
        ``"marks"``, ``"annotations"``, ``"saved_views"``, ``"recall_packs"``,
        ``"workspaces"``, ``"corrections"``, ``"blackboard"`` -- each listing
        that record kind (unfiltered, ``limit``-bounded, offset fixed at 0).

        ``projection="postmortem"``/``"pathologies"`` compile a distilled
        postmortem bundle / pathology-finding report over the session scope
        matched by origin/tag/repo/since/until (200-session analysis cap by
        default; ``limit`` raises or lowers it). ``projection`` also accepts
        ``"abandoned_sessions"`` (dangling-work terminal states) and
        ``"stuck_sessions"`` (latency-profile-flagged stuck sessions), scoped
        by the same origin/since/until filters.

        Continuation is not yet implemented for any of these non-default
        projections.
        """

        async def run() -> str:
            if projection == "sessions":
                if continuation is not None:
                    return hooks.error_json(
                        "query(projection='sessions') does not support continuation yet",
                        code="invalid_continuation",
                        tool="query",
                    )
                return await _query_sessions(
                    hooks,
                    expression=expression,
                    limit=limit,
                    origin=origin,
                    tag=tag,
                    repo=repo,
                    since=since,
                    until=until,
                    sort=sort,
                    min_messages=min_messages,
                    max_messages=max_messages,
                    min_words=min_words,
                )

            if projection in _PERSONAL_STATE_PROJECTIONS:
                if continuation is not None:
                    return hooks.error_json(
                        f"query(projection={projection!r}) does not support continuation yet",
                        code="invalid_continuation",
                        tool="query",
                    )
                return await _query_personal_state(hooks, projection, limit=limit)

            if projection in _INSIGHT_PROJECTIONS:
                if continuation is not None:
                    return hooks.error_json(
                        f"query(projection={projection!r}) does not support continuation yet",
                        code="invalid_continuation",
                        tool="query",
                    )
                return await _query_insight_projection(
                    hooks,
                    projection,
                    limit=limit,
                    origin=origin,
                    tag=tag,
                    repo=repo,
                    since=since,
                    until=until,
                )

            from polylogue.archive.query.transaction import (
                QueryArchiveEpochUnreadableError,
                QueryContinuationInvalidError,
                QueryContinuationStaleError,
            )

            try:
                with hooks.response_context("query", {}):
                    payload = await hooks.get_polylogue().query_units(
                        expression,
                        limit=limit,
                        continuation=continuation,
                    )
                    return hooks.json_payload(payload)
            except QueryContinuationInvalidError as exc:
                return hooks.error_json(str(exc), code=exc.code, tool="query")
            except QueryContinuationStaleError as exc:
                return hooks.error_json(str(exc), code=exc.code, tool="query")
            except QueryArchiveEpochUnreadableError as exc:
                return hooks.error_json(str(exc), code=exc.code, tool="query")

        return await hooks.async_safe_call("query", run)

    async def read(
        ref: str,
        view: str | None = None,
        limit: int | None = None,
        continuation: str | None = None,
    ) -> str:
        """Read a stable URI or public ref through an explicitly named view."""
        if continuation is not None:
            return hooks.error_json(
                "read continuations are not implemented for this view; use query for exhaustive rows",
                code="invalid_continuation",
                tool="read",
            )
        del limit

        normalized = _object_ref(ref)
        session_id = normalized.removeprefix("session:") if normalized.startswith("session:") else None

        async def run() -> str:
            if view == "topology":
                topology_session_id = normalized.removeprefix("session:")
                topology = await hooks.get_polylogue().get_session_topology(topology_session_id)
                if topology is None:
                    return hooks.error_json(f"object not found: {ref}", code="not_found", tool="read")
                return hooks.json_payload(session_topology_payload(topology, session_id=str(topology.target_id)))
            return hooks.json_payload(await hooks.get_polylogue().resolve_ref(normalized))

        return await hooks.async_safe_call("read", run, session_id=session_id)

    async def get(ref: str, projection: str | None = None) -> str:
        """Resolve one exact stable object or evidence identity."""
        del projection
        normalized = _object_ref(ref)
        session_id = normalized.removeprefix("session:") if normalized.startswith("session:") else None

        async def run() -> str:
            return hooks.json_payload(await hooks.get_polylogue().resolve_ref(normalized))

        return await hooks.async_safe_call("get", run, session_id=session_id)

    async def explain(
        subject: Literal["query", "capability", "ref", "result", "recovery"],
        expression: str | None = None,
        ref: str | None = None,
    ) -> str:
        """Explain parser grammar, capabilities, refs, result semantics, or recovery."""

        async def run() -> str:
            if subject == "query":
                if expression is None:
                    return hooks.error_json(
                        "query explanation requires expression", code="invalid_argument", tool="explain"
                    )
                explanation = await hooks.get_polylogue().explain_query_expression(expression)
                return hooks.json_payload(MCPRootPayload(root={"subject": subject, "explanation": explanation}))
            if subject == "ref":
                if ref is None:
                    return hooks.error_json("ref explanation requires ref", code="invalid_argument", tool="explain")
                resolution = await hooks.get_polylogue().resolve_ref(_object_ref(ref))
                return hooks.json_payload(
                    MCPRootPayload(root={"subject": subject, "resolution": resolution.model_dump(mode="json")})
                )
            from polylogue.archive.query.discovery import RESULT_SEMANTICS_TEACHING, query_discovery_examples

            read_views: list[object] = []
            if subject == "capability":
                read_views = list(await hooks.get_polylogue().list_read_view_profiles())
                return hooks.json_payload(
                    MCPRootPayload(root={"subject": subject, "read_views": read_views, "total": len(read_views)})
                )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "subject": subject,
                        "result_semantics": RESULT_SEMANTICS_TEACHING,
                        "examples": query_discovery_examples(),
                        "recovery": "Use a q2 continuation alone to resume an exhaustive query page.",
                        "read_views": read_views,
                        "total": len(read_views),
                    }
                )
            )

        return await hooks.async_safe_call("explain", run)

    async def context(
        intent: str,
        query: str | None = None,
        budget_tokens: int | None = None,
        result_ref: str | None = None,
        repo_path: str | None = None,
        cwd: str | None = None,
        recent_files: tuple[str, ...] = (),
        session_id: str | None = None,
        limit: int = 5,
    ) -> str:
        """Compile a policy-gated bounded context image with receipts.

        ``intent="resume"`` builds the SessionStart preamble (session
        lineage, ranked resume candidates, project git state, and
        provenance-gated assertion guidance) instead of the default
        seed-query/seed-ref context image. ``limit`` bounds the number of
        ranked resume candidates for that intent only.
        """
        del result_ref

        async def run() -> str:
            if intent == "resume":
                return await _resume_preamble(
                    hooks,
                    session_id=session_id,
                    repo_path=repo_path,
                    cwd=cwd,
                    recent_files=recent_files,
                    related_limit=hooks.clamp_limit(limit),
                )
            payload = await hooks.get_polylogue().context_image_payload(
                query=query,
                max_tokens=budget_tokens,
                max_sessions=5,
                include_messages=True,
                include_assertions=True,
                redact_paths=True,
            )
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("context", run, session_id=session_id)

    async def status(
        scope: Literal["archive", "sources", "embeddings", "coordination", "operation"],
        include: tuple[str, ...] = (),
        ref: str | None = None,
    ) -> str:
        """Report compact archive authority and readiness status.

        ``scope="sources"`` requires ``ref=<source_path>`` and returns bounded
        freshness evidence for that one configured source (inherently
        single-source, matching the retired ``named_source_freshness`` tool).
        ``scope="embeddings"`` returns embedding catch-up/readiness status;
        ``include=("detail",)`` and/or ``include=("bands",)`` add retrieval
        detail and band statistics. ``scope="coordination"`` returns the
        multi-agent coordination envelope (``include=("detail",)`` for the
        undiscounted view). ``scope="operation"`` returns readiness plus MCP
        call-delivery outbox pressure.
        """

        async def run() -> str:
            root: dict[str, object] = {"scope": scope}
            if scope == "operation":
                from dataclasses import asdict

                from polylogue.mcp.call_log import mcp_call_outbox_status
                from polylogue.mcp.payloads import MCPReadinessReportPayload
                from polylogue.readiness import get_readiness

                report = get_readiness(hooks.get_config())
                root["operation"] = MCPReadinessReportPayload.from_report(
                    report,
                    include_counts=True,
                    include_detail=True,
                    include_cached=True,
                    mcp_call_delivery=asdict(mcp_call_outbox_status()),
                ).model_dump(mode="json", exclude_none=True)
                return hooks.json_payload(MCPRootPayload(root=root), exclude_none=True)

            if scope == "coordination":
                from polylogue.coordination import build_coordination_envelope

                if "detail" in include:
                    envelope = build_coordination_envelope(view="status", detail=True)
                else:
                    envelope = _coordination_cache().get_or_build(view="status", cwd=None, limit=10)
                root["coordination"] = envelope.model_dump(mode="json", exclude_none=True)
                return hooks.json_payload(MCPRootPayload(root=root), exclude_none=True)

            if scope == "sources":
                if ref is None:
                    return hooks.error_json(
                        "status(scope='sources') requires ref=<source_path>",
                        code="invalid_argument",
                        tool="status",
                    )
                from polylogue.archive.query.source_freshness_surfaces import make_source_freshness_mcp_handler
                from polylogue.mcp.archive_support import mcp_archive_root

                freshness_handler = make_source_freshness_mcp_handler(lambda: mcp_archive_root(hooks.get_config()))
                root["sources"] = await freshness_handler(ref)
                return hooks.json_payload(MCPRootPayload(root=root), exclude_none=True)

            if scope == "embeddings":
                from polylogue.readiness.capability import component_from_embedding_payload
                from polylogue.storage.embeddings.status_payload import embedding_status_payload

                embeddings_payload = embedding_status_payload(
                    _EmbeddingStatusEnv(hooks.get_config()),
                    include_retrieval_bands="bands" in include,
                    include_detail="detail" in include,
                )
                embeddings_result = dict(embeddings_payload)
                embedding_component = component_from_embedding_payload(embeddings_result)
                embeddings_result["component_readiness"] = {
                    embedding_component.component: embedding_component.to_dict()
                }
                root["embeddings"] = embeddings_result
                return hooks.json_payload(MCPRootPayload(root=root), exclude_none=True)

            from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest
            from polylogue.mcp.archive_support import mcp_archive_root

            transaction = QueryTransaction(
                mcp_archive_root(hooks.get_config()),
                QueryTransactionRequest(
                    operation="status", arguments={"scope": scope}, page_size=1, projection="status"
                ),
            )
            stats = await transaction.run(lambda archive: archive.stats())
            if "provider_usage" not in include:
                root["archive"] = MCPArchiveStatsPayload.from_archive_stats(
                    stats, include_embedded=False, include_db_size=False
                ).model_dump(mode="json")
            if "provider_usage" in include:
                report_usage = await hooks.get_polylogue().provider_usage_report(
                    origin=ref,
                    limit=10,
                    detail="headline",
                )
                usage = report_usage.to_dict()
                root["provider_usage"] = {
                    "model_rollup_usage": usage["model_rollup_usage"],
                    "pricing_grain": usage["pricing_grain"],
                    "exact_total_tokens_evidence": usage["exact_total_tokens_evidence"],
                    "detail_level": usage["detail_level"],
                    "caveats": usage["caveats"],
                }
            return hooks.json_payload(MCPRootPayload(root=root), exclude_none=True)

        return await hooks.async_safe_call("status", run)

    for handler in (query, read, get, explain, context, status):
        register_declared_handler(mcp, handler, name=handler.__name__)


def _field(fields: dict[str, object] | None, name: str) -> object | None:
    return None if fields is None else fields.get(name)


def _require_field(hooks: ServerCallbacks, fields: dict[str, object] | None, name: str, *, operation: str) -> str:
    value = _field(fields, name)
    if value is None:
        raise _WriteFieldError(
            hooks.error_json(
                f"write(operation={operation!r}) requires fields[{name!r}]",
                code="invalid_argument",
                tool="write",
            )
        )
    if not isinstance(value, str):
        raise _WriteFieldError(
            hooks.error_json(
                f"write(operation={operation!r}) requires fields[{name!r}] to be a string",
                code="invalid_argument",
                tool="write",
            )
        )
    return value


class _WriteFieldError(Exception):
    """Carries a pre-built error payload out of field validation."""

    def __init__(self, payload: str) -> None:
        super().__init__(payload)
        self.payload = payload


async def _dispatch_write(hooks: ServerCallbacks, *, operation: str, kwargs: dict[str, Any]) -> str:
    """Apply one declared mutation operation, delegating to its existing typed owner.

    This is a thin adapter over the same ``Polylogue`` facade methods the
    retired per-operation MCP tools called -- it does not invent a new
    mutation policy (that is polylogue-t46.9's job).
    """
    from polylogue.mcp.mutation_support import resolve_session_or_error
    from polylogue.mcp.payloads import MutationResultPayload

    poly = hooks.get_polylogue()
    fields = kwargs.get("fields")
    fields = fields if isinstance(fields, dict) else None
    session_id = kwargs.get("session_id")
    session_ids = kwargs.get("session_ids")
    tag = kwargs.get("tag")
    tags = kwargs.get("tags")
    key = kwargs.get("key")
    value = kwargs.get("value")
    confirm = bool(kwargs.get("confirm") or False)

    try:
        if operation == "add_tag":
            if session_id is None or tag is None:
                return hooks.error_json(
                    "write(operation='add_tag') requires session_id and tag", code="invalid_argument"
                )
            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            author_ref = _field(fields, "author_ref")
            author_kind = _field(fields, "author_kind")
            add_tag_result = await poly.add_tag(
                resolved,
                tag,
                author_ref=author_ref if isinstance(author_ref, str) else None,
                author_kind=author_kind if isinstance(author_kind, str) else None,
            )
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if add_tag_result.outcome == "added" else "unchanged",
                    session_id=resolved,
                    tag=tag,
                    detail=add_tag_result.detail,
                    outcome=add_tag_result.outcome,
                ),
                exclude_none=True,
            )

        if operation == "remove_tag":
            if session_id is None or tag is None:
                return hooks.error_json(
                    "write(operation='remove_tag') requires session_id and tag", code="invalid_argument"
                )
            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            remove_tag_result = await poly.remove_tag(resolved, tag)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if remove_tag_result.outcome == "removed" else "not_found",
                    session_id=resolved,
                    tag=tag,
                    detail=remove_tag_result.detail,
                    outcome=remove_tag_result.outcome,
                ),
                exclude_none=True,
            )

        if operation == "bulk_tag_sessions":
            if not session_ids or not tags:
                return hooks.error_json(
                    "write(operation='bulk_tag_sessions') requires session_ids and tags", code="invalid_argument"
                )
            try:
                bulk_result = await poly.bulk_tag_sessions(list(session_ids), list(tags))
            except ValueError as exc:
                return hooks.error_json(str(exc))
            return hooks.json_payload(
                MutationResultPayload(
                    status=bulk_result.outcome,
                    session_count=bulk_result.session_count,
                    tag_count=bulk_result.tag_count,
                    affected_count=bulk_result.affected_count,
                    skipped_count=bulk_result.skipped_count,
                ),
                exclude_none=True,
            )

        if operation == "set_metadata":
            if session_id is None or key is None or value is None:
                return hooks.error_json(
                    "write(operation='set_metadata') requires session_id, key, and value", code="invalid_argument"
                )
            from polylogue.api.archive import SessionNotFoundError
            from polylogue.surfaces.payloads import MetadataKeyValidationError, validate_metadata_key

            key_error = validate_metadata_key(key)
            if key_error is not None:
                return hooks.error_json(key_error, session_id=session_id, code="invalid_key")
            import contextlib
            import json as _json

            parsed_value: object = value
            with contextlib.suppress(ValueError, TypeError):
                parsed_value = _json.loads(value)
            try:
                set_metadata_result = await poly.set_metadata(session_id, key, str(parsed_value))
            except MetadataKeyValidationError as exc:
                return hooks.error_json(str(exc), session_id=session_id, code="invalid_key")
            except SessionNotFoundError:
                return hooks.error_json("Session not found", code="not_found", session_id=session_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if set_metadata_result.outcome == "set" else "unchanged",
                    session_id=set_metadata_result.session_id,
                    key=set_metadata_result.key,
                    detail=set_metadata_result.detail,
                ),
                exclude_none=True,
            )

        if operation == "delete_metadata":
            if session_id is None or key is None:
                return hooks.error_json(
                    "write(operation='delete_metadata') requires session_id and key", code="invalid_argument"
                )
            from polylogue.api.archive import SessionNotFoundError
            from polylogue.surfaces.payloads import MetadataKeyValidationError, validate_metadata_key

            key_error = validate_metadata_key(key)
            if key_error is not None:
                return hooks.error_json(key_error, session_id=session_id, code="invalid_key")
            try:
                delete_metadata_result = await poly.delete_metadata(session_id, key)
            except MetadataKeyValidationError as exc:
                return hooks.error_json(str(exc), session_id=session_id, code="invalid_key")
            except SessionNotFoundError:
                return hooks.error_json("Session not found", code="not_found", session_id=session_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if delete_metadata_result.outcome == "deleted" else "not_found",
                    session_id=delete_metadata_result.session_id,
                    key=delete_metadata_result.key,
                    detail=delete_metadata_result.detail,
                ),
                exclude_none=True,
            )

        if operation == "delete_session":
            if session_id is None:
                return hooks.error_json(
                    "write(operation='delete_session') requires session_id", code="invalid_argument"
                )
            if not confirm:
                return hooks.error_json("Safety guard: set confirm=true to delete", session_id=session_id)
            delete_session_result = await poly.delete_session_safe(session_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if delete_session_result.outcome == "deleted" else "not_found",
                    session_id=delete_session_result.session_id,
                    detail=delete_session_result.detail,
                ),
                exclude_none=True,
            )

        if operation in ("add_mark", "remove_mark"):
            from polylogue.core.user_state_targets import MARK_TYPE_NAMES, TARGET_SESSION, is_mark_type_supported

            if session_id is None:
                return hooks.error_json(f"write(operation={operation!r}) requires session_id", code="invalid_argument")
            mark_type = _require_field(hooks, fields, "mark_type", operation=operation)
            if not is_mark_type_supported(mark_type):
                return hooks.error_json(f"mark_type must be one of: {', '.join(MARK_TYPE_NAMES)}", detail=mark_type)
            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            target_type = _field(fields, "target_type")
            target_type = target_type if isinstance(target_type, str) else TARGET_SESSION
            target_id = _field(fields, "target_id")
            message_id = _field(fields, "message_id")
            if operation == "add_mark":
                created = await poly.add_mark(
                    resolved,
                    mark_type,
                    target_type=target_type,
                    target_id=target_id if isinstance(target_id, str) else None,
                    message_id=message_id if isinstance(message_id, str) else None,
                )
                return hooks.json_payload(
                    MutationResultPayload(
                        status="ok" if created else "unchanged",
                        session_id=resolved,
                        detail=None if created else "already_present",
                        key=mark_type,
                        outcome="added" if created else "no_op",
                    ),
                    exclude_none=True,
                )
            deleted = await poly.remove_mark(
                resolved,
                mark_type,
                target_type=target_type,
                target_id=target_id if isinstance(target_id, str) else None,
                message_id=message_id if isinstance(message_id, str) else None,
            )
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if deleted else "not_found",
                    session_id=resolved,
                    detail=None if deleted else "mark_not_present",
                    key=mark_type,
                    outcome="removed" if deleted else "not_present",
                ),
                exclude_none=True,
            )

        if operation == "save_annotation":
            from polylogue.core.user_state_targets import TARGET_SESSION

            if session_id is None:
                return hooks.error_json(
                    "write(operation='save_annotation') requires session_id", code="invalid_argument"
                )
            annotation_id = _require_field(hooks, fields, "annotation_id", operation=operation)
            note_text = _require_field(hooks, fields, "note_text", operation=operation)
            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            target_type = _field(fields, "target_type")
            target_id = _field(fields, "target_id")
            message_id = _field(fields, "message_id")
            created = await poly.save_annotation(
                annotation_id,
                resolved,
                note_text,
                target_type=target_type if isinstance(target_type, str) else TARGET_SESSION,
                target_id=target_id if isinstance(target_id, str) else None,
                message_id=message_id if isinstance(message_id, str) else None,
            )
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok", session_id=resolved, key=annotation_id, outcome="added" if created else "updated"
                ),
                exclude_none=True,
            )

        if operation == "delete_annotation":
            annotation_id = _require_field(hooks, fields, "annotation_id", operation=operation)
            deleted = await poly.delete_annotation(annotation_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "annotation_not_found",
                    key=annotation_id,
                ),
                exclude_none=True,
            )

        if operation == "capture_assertion_candidate":
            from pathlib import Path

            from polylogue.api.archive import candidate_capture_kind

            body_text = _require_field(hooks, fields, "body_text", operation=operation)
            author_ref = _require_field(hooks, fields, "author_ref", operation=operation)
            if not author_ref.startswith("agent:") or author_ref == "agent:":
                return hooks.error_json("author_ref must be an agent:<session> ref", code="invalid_candidate_capture")
            kind = _field(fields, "kind")
            refs = _field(fields, "refs")
            scope_refs = _field(fields, "scope_refs")
            cwd = _field(fields, "cwd")
            candidate_result = await poly.capture_assertion_candidate(
                body_text=body_text,
                kind=candidate_capture_kind(kind if isinstance(kind, str) else "note"),
                refs=tuple(refs) if isinstance(refs, list) else (),
                scope_refs=tuple(scope_refs) if isinstance(scope_refs, list) else (),
                cwd=Path(cwd) if isinstance(cwd, str) else None,
                author_ref=author_ref,
                author_kind="agent",
            )
            return hooks.json_payload(candidate_result, exclude_none=False)

        if operation == "blackboard_post":
            from polylogue.mcp.archive_support import blackboard_note_payload

            kind = _require_field(hooks, fields, "kind", operation=operation)
            title = _require_field(hooks, fields, "title", operation=operation)
            content = _require_field(hooks, fields, "content", operation=operation)
            scope_repo = _field(fields, "scope_repo")
            scope_issue = _field(fields, "scope_issue")
            scope_path = _field(fields, "scope_path")
            related_sessions = _field(fields, "related_sessions")
            author_ref = _field(fields, "author_ref")
            author_kind = _field(fields, "author_kind")
            evidence_refs = _field(fields, "evidence_refs")
            staleness = _field(fields, "staleness")
            context_policy = _field(fields, "context_policy")
            try:
                note = await poly.post_blackboard_note(
                    kind=kind,
                    title=title,
                    content=content,
                    scope_repo=scope_repo if isinstance(scope_repo, str) else None,
                    scope_session=session_id,
                    scope_issue=scope_issue if isinstance(scope_issue, int) else None,
                    scope_path=scope_path if isinstance(scope_path, str) else None,
                    related_sessions=tuple(related_sessions) if isinstance(related_sessions, list) else (),
                    author_ref=author_ref if isinstance(author_ref, str) else None,
                    author_kind=author_kind if isinstance(author_kind, str) else "agent",
                    evidence_refs=tuple(evidence_refs) if isinstance(evidence_refs, list) else (),
                    staleness=staleness if isinstance(staleness, dict) else None,
                    context_policy=context_policy if isinstance(context_policy, dict) else None,
                )
            except ValueError as exc:
                return hooks.error_json(str(exc))
            return hooks.json_payload(blackboard_note_payload(note))

        if operation == "import_annotation_batch":
            from polylogue.annotations.importer import (
                AnnotationBatchImportError,
                AnnotationBatchImportRequest,
            )
            from polylogue.annotations.importer import import_annotation_batch as run_annotation_batch_import

            required = (
                "jsonl",
                "batch_id",
                "schema_id",
                "schema_version",
                "target_ref",
                "source_result_ref",
                "actor_ref",
                "model_ref",
                "prompt_ref",
            )
            values = {name: _field(fields, name) for name in required}
            missing = [name for name in required if values[name] is None]
            if missing:
                return hooks.error_json(
                    f"write(operation='import_annotation_batch') requires fields{sorted(missing)}",
                    code="invalid_argument",
                )
            schema_version_value = values["schema_version"]
            if not isinstance(schema_version_value, int):
                return hooks.error_json(
                    "write(operation='import_annotation_batch') requires fields['schema_version'] to be an integer",
                    code="invalid_argument",
                )
            metadata = _field(fields, "metadata")
            try:
                request = AnnotationBatchImportRequest(
                    jsonl=str(values["jsonl"]),
                    batch_id=str(values["batch_id"]),
                    schema_id=str(values["schema_id"]),
                    schema_version=schema_version_value,
                    target_ref=str(values["target_ref"]),
                    source_result_ref=str(values["source_result_ref"]),
                    actor_ref=str(values["actor_ref"]),
                    model_ref=str(values["model_ref"]),
                    prompt_ref=str(values["prompt_ref"]),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
                import_result = await run_annotation_batch_import(poly, request)
            except (AnnotationBatchImportError, ValueError) as exc:
                return hooks.error_json(str(exc), code="invalid_annotation_batch")
            return hooks.json_payload(import_result)

        if operation == "save_saved_view":
            import json as _json

            from polylogue.archive.query.spec import SessionQuerySpec

            name = _require_field(hooks, fields, "name", operation=operation)
            query_json = _require_field(hooks, fields, "query_json", operation=operation)
            view_id = _field(fields, "view_id")
            if not name.strip():
                return hooks.error_json("saved view name must not be empty")
            try:
                query = _json.loads(query_json)
            except _json.JSONDecodeError:
                return hooks.error_json("query_json must be valid JSON")
            if not isinstance(query, dict):
                return hooks.error_json("query_json must encode an object")
            try:
                SessionQuerySpec.from_params(query, strict=True)
            except Exception as exc:
                return hooks.error_json(
                    "query_json is not a valid SessionQuerySpec", detail=f"{type(exc).__name__}: {exc}"
                )
            canonical_query_json = _json.dumps(query, sort_keys=True, separators=(",", ":"))
            from hashlib import sha256

            if isinstance(view_id, str) and view_id:
                saved_id = view_id
            else:
                digest_input = f"{name.strip()}\0{canonical_query_json}"
                saved_id = f"saved-view-{sha256(digest_input.encode()).hexdigest()[:16]}"
            created = await poly.save_view(saved_id, name.strip(), canonical_query_json)
            return hooks.json_payload(
                MutationResultPayload(status="ok", key=saved_id, outcome="added" if created else "updated"),
                exclude_none=True,
            )

        if operation == "delete_saved_view":
            view_id = _require_field(hooks, fields, "view_id", operation=operation)
            deleted = await poly.delete_view(view_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "saved_view_not_found",
                    key=view_id,
                ),
                exclude_none=True,
            )

        if operation == "save_recall_pack":
            import json as _json

            pack_id = _require_field(hooks, fields, "pack_id", operation=operation)
            label = _require_field(hooks, fields, "label", operation=operation)
            payload_json = _field(fields, "payload_json")
            payload_json = payload_json if isinstance(payload_json, str) else "{}"
            if not pack_id.strip() or not label.strip():
                return hooks.error_json("pack_id and label must not be empty")
            try:
                payload = _json.loads(payload_json)
            except _json.JSONDecodeError:
                return hooks.error_json("payload_json must be valid JSON")
            if not isinstance(payload, dict):
                return hooks.error_json("payload_json must encode an object")
            items = payload.get("items")
            if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
                return hooks.error_json("payload_json must include an items list of objects")
            created = await poly.create_recall_pack(
                pack_id.strip(), label.strip(), _json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            return hooks.json_payload(
                MutationResultPayload(status="ok", key=pack_id.strip(), outcome="added" if created else "updated"),
                exclude_none=True,
            )

        if operation == "delete_recall_pack":
            pack_id = _require_field(hooks, fields, "pack_id", operation=operation)
            deleted = await poly.delete_recall_pack(pack_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "recall_pack_not_found",
                    key=pack_id,
                ),
                exclude_none=True,
            )

        if operation == "save_workspace":
            import json as _json

            workspace_id = _require_field(hooks, fields, "workspace_id", operation=operation)
            name = _require_field(hooks, fields, "name", operation=operation)
            mode = _field(fields, "mode")
            mode = mode if isinstance(mode, str) else "tabs"
            if mode not in {"tabs", "stack", "compare", "timeline"}:
                return hooks.error_json("mode must be one of: tabs, stack, compare, timeline")
            if not workspace_id.strip() or not name.strip():
                return hooks.error_json("workspace_id and name must not be empty")
            open_targets_json = _field(fields, "open_targets_json")
            layout_json = _field(fields, "layout_json")
            active_target_json = _field(fields, "active_target_json")
            try:
                open_targets = _json.loads(open_targets_json) if isinstance(open_targets_json, str) else []
                layout = _json.loads(layout_json) if isinstance(layout_json, str) else {}
                active_target = _json.loads(active_target_json) if isinstance(active_target_json, str) else {}
            except _json.JSONDecodeError:
                return hooks.error_json("open_targets_json/layout_json/active_target_json must be valid JSON")
            if not isinstance(open_targets, list) or not all(isinstance(item, dict) for item in open_targets):
                return hooks.error_json("open_targets_json must encode a list of objects")
            if not isinstance(layout, dict) or not isinstance(active_target, dict):
                return hooks.error_json("layout_json/active_target_json must encode objects")
            created = await poly.save_workspace(
                workspace_id.strip(),
                name.strip(),
                mode,
                _json.dumps(open_targets, sort_keys=True, separators=(",", ":")),
                _json.dumps(layout, sort_keys=True, separators=(",", ":")),
                _json.dumps(active_target, sort_keys=True, separators=(",", ":")),
            )
            return hooks.json_payload(
                MutationResultPayload(status="ok", key=workspace_id.strip(), outcome="added" if created else "updated"),
                exclude_none=True,
            )

        if operation == "delete_workspace":
            workspace_id = _require_field(hooks, fields, "workspace_id", operation=operation)
            deleted = await poly.delete_workspace(workspace_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "workspace_not_found",
                    key=workspace_id,
                ),
                exclude_none=True,
            )

        if operation == "record_correction":
            from polylogue.insights.feedback import UnknownCorrectionKindError

            if session_id is None:
                return hooks.error_json(
                    "write(operation='record_correction') requires session_id", code="invalid_argument"
                )
            kind = _require_field(hooks, fields, "kind", operation=operation)
            payload = _field(fields, "payload")
            if not isinstance(payload, dict):
                return hooks.error_json(
                    "write(operation='record_correction') requires fields['payload'] to be an object"
                )
            correction_note = _field(fields, "note")
            author_ref = _field(fields, "author_ref")
            author_kind = _field(fields, "author_kind")
            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            try:
                correction = await poly.record_correction(
                    resolved,
                    kind,
                    payload,
                    note=correction_note if isinstance(correction_note, str) else None,
                    author_ref=author_ref if isinstance(author_ref, str) else None,
                    author_kind=author_kind if isinstance(author_kind, str) else None,
                )
            except UnknownCorrectionKindError as exc:
                return hooks.error_json(str(exc), code="unknown_kind", kind=str(kind or ""))
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok", session_id=correction.session_id, outcome=correction.kind.value, detail=correction.note
                ),
                exclude_none=True,
            )

        if operation == "clear_corrections":
            from polylogue.insights.feedback import UnknownCorrectionKindError

            if session_id is None:
                return hooks.error_json(
                    "write(operation='clear_corrections') requires session_id", code="invalid_argument"
                )
            kind = _field(fields, "kind")
            kind = kind if isinstance(kind, str) else None
            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            try:
                if kind is None:
                    count = await poly.clear_corrections(resolved)
                    return hooks.json_payload(
                        MutationResultPayload(
                            status="ok", session_id=resolved, affected_count=count, outcome="cleared"
                        ),
                        exclude_none=True,
                    )
                removed = await poly.delete_correction(resolved, kind)
            except UnknownCorrectionKindError as exc:
                return hooks.error_json(str(exc), code="unknown_kind", kind=str(kind or ""))
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if removed else "not_found",
                    session_id=resolved,
                    outcome="deleted" if removed else "not_found",
                ),
                exclude_none=True,
            )

        return hooks.error_json(f"unknown write operation: {operation!r}", code="invalid_argument")
    except _WriteFieldError as exc:
        return exc.payload


async def _dispatch_run(hooks: ServerCallbacks, *, ref: str, limit: int | None) -> str:
    """Execute a saved-query ref through the same session-search machinery as ``query(projection="sessions")``."""
    import json as _json

    if not ref.startswith("saved-query:") and not ref.startswith("saved-view:"):
        return hooks.error_json(
            f"run() only supports saved-query/saved-view refs, got {ref!r}", code="invalid_argument", tool="run"
        )
    view_id = ref.split(":", 1)[1]
    poly = hooks.get_polylogue()
    rows = await poly.list_views()
    row = next((r for r in rows if r.get("view_id") == view_id), None)
    if row is None:
        return hooks.error_json(f"saved view not found: {view_id}", code="not_found", tool="run")
    try:
        query = _json.loads(row["query_json"])
    except (_json.JSONDecodeError, TypeError):
        query = {}
    if not isinstance(query, dict):
        query = {}
    return await _query_sessions(
        hooks,
        expression=query.get("query"),
        limit=limit,
        origin=query.get("origin"),
        tag=query.get("tag"),
        repo=query.get("repo"),
        since=query.get("since"),
        until=query.get("until"),
        sort=query.get("sort"),
        min_messages=query.get("min_messages"),
        max_messages=query.get("max_messages"),
        min_words=query.get("min_words"),
    )


def _build_mcp_scope_filter(
    *,
    session_ids: list[str] | None,
    origin: str | None,
    source_family: str | None,
    source_root: str | None,
    since: str | None,
    until: str | None,
    failure_kind: str | None,
    parser_version: str | None,
) -> MaintenanceScopeFilter:
    """Translate maintenance() args into a :class:`MaintenanceScopeFilter`.

    Shared by the preview and execute operations so they never drift on how
    scope filters are parsed.
    """
    from datetime import datetime
    from pathlib import Path

    from polylogue.core.enums import Origin
    from polylogue.maintenance.scope import MaintenanceScopeFilter

    time_range: tuple[datetime, datetime] | None
    if since is not None or until is not None:
        if since is None or until is None:
            raise ValueError("since and until must be supplied together")
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00") if since.endswith("Z") else since)
        until_dt = datetime.fromisoformat(until.replace("Z", "+00:00") if until.endswith("Z") else until)
        time_range = (since_dt, until_dt)
    else:
        time_range = None

    return MaintenanceScopeFilter(
        session_ids=tuple(session_ids) if session_ids else None,
        origin=Origin(origin).value if origin is not None else None,
        source_family=source_family,
        source_root=Path(source_root) if source_root else None,
        time_range=time_range,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )


async def _dispatch_maintenance(hooks: ServerCallbacks, *, operation: str, kwargs: dict[str, Any]) -> str:
    """Preview/execute/inspect maintenance operations, delegating to the existing planner/registry."""
    from polylogue.config import Config
    from polylogue.maintenance.envelope import envelope_from_operation
    from polylogue.paths import archive_root, render_root

    config = Config(archive_root=archive_root(), render_root=render_root(), sources=[])

    if operation in ("preview", "execute"):
        from polylogue.maintenance.planner import execute_backfill, preview_backfill

        targets = kwargs.get("targets")
        session_ids = kwargs.get("session_ids")
        try:
            scope_filter = _build_mcp_scope_filter(
                session_ids=list(session_ids) if session_ids else None,
                origin=kwargs.get("origin"),
                source_family=kwargs.get("source_family"),
                source_root=kwargs.get("source_root"),
                since=kwargs.get("since"),
                until=kwargs.get("until"),
                failure_kind=kwargs.get("failure_kind"),
                parser_version=kwargs.get("parser_version"),
            )
        except ValueError as exc:
            return hooks.error_json(str(exc), code="invalid_argument")
        resolved_targets = tuple(targets) if targets else ()
        if operation == "preview":
            result = preview_backfill(config, targets=resolved_targets, scope_filter=scope_filter)
            envelope = envelope_from_operation(result, origin="mcp", mode="preview")
        else:
            dry_run = bool(kwargs.get("dry_run") or False)
            result = execute_backfill(config, targets=resolved_targets, dry_run=dry_run, scope_filter=scope_filter)
            envelope = envelope_from_operation(result, origin="mcp", mode="execute")
        return hooks.json_payload(envelope)

    if operation == "status":
        from polylogue.maintenance.registry import MaintenanceOperationRegistry

        operation_id = kwargs.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            return hooks.error_json("maintenance(operation='status') requires operation_id", code="invalid_argument")
        registry = MaintenanceOperationRegistry(config=config)
        record = registry.get_operation(operation_id)
        if record is None:
            return hooks.error_json(f"Operation not found: {operation_id}", code="not_found")
        envelope = envelope_from_operation(record.operation, origin="mcp", mode="execute")
        return hooks.json_payload(
            MCPRootPayload(
                root={
                    "envelope": envelope.to_dict(),
                    "updated_at": record.updated_at,
                    "state_path": str(record.state_path),
                }
            )
        )

    if operation == "list":
        from polylogue.maintenance.registry import MaintenanceOperationRegistry

        registry = MaintenanceOperationRegistry(config=config)
        records = registry.list_operations()
        items = [
            {
                "envelope": envelope_from_operation(r.operation, origin="mcp", mode="execute").to_dict(),
                "updated_at": r.updated_at,
                "state_path": str(r.state_path),
            }
            for r in records
        ]
        return hooks.json_payload(MCPRootPayload(root={"items": items, "total": len(items)}))

    if operation == "rebuild_index":
        from polylogue.mcp.payloads import MCPMutationStatusPayload

        poly = hooks.get_polylogue()
        success = await poly.rebuild_index()
        status_info = await poly.get_index_status()
        return hooks.json_payload(
            MCPMutationStatusPayload(
                status="ok" if success else "failed",
                index_exists=bool(status_info.get("exists", False)),
                indexed_messages=int(status_info.get("count", 0)),
            ),
            exclude_none=True,
        )

    if operation == "update_index":
        from polylogue.mcp.payloads import MCPMutationStatusPayload

        session_ids = kwargs.get("session_ids")
        if not session_ids:
            return hooks.error_json(
                "maintenance(operation='update_index') requires session_ids", code="invalid_argument"
            )
        success = await hooks.get_polylogue().update_index(list(session_ids))
        return hooks.json_payload(
            MCPMutationStatusPayload(status="ok" if success else "failed", session_count=len(session_ids)),
            exclude_none=True,
        )

    if operation == "rebuild_insights":
        session_ids = kwargs.get("session_ids")
        counts = await hooks.get_polylogue().rebuild_insights(session_ids=list(session_ids) if session_ids else None)
        return hooks.json_payload(
            MCPRootPayload(
                root={
                    "status": "ok",
                    "session_count": len(session_ids) if session_ids else None,
                    "counts": counts.to_dict(),
                    "total": counts.total(),
                }
            ),
            exclude_none=True,
        )

    return hooks.error_json(f"unknown maintenance operation: {operation!r}", code="invalid_argument")


def register_cutover_privileged_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    """Register write/judge/run/maintenance for roles whose ladder includes them.

    Thin adapters over the same typed owners the retired per-operation MCP
    tools used (write, run, maintenance) or already used (judge). No new
    mutation policy is invented here -- that is polylogue-t46.9's job.
    """
    role = mcp.role  # type: ignore[attr-defined]

    if mcp_role_allows(role, "write"):

        async def write(
            operation: Literal[
                "add_tag",
                "remove_tag",
                "bulk_tag_sessions",
                "set_metadata",
                "delete_metadata",
                "delete_session",
                "add_mark",
                "remove_mark",
                "save_annotation",
                "delete_annotation",
                "capture_assertion_candidate",
                "blackboard_post",
                "import_annotation_batch",
                "save_saved_view",
                "delete_saved_view",
                "save_recall_pack",
                "delete_recall_pack",
                "save_workspace",
                "delete_workspace",
                "record_correction",
                "clear_corrections",
            ],
            session_id: str | None = None,
            session_ids: list[str] | None = None,
            tag: str | None = None,
            tags: list[str] | None = None,
            key: str | None = None,
            value: str | None = None,
            confirm: bool = False,
            fields: dict[str, object] | None = None,
        ) -> str:
            """Apply a declared mutation operation after shared authorization.

            ``session_id``/``session_ids``/``tag``/``tags``/``key``/``value``/
            ``confirm`` cover the common cases; ``fields`` carries every
            operation-specific value beyond those (see each operation's
            retired single-purpose tool for the exact field names, e.g.
            ``fields={"mark_type": "star"}`` for ``add_mark``).
            """

            async def run() -> str:
                return await _dispatch_write(
                    hooks,
                    operation=operation,
                    kwargs={
                        "session_id": session_id,
                        "session_ids": session_ids,
                        "tag": tag,
                        "tags": tags,
                        "key": key,
                        "value": value,
                        "confirm": confirm,
                        "fields": fields,
                    },
                )

            return await hooks.async_safe_call(
                "write", run, session_id=session_id, session_ids=tuple(session_ids or ())
            )

        register_declared_handler(mcp, write, name="write")

        async def run(
            ref: str,
            limit: int | None = None,
        ) -> str:
            """Execute a saved query or governed recipe ref."""

            async def _run() -> str:
                return await _dispatch_run(hooks, ref=ref, limit=limit)

            return await hooks.async_safe_call("run", _run)

        register_declared_handler(mcp, run, name="run")

    if mcp_role_allows(role, "review"):

        async def judge(
            items: list[dict[str, object]] | None = None,
            candidate_ref: str | None = None,
            decision: Literal["accept", "reject", "defer", "supersede"] | None = None,
            reason: str | None = None,
            inject: bool = False,
            actor_ref: str = "user:local",
            replacement_kind: str | None = None,
            replacement_body_text: str | None = None,
            replacement_value: object | None = None,
        ) -> str:
            """Accept, reject, defer, or supersede assertion candidates.

            Pass ``items`` for bulk judgment (independently reported partial
            success), or ``candidate_ref``+``decision`` for a single one.
            """

            async def _judge() -> str:
                from polylogue.storage.sqlite.archive_tiers.user_write import (
                    ArchiveAssertionBulkJudgmentItemEnvelope,
                )

                if items is not None:

                    def make_item(item: dict[str, object]) -> ArchiveAssertionBulkJudgmentItemEnvelope:
                        item_candidate_ref = item.get("candidate_ref")
                        item_decision = item.get("decision")
                        if not isinstance(item_candidate_ref, str) or not isinstance(item_decision, str):
                            raise ValueError("each judgment requires string candidate_ref and decision")
                        item_inject = item.get("inject", False)
                        if type(item_inject) is not bool:
                            raise ValueError("each judgment requires boolean inject")
                        item_reason = item.get("reason")
                        item_replacement_kind = item.get("replacement_kind")
                        item_replacement_body_text = item.get("replacement_body_text")
                        return ArchiveAssertionBulkJudgmentItemEnvelope(
                            candidate_ref=item_candidate_ref,
                            decision=item_decision,
                            reason=item_reason if isinstance(item_reason, str) else None,
                            inject=item_inject,
                            actor_ref=actor_ref,
                            replacement_kind=item_replacement_kind if isinstance(item_replacement_kind, str) else None,
                            replacement_body_text=item_replacement_body_text
                            if isinstance(item_replacement_body_text, str)
                            else None,
                            replacement_value=item.get("replacement_value"),
                        )

                    try:
                        judgments = tuple(make_item(item) for item in items)
                    except ValueError as exc:
                        return hooks.error_json(str(exc), code="invalid_assertion_judgment")
                elif candidate_ref is not None and decision is not None:
                    judgments = (
                        ArchiveAssertionBulkJudgmentItemEnvelope(
                            candidate_ref=candidate_ref,
                            decision=decision,
                            reason=reason,
                            inject=inject,
                            actor_ref=actor_ref,
                            replacement_kind=replacement_kind,
                            replacement_body_text=replacement_body_text,
                            replacement_value=replacement_value,
                        ),
                    )
                else:
                    return hooks.error_json(
                        "judge() requires items, or candidate_ref and decision", code="invalid_argument"
                    )
                payload = await hooks.get_polylogue().judge_assertion_candidates(items=judgments)
                return hooks.json_payload(payload, exclude_none=True)

            return await hooks.async_safe_call("judge", _judge)

        register_declared_handler(mcp, judge, name="judge")

    if mcp_role_allows(role, "admin"):

        async def maintenance(
            operation: Literal[
                "preview",
                "execute",
                "status",
                "list",
                "rebuild_index",
                "update_index",
                "rebuild_insights",
            ],
            targets: list[str] | None = None,
            dry_run: bool = False,
            session_ids: list[str] | None = None,
            origin: str | None = None,
            source_family: str | None = None,
            source_root: str | None = None,
            since: str | None = None,
            until: str | None = None,
            failure_kind: str | None = None,
            parser_version: str | None = None,
            operation_id: str | None = None,
        ) -> str:
            """Preview, execute, list, and inspect maintenance operations."""

            async def run() -> str:
                return await _dispatch_maintenance(
                    hooks,
                    operation=operation,
                    kwargs={
                        "targets": targets,
                        "dry_run": dry_run,
                        "session_ids": session_ids,
                        "origin": origin,
                        "source_family": source_family,
                        "source_root": source_root,
                        "since": since,
                        "until": until,
                        "failure_kind": failure_kind,
                        "parser_version": parser_version,
                        "operation_id": operation_id,
                    },
                )

            return await hooks.async_safe_call("maintenance", run, session_ids=tuple(session_ids or ()))

        register_declared_handler(mcp, maintenance, name="maintenance")


__all__ = ["register_cutover_privileged_tools", "register_cutover_read_tools"]
