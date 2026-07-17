"""Registry-driven MCP archive-insight tool registration.

Iterates INSIGHT_REGISTRY and registers a ``list_<name>`` MCP tool for
each insight type. Special one-off tools for single-item lookups and
derived distributions are registered directly.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from polylogue.insights.archive import (
    SessionLatencyProfileInsightQuery,
)
from polylogue.insights.registry import (
    INSIGHT_REGISTRY,
    InsightType,
    fetch_insights_async,
    insight_items_payload,
)
from polylogue.mcp.insight_tool_contracts import InsightListToolSpec
from polylogue.mcp.payloads import MCPRootPayload
from polylogue.mcp.query_contracts import MCPSessionQueryRequest

if TYPE_CHECKING:
    from polylogue.mcp.declarations.adapter import ToolRegistrar
    from polylogue.mcp.server_support import ServerCallbacks


_COMPLETE_BY_DEFAULT_INSIGHT_TOOLS = frozenset(
    {
        "cost_rollups",
        "session_costs",
        "tool_usage",
    }
)


def _object_int(value: object) -> int:
    if value is None:
        return 0
    return int(str(value))


def _register_list_tool(
    mcp: ToolRegistrar,
    hooks: ServerCallbacks,
    pt: InsightType,
) -> None:
    """Register one list-style MCP tool for an insight type."""
    spec = InsightListToolSpec.from_insight_type(pt)

    async def tool_fn(**kwargs: object) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            explicit_limit = "limit" in kwargs and kwargs["limit"] is not None
            normalized_kwargs = spec.normalize_kwargs(hooks.clamp_limit, kwargs)
            requested_limit = normalized_kwargs.get("limit") if explicit_limit else None
            requested_offset = _object_int(normalized_kwargs.get("offset"))
            if pt.name in _COMPLETE_BY_DEFAULT_INSIGHT_TOOLS:
                normalized_kwargs["limit"] = None
                normalized_kwargs["offset"] = 0
            insights = await fetch_insights_async(pt, poly, **normalized_kwargs)
            total = len(insights)
            page = insights
            if pt.name in _COMPLETE_BY_DEFAULT_INSIGHT_TOOLS:
                if requested_offset:
                    page = page[requested_offset:]
                if requested_limit is not None:
                    page = page[: _object_int(requested_limit)]
            payload = insight_items_payload(page, pt, item_key="items")
            if pt.name in _COMPLETE_BY_DEFAULT_INSIGHT_TOOLS:
                payload["total"] = total
                payload["returned"] = len(page)
                payload["limit"] = requested_limit
                payload["offset"] = requested_offset
                payload["truncated"] = requested_offset + len(page) < total
            return hooks.json_payload(MCPRootPayload(root=payload))

        return await hooks.async_safe_call(pt.name, run)

    async def wrapper(**kw: object) -> str:
        return await tool_fn(**kw)

    wrapper.__name__ = spec.name
    wrapper.__qualname__ = spec.name
    wrapper.__doc__ = spec.doc

    wrapper.__annotations__ = spec.signature.annotations
    wrapper.__kwdefaults__ = spec.signature.kwdefaults
    cast(Any, wrapper).__signature__ = inspect.Signature(
        parameters=spec.signature.parameters,
        return_annotation=str,
    )

    mcp.tool()(wrapper)


def register_insight_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    """Register all insight-type list tools plus special tools."""

    # Register generic list tools from the registry
    for pt in INSIGHT_REGISTRY.values():
        if pt.name == "archive_debt":
            continue
        if pt.query_model is not None and pt.operations_method_name:
            _register_list_tool(mcp, hooks, pt)

    # --- Special tools ---

    @mcp.tool()
    async def tool_call_latency_distribution(
        since: str | None = None,
        until: str | None = None,
        origin: str | None = None,
        tool_category: str | None = None,
        limit: int = 500,
    ) -> str:
        """Distribution of materialized per-session tool-call latency."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            result = await poly.tool_call_latency_distribution(
                since=since,
                until=until,
                origin=origin,
                tool_category=tool_category,
                limit=hooks.clamp_limit(limit),
            )
            return hooks.json_payload(MCPRootPayload(root=result))

        return await hooks.async_safe_call("tool_call_latency_distribution", run)

    @mcp.tool()
    async def session_latency_profile(session_id: str) -> str:
        """Get per-session latency profile by session ID."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            insight = await poly.get_session_latency_profile_insight(session_id)
            if insight is None:
                return hooks.error_json(
                    "Session not found",
                    code="not_found",
                    session_id=session_id,
                )
            return hooks.json_payload(
                MCPRootPayload(root=cast(dict[str, object], insight.model_dump(mode="json"))),
                exclude_none=True,
            )

        return await hooks.async_safe_call("session_latency_profile", run, session_id=session_id)

    @mcp.tool()
    async def find_stuck_sessions(since: str | None = None, limit: int = 20) -> str:
        """Find sessions with origin tool calls bounded as stuck."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            insights = await poly.find_stuck_session_latency_profile_insights(
                SessionLatencyProfileInsightQuery(
                    since=since,
                    limit=hooks.clamp_limit(limit),
                    only_stuck=True,
                )
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "total": len(insights),
                        "items": [insight.model_dump(mode="json") for insight in insights],
                    }
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("find_stuck_sessions", run)

    @mcp.tool()
    async def workflow_shape_distribution(
        since: str | None = None,
        until: str | None = None,
        group_by: str = "week",
        origin: str | None = None,
    ) -> str:
        """Histogram session workflow shapes by week, origin, or project."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            try:
                result = await poly.workflow_shape_distribution(
                    group_by=group_by,
                    since=since,
                    until=until,
                    origin=origin,
                )
            except ValueError:
                return hooks.error_json(
                    "Invalid group_by.",
                    code="invalid_argument",
                    detail="group_by must be one of week, origin, project",
                    tool="workflow_shape_distribution",
                )
            return hooks.json_payload(MCPRootPayload(root=result))

        return await hooks.async_safe_call("workflow_shape_distribution", run)

    @mcp.tool()
    async def find_abandoned_sessions(
        since: str | None = None,
        repo_path: str | None = None,
        min_severity: str = "question_left",
        limit: int = 20,
    ) -> str:
        """Find sessions whose terminal state indicates dangling work."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            try:
                result = await poly.find_abandoned_sessions(
                    since=since,
                    repo_path=repo_path,
                    min_severity=min_severity,
                    limit=hooks.clamp_limit(limit),
                )
            except ValueError:
                return hooks.error_json(
                    "Invalid min_severity.",
                    code="invalid_argument",
                    detail="min_severity must be one of question_left, error_left, tool_left, agent_hanging",
                    tool="find_abandoned_sessions",
                )
            return hooks.json_payload(MCPRootPayload(root=result))

        return await hooks.async_safe_call("find_abandoned_sessions", run)

    @mcp.tool()
    async def session_profile(session_id: str, tier: str = "merged") -> str:
        """Get a single session profile by session ID."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            insight = await poly.get_session_profile_insight(
                session_id,
                tier=tier,
            )
            if insight is None:
                return hooks.error_json("Session not found", code="not_found", session_id=session_id)
            return hooks.json_payload(
                MCPRootPayload(root=cast(dict[str, object], insight.model_dump(mode="json"))),
                exclude_none=True,
            )

        return await hooks.async_safe_call("session_profile", run, session_id=session_id)

    @mcp.tool()
    async def get_resume_brief(session_id: str, related_limit: int = 6) -> str:
        """Get a typed resume brief for one archived session.

        The brief composes already-materialized session insights (profile,
        enrichment, work events, phases, work thread) into a handoff
        payload. Provenance fields cite the session, message, work-event,
        and phase IDs that contributed.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            brief = await poly.resume_brief(session_id, related_limit=related_limit)
            if brief is None:
                return hooks.error_json("Session not found", code="not_found", session_id=session_id)
            return hooks.json_payload(brief, exclude_none=False)

        return await hooks.async_safe_call("get_resume_brief", run, session_id=session_id)

    @mcp.tool()
    async def find_resume_candidates(
        repo_path: str,
        cwd: str | None = None,
        recent_files: tuple[str, ...] = (),
        limit: int = 10,
    ) -> str:
        """Rank logical sessions likely to match the operator's current context."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            candidates = await poly.find_resume_candidates(
                repo_path=repo_path,
                cwd=cwd,
                recent_files=recent_files,
                limit=hooks.clamp_limit(limit),
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                        "total": len(candidates),
                    }
                ),
                exclude_none=False,
            )

        return await hooks.async_safe_call("find_resume_candidates", run)

    @mcp.tool()
    async def cost_outlook(plan: str, method: str = "linear") -> str:
        """Project the current billing cycle for a subscription plan (#1138).

        Returns the typed :class:`polylogue.cost.outlook.CycleOutlook`
        payload from #1137: cycle window, burn rate, projected total,
        quota pressure, overage rows, coverage ratio, and confidence.

        Parameters
        ----------
        plan:
            Subscription plan name (e.g. ``"claude-pro"``,
            ``"claude-max-5x"``, ``"chatgpt-plus"``). User-supplied rows
            from ``[[cost.subscription.plans]]`` are merged with the
            curated seed; user rows always win.
        method:
            Projection method tag — one of ``"linear"``,
            ``"trailing-7d-mean"``, ``"eom-naive"``. The chosen method
            is echoed in the response so callers cannot lose track of
            how the projection was made.

        Returns the JSON-serialized outlook on success, or a typed
        error envelope when the plan is unknown or has no cycle anchor.
        """
        from polylogue.cost.outlook import ProjectionMethod
        from polylogue.cost.plans import PlanLookupError

        async def run() -> str:
            try:
                projection_method = ProjectionMethod(method)
            except ValueError:
                return hooks.error_json(
                    f"Unknown projection method {method!r}.",
                    code="invalid_argument",
                    detail=f"plan={plan!r} method={method!r}",
                    tool="cost_outlook",
                )
            try:
                poly = hooks.get_polylogue()
                outlook = await poly.cost_outlook(plan, method=projection_method)
            except PlanLookupError as exc:
                return hooks.error_json(
                    str(exc),
                    code="not_found",
                    detail=f"plan={plan!r}",
                    tool="cost_outlook",
                )
            if outlook is None:
                return hooks.error_json(
                    f"Plan {plan!r} has no cycle_anchor_day; cannot project a cycle window.",
                    code="no_cycle_window",
                    detail=f"plan={plan!r}",
                    tool="cost_outlook",
                )
            return hooks.json_payload(outlook, exclude_none=False)

        return await hooks.async_safe_call("cost_outlook", run)

    @mcp.tool()
    async def insight_rigor_audit(sample_limit: int = 500) -> str:
        """Per-product rigor profile across materialized insights (#1275).

        Returns the JSON-serialized :class:`InsightRigorAuditReport`. For
        each contracted insight product, reports the share of rows that
        carry evidence/inference/fallback markers, the stale-version row
        count, and a confidence-bucket distribution.
        """

        async def run() -> str:
            from polylogue.insights.audit import InsightRigorAuditQuery

            poly = hooks.get_polylogue()
            report = await poly.insight_rigor_audit(InsightRigorAuditQuery(sample_limit=sample_limit))
            return hooks.json_payload(report, exclude_none=True)

        return await hooks.async_safe_call("insight_rigor_audit", run)

    @mcp.tool()
    async def aggregate_sessions(
        group_by: str = "workflow_shape",
        since: str | None = None,
        until: str | None = None,
        origin: str | None = None,
    ) -> str:
        """Aggregate session counts by a dimension (workflow_shape, terminal_state, origin).

        #1691: programmatic session analysis primitives — GROUP BY over session profiles.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            try:
                result = await poly.aggregate_sessions(
                    group_by=group_by,
                    since=since,
                    until=until,
                    origin=origin,
                )
            except ValueError as exc:
                return hooks.error_json(
                    str(exc),
                    code="invalid_argument",
                    tool="aggregate_sessions",
                )
            return hooks.json_payload(MCPRootPayload(root=result))

        return await hooks.async_safe_call("aggregate_sessions", run)

    @mcp.tool()
    async def compare_sessions(session_ids: str | None = None) -> str:
        """Compare multiple session profiles side-by-side.

        Takes a comma-separated list of 2-10 session IDs, fetches
        their merged-tier profiles, and returns a side-by-side comparison
        with highlighted differences.

        #1691: programmatic session analysis primitives.
        """

        async def run() -> str:
            if not session_ids:
                return hooks.error_json(
                    "No session_ids provided.",
                    code="invalid_argument",
                    detail="session_ids must be a comma-separated list of 2-10 session IDs.",
                    tool="compare_sessions",
                )
            ids = [s.strip() for s in session_ids.split(",") if s.strip()]
            if len(ids) < 2:
                return hooks.error_json(
                    "Need at least 2 session IDs to compare.",
                    code="invalid_argument",
                    detail=f"Got {len(ids)} ID(s); expected 2-10.",
                    tool="compare_sessions",
                )
            if len(ids) > 10:
                return hooks.error_json(
                    "Too many session IDs.",
                    code="invalid_argument",
                    detail=f"Got {len(ids)} IDs; maximum is 10.",
                    tool="compare_sessions",
                )
            poly = hooks.get_polylogue()
            result = await poly.compare_sessions(ids)
            return hooks.json_payload(MCPRootPayload(root=result), exclude_none=True)

        correlated_ids = tuple(s.strip() for s in (session_ids or "").split(",") if s.strip())
        return await hooks.async_safe_call("compare_sessions", run, session_ids=correlated_ids)

    @mcp.tool()
    async def find_similar_sessions(
        session_id: str,
        similarity_dimension: str = "auto",
        limit: int = 10,
    ) -> str:
        """Find sessions similar to a given session.

        If embeddings are enabled, delegates to neighbor_candidates.
        Otherwise falls back to metadata similarity: same workflow_shape,
        same origin, similar time window, overlapping tags.

        #1691: programmatic session analysis primitives.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            capped_limit = hooks.clamp_limit(limit)

            # Try neighbor_candidates first when embeddings may be active.
            use_neighbor = similarity_dimension in ("auto", "embedding")
            use_metadata = similarity_dimension in ("auto", "metadata")

            if use_neighbor and getattr(poly.config, "embedding_enabled", False):
                try:
                    neighbors = await poly.neighbor_candidates(
                        session_id=session_id,
                        limit=capped_limit,
                    )
                    if neighbors:
                        return hooks.json_payload(
                            MCPRootPayload(
                                root={
                                    "source_session_id": session_id,
                                    "method": "embedding",
                                    "similar": [
                                        {
                                            "session_id": n.session_id,
                                            "score": n.score,
                                            "rank": n.rank,
                                            "reasons": [
                                                {"kind": r.kind, "detail": r.detail, "evidence": r.evidence}
                                                for r in n.reasons
                                            ],
                                            "title": n.summary.title,
                                            "origin": str(n.summary.origin) if n.summary.origin else None,
                                        }
                                        for n in neighbors
                                    ],
                                }
                            ),
                            exclude_none=True,
                        )
                except Exception as exc:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug("find_similar_sessions: embedding path failed, falling back to metadata: %s", exc)
                    # Fall through to metadata similarity on any failure.

            if similarity_dimension == "embedding" and not use_metadata:
                return hooks.error_json(
                    "Embedding-based similarity is not available.",
                    code="embeddings_unavailable",
                    detail="Enable embeddings in polylogue.toml or use similarity_dimension='auto'.",
                    tool="find_similar_sessions",
                )

            if not use_metadata and not use_neighbor:
                return hooks.error_json(
                    "Invalid similarity_dimension.",
                    code="invalid_argument",
                    detail="similarity_dimension must be one of: auto, embedding, metadata.",
                    tool="find_similar_sessions",
                )

            # Metadata-based similarity fallback (#1691 / polylogue-9e5.24).
            result = await poly.find_similar_sessions_by_metadata(
                session_id,
                limit=capped_limit,
                candidate_pool_limit=hooks.clamp_limit(200),
            )
            if result is None:
                return hooks.error_json(
                    "Session not found.",
                    code="not_found",
                    session_id=session_id,
                    tool="find_similar_sessions",
                )
            return hooks.json_payload(MCPRootPayload(root=result), exclude_none=True)

        return await hooks.async_safe_call("find_similar_sessions", run, session_id=session_id)

    @mcp.tool()
    async def correlate_sessions(
        metric_x: str,
        metric_y: str,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> str:
        """Compute Pearson correlation between two numeric session metrics.

        Fetches all session profiles matching the optional filters, extracts
        the two named metrics, and returns the Pearson r with interpretation.

        Supported metric names from session profiles:
        message_count, word_count, tool_use_count, thinking_count,
        engaged_duration_ms, tool_active_duration_ms, wall_duration_ms,
        total_cost_usd, total_duration_ms, substantive_count.

        #1691: programmatic session analysis primitives.
        """

        from polylogue.insights.session_analytics import CORRELATABLE_SESSION_METRICS

        async def run() -> str:
            poly = hooks.get_polylogue()
            try:
                result = await poly.correlate_sessions(
                    metric_x=metric_x,
                    metric_y=metric_y,
                    origin=origin,
                    since=since,
                    until=until,
                )
            except ValueError as exc:
                return hooks.error_json(
                    str(exc),
                    code="invalid_argument",
                    detail=f"Supported metrics: {sorted(CORRELATABLE_SESSION_METRICS)}",
                    tool="correlate_sessions",
                )
            return hooks.json_payload(MCPRootPayload(root=result))

        return await hooks.async_safe_call("correlate_sessions", run)

    @mcp.tool()
    async def correlate_session(
        session_id: str,
        repo_path: str | None = None,
        since_hours: int = 2,
        confidence_threshold: float = 0.3,
    ) -> str:
        """Link a session to git commits and GitHub issue/PR references (#1690).

        Returns git commits likely produced by the session (via time-window
        analysis and file-overlap scoring), plus GitHub issue/PR references
        extracted from session message text.
        """

        async def run() -> str:
            from polylogue.archive.message.models import Message
            from polylogue.insights.session_commit import (
                build_correlation_result,
                correlation_result_to_payload,
            )

            poly = hooks.get_polylogue()
            conv = await poly.get_session(session_id)
            if conv is None:
                return hooks.error_json(
                    "Session not found",
                    code="not_found",
                    session_id=session_id,
                )

            # Build messages list from the session
            messages: list[dict[str, object]] = []
            for msg in conv.messages:
                if isinstance(msg, Message):
                    msg_dict: dict[str, object] = {
                        "id": msg.id,
                        "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                        "text": msg.text,
                        "content_blocks": list(msg.blocks) if msg.blocks else [],
                    }
                    messages.append(msg_dict)
                else:
                    msg_id = getattr(msg, "id", "")
                    msg_text = getattr(msg, "text", None)
                    content_blocks = getattr(msg, "blocks", None) or []
                    msg_dict = {
                        "id": str(msg_id),
                        "role": str(getattr(msg, "role", "")),
                        "text": msg_text,
                        "content_blocks": list(content_blocks) if content_blocks else [],
                    }
                    messages.append(msg_dict)

            repo: str = repo_path or "."
            if not repo_path:
                repo_url = getattr(conv, "git_repository_url", None)
                if isinstance(repo_url, str) and repo_url:
                    repo = repo_url
                else:
                    directories = getattr(conv, "working_directories", ()) or ()
                    repo = str(directories[0]) if directories else "."

            start = conv.created_at
            end = conv.updated_at

            result = build_correlation_result(
                session_id=session_id,
                messages=messages,
                session_created_at=start,
                session_updated_at=end,
                repo_path=repo,
                before_hours=since_hours,
                after_hours=since_hours,
                confidence_threshold=confidence_threshold,
            )

            return hooks.json_payload(
                MCPRootPayload(root=correlation_result_to_payload(result)),
                exclude_none=True,
            )

        return await hooks.async_safe_call("correlate_session", run, session_id=session_id)

    @mcp.tool()
    async def session_tool_timing(session_id: str) -> str:
        """Get per-tool timing breakdown with evidence provenance.

        Returns per-tool timing from OTLP spans when available, falling back
        to message-gap estimates. Each timing entry carries an
        ``evidence_source`` field: ``"otlp_span"`` for exact wallclock timing
        from instrumented tool calls, or ``"message_gap_estimate"`` for timing
        inferred from inter-message gaps.
        """

        async def run() -> str:
            # Verify the session exists
            poly = hooks.get_polylogue()
            conv = await poly.get_session(session_id)
            if conv is None:
                return hooks.error_json(
                    "Session not found",
                    code="not_found",
                    session_id=session_id,
                )

            from polylogue.insights.otlp_correlation import get_session_tool_timing
            from polylogue.paths import active_index_db_path

            timing = get_session_tool_timing(str(active_index_db_path()), session_id)
            return hooks.json_payload(
                MCPRootPayload(root=timing.as_dict()),
                exclude_none=True,
            )

        return await hooks.async_safe_call("session_tool_timing", run, session_id=session_id)

    @mcp.tool()
    async def get_postmortem_bundle(
        query: str | None = None,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tag: str | None = None,
        repo: str | None = None,
        limit: int | None = None,
    ) -> str:
        """Distilled postmortem bundle over a matched session scope (#2380).

        Read-shaped delegate to :meth:`Polylogue.postmortem_bundle`: resolves
        the matched session set from the filter scope, aggregates per-session
        session digests, and returns the typed
        :class:`polylogue.insights.postmortem.PostmortemBundle` (top sessions
        by cost, repos touched, tool/work-kind rollups, failure-mode signals).
        The analysis cap defaults to 200 sessions; a larger match marks the
        bundle ``truncated`` rather than silently capping.

        This is the agent-preferred distilled read over the matched scope.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            spec = MCPSessionQueryRequest(
                query=query,
                origin=origin,
                since=since,
                until=until,
                tag=tag,
                repo=repo,
            ).build_spec(hooks.clamp_limit)
            # build_spec writes the MCP default page limit (10) into spec.limit,
            # which `_archive_query_kwargs` prefers over the candidate-scope
            # default — that would cap the postmortem at 10 sessions and defeat
            # its own analysis cap. Drop the page limit so the full matched scope
            # is considered; `postmortem_bundle`'s `limit` is the analysis cap.
            spec = replace(spec, limit=None)
            bundle = await poly.postmortem_bundle(spec, limit=limit)
            return hooks.json_payload(bundle, exclude_none=True)

        return await hooks.async_safe_call("get_postmortem_bundle", run)

    @mcp.tool()
    async def get_pathologies(
        query: str | None = None,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tag: str | None = None,
        repo: str | None = None,
        limit: int | None = None,
    ) -> str:
        """Agent-workflow pathology distribution over a matched scope (#2383).

        Read-shaped delegate to :meth:`Polylogue.pathology_report`: runs the
        deterministic detectors (wasted-loop, missed-review, stale-context) over
        the matched sessions' run projections and returns the typed
        :class:`polylogue.insights.pathology.PathologyReport` — findings with
        evidence refs plus the per-kind distribution. Deterministic and
        rule-based (no LLM-as-judge); a rebuild over identical evidence is
        identical.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            spec = MCPSessionQueryRequest(
                query=query,
                origin=origin,
                since=since,
                until=until,
                tag=tag,
                repo=repo,
            ).build_spec(hooks.clamp_limit)
            spec = replace(spec, limit=None)
            report = await poly.pathology_report(spec, limit=limit)
            return hooks.json_payload(report, exclude_none=True)

        return await hooks.async_safe_call("get_pathologies", run)


__all__ = ["register_insight_tools"]
