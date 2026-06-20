"""Registry-driven MCP archive-insight tool registration.

Iterates INSIGHT_REGISTRY and registers a ``list_<name>`` MCP tool for
each insight type. Special one-off tools for single-item lookups and
derived distributions are registered directly.
"""

from __future__ import annotations

import inspect
from datetime import date
from math import ceil
from typing import TYPE_CHECKING, Any, cast

from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin, source_name_to_origin
from polylogue.insights.archive import (
    SessionLatencyProfileInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
)
from polylogue.insights.registry import (
    INSIGHT_REGISTRY,
    InsightType,
    fetch_insights_async,
    insight_items_payload,
    project_origin_payload,
)
from polylogue.mcp.insight_tool_contracts import InsightListToolSpec
from polylogue.mcp.payloads import MCPRootPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def _origin_to_provider_token(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return provider_from_origin(Origin(value)).value


def _project_origin_payload(value: object) -> object:
    return project_origin_payload(value)


def _register_list_tool(
    mcp: FastMCP,
    hooks: ServerCallbacks,
    pt: InsightType,
) -> None:
    """Register one list-style MCP tool for an insight type."""
    spec = InsightListToolSpec.from_insight_type(pt)

    async def tool_fn(**kwargs: object) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            normalized_kwargs = spec.normalize_kwargs(hooks.clamp_limit, kwargs)
            insights = await fetch_insights_async(pt, poly, **normalized_kwargs)
            payload = insight_items_payload(insights, pt, item_key="items")
            return hooks.json_payload(MCPRootPayload(root=cast(dict[str, object], _project_origin_payload(payload))))

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


def register_insight_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
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

        def percentile(values: list[int], p: float) -> int:
            if not values:
                return 0
            sorted_values = sorted(values)
            rank = max(0, ceil(p / 100.0 * len(sorted_values)) - 1)
            return sorted_values[rank]

        async def run() -> str:
            poly = hooks.get_polylogue()
            insights = await poly.list_session_latency_profile_insights(
                SessionLatencyProfileInsightQuery(
                    provider=_origin_to_provider_token(origin),
                    since=since,
                    until=until,
                    limit=hooks.clamp_limit(limit),
                )
            )
            if tool_category:
                insights = [
                    insight
                    for insight in insights
                    if insight.latency.tool_call_count_by_category.get(tool_category, 0) > 0
                ]
            medians = [
                insight.latency.median_tool_call_ms for insight in insights if insight.latency.median_tool_call_ms
            ]
            p90s = [insight.latency.p90_tool_call_ms for insight in insights if insight.latency.p90_tool_call_ms]
            maxes = [insight.latency.max_tool_call_ms for insight in insights if insight.latency.max_tool_call_ms]
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "total_sessions": len(insights),
                        "tool_category": tool_category,
                        "median_tool_call_ms": percentile(medians, 50),
                        "p90_tool_call_ms": percentile(p90s, 90),
                        "max_tool_call_ms": max(maxes) if maxes else 0,
                        "stuck_tool_count": sum(insight.latency.stuck_tool_count for insight in insights),
                        "construct_boundary": (
                            "distribution is over materialized per-session aggregates; "
                            "agent-response time includes both LLM inference and tool execution"
                        ),
                    }
                )
            )

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
                MCPRootPayload(root=cast(dict[str, object], _project_origin_payload(insight.model_dump(mode="json")))),
                exclude_none=True,
            )

        return await hooks.async_safe_call("session_latency_profile", run)

    @mcp.tool()
    async def find_stuck_sessions(since: str | None = None, limit: int = 20) -> str:
        """Find sessions with provider tool calls bounded as stuck."""

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
                        "items": [_project_origin_payload(insight.model_dump(mode="json")) for insight in insights],
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
            profiles = await poly.list_session_profile_insights(
                SessionProfileInsightQuery(
                    provider=_origin_to_provider_token(origin),
                    since=since,
                    until=until,
                    limit=None,
                )
            )
            allowed_group_by = {"week", "origin", "project"}
            if group_by not in allowed_group_by:
                return hooks.error_json(
                    "Invalid group_by.",
                    code="invalid_argument",
                    detail="group_by must be one of week, origin, project",
                    tool="workflow_shape_distribution",
                )
            buckets: dict[str, dict[str, int]] = {}
            for profile in profiles:
                evidence = profile.evidence
                inference = profile.inference
                shape = inference.workflow_shape if inference is not None else "unknown"
                keys: tuple[str, ...]
                if group_by == "origin":
                    keys = (source_name_to_origin(profile.source_name),)
                elif group_by == "project":
                    paths = evidence.cwd_paths if evidence is not None else ()
                    keys = tuple(paths) or ("unattributed",)
                else:
                    date_value = evidence.canonical_session_date if evidence is not None else None
                    if date_value:
                        try:
                            parsed = date.fromisoformat(date_value)
                            iso_year, iso_week, _ = parsed.isocalendar()
                            week_key = f"{iso_year}-W{iso_week:02d}"
                        except ValueError:
                            week_key = date_value[:7]
                    else:
                        week_key = "undated"
                    keys = (week_key,)
                for key in keys:
                    bucket = buckets.setdefault(key, {})
                    bucket[shape] = bucket.get(shape, 0) + 1
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "group_by": group_by,
                        "total_sessions": len(profiles),
                        "buckets": buckets,
                    }
                )
            )

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
            severity = {
                "question_left": 1,
                "error_left": 2,
                "tool_left": 3,
                "agent_hanging": 4,
            }
            if min_severity not in severity:
                return hooks.error_json(
                    "Invalid min_severity.",
                    code="invalid_argument",
                    detail="min_severity must be one of question_left, error_left, tool_left, agent_hanging",
                    tool="find_abandoned_sessions",
                )
            poly = hooks.get_polylogue()
            profiles = await poly.list_session_profile_insights(
                SessionProfileInsightQuery(
                    since=since,
                    limit=None,
                )
            )
            min_rank = severity[min_severity]
            items: list[dict[str, object]] = []
            for profile in profiles:
                inference = profile.inference
                evidence = profile.evidence
                state = inference.terminal_state if inference is not None else "unknown"
                if severity.get(state, 0) < min_rank:
                    continue
                cwd_paths = evidence.cwd_paths if evidence is not None else ()
                if repo_path and not any(repo_path in path for path in cwd_paths):
                    continue
                items.append(
                    {
                        "session_id": profile.session_id,
                        "origin": source_name_to_origin(profile.source_name),
                        "title": profile.title,
                        "terminal_state": state,
                        "terminal_state_confidence": (
                            inference.terminal_state_confidence if inference is not None else 0.0
                        ),
                        "workflow_shape": inference.workflow_shape if inference is not None else "unknown",
                        "canonical_session_date": evidence.canonical_session_date if evidence is not None else None,
                        "evidence": evidence.terminal_state_evidence if evidence is not None else {},
                    }
                )
            items.sort(key=lambda item: str(item.get("canonical_session_date") or ""), reverse=True)
            capped = items[: hooks.clamp_limit(limit)]
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "total": len(items),
                        "items": capped,
                    }
                )
            )

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
                MCPRootPayload(root=cast(dict[str, object], _project_origin_payload(insight.model_dump(mode="json")))),
                exclude_none=True,
            )

        return await hooks.async_safe_call("session_profile", run)

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

        return await hooks.async_safe_call("get_resume_brief", run)

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
            profiles = await poly.list_session_profile_insights(
                SessionProfileInsightQuery(
                    provider=_origin_to_provider_token(origin),
                    since=since,
                    until=until,
                    limit=hooks.clamp_limit(10000),
                )
            )
            buckets: dict[str, int] = {}
            for p in profiles:
                if group_by == "workflow_shape":
                    key = (p.inference.workflow_shape if p.inference else None) or "unknown"
                elif group_by == "terminal_state":
                    key = (p.inference.terminal_state if p.inference else None) or "unknown"
                elif group_by == "origin":
                    key = source_name_to_origin(p.source_name)
                else:
                    return hooks.error_json(
                        f"Unknown group_by: {group_by!r}. Supported: workflow_shape, terminal_state, origin.",
                        code="invalid_argument",
                        tool="aggregate_sessions",
                    )
                buckets[key] = buckets.get(key, 0) + 1
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "group_by": group_by,
                        "total_sessions": len(profiles),
                        "buckets": buckets,
                    }
                )
            )

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
            sessions: list[dict[str, object]] = []
            not_found: list[str] = []
            for conv_id in ids:
                profile = await poly.get_session_profile_insight(conv_id)
                if profile is None:
                    not_found.append(conv_id)
                    continue
                evidence = profile.evidence
                inference = profile.inference
                sessions.append(
                    {
                        "id": profile.session_id,
                        "origin": source_name_to_origin(profile.source_name),
                        "title": profile.title,
                        "workflow_shape": inference.workflow_shape if inference else "unknown",
                        "terminal_state": inference.terminal_state if inference else "unknown",
                        "message_count": evidence.message_count if evidence else 0,
                        "tool_call_count": evidence.tool_use_count if evidence else 0,
                        "engaged_duration_ms": inference.engaged_duration_ms if inference else 0,
                        "tool_active_duration_ms": evidence.tool_active_duration_ms if evidence else 0,
                        "word_count": evidence.word_count if evidence else 0,
                        "tags": list(evidence.tags) if evidence else [],
                        "auto_tags": list(inference.auto_tags) if inference else [],
                    }
                )

            # Collect per-key value sets across found sessions to surface differences.
            diff_keys = (
                "origin",
                "workflow_shape",
                "terminal_state",
                "message_count",
                "tool_call_count",
                "engaged_duration_ms",
                "tool_active_duration_ms",
                "word_count",
            )
            differences: dict[str, list[object]] = {}
            for key in diff_keys:
                vals = {s.get(key) for s in sessions}
                if len(vals) > 1:
                    differences[key] = sorted(vals, key=str)

            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "sessions": sessions,
                        "differences": differences,
                        "not_found": not_found,
                        "total_requested": len(ids),
                        "total_found": len(sessions),
                    }
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("compare_sessions", run)

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

            # Metadata-based similarity fallback.
            ref_profile = await poly.get_session_profile_insight(session_id)
            if ref_profile is None:
                return hooks.error_json(
                    "Session not found.",
                    code="not_found",
                    session_id=session_id,
                    tool="find_similar_sessions",
                )

            ref_evidence = ref_profile.evidence
            ref_inference = ref_profile.inference
            ref_shape = ref_inference.workflow_shape if ref_inference else None
            ref_source = ref_profile.source_name
            ref_origin = source_name_to_origin(ref_source)
            ref_date = ref_evidence.canonical_session_date if ref_evidence else None
            ref_tags = set(ref_evidence.tags) if ref_evidence else set()

            candidates = await poly.list_session_profile_insights(
                SessionProfileInsightQuery(
                    provider=ref_source,
                    limit=hooks.clamp_limit(200),
                )
            )

            scored: list[tuple[int, dict[str, object]]] = []
            for profile in candidates:
                if profile.session_id == session_id:
                    continue
                evidence = profile.evidence
                inference = profile.inference
                score = 0
                reasons: list[str] = []

                cand_shape = inference.workflow_shape if inference else None
                if ref_shape and cand_shape == ref_shape:
                    score += 3
                    reasons.append(f"same workflow_shape: {ref_shape}")

                cand_source = profile.source_name
                if ref_source and cand_source == ref_source:
                    score += 1
                    reasons.append(f"same origin: {ref_origin}")

                cand_date = evidence.canonical_session_date if evidence else None
                if ref_date and cand_date:
                    try:
                        ref_d = date.fromisoformat(ref_date)
                        cand_d = date.fromisoformat(cand_date)
                        delta = abs((ref_d - cand_d).days)
                        if delta <= 3:
                            score += 2
                            reasons.append(f"within 3 days (delta={delta})")
                        elif delta <= 14:
                            score += 1
                            reasons.append(f"within 14 days (delta={delta})")
                    except (ValueError, TypeError):
                        pass

                cand_tags = set(evidence.tags) if evidence else set()
                overlap = ref_tags & cand_tags
                if overlap:
                    score += len(overlap)
                    reasons.append(f"{len(overlap)} overlapping tags: {sorted(overlap)}")

                if score > 0:
                    scored.append(
                        (
                            -score,
                            {
                                "session_id": profile.session_id,
                                "title": profile.title,
                                "origin": source_name_to_origin(profile.source_name),
                                "workflow_shape": cand_shape or "unknown",
                                "terminal_state": inference.terminal_state if inference else "unknown",
                                "canonical_session_date": cand_date,
                                "similarity_score": score,
                                "similarity_reasons": reasons,
                            },
                        )
                    )

            scored.sort(key=lambda x: x[0])
            top = [item for _, item in scored[:capped_limit]]

            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "source_session_id": session_id,
                        "method": "metadata",
                        "similar": top,
                    }
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("find_similar_sessions", run)

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

        import math

        _numeric_metrics: set[str] = {
            "message_count",
            "word_count",
            "tool_use_count",
            "thinking_count",
            "engaged_duration_ms",
            "tool_active_duration_ms",
            "wall_duration_ms",
            "total_cost_usd",
            "total_duration_ms",
            "substantive_count",
        }

        async def run() -> str:
            if metric_x not in _numeric_metrics:
                return hooks.error_json(
                    f"Unknown metric_x: {metric_x!r}",
                    code="invalid_argument",
                    detail=f"Supported metrics: {sorted(_numeric_metrics)}",
                    tool="correlate_sessions",
                )
            if metric_y not in _numeric_metrics:
                return hooks.error_json(
                    f"Unknown metric_y: {metric_y!r}",
                    code="invalid_argument",
                    detail=f"Supported metrics: {sorted(_numeric_metrics)}",
                    tool="correlate_sessions",
                )

            poly = hooks.get_polylogue()
            profiles = await poly.list_session_profile_insights(
                SessionProfileInsightQuery(
                    provider=_origin_to_provider_token(origin),
                    since=since,
                    until=until,
                    limit=hooks.clamp_limit(10000),
                )
            )

            # Extract (x, y) pairs, resolving from evidence/inference.
            def _get_metric(profile: SessionProfileInsight, key: str) -> float | None:
                evidence = profile.evidence
                inference = profile.inference
                if key == "message_count":
                    return float(evidence.message_count) if evidence else None
                if key == "word_count":
                    return float(evidence.word_count) if evidence else None
                if key == "tool_use_count":
                    return float(evidence.tool_use_count) if evidence else None
                if key == "thinking_count":
                    return float(evidence.thinking_count) if evidence else None
                if key == "engaged_duration_ms":
                    return float(inference.engaged_duration_ms) if inference else None
                if key == "tool_active_duration_ms":
                    return float(evidence.tool_active_duration_ms) if evidence else None
                if key == "wall_duration_ms":
                    return float(evidence.wall_duration_ms) if evidence else None
                if key == "total_cost_usd":
                    return float(evidence.total_cost_usd) if evidence else None
                if key == "total_duration_ms":
                    return float(evidence.total_duration_ms) if evidence else None
                if key == "substantive_count":
                    return float(evidence.substantive_count) if evidence else None
                return None

            pairs: list[tuple[float, float]] = []
            for p in profiles:
                x = _get_metric(p, metric_x)
                y = _get_metric(p, metric_y)
                if x is not None and y is not None:
                    pairs.append((x, y))

            n = len(pairs)
            if n < 3:
                return hooks.json_payload(
                    MCPRootPayload(
                        root={
                            "metric_x": metric_x,
                            "metric_y": metric_y,
                            "pearson_r": None,
                            "sample_count": n,
                            "interpretation": "insufficient data (need at least 3 samples)",
                        }
                    )
                )

            sum_x = sum(p[0] for p in pairs)
            sum_y = sum(p[1] for p in pairs)
            sum_xy = sum(p[0] * p[1] for p in pairs)
            sum_x2 = sum(p[0] * p[0] for p in pairs)
            sum_y2 = sum(p[1] * p[1] for p in pairs)

            denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
            if denominator == 0:
                return hooks.json_payload(
                    MCPRootPayload(
                        root={
                            "metric_x": metric_x,
                            "metric_y": metric_y,
                            "pearson_r": None,
                            "sample_count": n,
                            "interpretation": "constant metric — zero variance, correlation undefined",
                        }
                    )
                )

            r = (n * sum_xy - sum_x * sum_y) / denominator
            r = max(-1.0, min(1.0, r))

            if abs(r) >= 0.7:
                direction = "strong positive" if r > 0 else "strong negative"
            elif abs(r) >= 0.4:
                direction = "moderate positive" if r > 0 else "moderate negative"
            elif abs(r) >= 0.2:
                direction = "weak positive" if r > 0 else "weak negative"
            else:
                direction = "negligible"

            interpretation = f"{direction} correlation (r={r:.3f})"

            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "metric_x": metric_x,
                        "metric_y": metric_y,
                        "pearson_r": round(r, 4),
                        "sample_count": n,
                        "interpretation": interpretation,
                    }
                )
            )

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

        return await hooks.async_safe_call("correlate_session", run)

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

        return await hooks.async_safe_call("session_tool_timing", run)


__all__ = ["register_insight_tools"]
