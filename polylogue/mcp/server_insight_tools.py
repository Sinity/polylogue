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

from polylogue.insights.archive import SessionLatencyProfileInsightQuery, SessionProfileInsightQuery
from polylogue.insights.registry import (
    INSIGHT_REGISTRY,
    InsightType,
    fetch_insights_async,
    insight_items_payload,
)
from polylogue.mcp.insight_tool_contracts import InsightListToolSpec
from polylogue.mcp.payloads import MCPRootPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


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
            return hooks.json_payload(MCPRootPayload(root=insight_items_payload(insights, pt, item_key="items")))

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
        if pt.query_model is not None and pt.operations_method_name:
            _register_list_tool(mcp, hooks, pt)

    # --- Special tools ---

    @mcp.tool()
    async def tool_call_latency_distribution(
        since: str | None = None,
        until: str | None = None,
        provider: str | None = None,
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
                    provider=provider,
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
    async def session_latency_profile(conversation_id: str) -> str:
        """Get per-session latency profile by conversation ID."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            insight = await poly.get_session_latency_profile_insight(conversation_id)
            if insight is None:
                return hooks.error_json(
                    "Conversation not found",
                    code="not_found",
                    conversation_id=conversation_id,
                )
            return hooks.json_payload(insight, exclude_none=True)

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
        provider: str | None = None,
    ) -> str:
        """Histogram session workflow shapes by week, provider, or project."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            profiles = await poly.list_session_profile_insights(
                SessionProfileInsightQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    limit=None,
                )
            )
            allowed_group_by = {"week", "provider", "project"}
            if group_by not in allowed_group_by:
                return hooks.error_json(
                    "Invalid group_by.",
                    code="invalid_argument",
                    detail="group_by must be one of week, provider, project",
                    tool="workflow_shape_distribution",
                )
            buckets: dict[str, dict[str, int]] = {}
            for profile in profiles:
                evidence = profile.evidence
                inference = profile.inference
                shape = inference.workflow_shape if inference is not None else "unknown"
                keys: tuple[str, ...]
                if group_by == "provider":
                    keys = (profile.source_name,)
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
                        "conversation_id": profile.conversation_id,
                        "source_name": profile.source_name,
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
    async def session_profile(conversation_id: str, tier: str = "merged") -> str:
        """Get a single session profile by conversation ID."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            insight = await poly.get_session_profile_insight(
                conversation_id,
                tier=tier,
            )
            if insight is None:
                return hooks.error_json("Conversation not found", code="not_found", conversation_id=conversation_id)
            return hooks.json_payload(insight, exclude_none=True)

        return await hooks.async_safe_call("session_profile", run)

    @mcp.tool()
    async def get_resume_brief(conversation_id: str, related_limit: int = 6) -> str:
        """Get a typed resume brief for one archived session.

        The brief composes already-materialized session insights (profile,
        enrichment, work events, phases, work thread) into a handoff
        payload. Provenance fields cite the session, message, work-event,
        and phase IDs that contributed.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            brief = await poly.resume_brief(conversation_id, related_limit=related_limit)
            if brief is None:
                return hooks.error_json("Conversation not found", code="not_found", conversation_id=conversation_id)
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
        provider: str | None = None,
    ) -> str:
        """Aggregate session counts by a dimension (workflow_shape, terminal_state, provider).

        #1691: programmatic session analysis primitives — GROUP BY over session profiles.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            profiles = await poly.list_session_profile_insights(
                SessionProfileInsightQuery(
                    provider=provider,
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
                elif group_by == "provider":
                    key = p.source_name or "unknown"
                else:
                    return hooks.json_payload(
                        MCPRootPayload(
                            root={
                                "error": f"Unknown group_by: {group_by!r}. Supported: workflow_shape, terminal_state, provider."
                            }
                        )
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


__all__ = ["register_insight_tools"]
