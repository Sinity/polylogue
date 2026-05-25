"""Registry-driven MCP archive-insight tool registration.

Iterates INSIGHT_REGISTRY and registers a ``list_<name>`` MCP tool for
each insight type. Special one-off tools (archive_coverage, single-item
lookups) are registered directly.
"""

from __future__ import annotations

import inspect
from datetime import date
from typing import TYPE_CHECKING, Any, cast

from polylogue.insights.archive import SessionProfileInsightQuery
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
                    keys = (profile.provider_name,)
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
                        "provider_name": profile.provider_name,
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
    async def session_classification(conversation_id: str) -> str:
        """Classify a session into the typed SessionCategory taxonomy.

        Returns the typed classification, confidence in [0, 1], coarse
        support_level ("strong"/"moderate"/"weak"), and evidence citations
        naming the SessionProfile fields that drove the decision.
        Heuristic / suggestion-grade; user-authored tags remain
        authoritative.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            classification = await poly.classify_session(conversation_id)
            if classification is None:
                return hooks.error_json(
                    "Conversation not found",
                    code="not_found",
                    conversation_id=conversation_id,
                )
            return hooks.json_payload(classification, exclude_none=True)

        return await hooks.async_safe_call("session_classification", run)

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
    async def archive_coverage() -> str:
        """Show archive coverage statistics."""

        async def run() -> str:
            from polylogue.archive.coverage import analyze_coverage

            summaries = await hooks.get_query_store().list_summaries()
            coverage = analyze_coverage(summaries)
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "total_conversations": coverage.total_conversations,
                        "total_messages": coverage.total_messages,
                        "provider_counts": coverage.provider_counts,
                        "provider_ranges": [
                            {
                                "provider": r.provider,
                                "first_date": r.first_date.isoformat(),
                                "last_date": r.last_date.isoformat(),
                                "count": r.count,
                            }
                            for r in coverage.provider_ranges
                        ],
                        "gaps": [
                            {
                                "start_date": g.start_date.isoformat(),
                                "end_date": g.end_date.isoformat(),
                                "days": g.days,
                            }
                            for g in coverage.gaps
                        ],
                        "truncated_sessions": coverage.truncated_sessions,
                        "date_range": [d.isoformat() if d else None for d in coverage.date_range],
                    }
                )
            )

        return await hooks.async_safe_call("archive_coverage", run)


__all__ = ["register_insight_tools"]
