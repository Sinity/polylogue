"""Durable insight methods for the async Polylogue facade."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from polylogue.api.archive import _active_archive_root, _provider_for_archive_origin
from polylogue.archive.query.spec import parse_query_date
from polylogue.archive.session.branch_type import BranchType
from polylogue.cost.aggregation import session_costs_to_daily_usd
from polylogue.cost.outlook import CycleOutlook, ProjectionMethod, build_cycle_outlook
from polylogue.cost.plans import resolve_plan
from polylogue.insights.archive import (
    ArchiveCoverageInsight,
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsight,
    ArchiveDebtInsightQuery,
    CostRollupInsight,
    CostRollupInsightQuery,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionLatencyProfileInsight,
    SessionLatencyProfileInsightQuery,
    SessionPhaseInsight,
    SessionPhaseInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    SessionWorkEventInsight,
    SessionWorkEventInsightQuery,
    ThreadInsight,
    ThreadInsightQuery,
    UsageTimelineInsight,
    UsageTimelineInsightQuery,
)
from polylogue.insights.cost_enrichment import enrich_session_cost_insights
from polylogue.insights.tag_rollups import synthesize_provider_tag_rollups
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery
from polylogue.insights.topology import (
    LogicalSession,
    SessionRef,
    SessionTopology,
    TopologyEdge,
    TopologyEdgeKind,
    TopologyNode,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.types import SessionId

if TYPE_CHECKING:
    from polylogue.config import Config

    class _InsightOperationsSurface(Protocol):
        async def list_session_tag_rollup_insights(
            self,
            query: SessionTagRollupQuery | None = None,
        ) -> list[SessionTagRollupInsight]: ...

        async def get_session_work_event_insights(
            self,
            session_id: str,
        ) -> list[SessionWorkEventInsight]: ...

        async def list_session_work_event_insights(
            self,
            query: SessionWorkEventInsightQuery | None = None,
        ) -> list[SessionWorkEventInsight]: ...

        async def get_session_phase_insights(
            self,
            session_id: str,
        ) -> list[SessionPhaseInsight]: ...

        async def list_session_phase_insights(
            self,
            query: SessionPhaseInsightQuery | None = None,
        ) -> list[SessionPhaseInsight]: ...

        async def get_thread_insight(self, thread_id: str) -> ThreadInsight | None: ...

        async def list_thread_insights(
            self,
            query: ThreadInsightQuery | None = None,
        ) -> list[ThreadInsight]: ...

        async def list_archive_coverage_insights(
            self,
            query: ArchiveCoverageInsightQuery | None = None,
        ) -> list[ArchiveCoverageInsight]: ...

        async def list_tool_usage_insights(
            self,
            query: ToolUsageInsightQuery | None = None,
        ) -> list[ToolUsageInsight]: ...

        async def list_session_cost_insights(
            self,
            query: SessionCostInsightQuery | None = None,
        ) -> list[SessionCostInsight]: ...

        async def get_session_latency_profile_insight(
            self,
            session_id: str,
        ) -> SessionLatencyProfileInsight | None: ...

        async def list_session_latency_profile_insights(
            self,
            query: SessionLatencyProfileInsightQuery | None = None,
        ) -> list[SessionLatencyProfileInsight]: ...

        async def find_stuck_session_latency_profile_insights(
            self,
            query: SessionLatencyProfileInsightQuery | None = None,
        ) -> list[SessionLatencyProfileInsight]: ...

        async def list_cost_rollup_insights(
            self,
            query: CostRollupInsightQuery | None = None,
        ) -> list[CostRollupInsight]: ...

        async def list_usage_timeline_insights(
            self,
            query: UsageTimelineInsightQuery | None = None,
        ) -> list[UsageTimelineInsight]: ...

        async def list_archive_debt_insights(
            self,
            query: ArchiveDebtInsightQuery | None = None,
        ) -> list[ArchiveDebtInsight]: ...


class _RepositorySurface(Protocol):
    async def get_session_topology(self, session_id: str) -> SessionTopology | None: ...

    async def resolve_id(self, session_id: str, *, strict: bool = False) -> object: ...


def _archive_query_date_ms(field: str, value: str | None) -> int | None:
    parsed = parse_query_date(field, value)
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


_DAY_MS = 86_400_000


def _session_date_lower_ms(value: str | None) -> int | None:
    """Map a ``--session-date-since`` date to the start-of-day epoch ms."""
    return _archive_query_date_ms("session_date_since", value)


def _session_date_upper_ms(value: str | None) -> int | None:
    """Map a ``--session-date-until`` date to an inclusive end-of-day bound.

    ``parse_query_date`` resolves a bare ``YYYY-MM-DD`` to midnight; a native
    ``<= until_ms`` test on that would exclude every session whose timestamp
    falls later in the same day. Widen the bound to the last millisecond of
    the named day so a one-day window (``since == until``) selects the whole
    day, matching the legacy canonical-session-date contract.
    """
    start = _archive_query_date_ms("session_date_until", value)
    if start is None:
        return None
    return start + _DAY_MS - 1


def _combine_lower_ms(*candidates: int | None) -> int | None:
    """Tightest (largest) lower bound across the provided candidates."""
    present = [value for value in candidates if value is not None]
    return max(present) if present else None


def _combine_upper_ms(*candidates: int | None) -> int | None:
    """Tightest (smallest) upper bound across the provided candidates."""
    present = [value for value in candidates if value is not None]
    return min(present) if present else None


def _archive_topology_edge_kind(branch_type: str | None) -> TopologyEdgeKind:
    if branch_type is None:
        return TopologyEdgeKind.UNKNOWN
    try:
        return TopologyEdgeKind.from_branch_type(BranchType(branch_type))
    except ValueError:
        return TopologyEdgeKind.UNKNOWN


def _archive_session_topology(archive: object, session_id: str) -> SessionTopology | None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_store = archive if isinstance(archive, ArchiveStore) else None
    if archive_store is None:
        return None
    try:
        target_id = archive_store.resolve_session_id(session_id)
    except KeyError:
        return None
    envelopes = archive_store.get_session_tree(target_id)
    if not envelopes:
        return None
    by_id = {envelope.session_id: envelope for envelope in envelopes}
    root_id = by_id[target_id].root_session_id or envelopes[0].session_id
    if root_id not in by_id:
        root_id = envelopes[0].session_id

    children: dict[str, list[str]] = {}
    edges: list[TopologyEdge] = []
    for envelope in envelopes:
        if envelope.parent_session_id and envelope.parent_session_id in by_id:
            children.setdefault(envelope.parent_session_id, []).append(envelope.session_id)
            edges.append(
                TopologyEdge(
                    child_id=SessionId(envelope.session_id),
                    parent_id=SessionId(envelope.parent_session_id),
                    kind=_archive_topology_edge_kind(envelope.branch_type),
                    resolved=True,
                )
            )

    # Surface unresolved parent links (#866/#1258): a parser asserted a
    # provider-native parent pointer that does not (yet) resolve to a stored
    # session. These rows live in session_links with resolved_dst_session_id IS NULL;
    # they must be reported so late repair has something to reconcile.
    placeholders = ", ".join("?" for _ in by_id)
    if placeholders:
        unresolved_rows = archive_store._conn.execute(
            f"""
            SELECT src_session_id, dst_native_id, link_type
            FROM session_links
            WHERE resolved_dst_session_id IS NULL
              AND src_session_id IN ({placeholders})
            ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_native_id, link_type
            """,
            tuple(by_id),
        ).fetchall()
        for row in unresolved_rows:
            edges.append(
                TopologyEdge(
                    child_id=SessionId(str(row["src_session_id"])),
                    parent_id=None,
                    parent_native_id=str(row["dst_native_id"]),
                    kind=TopologyEdgeKind.UNRESOLVED_NATIVE,
                    resolved=False,
                )
            )

    depths: dict[str, int] = {root_id: 0}
    queue = [root_id]
    seen = {root_id}
    cycle_detected = False
    while queue:
        current = queue.pop(0)
        for child_id in children.get(current, ()):
            if child_id in seen:
                cycle_detected = True
                continue
            seen.add(child_id)
            depths[child_id] = depths[current] + 1
            queue.append(child_id)

    nodes = tuple(
        TopologyNode(
            session_id=SessionId(envelope.session_id),
            source_name=_provider_for_archive_origin(envelope.origin).value,
            title=envelope.title,
            depth=depths.get(envelope.session_id, 0),
            is_root=envelope.session_id == root_id,
        )
        for envelope in sorted(envelopes, key=lambda item: (depths.get(item.session_id, 0), item.session_id))
    )
    return SessionTopology(
        target_id=SessionId(target_id),
        root_id=SessionId(root_id),
        nodes=nodes,
        edges=tuple(edges),
        cycle_detected=cycle_detected,
    )


class PolylogueInsightsMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def operations(self) -> _InsightOperationsSurface: ...

        @property
        def repository(self) -> _RepositorySurface: ...

        # Cross-mixin members from ``PolylogueArchiveMixin`` (api/archive.py),
        # a sibling in the composed ``Polylogue`` facade -- see #1691/9e5.24
        # analysis primitives, which read session profiles fetched there.
        async def list_session_profile_insights(
            self,
            query: SessionProfileInsightQuery | None = None,
        ) -> list[SessionProfileInsight]: ...

        async def get_session_profile_insight(
            self,
            session_id: str,
            *,
            tier: str = "merged",
        ) -> SessionProfileInsight | None: ...

    async def list_session_tag_rollup_insights(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        request = query or SessionTagRollupQuery()
        since_ms = _archive_query_date_ms("since", request.since)
        until_ms = _archive_query_date_ms("until", request.until)
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            # The archive rebuild does not write ``provider:<name>`` rows into
            # session_tags (provider identity lives on sessions.origin), so the
            # archive read only returns explicit/auto tags. Synthesize the
            # provider rollups to preserve the legacy ``insights tags``
            # contract, then merge them with the materialized tag rollups.
            materialized = archive.list_session_tag_rollup_insights(
                provider=request.provider,
                query=request.query,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=None,
                offset=0,
            )
            provider_rollups = synthesize_provider_tag_rollups(
                archive,
                provider=request.provider,
                query=request.query,
                since_ms=since_ms,
                until_ms=until_ms,
                materialized_at=datetime.now(UTC).isoformat(),
            )
        rollups = sorted(
            [*materialized, *provider_rollups],
            key=lambda rollup: (-rollup.session_count, rollup.tag),
        )
        if request.offset:
            rollups = rollups[request.offset :]
        if request.limit is not None:
            rollups = rollups[: max(int(request.limit), 0)]
        return rollups

    async def get_session_work_event_insights(
        self,
        session_id: str,
    ) -> list[SessionWorkEventInsight]:
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_session_work_event_insights(session_id)

    async def list_session_work_event_insights(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        request = query or SessionWorkEventInsightQuery()
        since_ms = _combine_lower_ms(
            _archive_query_date_ms("since", request.since),
            _session_date_lower_ms(request.session_date_since),
        )
        until_ms = _combine_upper_ms(
            _archive_query_date_ms("until", request.until),
            _session_date_upper_ms(request.session_date_until),
        )
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_session_work_event_insights(
                session_id=request.session_id,
                provider=request.provider,
                heuristic_label=request.heuristic_label,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=request.limit,
                offset=request.offset,
            )

    async def get_session_phase_insights(
        self,
        session_id: str,
    ) -> list[SessionPhaseInsight]:
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_session_phase_insights(session_id)

    async def list_session_phase_insights(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        request = query or SessionPhaseInsightQuery()
        since_ms = _combine_lower_ms(
            _archive_query_date_ms("since", request.since),
            _session_date_lower_ms(request.session_date_since),
        )
        until_ms = _combine_upper_ms(
            _archive_query_date_ms("until", request.until),
            _session_date_upper_ms(request.session_date_until),
        )
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_session_phase_insights(
                session_id=request.session_id,
                provider=request.provider,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=request.limit,
                offset=request.offset,
            )

    async def get_thread_insight(self, thread_id: str) -> ThreadInsight | None:
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_thread_insight(thread_id)

    async def list_thread_insights(
        self,
        query: ThreadInsightQuery | None = None,
    ) -> list[ThreadInsight]:
        request = query or ThreadInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_thread_insights(
                query=request.query,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )

    async def list_archive_coverage_insights(
        self,
        query: ArchiveCoverageInsightQuery | None = None,
    ) -> list[ArchiveCoverageInsight]:
        request = query or ArchiveCoverageInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_archive_coverage_insights(
                group_by=request.group_by,
                provider=request.provider,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )

    async def list_tool_usage_insights(
        self,
        query: ToolUsageInsightQuery | None = None,
    ) -> list[ToolUsageInsight]:
        with ArchiveStore(_active_archive_root(self.config)) as archive:
            return archive.list_tool_usage_insights(query)

    async def list_session_cost_insights(
        self,
        query: SessionCostInsightQuery | None = None,
    ) -> list[SessionCostInsight]:
        request = query or SessionCostInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            # The archive read returns a degraded estimate built from the
            # stored scalar cost. Re-derive the full estimate (model identity,
            # catalog basis, missing-reason taxonomy) from the session so
            # the public ``model``/``normalized_model`` fields are populated.
            # The ``model`` and ``status`` filters key on those re-derived
            # fields, so apply them after enrichment rather than asking the
            # archive (which cannot answer a model filter).
            insights = enrich_session_cost_insights(
                archive,
                archive.list_session_cost_insights(
                    session_id=request.session_id,
                    provider=request.provider,
                    status=None,
                    model=None,
                    since_ms=_archive_query_date_ms("since", request.since),
                    until_ms=_archive_query_date_ms("until", request.until),
                    limit=request.limit,
                    offset=request.offset,
                ),
            )
        if request.status is not None:
            insights = [insight for insight in insights if insight.estimate.status == request.status]
        if request.model is not None:
            insights = [
                insight
                for insight in insights
                if request.model in {insight.estimate.normalized_model, insight.estimate.model_name}
            ]
        return insights

    async def get_session_latency_profile_insight(
        self,
        session_id: str,
    ) -> SessionLatencyProfileInsight | None:
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_session_latency_profile_insight(session_id)

    async def list_session_latency_profile_insights(
        self,
        query: SessionLatencyProfileInsightQuery | None = None,
    ) -> list[SessionLatencyProfileInsight]:
        request = query or SessionLatencyProfileInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_session_latency_profile_insights(
                session_id=request.session_id,
                provider=request.provider,
                only_stuck=request.only_stuck,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )

    async def find_stuck_session_latency_profile_insights(
        self,
        query: SessionLatencyProfileInsightQuery | None = None,
    ) -> list[SessionLatencyProfileInsight]:
        request = query or SessionLatencyProfileInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.find_stuck_session_latency_profile_insights(
                provider=request.provider,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
            )

    async def list_cost_rollup_insights(
        self,
        query: CostRollupInsightQuery | None = None,
    ) -> list[CostRollupInsight]:
        request = query or CostRollupInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_cost_rollup_insights(
                provider=request.provider,
                model=request.model,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )

    async def list_usage_timeline_insights(
        self,
        query: UsageTimelineInsightQuery | None = None,
    ) -> list[UsageTimelineInsight]:
        request = query or UsageTimelineInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_usage_timeline_insights(
                provider=request.provider,
                model=request.model,
                group_by=request.group_by,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )

    async def list_archive_debt_insights(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        request = query or ArchiveDebtInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_archive_debt_insights(
                category=request.category,
                only_actionable=request.only_actionable,
                limit=request.limit,
                offset=request.offset,
            )

    async def cost_outlook(
        self,
        plan_name: str,
        *,
        now: datetime | None = None,
        method: ProjectionMethod = ProjectionMethod.linear,
    ) -> CycleOutlook | None:
        """Project the current billing cycle for ``plan_name``.

        Resolves the plan against user-supplied
        ``[[cost.subscription.plans]]`` rows merged with the curated
        seed, lists materialized session-cost insights, folds them into
        per-day USD usage, and projects the current cycle using
        ``method``.

        Returns ``None`` when the plan has no ``cycle_anchor_day`` —
        surfaces must report "no cycle window" explicitly. Raises
        :class:`polylogue.cost.plans.PlanLookupError` when ``plan_name``
        is non-empty but unknown.
        """
        from polylogue.config import load_polylogue_config

        polylogue_config = load_polylogue_config()
        plan = resolve_plan(plan_name, user_rows=polylogue_config.subscription_plans)
        if plan is None:
            return None
        when = (now or datetime.now(UTC)).astimezone(UTC)
        session_costs = await self.list_session_cost_insights()
        daily = session_costs_to_daily_usd(session_costs)
        return build_cycle_outlook(plan, daily, now=when, method=method)

    # ------------------------------------------------------------------
    # Session analysis primitives (#1691 / polylogue-9e5.24)
    #
    # These were originally inline MCP-only math in
    # ``mcp/server_insight_tools.py`` -- unreachable from the CLI or this
    # library facade. The reducers/heuristics themselves live in
    # ``insights/archive_rollups.py`` and ``insights/session_analytics.py``;
    # these methods are the fetch-then-reduce composition so MCP and any
    # other caller share one definition of the math.
    # ------------------------------------------------------------------

    async def aggregate_sessions(
        self,
        *,
        group_by: str = "workflow_shape",
        since: str | None = None,
        until: str | None = None,
        provider: str | None = None,
    ) -> dict[str, object]:
        """GROUP BY session counts over workflow_shape/terminal_state/origin.

        Raises ``ValueError`` for an unsupported ``group_by``.
        """
        from polylogue.insights.archive_rollups import aggregate_session_profiles_by_dimension

        profiles = await self.list_session_profile_insights(
            SessionProfileInsightQuery(provider=provider, since=since, until=until, limit=None)
        )
        buckets = aggregate_session_profiles_by_dimension(profiles, group_by)
        return {"group_by": group_by, "total_sessions": len(profiles), "buckets": buckets}

    async def workflow_shape_distribution(
        self,
        *,
        group_by: str = "week",
        since: str | None = None,
        until: str | None = None,
        provider: str | None = None,
    ) -> dict[str, object]:
        """Histogram session workflow shapes by week, origin, or project.

        Raises ``ValueError`` when ``group_by`` is not one of
        ``week``, ``origin``, ``project``.
        """
        from polylogue.insights.archive_rollups import workflow_shape_distribution_buckets

        profiles = await self.list_session_profile_insights(
            SessionProfileInsightQuery(provider=provider, since=since, until=until, limit=None)
        )
        buckets = workflow_shape_distribution_buckets(profiles, group_by)
        return {"group_by": group_by, "total_sessions": len(profiles), "buckets": buckets}

    async def find_abandoned_sessions(
        self,
        *,
        since: str | None = None,
        repo_path: str | None = None,
        min_severity: str = "question_left",
        limit: int = 20,
    ) -> dict[str, object]:
        """Sessions whose terminal state indicates dangling work.

        Raises ``ValueError`` for an unknown ``min_severity``.
        """
        from polylogue.insights.archive_rollups import abandoned_session_items

        profiles = await self.list_session_profile_insights(SessionProfileInsightQuery(since=since, limit=None))
        items = abandoned_session_items(profiles, min_severity=min_severity, repo_path=repo_path)
        return {"total": len(items), "items": items[:limit]}

    async def tool_call_latency_distribution(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        provider: str | None = None,
        tool_category: str | None = None,
        limit: int = 500,
    ) -> dict[str, object]:
        """Distribution of materialized per-session tool-call latency."""
        from polylogue.insights.archive_rollups import tool_call_latency_distribution_payload

        insights = await self.list_session_latency_profile_insights(
            SessionLatencyProfileInsightQuery(provider=provider, since=since, until=until, limit=limit)
        )
        return tool_call_latency_distribution_payload(insights, tool_category=tool_category)

    async def compare_sessions(self, session_ids: Sequence[str]) -> dict[str, object]:
        """Compare 2-10 session profiles side by side.

        Raises ``ValueError`` when ``session_ids`` has fewer than 2 or more
        than 10 entries.
        """
        from polylogue.insights.session_analytics import build_session_comparison_row, diff_session_comparison_rows

        if len(session_ids) < 2:
            raise ValueError(f"Need at least 2 session IDs to compare. Got {len(session_ids)} ID(s); expected 2-10.")
        if len(session_ids) > 10:
            raise ValueError(f"Too many session IDs. Got {len(session_ids)} IDs; maximum is 10.")

        sessions: list[dict[str, object]] = []
        not_found: list[str] = []
        for session_id in session_ids:
            profile = await self.get_session_profile_insight(session_id)
            if profile is None:
                not_found.append(session_id)
                continue
            sessions.append(build_session_comparison_row(profile))

        return {
            "sessions": sessions,
            "differences": diff_session_comparison_rows(sessions),
            "not_found": not_found,
            "total_requested": len(session_ids),
            "total_found": len(sessions),
        }

    async def find_similar_sessions_by_metadata(
        self,
        session_id: str,
        *,
        limit: int = 10,
        candidate_pool_limit: int = 200,
    ) -> dict[str, object] | None:
        """Metadata-similarity fallback for ``find_similar_sessions``.

        Used when embeddings are unavailable or the caller explicitly asks
        for ``similarity_dimension="metadata"``. Returns ``None`` when
        ``session_id`` does not resolve to a session profile.
        """
        from polylogue.insights.session_analytics import compute_metadata_similarity_candidates

        ref_profile = await self.get_session_profile_insight(session_id)
        if ref_profile is None:
            return None
        candidates = await self.list_session_profile_insights(
            SessionProfileInsightQuery(provider=ref_profile.source_name, limit=candidate_pool_limit)
        )
        scored = compute_metadata_similarity_candidates(ref_profile, candidates, exclude_session_id=session_id)
        return {
            "source_session_id": session_id,
            "method": "metadata",
            "similar": scored[:limit],
        }

    async def correlate_sessions(
        self,
        *,
        metric_x: str,
        metric_y: str,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, object]:
        """Pearson correlation between two numeric session metrics.

        Raises ``ValueError`` for an unknown metric name (validated before
        fetching profiles).
        """
        from polylogue.insights.session_analytics import ensure_known_session_metric, pearson_session_correlation

        ensure_known_session_metric(metric_x, "metric_x")
        ensure_known_session_metric(metric_y, "metric_y")

        profiles = await self.list_session_profile_insights(
            SessionProfileInsightQuery(provider=provider, since=since, until=until, limit=None)
        )
        return pearson_session_correlation(profiles, metric_x=metric_x, metric_y=metric_y)

    # ------------------------------------------------------------------
    # Topology read API (#1261 / #866 slice D)
    #
    # Surface the derived ``SessionTopology`` graph as a typed Python API
    # so MCP, future reader panes, and context image/bundle projections consume one read
    # model instead of re-walking parent pointers. Each helper accepts a
    # short or full session ID and resolves it before delegating.
    # ------------------------------------------------------------------

    async def _resolve_for_topology(self, session_id: str) -> str | None:
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            try:
                return archive.resolve_session_id(session_id)
            except KeyError:
                return None

    async def get_session_topology(self, session_id: str) -> SessionTopology | None:
        """Return the typed :class:`SessionTopology` for ``session_id``.

        Returns ``None`` when the session is unknown. Cycles and
        unresolved native parent edges are surfaced via the topology
        object itself; see :class:`SessionTopology`.
        """
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return _archive_session_topology(archive, session_id)

    async def get_ancestors(self, session_id: str) -> list[SessionRef]:
        """Return ancestor refs ordered root → parent.

        Empty list when the session is its own topology root, when
        it is unknown, or when no resolved ancestors exist.
        """

        resolved = await self._resolve_for_topology(session_id)
        if resolved is None:
            return []
        topology = await self.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.ancestor_refs(resolved)

    async def get_descendants(self, session_id: str) -> list[SessionRef]:
        """Return descendant refs in BFS order."""

        resolved = await self._resolve_for_topology(session_id)
        if resolved is None:
            return []
        topology = await self.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.descendant_refs(resolved)

    async def get_siblings(self, session_id: str) -> list[SessionRef]:
        """Return sibling refs (other children of the resolved parent)."""

        resolved = await self._resolve_for_topology(session_id)
        if resolved is None:
            return []
        topology = await self.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.sibling_refs(resolved)

    async def get_thread(self, session_id: str) -> list[SessionRef]:
        """Return the full lineage thread ordered ancestors → self → descendants."""

        resolved = await self._resolve_for_topology(session_id)
        if resolved is None:
            return []
        topology = await self.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.thread_refs(resolved)

    async def get_logical_session(self, session_id: str) -> LogicalSession | None:
        """Return the compact logical-session read-pull envelope."""

        resolved = await self._resolve_for_topology(session_id)
        if resolved is None:
            return None
        topology = await self.get_session_topology(resolved)
        if topology is None:
            return None
        return LogicalSession(
            session_id=SessionId(resolved),
            root_id=topology.root_id,
            thread=tuple(topology.thread_refs(resolved)),
            siblings=tuple(topology.sibling_refs(resolved)),
            descendants=tuple(topology.descendant_refs(resolved)),
            cycle_detected=topology.cycle_detected,
        )
