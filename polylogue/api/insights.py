"""Durable insight methods for the async Polylogue facade."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.session.session_profile import SessionProfile
from polylogue.cost.aggregation import session_costs_to_daily_usd
from polylogue.cost.outlook import CycleOutlook, ProjectionMethod, build_cycle_outlook
from polylogue.cost.plans import resolve_plan
from polylogue.insights.archive import (
    ArchiveDebtInsight,
    ArchiveDebtInsightQuery,
    CostRollupInsight,
    CostRollupInsightQuery,
    DaySessionSummaryInsight,
    DaySessionSummaryInsightQuery,
    ProviderAnalyticsInsight,
    ProviderAnalyticsInsightQuery,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionPhaseInsight,
    SessionPhaseInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    SessionWorkEventInsight,
    SessionWorkEventInsightQuery,
    WeekSessionSummaryInsight,
    WeekSessionSummaryInsightQuery,
    WorkThreadInsight,
    WorkThreadInsightQuery,
)
from polylogue.insights.classification import SessionClassification, classify_session
from polylogue.insights.feedback import LearningCorrection
from polylogue.insights.productivity import (
    ProductivityRollupInsight,
    ProductivityRollupInsightQuery,
)
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery
from polylogue.insights.topology import ConversationRef, SessionTopology

if TYPE_CHECKING:

    class _InsightOperationsSurface(Protocol):
        async def list_session_tag_rollup_insights(
            self,
            query: SessionTagRollupQuery | None = None,
        ) -> list[SessionTagRollupInsight]: ...

        async def get_session_work_event_insights(
            self,
            conversation_id: str,
        ) -> list[SessionWorkEventInsight]: ...

        async def list_session_work_event_insights(
            self,
            query: SessionWorkEventInsightQuery | None = None,
        ) -> list[SessionWorkEventInsight]: ...

        async def get_session_phase_insights(
            self,
            conversation_id: str,
        ) -> list[SessionPhaseInsight]: ...

        async def list_session_phase_insights(
            self,
            query: SessionPhaseInsightQuery | None = None,
        ) -> list[SessionPhaseInsight]: ...

        async def get_work_thread_insight(self, thread_id: str) -> WorkThreadInsight | None: ...

        async def list_work_thread_insights(
            self,
            query: WorkThreadInsightQuery | None = None,
        ) -> list[WorkThreadInsight]: ...

        async def list_day_session_summary_insights(
            self,
            query: DaySessionSummaryInsightQuery | None = None,
        ) -> list[DaySessionSummaryInsight]: ...

        async def list_week_session_summary_insights(
            self,
            query: WeekSessionSummaryInsightQuery | None = None,
        ) -> list[WeekSessionSummaryInsight]: ...

        async def list_provider_analytics_insights(
            self,
            query: ProviderAnalyticsInsightQuery | None = None,
        ) -> list[ProviderAnalyticsInsight]: ...

        async def list_tool_usage_insights(
            self,
            query: ToolUsageInsightQuery | None = None,
        ) -> list[ToolUsageInsight]: ...

        async def list_productivity_rollup_insights(
            self,
            query: ProductivityRollupInsightQuery | None = None,
        ) -> list[ProductivityRollupInsight]: ...

        async def list_session_cost_insights(
            self,
            query: SessionCostInsightQuery | None = None,
        ) -> list[SessionCostInsight]: ...

        async def list_cost_rollup_insights(
            self,
            query: CostRollupInsightQuery | None = None,
        ) -> list[CostRollupInsight]: ...

        async def list_archive_debt_insights(
            self,
            query: ArchiveDebtInsightQuery | None = None,
        ) -> list[ArchiveDebtInsight]: ...

        async def list_corrections(
            self,
            *,
            conversation_id: str | None = None,
            kind: str | None = None,
        ) -> list[LearningCorrection]: ...


class _RepositorySurface(Protocol):
    async def get_session_profile(self, conversation_id: str) -> SessionProfile | None: ...

    async def get_session_topology(self, conversation_id: str) -> SessionTopology | None: ...

    async def resolve_id(self, conversation_id: str, *, strict: bool = False) -> object: ...


class PolylogueInsightsMixin:
    if TYPE_CHECKING:

        @property
        def operations(self) -> _InsightOperationsSurface: ...

        @property
        def repository(self) -> _RepositorySurface: ...

    async def list_session_tag_rollup_insights(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        return await self.operations.list_session_tag_rollup_insights(query)

    async def get_session_work_event_insights(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventInsight]:
        return await self.operations.get_session_work_event_insights(conversation_id)

    async def list_session_work_event_insights(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        return await self.operations.list_session_work_event_insights(query)

    async def get_session_phase_insights(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseInsight]:
        return await self.operations.get_session_phase_insights(conversation_id)

    async def list_session_phase_insights(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        return await self.operations.list_session_phase_insights(query)

    async def get_work_thread_insight(self, thread_id: str) -> WorkThreadInsight | None:
        return await self.operations.get_work_thread_insight(thread_id)

    async def list_work_thread_insights(
        self,
        query: WorkThreadInsightQuery | None = None,
    ) -> list[WorkThreadInsight]:
        return await self.operations.list_work_thread_insights(query)

    async def list_day_session_summary_insights(
        self,
        query: DaySessionSummaryInsightQuery | None = None,
    ) -> list[DaySessionSummaryInsight]:
        return await self.operations.list_day_session_summary_insights(query)

    async def list_week_session_summary_insights(
        self,
        query: WeekSessionSummaryInsightQuery | None = None,
    ) -> list[WeekSessionSummaryInsight]:
        return await self.operations.list_week_session_summary_insights(query)

    async def list_provider_analytics_insights(
        self,
        query: ProviderAnalyticsInsightQuery | None = None,
    ) -> list[ProviderAnalyticsInsight]:
        return await self.operations.list_provider_analytics_insights(query)

    async def list_tool_usage_insights(
        self,
        query: ToolUsageInsightQuery | None = None,
    ) -> list[ToolUsageInsight]:
        return await self.operations.list_tool_usage_insights(query)

    async def list_productivity_rollup_insights(
        self,
        query: ProductivityRollupInsightQuery | None = None,
    ) -> list[ProductivityRollupInsight]:
        return await self.operations.list_productivity_rollup_insights(query)

    async def list_session_cost_insights(
        self,
        query: SessionCostInsightQuery | None = None,
    ) -> list[SessionCostInsight]:
        return await self.operations.list_session_cost_insights(query)

    async def list_cost_rollup_insights(
        self,
        query: CostRollupInsightQuery | None = None,
    ) -> list[CostRollupInsight]:
        return await self.operations.list_cost_rollup_insights(query)

    async def list_archive_debt_insights(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        return await self.operations.list_archive_debt_insights(query)

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

    async def classify_session(self, conversation_id: str) -> SessionClassification | None:
        """Classify a session into the typed :class:`SessionCategory` taxonomy.

        Computed on-the-fly from the hydrated :class:`SessionProfile`. The
        heuristic classifier is pure and deterministic; user corrections
        recorded via :meth:`record_correction` are applied on top so a
        rebuild always produces the same merged verdict (AC #1131).
        Returns ``None`` when no profile exists for ``conversation_id``.
        """

        from polylogue.insights.feedback import apply_correction_to_classification

        profile = await self.repository.get_session_profile(conversation_id)
        if profile is None:
            return None
        base = classify_session(profile)
        # Consult user corrections after the heuristic — corrections win
        # without altering the conversation's content hash.
        corrections = await self.operations.list_corrections(conversation_id=conversation_id)
        if not corrections:
            return base
        return apply_correction_to_classification(base, corrections)

    # ------------------------------------------------------------------
    # Topology read API (#1261 / #866 slice D)
    #
    # Surface the derived ``SessionTopology`` graph as a typed Python API
    # so MCP, future reader panes, and context packs consume one read
    # model instead of re-walking parent pointers. Each helper accepts a
    # short or full conversation ID and resolves it before delegating.
    # ------------------------------------------------------------------

    async def _resolve_for_topology(self, conversation_id: str) -> str | None:
        resolved = await self.repository.resolve_id(conversation_id, strict=True)
        if resolved is None:
            return None
        return str(resolved)

    async def get_session_topology(self, conversation_id: str) -> SessionTopology | None:
        """Return the typed :class:`SessionTopology` for ``conversation_id``.

        Returns ``None`` when the conversation is unknown. Cycles and
        unresolved native parent edges are surfaced via the topology
        object itself; see :class:`SessionTopology`.
        """

        resolved = await self._resolve_for_topology(conversation_id)
        if resolved is None:
            return None
        return await self.repository.get_session_topology(resolved)

    async def get_ancestors(self, conversation_id: str) -> list[ConversationRef]:
        """Return ancestor refs ordered root → parent.

        Empty list when the conversation is its own topology root, when
        it is unknown, or when no resolved ancestors exist.
        """

        resolved = await self._resolve_for_topology(conversation_id)
        if resolved is None:
            return []
        topology = await self.repository.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.ancestor_refs(resolved)

    async def get_descendants(self, conversation_id: str) -> list[ConversationRef]:
        """Return descendant refs in BFS order."""

        resolved = await self._resolve_for_topology(conversation_id)
        if resolved is None:
            return []
        topology = await self.repository.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.descendant_refs(resolved)

    async def get_siblings(self, conversation_id: str) -> list[ConversationRef]:
        """Return sibling refs (other children of the resolved parent)."""

        resolved = await self._resolve_for_topology(conversation_id)
        if resolved is None:
            return []
        topology = await self.repository.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.sibling_refs(resolved)

    async def get_thread(self, conversation_id: str) -> list[ConversationRef]:
        """Return the full lineage thread ordered ancestors → self → descendants."""

        resolved = await self._resolve_for_topology(conversation_id)
        if resolved is None:
            return []
        topology = await self.repository.get_session_topology(resolved)
        if topology is None:
            return []
        return topology.thread_refs(resolved)
