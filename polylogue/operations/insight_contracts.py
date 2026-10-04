"""Closed registry insight request and result contracts for resident pages."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from polylogue.analysis.archive import (
    ArchiveCoverageInsight,
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsight,
    ArchiveDebtInsightQuery,
    CostRollupInsight,
    CostRollupInsightQuery,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    ThreadInsight,
    ThreadInsightQuery,
    UsageTimelineInsight,
    UsageTimelineInsightQuery,
)
from polylogue.analysis.audit import InsightRigorAuditQuery, InsightRigorAuditReport
from polylogue.analysis.command_shapes import CommandShapeUsage, CommandShapeUsageQuery
from polylogue.analysis.readiness import InsightReadinessQuery, InsightReadinessReport, normalize_insight_readiness_name
from polylogue.analysis.tool_episodes import ToolEpisodeInsight, ToolEpisodeQuery
from polylogue.analysis.tool_usage import ToolUsageInsight, ToolUsageInsightQuery
from polylogue.surfaces.outcome import OutcomeEnvelope


class _InsightPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _InsightQuery(_InsightPage):
    @model_validator(mode="before")
    @classmethod
    def declared_query_fields(cls, value: object) -> object:
        from polylogue.analysis.registry import build_insight_query, get_insight_type

        if isinstance(value, dict) and isinstance(value.get("insight"), str) and isinstance(value.get("query"), dict):
            build_insight_query(get_insight_type(value["insight"]), **value["query"])
        return value


class SessionProfilesQuery(_InsightQuery):
    insight: Literal["session_profiles"]
    query: SessionProfileInsightQuery


class SessionProfilesPage(_InsightPage):
    insight: Literal["session_profiles"]
    items: list[SessionProfileInsight]
    total: int = Field(ge=0)


class ThreadsQuery(_InsightQuery):
    insight: Literal["threads"]
    query: ThreadInsightQuery


class ThreadsPage(_InsightPage):
    insight: Literal["threads"]
    items: list[ThreadInsight]
    total: int = Field(ge=0)


class SessionTagRollupsQuery(_InsightQuery):
    insight: Literal["session_tag_rollups"]
    query: SessionTagRollupQuery


class SessionTagRollupsPage(_InsightPage):
    insight: Literal["session_tag_rollups"]
    items: list[SessionTagRollupInsight]
    total: int = Field(ge=0)


class ArchiveCoverageQuery(_InsightQuery):
    insight: Literal["archive_coverage"]
    query: ArchiveCoverageInsightQuery


class ArchiveCoveragePage(_InsightPage):
    insight: Literal["archive_coverage"]
    items: list[ArchiveCoverageInsight]
    total: int = Field(ge=0)


class ToolUsageQuery(_InsightQuery):
    insight: Literal["tool_usage"]
    query: ToolUsageInsightQuery


class ToolUsagePage(_InsightPage):
    insight: Literal["tool_usage"]
    items: list[ToolUsageInsight]
    total: int = Field(ge=0)


class ToolEpisodesQuery(_InsightQuery):
    insight: Literal["tool_episodes"]
    query: ToolEpisodeQuery


class ToolEpisodesPage(_InsightPage):
    insight: Literal["tool_episodes"]
    items: list[ToolEpisodeInsight]
    total: int = Field(ge=0)


class CommandShapesQuery(_InsightQuery):
    insight: Literal["command_shapes"]
    query: CommandShapeUsageQuery


class CommandShapesPage(_InsightPage):
    insight: Literal["command_shapes"]
    items: list[CommandShapeUsage]
    total: int = Field(ge=0)


class SessionCostsQuery(_InsightQuery):
    insight: Literal["session_costs"]
    query: SessionCostInsightQuery


class SessionCostsPage(_InsightPage):
    insight: Literal["session_costs"]
    items: list[SessionCostInsight]
    total: int = Field(ge=0)


class CostRollupsQuery(_InsightQuery):
    insight: Literal["cost_rollups"]
    query: CostRollupInsightQuery


class CostRollupsPage(_InsightPage):
    insight: Literal["cost_rollups"]
    items: list[CostRollupInsight]
    total: int = Field(ge=0)


class UsageTimelineQuery(_InsightQuery):
    insight: Literal["usage_timeline"]
    query: UsageTimelineInsightQuery


class UsageTimelinePage(_InsightPage):
    insight: Literal["usage_timeline"]
    items: list[UsageTimelineInsight]
    total: int = Field(ge=0)


class ArchiveDebtQuery(_InsightQuery):
    insight: Literal["archive_debt"]
    query: ArchiveDebtInsightQuery


class ArchiveDebtPage(_InsightPage):
    insight: Literal["archive_debt"]
    items: list[ArchiveDebtInsight]
    total: int = Field(ge=0)


InsightQuery = Annotated[
    SessionProfilesQuery
    | ThreadsQuery
    | SessionTagRollupsQuery
    | ArchiveCoverageQuery
    | ToolUsageQuery
    | ToolEpisodesQuery
    | CommandShapesQuery
    | SessionCostsQuery
    | CostRollupsQuery
    | UsageTimelineQuery
    | ArchiveDebtQuery,
    Field(discriminator="insight"),
]

InsightPage = Annotated[
    SessionProfilesPage
    | ThreadsPage
    | SessionTagRollupsPage
    | ArchiveCoveragePage
    | ToolUsagePage
    | ToolEpisodesPage
    | CommandShapesPage
    | SessionCostsPage
    | CostRollupsPage
    | UsageTimelinePage
    | ArchiveDebtPage,
    Field(discriminator="insight"),
]


class InsightListRequest(_InsightPage):
    page: InsightQuery


class InsightListResult(_InsightPage):
    page: InsightPage
    outcome: OutcomeEnvelope


class InsightReadinessRequest(_InsightPage):
    query: InsightReadinessQuery

    @model_validator(mode="after")
    def validate_selected_targets(self) -> InsightReadinessRequest:
        for name in self.query.insights:
            normalize_insight_readiness_name(name)
        return self


class InsightReadinessResult(_InsightPage):
    report: InsightReadinessReport
    outcome: OutcomeEnvelope


class InsightRigorRequest(_InsightPage):
    query: InsightRigorAuditQuery


class InsightRigorResult(_InsightPage):
    report: InsightRigorAuditReport
    outcome: OutcomeEnvelope
