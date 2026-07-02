"""Versioned archive insight contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, model_validator

from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostEstimatePayload,
    CostModelBreakdown,
    CostUsagePayload,
)
from polylogue.archive.session.session_profile import SessionProfile
from polylogue.errors import PolylogueError
from polylogue.insights.archive_models import (
    ARCHIVE_INSIGHT_CONTRACT_VERSION,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveInsightModel,
    ArchiveInsightProvenance,
    DaySessionSummaryPayload,
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionLatencyProfilePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    ThreadPayload,
    WeekSessionSummaryPayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.storage.repair import ArchiveDebtStatus
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION

if TYPE_CHECKING:
    from polylogue.storage.runtime import (
        SessionPhaseRecord,
        SessionProfileRecord,
        SessionWorkEventRecord,
        ThreadRecord,
    )


class ArchiveInsightUnavailableError(PolylogueError):
    """Raised when a durable archive-insight surface is not ready to read."""

    is_transient = True
    http_status_code = 503


class PaginatedInsightQuery(ArchiveInsightModel):
    limit: int | None = 50
    offset: int = 0


class TimeWindowInsightQuery(PaginatedInsightQuery):
    since: str | None = None
    until: str | None = None


class SearchableTimeWindowInsightQuery(TimeWindowInsightQuery):
    query: str | None = None

    @property
    def wants_search(self) -> bool:
        return bool(self.query)


class ProviderTimeWindowInsightQuery(TimeWindowInsightQuery):
    provider: str | None = None


class ProviderSearchInsightQuery(SearchableTimeWindowInsightQuery):
    provider: str | None = None


class SessionWindowInsightQuery(ProviderSearchInsightQuery):
    first_message_since: str | None = None
    first_message_until: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None
    min_wallclock_seconds: int | None = None
    max_wallclock_seconds: int | None = None
    sort: str = "source"


class SessionTimelineWindowInsightQuery(ProviderTimeWindowInsightQuery):
    session_id: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None


class SearchableSessionTimelineInsightQuery(SessionTimelineWindowInsightQuery):
    query: str | None = None

    @property
    def wants_search(self) -> bool:
        return bool(self.query)


class SessionProfileInsightQuery(SessionWindowInsightQuery):
    tier: str = "merged"
    workflow_shape: str | None = None
    terminal_state: str | None = None


class SessionLatencyProfileInsightQuery(ProviderTimeWindowInsightQuery):
    session_id: str | None = None
    only_stuck: bool = False


class SessionWorkEventInsightQuery(SearchableSessionTimelineInsightQuery):
    heuristic_label: str | None = None


class SessionPhaseInsightQuery(SessionTimelineWindowInsightQuery):
    pass


class ThreadInsightQuery(SearchableTimeWindowInsightQuery):
    pass


class SessionTagRollupQuery(ProviderSearchInsightQuery):
    limit: int | None = 100


class ArchiveCoverageInsightQuery(ProviderTimeWindowInsightQuery):
    group_by: str = "provider"
    limit: int | None = None


class SessionCostInsightQuery(ProviderTimeWindowInsightQuery):
    session_id: str | None = None
    model: str | None = None
    status: str | None = None


class CostRollupInsightQuery(ProviderTimeWindowInsightQuery):
    model: str | None = None
    limit: int | None = None


class ArchiveDebtInsightQuery(PaginatedInsightQuery):
    category: str | None = None
    only_actionable: bool = False
    limit: int | None = None


class _InsightRecordWithProvenance(Protocol):
    materializer_version: int
    materialized_at: str
    source_updated_at: str | None
    source_sort_key: float | None


class _InsightRecordWithInference(_InsightRecordWithProvenance, Protocol):
    inference_version: int
    inference_family: str


class _InsightRecordWithEnrichment(_InsightRecordWithProvenance, Protocol):
    enrichment_version: int
    enrichment_family: str


def _record_provenance(record: _InsightRecordWithProvenance) -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(
        materializer_version=record.materializer_version,
        materialized_at=record.materialized_at,
        source_updated_at=record.source_updated_at,
        source_sort_key=record.source_sort_key,
    )


def _record_inference_provenance(record: _InsightRecordWithInference) -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=record.materializer_version,
        materialized_at=record.materialized_at,
        source_updated_at=record.source_updated_at,
        source_sort_key=record.source_sort_key,
        inference_version=record.inference_version,
        inference_family=record.inference_family,
    )


def _record_enrichment_provenance(record: _InsightRecordWithEnrichment) -> ArchiveEnrichmentProvenance:
    return ArchiveEnrichmentProvenance(
        materializer_version=record.materializer_version,
        materialized_at=record.materialized_at,
        source_updated_at=record.source_updated_at,
        source_sort_key=record.source_sort_key,
        enrichment_version=record.enrichment_version,
        enrichment_family=record.enrichment_family,
    )


class SessionProfileInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "session_profile"
    semantic_tier: str = "merged"
    session_id: str
    logical_session_id: str
    source_name: str
    title: str | None = None
    provenance: ArchiveInsightProvenance
    evidence: SessionEvidencePayload | None = None
    inference_provenance: ArchiveInferenceProvenance | None = None
    inference: SessionInferencePayload | None = None
    enrichment_provenance: ArchiveEnrichmentProvenance | None = None
    enrichment: SessionEnrichmentPayload | None = None

    @classmethod
    def from_record(
        cls,
        record: SessionProfileRecord,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight:
        include_evidence = tier in {"merged", "evidence"}
        include_inference = tier in {"merged", "inference"}
        include_enrichment = tier == "merged"
        inference = record.inference_payload if include_inference else None
        if inference is not None:
            # The denormalized native session_profiles columns are the
            # authoritative ranking signals (terminal_state / workflow_shape);
            # the inference payload JSON is the structured-evidence copy. The
            # writer keeps them in sync, so reconcile the insight object onto the
            # native columns so ranking/aggregation reads the queryable authority.
            inference = inference.model_copy(
                update={
                    "terminal_state": record.terminal_state,
                    "terminal_state_confidence": record.terminal_state_confidence,
                    "workflow_shape": record.workflow_shape,
                    "workflow_shape_confidence": record.workflow_shape_confidence,
                }
            )
        return cls(
            semantic_tier=tier,
            session_id=str(record.session_id),
            logical_session_id=str(record.logical_session_id),
            source_name=record.source_name,
            title=record.title,
            provenance=_record_provenance(record),
            evidence=(record.evidence_payload if include_evidence else None),
            inference_provenance=(_record_inference_provenance(record) if include_inference else None),
            inference=inference,
            enrichment_provenance=(_record_enrichment_provenance(record) if include_enrichment else None),
            enrichment=(record.enrichment_payload if include_enrichment else None),
        )


class SessionLatencyProfileInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "session_latency_profile"
    session_id: str
    source_name: str
    title: str | None = None
    provenance: ArchiveInsightProvenance
    latency: SessionLatencyProfilePayload

    @classmethod
    def from_record(cls, record: object) -> SessionLatencyProfileInsight:
        from polylogue.storage.runtime import SessionLatencyProfileRecord
        from polylogue.storage.sqlite.queries.mappers import _json_int_dict, _json_object, _parse_json

        if not isinstance(record, SessionLatencyProfileRecord):
            raise TypeError(f"Expected SessionLatencyProfileRecord, got {type(record).__name__}")
        evidence = _json_object(_parse_json(record.evidence_payload_json)) or {}
        tool_counts = _json_int_dict(_parse_json(record.tool_call_count_by_category_json))
        payload = SessionLatencyProfilePayload(
            median_tool_call_ms=record.median_tool_call_ms,
            p90_tool_call_ms=record.p90_tool_call_ms,
            max_tool_call_ms=record.max_tool_call_ms,
            stuck_tool_count=record.stuck_tool_count,
            median_agent_response_ms=record.median_agent_response_ms,
            median_user_response_ms=record.median_user_response_ms,
            tool_call_count_by_category=tool_counts,
            construct_boundary=str(
                evidence.get("construct_boundary") or SessionLatencyProfilePayload().construct_boundary
            ),
        )
        return cls(
            session_id=str(record.session_id),
            source_name=record.source_name,
            title=record.title,
            provenance=_record_provenance(record),
            latency=payload,
        )


class SessionWorkEventInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "session_work_event"
    semantic_tier: str = "inference"
    event_id: str
    session_id: str
    source_name: str
    event_index: int
    provenance: ArchiveInsightProvenance
    inference_provenance: ArchiveInferenceProvenance
    evidence: WorkEventEvidencePayload
    inference: WorkEventInferencePayload

    @classmethod
    def from_record(cls, record: SessionWorkEventRecord) -> SessionWorkEventInsight:
        return cls(
            event_id=record.event_id,
            session_id=record.session_id,
            source_name=record.source_name,
            event_index=record.event_index,
            provenance=_record_provenance(record),
            inference_provenance=_record_inference_provenance(record),
            evidence=record.evidence_payload,
            inference=record.inference_payload,
        )


class SessionPhaseInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "session_phase"
    semantic_tier: str = "evidence"
    phase_id: str
    session_id: str
    source_name: str
    phase_index: int
    provenance: ArchiveInsightProvenance
    inference_provenance: ArchiveInferenceProvenance | None = None
    evidence: SessionPhaseEvidencePayload
    inference: SessionPhaseInferencePayload | None = None

    @classmethod
    def from_record(cls, record: SessionPhaseRecord) -> SessionPhaseInsight:
        return cls(
            phase_id=record.phase_id,
            session_id=record.session_id,
            source_name=record.source_name,
            phase_index=record.phase_index,
            provenance=_record_provenance(record),
            evidence=record.evidence_payload,
        )


class ThreadInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "thread"
    thread_id: str
    root_id: str
    dominant_repo: str | None = None
    provenance: ArchiveInsightProvenance
    thread: ThreadPayload

    @classmethod
    def from_record(cls, record: ThreadRecord) -> ThreadInsight:
        return cls(
            thread_id=record.thread_id,
            root_id=record.root_id,
            dominant_repo=record.dominant_repo,
            provenance=ArchiveInsightProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.end_time or record.start_time,
                source_sort_key=None,
            ),
            thread=record.payload,
        )


class SessionTagRollupInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "session_tag_rollup"
    tag: str
    session_count: int
    logical_session_count: int = 0
    explicit_count: int
    auto_count: int
    origin_breakdown: dict[str, int]
    repo_breakdown: dict[str, int]
    provenance: ArchiveInsightProvenance

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_provider_breakdown(cls, data: object) -> object:
        if isinstance(data, Mapping) and "origin_breakdown" not in data and "provider_breakdown" in data:
            updated = dict(data)
            updated["origin_breakdown"] = updated["provider_breakdown"]
            return updated
        return data

    @property
    def provider_breakdown(self) -> dict[str, int]:
        return self.origin_breakdown


class DaySessionSummaryInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "day_session_summary"
    date: str
    provenance: ArchiveInsightProvenance
    summary: DaySessionSummaryPayload


class WeekSessionSummaryInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "week_session_summary"
    iso_week: str
    provenance: ArchiveInsightProvenance
    summary: WeekSessionSummaryPayload


class ArchiveCoverageInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "archive_coverage"
    group_by: str = "provider"
    bucket: str = ""
    source_name: str | None = None
    session_count: int
    logical_session_count: int = 0
    message_count: int = 0
    user_message_count: int = 0
    authored_user_message_count: int = 0
    assistant_message_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_tool_active_duration_ms: int = 0
    total_wall_duration_ms: int = 0
    total_words: int = 0
    avg_messages_per_session: float = 0.0
    avg_user_words: float = 0.0
    avg_authored_user_words: float = 0.0
    avg_assistant_words: float = 0.0
    tool_use_count: int = 0
    thinking_count: int = 0
    total_sessions_with_tools: int = 0
    total_sessions_with_thinking: int = 0
    tool_use_percentage: float = 0.0
    thinking_percentage: float = 0.0
    work_event_breakdown: dict[str, int] = Field(default_factory=dict)
    repos_active: tuple[str, ...] = ()
    origin_breakdown: dict[str, int] = Field(default_factory=dict)
    provenance: ArchiveInsightProvenance | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_provider_breakdown(cls, data: object) -> object:
        if isinstance(data, Mapping) and "origin_breakdown" not in data and "provider_breakdown" in data:
            updated = dict(data)
            updated["origin_breakdown"] = updated["provider_breakdown"]
            return updated
        return data

    @property
    def provider_breakdown(self) -> dict[str, int]:
        return self.origin_breakdown


class SessionCostInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "session_cost"
    semantic_tier: str = "estimate"
    session_id: str
    source_name: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    estimate: CostEstimatePayload
    provenance: ArchiveInsightProvenance


class CostRollupInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "cost_rollup"
    semantic_tier: str = "estimate"
    source_name: str
    model_name: str | None = None
    normalized_model: str | None = None
    session_count: int = 0
    priced_session_count: int = 0
    unavailable_session_count: int = 0
    status_counts: dict[str, int]
    # ``total_usd`` is the legacy summary draw: provider_reported_usd when
    # any exact totals were aggregated, else catalog_priced_usd. Consumers
    # that need a specific basis should read ``basis`` directly (#1136).
    total_usd: float = 0.0
    basis: CostBasisPayload = CostBasisPayload()
    unavailable_reason_counts: dict[str, int] = {}
    per_model_breakdown: tuple[CostModelBreakdown, ...] = ()
    usage: CostUsagePayload
    confidence: float = 0.0
    provenance: ArchiveInsightProvenance


class ArchiveDebtInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "archive_debt"
    debt_name: str
    category: str
    maintenance_target: str
    destructive: bool
    issue_count: int
    healthy: bool
    detail: str

    @classmethod
    def from_status(cls, status: ArchiveDebtStatus) -> ArchiveDebtInsight:
        return cls(
            debt_name=status.name,
            category=status.category.value,
            maintenance_target=status.maintenance_target,
            destructive=status.destructive,
            issue_count=status.issue_count,
            healthy=status.healthy,
            detail=status.detail,
        )


def profile_bucket_day(profile: SessionProfile) -> date | None:
    if profile.canonical_session_date is not None:
        return profile.canonical_session_date
    timestamp = profile.first_message_at or profile.created_at or profile.updated_at or profile.last_message_at
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        # Ensure timezone-awareness: assume UTC for naive datetimes
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.date()
    return timestamp


def profile_timestamp_values(profile: SessionProfile) -> tuple[list[str], list[float]]:
    timestamps = [
        timestamp
        for timestamp in (
            profile.updated_at,
            profile.last_message_at,
            profile.first_message_at,
            profile.created_at,
        )
        if timestamp is not None
    ]
    return (
        [timestamp.isoformat() for timestamp in timestamps],
        [timestamp.timestamp() for timestamp in timestamps],
    )


def _parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a timezone-aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def records_provenance(
    rows: Iterable[object],
    *,
    materialized_at_attr: str = "materialized_at",
    source_updated_at_attr: str = "source_updated_at",
    source_sort_key_attr: str = "source_sort_key",
) -> ArchiveInsightProvenance:
    row_list = list(rows)
    materialized_at_values = [
        _parse_iso_timestamp(str(getattr(row, materialized_at_attr)))
        for row in row_list
        if getattr(row, materialized_at_attr, None)
    ]
    materialized_at = max(materialized_at_values).isoformat() if materialized_at_values else "1970-01-01T00:00:00+00:00"
    source_updated_at_values = [
        _parse_iso_timestamp(str(getattr(row, source_updated_at_attr)))
        for row in row_list
        if getattr(row, source_updated_at_attr, None)
    ]
    source_updated_at = max(source_updated_at_values).isoformat() if source_updated_at_values else None
    source_sort_key = max(
        (
            float(getattr(row, source_sort_key_attr))
            for row in row_list
            if getattr(row, source_sort_key_attr, None) is not None
        ),
        default=None,
    )
    return ArchiveInsightProvenance(
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at=materialized_at,
        source_updated_at=source_updated_at,
        source_sort_key=source_sort_key,
    )


def day_after(iso_day: str) -> str:
    return (date_from_iso(iso_day) + timedelta(days=1)).isoformat()


def date_from_iso(value: str) -> date:
    return date.fromisoformat(value)


__all__ = [
    "ARCHIVE_INSIGHT_CONTRACT_VERSION",
    "ArchiveCoverageInsight",
    "ArchiveCoverageInsightQuery",
    "ArchiveDebtInsight",
    "ArchiveDebtInsightQuery",
    "ArchiveEnrichmentProvenance",
    "ArchiveInferenceProvenance",
    "ArchiveInsightModel",
    "ArchiveInsightProvenance",
    "CostBasisPayload",
    "CostModelBreakdown",
    "CostRollupInsight",
    "CostRollupInsightQuery",
    "ArchiveInsightUnavailableError",
    "DaySessionSummaryInsight",
    "SessionCostInsight",
    "SessionCostInsightQuery",
    "SessionEnrichmentPayload",
    "SessionEvidencePayload",
    "SessionInferencePayload",
    "SessionLatencyProfileInsight",
    "SessionLatencyProfileInsightQuery",
    "SessionLatencyProfilePayload",
    "SessionPhaseEvidencePayload",
    "SessionPhaseInferencePayload",
    "SessionPhaseInsight",
    "SessionPhaseInsightQuery",
    "SessionProfileInsight",
    "SessionProfileInsightQuery",
    "SessionTagRollupInsight",
    "SessionTagRollupQuery",
    "SessionWorkEventInsight",
    "SessionWorkEventInsightQuery",
    "WeekSessionSummaryInsight",
    "WorkEventEvidencePayload",
    "WorkEventInferencePayload",
    "ThreadInsight",
    "ThreadInsightQuery",
    "date_from_iso",
    "day_after",
    "profile_bucket_day",
    "profile_timestamp_values",
    "records_provenance",
]
