"""Versioned archive data product contracts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Protocol

from polylogue.archive.semantic.pricing import CostEstimatePayload, CostUsagePayload
from polylogue.archive.session.session_profile import SessionProfile
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
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    WeekSessionSummaryPayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
    WorkThreadPayload,
)
from polylogue.storage.repair import ArchiveDebtStatus
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION

if TYPE_CHECKING:
    from polylogue.storage.runtime import (
        SessionPhaseRecord,
        SessionProfileRecord,
        SessionWorkEventRecord,
        WorkThreadRecord,
    )


class ArchiveInsightUnavailableError(RuntimeError):
    """Raised when a durable archive-product surface is not ready to read."""


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


class ConversationTimelineWindowInsightQuery(ProviderTimeWindowInsightQuery):
    conversation_id: str | None = None
    kind: str | None = None


class SearchableConversationTimelineInsightQuery(ConversationTimelineWindowInsightQuery):
    query: str | None = None

    @property
    def wants_search(self) -> bool:
        return bool(self.query)


class SessionProfileInsightQuery(SessionWindowInsightQuery):
    tier: str = "merged"


class SessionEnrichmentInsightQuery(SessionWindowInsightQuery):
    pass


class SessionWorkEventInsightQuery(SearchableConversationTimelineInsightQuery):
    pass


class SessionPhaseInsightQuery(ConversationTimelineWindowInsightQuery):
    pass


class WorkThreadInsightQuery(SearchableTimeWindowInsightQuery):
    pass


class SessionTagRollupQuery(ProviderSearchInsightQuery):
    limit: int | None = 100


class DaySessionSummaryInsightQuery(ProviderTimeWindowInsightQuery):
    limit: int | None = 90


class WeekSessionSummaryInsightQuery(ProviderTimeWindowInsightQuery):
    limit: int | None = 52


class ProviderAnalyticsInsightQuery(PaginatedInsightQuery):
    provider: str | None = None
    limit: int | None = None


class SessionCostInsightQuery(ProviderTimeWindowInsightQuery):
    conversation_id: str | None = None
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
    product_kind: str = "session_profile"
    semantic_tier: str = "merged"
    conversation_id: str
    provider_name: str
    title: str | None = None
    provenance: ArchiveInsightProvenance
    evidence: SessionEvidencePayload | None = None
    inference_provenance: ArchiveInferenceProvenance | None = None
    inference: SessionInferencePayload | None = None

    @classmethod
    def from_record(
        cls,
        record: SessionProfileRecord,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight:
        include_evidence = tier in {"merged", "evidence"}
        include_inference = tier in {"merged", "inference"}
        return cls(
            semantic_tier=tier,
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            title=record.title,
            provenance=_record_provenance(record),
            evidence=(record.evidence_payload if include_evidence else None),
            inference_provenance=(_record_inference_provenance(record) if include_inference else None),
            inference=(record.inference_payload if include_inference else None),
        )


class SessionEnrichmentInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "session_enrichment"
    semantic_tier: str = "enrichment"
    conversation_id: str
    provider_name: str
    title: str | None = None
    provenance: ArchiveInsightProvenance
    enrichment_provenance: ArchiveEnrichmentProvenance
    enrichment: SessionEnrichmentPayload

    @classmethod
    def from_record(cls, record: SessionProfileRecord) -> SessionEnrichmentInsight:
        return cls(
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            title=record.title,
            provenance=_record_provenance(record),
            enrichment_provenance=_record_enrichment_provenance(record),
            enrichment=record.enrichment_payload,
        )


class SessionWorkEventInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "session_work_event"
    semantic_tier: str = "inference"
    event_id: str
    conversation_id: str
    provider_name: str
    event_index: int
    provenance: ArchiveInsightProvenance
    inference_provenance: ArchiveInferenceProvenance
    evidence: WorkEventEvidencePayload
    inference: WorkEventInferencePayload

    @classmethod
    def from_record(cls, record: SessionWorkEventRecord) -> SessionWorkEventInsight:
        return cls(
            event_id=record.event_id,
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            event_index=record.event_index,
            provenance=_record_provenance(record),
            inference_provenance=_record_inference_provenance(record),
            evidence=record.evidence_payload,
            inference=record.inference_payload,
        )


class SessionPhaseInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "session_phase"
    semantic_tier: str = "inference"
    phase_id: str
    conversation_id: str
    provider_name: str
    phase_index: int
    provenance: ArchiveInsightProvenance
    inference_provenance: ArchiveInferenceProvenance
    evidence: SessionPhaseEvidencePayload
    inference: SessionPhaseInferencePayload

    @classmethod
    def from_record(cls, record: SessionPhaseRecord) -> SessionPhaseInsight:
        return cls(
            phase_id=record.phase_id,
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            phase_index=record.phase_index,
            provenance=_record_provenance(record),
            inference_provenance=_record_inference_provenance(record),
            evidence=record.evidence_payload,
            inference=record.inference_payload,
        )


class WorkThreadInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "work_thread"
    thread_id: str
    root_id: str
    dominant_repo: str | None = None
    provenance: ArchiveInsightProvenance
    thread: WorkThreadPayload

    @classmethod
    def from_record(cls, record: WorkThreadRecord) -> WorkThreadInsight:
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
    product_kind: str = "session_tag_rollup"
    tag: str
    conversation_count: int
    explicit_count: int
    auto_count: int
    provider_breakdown: dict[str, int]
    repo_breakdown: dict[str, int]
    provenance: ArchiveInsightProvenance


class DaySessionSummaryInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "day_session_summary"
    date: str
    provenance: ArchiveInsightProvenance
    summary: DaySessionSummaryPayload


class WeekSessionSummaryInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "week_session_summary"
    iso_week: str
    provenance: ArchiveInsightProvenance
    summary: WeekSessionSummaryPayload


class ProviderAnalyticsInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "provider_analytics"
    provider_name: str
    conversation_count: int
    message_count: int
    user_message_count: int
    assistant_message_count: int
    avg_messages_per_conversation: float
    avg_user_words: float
    avg_assistant_words: float
    tool_use_count: int
    thinking_count: int
    total_conversations_with_tools: int
    total_conversations_with_thinking: int
    tool_use_percentage: float
    thinking_percentage: float


class SessionCostInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "session_cost"
    semantic_tier: str = "estimate"
    conversation_id: str
    provider_name: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    estimate: CostEstimatePayload
    provenance: ArchiveInsightProvenance


class CostRollupInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "cost_rollup"
    semantic_tier: str = "estimate"
    provider_name: str
    model_name: str | None = None
    normalized_model: str | None = None
    session_count: int = 0
    priced_session_count: int = 0
    unavailable_session_count: int = 0
    status_counts: dict[str, int]
    total_usd: float = 0.0
    usage: CostUsagePayload
    confidence: float = 0.0
    provenance: ArchiveInsightProvenance


class ArchiveDebtInsight(ArchiveInsightModel):
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    product_kind: str = "archive_debt"
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
    "ArchiveDebtInsight",
    "ArchiveDebtInsightQuery",
    "ArchiveEnrichmentProvenance",
    "ArchiveInferenceProvenance",
    "ArchiveInsightModel",
    "ArchiveInsightProvenance",
    "CostRollupInsight",
    "CostRollupInsightQuery",
    "ArchiveInsightUnavailableError",
    "DaySessionSummaryInsight",
    "DaySessionSummaryInsightQuery",
    "ProviderAnalyticsInsight",
    "ProviderAnalyticsInsightQuery",
    "SessionCostInsight",
    "SessionCostInsightQuery",
    "SessionEnrichmentPayload",
    "SessionEnrichmentInsight",
    "SessionEnrichmentInsightQuery",
    "SessionEvidencePayload",
    "SessionInferencePayload",
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
    "WeekSessionSummaryInsightQuery",
    "WorkEventEvidencePayload",
    "WorkEventInferencePayload",
    "WorkThreadInsight",
    "WorkThreadInsightQuery",
    "date_from_iso",
    "day_after",
    "profile_bucket_day",
    "profile_timestamp_values",
    "records_provenance",
]
