"""Versioned archive data product contracts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

from polylogue.archive_product_models import (
    ARCHIVE_PRODUCT_CONTRACT_VERSION,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveProductModel,
    ArchiveProductProvenance,
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
from polylogue.lib.pricing import CostEstimatePayload, CostUsagePayload
from polylogue.lib.session_profile import SessionProfile
from polylogue.storage.repair import ArchiveDebtStatus
from polylogue.storage.store import (
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


class ArchiveProductUnavailableError(RuntimeError):
    """Raised when a durable archive-product surface is not ready to read."""


class PaginatedProductQuery(ArchiveProductModel):
    limit: int | None = 50
    offset: int = 0


class TimeWindowProductQuery(PaginatedProductQuery):
    since: str | None = None
    until: str | None = None


class SearchableTimeWindowProductQuery(TimeWindowProductQuery):
    query: str | None = None

    @property
    def wants_search(self) -> bool:
        return bool(self.query)


class ProviderTimeWindowProductQuery(TimeWindowProductQuery):
    provider: str | None = None


class ProviderSearchProductQuery(SearchableTimeWindowProductQuery):
    provider: str | None = None


class SessionWindowProductQuery(ProviderSearchProductQuery):
    first_message_since: str | None = None
    first_message_until: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None
    min_wallclock_seconds: int | None = None
    max_wallclock_seconds: int | None = None
    sort: str = "source"


class ConversationTimelineWindowProductQuery(ProviderTimeWindowProductQuery):
    conversation_id: str | None = None
    kind: str | None = None


class SearchableConversationTimelineProductQuery(ConversationTimelineWindowProductQuery):
    query: str | None = None

    @property
    def wants_search(self) -> bool:
        return bool(self.query)


class SessionProfileProductQuery(SessionWindowProductQuery):
    tier: str = "merged"


class SessionEnrichmentProductQuery(SessionWindowProductQuery):
    pass


class SessionWorkEventProductQuery(SearchableConversationTimelineProductQuery):
    pass


class SessionPhaseProductQuery(ConversationTimelineWindowProductQuery):
    pass


class WorkThreadProductQuery(SearchableTimeWindowProductQuery):
    pass


class SessionTagRollupQuery(ProviderSearchProductQuery):
    limit: int | None = 100


class DaySessionSummaryProductQuery(ProviderTimeWindowProductQuery):
    limit: int | None = 90


class WeekSessionSummaryProductQuery(ProviderTimeWindowProductQuery):
    limit: int | None = 52


class ProviderAnalyticsProductQuery(PaginatedProductQuery):
    provider: str | None = None
    limit: int | None = None


class SessionCostProductQuery(ProviderTimeWindowProductQuery):
    conversation_id: str | None = None
    model: str | None = None
    status: str | None = None


class CostRollupProductQuery(ProviderTimeWindowProductQuery):
    model: str | None = None
    limit: int | None = None


class ArchiveDebtProductQuery(PaginatedProductQuery):
    category: str | None = None
    only_actionable: bool = False
    limit: int | None = None


class _ProductRecordWithProvenance(Protocol):
    materializer_version: int
    materialized_at: str
    source_updated_at: str | None
    source_sort_key: float | None


class _ProductRecordWithInference(_ProductRecordWithProvenance, Protocol):
    inference_version: int
    inference_family: str


class _ProductRecordWithEnrichment(_ProductRecordWithProvenance, Protocol):
    enrichment_version: int
    enrichment_family: str


def _record_provenance(record: _ProductRecordWithProvenance) -> ArchiveProductProvenance:
    return ArchiveProductProvenance(
        materializer_version=record.materializer_version,
        materialized_at=record.materialized_at,
        source_updated_at=record.source_updated_at,
        source_sort_key=record.source_sort_key,
    )


def _record_inference_provenance(record: _ProductRecordWithInference) -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=record.materializer_version,
        materialized_at=record.materialized_at,
        source_updated_at=record.source_updated_at,
        source_sort_key=record.source_sort_key,
        inference_version=record.inference_version,
        inference_family=record.inference_family,
    )


def _record_enrichment_provenance(record: _ProductRecordWithEnrichment) -> ArchiveEnrichmentProvenance:
    return ArchiveEnrichmentProvenance(
        materializer_version=record.materializer_version,
        materialized_at=record.materialized_at,
        source_updated_at=record.source_updated_at,
        source_sort_key=record.source_sort_key,
        enrichment_version=record.enrichment_version,
        enrichment_family=record.enrichment_family,
    )


class SessionProfileProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_profile"
    semantic_tier: str = "merged"
    conversation_id: str
    provider_name: str
    title: str | None = None
    provenance: ArchiveProductProvenance
    evidence: SessionEvidencePayload | None = None
    inference_provenance: ArchiveInferenceProvenance | None = None
    inference: SessionInferencePayload | None = None

    @classmethod
    def from_record(
        cls,
        record: SessionProfileRecord,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct:
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


class SessionEnrichmentProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_enrichment"
    semantic_tier: str = "enrichment"
    conversation_id: str
    provider_name: str
    title: str | None = None
    provenance: ArchiveProductProvenance
    enrichment_provenance: ArchiveEnrichmentProvenance
    enrichment: SessionEnrichmentPayload

    @classmethod
    def from_record(cls, record: SessionProfileRecord) -> SessionEnrichmentProduct:
        return cls(
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            title=record.title,
            provenance=_record_provenance(record),
            enrichment_provenance=_record_enrichment_provenance(record),
            enrichment=record.enrichment_payload,
        )


class SessionWorkEventProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_work_event"
    semantic_tier: str = "inference"
    event_id: str
    conversation_id: str
    provider_name: str
    event_index: int
    provenance: ArchiveProductProvenance
    inference_provenance: ArchiveInferenceProvenance
    evidence: WorkEventEvidencePayload
    inference: WorkEventInferencePayload

    @classmethod
    def from_record(cls, record: SessionWorkEventRecord) -> SessionWorkEventProduct:
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


class SessionPhaseProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_phase"
    semantic_tier: str = "inference"
    phase_id: str
    conversation_id: str
    provider_name: str
    phase_index: int
    provenance: ArchiveProductProvenance
    inference_provenance: ArchiveInferenceProvenance
    evidence: SessionPhaseEvidencePayload
    inference: SessionPhaseInferencePayload

    @classmethod
    def from_record(cls, record: SessionPhaseRecord) -> SessionPhaseProduct:
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


class WorkThreadProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "work_thread"
    thread_id: str
    root_id: str
    dominant_repo: str | None = None
    provenance: ArchiveProductProvenance
    thread: WorkThreadPayload

    @classmethod
    def from_record(cls, record: WorkThreadRecord) -> WorkThreadProduct:
        return cls(
            thread_id=record.thread_id,
            root_id=record.root_id,
            dominant_repo=record.dominant_repo,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.end_time or record.start_time,
                source_sort_key=None,
            ),
            thread=record.payload,
        )


class SessionTagRollupProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_tag_rollup"
    tag: str
    conversation_count: int
    explicit_count: int
    auto_count: int
    provider_breakdown: dict[str, int]
    repo_breakdown: dict[str, int]
    provenance: ArchiveProductProvenance


class DaySessionSummaryProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "day_session_summary"
    date: str
    provenance: ArchiveProductProvenance
    summary: DaySessionSummaryPayload


class WeekSessionSummaryProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "week_session_summary"
    iso_week: str
    provenance: ArchiveProductProvenance
    summary: WeekSessionSummaryPayload


class ProviderAnalyticsProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
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


class SessionCostProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_cost"
    semantic_tier: str = "estimate"
    conversation_id: str
    provider_name: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    estimate: CostEstimatePayload
    provenance: ArchiveProductProvenance


class CostRollupProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
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
    provenance: ArchiveProductProvenance


class ArchiveDebtProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "archive_debt"
    debt_name: str
    category: str
    maintenance_target: str
    destructive: bool
    issue_count: int
    healthy: bool
    detail: str

    @classmethod
    def from_status(cls, status: ArchiveDebtStatus) -> ArchiveDebtProduct:
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
) -> ArchiveProductProvenance:
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
    return ArchiveProductProvenance(
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        materialized_at=materialized_at,
        source_updated_at=source_updated_at,
        source_sort_key=source_sort_key,
    )


def day_after(iso_day: str) -> str:
    return (date_from_iso(iso_day) + timedelta(days=1)).isoformat()


def date_from_iso(value: str) -> date:
    return date.fromisoformat(value)


__all__ = [
    "ARCHIVE_PRODUCT_CONTRACT_VERSION",
    "ArchiveDebtProduct",
    "ArchiveDebtProductQuery",
    "ArchiveEnrichmentProvenance",
    "ArchiveInferenceProvenance",
    "ArchiveProductModel",
    "ArchiveProductProvenance",
    "CostRollupProduct",
    "CostRollupProductQuery",
    "ArchiveProductUnavailableError",
    "DaySessionSummaryProduct",
    "DaySessionSummaryProductQuery",
    "ProviderAnalyticsProduct",
    "ProviderAnalyticsProductQuery",
    "SessionCostProduct",
    "SessionCostProductQuery",
    "SessionEnrichmentPayload",
    "SessionEnrichmentProduct",
    "SessionEnrichmentProductQuery",
    "SessionEvidencePayload",
    "SessionInferencePayload",
    "SessionPhaseEvidencePayload",
    "SessionPhaseInferencePayload",
    "SessionPhaseProduct",
    "SessionPhaseProductQuery",
    "SessionProfileProduct",
    "SessionProfileProductQuery",
    "SessionTagRollupProduct",
    "SessionTagRollupQuery",
    "SessionWorkEventProduct",
    "SessionWorkEventProductQuery",
    "WeekSessionSummaryProduct",
    "WeekSessionSummaryProductQuery",
    "WorkEventEvidencePayload",
    "WorkEventInferencePayload",
    "WorkThreadProduct",
    "WorkThreadProductQuery",
    "date_from_iso",
    "day_after",
    "profile_bucket_day",
    "profile_timestamp_values",
    "records_provenance",
]
