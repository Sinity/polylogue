"""Versioned archive data product contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polylogue.maintenance_models import ArchiveDebtStatus
from polylogue.storage.store import (
    MaintenanceRunRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)

ARCHIVE_PRODUCT_CONTRACT_VERSION = 4


class ArchiveProductModel(BaseModel):
    """Shared base for public archive data product payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class ArchiveProductProvenance(ArchiveProductModel):
    materializer_version: int
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None


class ArchiveInferenceProvenance(ArchiveProductProvenance):
    inference_version: int
    inference_family: str


class ArchiveEnrichmentProvenance(ArchiveProductProvenance):
    enrichment_version: int
    enrichment_family: str


class SessionEvidencePayload(ArchiveProductModel):
    created_at: str | None = None
    updated_at: str | None = None
    first_message_at: str | None = None
    last_message_at: str | None = None
    canonical_session_date: str | None = None
    message_count: int = 0
    substantive_count: int = 0
    attachment_count: int = 0
    tool_use_count: int = 0
    thinking_count: int = 0
    word_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    wall_duration_ms: int = 0
    cost_is_estimated: bool = False
    tool_categories: dict[str, int] = Field(default_factory=dict)
    repo_paths: tuple[str, ...] = ()
    cwd_paths: tuple[str, ...] = ()
    branch_names: tuple[str, ...] = ()
    file_paths_touched: tuple[str, ...] = ()
    languages_detected: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    is_continuation: bool = False
    parent_id: str | None = None


class SessionInferencePayload(ArchiveProductModel):
    primary_work_kind: str | None = None
    canonical_projects: tuple[str, ...] = ()
    work_event_count: int = 0
    phase_count: int = 0
    engaged_duration_ms: int = 0
    engaged_minutes: float = 0.0
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    engaged_duration_source: str = "session_total_fallback"
    project_inference_strength: str = "weak"
    decision_signal_strength: str = "weak"
    auto_tags: tuple[str, ...] = ()
    work_events: tuple[dict[str, Any], ...] = ()
    phases: tuple[dict[str, Any], ...] = ()
    decisions: tuple[dict[str, Any], ...] = ()


class WorkEventEvidencePayload(ArchiveProductModel):
    start_index: int
    end_index: int
    start_time: str | None = None
    end_time: str | None = None
    canonical_session_date: str | None = None
    duration_ms: int = 0
    file_paths: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()


class WorkEventInferencePayload(ArchiveProductModel):
    kind: str
    summary: str
    confidence: float
    evidence: tuple[str, ...] = ()
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    fallback_inference: bool = False


class SessionPhaseEvidencePayload(ArchiveProductModel):
    start_time: str | None = None
    end_time: str | None = None
    canonical_session_date: str | None = None
    message_range: tuple[int, int] = (0, 0)
    duration_ms: int = 0
    tool_counts: dict[str, int] = Field(default_factory=dict)
    word_count: int = 0


class SessionPhaseInferencePayload(ArchiveProductModel):
    kind: str
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    fallback_inference: bool = False


class SessionEnrichmentPayload(ArchiveProductModel):
    intent_summary: str | None = None
    outcome_summary: str | None = None
    blockers: tuple[str, ...] = ()
    refined_work_kind: str | None = None
    confidence: float = 0.0
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    input_band_summary: dict[str, int] = Field(default_factory=dict)


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
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            evidence=(
                SessionEvidencePayload.model_validate(record.evidence_payload)
                if include_evidence
                else None
            ),
            inference_provenance=(
                ArchiveInferenceProvenance(
                    materializer_version=record.materializer_version,
                    materialized_at=record.materialized_at,
                    source_updated_at=record.source_updated_at,
                    source_sort_key=record.source_sort_key,
                    inference_version=record.inference_version,
                    inference_family=record.inference_family,
                )
                if include_inference
                else None
            ),
            inference=(
                SessionInferencePayload.model_validate(record.inference_payload)
                if include_inference
                else None
            ),
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
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            inference_provenance=ArchiveInferenceProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
                inference_version=record.inference_version,
                inference_family=record.inference_family,
            ),
            evidence=WorkEventEvidencePayload.model_validate(record.evidence_payload),
            inference=WorkEventInferencePayload.model_validate(record.inference_payload),
        )


class WorkThreadProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "work_thread"
    thread_id: str
    root_id: str
    dominant_project: str | None = None
    provenance: ArchiveProductProvenance
    thread: dict[str, Any]

    @classmethod
    def from_record(cls, record: WorkThreadRecord) -> WorkThreadProduct:
        return cls(
            thread_id=record.thread_id,
            root_id=record.root_id,
            dominant_project=record.dominant_project,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.end_time or record.start_time,
                source_sort_key=None,
            ),
            thread=dict(record.payload),
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
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            inference_provenance=ArchiveInferenceProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
                inference_version=record.inference_version,
                inference_family=record.inference_family,
            ),
            evidence=SessionPhaseEvidencePayload.model_validate(record.evidence_payload),
            inference=SessionPhaseInferencePayload.model_validate(record.inference_payload),
        )


class SessionTagRollupProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_tag_rollup"
    tag: str
    conversation_count: int
    explicit_count: int
    auto_count: int
    provider_breakdown: dict[str, int]
    project_breakdown: dict[str, int]
    provenance: ArchiveProductProvenance


class DaySessionSummaryProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "day_session_summary"
    date: str
    provenance: ArchiveProductProvenance
    summary: dict[str, Any]


class WeekSessionSummaryProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "week_session_summary"
    iso_week: str
    provenance: ArchiveProductProvenance
    summary: dict[str, Any]


class MaintenanceRunProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "maintenance_run"
    maintenance_run_id: str
    executed_at: str
    mode: str
    preview: bool
    repair_selected: bool
    cleanup_selected: bool
    vacuum_requested: bool
    target_names: tuple[str, ...] = ()
    success: bool
    schema_version: int
    manifest: dict[str, Any]

    @classmethod
    def from_record(cls, record: MaintenanceRunRecord) -> MaintenanceRunProduct:
        return cls(
            maintenance_run_id=record.maintenance_run_id,
            executed_at=record.executed_at,
            mode=record.mode,
            preview=record.preview,
            repair_selected=record.repair_selected,
            cleanup_selected=record.cleanup_selected,
            vacuum_requested=record.vacuum_requested,
            target_names=record.target_names,
            success=record.success,
            schema_version=record.schema_version,
            manifest=dict(record.manifest),
        )


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


class ArchiveDebtTargetLineage(ArchiveProductModel):
    latest_run_at: str | None = None
    latest_mode: str | None = None
    latest_preview_at: str | None = None
    latest_preview_issue_count: int | None = None
    latest_apply_at: str | None = None
    latest_successful_apply_at: str | None = None
    latest_validation_at: str | None = None
    latest_validation_issue_count: int | None = None
    latest_successful_validation_at: str | None = None
    latest_regressed_at: str | None = None


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
    governance_stage: str
    lineage: ArchiveDebtTargetLineage | None = None

    @classmethod
    def from_status(
        cls,
        status: ArchiveDebtStatus,
        *,
        governance_stage: str,
        lineage: ArchiveDebtTargetLineage | None = None,
    ) -> ArchiveDebtProduct:
        return cls(
            debt_name=status.name,
            category=status.category.value,
            maintenance_target=status.maintenance_target,
            destructive=status.destructive,
            issue_count=status.issue_count,
            healthy=status.healthy,
            detail=status.detail,
            governance_stage=governance_stage,
            lineage=lineage,
        )


class SessionProfileProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    first_message_since: str | None = None
    first_message_until: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None
    tier: str = "merged"
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


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
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            enrichment_provenance=ArchiveEnrichmentProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
                enrichment_version=record.enrichment_version,
                enrichment_family=record.enrichment_family,
            ),
            enrichment=SessionEnrichmentPayload.model_validate(record.enrichment_payload),
        )


class SessionEnrichmentProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    first_message_since: str | None = None
    first_message_until: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None
    refined_work_kind: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class SessionWorkEventProductQuery(ArchiveProductModel):
    conversation_id: str | None = None
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    kind: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class SessionPhaseProductQuery(ArchiveProductModel):
    conversation_id: str | None = None
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    kind: str | None = None
    limit: int | None = 50
    offset: int = 0


class WorkThreadProductQuery(ArchiveProductModel):
    since: str | None = None
    until: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class MaintenanceRunProductQuery(ArchiveProductModel):
    limit: int = Field(default=20, ge=1)


class SessionTagRollupQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = 100
    offset: int = 0
    query: str | None = None


class DaySessionSummaryProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = 90
    offset: int = 0


class WeekSessionSummaryProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = 52
    offset: int = 0


class ProviderAnalyticsProductQuery(ArchiveProductModel):
    provider: str | None = None
    limit: int | None = None
    offset: int = 0


class ArchiveDebtProductQuery(ArchiveProductModel):
    category: str | None = None
    only_actionable: bool = False
    limit: int | None = None
    offset: int = 0


__all__ = [
    "ArchiveDebtProduct",
    "ArchiveDebtProductQuery",
    "ARCHIVE_PRODUCT_CONTRACT_VERSION",
    "ArchiveDebtTargetLineage",
    "ArchiveEnrichmentProvenance",
    "ArchiveProductModel",
    "ArchiveProductProvenance",
    "ArchiveInferenceProvenance",
    "DaySessionSummaryProduct",
    "DaySessionSummaryProductQuery",
    "MaintenanceRunProduct",
    "MaintenanceRunProductQuery",
    "ProviderAnalyticsProduct",
    "ProviderAnalyticsProductQuery",
    "SessionPhaseProduct",
    "SessionPhaseProductQuery",
    "SessionTagRollupProduct",
    "SessionTagRollupQuery",
    "SessionEvidencePayload",
    "SessionEnrichmentPayload",
    "SessionEnrichmentProduct",
    "SessionEnrichmentProductQuery",
    "SessionInferencePayload",
    "SessionProfileProduct",
    "SessionProfileProductQuery",
    "SessionPhaseEvidencePayload",
    "SessionPhaseInferencePayload",
    "SessionWorkEventProduct",
    "SessionWorkEventProductQuery",
    "WorkEventEvidencePayload",
    "WorkEventInferencePayload",
    "WeekSessionSummaryProduct",
    "WeekSessionSummaryProductQuery",
    "WorkThreadProduct",
    "WorkThreadProductQuery",
]
