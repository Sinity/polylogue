"""Shared archive product base and typed payload contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from polylogue.archive.session.documents import SessionPhaseDocument, WorkEventDocument

ARCHIVE_INSIGHT_CONTRACT_VERSION = 5


class ArchiveInsightModel(BaseModel):
    """Shared base for public archive data product payloads."""

    # extra="ignore" tolerates legacy fields from older materialized records
    # (e.g. primary_work_kind, decisions removed in the March 2026 cleanup)
    model_config = ConfigDict(extra="ignore", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class ArchiveInsightProvenance(ArchiveInsightModel):
    materializer_version: int
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None


class ArchiveInferenceProvenance(ArchiveInsightProvenance):
    inference_version: int
    inference_family: str


class ArchiveEnrichmentProvenance(ArchiveInsightProvenance):
    enrichment_version: int
    enrichment_family: str


class SessionEvidencePayload(ArchiveInsightModel):
    created_at: str | None = None
    updated_at: str | None = None
    first_message_at: str | None = None
    last_message_at: str | None = None
    timestamped_message_count: int = 0
    untimestamped_message_count: int = 0
    timestamp_coverage: str = "none"
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
    compaction_count: int = 0
    has_compaction: bool = False
    tool_categories: dict[str, int] = Field(default_factory=dict)
    repo_paths: tuple[str, ...] = ()
    cwd_paths: tuple[str, ...] = ()
    branch_names: tuple[str, ...] = ()
    file_paths_touched: tuple[str, ...] = ()
    languages_detected: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    is_continuation: bool = False
    parent_id: str | None = None


class SessionInferencePayload(ArchiveInsightModel):
    repo_names: tuple[str, ...] = ()
    work_event_count: int = 0
    phase_count: int = 0
    engaged_duration_ms: int = 0
    engaged_minutes: float = 0.0
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    engaged_duration_source: str = "session_total_fallback"
    repo_inference_strength: str = "weak"
    auto_tags: tuple[str, ...] = ()
    work_events: tuple[WorkEventDocument, ...] = ()
    phases: tuple[SessionPhaseDocument, ...] = ()


class WorkEventEvidencePayload(ArchiveInsightModel):
    start_index: int
    end_index: int
    start_time: str | None = None
    end_time: str | None = None
    canonical_session_date: str | None = None
    duration_ms: int = 0
    file_paths: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()


class WorkEventInferencePayload(ArchiveInsightModel):
    kind: str
    summary: str
    confidence: float
    evidence: tuple[str, ...] = ()
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    fallback_inference: bool = False


class SessionPhaseEvidencePayload(ArchiveInsightModel):
    start_time: str | None = None
    end_time: str | None = None
    canonical_session_date: str | None = None
    message_range: tuple[int, int] = (0, 0)
    duration_ms: int = 0
    tool_counts: dict[str, int] = Field(default_factory=dict)
    word_count: int = 0


class SessionPhaseInferencePayload(ArchiveInsightModel):
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    fallback_inference: bool = False


class SessionEnrichmentPayload(ArchiveInsightModel):
    intent_summary: str | None = None
    outcome_summary: str | None = None
    blockers: tuple[str, ...] = ()
    confidence: float = 0.0
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    input_band_summary: dict[str, int] = Field(default_factory=dict)


class WorkThreadMemberEvidencePayload(ArchiveInsightModel):
    conversation_id: str
    parent_id: str | None = None
    role: str
    depth: int = 0
    confidence: float = 0.0
    support_signals: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class WorkThreadPayload(ArchiveInsightModel):
    start_time: str | None = None
    end_time: str | None = None
    dominant_repo: str | None = None
    session_ids: tuple[str, ...] = ()
    session_count: int = 0
    depth: int = 0
    branch_count: int = 0
    total_messages: int = 0
    total_cost_usd: float = 0.0
    wall_duration_ms: int = 0
    provider_breakdown: dict[str, int] = Field(default_factory=dict)
    work_event_breakdown: dict[str, int] = Field(default_factory=dict)
    confidence: float = 0.0
    support_level: str = "weak"
    support_signals: tuple[str, ...] = ()
    member_evidence: tuple[WorkThreadMemberEvidencePayload, ...] = ()


class DaySessionSummaryPayload(ArchiveInsightModel):
    date: str
    session_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_wall_duration_ms: int = 0
    total_messages: int = 0
    total_words: int = 0
    work_event_breakdown: dict[str, int] = Field(default_factory=dict)
    repos_active: tuple[str, ...] = ()
    providers: dict[str, int] = Field(default_factory=dict)


class WeekSessionSummaryPayload(ArchiveInsightModel):
    iso_week: str
    day_summaries: tuple[DaySessionSummaryPayload, ...] = ()
    session_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_messages: int = 0


__all__ = [
    "ARCHIVE_INSIGHT_CONTRACT_VERSION",
    "ArchiveEnrichmentProvenance",
    "ArchiveInferenceProvenance",
    "ArchiveInsightModel",
    "ArchiveInsightProvenance",
    "DaySessionSummaryPayload",
    "SessionEnrichmentPayload",
    "SessionEvidencePayload",
    "SessionInferencePayload",
    "SessionPhaseEvidencePayload",
    "SessionPhaseInferencePayload",
    "WeekSessionSummaryPayload",
    "WorkEventEvidencePayload",
    "WorkEventInferencePayload",
    "WorkThreadPayload",
    "WorkThreadMemberEvidencePayload",
]
