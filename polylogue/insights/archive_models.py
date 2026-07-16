"""Shared archive insight base and typed payload contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.archive.session.documents import SessionPhaseDocument, WorkEventDocument
from polylogue.core.sources import source_name_to_origin
from polylogue.insights.confidence import ConfidenceBand
from polylogue.insights.fallback import FallbackReason
from polylogue.insights.temporal_source import TimeConfidence

ARCHIVE_INSIGHT_CONTRACT_VERSION = 10


class ArchiveInsightModel(BaseModel):
    """Shared base for public archive insight payloads."""

    # Materialized insight records are sparse across materializer versions;
    # readers ignore unknown payload keys and consume the typed fields below.
    model_config = ConfigDict(extra="ignore", frozen=True, protected_namespaces=())

    @field_validator("origin", mode="before", check_fields=False)
    @classmethod
    def normalize_origin(cls, value: object) -> str | None:
        if value is None:
            return None
        return source_name_to_origin(value)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class ArchiveInsightProvenance(ArchiveInsightModel):
    materializer_version: int
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None
    time_confidence: TimeConfidence = "unknown"


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
    session_timestamp: str | None = None
    timestamp_source: str = "provider_supplied"
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
    tool_active_duration_ms: int = 0
    workflow_shape_features: dict[str, object] = Field(default_factory=dict)
    terminal_state_evidence: dict[str, object] = Field(default_factory=dict)
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
    logical_session_id: str | None = None
    thinking_duration_ms: int = 0
    output_duration_ms: int = 0
    tool_duration_ms: int = 0
    latency_percentiles_ms: dict[str, int] = Field(default_factory=dict)
    tool_calls_per_minute: float = 0.0
    timing_provenance: str = "sort_key_estimated"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_credit_cost: float = 0.0
    cost_provenance: str = "unknown"


class SessionInferencePayload(ArchiveInsightModel):
    inferred_topic: str | None = None
    inferred_topic_source: str = "absent"
    repo_names: tuple[str, ...] = ()
    work_event_count: int = 0
    phase_count: int = 0
    engaged_duration_ms: int = 0
    engaged_minutes: float = 0.0
    tool_active_duration_ms: int = 0
    tool_active_minutes: float = 0.0
    workflow_shape: str = "unknown"
    workflow_shape_confidence: float = 0.0
    terminal_state: str = "unknown"
    terminal_state_confidence: float = 0.0
    support_level: ConfidenceBand = ConfidenceBand.WEAK
    support_signals: tuple[str, ...] = ()
    engaged_duration_source: str = "session_total_fallback"
    repo_inference_strength: ConfidenceBand = ConfidenceBand.WEAK
    auto_tags: tuple[str, ...] = ()
    work_events: tuple[WorkEventDocument, ...] = ()
    phases: tuple[SessionPhaseDocument, ...] = ()
    fallback_reasons: tuple[FallbackReason, ...] = ()

    @field_validator("work_events", mode="before")
    @classmethod
    def _normalize_work_event_documents(cls, value: object) -> object:
        return _normalize_timed_documents(value)

    @field_validator("phases", mode="before")
    @classmethod
    def _normalize_phase_documents(cls, value: object) -> object:
        return _normalize_timed_documents(value)


class SessionLatencyProfilePayload(ArchiveInsightModel):
    median_tool_call_ms: int = 0
    p90_tool_call_ms: int = 0
    max_tool_call_ms: int = 0
    stuck_tool_count: int = 0
    median_agent_response_ms: int = 0
    median_user_response_ms: int = 0
    tool_call_count_by_category: dict[str, int] = Field(default_factory=dict)
    construct_boundary: str = (
        "agent-response time includes both model output delay and any intervening tool execution; "
        "provider tool latency requires timestamped session-event pairs"
    )


class WorkEventEvidencePayload(ArchiveInsightModel):
    start_index: int
    end_index: int
    start_time: str | None = None
    end_time: str | None = None
    canonical_session_date: str | None = None
    timing_provenance: str = "untimestamped"
    date_provenance: str = "none"
    duration_ms: int = 0
    file_paths: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()


class WorkEventInferencePayload(ArchiveInsightModel):
    heuristic_label: str
    summary: str
    confidence: float
    evidence: tuple[str, ...] = ()
    support_level: ConfidenceBand = ConfidenceBand.WEAK
    support_signals: tuple[str, ...] = ()
    fallback_inference: bool = False
    fallback_reasons: tuple[FallbackReason, ...] = ()


class SessionPhaseEvidencePayload(ArchiveInsightModel):
    start_time: str | None = None
    end_time: str | None = None
    canonical_session_date: str | None = None
    timing_provenance: str = "untimestamped"
    date_provenance: str = "none"
    message_range: tuple[int, int] = (0, 0)
    duration_ms: int = 0
    phase_idle_threshold_ms: int = 300_000
    tool_counts: dict[str, int] = Field(default_factory=dict)
    word_count: int = 0


class SessionPhaseInferencePayload(ArchiveInsightModel):
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    support_level: ConfidenceBand = ConfidenceBand.WEAK
    support_signals: tuple[str, ...] = ()
    fallback_inference: bool = False
    fallback_reasons: tuple[FallbackReason, ...] = ()


class SessionEnrichmentPayload(ArchiveInsightModel):
    intent_summary: str | None = None
    outcome_summary: str | None = None
    blockers: tuple[str, ...] = ()
    confidence: float = 0.0
    support_level: ConfidenceBand = ConfidenceBand.WEAK
    support_signals: tuple[str, ...] = ()
    input_band_summary: dict[str, int] = Field(default_factory=dict)
    fallback_reasons: tuple[FallbackReason, ...] = ()
    # #1687: goal-driven session detection from /goal command in first user message.
    is_goal_session: bool = False
    goal_text: str | None = None
    # Boundary-derived posture, not task-success judgment.
    goal_outcome: str | None = None


def _normalize_timed_documents(value: object) -> object:
    if not isinstance(value, list | tuple):
        return value
    return tuple(_normalize_timed_document(item) for item in value)


def _normalize_timed_document(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    document: dict[str, Any] = dict(value)
    start_time = _optional_str(document.get("start_time"))
    end_time = _optional_str(document.get("end_time"))
    canonical_session_date = _optional_str(document.get("canonical_session_date"))
    document.setdefault("timing_provenance", _range_timing_provenance(start_time, end_time))
    document.setdefault("date_provenance", _date_provenance(canonical_session_date, start_time, end_time))
    return document


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _range_timing_provenance(start_time: str | None, end_time: str | None) -> str:
    if start_time is not None and end_time is not None:
        return "timestamped_range"
    if start_time is not None:
        return "start_timestamp_only"
    if end_time is not None:
        return "end_timestamp_only"
    return "untimestamped"


def _date_provenance(canonical_session_date: str | None, start_time: str | None, end_time: str | None) -> str:
    if canonical_session_date is None:
        return "none"
    if start_time is not None or end_time is not None:
        return "event_timestamp"
    return "date_only"


class ThreadMemberEvidencePayload(ArchiveInsightModel):
    session_id: str
    parent_id: str | None = None
    role: str
    depth: int = 0
    confidence: float = 0.0
    support_signals: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class ThreadPayload(ArchiveInsightModel):
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
    origin_breakdown: dict[str, int] = Field(default_factory=dict)
    work_event_breakdown: dict[str, int] = Field(default_factory=dict)
    confidence: float = 0.0
    support_level: ConfidenceBand = ConfidenceBand.WEAK
    support_signals: tuple[str, ...] = ()
    member_evidence: tuple[ThreadMemberEvidencePayload, ...] = ()


class DaySessionSummaryPayload(ArchiveInsightModel):
    date: str
    session_count: int = 0
    logical_session_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_tool_active_duration_ms: int = 0
    total_wall_duration_ms: int = 0
    total_messages: int = 0
    total_words: int = 0
    work_event_breakdown: dict[str, int] = Field(default_factory=dict)
    repos_active: tuple[str, ...] = ()
    origins: dict[str, int] = Field(default_factory=dict)


class WeekSessionSummaryPayload(ArchiveInsightModel):
    iso_week: str
    day_summaries: tuple[DaySessionSummaryPayload, ...] = ()
    session_count: int = 0
    logical_session_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_tool_active_duration_ms: int = 0
    total_messages: int = 0


__all__ = [
    "ARCHIVE_INSIGHT_CONTRACT_VERSION",
    "ArchiveEnrichmentProvenance",
    "ArchiveInferenceProvenance",
    "ArchiveInsightModel",
    "ArchiveInsightProvenance",
    "DaySessionSummaryPayload",
    "FallbackReason",
    "SessionEnrichmentPayload",
    "SessionEvidencePayload",
    "SessionInferencePayload",
    "SessionPhaseEvidencePayload",
    "SessionPhaseInferencePayload",
    "WeekSessionSummaryPayload",
    "WorkEventEvidencePayload",
    "WorkEventInferencePayload",
    "ThreadPayload",
    "ThreadMemberEvidencePayload",
]
