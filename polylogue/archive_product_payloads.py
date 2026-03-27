"""Archive-product evidence, inference, enrichment, and lineage payloads."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from polylogue.archive_product_base import ArchiveProductModel


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


__all__ = [
    "ArchiveDebtTargetLineage",
    "SessionEnrichmentPayload",
    "SessionEvidencePayload",
    "SessionInferencePayload",
    "SessionPhaseEvidencePayload",
    "SessionPhaseInferencePayload",
    "WorkEventEvidencePayload",
    "WorkEventInferencePayload",
]
