"""Shared typed document payloads for session-derived runtime surfaces."""

from __future__ import annotations

from typing_extensions import NotRequired, TypedDict


class WorkEventDocument(TypedDict):
    heuristic_label: str
    start_index: int
    end_index: int
    start_time: str | None
    end_time: str | None
    canonical_session_date: str | None
    timing_provenance: str
    date_provenance: str
    duration_ms: int
    confidence: float
    evidence: list[str]
    file_paths: list[str]
    tools_used: list[str]
    summary: str


class SessionPhaseDocument(TypedDict):
    start_time: str | None
    end_time: str | None
    canonical_session_date: str | None
    timing_provenance: str
    date_provenance: str
    message_range: list[int]
    duration_ms: int
    phase_idle_threshold_ms: int
    tool_counts: dict[str, int]
    word_count: int
    confidence: float
    evidence: list[str]


class SessionProfileDocument(TypedDict):
    session_id: str
    origin: str
    title: str | None
    inferred_topic: str | None
    inferred_topic_source: str
    created_at: str | None
    updated_at: str | None
    message_count: int
    substantive_count: int
    tool_use_count: int
    thinking_count: int
    attachment_count: int
    word_count: int
    total_cost_usd: float
    total_duration_ms: int
    tool_categories: dict[str, int]
    repo_paths: list[str]
    cwd_paths: list[str]
    branch_names: list[str]
    file_paths_touched: list[str]
    languages_detected: list[str]
    repo_names: list[str]
    work_events: list[WorkEventDocument]
    phases: list[SessionPhaseDocument]
    first_message_at: str | None
    last_message_at: str | None
    timestamp_source: str
    timestamped_message_count: int
    untimestamped_message_count: int
    timestamp_coverage: str
    canonical_session_date: str | None
    engaged_duration_ms: int
    engaged_minutes: float
    tool_active_duration_ms: int
    tool_active_minutes: float
    wall_duration_ms: int
    workflow_shape: str
    workflow_shape_confidence: float
    workflow_shape_features: dict[str, object]
    terminal_state: str
    terminal_state_confidence: float
    terminal_state_evidence: dict[str, object]
    cost_is_estimated: bool
    logical_session_id: str | None
    compaction_count: int
    thread_id: str | None
    continuation_depth: int
    tags: list[str]
    auto_tags: list[str]
    is_continuation: bool
    parent_id: str | None
    thinking_duration_ms: int
    output_duration_ms: int
    tool_duration_ms: int
    latency_percentiles_ms: dict[str, int]
    tool_calls_per_minute: float
    timing_provenance: str
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    total_credit_cost: float
    cost_provenance: str
    per_model_cost_json: str
    primary_model_name: str | None
    primary_model_family: str | None


class ThreadMemberEvidenceDocument(TypedDict):
    session_id: str
    parent_id: str | None
    role: str
    depth: int
    confidence: float
    support_signals: list[str]
    evidence: list[str]


class ThreadDocument(TypedDict):
    thread_id: str
    root_id: str
    session_ids: list[str]
    session_count: int
    depth: int
    branch_count: int
    start_time: str | None
    end_time: str | None
    wall_duration_ms: int
    total_messages: int
    total_cost_usd: float
    dominant_repo: str | None
    origin_breakdown: dict[str, int]
    provider_breakdown: NotRequired[dict[str, int]]
    work_event_breakdown: dict[str, int]
    confidence: float
    support_level: str
    support_signals: list[str]
    member_evidence: list[ThreadMemberEvidenceDocument]


__all__ = [
    "SessionPhaseDocument",
    "SessionProfileDocument",
    "WorkEventDocument",
    "ThreadDocument",
    "ThreadMemberEvidenceDocument",
]
