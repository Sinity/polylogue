"""Shared typed document payloads for session-derived runtime surfaces."""

from __future__ import annotations

from typing import TypedDict


class WorkEventDocument(TypedDict):
    kind: str
    start_index: int
    end_index: int
    start_time: str | None
    end_time: str | None
    canonical_session_date: str | None
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
    message_range: list[int]
    duration_ms: int
    tool_counts: dict[str, int]
    word_count: int
    confidence: float
    evidence: list[str]


class SessionProfileDocument(TypedDict):
    conversation_id: str
    provider: str
    title: str | None
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
    timestamped_message_count: int
    untimestamped_message_count: int
    timestamp_coverage: str
    canonical_session_date: str | None
    engaged_duration_ms: int
    engaged_minutes: float
    wall_duration_ms: int
    cost_is_estimated: bool
    compaction_count: int
    thread_id: str | None
    continuation_depth: int
    tags: list[str]
    auto_tags: list[str]
    is_continuation: bool
    parent_id: str | None


class WorkThreadMemberEvidenceDocument(TypedDict):
    conversation_id: str
    parent_id: str | None
    role: str
    depth: int
    confidence: float
    support_signals: list[str]
    evidence: list[str]


class WorkThreadDocument(TypedDict):
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
    provider_breakdown: dict[str, int]
    work_event_breakdown: dict[str, int]
    confidence: float
    support_level: str
    support_signals: list[str]
    member_evidence: list[WorkThreadMemberEvidenceDocument]


__all__ = [
    "SessionPhaseDocument",
    "SessionProfileDocument",
    "WorkEventDocument",
    "WorkThreadDocument",
    "WorkThreadMemberEvidenceDocument",
]
