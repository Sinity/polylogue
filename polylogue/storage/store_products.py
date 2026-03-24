"""Derived/session-product storage record models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from polylogue.types import ConversationId

from .store_core import (
    MAINTENANCE_RUN_SCHEMA_VERSION,
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
)


class SessionProfileRecord(BaseModel):
    conversation_id: ConversationId
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    provider_name: str
    title: str | None = None
    first_message_at: str | None = None
    last_message_at: str | None = None
    canonical_session_date: str | None = None
    primary_work_kind: str | None = None
    repo_paths: tuple[str, ...] = ()
    canonical_projects: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()
    message_count: int = 0
    substantive_count: int = 0
    attachment_count: int = 0
    work_event_count: int = 0
    phase_count: int = 0
    word_count: int = 0
    tool_use_count: int = 0
    thinking_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    engaged_duration_ms: int = 0
    wall_duration_ms: int = 0
    cost_is_estimated: bool = False
    evidence_payload: dict[str, Any]
    inference_payload: dict[str, Any]
    search_text: str
    evidence_search_text: str
    inference_search_text: str
    enrichment_payload: dict[str, Any]
    enrichment_search_text: str
    enrichment_version: int = SESSION_ENRICHMENT_VERSION
    enrichment_family: str = SESSION_ENRICHMENT_FAMILY
    inference_version: int = SESSION_INFERENCE_VERSION
    inference_family: str = SESSION_INFERENCE_FAMILY

    @field_validator(
        "conversation_id",
        "provider_name",
        "materialized_at",
        "search_text",
        "evidence_search_text",
        "inference_search_text",
        "enrichment_search_text",
        "enrichment_family",
        "inference_family",
    )
    @classmethod
    def profile_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class SessionWorkEventRecord(BaseModel):
    event_id: str
    conversation_id: ConversationId
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    provider_name: str
    event_index: int
    kind: str
    confidence: float
    start_index: int
    end_index: int
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: int = 0
    canonical_session_date: str | None = None
    summary: str
    file_paths: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    evidence_payload: dict[str, Any]
    inference_payload: dict[str, Any]
    search_text: str
    inference_version: int = SESSION_INFERENCE_VERSION
    inference_family: str = SESSION_INFERENCE_FAMILY

    @field_validator(
        "event_id",
        "conversation_id",
        "materialized_at",
        "provider_name",
        "kind",
        "summary",
        "search_text",
        "inference_family",
    )
    @classmethod
    def work_event_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class SessionPhaseRecord(BaseModel):
    phase_id: str
    conversation_id: ConversationId
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    provider_name: str
    phase_index: int
    kind: str
    start_index: int
    end_index: int
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: int = 0
    canonical_session_date: str | None = None
    confidence: float = 0.0
    evidence_reasons: tuple[str, ...] = ()
    tool_counts: dict[str, int]
    word_count: int = 0
    evidence_payload: dict[str, Any]
    inference_payload: dict[str, Any]
    search_text: str
    inference_version: int = SESSION_INFERENCE_VERSION
    inference_family: str = SESSION_INFERENCE_FAMILY

    @field_validator(
        "phase_id",
        "conversation_id",
        "materialized_at",
        "provider_name",
        "kind",
        "search_text",
        "inference_family",
    )
    @classmethod
    def session_phase_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class WorkThreadRecord(BaseModel):
    thread_id: str
    root_id: ConversationId
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    start_time: str | None = None
    end_time: str | None = None
    dominant_project: str | None = None
    session_ids: tuple[str, ...] = ()
    session_count: int = 0
    depth: int = 0
    branch_count: int = 0
    total_messages: int = 0
    total_cost_usd: float = 0.0
    wall_duration_ms: int = 0
    work_event_breakdown: dict[str, int] | None = None
    payload: dict[str, Any]
    search_text: str

    @field_validator("thread_id", "root_id", "materialized_at", "search_text")
    @classmethod
    def work_thread_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class SessionTagRollupRecord(BaseModel):
    tag: str
    bucket_day: str
    provider_name: str
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    conversation_count: int = 0
    explicit_count: int = 0
    auto_count: int = 0
    project_breakdown: dict[str, int]
    search_text: str

    @field_validator("tag", "bucket_day", "provider_name", "materialized_at", "search_text")
    @classmethod
    def tag_rollup_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class DaySessionSummaryRecord(BaseModel):
    day: str
    provider_name: str
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    conversation_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_wall_duration_ms: int = 0
    total_messages: int = 0
    total_words: int = 0
    work_event_breakdown: dict[str, int]
    projects_active: tuple[str, ...] = ()
    payload: dict[str, Any]
    search_text: str

    @field_validator("day", "provider_name", "materialized_at", "search_text")
    @classmethod
    def day_summary_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class MaintenanceRunRecord(BaseModel):
    maintenance_run_id: str
    schema_version: int = MAINTENANCE_RUN_SCHEMA_VERSION
    executed_at: str
    mode: str
    preview: bool = False
    repair_selected: bool = False
    cleanup_selected: bool = False
    vacuum_requested: bool = False
    target_names: tuple[str, ...] = ()
    success: bool = True
    manifest: dict[str, Any]

    @field_validator("maintenance_run_id", "executed_at", "mode")
    @classmethod
    def maintenance_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


__all__ = [
    "DaySessionSummaryRecord",
    "MaintenanceRunRecord",
    "SessionPhaseRecord",
    "SessionProfileRecord",
    "SessionTagRollupRecord",
    "SessionWorkEventRecord",
    "WorkThreadRecord",
]
