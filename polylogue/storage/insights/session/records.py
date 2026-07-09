"""Session-level derived insight storage models."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from polylogue.insights.archive_models import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    ThreadPayload,
)
from polylogue.storage.runtime.store_constants import (
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)
from polylogue.types import SessionId


class SessionProfileRecord(BaseModel):
    session_id: SessionId
    logical_session_id: SessionId
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None
    input_row_count: int = 0
    source_name: str
    title: str | None = None
    first_message_at: str | None = None
    last_message_at: str | None = None
    canonical_session_date: str | None = None
    repo_paths: tuple[str, ...] = ()
    repo_names: tuple[str, ...] = ()
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
    paste_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    engaged_duration_ms: int = 0
    tool_active_duration_ms: int = 0
    wall_duration_ms: int = 0
    workflow_shape: str = "unknown"
    workflow_shape_confidence: float = 0.0
    workflow_shape_features_json: str = "{}"
    terminal_state: str = "unknown"
    terminal_state_confidence: float = 0.0
    terminal_state_evidence_json: str = "{}"
    cost_is_estimated: bool = False
    thinking_duration_ms: int = 0
    output_duration_ms: int = 0
    tool_duration_ms: int = 0
    latency_percentiles_ms_json: str = "{}"
    tool_calls_per_minute: float = 0.0
    timing_provenance: str = "sort_key_estimated"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_credit_cost: float = 0.0
    cost_provenance: str = "unknown"
    per_model_cost_json: str = "{}"
    primary_model_name: str | None = None
    primary_model_family: str | None = None
    evidence_payload: SessionEvidencePayload
    inference_payload: SessionInferencePayload
    search_text: str
    evidence_search_text: str
    inference_search_text: str
    enrichment_payload: SessionEnrichmentPayload
    enrichment_search_text: str
    enrichment_version: int = SESSION_ENRICHMENT_VERSION
    enrichment_family: str = SESSION_ENRICHMENT_FAMILY
    inference_version: int = SESSION_INFERENCE_VERSION
    inference_family: str = SESSION_INFERENCE_FAMILY

    @field_validator(
        "session_id",
        "logical_session_id",
        "source_name",
        "materialized_at",
        "search_text",
        "evidence_search_text",
        "inference_search_text",
        "enrichment_search_text",
        "enrichment_family",
        "inference_family",
    )
    @classmethod
    def profile_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


class SessionLatencyProfileRecord(BaseModel):
    session_id: SessionId
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None
    input_row_count: int = 0
    source_name: str
    title: str | None = None
    first_message_at: str | None = None
    last_message_at: str | None = None
    canonical_session_date: str | None = None
    median_tool_call_ms: int = 0
    p90_tool_call_ms: int = 0
    max_tool_call_ms: int = 0
    stuck_tool_count: int = 0
    median_agent_response_ms: int = 0
    median_user_response_ms: int = 0
    tool_call_count_by_category_json: str = "{}"
    evidence_payload_json: str = "{}"
    search_text: str = ""

    @field_validator("session_id", "source_name", "materialized_at")
    @classmethod
    def latency_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


class ThreadRecord(BaseModel):
    thread_id: str
    root_id: SessionId
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None
    input_row_count: int = 0
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
    work_event_breakdown: dict[str, int] | None = None
    payload: ThreadPayload
    search_text: str

    @field_validator("thread_id", "root_id", "materialized_at", "search_text")
    @classmethod
    def thread_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


__all__ = ["SessionLatencyProfileRecord", "SessionProfileRecord", "ThreadRecord"]
