"""Session-level derived product storage models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from polylogue.types import ConversationId

from .store_constants import (
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
    def profile_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


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
    def work_thread_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


__all__ = ["SessionProfileRecord", "WorkThreadRecord"]
