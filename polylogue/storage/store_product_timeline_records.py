"""Timeline-oriented derived product storage models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from polylogue.types import ConversationId

from .store_constants import (
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
)


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
    def work_event_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


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
    def session_phase_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


__all__ = ["SessionPhaseRecord", "SessionWorkEventRecord"]
