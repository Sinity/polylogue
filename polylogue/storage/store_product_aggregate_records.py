"""Aggregate derived product storage models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from .store_core import SESSION_PRODUCT_MATERIALIZER_VERSION


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
    def tag_rollup_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


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
    def day_summary_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


__all__ = ["DaySessionSummaryRecord", "SessionTagRollupRecord"]
