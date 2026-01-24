"""Data models for the ingestion pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PlanResult(BaseModel):
    timestamp: int
    counts: dict[str, int]
    sources: list[str]
    cursors: dict[str, dict[str, Any]]


class RunResult(BaseModel):
    run_id: str
    counts: dict[str, int]
    drift: dict[str, dict[str, int]]
    indexed: bool
    index_error: str | None
    duration_ms: int
    render_failures: list[dict[str, str]] = []


class ExistingConversation(BaseModel):
    conversation_id: str
    content_hash: str
