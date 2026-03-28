"""Data models for the ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class PlanResult(BaseModel):
    timestamp: int
    counts: dict[str, int]
    sources: list[str]
    cursors: dict[str, dict]


class RunResult(BaseModel):
    run_id: str
    counts: dict[str, int]
    drift: dict[str, dict]
    indexed: bool
    index_error: str | None
    duration_ms: int


class ExistingConversation(BaseModel):
    conversation_id: str
    content_hash: str
