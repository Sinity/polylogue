"""Typed phase semantic models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class SessionPhase:
    """A temporal phase within a conversation session."""

    kind: str
    start_time: datetime | None
    end_time: datetime | None
    canonical_session_date: date | None
    message_range: tuple[int, int]
    duration_ms: int
    tool_counts: dict[str, int]
    word_count: int
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()


__all__ = ["SessionPhase"]
