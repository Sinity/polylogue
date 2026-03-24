"""Typed work-event semantic models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class WorkEventKind(str, Enum):
    """Classification of work performed in a message range."""

    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    DEBUGGING = "debugging"
    REVIEW = "review"
    TESTING = "testing"
    RESEARCH = "research"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    DATA_ANALYSIS = "data_analysis"
    CONVERSATION = "conversation"


@dataclass(frozen=True)
class WorkEvent:
    """A classified segment of work within a conversation."""

    kind: WorkEventKind
    start_index: int
    end_index: int
    confidence: float
    evidence: tuple[str, ...]
    file_paths: tuple[str, ...]
    tools_used: tuple[str, ...]
    summary: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    canonical_session_date: date | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "canonical_session_date": (
                self.canonical_session_date.isoformat()
                if self.canonical_session_date
                else None
            ),
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "file_paths": list(self.file_paths),
            "tools_used": list(self.tools_used),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WorkEvent:
        return cls(
            kind=WorkEventKind(str(payload["kind"])),
            start_index=int(payload.get("start_index", 0) or 0),
            end_index=int(payload.get("end_index", 0) or 0),
            start_time=(
                datetime.fromisoformat(str(payload["start_time"]))
                if payload.get("start_time")
                else None
            ),
            end_time=(
                datetime.fromisoformat(str(payload["end_time"]))
                if payload.get("end_time")
                else None
            ),
            canonical_session_date=(
                date.fromisoformat(str(payload["canonical_session_date"]))
                if payload.get("canonical_session_date")
                else None
            ),
            duration_ms=int(payload.get("duration_ms", 0) or 0),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            evidence=tuple(str(item) for item in payload.get("evidence", []) or []),
            file_paths=tuple(str(item) for item in payload.get("file_paths", []) or []),
            tools_used=tuple(str(item) for item in payload.get("tools_used", []) or []),
            summary=str(payload.get("summary", "") or ""),
        )


__all__ = ["WorkEvent", "WorkEventKind"]
