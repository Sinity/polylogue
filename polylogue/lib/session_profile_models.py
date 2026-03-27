"""Typed session-profile semantic models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from polylogue.lib.attribution import ConversationAttribution
from polylogue.lib.phase_extraction import SessionPhase
from polylogue.lib.project_normalization import normalize_project_names, normalize_repo_paths
from polylogue.lib.semantic_facts import ConversationSemanticFacts
from polylogue.lib.work_event_extraction import WorkEvent


@dataclass(frozen=True)
class SessionProfile:
    """Complete semantic profile of a conversation session."""

    conversation_id: str
    provider: str
    title: str | None
    created_at: datetime | None
    updated_at: datetime | None
    message_count: int
    substantive_count: int
    tool_use_count: int
    thinking_count: int
    attachment_count: int
    word_count: int
    total_cost_usd: float
    total_duration_ms: int
    tool_categories: dict[str, int]
    repo_paths: tuple[str, ...]
    cwd_paths: tuple[str, ...]
    branch_names: tuple[str, ...]
    file_paths_touched: tuple[str, ...]
    languages_detected: tuple[str, ...]
    canonical_projects: tuple[str, ...]
    work_events: tuple[WorkEvent, ...]
    phases: tuple[SessionPhase, ...]
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None
    canonical_session_date: date | None = None
    engaged_duration_ms: int = 0
    wall_duration_ms: int = 0
    cost_is_estimated: bool = False
    thread_id: str | None = None
    continuation_depth: int = 0
    tags: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()
    is_continuation: bool = False
    parent_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.message_count,
            "substantive_count": self.substantive_count,
            "tool_use_count": self.tool_use_count,
            "thinking_count": self.thinking_count,
            "attachment_count": self.attachment_count,
            "word_count": self.word_count,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_ms": self.total_duration_ms,
            "tool_categories": self.tool_categories,
            "repo_paths": list(self.repo_paths),
            "cwd_paths": list(self.cwd_paths),
            "branch_names": list(self.branch_names),
            "file_paths_touched": list(self.file_paths_touched),
            "languages_detected": list(self.languages_detected),
            "canonical_projects": list(self.canonical_projects),
            "work_events": [event.to_dict() for event in self.work_events],
            "phases": [
                {
                    "start_time": phase.start_time.isoformat() if phase.start_time else None,
                    "end_time": phase.end_time.isoformat() if phase.end_time else None,
                    "canonical_session_date": (
                        phase.canonical_session_date.isoformat()
                        if phase.canonical_session_date
                        else None
                    ),
                    "message_range": list(phase.message_range),
                    "duration_ms": phase.duration_ms,
                    "tool_counts": phase.tool_counts,
                    "word_count": phase.word_count,
                    "confidence": phase.confidence,
                    "evidence": list(phase.evidence),
                }
                for phase in self.phases
            ],
            "first_message_at": self.first_message_at.isoformat() if self.first_message_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "canonical_session_date": (
                self.canonical_session_date.isoformat()
                if self.canonical_session_date
                else None
            ),
            "engaged_duration_ms": self.engaged_duration_ms,
            "engaged_minutes": round(self.engaged_duration_ms / 60_000.0, 4),
            "wall_duration_ms": self.wall_duration_ms,
            "cost_is_estimated": self.cost_is_estimated,
            "thread_id": self.thread_id,
            "continuation_depth": self.continuation_depth,
            "tags": list(self.tags),
            "auto_tags": list(self.auto_tags),
            "is_continuation": self.is_continuation,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SessionProfile:
        repo_paths = normalize_repo_paths(payload.get("repo_paths", []) or [])
        canonical_projects = normalize_project_names(
            payload.get("canonical_projects", []) or [],
            repo_paths=repo_paths,
        )
        return cls(
            conversation_id=str(payload["conversation_id"]),
            provider=str(payload["provider"]),
            title=str(payload["title"]) if payload.get("title") is not None else None,
            created_at=datetime.fromisoformat(str(payload["created_at"])) if payload.get("created_at") else None,
            updated_at=datetime.fromisoformat(str(payload["updated_at"])) if payload.get("updated_at") else None,
            message_count=int(payload.get("message_count", 0) or 0),
            substantive_count=int(payload.get("substantive_count", 0) or 0),
            tool_use_count=int(payload.get("tool_use_count", 0) or 0),
            thinking_count=int(payload.get("thinking_count", 0) or 0),
            attachment_count=int(payload.get("attachment_count", 0) or 0),
            word_count=int(payload.get("word_count", 0) or 0),
            total_cost_usd=float(payload.get("total_cost_usd", 0.0) or 0.0),
            total_duration_ms=int(payload.get("total_duration_ms", 0) or 0),
            tool_categories={
                str(key): int(value or 0)
                for key, value in (payload.get("tool_categories", {}) or {}).items()
            },
            repo_paths=repo_paths,
            cwd_paths=tuple(str(item) for item in payload.get("cwd_paths", []) or []),
            branch_names=tuple(str(item) for item in payload.get("branch_names", []) or []),
            file_paths_touched=tuple(str(item) for item in payload.get("file_paths_touched", []) or []),
            languages_detected=tuple(str(item) for item in payload.get("languages_detected", []) or []),
            canonical_projects=canonical_projects,
            work_events=tuple(
                WorkEvent.from_dict(item)
                for item in payload.get("work_events", []) or []
                if isinstance(item, dict)
            ),
            phases=tuple(
                SessionPhase(
                    start_time=datetime.fromisoformat(str(item["start_time"])) if item.get("start_time") else None,
                    end_time=datetime.fromisoformat(str(item["end_time"])) if item.get("end_time") else None,
                    canonical_session_date=(
                        date.fromisoformat(str(item["canonical_session_date"]))
                        if item.get("canonical_session_date")
                        else None
                    ),
                    message_range=tuple(int(value) for value in item.get("message_range", [0, 0])),
                    duration_ms=int(item.get("duration_ms", 0) or 0),
                    tool_counts={str(key): int(value or 0) for key, value in (item.get("tool_counts", {}) or {}).items()},
                    word_count=int(item.get("word_count", 0) or 0),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    evidence=tuple(str(value) for value in item.get("evidence", []) or []),
                )
                for item in payload.get("phases", []) or []
                if isinstance(item, dict)
            ),
            first_message_at=(
                datetime.fromisoformat(str(payload["first_message_at"]))
                if payload.get("first_message_at")
                else None
            ),
            last_message_at=(
                datetime.fromisoformat(str(payload["last_message_at"]))
                if payload.get("last_message_at")
                else None
            ),
            canonical_session_date=(
                date.fromisoformat(str(payload["canonical_session_date"]))
                if payload.get("canonical_session_date")
                else None
            ),
            engaged_duration_ms=int(payload.get("engaged_duration_ms", 0) or 0),
            wall_duration_ms=int(payload.get("wall_duration_ms", 0) or 0),
            cost_is_estimated=bool(payload.get("cost_is_estimated", False)),
            thread_id=str(payload["thread_id"]) if payload.get("thread_id") is not None else None,
            continuation_depth=int(payload.get("continuation_depth", 0) or 0),
            tags=tuple(str(item) for item in payload.get("tags", []) or []),
            auto_tags=tuple(str(item) for item in payload.get("auto_tags", []) or []),
            is_continuation=bool(payload.get("is_continuation", False)),
            parent_id=str(payload["parent_id"]) if payload.get("parent_id") is not None else None,
        )


@dataclass(frozen=True)
class SessionAnalysis:
    facts: ConversationSemanticFacts
    attribution: ConversationAttribution
    work_events: tuple[WorkEvent, ...]
    phases: tuple[SessionPhase, ...]


__all__ = ["SessionAnalysis", "SessionProfile"]
