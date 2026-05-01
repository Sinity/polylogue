"""Typed session-profile semantic models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

from polylogue.archive.phase.extraction import SessionPhase
from polylogue.archive.semantic.facts import ConversationSemanticFacts
from polylogue.lib.conversation.attribution import ConversationAttribution
from polylogue.lib.conversation.extraction import WorkEvent
from polylogue.lib.payload_coercion import (
    coerce_float,
    coerce_int,
    int_pair,
    mapping_sequence,
    optional_date,
    optional_datetime,
    optional_string,
    string_int_mapping,
    string_sequence,
)
from polylogue.lib.repo_identity import normalize_repo_names, normalize_repo_paths
from polylogue.lib.session.documents import (
    SessionPhaseDocument,
    SessionProfileDocument,
)

SessionPhasePayload = SessionPhaseDocument
SessionProfilePayload = SessionProfileDocument


def _phase_to_dict(phase: SessionPhase) -> SessionPhasePayload:
    return {
        "start_time": phase.start_time.isoformat() if phase.start_time else None,
        "end_time": phase.end_time.isoformat() if phase.end_time else None,
        "canonical_session_date": phase.canonical_session_date.isoformat() if phase.canonical_session_date else None,
        "message_range": list(phase.message_range),
        "duration_ms": phase.duration_ms,
        "tool_counts": phase.tool_counts,
        "word_count": phase.word_count,
        "confidence": phase.confidence,
        "evidence": list(phase.evidence),
    }


def _phase_from_mapping(payload: SessionPhasePayload | Mapping[str, object]) -> SessionPhase:
    return SessionPhase(
        start_time=optional_datetime(payload.get("start_time")),
        end_time=optional_datetime(payload.get("end_time")),
        canonical_session_date=optional_date(payload.get("canonical_session_date")),
        message_range=int_pair(payload.get("message_range")),
        duration_ms=coerce_int(payload.get("duration_ms"), 0),
        tool_counts=string_int_mapping(payload.get("tool_counts")),
        word_count=coerce_int(payload.get("word_count"), 0),
        confidence=coerce_float(payload.get("confidence"), 0.0),
        evidence=string_sequence(payload.get("evidence")),
    )


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
    repo_names: tuple[str, ...]
    work_events: tuple[WorkEvent, ...]
    phases: tuple[SessionPhase, ...]
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None
    timestamped_message_count: int = 0
    untimestamped_message_count: int = 0
    timestamp_coverage: str = "none"
    canonical_session_date: date | None = None
    engaged_duration_ms: int = 0
    wall_duration_ms: int = 0
    cost_is_estimated: bool = False
    thread_id: str | None = None
    continuation_depth: int = 0
    compaction_count: int = 0
    tags: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()
    is_continuation: bool = False
    parent_id: str | None = None

    def to_dict(self) -> SessionProfilePayload:
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
            "repo_names": list(self.repo_names),
            "work_events": [event.to_dict() for event in self.work_events],
            "phases": [_phase_to_dict(phase) for phase in self.phases],
            "first_message_at": self.first_message_at.isoformat() if self.first_message_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "timestamped_message_count": self.timestamped_message_count,
            "untimestamped_message_count": self.untimestamped_message_count,
            "timestamp_coverage": self.timestamp_coverage,
            "canonical_session_date": self.canonical_session_date.isoformat() if self.canonical_session_date else None,
            "engaged_duration_ms": self.engaged_duration_ms,
            "engaged_minutes": round(self.engaged_duration_ms / 60_000.0, 4),
            "wall_duration_ms": self.wall_duration_ms,
            "cost_is_estimated": self.cost_is_estimated,
            "compaction_count": self.compaction_count,
            "thread_id": self.thread_id,
            "continuation_depth": self.continuation_depth,
            "tags": list(self.tags),
            "auto_tags": list(self.auto_tags),
            "is_continuation": self.is_continuation,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, payload: SessionProfilePayload | Mapping[str, object]) -> SessionProfile:
        repo_paths = normalize_repo_paths(string_sequence(payload.get("repo_paths")))
        explicit_repo_names = tuple(
            sorted({item.strip() for item in string_sequence(payload.get("repo_names")) if item.strip()})
        )
        repo_names = explicit_repo_names or normalize_repo_names(repo_paths=repo_paths)
        return cls(
            conversation_id=str(payload["conversation_id"]),
            provider=str(payload["provider"]),
            title=optional_string(payload.get("title")),
            created_at=optional_datetime(payload.get("created_at")),
            updated_at=optional_datetime(payload.get("updated_at")),
            message_count=coerce_int(payload.get("message_count"), 0),
            substantive_count=coerce_int(payload.get("substantive_count"), 0),
            tool_use_count=coerce_int(payload.get("tool_use_count"), 0),
            thinking_count=coerce_int(payload.get("thinking_count"), 0),
            attachment_count=coerce_int(payload.get("attachment_count"), 0),
            word_count=coerce_int(payload.get("word_count"), 0),
            total_cost_usd=coerce_float(payload.get("total_cost_usd"), 0.0),
            total_duration_ms=coerce_int(payload.get("total_duration_ms"), 0),
            tool_categories=string_int_mapping(payload.get("tool_categories")),
            repo_paths=repo_paths,
            cwd_paths=string_sequence(payload.get("cwd_paths")),
            branch_names=string_sequence(payload.get("branch_names")),
            file_paths_touched=string_sequence(payload.get("file_paths_touched")),
            languages_detected=string_sequence(payload.get("languages_detected")),
            repo_names=repo_names,
            work_events=tuple(WorkEvent.from_dict(item) for item in mapping_sequence(payload.get("work_events"))),
            phases=tuple(_phase_from_mapping(item) for item in mapping_sequence(payload.get("phases"))),
            first_message_at=optional_datetime(payload.get("first_message_at")),
            last_message_at=optional_datetime(payload.get("last_message_at")),
            timestamped_message_count=coerce_int(payload.get("timestamped_message_count"), 0),
            untimestamped_message_count=coerce_int(payload.get("untimestamped_message_count"), 0),
            timestamp_coverage=optional_string(payload.get("timestamp_coverage")) or "none",
            canonical_session_date=optional_date(payload.get("canonical_session_date")),
            engaged_duration_ms=coerce_int(payload.get("engaged_duration_ms"), 0),
            wall_duration_ms=coerce_int(payload.get("wall_duration_ms"), 0),
            cost_is_estimated=bool(payload.get("cost_is_estimated", False)),
            compaction_count=coerce_int(payload.get("compaction_count"), 0),
            thread_id=optional_string(payload.get("thread_id")),
            continuation_depth=coerce_int(payload.get("continuation_depth"), 0),
            tags=string_sequence(payload.get("tags")),
            auto_tags=string_sequence(payload.get("auto_tags")),
            is_continuation=bool(payload.get("is_continuation", False)),
            parent_id=optional_string(payload.get("parent_id")),
        )


@dataclass(frozen=True)
class SessionAnalysis:
    facts: ConversationSemanticFacts
    attribution: ConversationAttribution
    work_events: tuple[WorkEvent, ...]
    phases: tuple[SessionPhase, ...]


__all__ = ["SessionAnalysis", "SessionProfile"]
