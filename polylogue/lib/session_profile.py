"""Semantic session-profile runtime builders."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import TYPE_CHECKING

from polylogue.lib.attribution import ConversationAttribution, extract_attribution
from polylogue.lib.decisions import Decision, extract_decisions
from polylogue.lib.phases import SessionPhase, extract_phases
from polylogue.lib.project_normalization import normalize_project_names, normalize_repo_paths
from polylogue.lib.semantic_facts import ConversationSemanticFacts, build_conversation_semantic_facts
from polylogue.lib.work_events import WorkEvent, extract_work_events

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation


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
    decisions: tuple[Decision, ...] = ()
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
                    "kind": phase.kind,
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
                }
                for phase in self.phases
            ],
            "decisions": [
                {
                    "index": decision.index,
                    "summary": decision.summary,
                    "confidence": decision.confidence,
                    "context": decision.context,
                }
                for decision in self.decisions
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
                    kind=str(item["kind"]),
                    start_time=datetime.fromisoformat(str(item["start_time"])) if item.get("start_time") else None,
                    end_time=datetime.fromisoformat(str(item["end_time"])) if item.get("end_time") else None,
                    canonical_session_date=(
                        date.fromisoformat(str(item["canonical_session_date"]))
                        if item.get("canonical_session_date")
                        else None
                    ),
                    message_range=tuple(int(v) for v in item.get("message_range", [0, 0])),
                    duration_ms=int(item.get("duration_ms", 0) or 0),
                    tool_counts={str(k): int(v or 0) for k, v in (item.get("tool_counts", {}) or {}).items()},
                    word_count=int(item.get("word_count", 0) or 0),
                )
                for item in payload.get("phases", []) or []
                if isinstance(item, dict)
            ),
            decisions=tuple(
                Decision(
                    index=int(item.get("index", 0) or 0),
                    summary=str(item.get("summary", "") or ""),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    context=str(item.get("context", "") or ""),
                )
                for item in payload.get("decisions", []) or []
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
    decisions: tuple[Decision, ...]


def build_session_analysis(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> SessionAnalysis:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    return SessionAnalysis(
        facts=semantic_facts,
        attribution=extract_attribution(conversation, facts=semantic_facts),
        work_events=tuple(extract_work_events(conversation, facts=semantic_facts)),
        phases=tuple(extract_phases(conversation, facts=semantic_facts)),
        decisions=tuple(extract_decisions(conversation, facts=semantic_facts)),
    )


def infer_auto_tags(profile: SessionProfile) -> tuple[str, ...]:
    tags: list[str] = [f"provider:{profile.provider}"]
    for project in list(profile.canonical_projects)[:3]:
        tags.append(f"project:{project}")
    if profile.work_events:
        kind_counter: Counter[str] = Counter()
        for event in profile.work_events:
            kind_counter[event.kind.value if hasattr(event.kind, "value") else str(event.kind)] += 1
        tags.append(f"kind:{kind_counter.most_common(1)[0][0]}")
    if profile.is_continuation:
        tags.append("continuation")
    if profile.continuation_depth >= 3:
        tags.append("deep-thread")
    if len(profile.canonical_projects) > 1:
        tags.append("multi-project")
    if profile.total_cost_usd >= 1.0:
        tags.append("costly")
    return tuple(sorted(set(tags)))


def build_session_profile(
    conversation: Conversation,
    *,
    analysis: SessionAnalysis | None = None,
) -> SessionProfile:
    from polylogue.lib.pricing import harmonize_session_cost

    session_analysis = analysis or build_session_analysis(conversation)
    facts = session_analysis.facts
    attribution = session_analysis.attribution
    cost_usd, cost_is_estimated = harmonize_session_cost(conversation)
    engaged_duration_ms = sum(int(phase.duration_ms or 0) for phase in session_analysis.phases)
    if engaged_duration_ms <= 0:
        engaged_duration_ms = max(int(conversation.total_duration_ms or 0), 0)
    canonical_session_at = (
        facts.first_message_at
        or conversation.created_at
        or conversation.updated_at
        or facts.last_message_at
    )
    partial = SessionProfile(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=facts.total_messages,
        substantive_count=facts.substantive_messages,
        tool_use_count=facts.tool_messages,
        thinking_count=facts.thinking_messages,
        word_count=facts.word_count,
        total_cost_usd=cost_usd,
        total_duration_ms=conversation.total_duration_ms,
        tool_categories=facts.tool_category_counts,
        repo_paths=attribution.repo_paths,
        cwd_paths=attribution.cwd_paths,
        branch_names=attribution.branch_names,
        file_paths_touched=attribution.file_paths_touched,
        languages_detected=attribution.languages_detected,
        canonical_projects=attribution.canonical_projects,
        work_events=session_analysis.work_events,
        phases=session_analysis.phases,
        decisions=session_analysis.decisions,
        first_message_at=facts.first_message_at,
        last_message_at=facts.last_message_at,
        canonical_session_date=canonical_session_at.date() if canonical_session_at else None,
        engaged_duration_ms=engaged_duration_ms,
        wall_duration_ms=facts.wall_duration_ms,
        cost_is_estimated=cost_is_estimated,
        tags=tuple(conversation.tags),
        is_continuation=conversation.is_continuation,
        parent_id=str(conversation.parent_id) if conversation.parent_id else None,
    )
    return replace(partial, auto_tags=infer_auto_tags(partial))


__all__ = [
    "SessionAnalysis",
    "SessionProfile",
    "build_session_analysis",
    "build_session_profile",
    "infer_auto_tags",
]
