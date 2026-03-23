"""Derived semantic product models built on the canonical runtime fact layers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import TYPE_CHECKING

from polylogue.lib.attribution import ConversationAttribution, extract_attribution
from polylogue.lib.decisions import Decision, extract_decisions
from polylogue.lib.phases import SessionPhase, extract_phases
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
            "work_events": [
                {
                    "kind": event.kind.value,
                    "start_index": event.start_index,
                    "end_index": event.end_index,
                    "confidence": event.confidence,
                    "evidence": list(event.evidence),
                    "file_paths": list(event.file_paths),
                    "tools_used": list(event.tools_used),
                    "summary": event.summary,
                }
                for event in self.work_events
            ],
            "phases": [
                {
                    "kind": phase.kind,
                    "start_time": phase.start_time.isoformat() if phase.start_time else None,
                    "end_time": phase.end_time.isoformat() if phase.end_time else None,
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
            "wall_duration_ms": self.wall_duration_ms,
            "cost_is_estimated": self.cost_is_estimated,
            "thread_id": self.thread_id,
            "continuation_depth": self.continuation_depth,
            "tags": list(self.tags),
            "auto_tags": list(self.auto_tags),
            "is_continuation": self.is_continuation,
            "parent_id": self.parent_id,
        }


@dataclass(frozen=True)
class WorkThread:
    thread_id: str
    root_id: str
    session_ids: tuple[str, ...]
    depth: int
    branch_count: int
    start_time: datetime | None
    end_time: datetime | None
    wall_duration_ms: int
    total_messages: int
    total_cost_usd: float
    dominant_project: str | None
    work_event_breakdown: dict[str, int]


@dataclass(frozen=True)
class DaySessionSummary:
    date: date
    session_count: int
    total_cost_usd: float
    total_duration_ms: int
    total_wall_duration_ms: int
    total_messages: int
    total_words: int
    work_event_breakdown: dict[str, int]
    projects_active: tuple[str, ...]
    providers: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "date": self.date.isoformat(),
            "session_count": self.session_count,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_ms": self.total_duration_ms,
            "total_wall_duration_ms": self.total_wall_duration_ms,
            "total_messages": self.total_messages,
            "total_words": self.total_words,
            "work_event_breakdown": self.work_event_breakdown,
            "projects_active": list(self.projects_active),
            "providers": self.providers,
        }


@dataclass(frozen=True)
class WeekSessionSummary:
    iso_week: str
    day_summaries: tuple[DaySessionSummary, ...]
    session_count: int
    total_cost_usd: float
    total_duration_ms: int
    total_messages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "iso_week": self.iso_week,
            "day_summaries": [day.to_dict() for day in self.day_summaries],
            "session_count": self.session_count,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_ms": self.total_duration_ms,
            "total_messages": self.total_messages,
        }


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
        wall_duration_ms=facts.wall_duration_ms,
        cost_is_estimated=cost_is_estimated,
        tags=tuple(conversation.tags),
        is_continuation=conversation.is_continuation,
        parent_id=str(conversation.parent_id) if conversation.parent_id else None,
    )
    return replace(partial, auto_tags=infer_auto_tags(partial))


def _bfs_depth(adjacency: dict[str, list[str]], root: str) -> int:
    visited = {root}
    frontier = [root]
    depth = 0
    while frontier:
        next_frontier = []
        for node in frontier:
            for child in adjacency.get(node, []):
                if child not in visited:
                    visited.add(child)
                    next_frontier.append(child)
        if not next_frontier:
            break
        depth += 1
        frontier = next_frontier
    return depth


def build_session_threads(profiles: Iterable[SessionProfile]) -> list[WorkThread]:
    all_profiles = list(profiles)
    by_id = {profile.conversation_id: profile for profile in all_profiles}
    children: dict[str, list[str]] = defaultdict(list)
    for profile in all_profiles:
        if profile.parent_id and profile.parent_id in by_id:
            children[profile.parent_id].append(profile.conversation_id)
    child_ids = {child_id for child_list in children.values() for child_id in child_list}
    roots = [profile for profile in all_profiles if profile.conversation_id not in child_ids]

    threads: list[WorkThread] = []
    for root in roots:
        thread_ids: list[str] = []
        frontier = [root.conversation_id]
        while frontier:
            next_frontier = []
            for conversation_id in frontier:
                thread_ids.append(conversation_id)
                next_frontier.extend(children.get(conversation_id, []))
            frontier = next_frontier

        thread_profiles = [by_id[conversation_id] for conversation_id in thread_ids if conversation_id in by_id]
        timestamps_start = [profile.first_message_at for profile in thread_profiles if profile.first_message_at]
        timestamps_end = [profile.last_message_at for profile in thread_profiles if profile.last_message_at]
        start_time = min(timestamps_start) if timestamps_start else None
        end_time = max(timestamps_end) if timestamps_end else None
        wall_ms = 0
        if start_time and end_time:
            wall_ms = max(int((end_time - start_time).total_seconds() * 1000), 0)

        project_counter: Counter[str] = Counter()
        work_event_counter: Counter[str] = Counter()
        for profile in thread_profiles:
            project_counter.update(profile.canonical_projects)
            work_event_counter.update(
                event.kind.value if hasattr(event.kind, "value") else str(event.kind)
                for event in profile.work_events
            )

        threads.append(
            WorkThread(
                thread_id=root.conversation_id,
                root_id=root.conversation_id,
                session_ids=tuple(thread_ids),
                depth=_bfs_depth(children, root.conversation_id),
                branch_count=sum(1 for conversation_id in thread_ids if not children.get(conversation_id)),
                start_time=start_time,
                end_time=end_time,
                wall_duration_ms=wall_ms,
                total_messages=sum(profile.message_count for profile in thread_profiles),
                total_cost_usd=sum(profile.total_cost_usd for profile in thread_profiles),
                dominant_project=project_counter.most_common(1)[0][0] if project_counter else None,
                work_event_breakdown=dict(work_event_counter),
            )
        )
    return threads


def _profile_date(profile: SessionProfile) -> date | None:
    timestamp = profile.first_message_at or profile.created_at
    if timestamp is None:
        return None
    return timestamp.date() if isinstance(timestamp, datetime) else timestamp


def summarize_day(
    profiles: Sequence[SessionProfile],
    target_date: date,
) -> DaySessionSummary:
    work_events: Counter[str] = Counter()
    projects: set[str] = set()
    providers: Counter[str] = Counter()
    total_cost = 0.0
    total_duration = 0
    total_wall = 0
    total_messages = 0
    total_words = 0
    for profile in profiles:
        total_cost += profile.total_cost_usd
        total_duration += profile.total_duration_ms
        total_wall += profile.wall_duration_ms
        total_messages += profile.message_count
        total_words += profile.word_count
        providers[profile.provider] += 1
        work_events.update(
            event.kind.value if hasattr(event.kind, "value") else str(event.kind)
            for event in profile.work_events
        )
        projects.update(profile.repo_paths)
    return DaySessionSummary(
        date=target_date,
        session_count=len(profiles),
        total_cost_usd=total_cost,
        total_duration_ms=total_duration,
        total_wall_duration_ms=total_wall,
        total_messages=total_messages,
        total_words=total_words,
        work_event_breakdown=dict(work_events),
        projects_active=tuple(sorted(projects)),
        providers=dict(providers),
    )


def summarize_week(day_summaries: Sequence[DaySessionSummary]) -> WeekSessionSummary:
    if not day_summaries:
        return WeekSessionSummary(
            iso_week="",
            day_summaries=(),
            session_count=0,
            total_cost_usd=0.0,
            total_duration_ms=0,
            total_messages=0,
        )
    first_date = day_summaries[0].date
    iso = first_date.isocalendar()
    return WeekSessionSummary(
        iso_week=f"{iso[0]}-W{iso[1]:02d}",
        day_summaries=tuple(day_summaries),
        session_count=sum(day.session_count for day in day_summaries),
        total_cost_usd=sum(day.total_cost_usd for day in day_summaries),
        total_duration_ms=sum(day.total_duration_ms for day in day_summaries),
        total_messages=sum(day.total_messages for day in day_summaries),
    )


__all__ = [
    "DaySessionSummary",
    "SessionAnalysis",
    "SessionProfile",
    "WeekSessionSummary",
    "WorkThread",
    "build_session_analysis",
    "build_session_profile",
    "build_session_threads",
    "infer_auto_tags",
    "summarize_day",
    "summarize_week",
]
