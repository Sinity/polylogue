"""Unified session profile — the semantic contract between Polylogue and Lynchpin.

Composes work event extraction, path/repo attribution, and temporal
phase detection into a single frozen profile per conversation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation

from polylogue.lib.attribution import ConversationAttribution, extract_attribution
from polylogue.lib.decisions import Decision, extract_decisions
from polylogue.lib.phases import SessionPhase, extract_phases
from polylogue.lib.work_events import WorkEvent, WorkEventKind, extract_work_events



@dataclass(frozen=True)
class SessionProfile:
    """Complete semantic profile of a conversation session.

    This is the contract between Polylogue (extraction) and Lynchpin (consumption).
    All fields are derived from conversation content — no external state needed.
    """

    # Identity
    conversation_id: str
    provider: str
    title: str | None
    created_at: datetime | None
    updated_at: datetime | None

    # Counts
    message_count: int
    substantive_count: int
    tool_use_count: int
    thinking_count: int
    word_count: int

    # Cost / timing
    total_cost_usd: float
    total_duration_ms: int

    # Tool breakdown
    tool_categories: dict[str, int]

    # Attribution
    repo_paths: tuple[str, ...]           # raw /realm/project/... paths
    cwd_paths: tuple[str, ...]
    branch_names: tuple[str, ...]
    file_paths_touched: tuple[str, ...]
    languages_detected: tuple[str, ...]
    canonical_projects: tuple[str, ...]   # resolved project names (e.g. "sinex")

    # Semantic extraction
    work_events: tuple[WorkEvent, ...]
    phases: tuple[SessionPhase, ...]
    decisions: tuple[Decision, ...] = ()

    # Temporal
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None
    wall_duration_ms: int = 0

    # Cost (harmonized — estimated for non-Claude Code providers)
    cost_is_estimated: bool = False       # False = actual Claude Code cost; True = estimated

    # Thread linkage (set by build_session_threads(), None when building individual profiles)
    thread_id: str | None = None          # root conversation_id of the continuation chain
    continuation_depth: int = 0           # 0 = root, 1 = first continuation, etc.

    # Metadata
    tags: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()       # inferred tags (project:X, kind:Y, etc.)
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
            "work_events": [
                {
                    "kind": we.kind.value,
                    "start_index": we.start_index,
                    "end_index": we.end_index,
                    "confidence": we.confidence,
                    "evidence": list(we.evidence),
                    "file_paths": list(we.file_paths),
                    "tools_used": list(we.tools_used),
                    "summary": we.summary,
                }
                for we in self.work_events
            ],
            "phases": [
                {
                    "kind": p.kind,
                    "start_time": p.start_time.isoformat() if p.start_time else None,
                    "end_time": p.end_time.isoformat() if p.end_time else None,
                    "message_range": list(p.message_range),
                    "duration_ms": p.duration_ms,
                    "tool_counts": p.tool_counts,
                    "word_count": p.word_count,
                }
                for p in self.phases
            ],
            "decisions": [
                {
                    "index": d.index,
                    "summary": d.summary,
                    "confidence": d.confidence,
                    "context": d.context,
                }
                for d in self.decisions
            ],
            "first_message_at": self.first_message_at.isoformat() if self.first_message_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "wall_duration_ms": self.wall_duration_ms,
            "canonical_projects": list(self.canonical_projects),
            "cost_is_estimated": self.cost_is_estimated,
            "thread_id": self.thread_id,
            "continuation_depth": self.continuation_depth,
            "tags": list(self.tags),
            "auto_tags": list(self.auto_tags),
            "is_continuation": self.is_continuation,
            "parent_id": self.parent_id,
        }


def build_session_profile(conversation: Conversation) -> SessionProfile:
    """Build a complete session profile from a conversation."""
    from polylogue.lib.pricing import harmonize_session_cost
    from polylogue.lib.tagging import infer_tags

    # Extract components
    attribution = extract_attribution(conversation)
    work_events = extract_work_events(conversation)
    phases = extract_phases(conversation)
    decisions = extract_decisions(conversation)

    # Count message types
    messages = list(conversation.messages)
    substantive_count = sum(1 for m in messages if m.is_substantive)
    tool_use_count = sum(1 for m in messages if m.is_tool_use)
    thinking_count = sum(1 for m in messages if m.is_thinking)

    # Aggregate tool categories
    tool_categories: Counter[str] = Counter()
    for msg in messages:
        harmonized = msg.harmonized
        if harmonized is None:
            continue
        calls = getattr(harmonized, "tool_calls", None)
        if calls:
            for tc in calls:
                tool_categories[tc.category.value] += 1

    # Temporal bounds
    timestamps = [m.timestamp for m in messages if m.timestamp]
    first_message_at = min(timestamps) if timestamps else None
    last_message_at = max(timestamps) if timestamps else None
    wall_duration_ms = 0
    if first_message_at and last_message_at:
        wall_duration_ms = max(int((last_message_at - first_message_at).total_seconds() * 1000), 0)

    cost_usd, cost_is_estimated = harmonize_session_cost(conversation)

    # Build partial profile without auto_tags so infer_tags can read canonical_projects
    partial = SessionProfile(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(messages),
        substantive_count=substantive_count,
        tool_use_count=tool_use_count,
        thinking_count=thinking_count,
        word_count=conversation.word_count,
        total_cost_usd=cost_usd,
        total_duration_ms=conversation.total_duration_ms,
        tool_categories=dict(tool_categories),
        repo_paths=attribution.repo_paths,
        cwd_paths=attribution.cwd_paths,
        branch_names=attribution.branch_names,
        file_paths_touched=attribution.file_paths_touched,
        languages_detected=attribution.languages_detected,
        canonical_projects=attribution.canonical_projects,
        work_events=tuple(work_events),
        phases=tuple(phases),
        decisions=tuple(decisions),
        first_message_at=first_message_at,
        last_message_at=last_message_at,
        wall_duration_ms=wall_duration_ms,
        cost_is_estimated=cost_is_estimated,
        tags=tuple(conversation.tags),
        is_continuation=conversation.is_continuation,
        parent_id=str(conversation.parent_id) if conversation.parent_id else None,
    )
    auto_tags = infer_tags(partial)
    return replace(partial, auto_tags=auto_tags)
