"""Heuristic work-event extraction from conversation message sequences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

from polylogue.lib.phase_extraction import extract_phases
from polylogue.lib.semantic_facts import (
    ConversationSemanticFacts,
    MessageSemanticFacts,
    build_conversation_semantic_facts,
)


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

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation


_DEBUGGING_PATTERNS = ("error", "traceback", "failed", "bug", "fix", "broken", "crash", "exception", "stack trace", "panic")
_PLANNING_PATTERNS = ("plan", "design", "approach", "architecture", "strategy", "should we", "how should", "let's think")
_TESTING_PATTERNS = ("test", "pytest", "cargo test", "npm test", "assert", "spec")
_REVIEW_PATTERNS = ("review", "looks good", "lgtm", "nit", "suggestion", "feedback")
_REFACTORING_PATTERNS = ("refactor", "rename", "extract", "move", "restructure", "clean up")
_DOCUMENTATION_PATTERNS = ("document", "readme", "docstring", "comment", "explain")
_CONFIGURATION_PATTERNS = ("config", "toml", "yaml", "nix", "flake", "settings", "env")
_DATA_ANALYSIS_PATTERNS = ("data", "analysis", "query", "sql", "duckdb", "pandas", "plot", "chart", "csv")


def _classify_message_range(
    messages: list[MessageSemanticFacts],
    start: int,
    end: int,
) -> tuple[WorkEventKind, float, list[str]]:
    category_counts: dict[str, int] = {}
    user_text = ""
    evidence: list[str] = []

    for i in range(start, end):
        msg = messages[i]
        if msg.is_user and msg.text and not msg.is_context_dump:
            user_text += " " + msg.text.lower()
        for action in msg.action_events:
            cat = action.kind.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

    text_lower = user_text.strip()
    if text_lower:
        if any(pattern in text_lower for pattern in _DEBUGGING_PATTERNS):
            return WorkEventKind.DEBUGGING, 0.8, ["user_text_debugging"]
        if any(pattern in text_lower for pattern in _PLANNING_PATTERNS):
            return WorkEventKind.PLANNING, 0.8, ["user_text_planning"]
        if any(pattern in text_lower for pattern in _TESTING_PATTERNS):
            return WorkEventKind.TESTING, 0.75, ["user_text_testing"]
        if any(pattern in text_lower for pattern in _REVIEW_PATTERNS):
            return WorkEventKind.REVIEW, 0.75, ["user_text_review"]
        if any(pattern in text_lower for pattern in _REFACTORING_PATTERNS):
            return WorkEventKind.REFACTORING, 0.75, ["user_text_refactoring"]
        if any(pattern in text_lower for pattern in _DOCUMENTATION_PATTERNS):
            return WorkEventKind.DOCUMENTATION, 0.7, ["user_text_documentation"]
        if any(pattern in text_lower for pattern in _CONFIGURATION_PATTERNS):
            return WorkEventKind.CONFIGURATION, 0.7, ["user_text_configuration"]
        if any(pattern in text_lower for pattern in _DATA_ANALYSIS_PATTERNS):
            return WorkEventKind.DATA_ANALYSIS, 0.7, ["user_text_data_analysis"]

    edit_count = category_counts.get("file_edit", 0) + category_counts.get("file_write", 0)
    read_count = category_counts.get("file_read", 0)
    search_count = category_counts.get("search", 0)
    shell_count = category_counts.get("shell", 0)
    git_count = category_counts.get("git", 0)
    agent_count = category_counts.get("agent", 0) + category_counts.get("subagent", 0)

    if edit_count >= 2:
        evidence.append("file_edits")
        if shell_count and any(pattern in user_text for pattern in _TESTING_PATTERNS):
            return WorkEventKind.TESTING, 0.7, evidence + ["shell_test"]
        return WorkEventKind.IMPLEMENTATION, 0.75, evidence
    if agent_count >= 2 and edit_count == 0:
        return WorkEventKind.PLANNING, 0.7, ["agent_orchestration"]
    if search_count >= 2 or (read_count >= 3 and edit_count == 0):
        return WorkEventKind.RESEARCH, 0.7, ["search_or_read_dominant"]
    if git_count >= 1:
        return WorkEventKind.REVIEW, 0.6, ["git_operations"]
    if shell_count >= 2:
        if any(pattern in user_text for pattern in _TESTING_PATTERNS):
            return WorkEventKind.TESTING, 0.65, ["shell_testing"]
        if any(pattern in user_text for pattern in _DEBUGGING_PATTERNS):
            return WorkEventKind.DEBUGGING, 0.65, ["shell_debugging"]
        return WorkEventKind.IMPLEMENTATION, 0.5, ["shell_default"]
    if not category_counts:
        return WorkEventKind.CONVERSATION, 0.6, ["no_tools"]
    return WorkEventKind.IMPLEMENTATION, 0.4, ["weak_signal"]


def _compute_phase_ranges(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> list[tuple[int, int]]:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    phases = extract_phases(conversation, facts=semantic_facts)
    if not phases:
        msg_count = len(semantic_facts.message_facts)
        return [(0, msg_count)] if msg_count > 0 else []

    ranges: list[tuple[int, int]] = []
    messages = list(semantic_facts.message_facts)
    for phase in phases:
        start, end = phase.message_range
        if start >= end:
            continue
        if end - start <= 15:
            ranges.append((start, end))
            continue
        sub_start = start
        prev_dominant = None
        for i in range(start, end):
            actions = messages[i].action_events
            dominant = actions[0].kind.value if actions else None
            if (
                prev_dominant is not None
                and dominant is not None
                and dominant != prev_dominant
                and i - sub_start >= 3
            ):
                ranges.append((sub_start, i))
                sub_start = i
            prev_dominant = dominant
        if sub_start < end:
            ranges.append((sub_start, end))
    return ranges if ranges else [(0, len(messages))]


def _merge_adjacent(events: list[WorkEvent]) -> list[WorkEvent]:
    if len(events) <= 1:
        return events

    merged: list[WorkEvent] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        if prev.kind != event.kind:
            merged.append(event)
            continue
        merged[-1] = WorkEvent(
            kind=prev.kind,
            start_index=prev.start_index,
            end_index=event.end_index,
            start_time=prev.start_time or event.start_time,
            end_time=event.end_time or prev.end_time,
            canonical_session_date=prev.canonical_session_date or event.canonical_session_date,
            duration_ms=(
                max(
                    int(
                        (
                            (event.end_time or prev.end_time)
                            - (prev.start_time or event.start_time)
                        ).total_seconds()
                        * 1000
                    ),
                    0,
                )
                if (prev.start_time or event.start_time) and (event.end_time or prev.end_time)
                else prev.duration_ms + event.duration_ms
            ),
            confidence=max(prev.confidence, event.confidence),
            evidence=tuple(dict.fromkeys(prev.evidence + event.evidence)),
            file_paths=tuple(dict.fromkeys(prev.file_paths + event.file_paths)),
            tools_used=tuple(dict.fromkeys(prev.tools_used + event.tools_used)),
            summary=prev.summary,
        )
    return merged


def extract_work_events(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> list[WorkEvent]:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    messages = list(semantic_facts.message_facts)
    if not messages:
        return []

    events: list[WorkEvent] = []
    ranges = _compute_phase_ranges(conversation, facts=semantic_facts)
    for chunk_start, chunk_end in ranges:
        kind, confidence, evidence = _classify_message_range(messages, chunk_start, chunk_end)
        file_paths: list[str] = []
        tools_used: list[str] = []
        for index in range(chunk_start, chunk_end):
            for action in messages[index].action_events:
                tools_used.append(action.tool_name)
                file_paths.extend(action.affected_paths)

        user_texts = [
            message.text[:100]
            for message in messages[chunk_start:chunk_end]
            if message.is_user and message.text
        ]
        summary = "; ".join(user_texts)[:200] if user_texts else kind.value
        timestamps = [
            message.timestamp
            for message in messages[chunk_start:chunk_end]
            if message.timestamp is not None
        ]
        start_time = timestamps[0] if timestamps else None
        end_time = timestamps[-1] if timestamps else None
        duration_ms = 0
        if start_time and end_time:
            duration_ms = max(int((end_time - start_time).total_seconds() * 1000), 0)

        events.append(
            WorkEvent(
                kind=kind,
                start_index=chunk_start,
                end_index=chunk_end,
                start_time=start_time,
                end_time=end_time,
                canonical_session_date=(start_time or end_time).date() if (start_time or end_time) else None,
                duration_ms=duration_ms,
                confidence=confidence,
                evidence=tuple(dict.fromkeys(evidence)),
                file_paths=tuple(dict.fromkeys(file_paths)),
                tools_used=tuple(dict.fromkeys(tools_used)),
                summary=summary,
            )
        )

    return _merge_adjacent(events)


__all__ = ["WorkEvent", "WorkEventKind", "extract_work_events"]
