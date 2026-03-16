"""Work event extraction from conversation message sequences.

Classifies contiguous message ranges into work event kinds (implementation,
debugging, research, etc.) based on tool call categories and message content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, Message
    from polylogue.lib.viewports import ToolCall, ToolCategory


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


# Patterns for classifying user messages
_DEBUGGING_PATTERNS = ("error", "traceback", "failed", "bug", "fix", "broken", "crash", "exception", "stack trace", "panic")
_PLANNING_PATTERNS = ("plan", "design", "approach", "architecture", "strategy", "should we", "how should", "let's think")
_TESTING_PATTERNS = ("test", "pytest", "cargo test", "npm test", "assert", "spec")
_REVIEW_PATTERNS = ("review", "looks good", "lgtm", "nit", "suggestion", "feedback")
_REFACTORING_PATTERNS = ("refactor", "rename", "extract", "move", "restructure", "clean up")
_DOCUMENTATION_PATTERNS = ("document", "readme", "docstring", "comment", "explain")
_CONFIGURATION_PATTERNS = ("config", "toml", "yaml", "nix", "flake", "settings", "env")
_DATA_ANALYSIS_PATTERNS = ("data", "analysis", "query", "sql", "duckdb", "pandas", "plot", "chart", "csv")


def _get_tool_calls(message: Message) -> list[ToolCall]:
    """Extract tool calls from a message's harmonized viewport."""
    harmonized = message.harmonized
    if harmonized is None:
        return []
    calls = getattr(harmonized, "tool_calls", None)
    return list(calls) if calls else []


def _classify_message_range(
    messages: list[Message],
    start: int,
    end: int,
) -> tuple[WorkEventKind, float, list[str]]:
    """Classify a contiguous range of messages into a work event kind."""
    from polylogue.lib.viewports import ToolCategory

    category_counts: dict[str, int] = {}
    all_tools: list[str] = []
    user_text = ""
    evidence: list[str] = []

    for i in range(start, end):
        msg = messages[i]
        if msg.is_user and msg.text:
            user_text += " " + msg.text.lower()
        for tc in _get_tool_calls(msg):
            cat = tc.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            all_tools.append(tc.name)

    # Check user text patterns first (strongest signal)
    text_lower = user_text.strip()
    if text_lower:
        if any(p in text_lower for p in _DEBUGGING_PATTERNS):
            return WorkEventKind.DEBUGGING, 0.8, ["user_text_debugging"]
        if any(p in text_lower for p in _PLANNING_PATTERNS):
            return WorkEventKind.PLANNING, 0.8, ["user_text_planning"]
        if any(p in text_lower for p in _TESTING_PATTERNS):
            return WorkEventKind.TESTING, 0.75, ["user_text_testing"]
        if any(p in text_lower for p in _REVIEW_PATTERNS):
            return WorkEventKind.REVIEW, 0.75, ["user_text_review"]
        if any(p in text_lower for p in _REFACTORING_PATTERNS):
            return WorkEventKind.REFACTORING, 0.75, ["user_text_refactoring"]
        if any(p in text_lower for p in _DOCUMENTATION_PATTERNS):
            return WorkEventKind.DOCUMENTATION, 0.7, ["user_text_documentation"]
        if any(p in text_lower for p in _CONFIGURATION_PATTERNS):
            return WorkEventKind.CONFIGURATION, 0.7, ["user_text_configuration"]
        if any(p in text_lower for p in _DATA_ANALYSIS_PATTERNS):
            return WorkEventKind.DATA_ANALYSIS, 0.7, ["user_text_data_analysis"]

    # Fall back to tool category patterns
    edit_count = category_counts.get("file_edit", 0) + category_counts.get("file_write", 0)
    read_count = category_counts.get("file_read", 0)
    search_count = category_counts.get("search", 0)
    shell_count = category_counts.get("shell", 0)
    git_count = category_counts.get("git", 0)

    if edit_count >= 2:
        evidence.append("file_edits")
        # Check if shell commands contain test patterns
        if shell_count and any(p in user_text for p in _TESTING_PATTERNS):
            return WorkEventKind.TESTING, 0.7, evidence + ["shell_test"]
        return WorkEventKind.IMPLEMENTATION, 0.75, evidence
    if search_count >= 2 or (read_count >= 3 and edit_count == 0):
        return WorkEventKind.RESEARCH, 0.7, ["search_or_read_dominant"]
    if git_count >= 1:
        return WorkEventKind.REVIEW, 0.6, ["git_operations"]
    if shell_count >= 2:
        # Shell-heavy could be testing or debugging
        if any(p in user_text for p in _TESTING_PATTERNS):
            return WorkEventKind.TESTING, 0.65, ["shell_testing"]
        if any(p in user_text for p in _DEBUGGING_PATTERNS):
            return WorkEventKind.DEBUGGING, 0.65, ["shell_debugging"]
        return WorkEventKind.IMPLEMENTATION, 0.5, ["shell_default"]

    # No strong signal — conversation
    if not category_counts:
        return WorkEventKind.CONVERSATION, 0.6, ["no_tools"]

    return WorkEventKind.IMPLEMENTATION, 0.4, ["weak_signal"]


def _compute_phase_ranges(conversation: Conversation) -> list[tuple[int, int]]:
    """Compute chunk ranges aligned to phase boundaries.

    Uses temporal phase detection (5-min gaps) to find natural work
    boundaries. Within long phases (>15 messages), sub-chunks at
    tool-category shift points.
    """
    from polylogue.lib.phases import extract_phases

    phases = extract_phases(conversation)
    if not phases:
        # Fallback: single chunk for entire conversation
        msg_count = len(list(conversation.messages))
        return [(0, msg_count)] if msg_count > 0 else []

    ranges: list[tuple[int, int]] = []
    messages = list(conversation.messages)

    for phase in phases:
        start, end = phase.message_range
        if start >= end:
            continue
        # If phase is short enough, use it directly
        if end - start <= 15:
            ranges.append((start, end))
            continue
        # Sub-chunk long phases at tool-category transitions
        sub_start = start
        prev_dominant = None
        for i in range(start, end):
            calls = _get_tool_calls(messages[i])
            if calls:
                dominant = calls[0].category.value
            else:
                dominant = None
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


def extract_work_events(conversation: Conversation) -> list[WorkEvent]:
    """Extract work events from a conversation.

    Uses phase-boundary-aligned chunking: temporal phases (5-min gaps)
    define natural work boundaries. Long phases are sub-chunked at
    tool-category shift points.
    """
    messages = list(conversation.messages)
    if not messages:
        return []

    events: list[WorkEvent] = []
    ranges = _compute_phase_ranges(conversation)

    for chunk_start, chunk_end in ranges:
        kind, confidence, evidence = _classify_message_range(messages, chunk_start, chunk_end)

        # Collect file paths and tools from this range
        file_paths: list[str] = []
        tools_used: list[str] = []
        for j in range(chunk_start, chunk_end):
            for tc in _get_tool_calls(messages[j]):
                tools_used.append(tc.name)
                file_paths.extend(tc.affected_paths)

        # Build summary from user messages in range
        user_texts = [
            m.text[:100] for m in messages[chunk_start:chunk_end]
            if m.is_user and m.text
        ]
        summary = "; ".join(user_texts)[:200] if user_texts else kind.value

        events.append(WorkEvent(
            kind=kind,
            start_index=chunk_start,
            end_index=chunk_end,
            confidence=confidence,
            evidence=tuple(dict.fromkeys(evidence)),
            file_paths=tuple(dict.fromkeys(file_paths)),
            tools_used=tuple(dict.fromkeys(tools_used)),
            summary=summary,
        ))

    # Merge adjacent events of the same kind
    if len(events) <= 1:
        return events

    merged: list[WorkEvent] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        if prev.kind == event.kind:
            merged[-1] = WorkEvent(
                kind=prev.kind,
                start_index=prev.start_index,
                end_index=event.end_index,
                confidence=max(prev.confidence, event.confidence),
                evidence=tuple(dict.fromkeys(prev.evidence + event.evidence)),
                file_paths=tuple(dict.fromkeys(prev.file_paths + event.file_paths)),
                tools_used=tuple(dict.fromkeys(prev.tools_used + event.tools_used)),
                summary=prev.summary,
            )
        else:
            merged.append(event)

    return merged
