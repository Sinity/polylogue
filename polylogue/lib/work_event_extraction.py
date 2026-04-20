"""Heuristic work-event extraction from conversation message sequences."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from polylogue.lib.payload_coercion import (
    coerce_float,
    coerce_int,
    optional_date,
    optional_datetime,
    string_sequence,
)
from polylogue.lib.phase_extraction import SessionPhase, extract_phases
from polylogue.lib.semantic_facts import (
    ConversationSemanticFacts,
    MessageSemanticFacts,
    build_conversation_semantic_facts,
)
from polylogue.lib.session_payload_documents import WorkEventDocument

# Strip XML-like protocol artifacts from user messages before summarizing.
# Claude Code sessions contain <command-name>, <task-notification>,
# <local-command-caveat>, <system-reminder> etc. which are tool protocol
# noise, not human-readable content.
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_SUMMARY_PROTOCOL_BLOCKS = (
    ("<system-reminder>", re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)),
    ("<task-notification>", re.compile(r"<task-notification>.*?</task-notification>", re.DOTALL)),
    ("<local-command-caveat>", re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.DOTALL)),
    ("<local-command-stdout>", re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.DOTALL)),
    ("<command-name>", re.compile(r"<command-name>.*?</command-name>", re.DOTALL)),
    ("<command-message>", re.compile(r"<command-message>.*?</command-message>", re.DOTALL)),
    ("<command-args>", re.compile(r"<command-args>.*?</command-args>", re.DOTALL)),
)
_SUMMARY_MAX_TEXT_LEN = 100
_SUMMARY_MAX_JOINED_LEN = 200


def _normalize_summary_whitespace(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if "\n" not in stripped and "\r" not in stripped and "\t" not in stripped and "  " not in stripped:
        return stripped
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _clean_summary_text(text: str) -> str:
    """Extract human-readable content from user message text.

    Strips XML tags, system-reminder blocks, and protocol artifacts
    that appear in Claude Code session transcripts.
    """
    if not text:
        return ""
    if "<" not in text:
        cleaned = _normalize_summary_whitespace(text)
        return cleaned[:_SUMMARY_MAX_TEXT_LEN] if cleaned else ""

    cleaned = text
    for marker, pattern in _SUMMARY_PROTOCOL_BLOCKS:
        if marker in cleaned:
            cleaned = pattern.sub("", cleaned)
    if "<" in cleaned and ">" in cleaned:
        cleaned = _TAG_RE.sub("", cleaned)
    cleaned = _normalize_summary_whitespace(cleaned)
    return cleaned[:_SUMMARY_MAX_TEXT_LEN] if cleaned else ""


def _duration_ms_between(start_time: datetime | None, end_time: datetime | None) -> int:
    if start_time is None or end_time is None:
        return 0
    return max(int((end_time - start_time).total_seconds() * 1000), 0)


def _canonical_event_date(start_time: datetime | None, end_time: datetime | None) -> date | None:
    reference_time = start_time or end_time
    return reference_time.date() if reference_time is not None else None


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


WorkEventPayload: TypeAlias = WorkEventDocument


@dataclass(frozen=True)
class MessageRange:
    start: int
    end: int

    def iter_messages(self, messages: Sequence[MessageSemanticFacts]) -> Sequence[MessageSemanticFacts]:
        return messages[self.start : self.end]


@dataclass(frozen=True)
class WorkEventSignalBundle:
    action_category_counts: dict[str, int]
    normalized_user_text: str
    text_signal: WorkEventKind | None
    text_signal_name: str | None


@dataclass(frozen=True)
class WorkEventClassifiedRange:
    kind: WorkEventKind
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class WorkEventArtifacts:
    file_paths: tuple[str, ...]
    tools_used: tuple[str, ...]


@dataclass(frozen=True)
class WorkEventTiming:
    start_time: datetime | None
    end_time: datetime | None
    canonical_session_date: date | None
    duration_ms: int


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

    def to_dict(self) -> WorkEventPayload:
        return {
            "kind": self.kind.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "canonical_session_date": (
                self.canonical_session_date.isoformat() if self.canonical_session_date else None
            ),
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "file_paths": list(self.file_paths),
            "tools_used": list(self.tools_used),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: WorkEventPayload | Mapping[str, object]) -> WorkEvent:
        return cls(
            kind=WorkEventKind(str(payload["kind"])),
            start_index=coerce_int(payload.get("start_index"), 0),
            end_index=coerce_int(payload.get("end_index"), 0),
            start_time=optional_datetime(payload.get("start_time")),
            end_time=optional_datetime(payload.get("end_time")),
            canonical_session_date=optional_date(payload.get("canonical_session_date")),
            duration_ms=coerce_int(payload.get("duration_ms"), 0),
            confidence=coerce_float(payload.get("confidence"), 0.0),
            evidence=string_sequence(payload.get("evidence")),
            file_paths=string_sequence(payload.get("file_paths")),
            tools_used=string_sequence(payload.get("tools_used")),
            summary=str(payload.get("summary", "") or ""),
        )


if TYPE_CHECKING:
    from polylogue.lib.models import Conversation


_DEBUGGING_PATTERNS = (
    "error",
    "traceback",
    "failed",
    "bug",
    "fix",
    "broken",
    "crash",
    "exception",
    "stack trace",
    "panic",
)
_PLANNING_PATTERNS = (
    "plan",
    "design",
    "approach",
    "architecture",
    "strategy",
    "should we",
    "how should",
    "let's think",
)
_TESTING_PATTERNS = ("test", "pytest", "cargo test", "npm test", "assert", "spec")
_REVIEW_PATTERNS = ("review", "looks good", "lgtm", "nit", "suggestion", "feedback")
_REFACTORING_PATTERNS = ("refactor", "rename", "extract", "move", "restructure", "clean up")
_DOCUMENTATION_PATTERNS = ("document", "readme", "docstring", "comment", "explain")
_CONFIGURATION_PATTERNS = ("config", "toml", "yaml", "nix", "flake", "settings", "env")
_DATA_ANALYSIS_PATTERNS = ("data", "analysis", "query", "sql", "duckdb", "pandas", "plot", "chart", "csv")

# Text signals as weighted inputs (not short-circuit overrides).
# Order matters — first match wins. These are checked only after action
# evidence, or as a fallback when no action evidence is available.
TextSignalTable: TypeAlias = list[tuple[tuple[str, ...], WorkEventKind, str]]


_TEXT_SIGNAL_TABLE: TextSignalTable = [
    (_DEBUGGING_PATTERNS, WorkEventKind.DEBUGGING, "user_text_debugging"),
    (_PLANNING_PATTERNS, WorkEventKind.PLANNING, "user_text_planning"),
    (_TESTING_PATTERNS, WorkEventKind.TESTING, "user_text_testing"),
    (_REVIEW_PATTERNS, WorkEventKind.REVIEW, "user_text_review"),
    (_REFACTORING_PATTERNS, WorkEventKind.REFACTORING, "user_text_refactoring"),
    (_DOCUMENTATION_PATTERNS, WorkEventKind.DOCUMENTATION, "user_text_documentation"),
    (_CONFIGURATION_PATTERNS, WorkEventKind.CONFIGURATION, "user_text_configuration"),
    (_DATA_ANALYSIS_PATTERNS, WorkEventKind.DATA_ANALYSIS, "user_text_data_analysis"),
]


def _collect_range_signals(
    messages: Sequence[MessageSemanticFacts],
    message_range: MessageRange,
) -> WorkEventSignalBundle:
    action_category_counts: dict[str, int] = {}
    user_text_parts: list[str] = []

    for message in message_range.iter_messages(messages):
        if message.is_user and message.text and not message.is_context_dump:
            user_text_parts.append(message.text.lower())
        for action in message.action_events:
            category = action.kind.value
            action_category_counts[category] = action_category_counts.get(category, 0) + 1

    normalized_user_text = " ".join(part for part in user_text_parts if part).strip()
    text_signal: WorkEventKind | None = None
    text_signal_name: str | None = None
    if normalized_user_text:
        for patterns, kind, name in _TEXT_SIGNAL_TABLE:
            if any(pattern in normalized_user_text for pattern in patterns):
                text_signal = kind
                text_signal_name = name
                break
    return WorkEventSignalBundle(
        action_category_counts=action_category_counts,
        normalized_user_text=normalized_user_text,
        text_signal=text_signal,
        text_signal_name=text_signal_name,
    )


def _classify_range(signals: WorkEventSignalBundle) -> WorkEventClassifiedRange:
    """Classify a message range from action evidence plus weighted text signals."""

    evidence: list[str] = []
    edit_count = signals.action_category_counts.get("file_edit", 0) + signals.action_category_counts.get(
        "file_write",
        0,
    )
    read_count = signals.action_category_counts.get("file_read", 0)
    search_count = signals.action_category_counts.get("search", 0)
    shell_count = signals.action_category_counts.get("shell", 0)
    git_count = signals.action_category_counts.get("git", 0)
    agent_count = signals.action_category_counts.get("agent", 0) + signals.action_category_counts.get("subagent", 0)

    if edit_count >= 2:
        evidence.append("file_edits")
        if shell_count and signals.text_signal == WorkEventKind.TESTING:
            return WorkEventClassifiedRange(
                WorkEventKind.TESTING,
                0.7,
                tuple(evidence + ["shell_test", signals.text_signal_name or ""]),
            )
        if signals.text_signal == WorkEventKind.REFACTORING:
            return WorkEventClassifiedRange(
                WorkEventKind.REFACTORING,
                0.7,
                tuple(evidence + [signals.text_signal_name or ""]),
            )
        return WorkEventClassifiedRange(WorkEventKind.IMPLEMENTATION, 0.75, tuple(evidence))
    if agent_count >= 2 and edit_count == 0:
        return WorkEventClassifiedRange(WorkEventKind.PLANNING, 0.7, ("agent_orchestration",))
    if search_count >= 2 or (read_count >= 3 and edit_count == 0):
        return WorkEventClassifiedRange(WorkEventKind.RESEARCH, 0.7, ("search_or_read_dominant",))
    if git_count >= 1:
        return WorkEventClassifiedRange(WorkEventKind.REVIEW, 0.6, ("git_operations",))
    if shell_count >= 2:
        if signals.text_signal == WorkEventKind.TESTING:
            return WorkEventClassifiedRange(
                WorkEventKind.TESTING,
                0.65,
                ("shell_testing", signals.text_signal_name or ""),
            )
        if signals.text_signal == WorkEventKind.DEBUGGING:
            return WorkEventClassifiedRange(
                WorkEventKind.DEBUGGING,
                0.65,
                ("shell_debugging", signals.text_signal_name or ""),
            )
        return WorkEventClassifiedRange(WorkEventKind.IMPLEMENTATION, 0.5, ("shell_default",))

    if signals.text_signal is not None and signals.text_signal_name:
        return WorkEventClassifiedRange(signals.text_signal, 0.5, (signals.text_signal_name,))
    if not signals.action_category_counts:
        return WorkEventClassifiedRange(WorkEventKind.CONVERSATION, 0.6, ("no_tools",))
    return WorkEventClassifiedRange(WorkEventKind.IMPLEMENTATION, 0.4, ("weak_signal",))


def _compute_phase_ranges(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
    phases: Sequence[SessionPhase] | None = None,
) -> list[tuple[int, int]]:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    resolved_phases = list(phases) if phases is not None else extract_phases(conversation, facts=semantic_facts)
    if not resolved_phases:
        msg_count = len(semantic_facts.message_facts)
        return [(0, msg_count)] if msg_count > 0 else []

    ranges: list[tuple[int, int]] = []
    messages = list(semantic_facts.message_facts)
    for phase in resolved_phases:
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
            if prev_dominant is not None and dominant is not None and dominant != prev_dominant and i - sub_start >= 3:
                ranges.append((sub_start, i))
                sub_start = i
            prev_dominant = dominant
        if sub_start < end:
            ranges.append((sub_start, end))
    return ranges if ranges else [(0, len(messages))]


def _range_payload(range_bounds: tuple[int, int]) -> MessageRange:
    return MessageRange(start=range_bounds[0], end=range_bounds[1])


def _collect_range_artifacts(
    messages: Sequence[MessageSemanticFacts],
    message_range: MessageRange,
) -> WorkEventArtifacts:
    file_paths: list[str] = []
    tools_used: list[str] = []
    for message in message_range.iter_messages(messages):
        for action in message.action_events:
            tools_used.append(action.tool_name)
            file_paths.extend(action.affected_paths)
    return WorkEventArtifacts(
        file_paths=tuple(dict.fromkeys(file_paths)),
        tools_used=tuple(dict.fromkeys(tools_used)),
    )


def _build_range_summary(
    messages: Sequence[MessageSemanticFacts],
    message_range: MessageRange,
    *,
    fallback: WorkEventKind,
) -> str:
    summary_parts: list[str] = []
    summary_length = 0
    for message in message_range.iter_messages(messages):
        if not message.is_user or not message.text:
            continue
        cleaned_text = _clean_summary_text(message.text)
        if not cleaned_text:
            continue
        separator_length = 2 if summary_parts else 0
        summary_parts.append(cleaned_text)
        summary_length += separator_length + len(cleaned_text)
        if summary_length >= _SUMMARY_MAX_JOINED_LEN:
            break
    return "; ".join(summary_parts)[:_SUMMARY_MAX_JOINED_LEN] if summary_parts else fallback.value


def _range_timing(
    messages: Sequence[MessageSemanticFacts],
    message_range: MessageRange,
) -> WorkEventTiming:
    timestamps = [
        message.timestamp for message in message_range.iter_messages(messages) if message.timestamp is not None
    ]
    start_time = timestamps[0] if timestamps else None
    end_time = timestamps[-1] if timestamps else None
    return WorkEventTiming(
        start_time=start_time,
        end_time=end_time,
        canonical_session_date=_canonical_event_date(start_time, end_time),
        duration_ms=_duration_ms_between(start_time, end_time),
    )


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
                _duration_ms_between(prev.start_time or event.start_time, event.end_time or prev.end_time)
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
    phases: Sequence[SessionPhase] | None = None,
) -> list[WorkEvent]:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    messages = list(semantic_facts.message_facts)
    if not messages:
        return []

    events: list[WorkEvent] = []
    ranges = _compute_phase_ranges(conversation, facts=semantic_facts, phases=phases)
    for range_bounds in ranges:
        message_range = _range_payload(range_bounds)
        signals = _collect_range_signals(messages, message_range)
        classified = _classify_range(signals)
        artifacts = _collect_range_artifacts(messages, message_range)
        timing = _range_timing(messages, message_range)
        summary = _build_range_summary(messages, message_range, fallback=classified.kind)

        events.append(
            WorkEvent(
                kind=classified.kind,
                start_index=message_range.start,
                end_index=message_range.end,
                start_time=timing.start_time,
                end_time=timing.end_time,
                canonical_session_date=timing.canonical_session_date,
                duration_ms=timing.duration_ms,
                confidence=classified.confidence,
                evidence=classified.evidence,
                file_paths=artifacts.file_paths,
                tools_used=artifacts.tools_used,
                summary=summary,
            )
        )

    return _merge_adjacent(events)


__all__ = [
    "MessageRange",
    "WorkEvent",
    "WorkEventArtifacts",
    "WorkEventClassifiedRange",
    "WorkEventKind",
    "WorkEventPayload",
    "WorkEventSignalBundle",
    "WorkEventTiming",
    "extract_work_events",
]
