"""Heuristic work-event extraction from session message sequences."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from polylogue.archive.phase.extraction import SessionPhase, extract_phases
from polylogue.archive.semantic.facts import (
    MessageSemanticFacts,
    SessionSemanticFacts,
    build_session_semantic_facts,
)
from polylogue.archive.session.documents import WorkEventDocument
from polylogue.core.payload_coercion import (
    coerce_float,
    coerce_int,
    optional_date,
    optional_datetime,
    string_sequence,
)

# Strip XML-like protocol artifacts from authored prompt messages before summarizing.
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
_SIGNAL_TEXT_PREVIEW_MAX_LEN = 512


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


class WorkEventHeuristicLabel(str, Enum):
    """Weak heuristic label for a message-range work event."""

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
    SESSION = "session"


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


@dataclass(frozen=True)
class WorkEventClassifiedRange:
    heuristic_label: WorkEventHeuristicLabel
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
    """A weakly labelled segment of work within a session."""

    heuristic_label: WorkEventHeuristicLabel
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
        start_time = self.start_time.isoformat() if self.start_time else None
        end_time = self.end_time.isoformat() if self.end_time else None
        canonical_session_date = self.canonical_session_date.isoformat() if self.canonical_session_date else None
        return {
            "heuristic_label": self.heuristic_label.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": start_time,
            "end_time": end_time,
            "canonical_session_date": canonical_session_date,
            "timing_provenance": _range_timing_provenance(start_time, end_time),
            "date_provenance": _date_provenance(canonical_session_date, start_time, end_time),
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
            heuristic_label=WorkEventHeuristicLabel(str(payload["heuristic_label"])),
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
    from polylogue.archive.models import Session


def _collect_range_signals(
    messages: Sequence[MessageSemanticFacts],
    message_range: MessageRange,
) -> WorkEventSignalBundle:
    action_category_counts: dict[str, int] = {}

    for message in message_range.iter_messages(messages):
        for action in message.actions:
            category = action.kind.value
            action_category_counts[category] = action_category_counts.get(category, 0) + 1

    return WorkEventSignalBundle(action_category_counts=action_category_counts)


def _classify_range(signals: WorkEventSignalBundle) -> WorkEventClassifiedRange:
    """Classify a message range from structural action evidence only.

    Prose-keyword text signals were deleted per the polylogue-ve9z decision
    (evidence-authority ladder): the never-measured keyword table may not
    refine labels. Where action evidence does not determine a finer label,
    the honest coarse label wins over a keyword guess.
    """

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
        return WorkEventClassifiedRange(WorkEventHeuristicLabel.IMPLEMENTATION, 0.75, ("file_edits",))
    if agent_count >= 2 and edit_count == 0:
        return WorkEventClassifiedRange(WorkEventHeuristicLabel.PLANNING, 0.7, ("agent_orchestration",))
    if search_count >= 2 or (read_count >= 3 and edit_count == 0):
        return WorkEventClassifiedRange(WorkEventHeuristicLabel.RESEARCH, 0.7, ("search_or_read_dominant",))
    if git_count >= 1:
        return WorkEventClassifiedRange(WorkEventHeuristicLabel.REVIEW, 0.6, ("git_operations",))
    if shell_count >= 2:
        return WorkEventClassifiedRange(WorkEventHeuristicLabel.IMPLEMENTATION, 0.5, ("shell_default",))
    if not signals.action_category_counts:
        return WorkEventClassifiedRange(WorkEventHeuristicLabel.SESSION, 0.6, ("no_tools",))
    return WorkEventClassifiedRange(WorkEventHeuristicLabel.IMPLEMENTATION, 0.4, ("weak_signal",))


def _compute_phase_ranges(
    session: Session,
    *,
    facts: SessionSemanticFacts | None = None,
    phases: Sequence[SessionPhase] | None = None,
) -> list[tuple[int, int]]:
    semantic_facts = facts or build_session_semantic_facts(session)
    resolved_phases = list(phases) if phases is not None else extract_phases(session, facts=semantic_facts)
    if not resolved_phases:
        msg_count = len(semantic_facts.message_facts)
        return [(0, msg_count)] if msg_count > 0 else []

    ranges: list[tuple[int, int]] = []
    messages = semantic_facts.message_facts
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
            actions = messages[i].actions
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
        for action in message.actions:
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
    fallback: WorkEventHeuristicLabel,
) -> str:
    summary_parts: list[str] = []
    summary_length = 0
    for message in message_range.iter_messages(messages):
        if not message.is_candidate_human_authored or not message.text:
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


def _range_timing_provenance(start_time: str | None, end_time: str | None) -> str:
    if start_time is not None and end_time is not None:
        return "timestamped_range"
    if start_time is not None:
        return "start_timestamp_only"
    if end_time is not None:
        return "end_timestamp_only"
    return "untimestamped"


def _date_provenance(canonical_session_date: str | None, start_time: str | None, end_time: str | None) -> str:
    if canonical_session_date is None:
        return "none"
    if start_time is not None or end_time is not None:
        return "event_timestamp"
    return "date_only"


def _merge_adjacent(events: list[WorkEvent]) -> list[WorkEvent]:
    if len(events) <= 1:
        return events

    merged: list[WorkEvent] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        if prev.heuristic_label != event.heuristic_label:
            merged.append(event)
            continue
        merged[-1] = WorkEvent(
            heuristic_label=prev.heuristic_label,
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
    session: Session,
    *,
    facts: SessionSemanticFacts | None = None,
    phases: Sequence[SessionPhase] | None = None,
) -> list[WorkEvent]:
    semantic_facts = facts or build_session_semantic_facts(session)
    messages = semantic_facts.message_facts
    if not messages:
        return []

    events: list[WorkEvent] = []
    ranges = _compute_phase_ranges(session, facts=semantic_facts, phases=phases)
    for range_bounds in ranges:
        message_range = _range_payload(range_bounds)
        signals = _collect_range_signals(messages, message_range)
        classified = _classify_range(signals)
        artifacts = _collect_range_artifacts(messages, message_range)
        timing = _range_timing(messages, message_range)
        summary = _build_range_summary(messages, message_range, fallback=classified.heuristic_label)

        events.append(
            WorkEvent(
                heuristic_label=classified.heuristic_label,
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
    "WorkEventHeuristicLabel",
    "WorkEventPayload",
    "WorkEventSignalBundle",
    "WorkEventTiming",
    "extract_work_events",
]
