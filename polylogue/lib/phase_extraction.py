"""Session temporal phase extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from polylogue.lib.semantic_facts import (
    ConversationSemanticFacts,
    MessageSemanticFacts,
    build_conversation_semantic_facts,
)

if TYPE_CHECKING:
    from datetime import datetime

    from polylogue.lib.models import Conversation


@dataclass(frozen=True)
class SessionPhase:
    """A temporal phase within a conversation session."""

    kind: str
    start_time: datetime | None
    end_time: datetime | None
    canonical_session_date: date | None
    message_range: tuple[int, int]
    duration_ms: int
    tool_counts: dict[str, int]
    word_count: int
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()


_PHASE_GAP = timedelta(minutes=5)


def _classify_phase(tool_counts: dict[str, int]) -> tuple[str, float, tuple[str, ...]]:
    if not tool_counts:
        return "conversation", 0.55, ("no_tools",)
    dominant = max(tool_counts, key=tool_counts.get)
    dominant_count = int(tool_counts.get(dominant, 0) or 0)
    total = max(sum(tool_counts.values()), 1)
    confidence = round(min(0.95, 0.45 + (dominant_count / total) * 0.5), 3)
    evidence = (f"dominant_tool:{dominant}", f"dominant_ratio:{dominant_count}/{total}")
    if dominant in ("file_edit", "file_write"):
        return "implementation", confidence, evidence
    if dominant == "shell":
        return "execution", confidence, evidence
    if dominant in ("file_read", "search"):
        return "exploration", confidence, evidence
    if dominant == "git":
        return "version_control", confidence, evidence
    if dominant in ("agent", "subagent"):
        return "delegation", confidence, evidence
    if dominant == "web":
        return "web_research", confidence, evidence
    return "mixed", max(confidence - 0.1, 0.35), evidence


def _build_phase(
    messages: list[MessageSemanticFacts],
    start_idx: int,
    end_idx: int,
    start_time: datetime | None,
    end_time: datetime | None,
) -> SessionPhase:
    tool_counts: Counter[str] = Counter()
    word_count = 0
    for message in messages[start_idx:end_idx]:
        word_count += message.word_count
        for tool_call in message.tool_calls:
            tool_counts[tool_call.category.value] += 1

    duration_ms = 0
    if start_time and end_time:
        duration_ms = max(int((end_time - start_time).total_seconds() * 1000), 0)

    kind, confidence, evidence = _classify_phase(dict(tool_counts))
    return SessionPhase(
        kind=kind,
        start_time=start_time,
        end_time=end_time,
        canonical_session_date=(start_time or end_time).date() if (start_time or end_time) else None,
        message_range=(start_idx, end_idx),
        duration_ms=duration_ms,
        tool_counts=dict(tool_counts),
        word_count=word_count,
        confidence=confidence,
        evidence=evidence,
    )


def extract_phases(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> list[SessionPhase]:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    messages = list(semantic_facts.message_facts)
    if not messages:
        return []

    phases: list[SessionPhase] = []
    phase_start_idx = 0
    phase_start_time = None
    prev_time = None

    for index, message in enumerate(messages):
        timestamp = message.timestamp
        if timestamp is None:
            continue
        if phase_start_time is None:
            phase_start_time = timestamp
        if prev_time is not None and (timestamp - prev_time) > _PHASE_GAP:
            phases.append(_build_phase(messages, phase_start_idx, index, phase_start_time, prev_time))
            phase_start_idx = index
            phase_start_time = timestamp
        prev_time = timestamp

    if phase_start_time is not None:
        phases.append(_build_phase(messages, phase_start_idx, len(messages), phase_start_time, prev_time))

    return phases


__all__ = ["SessionPhase", "extract_phases"]
