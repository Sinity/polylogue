"""Session temporal phase detection.

Segments a conversation into phases based on timestamp gaps between
messages. Each phase is classified by the dominant tool category
pattern within it.
"""

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
    from polylogue.lib.models import Conversation

_PHASE_GAP = timedelta(minutes=5)


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


def _classify_phase(tool_counts: dict[str, int], word_count: int) -> str:
    """Classify a phase by its dominant activity."""
    if not tool_counts:
        return "conversation"
    dominant = max(tool_counts, key=tool_counts.get)
    if dominant in ("file_edit", "file_write"):
        return "implementation"
    if dominant == "shell":
        return "execution"
    if dominant in ("file_read", "search"):
        return "exploration"
    if dominant == "git":
        return "version_control"
    if dominant in ("agent", "subagent"):
        return "delegation"
    if dominant == "web":
        return "web_research"
    return "mixed"


def extract_phases(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> list[SessionPhase]:
    """Extract temporal phases from a conversation.

    Phase boundaries are defined by gaps of 5+ minutes between
    consecutive messages with timestamps.
    """
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    messages = list(semantic_facts.message_facts)
    if not messages:
        return []

    phases: list[SessionPhase] = []
    phase_start_idx = 0
    phase_start_time: datetime | None = None
    prev_time: datetime | None = None

    for i, msg in enumerate(messages):
        ts = msg.timestamp
        if ts is None:
            continue

        if phase_start_time is None:
            phase_start_time = ts

        # Check for gap
        if prev_time is not None and (ts - prev_time) > _PHASE_GAP:
            # Emit previous phase
            phases.append(_build_phase(messages, phase_start_idx, i, phase_start_time, prev_time))
            phase_start_idx = i
            phase_start_time = ts

        prev_time = ts

    # Emit final phase
    if phase_start_time is not None:
        phases.append(_build_phase(messages, phase_start_idx, len(messages), phase_start_time, prev_time))

    return phases


def _build_phase(
    messages: list[MessageSemanticFacts],
    start_idx: int,
    end_idx: int,
    start_time: datetime | None,
    end_time: datetime | None,
) -> SessionPhase:
    """Build a SessionPhase from a message range."""
    tool_counts: Counter[str] = Counter()
    word_count = 0

    for msg in messages[start_idx:end_idx]:
        word_count += msg.word_count
        for tc in msg.tool_calls:
            tool_counts[tc.category.value] += 1

    duration_ms = 0
    if start_time and end_time:
        duration_ms = max(int((end_time - start_time).total_seconds() * 1000), 0)

    return SessionPhase(
        kind=_classify_phase(dict(tool_counts), word_count),
        start_time=start_time,
        end_time=end_time,
        canonical_session_date=(start_time or end_time).date() if (start_time or end_time) else None,
        message_range=(start_idx, end_idx),
        duration_ms=duration_ms,
        tool_counts=dict(tool_counts),
        word_count=word_count,
    )
