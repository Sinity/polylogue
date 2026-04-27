"""Session temporal phase extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from polylogue.lib.semantic.facts import (
    ConversationSemanticFacts,
    MessageSemanticFacts,
    build_conversation_semantic_facts,
)

if TYPE_CHECKING:
    from datetime import datetime

    from polylogue.lib.models import Conversation


@dataclass(frozen=True)
class SessionPhase:
    """A temporal phase within a conversation session.

    Phases are time-gap-segmented intervals, not intent-classified.
    The `kind` field was removed — phases represent when activity
    happened, not what kind of activity it was.
    """

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

    ref_time = start_time or end_time
    return SessionPhase(
        start_time=start_time,
        end_time=end_time,
        canonical_session_date=ref_time.date() if ref_time else None,
        message_range=(start_idx, end_idx),
        duration_ms=duration_ms,
        tool_counts=dict(tool_counts),
        word_count=word_count,
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
