"""Session temporal phase extraction."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import TYPE_CHECKING

from polylogue.lib.phase_models import SessionPhase
from polylogue.lib.semantic_facts import (
    ConversationSemanticFacts,
    MessageSemanticFacts,
    build_conversation_semantic_facts,
)

if TYPE_CHECKING:
    from datetime import datetime

    from polylogue.lib.models import Conversation


_PHASE_GAP = timedelta(minutes=5)


def _classify_phase(tool_counts: dict[str, int]) -> str:
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

    return SessionPhase(
        kind=_classify_phase(dict(tool_counts)),
        start_time=start_time,
        end_time=end_time,
        canonical_session_date=(start_time or end_time).date() if (start_time or end_time) else None,
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


__all__ = ["extract_phases"]
