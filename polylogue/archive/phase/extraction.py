"""Session temporal phase extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from polylogue.archive.semantic.facts import (
    MessageSemanticFacts,
    SessionSemanticFacts,
    build_session_semantic_facts,
)

if TYPE_CHECKING:
    from datetime import datetime

    from polylogue.archive.models import Session


_PHASE_GAP = timedelta(minutes=5)
PHASE_IDLE_THRESHOLD_MS = int(_PHASE_GAP.total_seconds() * 1000)


@dataclass(frozen=True)
class SessionPhase:
    """A temporal phase within a session session.

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
    phase_idle_threshold_ms: int = PHASE_IDLE_THRESHOLD_MS
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()


def _build_phase(
    messages: Sequence[MessageSemanticFacts],
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
        phase_idle_threshold_ms=PHASE_IDLE_THRESHOLD_MS,
    )


def extract_phases(
    session: Session,
    *,
    facts: SessionSemanticFacts | None = None,
) -> list[SessionPhase]:
    semantic_facts = facts or build_session_semantic_facts(session)
    messages = semantic_facts.message_facts
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

    # Fallback for sources where messages carry no timestamps but
    # session_events do (codex pre-Dec-2025, hermes per-request dumps).
    # See #1624 — without this, session_phases is empty for 30% of the
    # archive, blinding `find_resume_candidates`, the phases lens, and
    # workflow_shape_distribution to anything codex/hermes.
    return _phases_from_session_events(session, messages)


def _phases_from_session_events(
    session: Session,
    messages: Sequence[MessageSemanticFacts],
) -> list[SessionPhase]:
    event_timestamps = sorted(event.timestamp for event in session.session_events if event.timestamp is not None)
    if not event_timestamps:
        return []

    phases: list[SessionPhase] = []
    phase_start_time = event_timestamps[0]
    prev_time = event_timestamps[0]
    for timestamp in event_timestamps[1:]:
        if (timestamp - prev_time) > _PHASE_GAP:
            phases.append(_build_phase(messages, 0, len(messages), phase_start_time, prev_time))
            phase_start_time = timestamp
        prev_time = timestamp
    phases.append(_build_phase(messages, 0, len(messages), phase_start_time, prev_time))
    return phases


__all__ = ["PHASE_IDLE_THRESHOLD_MS", "SessionPhase", "extract_phases"]
