"""Sort-key-based session timing computation.

Computes per-session timing aggregates from message timestamps and content
blocks: thinking/output split, tool-call duration, latency percentiles, and
tool throughput.  All metrics are derived from existing stored sort keys and
work retroactively on all historical data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.stats import percentile

if TYPE_CHECKING:
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.events import SessionEvent


TOOL_ACTIVE_START_EVENT_TYPES = frozenset({"function_call", "custom_tool_call", "web_search_call", "tool_search_call"})
TOOL_ACTIVE_OUTPUT_EVENT_TYPES = frozenset(
    {"function_call_output", "custom_tool_call_output", "web_search_output", "tool_search_output"}
)


def _gap_ms(earlier: datetime | None, later: datetime | None) -> int:
    """Return the non-negative millisecond gap between two timestamps."""
    if earlier is None or later is None:
        return 0
    return max(int((later - earlier).total_seconds() * 1000), 0)


@dataclass(frozen=True, slots=True)
class SessionTimingFacts:
    """Aggregated timing facts derived from intra-session message gaps."""

    thinking_duration_ms: int = 0
    output_duration_ms: int = 0
    tool_duration_ms: int = 0
    latency_percentiles_ms: dict[str, int] = field(default_factory=dict)
    tool_calls_per_minute: float = 0.0
    timing_provenance: str = "sort_key_estimated"

    def __post_init__(self) -> None:
        if not isinstance(self.latency_percentiles_ms, dict):
            object.__setattr__(self, "latency_percentiles_ms", dict(self.latency_percentiles_ms or {}))

    @property
    def computed_total_ms(self) -> int:
        """Sum of all computed timing categories.

        This may be less than wall_duration_ms because context-dump,
        system-message, and untimestamped gaps are excluded.
        """
        return self.thinking_duration_ms + self.output_duration_ms + self.tool_duration_ms

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "thinking_duration_ms": self.thinking_duration_ms,
                "output_duration_ms": self.output_duration_ms,
                "tool_duration_ms": self.tool_duration_ms,
                "latency_percentiles_ms": dict(self.latency_percentiles_ms),
                "tool_calls_per_minute": self.tool_calls_per_minute,
                "timing_provenance": self.timing_provenance,
            }
        )


def compute_session_timing(
    messages: Sequence[Message],
    *,
    tool_use_count: int = 0,
    wall_duration_ms: int = 0,
) -> SessionTimingFacts:
    """Compute timing aggregates from a chronological message sequence.

    Messages are assumed to be in chronological order (as returned by the
    storage layer, sorted by ``sort_key``).  Timing provenance defaults to
    ``"sort_key_estimated"`` because these values are derived from inter-
    message gaps rather than hook-level precise measurements.
    """
    if not messages:
        return SessionTimingFacts()

    # ------------------------------------------------------------------
    # Per-message gap computation
    # ------------------------------------------------------------------
    gaps: list[int] = []
    thinking_duration_ms = 0
    output_duration_ms = 0
    tool_duration_ms = 0
    prev_message: Message | None = None

    for message in messages:
        if prev_message is not None:
            gap = _gap_ms(prev_message.timestamp, message.timestamp)
            gaps.append(gap)

            # Classify the gap based on *this* message's content.
            # A tool-call gap is the time before a tool_result message.
            if (
                message.is_tool_use or any(block.get("type") == "tool_result" for block in message.blocks)
            ) and prev_message.is_assistant:
                tool_duration_ms += gap
            elif message.is_thinking and not message.is_tool_use:
                thinking_duration_ms += gap
            elif message.is_assistant and message.is_substantive:
                # Assistant response that is neither thinking-only nor tool:
                # treat as output.
                output_duration_ms += gap

        prev_message = message

    # ------------------------------------------------------------------
    # Latency percentiles  (p50, p95, p99)
    # ------------------------------------------------------------------
    sorted_gaps = sorted(g for g in gaps if g > 0)
    latency_percentiles_ms = {
        "p50": int(percentile(sorted_gaps, 0.5, method="nearest")),
        "p95": int(percentile(sorted_gaps, 0.95, method="nearest")),
        "p99": int(percentile(sorted_gaps, 0.99, method="nearest")),
    }

    # ------------------------------------------------------------------
    # Tool calls per minute
    # ------------------------------------------------------------------
    tool_calls_per_minute = 0.0
    if tool_use_count > 0 and wall_duration_ms > 0:
        minutes = wall_duration_ms / 60_000.0
        if minutes > 0:
            tool_calls_per_minute = round(tool_use_count / minutes, 2)

    return SessionTimingFacts(
        thinking_duration_ms=thinking_duration_ms,
        output_duration_ms=output_duration_ms,
        tool_duration_ms=tool_duration_ms,
        latency_percentiles_ms=latency_percentiles_ms,
        tool_calls_per_minute=tool_calls_per_minute,
        timing_provenance="sort_key_estimated",
    )


@dataclass(frozen=True, slots=True)
class SessionLatencyProfileFacts:
    """Per-session latency aggregate derived from messages and session events."""

    median_tool_call_ms: int = 0
    p90_tool_call_ms: int = 0
    max_tool_call_ms: int = 0
    stuck_tool_count: int = 0
    median_agent_response_ms: int = 0
    median_user_response_ms: int = 0
    tool_call_count_by_category: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_count_by_category, dict):
            object.__setattr__(
                self,
                "tool_call_count_by_category",
                dict(self.tool_call_count_by_category or {}),
            )

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "median_tool_call_ms": self.median_tool_call_ms,
                "p90_tool_call_ms": self.p90_tool_call_ms,
                "max_tool_call_ms": self.max_tool_call_ms,
                "stuck_tool_count": self.stuck_tool_count,
                "median_agent_response_ms": self.median_agent_response_ms,
                "median_user_response_ms": self.median_user_response_ms,
                "tool_call_count_by_category": dict(self.tool_call_count_by_category),
            }
        )


def _median_ms(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) // 2


def compute_tool_active_duration_ms(session_events: Sequence[SessionEvent]) -> int:
    """Sum paired provider tool-call windows with explicit event timestamps.

    This is a session-event measurement, not an inter-message estimate: a
    start event opens a tool window and the matching output event closes it.
    Unpaired or untimestamped events are ignored because they cannot establish
    a bounded active interval.
    """

    pending_by_call_id: dict[str, datetime] = {}
    pending_without_call_id: list[datetime] = []
    total_ms = 0
    for event in sorted(session_events, key=lambda item: item.event_index):
        if event.timestamp is None:
            continue
        event_type = str(event.event_type).strip().lower()
        call_id_value = event.payload.get("call_id")
        call_id = call_id_value.strip() if isinstance(call_id_value, str) and call_id_value.strip() else None
        if event_type in TOOL_ACTIVE_START_EVENT_TYPES:
            if call_id is not None:
                pending_by_call_id[call_id] = event.timestamp
            else:
                pending_without_call_id.append(event.timestamp)
            continue
        if event_type not in TOOL_ACTIVE_OUTPUT_EVENT_TYPES:
            continue
        started_at: datetime | None = None
        if call_id is not None:
            started_at = pending_by_call_id.pop(call_id, None)
        elif pending_without_call_id:
            started_at = pending_without_call_id.pop(0)
        if started_at is not None:
            total_ms += max(int((event.timestamp - started_at).total_seconds() * 1000), 0)
    return total_ms


def _provider_tool_latencies(
    session_events: Sequence[SessionEvent],
    *,
    session_end: datetime | None,
    stuck_threshold_ms: int,
) -> tuple[list[int], int]:
    pending_by_call_id: dict[str, datetime] = {}
    pending_without_call_id: list[datetime] = []
    durations: list[int] = []
    for event in sorted(session_events, key=lambda item: item.event_index):
        if event.timestamp is None:
            continue
        event_type = str(event.event_type).strip().lower()
        call_id_value = event.payload.get("call_id")
        call_id = call_id_value.strip() if isinstance(call_id_value, str) and call_id_value.strip() else None
        if event_type in TOOL_ACTIVE_START_EVENT_TYPES:
            if call_id is not None:
                pending_by_call_id[call_id] = event.timestamp
            else:
                pending_without_call_id.append(event.timestamp)
            continue
        if event_type not in TOOL_ACTIVE_OUTPUT_EVENT_TYPES:
            continue
        started_at: datetime | None = None
        if call_id is not None:
            started_at = pending_by_call_id.pop(call_id, None)
        elif pending_without_call_id:
            started_at = pending_without_call_id.pop(0)
        if started_at is not None:
            durations.append(max(int((event.timestamp - started_at).total_seconds() * 1000), 0))
    stuck_count = 0
    if session_end is not None:
        for started_at in (*pending_by_call_id.values(), *pending_without_call_id):
            if max(int((session_end - started_at).total_seconds() * 1000), 0) > stuck_threshold_ms:
                stuck_count += 1
    return durations, stuck_count


def _message_response_latencies(messages: Sequence[Message]) -> tuple[list[int], list[int]]:
    agent_response_ms: list[int] = []
    user_response_ms: list[int] = []
    timestamped = [message for message in messages if message.timestamp is not None]
    for index, message in enumerate(timestamped):
        next_message = next(
            (
                candidate
                for candidate in timestamped[index + 1 :]
                if candidate.is_candidate_human_authored or candidate.is_assistant
            ),
            None,
        )
        if next_message is None or next_message.timestamp is None or message.timestamp is None:
            continue
        delta_ms = max(int((next_message.timestamp - message.timestamp).total_seconds() * 1000), 0)
        if message.is_candidate_human_authored and next_message.is_assistant:
            agent_response_ms.append(delta_ms)
        elif message.is_assistant and next_message.is_candidate_human_authored and delta_ms <= 1_800_000:
            user_response_ms.append(delta_ms)
    return agent_response_ms, user_response_ms


def compute_session_latency_profile(
    messages: Sequence[Message],
    session_events: Sequence[SessionEvent],
    *,
    session_end: datetime | None = None,
    tool_call_count_by_category: dict[str, int] | None = None,
    stuck_threshold_ms: int = 300_000,
) -> SessionLatencyProfileFacts:
    """Compute per-session latency facts without causal interpretation."""

    tool_durations, stuck_tool_count = _provider_tool_latencies(
        session_events,
        session_end=session_end,
        stuck_threshold_ms=stuck_threshold_ms,
    )
    agent_response_ms, user_response_ms = _message_response_latencies(messages)
    sorted_tool_durations = sorted(tool_durations)
    return SessionLatencyProfileFacts(
        median_tool_call_ms=_median_ms(tool_durations),
        p90_tool_call_ms=int(percentile(sorted_tool_durations, 0.9, method="nearest")),
        max_tool_call_ms=max(tool_durations) if tool_durations else 0,
        stuck_tool_count=stuck_tool_count,
        median_agent_response_ms=_median_ms(agent_response_ms),
        median_user_response_ms=_median_ms(user_response_ms),
        tool_call_count_by_category=dict(tool_call_count_by_category or {}),
    )


__all__ = [
    "SessionTimingFacts",
    "SessionLatencyProfileFacts",
    "compute_session_latency_profile",
    "compute_session_timing",
    "compute_tool_active_duration_ms",
]
