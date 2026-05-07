"""Sort-key-based session timing computation.

Computes per-session timing aggregates from message timestamps and content
blocks: thinking/output split, tool-call duration, latency percentiles, and
tool throughput.  All metrics are derived from existing stored sort keys and
work retroactively on all historical data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.core.json import JSONDocument, json_document

if TYPE_CHECKING:
    from polylogue.archive.message.models import Message


def _gap_ms(earlier: datetime | None, later: datetime | None) -> int:
    """Return the non-negative millisecond gap between two timestamps."""
    if earlier is None or later is None:
        return 0
    return max(int((later - earlier).total_seconds() * 1000), 0)


def _percentile(values: list[int], p: float) -> int:
    """Return the *p*-th percentile of a sorted list of ints (nearest-rank)."""
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    rank = max(0, int(math.ceil(p / 100.0 * len(values))) - 1)
    return values[rank]


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
                message.is_tool_use or any(block.get("type") == "tool_result" for block in message.content_blocks)
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
        "p50": _percentile(sorted_gaps, 50),
        "p95": _percentile(sorted_gaps, 95),
        "p99": _percentile(sorted_gaps, 99),
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


__all__ = [
    "SessionTimingFacts",
    "compute_session_timing",
]
