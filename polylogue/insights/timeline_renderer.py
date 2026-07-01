"""Per-session timeline renderer with timing-fidelity tags.

Renders a chronological timeline for a single session by merging the two
already-materialized timeline read models — work events and session phases —
and tags each entry with a *fidelity* label that distinguishes hook-precise
timing from sort-key-reconstructed timing.

The renderer is a pure function over insight payloads. It does not query
storage and does not recompute timing. Storage access lives in the CLI / API
caller. See ``polylogue/cli/commands/insights.py`` for the wiring.

Fidelity tag derivation
-----------------------

Every materialized timeline row carries a ``timing_provenance`` string on
its evidence payload. The mapping is:

- ``timestamped_range`` -> ``hook`` — both endpoints came from a recorded
  timestamp (the precise case: hook-event timing or first-class message
  timestamps that bound the entry exactly).
- ``start_timestamp_only`` / ``end_timestamp_only`` / ``untimestamped`` ->
  ``sort_key`` — at least one endpoint was reconstructed from message
  sort-key ordering rather than read from a recorded timestamp.

The invariant ``hook -> timestamped_range`` is preserved across the
renderer surface. The tag is the user-visible name; the underlying
``timing_provenance`` value is preserved in the JSON payload so callers
can recover the exact provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from polylogue.insights.archive import SessionPhaseInsight, SessionWorkEventInsight

FidelityTag = Literal["hook", "sort_key"]

# Public mapping from raw timing_provenance to the user-visible fidelity tag.
# Anything not present here is treated as ``sort_key``: the precise-timing
# branch is the explicitly-recognized one, so unknown provenance values do
# not silently get promoted.
_HOOK_PROVENANCES: frozenset[str] = frozenset({"timestamped_range"})


def fidelity_for(timing_provenance: str | None) -> FidelityTag:
    """Map a raw ``timing_provenance`` value to a user-visible fidelity tag."""
    if timing_provenance is not None and timing_provenance in _HOOK_PROVENANCES:
        return "hook"
    return "sort_key"


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """A single rendered timeline row.

    ``source`` distinguishes work-event entries (semantic classification:
    debugging, refactor, ...) from session-phase entries (gap-segmented
    activity intervals). Both are materialized from the same underlying
    message stream and share the fidelity-tag derivation.
    """

    source: Literal["work_event", "phase"]
    entry_id: str
    session_id: str
    kind: str
    summary: str
    start_time: str | None
    end_time: str | None
    duration_ms: int
    fidelity: FidelityTag
    timing_provenance: str
    confidence: float = 0.0
    tools_used: tuple[str, ...] = ()
    file_paths: tuple[str, ...] = ()
    word_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "summary": self.summary,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "fidelity": self.fidelity,
            "timing_provenance": self.timing_provenance,
            "confidence": self.confidence,
            "tools_used": list(self.tools_used),
            "file_paths": list(self.file_paths),
            "word_count": self.word_count,
        }


@dataclass(frozen=True, slots=True)
class SessionTimeline:
    """Rendered session timeline payload.

    ``entries`` is ordered chronologically (by ``start_time``, with entries
    that lack ``start_time`` placed last in their original input order).
    ``fidelity_counts`` is a per-tag count surfaced so callers can warn when
    a session is dominated by sort-key-reconstructed timing.
    """

    session_id: str
    entries: tuple[TimelineEntry, ...]
    fidelity_counts: dict[FidelityTag, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "fidelity_counts": dict(self.fidelity_counts),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _work_event_entry(insight: SessionWorkEventInsight) -> TimelineEntry:
    evidence = insight.evidence
    inference = insight.inference
    return TimelineEntry(
        source="work_event",
        entry_id=insight.event_id,
        session_id=insight.session_id,
        kind=inference.heuristic_label,
        summary=inference.summary,
        start_time=evidence.start_time,
        end_time=evidence.end_time,
        duration_ms=evidence.duration_ms,
        fidelity=fidelity_for(evidence.timing_provenance),
        timing_provenance=evidence.timing_provenance,
        confidence=inference.confidence,
        tools_used=tuple(evidence.tools_used),
        file_paths=tuple(evidence.file_paths),
    )


def _phase_entry(insight: SessionPhaseInsight, *, phase_kind: str = "phase") -> TimelineEntry:
    evidence = insight.evidence
    inference = insight.inference
    return TimelineEntry(
        source="phase",
        entry_id=insight.phase_id,
        session_id=insight.session_id,
        kind=phase_kind,
        summary=f"phase #{insight.phase_index} ({evidence.word_count} words)",
        start_time=evidence.start_time,
        end_time=evidence.end_time,
        duration_ms=evidence.duration_ms,
        fidelity=fidelity_for(evidence.timing_provenance),
        timing_provenance=evidence.timing_provenance,
        confidence=inference.confidence if inference is not None else 0.0,
        word_count=evidence.word_count,
    )


def _sort_key(entry: TimelineEntry) -> tuple[int, str, int]:
    # Entries with a start_time come first, ordered lexicographically (ISO
    # timestamps sort correctly). Entries without a start_time come last,
    # preserved in input order via the secondary index passed by the caller.
    if entry.start_time is None:
        return (1, "", 0)
    return (0, entry.start_time, 0)


def build_timeline_entries(
    work_events: Iterable[SessionWorkEventInsight],
    phases: Iterable[SessionPhaseInsight],
) -> list[TimelineEntry]:
    """Merge work events and phases into a chronologically ordered list."""
    entries: list[TimelineEntry] = []
    entries.extend(_work_event_entry(we) for we in work_events)
    entries.extend(_phase_entry(p) for p in phases)
    # Stable sort: entries with start_time sort by ISO timestamp, untimed
    # entries fall to the end in input order.
    indexed = list(enumerate(entries))
    indexed.sort(key=lambda pair: (_sort_key(pair[1]), pair[0]))
    return [entry for _, entry in indexed]


def build_session_timeline(
    session_id: str,
    work_events: Iterable[SessionWorkEventInsight],
    phases: Iterable[SessionPhaseInsight],
) -> SessionTimeline:
    """Compose the full SessionTimeline payload."""
    entries = build_timeline_entries(work_events, phases)
    counts: dict[FidelityTag, int] = {"hook": 0, "sort_key": 0}
    for entry in entries:
        counts[entry.fidelity] += 1
    return SessionTimeline(
        session_id=session_id,
        entries=tuple(entries),
        fidelity_counts=counts,
    )


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _format_duration(ms: int) -> str:
    if ms <= 0:
        return "-"
    if ms < 1000:
        return f"{ms}ms"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.1f}h"


def _format_time(value: str | None) -> str:
    if not value:
        return "-"
    return value


def _fidelity_marker(tag: FidelityTag) -> str:
    # Single-letter marker so the column stays narrow; explicit hint
    # appears in the legend at the bottom of plain output.
    return "H" if tag == "hook" else "S"


def render_plain(timeline: SessionTimeline) -> str:
    """Render the timeline as a fixed-width plain-text table.

    The fidelity column uses ``H`` for hook-precise entries and ``S`` for
    sort-key-reconstructed entries. The legend appears beneath the table.
    """
    lines: list[str] = []
    lines.append(f"Timeline for {timeline.session_id}")
    counts = timeline.fidelity_counts
    lines.append(f"  fidelity: hook={counts.get('hook', 0)} sort_key={counts.get('sort_key', 0)}")
    lines.append("")
    if not timeline.entries:
        lines.append("(no timeline rows)")
        return "\n".join(lines)

    lines.append(f"{'fid':<3} {'source':<10} {'kind':<14} {'start':<20} {'dur':>7}  summary")
    lines.append("-" * 72)
    for entry in timeline.entries:
        lines.append(
            f"{_fidelity_marker(entry.fidelity):<3} "
            f"{entry.source:<10} "
            f"{entry.kind:<14} "
            f"{_format_time(entry.start_time):<20} "
            f"{_format_duration(entry.duration_ms):>7}  "
            f"{entry.summary}"
        )
    lines.append("")
    lines.append("legend: H = hook-precise (timestamped range), S = sort-key-reconstructed")
    return "\n".join(lines)


def render_markdown(timeline: SessionTimeline) -> str:
    """Render the timeline as a GitHub-flavored Markdown table."""
    lines: list[str] = []
    lines.append(f"# Timeline — `{timeline.session_id}`")
    counts = timeline.fidelity_counts
    lines.append("")
    lines.append(f"**Fidelity:** hook={counts.get('hook', 0)}, sort_key={counts.get('sort_key', 0)}")
    lines.append("")
    if not timeline.entries:
        lines.append("_No timeline rows._")
        return "\n".join(lines)
    lines.append("| Fidelity | Source | Kind | Start | Duration | Summary |")
    lines.append("|---|---|---|---|---|---|")
    for entry in timeline.entries:
        summary = entry.summary.replace("|", "\\|")
        lines.append(
            f"| {entry.fidelity} | {entry.source} | {entry.kind} | "
            f"{_format_time(entry.start_time)} | {_format_duration(entry.duration_ms)} | "
            f"{summary} |"
        )
    lines.append("")
    lines.append(
        "_`hook` = timing came from a recorded timestamped range. "
        "`sort_key` = at least one endpoint reconstructed from message sort-key order._"
    )
    return "\n".join(lines)


def render(
    work_events: Sequence[SessionWorkEventInsight],
    phases: Sequence[SessionPhaseInsight],
    *,
    session_id: str,
    output: Literal["plain", "markdown"] = "plain",
) -> str:
    """Convenience: build the SessionTimeline and render in one call."""
    timeline = build_session_timeline(session_id, work_events, phases)
    if output == "markdown":
        return render_markdown(timeline)
    return render_plain(timeline)


__all__ = [
    "FidelityTag",
    "SessionTimeline",
    "TimelineEntry",
    "build_session_timeline",
    "build_timeline_entries",
    "fidelity_for",
    "render",
    "render_markdown",
    "render_plain",
]
