"""Storage-free temporal evidence projection primitives.

The archive/query layer should select occurrence rows; this module gives CLI,
MCP, daemon, and demo renderers a shared way to compose those rows into a time
window without inventing recovery/export-specific report DTOs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from polylogue.surfaces.payloads import SurfacePayloadModel


class TemporalBucket(str, Enum):
    """Supported deterministic temporal aggregation grains."""

    HOUR = "hour"
    DAY = "day"


class TemporalEvidenceEvent(SurfacePayloadModel):
    """One timestamped occurrence from any evidence family."""

    event_id: str
    occurred_at: datetime
    family: str
    kind: str
    label: str
    source_ref: str | None = None
    evidence_refs: tuple[str, ...] = ()
    phase: str | None = None

    @model_validator(mode="after")
    def _require_timezone(self) -> TemporalEvidenceEvent:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return self


class TemporalCountBucket(SurfacePayloadModel):
    """Count of events for one bucket/family/kind coordinate."""

    bucket_start: datetime
    family: str
    kind: str
    count: int = Field(ge=0)


class TemporalPhaseSpan(SurfacePayloadModel):
    """Elapsed time between adjacent phase-bearing events."""

    from_phase: str
    to_phase: str
    start_at: datetime
    end_at: datetime
    duration_seconds: float = Field(ge=0)
    start_event_id: str
    end_event_id: str


class TemporalActivityBand(SurfacePayloadModel):
    """Dense activity bucket with compositional family/kind counts."""

    bucket_start: datetime
    event_count: int = Field(ge=0)
    family_counts: dict[str, int] = Field(default_factory=dict)
    kind_counts: dict[str, int] = Field(default_factory=dict)


class TemporalEvidenceWindow(SurfacePayloadModel):
    """Composable temporal projection over selected evidence events."""

    since: datetime | None = None
    until: datetime | None = None
    bucket: TemporalBucket = TemporalBucket.HOUR
    events: tuple[TemporalEvidenceEvent, ...] = ()
    event_count: int = Field(ge=0)
    family_counts: dict[str, int] = Field(default_factory=dict)
    kind_counts: dict[str, int] = Field(default_factory=dict)
    buckets: tuple[TemporalCountBucket, ...] = ()
    activity_bands: tuple[TemporalActivityBand, ...] = ()
    phase_spans: tuple[TemporalPhaseSpan, ...] = ()
    caveats: tuple[str, ...] = ()


def build_temporal_evidence_window(
    events: Iterable[TemporalEvidenceEvent],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    bucket: TemporalBucket = TemporalBucket.HOUR,
    caveats: Iterable[str] = (),
) -> TemporalEvidenceWindow:
    """Build a deterministic temporal window from already-selected events."""

    _validate_bound("since", since)
    _validate_bound("until", until)
    selected = tuple(
        sorted(
            (event for event in events if _in_window(event, since=since, until=until)),
            key=lambda event: (event.occurred_at, event.event_id),
        )
    )
    family_counts = Counter(event.family for event in selected)
    kind_counts = Counter(event.kind for event in selected)
    caveat_list = list(caveats)
    if since is None or until is None:
        caveat_list.append("window_bound_open")
    return TemporalEvidenceWindow(
        since=since,
        until=until,
        bucket=bucket,
        events=selected,
        event_count=len(selected),
        family_counts=dict(sorted(family_counts.items())),
        kind_counts=dict(sorted(kind_counts.items())),
        buckets=_count_buckets(selected, bucket=bucket),
        activity_bands=_activity_bands(selected, bucket=bucket),
        phase_spans=_phase_spans(selected),
        caveats=tuple(dict.fromkeys(caveat_list)),
    )


def summary_to_temporal_event(summary: Any) -> TemporalEvidenceEvent | None:
    """Project a session summary-like object into a temporal occurrence event."""

    occurred_at = getattr(summary, "created_at", None) or getattr(summary, "updated_at", None)
    if occurred_at is None:
        return None
    session_id = str(summary.id)
    label = str(getattr(summary, "title", None) or getattr(summary, "display_title", None) or session_id)
    return TemporalEvidenceEvent(
        event_id=f"session:{session_id}:session",
        occurred_at=_timezone_aware(occurred_at),
        family="archive-session",
        kind="session",
        label=label,
        source_ref=f"session:{session_id}",
        evidence_refs=(f"session:{session_id}",),
    )


def message_row_to_temporal_event(row: Any) -> TemporalEvidenceEvent | None:
    """Project a message query row into a temporal occurrence event."""

    occurred_at_ms = getattr(row, "occurred_at_ms", None)
    if occurred_at_ms is None:
        return None
    message_id = str(row.message_id)
    session_id = str(row.session_id)
    role = str(getattr(row, "role", "unknown") or "unknown")
    message_type = str(getattr(row, "message_type", "message") or "message")
    position = int(getattr(row, "position", 0) or 0)
    text = str(getattr(row, "text", "") or "").replace("\n", " ").strip()
    label = f"{role} {message_type} #{position}"
    if text:
        label = f"{label}: {text[:80]}"
    return TemporalEvidenceEvent(
        event_id=f"message:{message_id}:message",
        occurred_at=datetime.fromtimestamp(int(occurred_at_ms) / 1000, UTC),
        family="archive-message",
        kind=message_type,
        label=label,
        source_ref=f"message:{message_id}",
        evidence_refs=(f"session:{session_id}", f"message:{message_id}"),
        phase=role,
    )


def action_row_to_temporal_event(row: Any) -> TemporalEvidenceEvent | None:
    """Project an action query row into a temporal occurrence event."""

    occurred_at_ms = getattr(row, "occurred_at_ms", None)
    if occurred_at_ms is None:
        return None
    session_id = str(row.session_id)
    message_id = str(row.message_id)
    tool_use_block_id = str(row.tool_use_block_id)
    tool_name = str(getattr(row, "tool_name", "") or "tool")
    semantic_type = str(getattr(row, "semantic_type", "") or "action")
    command = str(getattr(row, "tool_command", "") or "").replace("\n", " ").strip()
    path = str(getattr(row, "tool_path", "") or "").replace("\n", " ").strip()
    label = f"{semantic_type} via {tool_name}"
    if command:
        label = f"{label}: {command[:80]}"
    elif path:
        label = f"{label}: {path[:80]}"
    return TemporalEvidenceEvent(
        event_id=f"action:{tool_use_block_id}:action",
        occurred_at=datetime.fromtimestamp(int(occurred_at_ms) / 1000, UTC),
        family="archive-action",
        kind=semantic_type,
        label=label,
        source_ref=f"action:{tool_use_block_id}",
        evidence_refs=(f"session:{session_id}", f"message:{message_id}", f"action:{tool_use_block_id}"),
        phase=semantic_type,
    )


def _timezone_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _validate_bound(name: str, value: datetime | None) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


def _in_window(event: TemporalEvidenceEvent, *, since: datetime | None, until: datetime | None) -> bool:
    occurred_at = event.occurred_at
    if since is not None and occurred_at < since:
        return False
    return not (until is not None and occurred_at > until)


def _bucket_start(value: datetime, bucket: TemporalBucket) -> datetime:
    normalized = value.astimezone(UTC)
    if bucket is TemporalBucket.DAY:
        return normalized.replace(hour=0, minute=0, second=0, microsecond=0)
    return normalized.replace(minute=0, second=0, microsecond=0)


def _count_buckets(
    events: tuple[TemporalEvidenceEvent, ...],
    *,
    bucket: TemporalBucket,
) -> tuple[TemporalCountBucket, ...]:
    counts: Counter[tuple[datetime, str, str]] = Counter(
        (_bucket_start(event.occurred_at, bucket), event.family, event.kind) for event in events
    )
    return tuple(
        TemporalCountBucket(bucket_start=start, family=family, kind=kind, count=count)
        for (start, family, kind), count in sorted(counts.items(), key=lambda item: item[0])
    )


def _activity_bands(
    events: tuple[TemporalEvidenceEvent, ...],
    *,
    bucket: TemporalBucket,
    limit: int = 5,
) -> tuple[TemporalActivityBand, ...]:
    grouped: dict[datetime, list[TemporalEvidenceEvent]] = {}
    for event in events:
        grouped.setdefault(_bucket_start(event.occurred_at, bucket), []).append(event)
    ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))[:limit]
    return tuple(
        TemporalActivityBand(
            bucket_start=start,
            event_count=len(bucket_events),
            family_counts=dict(sorted(Counter(event.family for event in bucket_events).items())),
            kind_counts=dict(sorted(Counter(event.kind for event in bucket_events).items())),
        )
        for start, bucket_events in ranked
    )


def _phase_spans(events: tuple[TemporalEvidenceEvent, ...]) -> tuple[TemporalPhaseSpan, ...]:
    phase_events = [event for event in events if event.phase]
    spans: list[TemporalPhaseSpan] = []
    for start, end in zip(phase_events, phase_events[1:], strict=False):
        duration = (end.occurred_at - start.occurred_at).total_seconds()
        if duration < 0:
            continue
        spans.append(
            TemporalPhaseSpan(
                from_phase=str(start.phase),
                to_phase=str(end.phase),
                start_at=start.occurred_at,
                end_at=end.occurred_at,
                duration_seconds=duration,
                start_event_id=start.event_id,
                end_event_id=end.event_id,
            )
        )
    return tuple(spans)


__all__ = [
    "TemporalActivityBand",
    "TemporalBucket",
    "TemporalCountBucket",
    "TemporalEvidenceEvent",
    "TemporalEvidenceWindow",
    "TemporalPhaseSpan",
    "action_row_to_temporal_event",
    "build_temporal_evidence_window",
    "message_row_to_temporal_event",
    "summary_to_temporal_event",
]
