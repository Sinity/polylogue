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
    phase_spans: tuple[TemporalPhaseSpan, ...] = ()
    caveats: tuple[str, ...] = ()


def build_temporal_evidence_window(
    events: Iterable[TemporalEvidenceEvent],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    bucket: TemporalBucket = TemporalBucket.HOUR,
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
    caveats: list[str] = []
    if since is None or until is None:
        caveats.append("window_bound_open")
    return TemporalEvidenceWindow(
        since=since,
        until=until,
        bucket=bucket,
        events=selected,
        event_count=len(selected),
        family_counts=dict(sorted(family_counts.items())),
        kind_counts=dict(sorted(kind_counts.items())),
        buckets=_count_buckets(selected, bucket=bucket),
        phase_spans=_phase_spans(selected),
        caveats=tuple(caveats),
    )


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
    "TemporalBucket",
    "TemporalCountBucket",
    "TemporalEvidenceEvent",
    "TemporalEvidenceWindow",
    "TemporalPhaseSpan",
    "build_temporal_evidence_window",
]
