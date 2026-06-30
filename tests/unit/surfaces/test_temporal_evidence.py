from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from polylogue.surfaces.temporal_evidence import (
    TemporalBucket,
    TemporalEvidenceEvent,
    build_temporal_evidence_window,
)


def _event(
    event_id: str,
    hour: int,
    minute: int,
    *,
    family: str,
    kind: str,
    phase: str | None = None,
) -> TemporalEvidenceEvent:
    return TemporalEvidenceEvent(
        event_id=event_id,
        occurred_at=datetime(2026, 6, 30, hour, minute, tzinfo=UTC),
        family=family,
        kind=kind,
        label=event_id,
        phase=phase,
        evidence_refs=(f"event:{event_id}",),
    )


def test_temporal_window_filters_sorts_and_counts_events() -> None:
    events = [
        _event("git-1", 10, 10, family="git", kind="commit"),
        _event("log-2", 11, 5, family="devloop-log", kind="focus", phase="Proof"),
        _event("log-1", 10, 5, family="devloop-log", kind="focus", phase="Construction"),
        _event("archive-1", 12, 0, family="archive", kind="message"),
    ]

    window = build_temporal_evidence_window(
        events,
        since=datetime(2026, 6, 30, 10, 0, tzinfo=UTC),
        until=datetime(2026, 6, 30, 11, 59, tzinfo=UTC),
    )

    assert [event.event_id for event in window.events] == ["log-1", "git-1", "log-2"]
    assert window.event_count == 3
    assert window.family_counts == {"devloop-log": 2, "git": 1}
    assert window.kind_counts == {"commit": 1, "focus": 2}
    assert window.caveats == ()
    assert [(bucket.bucket_start.hour, bucket.family, bucket.kind, bucket.count) for bucket in window.buckets] == [
        (10, "devloop-log", "focus", 1),
        (10, "git", "commit", 1),
        (11, "devloop-log", "focus", 1),
    ]
    assert [(band.bucket_start.hour, band.event_count) for band in window.activity_bands] == [(10, 2), (11, 1)]
    assert window.activity_bands[0].family_counts == {"devloop-log": 1, "git": 1}
    assert window.activity_bands[0].kind_counts == {"commit": 1, "focus": 1}


def test_temporal_window_preserves_explicit_caveats() -> None:
    window = build_temporal_evidence_window(
        [_event("a", 9, 0, family="archive-message", kind="message")],
        caveats=("message_events_capped", "message_events_capped"),
    )

    assert window.caveats == ("message_events_capped", "window_bound_open")


def test_temporal_window_computes_adjacent_phase_spans() -> None:
    window = build_temporal_evidence_window(
        [
            _event("a", 9, 0, family="devloop-log", kind="focus", phase="Direction"),
            _event("b", 9, 15, family="devloop-log", kind="focus", phase="Evidence"),
            _event("c", 10, 0, family="devloop-log", kind="focus", phase="Proof"),
        ]
    )

    assert [(span.from_phase, span.to_phase, span.duration_seconds) for span in window.phase_spans] == [
        ("Direction", "Evidence", 900.0),
        ("Evidence", "Proof", 2700.0),
    ]
    assert window.caveats == ("window_bound_open",)


def test_temporal_window_supports_day_buckets() -> None:
    window = build_temporal_evidence_window(
        [
            _event("a", 1, 0, family="archive", kind="message"),
            _event("b", 23, 0, family="archive", kind="message"),
        ],
        bucket=TemporalBucket.DAY,
    )

    assert len(window.buckets) == 1
    assert window.buckets[0].bucket_start == datetime(2026, 6, 30, tzinfo=UTC)
    assert window.buckets[0].count == 2
    assert window.activity_bands[0].bucket_start == datetime(2026, 6, 30, tzinfo=UTC)
    assert window.activity_bands[0].event_count == 2


def test_temporal_window_activity_bands_keep_top_dense_buckets() -> None:
    events = [
        _event("h10-a", 10, 0, family="devloop-log", kind="checkpoint"),
        _event("h10-b", 10, 5, family="git", kind="commit"),
        _event("h12-a", 12, 0, family="git", kind="commit"),
        _event("h11-a", 11, 0, family="devloop-log", kind="focus"),
        _event("h11-b", 11, 1, family="devloop-log", kind="focus"),
        _event("h11-c", 11, 2, family="git", kind="commit"),
    ]

    window = build_temporal_evidence_window(events)

    assert [(band.bucket_start.hour, band.event_count) for band in window.activity_bands] == [
        (11, 3),
        (10, 2),
        (12, 1),
    ]
    assert window.activity_bands[0].family_counts == {"devloop-log": 2, "git": 1}
    assert window.activity_bands[0].kind_counts == {"commit": 1, "focus": 2}


def test_temporal_events_and_bounds_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        TemporalEvidenceEvent(
            event_id="naive",
            occurred_at=datetime(2026, 6, 30, 10, 0),
            family="archive",
            kind="message",
            label="naive",
        )
    with pytest.raises(ValueError, match="since must be timezone-aware"):
        build_temporal_evidence_window([], since=datetime(2026, 6, 30, 10, 0))
