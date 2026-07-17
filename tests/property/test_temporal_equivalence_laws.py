"""Survivor laws for equivalent instants across Polylogue temporal routes.

Expected instants come from ``tests.infra.strategies.temporal``.  That oracle
uses only standard-library datetime/Decimal arithmetic and explicit closed
interval rules; it never calls a Polylogue parser, storage converter, query
helper, or surface canonicalizer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings

from polylogue import Polylogue
from polylogue.core.timestamps import canonical_timestamp_text, parse_timestamp
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceEvent,
    build_temporal_evidence_window,
    summary_to_temporal_event,
)
from tests.infra.storage_records import SessionBuilder
from tests.infra.strategies.temporal import (
    EquivalentInstantCase,
    equivalent_instant_case_strategy,
    standard_datetime,
    standard_epoch_microseconds,
    wire_datetime,
)

_EXACT_EQUIVALENT_WIRES = (
    (
        "leap-day/month-boundary",
        "2024-02-29T23:59:59.123456Z",
        "2024-03-01T05:29:59.123456+05:30",
    ),
    (
        "year-boundary/max-positive-offset",
        "2024-12-31T23:59:59.999999Z",
        "2025-01-01T13:59:59.999999+14:00",
    ),
    (
        "naive-provider-input-is-declared-utc",
        "2024-02-29T23:59:59.123456",
        "2024-02-29T23:59:59.123456Z",
    ),
    (
        "fractional-epoch-provider-wire",
        "1709251199.123456",
        "2024-02-29T23:59:59.123456Z",
    ),
)


def _event(event_id: str, occurred_at: datetime, *, phase: str | None = None) -> TemporalEvidenceEvent:
    return TemporalEvidenceEvent(
        event_id=event_id,
        occurred_at=occurred_at,
        family="provider-wire",
        kind="message",
        label=event_id,
        evidence_refs=(f"wire:{event_id}",),
        phase=phase,
    )


@pytest.mark.parametrize(("witness", "left_wire", "right_wire"), _EXACT_EQUIVALENT_WIRES)
def test_exact_provider_wire_witnesses_name_the_same_standard_instant(
    witness: str,
    left_wire: str,
    right_wire: str,
) -> None:
    """Exact non-dominated witnesses kill timezone-strip and precision-truncation mutations."""

    del witness
    expected = standard_epoch_microseconds(left_wire)
    assert expected is not None
    assert standard_epoch_microseconds(right_wire) == expected

    left = parse_timestamp(left_wire)
    right = parse_timestamp(right_wire)

    assert left is not None
    assert right is not None
    assert standard_epoch_microseconds(left) == expected
    assert standard_epoch_microseconds(right) == expected
    assert left == right
    assert canonical_timestamp_text(left_wire) == canonical_timestamp_text(right_wire)


@given(case=equivalent_instant_case_strategy())
@settings(max_examples=80, deadline=None)
def test_parse_timestamp_matches_independent_standard_instant(case: EquivalentInstantCase) -> None:
    """Equivalent offsets survive parsing; local-time comparison and timezone-strip mutations die."""

    first = parse_timestamp(case.first_wire)
    second = parse_timestamp(case.second_wire)

    assert first is not None
    assert second is not None
    assert standard_epoch_microseconds(first) == case.epoch_microseconds
    assert standard_epoch_microseconds(second) == case.epoch_microseconds
    assert first == second


@given(case=equivalent_instant_case_strategy())
@settings(max_examples=60, deadline=None)
def test_closed_window_and_public_rendering_are_representation_invariant(case: EquivalentInstantCase) -> None:
    """A closed [since, until] range includes both endpoints and renders one UTC form.

    This kills exclusive-boundary (``>``/``<``), local-wall ordering, timezone
    strip, and public-offset-leak mutations.
    """

    first = wire_datetime(case.first_wire)
    second = wire_datetime(case.second_wire)
    left = build_temporal_evidence_window(
        [_event("same-instant", first)],
        since=first,
        until=first,
    )
    right = build_temporal_evidence_window(
        [_event("same-instant", second)],
        since=second,
        until=second,
    )

    assert left.event_count == right.event_count == 1
    assert left.model_dump(mode="json") == right.model_dump(mode="json")
    assert left.to_json() == right.to_json()
    assert left.since is not None and left.since.tzinfo is UTC
    assert left.until is not None and left.until.tzinfo is UTC
    assert left.events[0].occurred_at.tzinfo is UTC


def test_equal_instants_order_by_event_id_without_truncating_subseconds() -> None:
    """Instant ordering uses absolute time, then event ID, while retaining one-microsecond distinctions."""

    equal_fold_wire = wire_datetime("2024-11-03T01:30:00.123456-04:00")
    equal_utc_wire = wire_datetime("2024-11-03T05:30:00.123456Z")
    one_microsecond_later = wire_datetime("2024-11-03T01:30:00.123457-04:00")

    window = build_temporal_evidence_window(
        [
            _event("z-equal", equal_fold_wire),
            _event("later", one_microsecond_later),
            _event("a-equal", equal_utc_wire),
        ],
        since=wire_datetime("2024-11-03T00:30:00.123456-05:00"),
        until=wire_datetime("2024-11-03T05:30:00.123457Z"),
    )

    assert [event.event_id for event in window.events] == ["a-equal", "z-equal", "later"]
    assert window.events[0].occurred_at.microsecond == 123456
    assert window.events[2].occurred_at.microsecond == 123457


def test_dst_fold_witnesses_remain_distinct_absolute_instants() -> None:
    """The repeated 01:30 wall time in New York must not collapse to one instant."""

    first_fold = parse_timestamp("2024-11-03T01:30:00-04:00")
    second_fold = parse_timestamp("2024-11-03T01:30:00-05:00")

    assert first_fold is not None
    assert second_fold is not None
    first_microseconds = standard_epoch_microseconds(first_fold)
    second_microseconds = standard_epoch_microseconds(second_fold)
    assert first_microseconds is not None
    assert second_microseconds is not None
    assert standard_epoch_microseconds("2024-11-03T05:30:00Z") == first_microseconds
    assert standard_epoch_microseconds("2024-11-03T06:30:00Z") == second_microseconds
    assert second_microseconds - first_microseconds == 3_600_000_000
    assert first_fold < second_fold


def test_dst_gap_adjacent_events_expose_an_inferred_one_microsecond_gap() -> None:
    """Adjacent-event gaps are inferred evidence, never provider or model-compute duration.

    The New York spring transition jumps from 01:59:59.999999-05:00 to
    03:00:00-04:00 in one microsecond.  A local-wall subtraction mutation
    reports one hour; a precision-truncation mutation reports zero.
    """

    before_gap = wire_datetime("2024-03-10T01:59:59.999999-05:00")
    after_gap = wire_datetime("2024-03-10T03:00:00-04:00")
    window = build_temporal_evidence_window(
        [
            _event("before-gap", before_gap, phase="provider-message"),
            _event("after-gap", after_gap, phase="provider-message"),
        ]
    )

    assert len(window.phase_spans) == 1
    span = window.phase_spans[0]
    assert span.duration_seconds == pytest.approx(0.000001)
    assert span.duration_semantics == "inferred_event_gap"
    assert window.model_dump(mode="json")["phase_spans"][0]["duration_semantics"] == "inferred_event_gap"
    assert "model_compute" not in span.duration_semantics
    assert span.start_at == standard_datetime("2024-03-10T06:59:59.999999Z")
    assert span.end_at == standard_datetime("2024-03-10T07:00:00Z")


def test_absent_provider_timestamps_remain_absent() -> None:
    """Missing timestamps are not converted to epoch zero or synthetic occurrences."""

    assert parse_timestamp(None) is None
    assert canonical_timestamp_text(None) is None
    summary = SimpleNamespace(id="no-time", title="No timestamp", created_at=None, updated_at=None)
    assert summary_to_temporal_event(summary) is None


@pytest.mark.asyncio
async def test_archive_query_closed_range_is_inclusive_and_offset_invariant(tmp_path: Path) -> None:
    """The real SQLite query route applies the same closed interval to equivalent bounds.

    Replacing either SQL ``>=``/``<=`` with an exclusive comparator, comparing
    local wall fields, or stripping an offset makes this survivor fail.
    """

    # Archive timestamps are stored at millisecond precision, so these unique
    # witnesses differ by one declared storage unit rather than hiding a
    # microsecond expectation behind the schema's documented precision.
    archive_root = tmp_path
    db_path = archive_root / "index.db"
    target_utc = datetime(2024, 2, 29, 23, 59, 59, 123000, tzinfo=UTC)
    before_utc = target_utc - timedelta(milliseconds=1)
    after_utc = target_utc + timedelta(milliseconds=1)

    before = (
        SessionBuilder(db_path, "before-boundary").created_at(before_utc.isoformat()).updated_at(before_utc.isoformat())
    )
    at = (
        SessionBuilder(db_path, "at-boundary")
        .created_at("2024-03-01T05:29:59.123+05:30")
        .updated_at("2024-03-01T05:29:59.123+05:30")
    )
    after = (
        SessionBuilder(db_path, "after-boundary").created_at(after_utc.isoformat()).updated_at(after_utc.isoformat())
    )
    expected_id = at.native_session_id()
    before.save()
    at.save()
    after.save()

    async with Polylogue(archive_root=archive_root, db_path=db_path) as archive:
        offset_rows = await (
            archive.filter()
            .since("2024-03-01T00:59:59.123+01:00")
            .until("2024-02-29T18:59:59.123-05:00")
            .list_summaries()
        )
        utc_rows = await (
            archive.filter().since("2024-02-29T23:59:59.123Z").until("2024-02-29T23:59:59.123+00:00").list_summaries()
        )

    assert [str(row.id) for row in offset_rows] == [expected_id]
    assert [str(row.id) for row in utc_rows] == [expected_id]
    assert offset_rows == utc_rows
