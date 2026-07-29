"""Work-event extraction regression tests for the traversal-order-vs-wall-clock
mismatch fixed alongside session_phases (see test_phase_extraction.py's
``test_extract_phases_never_inverts_start_after_end_for_out_of_order_variant``).

session_work_events shares the exact same defect shape as session_phases:
2,614 of the 13,743 live-archive inverted rows (``ended_at_ms < started_at_ms``)
were session_work_events, all downstream of
``polylogue.archive.session.extraction._range_timing`` taking the first/last
message timestamp *in (position, variant_index) traversal order* rather than
the true min/max envelope.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.session.extraction import extract_work_events
from polylogue.core.enums import Provider
from tests.infra.builders import make_conv, make_msg


def test_extract_work_events_never_inverts_start_after_end_for_out_of_order_variant() -> None:
    day1_1757 = datetime(2024, 1, 11, 17, 57, 26, tzinfo=timezone.utc)
    day1_2150 = datetime(2024, 1, 11, 21, 50, 54, tzinfo=timezone.utc)
    day1_2152 = datetime(2024, 1, 11, 21, 52, 51, tzinfo=timezone.utc)
    day1_2155 = datetime(2024, 1, 11, 21, 55, 50, tzinfo=timezone.utc)
    day1_1757b = datetime(2024, 1, 11, 17, 57, 27, tzinfo=timezone.utc)  # abandoned-branch reply
    day1_2151 = datetime(2024, 1, 11, 21, 50, 55, tzinfo=timezone.utc)

    session = make_conv(
        id="conv-chatgpt-work-event-reorder",
        origin=Provider.CHATGPT,
        title="ChatGPT edited-turn regeneration",
        messages=MessageCollection(
            messages=[
                make_msg(id="m0", role="user", origin="chatgpt", text="hi", timestamp=day1_1757, branch_index=0),
                make_msg(id="m1", role="user", origin="chatgpt", text="edit 1", timestamp=day1_2150, branch_index=1),
                make_msg(id="m2", role="user", origin="chatgpt", text="edit 2", timestamp=day1_2152, branch_index=2),
                make_msg(id="m3", role="user", origin="chatgpt", text="edit 3", timestamp=day1_2155, branch_index=3),
                make_msg(
                    id="m4",
                    role="assistant",
                    origin="chatgpt",
                    text="stale reply to the abandoned first edit",
                    timestamp=day1_1757b,
                    branch_index=0,
                ),
                make_msg(
                    id="m5",
                    role="assistant",
                    origin="chatgpt",
                    text="accepted reply",
                    timestamp=day1_2151,
                    branch_index=0,
                ),
            ]
        ),
    )

    events = extract_work_events(session)

    assert events, "expected at least one work event"
    for event in events:
        if event.start_time is not None and event.end_time is not None:
            assert event.end_time >= event.start_time, (
                f"inverted work event: start={event.start_time} end={event.end_time} "
                f"range=({event.start_index}, {event.end_index})"
            )


def test_extract_work_events_flags_reorder_evidence_without_inverting() -> None:
    """A within-range regression too small to force a phase split can still
    make the envelope diverge from naive first/last order; that must be
    visible in the event's evidence rather than silently reported as a
    clean range.
    """
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    session = make_conv(
        id="conv-work-event-small-regression",
        origin=Provider.CHATGPT,
        title="Small in-range regression",
        messages=MessageCollection(
            messages=[
                make_msg(id="m0", role="user", origin="chatgpt", text="a", timestamp=base + timedelta(minutes=3)),
                make_msg(id="m1", role="assistant", origin="chatgpt", text="b", timestamp=base + timedelta(minutes=1)),
                make_msg(id="m2", role="user", origin="chatgpt", text="c", timestamp=base + timedelta(minutes=4)),
            ]
        ),
    )

    events = extract_work_events(session)

    assert len(events) == 1
    event = events[0]
    assert event.start_time is not None
    assert event.end_time is not None
    assert event.end_time >= event.start_time
    assert event.start_time == base + timedelta(minutes=1)
    assert event.end_time == base + timedelta(minutes=4)
    assert "chronological_reorder" in event.evidence
