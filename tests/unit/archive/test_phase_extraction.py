"""Phase-extraction regression tests covering the session-events fallback (#1624)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.phase.extraction import extract_phases
from polylogue.archive.session.events import SessionEvent
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.core.types import SessionEventId, SessionId
from tests.infra.builders import make_conv, make_msg


def _untimed_msg(idx: int) -> Message:
    return make_msg(
        id=f"m{idx}",
        role="user" if idx % 2 == 0 else "assistant",
        origin="codex",
        text=f"message {idx}",
        timestamp=None,
    )


def test_extract_phases_falls_back_to_session_events_when_messages_have_no_timestamps() -> None:
    started_at = datetime(2026, 5, 24, 10, 0, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(minutes=2)
    session = make_conv(
        id="conv-codex-no-msg-ts",
        origin=Provider.CODEX,
        title="Codex pre-Dec-2025",
        messages=MessageCollection(messages=[_untimed_msg(0), _untimed_msg(1)]),
        session_events=(
            SessionEvent(
                id=SessionEventId("conv-codex-no-msg-ts:event-0"),
                session_id=SessionId("conv-codex-no-msg-ts"),
                origin=origin_from_provider(Provider.CODEX),
                event_index=0,
                event_type="function_call",
                timestamp=started_at,
                payload={"call_id": "c1", "name": "exec"},
            ),
            SessionEvent(
                id=SessionEventId("conv-codex-no-msg-ts:event-1"),
                session_id=SessionId("conv-codex-no-msg-ts"),
                origin=origin_from_provider(Provider.CODEX),
                event_index=1,
                event_type="function_call_output",
                timestamp=ended_at,
                payload={"call_id": "c1", "status": "ok"},
            ),
        ),
    )

    phases = extract_phases(session)

    assert len(phases) == 1
    phase = phases[0]
    assert phase.start_time == started_at
    assert phase.end_time == ended_at
    assert phase.duration_ms == 120_000
    assert phase.message_range == (0, 2)


def test_extract_phases_splits_session_events_on_idle_gap() -> None:
    burst_a_start = datetime(2026, 5, 24, 10, 0, tzinfo=timezone.utc)
    burst_a_end = burst_a_start + timedelta(minutes=1)
    burst_b_start = burst_a_end + timedelta(minutes=10)
    burst_b_end = burst_b_start + timedelta(minutes=1)
    session = make_conv(
        id="conv-codex-bursts",
        origin=Provider.CODEX,
        title="Codex two bursts",
        messages=MessageCollection(messages=[_untimed_msg(i) for i in range(4)]),
        session_events=tuple(
            SessionEvent(
                id=SessionEventId(f"conv-codex-bursts:event-{i}"),
                session_id=SessionId("conv-codex-bursts"),
                origin=origin_from_provider(Provider.CODEX),
                event_index=i,
                event_type="function_call",
                timestamp=ts,
                payload={"call_id": f"c{i}"},
            )
            for i, ts in enumerate([burst_a_start, burst_a_end, burst_b_start, burst_b_end])
        ),
    )

    phases = extract_phases(session)

    assert len(phases) == 2
    assert (phases[0].start_time, phases[0].end_time) == (burst_a_start, burst_a_end)
    assert (phases[1].start_time, phases[1].end_time) == (burst_b_start, burst_b_end)


def test_extract_phases_returns_empty_when_no_timestamps_anywhere() -> None:
    session = make_conv(
        id="conv-codex-zero",
        origin=Provider.CODEX,
        title="No times at all",
        messages=MessageCollection(messages=[_untimed_msg(0)]),
        session_events=(),
    )

    assert extract_phases(session) == []


def test_extract_phases_prefers_message_timestamps_when_present() -> None:
    started_at = datetime(2026, 5, 24, 10, 0, tzinfo=timezone.utc)
    session = make_conv(
        id="conv-claude-code",
        origin=Provider.CLAUDE_CODE,
        title="Normal claude-code",
        messages=MessageCollection(
            messages=[
                make_msg(id="m0", role="user", origin="claude-code", text="hi", timestamp=started_at),
                make_msg(
                    id="m1",
                    role="assistant",
                    origin="claude-code",
                    text="hello",
                    timestamp=started_at + timedelta(minutes=1),
                ),
            ]
        ),
        session_events=(
            SessionEvent(
                id=SessionEventId("conv-claude-code:event-0"),
                session_id=SessionId("conv-claude-code"),
                origin=origin_from_provider(Provider.CLAUDE_CODE),
                event_index=0,
                event_type="session_meta",
                timestamp=started_at - timedelta(hours=10),
                payload={},
            ),
        ),
    )

    phases = extract_phases(session)

    assert len(phases) == 1
    assert phases[0].start_time == started_at
    assert phases[0].end_time == started_at + timedelta(minutes=1)


def test_extract_phases_never_inverts_start_after_end_for_out_of_order_variant() -> None:
    """Regression for the 13,743 inverted session_phases/session_work_events
    rows found in the live archive (99%+ chatgpt-export).

    ChatGPT exports flatten a message tree into (position, variant_index)
    traversal order, which is not chronological: a regenerated/edited
    sibling variant can carry a real timestamp that predates (or postdates)
    the position it was flattened next to. Concretely reproduced from
    session chatgpt-export:0012f391-... in the live archive: an assistant
    reply belonging to an abandoned (non-active-path) branch was flattened
    between two later, accepted-path messages, but its own timestamp was
    ~4 hours earlier than both of them. Taking the first/last timestamp
    *in traversal order* as phase start/end inverted start > end.
    """
    day1_1757 = datetime(2024, 1, 11, 17, 57, 26, tzinfo=timezone.utc)
    day1_2150 = datetime(2024, 1, 11, 21, 50, 54, tzinfo=timezone.utc)
    day1_2152 = datetime(2024, 1, 11, 21, 52, 51, tzinfo=timezone.utc)
    day1_2155 = datetime(2024, 1, 11, 21, 55, 50, tzinfo=timezone.utc)
    day1_1757b = datetime(2024, 1, 11, 17, 57, 27, tzinfo=timezone.utc)  # abandoned-branch reply
    day1_2151 = datetime(2024, 1, 11, 21, 50, 55, tzinfo=timezone.utc)

    session = make_conv(
        id="conv-chatgpt-variant-reorder",
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

    phases = extract_phases(session)

    for phase in phases:
        if phase.start_time is not None and phase.end_time is not None:
            assert phase.end_time >= phase.start_time, (
                f"inverted phase: start={phase.start_time} end={phase.end_time} range={phase.message_range}"
            )

    # The out-of-order timestamp must land in its own phase (index 4 sits
    # more than 5 minutes, in either direction, from its traversal-order
    # neighbours), and the phase containing it must be flagged as having
    # required chronological reordering rather than silently reported as a
    # trustworthy monotonic range.
    outlier_phase = next(phase for phase in phases if phase.message_range == (4, 5))
    assert outlier_phase.start_time == day1_1757b
    assert outlier_phase.end_time == day1_1757b
    assert "chronological_reorder" not in outlier_phase.evidence  # single-message span, nothing to reorder

    # The phase spanning the edited-user-turn variants (m1..m3) must report
    # its true min/max envelope, not a raw first/last-in-traversal value.
    edit_phase = next(phase for phase in phases if phase.message_range == (1, 4))
    assert edit_phase.start_time == day1_2150
    assert edit_phase.end_time == day1_2155


def test_extract_phases_flags_reorder_evidence_without_inverting() -> None:
    """A within-phase timestamp regression too small to trigger a phase
    split on its own (each adjacent jump stays under the 5-minute gap) can
    still make the envelope (min/max) diverge from the naive first/last
    seen while walking messages in traversal order. That divergence must be
    surfaced honestly via the phase's `evidence`, even though it never
    produces an inverted start/end in this milder case.
    """
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    session = make_conv(
        id="conv-small-regression",
        origin=Provider.CHATGPT,
        title="Small in-phase regression",
        messages=MessageCollection(
            messages=[
                make_msg(id="m0", role="user", origin="chatgpt", text="a", timestamp=base + timedelta(minutes=3)),
                make_msg(id="m1", role="assistant", origin="chatgpt", text="b", timestamp=base + timedelta(minutes=1)),
                make_msg(id="m2", role="user", origin="chatgpt", text="c", timestamp=base + timedelta(minutes=4)),
            ]
        ),
    )

    phases = extract_phases(session)

    assert len(phases) == 1
    phase = phases[0]
    assert phase.start_time is not None
    assert phase.end_time is not None
    assert phase.end_time >= phase.start_time
    assert phase.start_time == base + timedelta(minutes=1)
    assert phase.end_time == base + timedelta(minutes=4)
    assert "chronological_reorder" in phase.evidence
