"""Phase-extraction regression tests covering the session-events fallback (#1624)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.phase.extraction import extract_phases
from polylogue.archive.session.events import SessionEvent
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.types import SessionEventId, SessionId
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
