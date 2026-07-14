"""Hermes lifecycle-event taxonomy and snapshot reconciliation (fs1.7)."""

from __future__ import annotations

from polylogue.sources.parsers.hermes_lifecycle import (
    DURABLE_FINALIZE,
    PER_TURN_END,
    TOOL_FINISH,
    TOOL_START,
    HermesLifecycleEvent,
    reconcile_lifecycle_events,
)


def _event(event_id: str, event_type: str, at_ms: int, **payload: object) -> HermesLifecycleEvent:
    return HermesLifecycleEvent(
        event_id=event_id,
        event_type=event_type,
        session_native_id="hermes-session-1",
        observed_at_ms=at_ms,
        payload=payload,
    )


def test_reconciliation_reports_complete_stream_as_complete() -> None:
    events = [
        _event("e1", TOOL_START, 0, tool_call_id="call-1"),
        _event("e2", TOOL_FINISH, 1, tool_call_id="call-1"),
        _event("e3", DURABLE_FINALIZE, 2),
    ]
    report = reconcile_lifecycle_events("hermes-session-1", events)
    assert report.complete
    assert report.finalized
    assert report.unpaired_event_ids == ()
    assert report.caveats == ()


def test_reconciliation_surfaces_unpaired_start_without_finish() -> None:
    events = [_event("e1", TOOL_START, 0, tool_call_id="call-1")]
    report = reconcile_lifecycle_events("hermes-session-1", events)
    assert not report.complete
    assert report.unpaired_event_ids == ("e1",)
    assert any("paired counterpart" in caveat for caveat in report.caveats)


def test_reconciliation_flags_per_turn_end_without_durable_finalize() -> None:
    """fs1.7 AC: per-turn end is never conflated with durable finalization."""

    events = [_event("e1", PER_TURN_END, 0, turn_id="turn-1")]
    report = reconcile_lifecycle_events("hermes-session-1", events)
    assert not report.finalized
    assert any("no durable on_session_finalize" in caveat for caveat in report.caveats)


def test_reconciliation_flags_events_referencing_unknown_snapshot_messages() -> None:
    events = [_event("e1", TOOL_START, 0, tool_call_id="call-1", message_id="msg-does-not-exist")]
    report = reconcile_lifecycle_events(
        "hermes-session-1",
        events,
        snapshot_message_ids=frozenset({"msg-1", "msg-2"}),
    )
    assert report.events_referencing_unknown_messages == ("e1",)
    assert not report.complete


def test_reconciliation_accepts_events_referencing_known_snapshot_messages() -> None:
    events = [_event("e1", TOOL_START, 0, tool_call_id="call-1", message_id="msg-1")]
    report = reconcile_lifecycle_events(
        "hermes-session-1",
        events,
        snapshot_message_ids=frozenset({"msg-1"}),
    )
    assert report.events_referencing_unknown_messages == ()
