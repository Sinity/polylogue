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


# ── Adversarial/malformed-stream shapes (review fix: happy-path-only was the
# gap; a real drained stream is not guaranteed to arrive in causal order or
# without redelivery duplicates) ────────────────────────────────────────────


def test_reconciliation_pairs_a_finish_event_whose_timestamp_precedes_its_start() -> None:
    """Clock skew / delayed-but-earlier-stamped delivery: the finish event's
    ``observed_at_ms`` is smaller than its start's. Pairing must be by
    correlation-id membership, not temporal processing order -- a real
    finish that arrived must never be reported as a gap merely because
    sorting placed it before the start it pairs with."""

    events = [
        _event("e-finish", TOOL_FINISH, 0, tool_call_id="call-1"),
        _event("e-start", TOOL_START, 100, tool_call_id="call-1"),
    ]
    report = reconcile_lifecycle_events("hermes-session-1", events)
    assert report.complete
    assert report.unpaired_event_ids == ()


def test_reconciliation_collapses_duplicate_start_markers_onto_one_pairing_slot() -> None:
    """At-least-once hook redelivery can enqueue the same start event twice
    (same correlation id, distinct event ids). A single finish must satisfy
    both -- redelivery duplication is not itself an application-level gap."""

    events = [
        _event("e1", TOOL_START, 0, tool_call_id="call-1"),
        _event("e1-retry", TOOL_START, 1, tool_call_id="call-1"),
        _event("e2", TOOL_FINISH, 2, tool_call_id="call-1"),
    ]
    report = reconcile_lifecycle_events("hermes-session-1", events)
    assert report.complete
    assert report.unpaired_event_ids == ()
    assert report.events_by_type[TOOL_START] == 2


def test_reconciliation_survives_duplicate_durable_finalize_markers() -> None:
    """A redelivered ``on_session_finalize`` must not un-finalize or double-count
    in a way that breaks ``finalized``; presence, not count, is what matters."""

    events = [
        _event("e1", DURABLE_FINALIZE, 0),
        _event("e2", DURABLE_FINALIZE, 1),
    ]
    report = reconcile_lifecycle_events("hermes-session-1", events)
    assert report.finalized
    assert report.events_by_type[DURABLE_FINALIZE] == 2


def test_reconciliation_still_reports_a_real_gap_when_only_one_of_two_calls_finishes() -> None:
    """Two independent tool calls, only one finishes: the genuine gap (call-2)
    must still surface even in the presence of a duplicate/out-of-order call-1
    pairing -- the order-independent rewrite must not paper over real gaps."""

    events = [
        _event("e1", TOOL_FINISH, 0, tool_call_id="call-1"),  # out-of-order finish
        _event("e2", TOOL_START, 1, tool_call_id="call-1"),
        _event("e3", TOOL_START, 2, tool_call_id="call-2"),  # never finishes
    ]
    report = reconcile_lifecycle_events("hermes-session-1", events)
    assert not report.complete
    assert report.unpaired_event_ids == ("e3",)
