"""Hermes NeMo Relay ATOF/ATIF observer-trace importer (fs1.2)."""

from __future__ import annotations

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers import hermes_spans


def _spans() -> list[JSONDocument]:
    return [
        {
            "hook_type": "pre_api_request",
            "span_id": "req-1",
            "timestamp": "2026-07-10T09:00:00Z",
            "model": "claude-example",
        },
        {
            "hook_type": "post_api_request",
            "span_id": "req-1",
            "timestamp": "2026-07-10T09:00:01Z",
            "duration_ms": 900,
            "input_tokens": 120,
            "output_tokens": 30,
        },
        {
            "hook_type": "pre_tool_call",
            "span_id": "tool-1",
            "tool_call_id": "call-1",
            "tool_name": "read_file",
            "timestamp": "2026-07-10T09:00:02Z",
        },
        {
            "hook_type": "post_tool_call",
            "span_id": "tool-1",
            "tool_call_id": "call-1",
            "timestamp": "2026-07-10T09:00:03Z",
            "duration_ms": 40,
            "status": "ok",
        },
        {
            "hook_type": "approval_request",
            "span_id": "appr-1",
            "timestamp": "2026-07-10T09:00:04Z",
        },
        # No matching approval_response: deliberately unpaired.
        {
            "hook_type": "error",
            "span_id": "err-1",
            "timestamp": "2026-07-10T09:00:05Z",
            "error_type": "timeout",
            "retryable": True,
        },
    ]


def test_looks_like_atif_payload_requires_marker_and_session_and_spans() -> None:
    payload = hermes_spans.marker_payload("hermes-session-1", _spans())
    assert hermes_spans.looks_like_atif_payload(payload)
    assert not hermes_spans.looks_like_atif_payload({"session_id": "x", "spans": []})
    assert not hermes_spans.looks_like_atif_payload({"polylogue_artifact": "hermes_atif_trace", "session_id": "x"})


def test_dispatch_detects_and_parses_atif_trace_through_the_real_pipeline() -> None:
    """Production route: the shared detector/parser dispatch, not a bespoke test-only call."""

    payload = hermes_spans.marker_payload("hermes-session-1", _spans())
    assert detect_provider(payload) is Provider.HERMES

    sessions = parse_payload(Provider.HERMES, payload, "fallback-id")
    assert len(sessions) == 1
    session = sessions[0]
    assert session.provider_session_id == "observer:hermes-session-1"
    assert session.source_name is Provider.HERMES
    # Never a duplicated transcript: the only "message" is a bounded summary.
    assert len(session.messages) == 1
    assert len(session.messages[0].text or "") < 500


def test_atif_parse_is_idempotent_and_deterministic() -> None:
    """Same document parsed twice yields byte-identical structural output."""

    payload = hermes_spans.marker_payload("hermes-session-1", _spans())
    first = hermes_spans.parse_atif_document(payload, "fallback-id")
    second = hermes_spans.parse_atif_document(payload, "fallback-id")
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_unpaired_spans_are_reported_not_silently_dropped() -> None:
    payload = hermes_spans.marker_payload("hermes-session-1", _spans())
    session = hermes_spans.parse_atif_document(payload, "fallback-id")

    correlation_events = [
        event for event in session.session_events if event.event_type != "hermes_observer_trace_correlation"
    ]
    assert len(correlation_events) == len(_spans())  # nothing dropped

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert any("never observed their paired counterpart" in caveat for caveat in fidelity.caveats)
    assert fidelity.capabilities["decision_points"].status == "degraded"
    # error hooks are unpaired-exempt (no counterpart expected) and still counted.
    assert fidelity.capabilities["error_taxonomy"].observed == 1


def test_unrecognized_hook_type_becomes_generic_observer_span_not_dropped() -> None:
    """Ambiguous input is handled deterministically (AC): unknown kinds are visible, not lost."""

    spans: list[JSONValue] = [{"hook_type": "future_hook_kind", "span_id": "s-1", "timestamp": "2026-07-10T09:00:00Z"}]
    payload = hermes_spans.marker_payload("hermes-session-1", spans)
    session = hermes_spans.parse_atif_document(payload, "fallback-id")

    generic_events = [event for event in session.session_events if event.event_type == "hermes_observer_span"]
    assert len(generic_events) == 1
    assert generic_events[0].payload["hook_type"] == "future_hook_kind"


def test_malformed_spans_are_skipped_and_counted_not_crashing() -> None:
    spans: list[JSONValue] = [
        {"hook_type": "pre_tool_call"},  # missing span_id
        {"span_id": "s-2"},  # missing hook_type
        "not-a-dict",  # not even a document
    ]
    payload = hermes_spans.marker_payload("hermes-session-1", spans)
    session = hermes_spans.parse_atif_document(payload, "fallback-id")
    real_events = [event for event in session.session_events if event.event_type != "hermes_observer_trace_correlation"]
    assert real_events == []


def test_observer_session_id_correlates_with_qualified_state_db_session_id() -> None:
    """Read-side join key: the observer-evidence session and the state-db session
    share the raw Hermes session id, even though the state-db session id is
    profile-qualified."""

    qualified = "hermes-session-1@profile-abc123def456"
    assert hermes_spans.hermes_observer_session_id_for(qualified) == "observer:hermes-session-1"
    assert hermes_spans.hermes_observer_session_id_for(qualified) == hermes_spans.observer_session_provider_id(
        "hermes-session-1"
    )
