"""Hermes NeMo Relay ATIF trajectory importer (fs1.2).

Fixtures below build real-shaped ATIF-v1.7 documents (see the module
docstring of ``hermes_spans.py`` for the external NVIDIA/Hermes-fork sources
this schema is grounded in), not the pre-fix synthetic
``polylogue_artifact: "hermes_atif_trace"`` marker shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.sources.dispatch import detect_provider, parse_payload, parse_stream_payload
from polylogue.sources.import_explain import explain_import_path
from polylogue.sources.parsers import hermes_spans

REAL_ATIF_FIXTURE = Path(__file__).parents[3] / "fixtures/hermes/atif/nemo_relay_atif_v1.7_real_redacted.json"
REAL_ATOF_FIXTURE = Path(__file__).parents[3] / "fixtures/hermes/atof/nemo_relay_atof_v0.1_real_redacted.jsonl"


def _steps() -> list[JSONDocument]:
    return [
        {
            "source": "agent",
            "tool_calls": [
                {"function_name": "read_file", "tool_call_id": "call-1", "arguments": {"path": "README.md"}}
            ],
            "observation": {"results": [{"content": "file contents..."}]},
        },
        {"source": "agent", "message": "I read the file and it looks fine."},
    ]


def test_looks_like_atif_payload_requires_real_atif_schema_version_and_session_and_steps() -> None:
    payload = hermes_spans.marker_payload("hermes-session-1", _steps())
    assert hermes_spans.looks_like_atif_payload(payload)
    # No schema_version at all.
    assert not hermes_spans.looks_like_atif_payload({"session_id": "x", "steps": []})
    # schema_version present but not the ATIF family.
    assert not hermes_spans.looks_like_atif_payload({"schema_version": "OTEL-v1", "session_id": "x", "steps": []})
    # The old, pre-fix synthetic marker shape must NOT match anymore -- that
    # was exactly the review finding (a self-referential detector that could
    # only ever recognize this repo's own test fixture).
    assert not hermes_spans.looks_like_atif_payload(
        {"polylogue_artifact": "hermes_atif_trace", "session_id": "x", "spans": []}
    )
    # Missing steps.
    assert not hermes_spans.looks_like_atif_payload({"schema_version": "ATIF-v1.7", "session_id": "x"})


def test_dispatch_detects_and_parses_atif_trace_through_the_real_pipeline() -> None:
    """Production route: the shared detector/parser dispatch, not a bespoke test-only call."""

    payload = hermes_spans.marker_payload("hermes-session-1", _steps())
    assert detect_provider(payload) is Provider.HERMES

    sessions = parse_payload(Provider.HERMES, payload, "fallback-id")
    assert len(sessions) == 1
    session = sessions[0]
    assert session.provider_session_id == "observer:hermes-session-1"
    assert session.source_name is Provider.HERMES
    # Never a duplicated transcript: the only "message" is a bounded summary.
    assert len(session.messages) == 1
    assert len(session.messages[0].text or "") < 500


def test_real_nemo_relay_atif_fixture_reaches_the_hermes_parser() -> None:
    """A redacted live export guards the actual wire shape, not a synthetic marker."""

    payload = json.loads(REAL_ATIF_FIXTURE.read_text())
    assert hermes_spans.looks_like_atif_payload(payload)
    assert detect_provider(payload) is Provider.HERMES

    session = parse_payload(Provider.HERMES, payload, "fallback-id")[0]
    assert session.provider_session_id == "observer:real-nemo-relay-session-redacted"
    assert len(payload["steps"]) == 6
    llm_events = [event for event in session.session_events if event.event_type == "hermes_llm_request_span"]
    assert len(llm_events) == 5
    assert all(event.payload["message_char_len"] == len("<redacted>") for event in llm_events)

    # Step 6 (real evidence: 4 parallel tool_calls plus observation.results,
    # drawn from a separate live trajectory -- see fixtures/hermes/atif/README.md)
    # proves the tool_calls step shape without copying arguments/observation content.
    tool_events = [event for event in session.session_events if event.event_type == "hermes_tool_execution_span"]
    assert len(tool_events) == 4
    assert {event.payload["function_name"] for event in tool_events} == {"process", "terminal", "search_files"}
    assert all(event.payload["has_arguments"] is True for event in tool_events)
    assert all(event.payload["has_observation"] is True for event in tool_events)

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["llm_request_spans"].status == "exact"
    assert fidelity.capabilities["tool_execution_spans"].status == "exact"


def test_real_nemo_relay_atof_fixture_reaches_the_stream_parser_without_copying_content() -> None:
    """Production dependencies: JSONL detection, stream dispatch, session grouping.

    Mutation: remove Hermes from the stream providers, classify this envelope as
    a generic hook sidecar, or copy ATOF data blobs into events. This test's
    detector, produced-session, semantic-count, and payload-hygiene assertions
    fail through the real import route.
    """
    records = [json.loads(line) for line in REAL_ATOF_FIXTURE.read_text().splitlines()]
    assert detect_provider(records) is Provider.HERMES
    assert hermes_spans.looks_like_atof_payload(records[0])

    sessions = parse_stream_payload(
        Provider.HERMES,
        iter(records),
        "fallback-id",
        source_path=str(REAL_ATOF_FIXTURE),
    )
    assert len(sessions) == 1
    session = sessions[0]
    assert session.provider_session_id == "observer:real-nemo-relay-session-redacted"
    events = session.session_events
    assert sum(event.event_type == "hermes_llm_request_span" for event in events) == 3
    assert sum(event.event_type == "hermes_tool_execution_span" for event in events) == 2
    assert sum(event.event_type == "hermes_decision_span" for event in events) == 2
    assert sum(event.event_type == "hermes_subagent_span" for event in events) == 1
    assert sum(event.event_type == "hermes_error_span" for event in events) == 1
    # The outer session-scope start/end pair (kind=scope,category=agent) and
    # the hermes.session.end mark are real, well-defined lifecycle evidence
    # (confirmed against ~/.hermes/observability/nemo-relay/atof/events.jsonl)
    # -- admitted as typed context_span, not left in the unrecognized bucket.
    assert sum(event.event_type == "hermes_context_span" for event in events) == 3
    assert sum(event.event_type == "hermes_observer_span" for event in events) == 0
    assert "<redacted>" not in repr([event.payload for event in events])

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.acquisition_method == "jsonl_stream"
    assert fidelity.capabilities["llm_request_spans"].status == "exact"
    assert fidelity.capabilities["tool_execution_spans"].status == "exact"
    assert fidelity.capabilities["decision_points"].status == "exact"
    assert fidelity.capabilities["error_taxonomy"].status == "exact"
    assert fidelity.capabilities["subagent_delegation"].status == "exact"
    assert fidelity.capabilities["context_events"].status == "exact"
    assert "unrecognized_atof_events" not in fidelity.capabilities


def test_atof_stream_is_idempotent_across_append_replay_and_keeps_scope_edges() -> None:
    records = [json.loads(line) for line in REAL_ATOF_FIXTURE.read_text().splitlines()]
    once = parse_stream_payload(Provider.HERMES, iter(records), "fallback-id")
    replayed = parse_stream_payload(Provider.HERMES, iter([*records, *records]), "fallback-id")
    assert [session.model_dump(mode="json") for session in replayed] == [
        session.model_dump(mode="json") for session in once
    ]

    tool_phases = [
        event.payload["scope_category"]
        for event in once[0].session_events
        if event.event_type == "hermes_tool_execution_span"
    ]
    assert tool_phases == ["start", "end"]


def test_atof_unmatched_response_and_turn_context_are_normalized_not_dropped() -> None:
    records: list[JSONDocument] = [
        {
            "atof_version": "0.1",
            "kind": "mark",
            "uuid": "turn-1",
            "timestamp": "2026-07-18T09:00:00Z",
            "name": "hermes.turn.start",
            "metadata": {"session_id": "session-1"},
        },
        {
            "atof_version": "0.1",
            "kind": "mark",
            "uuid": "unmatched-1",
            "timestamp": "2026-07-18T09:00:01Z",
            "name": "hermes.api.response.unmatched",
            "metadata": {"session_id": "session-1"},
        },
    ]
    session = hermes_spans.parse_atof_stream(records, "fallback-id")[0]
    assert [event.event_type for event in session.session_events[1:]] == [
        "hermes_context_span",
        "hermes_error_span",
    ]
    assert session.session_events[-1].payload["outcome"] == "unmatched_response"


def test_atof_agent_scope_and_session_end_mark_become_typed_context_spans() -> None:
    """The outer session-scope start/end pair (kind=scope,category=agent) and a
    hermes.session.end mark are real lifecycle evidence -- confirmed against
    ~/.hermes/observability/nemo-relay/atof/events.jsonl -- and must not fall
    into the generic unrecognized-event bucket."""

    records: list[JSONDocument] = [
        {
            "atof_version": "0.1",
            "kind": "scope",
            "category": "agent",
            "scope_category": "start",
            "uuid": "scope-parent",
            "timestamp": "2026-07-18T09:00:00Z",
            "name": "hermes-session-example",
            "metadata": {"session_id": "session-1"},
        },
        {
            "atof_version": "0.1",
            "kind": "mark",
            "uuid": "session-end-1",
            "timestamp": "2026-07-18T09:00:01Z",
            "name": "hermes.session.end",
            "metadata": {"session_id": "session-1"},
        },
        {
            "atof_version": "0.1",
            "kind": "scope",
            "category": "agent",
            "scope_category": "end",
            "uuid": "scope-parent",
            "timestamp": "2026-07-18T09:00:02Z",
            "name": "hermes-session-example",
            "metadata": {"session_id": "session-1"},
        },
    ]
    session = hermes_spans.parse_atof_stream(records, "fallback-id")[0]
    context_events = [event for event in session.session_events if event.event_type == "hermes_context_span"]
    assert [event.payload["context_event"] for event in context_events] == [
        "hermes.session.start",
        "hermes.session.end",
        "hermes.session.end",
    ]
    assert not any(event.event_type == "hermes_observer_span" for event in session.session_events)


def test_import_explain_uses_atof_stream_fidelity_for_a_real_fixture() -> None:
    entry = explain_import_path(REAL_ATOF_FIXTURE, source_name="hermes").entries[0]
    assert entry.detected_provider == "hermes"
    assert entry.parser == "hermes"
    assert entry.produced.sessions == 1
    assert entry.fidelity is not None
    assert entry.fidelity.acquisition_method == "jsonl_stream"


def test_import_explain_uses_atif_fidelity_for_a_real_fixture() -> None:
    entry = explain_import_path(REAL_ATIF_FIXTURE).entries[0]
    assert entry.detected_provider == "hermes"
    assert entry.parser == "hermes"
    assert entry.produced.sessions == 1
    assert entry.fidelity is not None
    assert entry.fidelity.capabilities["llm_request_spans"].status == "exact"


def test_atif_parse_is_idempotent_and_deterministic() -> None:
    """Same document parsed twice yields byte-identical structural output."""

    payload = hermes_spans.marker_payload("hermes-session-1", _steps())
    first = hermes_spans.parse_atif_document(payload, "fallback-id")
    second = hermes_spans.parse_atif_document(payload, "fallback-id")
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_tool_call_steps_become_tool_execution_spans_without_copying_arguments() -> None:
    """Payload hygiene: a tool_calls entry's actual ``arguments``/``observation``
    content is never copied into the span event -- only bounded presence
    evidence (see module docstring)."""

    payload = hermes_spans.marker_payload("hermes-session-1", _steps())
    session = hermes_spans.parse_atif_document(payload, "fallback-id")

    tool_events = [event for event in session.session_events if event.event_type == "hermes_tool_execution_span"]
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.payload["function_name"] == "read_file"
    assert event.payload["tool_call_id"] == "call-1"
    assert event.payload["has_arguments"] is True
    assert event.payload["has_observation"] is True
    # The real argument/observation content must not appear anywhere in the payload.
    assert "README.md" not in repr(event.payload)
    assert "file contents" not in repr(event.payload)


def test_message_only_steps_become_llm_response_evidence_without_copying_text() -> None:
    payload = hermes_spans.marker_payload(
        "hermes-session-1", _steps(), agent={"name": "hermes", "version": "1", "model_name": "claude-example"}
    )
    session = hermes_spans.parse_atif_document(payload, "fallback-id")

    llm_events = [event for event in session.session_events if event.event_type == "hermes_llm_request_span"]
    assert len(llm_events) == 1
    event = llm_events[0]
    assert event.payload["model"] == "claude-example"
    assert event.payload["message_char_len"] == len("I read the file and it looks fine.")
    assert "I read the file and it looks fine." not in repr(event.payload)


def test_subagent_trajectories_become_subagent_span_events() -> None:
    subagents: list[JSONValue] = [
        {
            "session_id": "docs-child-session",
            "agent": {"name": "Hermes Agent E2E"},
            "steps": [{"source": "agent", "tool_calls": [{"function_name": "terminal", "tool_call_id": "call-2"}]}],
        }
    ]
    payload = hermes_spans.marker_payload("hermes-session-1", [], subagent_trajectories=subagents)
    session = hermes_spans.parse_atif_document(payload, "fallback-id")

    subagent_events = [event for event in session.session_events if event.event_type == "hermes_subagent_span"]
    assert len(subagent_events) == 1
    event = subagent_events[0]
    assert event.payload["subagent_session_id"] == "docs-child-session"
    assert event.payload["subagent_agent_name"] == "Hermes Agent E2E"
    assert event.payload["subagent_step_count"] == 1

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["subagent_delegation"].status == "inferred"
    # Physical topology_edges materialization remains explicitly out of scope.
    assert fidelity.capabilities["topology_edges"].status == "absent"


def test_unrecognized_step_shape_becomes_generic_observer_span_not_dropped() -> None:
    """Ambiguous input is handled deterministically (AC): a step with none of the
    documented shapes (tool_calls/message/observation) is visible, not lost."""

    steps: list[JSONValue] = [{"source": "agent", "unexpected_field": "future-shape"}]
    payload = hermes_spans.marker_payload("hermes-session-1", steps)
    session = hermes_spans.parse_atif_document(payload, "fallback-id")

    generic_events = [event for event in session.session_events if event.event_type == "hermes_observer_span"]
    assert len(generic_events) == 1
    assert generic_events[0].payload["shape"] == "unrecognized"

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["unrecognized_step_shapes"].status == "degraded"
    assert fidelity.capabilities["unrecognized_step_shapes"].observed == 1


def test_malformed_steps_and_tool_calls_are_skipped_and_counted_not_crashing() -> None:
    steps: list[JSONValue] = [
        "not-a-dict",  # not even a document
        {"source": "agent", "tool_calls": [{"function_name": "x"}]},  # tool_call missing tool_call_id
        {"source": "agent", "tool_calls": ["not-a-dict"]},  # tool_calls entry not even a document
    ]
    payload = hermes_spans.marker_payload("hermes-session-1", steps)
    session = hermes_spans.parse_atif_document(payload, "fallback-id")
    real_events = [
        event
        for event in session.session_events
        if event.event_type not in {"hermes_observer_span", "hermes_observer_trace_correlation"}
    ]
    assert real_events == []

    # A non-object step is genuinely skipped-and-counted, not silently
    # coerced into a generic ``hermes_observer_span`` event (review-adjacent
    # fix: ``json_document()`` returns ``{}``, never ``None``, on coercion
    # failure, so the prior ``step is None`` check was dead code that let a
    # non-object step slip through as a fabricated "unrecognized shape"
    # event instead of being counted as unparseable).
    generic_events = [event for event in session.session_events if event.event_type == "hermes_observer_span"]
    assert generic_events == []
    summary_message = session.messages[0]
    assert summary_message.text is not None
    assert "3 unparseable" in summary_message.text


def test_decision_points_and_error_taxonomy_are_honestly_absent_not_fabricated() -> None:
    """ATIF's documented step schema carries no approval/error-hook vocabulary
    (that lives only in the separate raw ATOF event stream, not ingested by
    this pass) -- these capabilities must be declared 'absent', never guessed
    from a schema that doesn't carry them."""

    payload = hermes_spans.marker_payload("hermes-session-1", _steps())
    session = hermes_spans.parse_atif_document(payload, "fallback-id")
    fidelity = hermes_spans.import_fidelity_declaration(session)

    assert fidelity.capabilities["decision_points"].status == "absent"
    assert fidelity.capabilities["error_taxonomy"].status == "absent"
    assert "raw ATOF event stream" in fidelity.capabilities["decision_points"].detail
    # llm_request_spans and tool_execution_spans are the two capabilities whose
    # field mapping has been independently confirmed against real ATIF bytes
    # (see fixtures/hermes/atif/README.md) -- "exact" here means the mapping
    # itself is proven, not that this particular (synthetic) session is real.
    assert fidelity.capabilities["llm_request_spans"].status == "exact"
    assert fidelity.capabilities["tool_execution_spans"].status == "exact"
    exempt = {"llm_request_spans", "tool_execution_spans"}
    assert all(cap.status != "exact" for name, cap in fidelity.capabilities.items() if name not in exempt)


def test_observer_session_id_correlates_with_qualified_state_db_session_id() -> None:
    """Read-side join key: the observer-evidence session and the state-db session
    share the raw Hermes session id, even though the state-db session id is
    profile-qualified."""

    qualified = "hermes-session-1@profile-abc123def456"
    assert hermes_spans.hermes_observer_session_id_for(qualified) == "observer:hermes-session-1"
    assert hermes_spans.hermes_observer_session_id_for(qualified) == hermes_spans.observer_session_provider_id(
        "hermes-session-1"
    )
