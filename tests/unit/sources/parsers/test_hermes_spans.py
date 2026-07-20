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
    # fs1.14: dispatch derives profile_root from source_path's parent
    # directory (production route, not a bespoke test-only call), so a real
    # source_path now yields a profile-qualified identity and a parent
    # session_links join key -- see test_dispatch_threads_source_path_directory_as_profile_root_for_atif.
    from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id

    expected_key = profile_key(REAL_ATOF_FIXTURE.parent)
    assert session.provider_session_id == f"observer:real-nemo-relay-session-redacted@profile-{expected_key}"
    assert session.parent_session_provider_id == qualified_session_id("real-nemo-relay-session-redacted", expected_key)
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

    # llm-error-1 (record 8) is a real scope=end with no matching scope=start
    # in this fixture -- genuine acquisition debt, surfaced explicitly rather
    # than silently indistinguishable from a normal completed llm/tool pair.
    unpaired = [event for event in events if event.event_type == "hermes_atof_unpaired_scope"]
    assert len(unpaired) == 1
    assert unpaired[0].payload["event_uuid"] == "llm-error-1"
    assert unpaired[0].payload["phase_observed"] == "end"
    assert unpaired[0].payload["phase_missing"] == "start"

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["unpaired_scope_debt"].status == "degraded"
    assert fidelity.capabilities["unpaired_scope_debt"].observed == 1

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


def test_atof_scope_start_without_end_is_surfaced_as_unpaired_debt() -> None:
    """A crashed request, or a truncated/rotated file cutting a pending scope
    in half, leaves a real scope=start with no matching scope=end. This must
    be visible as acquisition debt, not silently indistinguishable from a
    normal completed pair."""

    records: list[JSONDocument] = [
        {
            "atof_version": "0.1",
            "kind": "scope",
            "category": "tool",
            "scope_category": "start",
            "uuid": "tool-crashed-1",
            "timestamp": "2026-07-18T09:00:00Z",
            "name": "terminal",
            "metadata": {"session_id": "session-1", "tool_call_id": "call-1"},
        },
    ]
    session = hermes_spans.parse_atof_stream(records, "fallback-id")[0]

    tool_events = [e for e in session.session_events if e.event_type == "hermes_tool_execution_span"]
    assert len(tool_events) == 1  # the start-phase span itself is still recorded

    unpaired = [e for e in session.session_events if e.event_type == "hermes_atof_unpaired_scope"]
    assert len(unpaired) == 1
    assert unpaired[0].payload["event_uuid"] == "tool-crashed-1"
    assert unpaired[0].payload["category"] == "tool"
    assert unpaired[0].payload["phase_observed"] == "start"
    assert unpaired[0].payload["phase_missing"] == "end"

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["unpaired_scope_debt"].status == "degraded"
    assert fidelity.capabilities["unpaired_scope_debt"].observed == 1


def test_atof_scope_with_both_phases_is_not_flagged_unpaired() -> None:
    records: list[JSONDocument] = [
        {
            "atof_version": "0.1",
            "kind": "scope",
            "category": "tool",
            "scope_category": "start",
            "uuid": "tool-complete-1",
            "timestamp": "2026-07-18T09:00:00Z",
            "name": "terminal",
            "metadata": {"session_id": "session-1", "tool_call_id": "call-1"},
        },
        {
            "atof_version": "0.1",
            "kind": "scope",
            "category": "tool",
            "scope_category": "end",
            "uuid": "tool-complete-1",
            "timestamp": "2026-07-18T09:00:01Z",
            "name": "terminal",
            "metadata": {"session_id": "session-1", "tool_call_id": "call-1", "status": "completed"},
        },
    ]
    session = hermes_spans.parse_atof_stream(records, "fallback-id")[0]

    assert not any(e.event_type == "hermes_atof_unpaired_scope" for e in session.session_events)
    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert "unpaired_scope_debt" not in fidelity.capabilities


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
    # The summary message text is deliberately stable across replays (see
    # _atof_session/parse_atif_document's own note) and never embeds a live
    # skip count -- real_events == [] and generic_events == [] above already
    # prove the 3 malformed steps were skipped-and-counted, not silently
    # coerced into a fabricated event.
    summary_message = session.messages[0]
    assert summary_message.text == "Hermes ATIF trajectory: hermes-session-1"


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
    share the same profile-qualified identity -- fs1.14 fixed the prior
    behavior where ``hermes_observer_session_id_for`` stripped the profile
    qualifier off a real conversational session id, which is exactly what let
    two separate Hermes installs (profiles) reusing the same raw session id
    silently collapse onto one observer-evidence archive session."""

    qualified = "hermes-session-1@profile-abc123def456"
    assert hermes_spans.hermes_observer_session_id_for(qualified) == "observer:hermes-session-1@profile-abc123def456"
    assert hermes_spans.hermes_observer_session_id_for(qualified) == hermes_spans.observer_session_provider_id(
        "hermes-session-1", "abc123def456"
    )
    # Two different profiles reusing the same raw session id must NOT
    # collapse onto the same observer session identity.
    other_profile = "hermes-session-1@profile-fedcba654321"
    assert hermes_spans.hermes_observer_session_id_for(qualified) != hermes_spans.hermes_observer_session_id_for(
        other_profile
    )
    # A legacy/unqualified conversational id (no profile marker at all) falls
    # back to an unqualified observer id rather than fabricating a profile.
    assert hermes_spans.hermes_observer_session_id_for("hermes-session-1") == "observer:hermes-session-1"


def test_atif_asserts_profile_qualified_parent_session_link_when_profile_root_known(tmp_path: Path) -> None:
    """fs1.14: a known profile root makes this parser assert a real
    session_links parent edge back to the matching state-db conversational
    session, using the exact qualifier hermes_state.py computes for it."""

    profile_root = tmp_path / "hermes-install"
    payload = hermes_spans.marker_payload("hermes-session-1", _steps())
    session = hermes_spans.parse_atif_document(payload, "fallback-id", profile_root=profile_root)

    from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id

    expected_key = profile_key(profile_root)
    assert session.parent_session_provider_id == qualified_session_id("hermes-session-1", expected_key)
    assert session.provider_session_id == f"observer:hermes-session-1@profile-{expected_key}"

    correlation_event = session.session_events[0]
    assert correlation_event.event_type == "hermes_observer_trace_correlation"
    assert correlation_event.payload["profile_qualified"] is True
    assert correlation_event.payload["asserted_parent_session_provider_id"] == session.parent_session_provider_id

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["parent_session_link"].status == "inferred"


def test_atif_fails_closed_on_unknown_profile_root() -> None:
    """No profile root means no parent edge is asserted -- fail closed rather
    than guessing an unqualified, potentially cross-profile-colliding join key."""

    payload = hermes_spans.marker_payload("hermes-session-1", _steps())
    session = hermes_spans.parse_atif_document(payload, "fallback-id")

    assert session.parent_session_provider_id is None
    assert session.provider_session_id == "observer:hermes-session-1"

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["parent_session_link"].status == "absent"


def test_two_profiles_with_the_same_raw_session_id_do_not_collapse(tmp_path: Path) -> None:
    """The actual collapse bug fs1.14 fixes: two separate Hermes installs
    (profiles) reusing the same raw session id must produce two distinct
    observer-evidence archive sessions with two distinct parent edges, never
    one session silently overwriting the other."""

    profile_a = tmp_path / "install-a"
    profile_b = tmp_path / "install-b"
    payload = hermes_spans.marker_payload("hermes-session-1", _steps())

    session_a = hermes_spans.parse_atif_document(payload, "fallback-id", profile_root=profile_a)
    session_b = hermes_spans.parse_atif_document(payload, "fallback-id", profile_root=profile_b)

    assert session_a.provider_session_id != session_b.provider_session_id
    assert session_a.parent_session_provider_id != session_b.parent_session_provider_id


def test_atof_stream_asserts_profile_qualified_parent_session_link(tmp_path: Path) -> None:
    """Same fs1.14 wiring on the ATOF (JSONL scope/mark stream) route."""

    profile_root = tmp_path / "hermes-install"
    records = [
        {
            "atof_version": "0.1",
            "kind": "scope",
            "uuid": "u1",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "llm.request",
            "category": "llm",
            "scope_category": "start",
            "metadata": {"session_id": "hermes-session-2", "model": "m", "provider": "p"},
            "data": {},
        },
        {
            "atof_version": "0.1",
            "kind": "scope",
            "uuid": "u1",
            "timestamp": "2026-01-01T00:00:01Z",
            "name": "llm.request",
            "category": "llm",
            "scope_category": "end",
            "metadata": {"session_id": "hermes-session-2", "model": "m", "provider": "p"},
            "data": {},
        },
    ]
    sessions = hermes_spans.parse_atof_stream(records, "fallback-id", profile_root=profile_root)
    assert len(sessions) == 1
    session = sessions[0]

    from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id

    expected_key = profile_key(profile_root)
    assert session.parent_session_provider_id == qualified_session_id("hermes-session-2", expected_key)

    fidelity = hermes_spans.import_fidelity_declaration(session)
    assert fidelity.capabilities["parent_session_link"].status == "inferred"

    sessions_unknown_profile = hermes_spans.parse_atof_stream(records, "fallback-id")
    assert sessions_unknown_profile[0].parent_session_provider_id is None
    fidelity_unknown = hermes_spans.import_fidelity_declaration(sessions_unknown_profile[0])
    assert fidelity_unknown.capabilities["parent_session_link"].status == "absent"


def test_dispatch_threads_source_path_directory_as_profile_root_for_atif() -> None:
    """Production route: dispatch.py derives profile_root from spec.source_path's
    parent directory (the same convention hermes_state.py's own callers use),
    not a bespoke test-only code path."""

    payload = hermes_spans.marker_payload("hermes-session-3", _steps())
    sessions = parse_payload(
        Provider.HERMES,
        payload,
        "fallback-id",
        source_path="/home/example/.hermes/atif/hermes-session-3.json",
    )
    assert len(sessions) == 1
    session = sessions[0]

    from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id

    expected_key = profile_key(Path("/home/example/.hermes/atif"))
    assert session.parent_session_provider_id == qualified_session_id("hermes-session-3", expected_key)
