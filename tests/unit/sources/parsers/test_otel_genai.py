"""Synthetic OTLP-JSON GenAI parser and dispatch contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

from polylogue.core.enums import Origin, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, detect_provider, parse_payload
from polylogue.sources.origin_specs import SEMCONV_SCHEMA_URL
from polylogue.sources.parsers import otel_genai

FIXTURE = Path(__file__).parents[3] / "fixtures" / "otel-genai" / "trace.json"


def _payload() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], payload["resourceSpans"][0]["scopeSpans"][0]["spans"])


def test_otel_genai_reaches_dispatch_and_preserves_typed_messages() -> None:
    payload = _payload()

    assert detect_provider(payload) is Provider.OTEL_GENAI
    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert len(sessions) == 1
    session = sessions[0]
    assert session.provider_session_id == "synthetic-agent:conversation:conversation-demo-7"
    assert origin_from_provider(session.source_name) is Origin.OTEL_GENAI
    assert [message.text for message in session.messages[:2]] == [
        "Find tomorrow's weather.",
        "I'll check the forecast.",
    ]
    assistant = session.messages[1]
    assert (assistant.input_tokens, assistant.output_tokens, assistant.cache_read_tokens) == (18, 6, 0)
    blocks = [block for message in session.messages for block in message.blocks]
    assert [block.type.value for block in blocks] == ["tool_use", "tool_result", "tool_use", "tool_result"]
    assert blocks[1].tool_outcome is ToolOutcome.UNKNOWN
    assert blocks[1].outcome_unknown_reason == ToolResultUnknownReason.NOT_REPORTED.value
    assert blocks[3].tool_outcome is ToolOutcome.ERROR
    evidence = [event.payload for event in session.session_events if event.event_type == "otel_span_evidence"]
    tool = next(row for row in evidence if row["span_id"] == "00f067aa0ba902b7")
    assert tool["parent_span_id"] == "b7ad6b7169203331"
    assert tool["attributes"]["vendor.experimental.signal"] == "retained-verbatim"  # type: ignore[index]


def test_otel_genai_keeps_unsupported_schema_as_evidence_without_messages() -> None:
    payload = _payload()
    unsupported_scope = copy.deepcopy(payload["resourceSpans"][0]["scopeSpans"][0])
    unsupported_scope["schemaUrl"] = "https://example.invalid/genai/99.0.0"
    unsupported_span = unsupported_scope["spans"][1]
    unsupported_span["traceId"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    unsupported_span["spanId"] = "bbbbbbbbbbbbbbbb"
    payload["resourceSpans"][0]["scopeSpans"].append(unsupported_scope)

    session = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")[0]

    event = next(
        event
        for event in session.session_events
        if event.event_type == "otel_span_evidence" and event.payload["span_id"] == "bbbbbbbbbbbbbbbb"
    )
    assert event.payload["schema_url_status"] == "unsupported"
    assert not any(
        message.provider_message_id.startswith("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:") for message in session.messages
    )


def test_otel_genai_falls_back_to_trace_identity_when_conversation_is_absent() -> None:
    payload = _payload()
    for span in _spans(payload):
        span["attributes"] = [
            attribute for attribute in span["attributes"] if attribute["key"] != "gen_ai.conversation.id"
        ]

    session = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")[0]

    assert session.provider_session_id.endswith(":trace:4bf92f3577b34da6a3ce929d0e0e4736")
    assert otel_genai.looks_like(payload)


def test_otel_genai_retains_usage_only_chat_span() -> None:
    payload = _payload()
    chat = _spans(payload)[1]
    _spans(payload)[:] = [chat]
    chat["attributes"] = [
        attribute
        for attribute in chat["attributes"]
        if attribute["key"] not in {"gen_ai.input.messages", "gen_ai.output.messages"}
    ]

    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert len(sessions) == 1
    assert sessions[0].messages == []
    assert sessions[0].session_events[0].payload["usage_fidelity"] == "present"
    usage_events = [event for event in sessions[0].session_events if event.event_type == "message_usage"]
    assert len(usage_events) == 1
    assert usage_events[0].timestamp == "2025-01-01T00:00:01+00:00"
    assert usage_events[0].source_message_provider_id is None
    assert usage_events[0].payload == {
        "last_token_usage": {"input_tokens": 18, "output_tokens": 6, "cached_input_tokens": 0},
        "model": "gpt-4.1-mini",
    }
    assert admit_parsed_sessions_for_publication(sessions, provider=Provider.OTEL_GENAI, source_path=None) == sessions


def test_otel_genai_inherits_conversation_from_trace_parent() -> None:
    payload = _payload()
    for span in (_spans(payload)[0], _spans(payload)[2]):
        span["attributes"] = [
            attribute for attribute in span["attributes"] if attribute["key"] != "gen_ai.conversation.id"
        ]

    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "synthetic-agent:conversation:conversation-demo-7"
    assert len(sessions[0].session_events) == 3
    assert sum(block.type.value == "tool_result" for message in sessions[0].messages for block in message.blocks) == 2


def test_otel_genai_assigns_span_usage_to_one_assistant_output() -> None:
    payload = _payload()
    chat = _spans(payload)[1]
    output = next(attribute for attribute in chat["attributes"] if attribute["key"] == "gen_ai.output.messages")
    output["value"] = {
        "stringValue": json.dumps(
            [
                {"role": "assistant", "content": "First response."},
                {"role": "assistant", "content": "Second response."},
            ]
        )
    }

    session = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")[0]
    outputs = [message for message in session.messages if ":output:" in message.provider_message_id]

    assert [message.text for message in outputs] == ["First response.", "Second response."]
    assert [(message.input_tokens, message.output_tokens, message.cache_read_tokens) for message in outputs] == [
        (18, 6, 0),
        (None, None, None),
    ]
    assert not any(event.event_type == "message_usage" for event in session.session_events)


def test_otel_genai_token_bearing_output_names_its_model() -> None:
    """Usage rollups aggregate by model, so the token-bearing message carries one.

    Anti-vacuity: drop ``model_name`` from ``_messages_for_span`` and the
    assistant output's tokens join no model's usage.
    """
    session = parse_payload(Provider.OTEL_GENAI, _payload(), "ignored-file-stem")[0]
    token_bearing = [message for message in session.messages if message.output_tokens is not None]

    assert token_bearing
    assert {message.model_name for message in token_bearing} == {"gpt-4.1-mini"}
    assert all(message.model_name is None for message in session.messages if ":input:" in message.provider_message_id)


def test_duplicate_span_copies_normalize_once_in_any_wire_order() -> None:
    payload = _payload()
    chat = _spans(payload)[1]
    _spans(payload).append(copy.deepcopy(chat))

    forward = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")
    _spans(payload).reverse()
    reversed_result = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert [session.model_dump(mode="json") for session in forward] == [
        session.model_dump(mode="json") for session in reversed_result
    ]
    assert len(forward[0].messages) == 6
    assert not any(event.event_type == "otel_conflicting_span_id" for event in forward[0].session_events)


def test_conflicting_span_chooses_one_copy_before_conversation_grouping() -> None:
    payload = _payload()
    chat = _spans(payload)[1]
    conflict = copy.deepcopy(chat)
    conflict["startTimeUnixNano"] = "1735689609000000000"
    conversation = next(attr for attr in conflict["attributes"] if attr["key"] == "gen_ai.conversation.id")
    conversation["value"] = {"stringValue": "different-conversation"}
    _spans(payload).extend((copy.deepcopy(chat), conflict, copy.deepcopy(conflict)))

    forward = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")
    _spans(payload).reverse()
    reversed_result = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert [session.model_dump(mode="json") for session in forward] == [
        session.model_dump(mode="json") for session in reversed_result
    ]
    assert len(forward) == 1
    assert forward[0].provider_session_id == "synthetic-agent:conversation:conversation-demo-7"
    assert len(forward[0].messages) == 6
    conflicts = [event for event in forward[0].session_events if event.event_type == "otel_conflicting_span_id"]
    assert len(conflicts) == 1
    conflicting_span = cast(dict[str, object], conflicts[0].payload["conflicting_span"])
    assert conflicting_span["startTimeUnixNano"] == "1735689609000000000"


def test_supported_schema_wins_conflicting_unsupported_copy() -> None:
    payload = _payload()
    scope = payload["resourceSpans"][0]["scopeSpans"][0]
    unsupported = copy.deepcopy(scope)
    unsupported["schemaUrl"] = "https://example.invalid/genai/99.0.0"
    unsupported["spans"] = [copy.deepcopy(_spans(payload)[1])]
    unsupported["spans"][0]["startTimeUnixNano"] = "1735689600000000000"
    payload["resourceSpans"][0]["scopeSpans"].insert(0, unsupported)

    session = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")[0]

    assert len(session.messages) == 6
    conflicts = [event for event in session.session_events if event.event_type == "otel_conflicting_span_id"]
    assert len(conflicts) == 1
    assert conflicts[0].payload["schema_url"] == "https://example.invalid/genai/99.0.0"


def test_canonical_span_precedes_schema_url_for_equal_rank_and_start() -> None:
    payload = _payload()
    canonical_span = copy.deepcopy(_spans(payload)[1])
    canonical_span["traceId"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    canonical_span["spanId"] = "bbbbbbbbbbbbbbbb"
    conversation = next(attr for attr in canonical_span["attributes"] if attr["key"] == "gen_ai.conversation.id")
    conversation["value"] = {"stringValue": "a-canonical"}
    schema_first_span = copy.deepcopy(canonical_span)
    schema_conversation = next(
        attr for attr in schema_first_span["attributes"] if attr["key"] == "gen_ai.conversation.id"
    )
    schema_conversation["value"] = {"stringValue": "z-schema"}
    scope_z = {"schemaUrl": "https://example.invalid/z", "spans": [canonical_span]}
    scope_a = {"schemaUrl": "https://example.invalid/a", "spans": [schema_first_span]}
    payload["resourceSpans"][0]["scopeSpans"].extend((scope_a, scope_z))

    forward = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")
    payload["resourceSpans"][0]["scopeSpans"].reverse()
    reversed_result = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert [session.model_dump(mode="json") for session in forward] == [
        session.model_dump(mode="json") for session in reversed_result
    ]
    # The two copies name different conversations: the coordinate is
    # identity-ambiguous and keys by its trace (Codex P2, #5711).
    assert {session.provider_session_id for session in forward} == {
        "synthetic-agent:trace:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "synthetic-agent:conversation:conversation-demo-7",
    }
    selected = next(session for session in forward if session.provider_session_id.endswith(":trace:" + "a" * 32))
    assert selected.messages == []
    assert [event.event_type for event in selected.session_events] == [
        "otel_span_evidence",
        "otel_conflicting_span_id",
    ]
    assert selected.session_events[0].payload["schema_url"] == "https://example.invalid/z"
    assert selected.session_events[1].payload["schema_url"] == "https://example.invalid/a"


def test_evidence_only_genai_span_remains_an_admitted_session() -> None:
    payload = _payload()
    chat = _spans(payload)[1]
    _spans(payload)[:] = [chat]
    chat["attributes"] = [
        attr
        for attr in chat["attributes"]
        if attr["key"] not in {"gen_ai.input.messages", "gen_ai.output.messages"}
        and not attr["key"].startswith("gen_ai.usage.")
    ]

    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert len(sessions) == 1
    assert sessions[0].messages == []
    assert [event.event_type for event in sessions[0].session_events] == ["otel_span_evidence"]
    assert admit_parsed_sessions_for_publication(sessions, provider=Provider.OTEL_GENAI, source_path=None) == sessions


def _attr(key: str, value: object) -> dict[str, object]:
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    return {"key": key, "value": {"stringValue": value if isinstance(value, str) else json.dumps(value)}}


def _span(trace_id: str, span_id: str, start_ns: int, attributes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "name": "span",
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(start_ns + 1),
        "attributes": attributes,
    }


def _document(*resources: tuple[list[dict[str, object]], list[dict[str, object]]]) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attributes},
                "scopeSpans": [{"schemaUrl": SEMCONV_SCHEMA_URL, "spans": spans}],
            }
            for resource_attributes, spans in resources
        ]
    }


def _chat(trace_id: str, span_id: str, start_ns: int, inputs: list[str], output: str) -> dict[str, object]:
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": text} for index, text in enumerate(inputs)
    ]
    return _span(
        trace_id,
        span_id,
        start_ns,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.input.messages", history),
            _attr("gen_ai.output.messages", [{"role": "assistant", "content": output}]),
        ],
    )


def test_repeated_input_history_is_emitted_once() -> None:
    """Each request's full history adds only its unseen suffix to the transcript.

    Anti-vacuity: emit every input entry per span again and the transcript
    reads ``Q1, A1, Q1, A1, Q2, A2``.
    """
    trace = "1" * 32
    payload = _document(
        (
            [_attr("service.name", "agent")],
            [
                _chat(trace, "a" * 16, 1_000, ["Q1"], "A1"),
                _chat(trace, "b" * 16, 2_000, ["Q1", "A1", "Q2"], "A2"),
            ],
        )
    )

    (session,) = otel_genai.parse(payload, "ignored")

    assert [message.text for message in session.messages] == ["Q1", "A1", "Q2", "A2"]


def test_unnamed_resources_with_different_attributes_stay_distinct() -> None:
    """Resources without ``service.name`` are not merged into one session.

    Anti-vacuity: fall back to one shared resource id and both conversations
    ``chat-1`` group into a single session.
    """
    payload = _document(
        ([_attr("deployment.environment", "staging")], [_chat("2" * 32, "c" * 16, 1_000, ["Q-staging"], "A")]),
        ([_attr("deployment.environment", "prod")], [_chat("3" * 32, "d" * 16, 1_000, ["Q-prod"], "A")]),
    )

    sessions = otel_genai.parse(payload, "ignored")

    assert len(sessions) == 2
    assert len({session.provider_session_id for session in sessions}) == 2


def test_non_genai_trace_does_not_become_a_session() -> None:
    """An ordinary trace beside a GenAI trace in one export is not admitted.

    Anti-vacuity: keep every span and the HTTP trace becomes its own
    zero-message session.
    """
    http_span = _span("5" * 32, "f" * 16, 1_000, [_attr("http.method", "GET")])
    payload = _document(([_attr("service.name", "agent")], [_chat("4" * 32, "e" * 16, 1_000, ["Q"], "A"), http_span]))

    sessions = otel_genai.parse(payload, "ignored")

    assert len(sessions) == 1
    assert [message.text for message in sessions[0].messages] == ["Q", "A"]


def test_usage_survives_for_non_chat_generation_operations() -> None:
    """A usage-only ``text_completion`` span keeps its counters.

    Anti-vacuity: restore the ``== "chat"`` guard and no ``message_usage``
    event is emitted.
    """
    span = _span(
        "6" * 32,
        "1" * 16,
        1_000,
        [
            _attr("gen_ai.operation.name", "text_completion"),
            _attr("gen_ai.request.model", "m"),
            _attr("gen_ai.usage.input_tokens", 11),
            _attr("gen_ai.usage.output_tokens", 7),
        ],
    )
    payload = _document(([_attr("service.name", "agent")], [span]))

    (session,) = otel_genai.parse(payload, "ignored")

    usage = [event.payload for event in session.session_events if event.event_type == "message_usage"]
    assert usage == [{"last_token_usage": {"input_tokens": 11, "output_tokens": 7}, "model": "m"}]


def test_ordinary_ancestor_joins_its_genai_child_conversation() -> None:
    """An HTTP root span over a GenAI child stays in the child's conversation.

    Anti-vacuity: group spans only by their own or an ancestor's
    conversation id and the root becomes a second, zero-message
    ``trace:<id>`` session.
    """
    trace = "7" * 32
    root = _span(trace, "2" * 16, 500, [_attr("http.method", "POST")])
    child = {**_chat(trace, "3" * 16, 1_000, ["Q"], "A"), "parentSpanId": "2" * 16}
    payload = _document(([_attr("service.name", "agent")], [root, child]))

    (session,) = otel_genai.parse(payload, "ignored")

    assert session.provider_session_id == "agent:conversation:chat-1"
    span_ids = [
        event.payload["span_id"] for event in session.session_events if event.event_type == "otel_span_evidence"
    ]
    assert span_ids == ["2" * 16, "3" * 16]


def test_tool_exchange_replayed_in_next_request_is_emitted_once() -> None:
    """A tool span's call and result count as history for the next chat span.

    Anti-vacuity: leave tool spans out of the transcript and the second chat
    span re-emits the replayed tool call and result as input messages.
    """
    trace = "8" * 32
    call = {"role": "assistant", "parts": [{"type": "tool_call", "id": "call-1", "name": "search", "arguments": {}}]}
    result = {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call-1", "response": "found"}]}
    first = _span(
        trace,
        "4" * 16,
        1_000,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.input.messages", [{"role": "user", "content": "Q"}]),
            _attr("gen_ai.output.messages", [call]),
        ],
    )
    tool = _span(
        trace,
        "5" * 16,
        2_000,
        [
            _attr("gen_ai.operation.name", "execute_tool"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.tool.name", "search"),
            _attr("gen_ai.tool.call.id", "call-1"),
            _attr("gen_ai.tool.call.result", "found"),
        ],
    )
    second = _span(
        trace,
        "6" * 16,
        3_000,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.input.messages", [{"role": "user", "content": "Q"}, call, result]),
            _attr("gen_ai.output.messages", [{"role": "assistant", "content": "A"}]),
        ],
    )
    payload = _document(([_attr("service.name", "agent")], [first, tool, second]))

    (session,) = otel_genai.parse(payload, "ignored")

    second_span_inputs = [
        message.provider_message_id
        for message in session.messages
        if message.provider_message_id.startswith(f"{trace}:{'6' * 16}:input")
    ]
    assert second_span_inputs == []
    assert [message.text for message in session.messages if message.text] == ["Q", "A"]


def test_truncated_history_around_a_pinned_system_prompt_is_emitted_once() -> None:
    """A retained system prompt must not defeat the suffix history match.

    Anti-vacuity (Codex P2, #5711): transcript ``[S, Q1, A1, Q2, A2]``
    truncates to a third request's inputs ``[S, Q2, A2, Q3]`` -- the
    retained system prompt interrupts what would otherwise be a clean
    suffix match, since no leading slice of the new inputs equals a
    trailing slice of the transcript with ``S`` sitting at index 0. Before
    matching the pinned prefix separately, this re-emitted ``S``, ``Q2``
    and ``A2`` as duplicates; only ``Q3`` is genuinely new.
    """
    trace = "9" * 32
    system = {"role": "system", "content": "S"}
    first = _span(
        trace,
        "1" * 16,
        1_000,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.input.messages", [system, {"role": "user", "content": "Q1"}]),
            _attr("gen_ai.output.messages", [{"role": "assistant", "content": "A1"}]),
        ],
    )
    second = _span(
        trace,
        "2" * 16,
        2_000,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr(
                "gen_ai.input.messages",
                [
                    system,
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"},
                ],
            ),
            _attr("gen_ai.output.messages", [{"role": "assistant", "content": "A2"}]),
        ],
    )
    # Context truncation drops Q1/A1 but keeps the pinned system prompt.
    third = _span(
        trace,
        "3" * 16,
        3_000,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr(
                "gen_ai.input.messages",
                [
                    system,
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                    {"role": "user", "content": "Q3"},
                ],
            ),
            _attr("gen_ai.output.messages", [{"role": "assistant", "content": "A3"}]),
        ],
    )
    payload = _document(([_attr("service.name", "agent")], [first, second, third]))

    (session,) = otel_genai.parse(payload, "ignored")

    assert [message.text for message in session.messages] == ["S", "Q1", "A1", "Q2", "A2", "Q3", "A3"]


def test_usage_only_span_attributes_usage_to_response_model() -> None:
    """A usage-only span without a request model uses its response model.

    Anti-vacuity: read only ``gen_ai.request.model`` and the usage event
    carries ``model=None`` with ``models_used`` empty.
    """
    span = _span(
        "9" * 32,
        "7" * 16,
        1_000,
        [
            _attr("gen_ai.operation.name", "generate_content"),
            _attr("gen_ai.response.model", "served-model"),
            _attr("gen_ai.usage.input_tokens", 3),
        ],
    )
    payload = _document(([_attr("service.name", "agent")], [span]))

    (session,) = otel_genai.parse(payload, "ignored")

    usage = [event.payload for event in session.session_events if event.event_type == "message_usage"]
    assert usage == [{"last_token_usage": {"input_tokens": 3}, "model": "served-model"}]
    assert session.models_used == ["served-model"]


def test_repeated_input_only_turn_is_a_second_occurrence() -> None:
    """Two input-only spans carrying the same message both emit it.

    Anti-vacuity: let the overlap cover the whole input and the second
    ``continue`` is swallowed as replayed history.
    """
    trace = "a" * 32

    def input_only(span_id: str, start_ns: int) -> dict[str, object]:
        return _span(
            trace,
            span_id,
            start_ns,
            [
                _attr("gen_ai.operation.name", "chat"),
                _attr("gen_ai.conversation.id", "chat-1"),
                _attr("gen_ai.input.messages", [{"role": "user", "content": "continue"}]),
            ],
        )

    payload = _document(([_attr("service.name", "agent")], [input_only("8" * 16, 1_000), input_only("9" * 16, 2_000)]))

    (session,) = otel_genai.parse(payload, "ignored")

    assert [message.text for message in session.messages] == ["continue", "continue"]


def test_genai_trace_found_only_in_a_conflicting_copy_is_kept() -> None:
    """A trace whose GenAI attributes live only in a conflict variant stays.

    Anti-vacuity: decide trace membership from the selected copies alone and
    the document yields no session and loses the conflict evidence.
    """
    trace = "b" * 32
    plain = _span(trace, "a" * 16, 1_000, [])
    genai = _span(trace, "a" * 16, 1_000, [_attr("gen_ai.operation.name", "chat")])
    payload = _document(([_attr("service.name", "agent")], [plain, genai]))

    sessions = otel_genai.parse(payload, "ignored")

    assert len(sessions) == 1
    assert "otel_conflicting_span_id" in [event.event_type for event in sessions[0].session_events]


def test_tool_call_beside_text_is_recognised_in_replayed_history() -> None:
    """An output message with text and a tool call is matched by its call id.

    Anti-vacuity: require the recorded call entry to carry no text and the
    tool span records a second call, so the replayed history is emitted again.
    """
    trace = "c" * 32
    call = {
        "role": "assistant",
        "parts": [
            {"type": "text", "content": "Searching."},
            {"type": "tool_call", "id": "call-2", "name": "search", "arguments": {}},
        ],
    }
    result = {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call-2", "response": "found"}]}
    first = _span(
        trace,
        "1" * 16,
        1_000,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.input.messages", [{"role": "user", "content": "Q"}]),
            _attr("gen_ai.output.messages", [call]),
        ],
    )
    tool = _span(
        trace,
        "2" * 16,
        2_000,
        [
            _attr("gen_ai.operation.name", "execute_tool"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.tool.name", "search"),
            _attr("gen_ai.tool.call.id", "call-2"),
        ],
    )
    second = _span(
        trace,
        "3" * 16,
        3_000,
        [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.conversation.id", "chat-1"),
            _attr("gen_ai.input.messages", [{"role": "user", "content": "Q"}, call, result]),
            _attr("gen_ai.output.messages", [{"role": "assistant", "content": "A"}]),
        ],
    )
    payload = _document(([_attr("service.name", "agent")], [first, tool, second]))

    (session,) = otel_genai.parse(payload, "ignored")

    assert not [message for message in session.messages if f"{trace}:{'3' * 16}:input" in message.provider_message_id]


def test_unnamed_resource_identity_survives_an_application_upgrade() -> None:
    """An upgraded application exports the same conversation under the same id.

    Anti-vacuity (Codex P2, #5711): hash ``service.version`` into the
    fallback resource identity and the post-upgrade export imports as a
    second session. A changed non-version attribute still separates the
    unnamed resources of one document.
    """

    def resource(version: str, environment: str = "prod") -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return (
            [_attr("deployment.environment", environment), _attr("service.version", version)],
            [_chat("e" * 32, "5" * 16, 1_000, ["Q"], "A")],
        )

    def export(*resources: tuple[list[dict[str, object]], list[dict[str, object]]]) -> list[str]:
        return sorted(session.provider_session_id for session in otel_genai.parse(_document(*resources), "ignored"))

    assert export(resource("1.0.0"), resource("1.0.0", "staging")) == export(
        resource("1.1.0"), resource("1.1.0", "staging")
    )
    assert len(export(resource("1.0.0"), resource("1.0.0", "staging"))) == 2


def test_unnamed_resource_identity_ignores_instance_attributes() -> None:
    """A restarted process exports the same conversation under the same id.

    Anti-vacuity: hash every resource attribute and the changed
    ``process.pid`` gives the second export a different session id.
    """

    def export(pid: int) -> str:
        payload = _document(
            (
                [_attr("deployment.environment", "prod"), _attr("process.pid", pid)],
                [_chat("d" * 32, "4" * 16, 1_000, ["Q"], "A")],
            )
        )
        (session,) = otel_genai.parse(payload, "ignored")
        return session.provider_session_id

    assert export(100) == export(200)


def test_every_session_of_a_multi_conversation_document_is_admitted() -> None:
    """Each session drawn from one OTLP document carries an admission proof.

    Anti-vacuity (Codex P2, #5711): exempt multi-session results from the
    admission boundary and both sessions reach the writer with
    ``unit_accounting=None``, skipping its conservation check.
    """
    payload = _document(
        ([_attr("service.name", "alpha")], [_chat("a" * 32, "1" * 16, 1_000, ["Q1"], "A1")]),
        ([_attr("service.name", "beta")], [_chat("b" * 32, "2" * 16, 2_000, ["Q2"], "A2")]),
    )
    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert len(sessions) == 2
    assert all(session.unit_accounting is not None for session in sessions)


def test_an_unnamed_resource_keeps_its_id_when_a_second_resource_joins() -> None:
    """A later export that adds a second unnamed resource keeps the first's session.

    Anti-vacuity (Codex P2, #5711): key the lone unnamed resource as bare
    ``resource`` and hash it only once a second one appears, and the
    two-resource export names the original conversation differently,
    importing it as a second session.
    """
    original = ([_attr("deployment.environment", "prod")], [_chat("a" * 32, "1" * 16, 1_000, ["Q"], "A")])
    added = ([_attr("deployment.environment", "staging")], [_chat("b" * 32, "2" * 16, 2_000, ["Q2"], "A2")])

    (alone,) = otel_genai.parse(_document(original), "ignored")
    together = otel_genai.parse(_document(original, added), "ignored")

    assert alone.provider_session_id in {session.provider_session_id for session in together}
    assert len({session.provider_session_id for session in together}) == 2


def test_tool_arguments_in_plain_json_attributes_are_not_wire_types() -> None:
    """A tool argument ``{"type": "unknown"}`` does not mark the document unknown.

    Anti-vacuity (Codex P2, #5711): admit OTel documents with the recursive
    default scanner and the argument inside the plain-JSON attribute map
    becomes an ``otel_genai_unknown_input`` event.
    """
    chat = _chat("c" * 32, "3" * 16, 1_000, ["Q"], "A")
    tool: dict[str, object] = {
        "traceId": "c" * 32,
        "spanId": "4" * 16,
        "name": "execute_tool",
        "startTimeUnixNano": "2000",
        "endTimeUnixNano": "2001",
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "search",
            "gen_ai.tool.call.id": "call-1",
            "gen_ai.tool.call.arguments": {"type": "unknown"},
        },
    }
    payload = _document(([_attr("service.name", "agent")], [chat, tool]))

    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert sessions
    for session in sessions:
        assert not [event for event in session.session_events if event.event_type.endswith("_unknown_input")]


def test_a_multi_conversation_document_is_scanned_once(monkeypatch: Any) -> None:
    """Every conversation of one document shares a single admission scan.

    Anti-vacuity (Codex P2, #5711): build one observer per emitted session
    and the document is scanned once per conversation (O(N^2) work).
    """
    from polylogue.sources.parsers import base_support

    calls: list[object] = []
    scan = base_support._ADMISSION_SCANS["opentelemetry"]

    def counting(value: object) -> str | None:
        calls.append(value)
        return scan(value)

    monkeypatch.setitem(base_support._ADMISSION_SCANS, "opentelemetry", counting)
    payload = _document(
        *(
            ([_attr("service.name", f"svc-{index}")], [_chat(f"{index:032x}", "1" * 16, 1_000, ["Q"], "A")])
            for index in range(1, 5)
        )
    )

    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-file-stem")

    assert len(sessions) == 4
    assert all(session.unit_accounting is not None for session in sessions)
    assert len(calls) == 1


def test_a_conversation_id_in_a_conflicting_copy_names_the_session() -> None:
    """A conversation id surviving only in a conflicting copy still keys the session.

    Anti-vacuity (Codex P2, #5711): read the conversation from the selected
    ordinary copy alone and the session is keyed by trace, so a later clean
    export of the GenAI copy imports a second session.
    """
    trace = "d" * 32
    plain = _span(trace, "a" * 16, 1_000, [])
    genai = _span(
        trace, "a" * 16, 1_000, [_attr("gen_ai.operation.name", "chat"), _attr("gen_ai.conversation.id", "chat-9")]
    )
    conflicted = otel_genai.parse(_document(([_attr("service.name", "agent")], [plain, genai])), "ignored")
    clean = otel_genai.parse(_document(([_attr("service.name", "agent")], [genai])), "ignored")

    assert [session.provider_session_id for session in conflicted] == [session.provider_session_id for session in clean]


def test_copies_naming_different_conversations_choose_neither() -> None:
    """A coordinate whose copies disagree on the conversation has no conversation identity.

    Anti-vacuity (Codex P2, #5711): pick the first sorted conversation and a
    later clean export of the other copy keys a different session.
    """
    trace = "e" * 32

    def copy(conversation: str) -> dict[str, object]:
        return _span(
            trace,
            "b" * 16,
            1_000,
            [_attr("gen_ai.operation.name", "chat"), _attr("gen_ai.conversation.id", conversation)],
        )

    sessions = otel_genai.parse(_document(([_attr("service.name", "agent")], [copy("chat-a"), copy("chat-b")])), "x")

    assert [session.provider_session_id for session in sessions] == [f"agent:trace:{trace}"]


def test_a_shared_root_stays_with_its_first_conversation_when_a_sibling_joins() -> None:
    """An ancestor shared by conversations keeps its owner as the export grows.

    Anti-vacuity (Codex P2, #5711): assign the ancestor to the lowest
    conversation id and appending conversation ``a`` moves the root out of
    ``z``, turning append-only growth into a conflict.
    """
    trace = "f" * 32
    root = _span(trace, "0" * 16, 1_000, [])

    def child(span_id: str, start: int, conversation: str) -> dict[str, object]:
        span = _span(
            trace,
            span_id,
            start,
            [_attr("gen_ai.operation.name", "chat"), _attr("gen_ai.conversation.id", conversation)],
        )
        span["parentSpanId"] = "0" * 16
        return span

    def root_owner(spans: list[dict[str, object]]) -> str:
        sessions = otel_genai.parse(_document(([_attr("service.name", "agent")], spans)), "x")
        return next(
            session.provider_session_id
            for session in sessions
            for event in session.session_events
            if event.event_type == "otel_span_evidence" and event.payload.get("span_id") == "0" * 16
        )

    before = root_owner([root, child("1" * 16, 2_000, "z")])
    after = root_owner([root, child("1" * 16, 2_000, "z"), child("2" * 16, 3_000, "a")])

    assert before == after == "agent:conversation:z"


def test_the_history_overlap_is_linear_in_the_history() -> None:
    """The prefix/suffix overlap search makes a linear number of comparisons.

    Anti-vacuity (Codex P2, #5711): try every overlap length with fresh slices
    and a 2,000-entry history with no overlap costs about two million
    comparisons.
    """
    import random

    compared = {"count": 0}

    class Entry(tuple[str, str, bool]):
        def __eq__(self, other: object) -> bool:
            compared["count"] += 1
            return tuple.__eq__(self, other)

        def __ne__(self, other: object) -> bool:
            return not self.__eq__(other)

        __hash__ = tuple.__hash__

    history = [Entry(("user", f"m{index}", False)) for index in range(2_000)]
    retained = [Entry(("user", f"r{index}", False)) for index in range(2_000)]
    assert otel_genai._prefix_suffix_overlap(retained, history) == 0  # type: ignore[arg-type]
    assert compared["count"] < 10 * 4_000

    rng = random.Random(7)
    for _ in range(200):
        text = [rng.choice("ab") for _ in range(rng.randint(0, 12))]
        pattern = [rng.choice("ab") for _ in range(rng.randint(0, 12))]
        brute = max(
            (size for size in range(min(len(text), len(pattern)) + 1) if pattern[:size] == text[len(text) - size :]),
            default=0,
        )
        assert otel_genai._prefix_suffix_overlap(pattern, text) == brute  # type: ignore[arg-type]


def test_a_cross_resource_root_is_kept_with_its_trace() -> None:
    """A trace's HTTP root under another resource stays as evidence of the conversation.

    Anti-vacuity (Codex P1, #5711): decide trace membership per resource and
    the frontend root is dropped before grouping.
    """
    trace = "a" * 32
    frontend_root = _span(trace, "0" * 16, 1_000, [_attr("http.request.method", "POST")])
    genai = _span(
        trace, "1" * 16, 2_000, [_attr("gen_ai.operation.name", "chat"), _attr("gen_ai.conversation.id", "chat-x")]
    )
    genai["parentSpanId"] = "0" * 16

    sessions = otel_genai.parse(
        _document(([_attr("service.name", "frontend")], [frontend_root]), ([_attr("service.name", "agent")], [genai])),
        "x",
    )

    (session,) = sessions
    assert session.provider_session_id == "agent:conversation:chat-x"
    root_evidence = [
        event.payload
        for event in session.session_events
        if event.event_type == "otel_span_evidence" and event.payload.get("span_id") == "0" * 16
    ]
    assert root_evidence and root_evidence[0]["resource_id"] == "frontend"


def test_dispatch_preserves_structured_tool_results() -> None:
    """A kvlist, array, numeric or boolean tool result reaches the result block.

    Anti-vacuity: keep only string results and every case here has text=None.
    """
    # OTLP JSON carries int64 as a string or a number; the decoded value is
    # what the result block must serialize.
    cases: list[tuple[object, dict[str, object]]] = [
        (
            {"found": ["1", False]},
            {
                "kvlistValue": {
                    "values": [
                        {
                            "key": "found",
                            "value": {"arrayValue": {"values": [{"intValue": "1"}, {"boolValue": False}]}},
                        }
                    ]
                }
            },
        ),
        (["item", 2], {"arrayValue": {"values": [{"stringValue": "item"}, {"intValue": 2}]}}),
        (0, {"intValue": 0}),
        (1.5, {"doubleValue": 1.5}),
        (False, {"boolValue": False}),
    ]
    for expected, value in cases:
        payload = _payload()
        span = next(span for span in _spans(payload) if span["name"].startswith("execute_tool"))
        span["attributes"] = [
            attribute for attribute in span["attributes"] if attribute["key"] != "gen_ai.tool.call.result"
        ]
        span["attributes"].append({"key": "gen_ai.tool.call.result", "value": value})
        session = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
        message = next(
            message
            for message in session.messages
            if message.provider_message_id.endswith(f":{span['spanId']}:tool-result")
        )
        assert message.blocks[0].text is not None
        assert json.loads(message.blocks[0].text) == expected


def test_dispatch_retains_unprojected_span_fields_in_evidence() -> None:
    """End time, links, trace state, flags, dropped counts and unknown keys survive.

    Anti-vacuity: build the evidence payload from the projected keys alone and
    none of these fields is recoverable from ``otel_span_evidence``.
    """
    payload = _payload()
    span = _spans(payload)[0]
    extra = {
        "endTimeUnixNano": "1700000001000000000",
        "traceState": "neutral=fixture",
        "flags": 1,
        "droppedAttributesCount": 2,
        "droppedEventsCount": 3,
        "droppedLinksCount": 4,
        "links": [{"traceId": "1" * 32, "spanId": "2" * 16, "flags": 1}],
        "vendorEnvelopeField": {"neutral": True},
    }
    span.update(extra)

    session = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
    event = next(
        event
        for event in session.session_events
        if event.event_type == "otel_span_evidence" and event.payload["span_id"] == span["spanId"]
    )
    unprojected = cast(dict[str, object], event.payload["unprojected_span_fields"])
    assert {key: unprojected[key] for key in extra} == extra
    assert "attributes" not in unprojected


def _conversation_resource(
    service: str, conversation: str, trace_id: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    span = _chat(trace_id, "1" * 16, 1_000, ["Q"], f"A from {service}")
    span["attributes"] = [
        _attr("gen_ai.conversation.id", conversation) if attribute["key"] == "gen_ai.conversation.id" else attribute
        for attribute in cast(list[dict[str, object]], span["attributes"])
    ]
    return ([_attr("service.name", service)], [span])


def test_session_identity_components_cannot_run_into_each_other(tmp_path: Path) -> None:
    """A ``:`` inside a service name or conversation id is not a separator.

    Anti-vacuity: join the raw components with ``:`` and service
    ``svc:conversation`` / conversation ``x`` and service ``svc`` /
    conversation ``conversation:x`` both become
    ``svc:conversation:conversation:x`` and are written as one session.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    sessions = parse_payload(
        Provider.OTEL_GENAI,
        _document(
            _conversation_resource("svc:conversation", "x", "a" * 32),
            _conversation_resource("svc", "conversation:x", "b" * 32),
        ),
        "ignored",
    )

    assert sorted(session.provider_session_id for session in sessions) == [
        "svc%3Aconversation:conversation:x",
        "svc:conversation:conversation%3Ax",
    ]
    with ArchiveStore(tmp_path / "archive") as archive:
        stored = {write_index_session(archive, session) for session in sessions}
    assert len(stored) == 2


def test_session_identity_escapes_the_escape_character() -> None:
    """``%`` is escaped too, so a literal ``%3A`` never reads as an escaped ``:``."""
    (literal,) = otel_genai.parse(_document(_conversation_resource("svc%3Ax", "c", "c" * 32)), "ignored")
    (escaped,) = otel_genai.parse(_document(_conversation_resource("svc:x", "c", "d" * 32)), "ignored")

    assert literal.provider_session_id == "svc%253Ax:conversation:c"
    assert escaped.provider_session_id == "svc%3Ax:conversation:c"
