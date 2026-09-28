"""Synthetic OTLP-JSON GenAI parser and dispatch contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

from polylogue.core.enums import Origin, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import detect_provider, parse_payload, require_positive_conversational_evidence
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
    assert (
        require_positive_conversational_evidence(sessions, provider=Provider.OTEL_GENAI, source_path=None) == sessions
    )


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
    assert {session.provider_session_id for session in forward} == {
        "synthetic-agent:conversation:a-canonical",
        "synthetic-agent:conversation:conversation-demo-7",
    }
    selected = next(session for session in forward if session.provider_session_id.endswith(":a-canonical"))
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
    assert (
        require_positive_conversational_evidence(sessions, provider=Provider.OTEL_GENAI, source_path=None) == sessions
    )


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
                "scopeSpans": [{"schemaUrl": otel_genai.SEMCONV_SCHEMA_URL, "spans": spans}],
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
