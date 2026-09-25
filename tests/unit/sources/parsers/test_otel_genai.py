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
