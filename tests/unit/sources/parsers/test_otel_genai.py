"""Synthetic OTLP-JSON GenAI parser and dispatch contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from polylogue.core.enums import Origin, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers import otel_genai

FIXTURE = Path(__file__).parents[3] / "fixtures" / "otel-genai" / "trace.json"


def _payload() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _spans(payload: dict[str, object]) -> list[dict[str, object]]:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]  # type: ignore[index,return-value]


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
    unsupported_scope = copy.deepcopy(payload["resourceSpans"][0]["scopeSpans"][0])  # type: ignore[index]
    unsupported_scope["schemaUrl"] = "https://example.invalid/genai/99.0.0"
    unsupported_span = unsupported_scope["spans"][1]
    unsupported_span["traceId"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    unsupported_span["spanId"] = "bbbbbbbbbbbbbbbb"
    payload["resourceSpans"][0]["scopeSpans"].append(unsupported_scope)  # type: ignore[index]

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
