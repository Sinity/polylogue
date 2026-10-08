"""OTLP-JSON GenAI parser contracts: acquisition, identity, fidelity, schema and lineage states."""

from __future__ import annotations

import asyncio
import copy
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, cast

from polylogue.core.enums import Origin, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.origin_specs import SEMCONV_SCHEMA_URL, origin_specs
from polylogue.sources.parsers import otel_genai
from polylogue.sources.source_layout import export_drop_layout

FIXTURE = Path(__file__).parents[3] / "fixtures" / "otel-genai" / "trace.json"


def test_configured_file_root_acquires_and_archives_otel_trace(tmp_path: Path) -> None:
    """The configured OTel root must reach the ordinary live archive route."""
    from polylogue.sources.live import WatchSource
    from tests.infra.live_batch import prepared_live_batch_processor

    async def ingest() -> tuple[Path, Path, Path, Any]:
        source_root = tmp_path / "configured-otel-root"
        source_root.mkdir()
        source_path = source_root / "trace.json"
        shutil.copyfile(FIXTURE, source_path)

        archive_root = tmp_path / "archive"
        # The live processor publishes through the daemon's retained Raw owner.
        async with prepared_live_batch_processor(
            archive_root,
            (WatchSource(name="otel-genai", root=source_root, layout=export_drop_layout((".json",))),),
            parser_fingerprint="test-otel-parser",
        ) as processor:
            result = await processor.ingest_files([source_path], emit_event=False)
        return source_path, archive_root / "source.db", archive_root / "index.db", result

    source_path, source_db, index_db, result = asyncio.run(ingest())

    assert result.full_file_count == result.succeeded_file_count == 1
    assert result.ingested_session_count == 1
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT source_path, origin FROM raw_sessions").fetchone() == (
            str(source_path),
            "otel-genai",
        )
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT origin, native_id, message_count FROM sessions").fetchone() == (
            "otel-genai",
            "synthetic-agent:conversation:conversation-demo-7",
            6,
        )
        assert conn.execute(
            "SELECT b.text FROM messages AS m JOIN blocks AS b USING (message_id) "
            "WHERE m.role = 'user' ORDER BY m.position LIMIT 1"
        ).fetchone() == ("Find tomorrow's weather.",)


def _payload() -> Any:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _spans(payload: Any) -> list[Any]:
    return cast(list[Any], payload["resourceSpans"][0]["scopeSpans"][0]["spans"])


def test_otlp_json_detects_messages_tool_blocks_usage_and_unknown_status() -> None:
    payload = _payload()
    assert detect_provider(payload) is Provider.OTEL_GENAI
    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored-filename-stem")
    replay = parse_payload(Provider.OTEL_GENAI, payload, "another-filename-stem")
    assert len(sessions) == 1
    session = sessions[0]
    assert session.model_dump(mode="json") == replay[0].model_dump(mode="json")
    assert Provider.from_string("otel-genai") is Provider.OTEL_GENAI
    assert canonical_acquisition_provider(None, source_name="otel-genai") == "opentelemetry"
    assert session.provider_session_id == "synthetic-agent:conversation:conversation-demo-7"
    assert [message.text for message in session.messages[:2]] == [
        "Find tomorrow's weather.",
        "I'll check the forecast.",
    ]
    assert [message.provider_message_id for message in session.messages[:2]] == [
        "4bf92f3577b34da6a3ce929d0e0e4736:b7ad6b7169203331:input:0",
        "4bf92f3577b34da6a3ce929d0e0e4736:b7ad6b7169203331:output:0",
    ]
    assistant = session.messages[1]
    assert (assistant.input_tokens, assistant.output_tokens, assistant.cache_read_tokens) == (18, 6, 0)
    blocks = [block for message in session.messages for block in message.blocks]
    assert [block.type.value for block in blocks] == ["tool_use", "tool_result", "tool_use", "tool_result"]
    assert blocks[1].tool_outcome is ToolOutcome.UNKNOWN
    assert blocks[1].outcome_unknown_reason == ToolResultUnknownReason.NOT_REPORTED.value
    assert blocks[3].tool_outcome is ToolOutcome.ERROR and blocks[3].is_error is True
    assert any(
        message.provider_message_id == "4bf92f3577b34da6a3ce929d0e0e4736:00f067aa0ba902b7:tool-use"
        and message.parent_message_provider_id == "4bf92f3577b34da6a3ce929d0e0e4736:b7ad6b7169203331:output:0"
        for message in session.messages
    )
    assert origin_from_provider(session.source_name) is Origin.OTEL_GENAI
    evidence = [event.payload for event in session.session_events if event.event_type == "otel_span_evidence"]
    tool = next(row for row in evidence if row["span_id"] == "00f067aa0ba902b7")
    assert tool["parent_span_id"] == "b7ad6b7169203331"
    assert cast(Any, tool["attributes"])["vendor.experimental.signal"] == "retained-verbatim"


def test_duplicate_export_deduplicates_and_conflicting_copy_is_order_independent() -> None:
    payload = _payload()
    model = next(span for span in _spans(payload) if span["spanId"] == "b7ad6b7169203331")
    spans = _spans(payload)
    spans.append(copy.deepcopy(model))
    parsed = parse_payload(Provider.OTEL_GENAI, payload, "file-a")[0]
    prefix = "4bf92f3577b34da6a3ce929d0e0e4736:b7ad6b7169203331:"
    assert sum(m.provider_message_id.startswith(prefix) for m in parsed.messages) == 2
    conflict = copy.deepcopy(model)
    conflict["attributes"].append({"key": "vendor.conflict.marker", "value": {"stringValue": "alternate-fragment"}})
    spans.append(conflict)
    parsed = parse_payload(Provider.OTEL_GENAI, payload, "file-a")[0]
    reversed_payload = copy.deepcopy(payload)
    _spans(reversed_payload).reverse()
    reversed_parsed = parse_payload(Provider.OTEL_GENAI, reversed_payload, "file-a")[0]
    assert [m.model_dump(mode="json") for m in parsed.messages] == [
        m.model_dump(mode="json") for m in reversed_parsed.messages
    ]
    conflict_event = next(e for e in parsed.session_events if e.event_type == "otel_conflicting_span_id")
    assert conflict_event.payload["span_id"] == "b7ad6b7169203331"
    assert conflict_event.payload["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert conflict_event.payload["conflicting_span"]


def test_same_span_id_in_distinct_traces_within_a_conversation_is_preserved() -> None:
    payload = _payload()
    original = next(span for span in _spans(payload) if span["spanId"] == "b7ad6b7169203331")
    second_trace = copy.deepcopy(original)
    second_trace["traceId"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    second_trace["startTimeUnixNano"] = "1735689605000000000"
    second_trace["attributes"] = [
        attribute for attribute in second_trace["attributes"] if attribute["key"] != "gen_ai.output.messages"
    ]
    second_trace["attributes"].append(
        {
            "key": "gen_ai.output.messages",
            "value": {"stringValue": json.dumps([{"role": "assistant", "content": "second trace turn"}])},
        }
    )
    _spans(payload).append(second_trace)

    sessions = parse_payload(Provider.OTEL_GENAI, payload, "ignored")
    assert len(sessions) == 1
    session = sessions[0]
    assistant_messages = [message for message in session.messages if message.role == "assistant"]
    assert {message.text for message in assistant_messages} >= {"I'll check the forecast.", "second trace turn"}
    assert {
        message.provider_message_id
        for message in assistant_messages
        if message.text in {"I'll check the forecast.", "second trace turn"}
    } == {
        "4bf92f3577b34da6a3ce929d0e0e4736:b7ad6b7169203331:output:0",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:b7ad6b7169203331:output:0",
    }
    assert not any(event.event_type == "otel_conflicting_span_id" for event in session.session_events)


def test_trace_fallback_and_empty_message_state_are_explicit() -> None:
    payload = _payload()
    # A trace falls back to its own session only when no span in it names a
    # conversation: a conversation-less root is otherwise adopted by the
    # conversation beneath it.
    for traced in _spans(payload):
        traced["attributes"] = [a for a in traced["attributes"] if a["key"] != "gen_ai.conversation.id"]
    span = _spans(payload)[1]
    span["attributes"].append({"key": "gen_ai.input.messages", "value": {"arrayValue": {"values": []}}})
    session = next(
        s
        for s in parse_payload(Provider.OTEL_GENAI, payload, "ignored")
        if s.provider_session_id.endswith(":trace:4bf92f3577b34da6a3ce929d0e0e4736")
    )
    event = next(
        e
        for e in session.session_events
        if e.event_type == "otel_span_evidence" and e.payload["span_id"] == "b7ad6b7169203331"
    )
    assert event.payload["message_fidelity"] == {"gen_ai.input.messages": "empty", "gen_ai.output.messages": "present"}
    assert event.payload["schema_url"] == SEMCONV_SCHEMA_URL
    assert event.payload["schema_url_status"] == "supported"
    spec = next(spec for spec in origin_specs() if spec.origin is Origin.OTEL_GENAI)
    assert spec.acquisition_modes == ("otlp-json-file",)
    assert spec.detector_tightness == 95
    assert spec.parser_fingerprint()
    assert spec.completeness_modes[0].capture_mode == "otlp-json-file"


def test_explicit_empty_attribute_is_not_replaced_by_compatibility_events() -> None:
    payload = _payload()
    span = next(span for span in _spans(payload) if span["spanId"] == "b7ad6b7169203331")
    span["attributes"] = [
        attr for attr in span["attributes"] if attr["key"] not in {"gen_ai.input.messages", "gen_ai.output.messages"}
    ]
    span["attributes"].append({"key": "gen_ai.input.messages", "value": {"arrayValue": {"values": []}}})
    span["events"] = [
        {
            "name": "gen_ai.prompt",
            "timeUnixNano": "1735689601000000000",
            "attributes": [{"key": "content", "value": {"stringValue": "legacy prompt"}}],
        }
    ]

    parsed = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
    assert not any(message.text == "legacy prompt" for message in parsed.messages)


def test_late_parent_is_retained_then_resolved_on_a_later_file_revision() -> None:
    payload = _payload()
    spans = _spans(payload)
    parent = next(span for span in spans if span["spanId"] == "b7ad6b7169203331")
    spans.remove(parent)
    partial = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
    child_evidence = next(
        event
        for event in partial.session_events
        if event.event_type == "otel_span_evidence" and event.payload["span_id"] == "00f067aa0ba902b7"
    )
    assert child_evidence.payload["parent_span_id"] == "b7ad6b7169203331"
    assert not any(
        message.provider_message_id.startswith("4bf92f3577b34da6a3ce929d0e0e4736:b7ad6b7169203331:")
        for message in partial.messages
    )

    spans.append(parent)
    complete = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
    tool_use = next(
        message
        for message in complete.messages
        if message.provider_message_id == "4bf92f3577b34da6a3ce929d0e0e4736:00f067aa0ba902b7:tool-use"
    )
    assert tool_use.parent_message_provider_id == "4bf92f3577b34da6a3ce929d0e0e4736:b7ad6b7169203331:output:0"


def test_scope_schema_url_is_preserved_and_unsupported_scope_is_not_normalized() -> None:
    payload = _payload()
    resource_spans = payload["resourceSpans"][0]
    scope = resource_spans["scopeSpans"][0]
    del scope["schemaUrl"]

    without_url = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
    event = next(event for event in without_url.session_events if event.event_type == "otel_span_evidence")
    assert event.payload["schema_url"] is None
    assert event.payload["schema_url_status"] == "missing"

    payload = _payload()
    resource_spans = payload["resourceSpans"][0]
    scope = resource_spans["scopeSpans"][0]
    unsupported_scope = copy.deepcopy(scope)
    unsupported_scope["schemaUrl"] = "https://example.invalid/schemas/gen-ai/99.0.0"
    unsupported_span = unsupported_scope["spans"][1]
    unsupported_span["traceId"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    unsupported_span["spanId"] = "bbbbbbbbbbbbbbbb"
    resource_spans["scopeSpans"].append(unsupported_scope)

    assert otel_genai.looks_like(payload)
    session = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
    event = next(
        event
        for event in session.session_events
        if event.event_type == "otel_span_evidence" and event.payload["span_id"] == "bbbbbbbbbbbbbbbb"
    )
    assert event.payload["schema_url"] == "https://example.invalid/schemas/gen-ai/99.0.0"
    assert event.payload["schema_url_status"] == "unsupported"
    assert not any(
        message.provider_message_id.startswith("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:") for message in session.messages
    )


def test_redacted_truncated_and_missing_usage_states_are_not_collapsed() -> None:
    payload = _payload()
    spans = _spans(payload)
    missing_usage_tool = spans[0]
    missing_usage_tool["attributes"].append({"key": "polylogue.evidence.truncated", "value": {"boolValue": True}})
    model = next(span for span in spans if span["spanId"] == "b7ad6b7169203331")
    model["attributes"].extend(
        [
            {"key": "polylogue.evidence.redacted", "value": {"boolValue": True}},
            {"key": "polylogue.usage.truncated", "value": {"boolValue": True}},
        ]
    )
    session = parse_payload(Provider.OTEL_GENAI, payload, "ignored")[0]
    evidence: dict[Any, Any] = {
        event.payload["span_id"]: event.payload
        for event in session.session_events
        if event.event_type == "otel_span_evidence"
    }
    assert evidence["b7ad6b7169203331"]["message_fidelity"]["gen_ai.input.messages"] == "redacted"
    assert evidence["b7ad6b7169203331"]["usage_fidelity"] == "truncated"
    assert evidence["00f067aa0ba902b7"]["usage_fidelity"] == "missing"
    assert evidence["00f067aa0ba902b7"]["message_fidelity"]["gen_ai.input.messages"] == "truncated"
