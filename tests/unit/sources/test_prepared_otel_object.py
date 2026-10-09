"""A single OTLP-JSON export streams its spans through sealed preparation."""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import polylogue.sources.prepared_jsonl as prepared_jsonl
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import spill_otlp_spans
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_payload
from polylogue.sources.origin_specs import SEMCONV_SCHEMA_URL
from polylogue.sources.parsers import otel_genai
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.prepared_jsonl import _index_otlp_spans, _otlp_envelope, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteSessionEventSink
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.retained_jsonl import retained_raw_fixture


def _attribute(key: str, value: object) -> dict[str, object]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": value}}


def _chat_span(
    trace: str, span: str, start: int, *, conversation: str | None, parent: str | None = None
) -> dict[str, Any]:
    attributes = [
        _attribute("gen_ai.operation.name", "chat"),
        _attribute("gen_ai.request.model", f"model-{start % 3}"),
        _attribute("gen_ai.input.messages", json.dumps([{"role": "user", "content": f"question {span}"}])),
        _attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "content": f"answer {span}"}])),
        _attribute("gen_ai.usage.input_tokens", 5),
        _attribute("gen_ai.usage.output_tokens", 7),
    ]
    if conversation is not None:
        attributes.append(_attribute("gen_ai.conversation.id", conversation))
    body: dict[str, Any] = {
        "traceId": trace,
        "spanId": span,
        "name": "chat",
        "startTimeUnixNano": str(start),
        "attributes": attributes,
        "status": {"code": 1},
    }
    if parent is not None:
        body["parentSpanId"] = parent
    return body


def _tool_span(trace: str, span: str, start: int, parent: str) -> dict[str, Any]:
    return {
        "traceId": trace,
        "spanId": span,
        "parentSpanId": parent,
        "name": "execute_tool",
        "startTimeUnixNano": str(start),
        "attributes": [
            _attribute("gen_ai.operation.name", "execute_tool"),
            _attribute("gen_ai.tool.name", "search"),
            _attribute("gen_ai.tool.call.result", f"result {span}"),
        ],
        "status": {"code": 2},
    }


def _document(span_count: int = 120) -> dict[str, Any]:
    """Resources, scopes and spans in the orders the parser must normalize.

    Covers: resource identity and scope schema URL after their spans, both
    scope array spellings, a dotted key posing as a nested path, conflicting
    and identical copies of one span, conversation inherited through a parent
    chain, a parent cycle, an unsupported schema, spans without identity and
    non-object members, and start times that are equal, negative or missing.
    """
    spans: list[Any] = []
    for index in range(span_count):
        trace = f"trace-{index % 4}"
        conversation = f"conversation-{index % 3}" if index % 5 == 0 else None
        parent = f"span-{index - 4}" if index >= 4 and index % 5 else None
        spans.append(
            _chat_span(trace, f"span-{index}", 1_000 + (index * 7919) % 97, conversation=conversation, parent=parent)
        )
        if index % 11 == 0:
            spans.append(_tool_span(trace, f"tool-{index}", 1_000 + index, f"span-{index}"))
    copied = next(span for span in spans if span["spanId"] == "span-3")
    spans.append(dict(copied))
    spans.append({**copied, "name": "chat conflicting copy"})
    spans.append({**copied, "name": "chat conflicting copy"})
    spans.append(_chat_span("trace-cycle", "cycle-a", 5, conversation=None, parent="cycle-b"))
    spans.append(_chat_span("trace-cycle", "cycle-b", 5, conversation=None, parent="cycle-a"))
    spans.append({**_chat_span("trace-missing-start", "unstarted", 0, conversation=None), "startTimeUnixNano": "later"})
    spans.append({"traceId": "", "spanId": "no-trace", "attributes": []})
    spans.append("not a span")
    spans.append(["not", "a", "span"])
    unsupported_spans = [
        {**_chat_span("trace-0", "span-0", 1_000, conversation="conversation-0"), "name": "unsupported copy"},
        _chat_span("trace-9", "unsupported-only", 3, conversation=None),
    ]
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {"spans": spans, "schemaUrl": SEMCONV_SCHEMA_URL, "scope": {"name": "synthetic"}},
                    {"schemaUrl": "https://example.invalid/genai/99", "spans": unsupported_spans},
                    "not a scope",
                    {"spans": "not an array"},
                ],
                "scopeSpans.item.spans": [_chat_span("trace-posing", "posing", 1, conversation="posing")],
                "instrumentationLibrarySpans": [
                    {"spans": [_chat_span("trace-ignored", "ignored", 1, conversation=None)]}
                ],
                "resource": {"attributes": [_attribute("service.name", "neutral-agent")]},
            },
            {
                "instrumentationLibrarySpans": [
                    {"spans": [_chat_span("trace-0", "span-0", 1_000, conversation=None)]},
                ],
            },
            "not a resource",
        ],
        "resource_spans": [{"scopeSpans": [{"spans": [_chat_span("trace-other", "other", 1, conversation=None)]}]}],
        "exportedBy": "neutral-exporter",
    }


def _source(tmp_path: Path, document: dict[str, Any] | str) -> Path:
    source = tmp_path / "otel" / "trace.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")
    return source


def _expected(document: dict[str, Any], source: Path) -> list[ParsedSession]:
    sessions = admit_parsed_sessions_for_publication(
        parse_payload(Provider.OTEL_GENAI, [document], "fallback", source_path=str(source)),
        provider=Provider.OTEL_GENAI,
        source_path=str(source),
    )
    for session in sessions:
        session.content_hash = session_content_hash(session)
    return sessions


def _assert_same_publication(
    actual: list[ParsedSession], expected: list[ParsedSession], shard_path: Path, tmp_path: Path
) -> None:
    assert [session.provider_session_id for session in actual] == [session.provider_session_id for session in expected]
    for left, right in zip(actual, expected, strict=True):
        assert left.content_hash == right.content_hash
        assert len(left.messages) == len(right.messages)
        assert [event.model_dump(mode="json") for event in left.session_events] == [
            event.model_dump(mode="json") for event in right.session_events
        ]
        exclude = {"messages", "session_events", "attachments"}
        assert left.model_dump(mode="json", exclude=exclude) == right.model_dump(mode="json", exclude=exclude)
    expected_shard = prepare_session_shard(tmp_path / "expected", expected)
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )


def _refuse_whole_document(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("whole-document decode or parse was used")

    monkeypatch.setattr(prepared_jsonl, "owned_json_records", refuse)
    monkeypatch.setattr(prepared_jsonl, "iter_parsed_payload", refuse)
    monkeypatch.setattr(otel_genai, "parse", refuse)


def test_otlp_export_streams_spans_to_scratch_before_eof_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document(span_count=3_000)
    source = _source(tmp_path, document)
    expected = _expected(document, source)
    assert len(expected) > 4
    events = [event for session in expected for event in session.session_events]
    assert any(event.event_type == "otel_conflicting_span_id" for event in events)
    assert any(
        event.event_type == "otel_span_evidence" and event.payload["schema_url_status"] == "unsupported"
        for event in events
    )
    assert not any(session.provider_session_id.endswith(":posing") for session in expected)

    _refuse_whole_document(monkeypatch)
    source_size = source.stat().st_size
    first_span_offset: int | None = None
    original_walk = spill_otlp_spans

    def tracked_walk(handle: Any, root_key: str, *, on_resource: Any, on_scope: Any, on_span: Any) -> bool:
        def record_span(*args: Any) -> None:
            nonlocal first_span_offset
            if first_span_offset is None:
                first_span_offset = handle.tell()
            on_span(*args)

        return original_walk(handle, root_key, on_resource=on_resource, on_scope=on_scope, on_span=record_span)

    monkeypatch.setattr(prepared_jsonl, "spill_otlp_spans", tracked_walk)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.OTEL_GENAI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered
    # The first span reached scratch while most of the document was unread.
    assert first_span_offset is not None and first_span_offset < source_size // 4
    actual = list(artifact.iter_sessions())
    assert all(isinstance(session.messages, SqliteMessageSink) for session in actual)
    assert all(isinstance(session.session_events, SqliteSessionEventSink) for session in actual)
    assert artifact.shard_path is not None
    _assert_same_publication(actual, expected, artifact.shard_path, tmp_path)
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not {table for table in tables if table.startswith(("otel_", "otlp_"))}


def test_otel_object_parser_keeps_order_conflicts_and_inherited_conversations() -> None:
    sessions = otel_genai.parse(_document(span_count=40), "ignored")
    identities = [session.provider_session_id for session in sessions]
    assert identities == sorted(identities, key=lambda identity: tuple(identity.split(":", 2)))
    by_identity = {session.provider_session_id: session for session in sessions}
    # span-9 has no conversation of its own; its parent span-5 carries one.
    conversation = by_identity["neutral-agent:conversation:conversation-2"]
    assert any(event.payload.get("span_id") == "span-9" for event in conversation.session_events)
    cycle = by_identity["neutral-agent:trace:trace-cycle"]
    assert [event.payload["span_id"] for event in cycle.session_events if event.event_type == "otel_span_evidence"] == [
        "cycle-a",
        "cycle-b",
    ]
    conflicts = [
        event.payload
        for session in sessions
        for event in session.session_events
        if event.event_type == "otel_conflicting_span_id"
    ]
    assert [(payload["span_id"], payload["schema_url"]) for payload in conflicts] == [
        ("span-0", "https://example.invalid/genai/99"),
        ("span-3", SEMCONV_SCHEMA_URL),
    ]
    assert all(session.models_used == sorted(session.models_used) for session in sessions)


def _topology_document() -> dict[str, Any]:
    """Cross-resource traces, adoption, membership and replayed history.

    The HTTP root of trace ``t-x`` lives under an unnamed resource while its
    GenAI children live under ``neutral-agent``; two conversations share that
    root, and the one whose span started first adopts it. Chat inputs replay
    earlier history behind a pinned system prompt, a tool exchange is
    replayed by id, and a ``text_completion`` span carries usage and a
    response model without messages. A plain HTTP trace is not this origin's
    material; a trace whose GenAI attribute survives only in a conflicting
    copy is. Copies naming different conversations leave their span on its
    trace, and an unidentified span with a future ``kind`` still reaches the
    admission proof.
    """

    def chat(span: str, start: int, conversation: str | None, inputs: list[Any], output: str, **extra: Any) -> Any:
        attributes = [
            _attribute("gen_ai.operation.name", "chat"),
            _attribute("gen_ai.input.messages", json.dumps(inputs)),
            _attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "content": output}])),
            _attribute("gen_ai.response.model", "response-model"),
        ]
        if conversation is not None:
            attributes.append(_attribute("gen_ai.conversation.id", conversation))
        return {
            "traceId": "t-x",
            "spanId": span,
            "parentSpanId": "http-root",
            "startTimeUnixNano": str(start),
            "attributes": attributes,
            **extra,
        }

    system = {"role": "system", "content": "pinned"}
    call = {"role": "assistant", "parts": [{"type": "tool_call", "id": "call-1"}]}
    result = {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call-1"}]}
    agent_spans = [
        chat("g1", 10, "conv-1", [system, {"role": "user", "content": "q1"}], "a1"),
        chat(
            "g2",
            20,
            "conv-1",
            [system, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, call, result],
            "a2",
        ),
        chat("g3", 15, "conv-2", [{"role": "user", "content": "other"}], "reply"),
        {
            "traceId": "t-x",
            "spanId": "tool-1",
            "parentSpanId": "g2",
            "startTimeUnixNano": "18",
            "attributes": [
                _attribute("gen_ai.operation.name", "execute_tool"),
                _attribute("gen_ai.tool.name", "search"),
                _attribute("gen_ai.tool.call.id", "call-1"),
                _attribute("gen_ai.tool.call.result", "found"),
            ],
        },
        {
            "traceId": "t-x",
            "spanId": "completion",
            "parentSpanId": "g1",
            "startTimeUnixNano": "25",
            "attributes": [
                _attribute("gen_ai.operation.name", "text_completion"),
                _attribute("gen_ai.request.model", "request-model"),
                _attribute("gen_ai.usage.input_tokens", 3),
            ],
        },
    ]
    ambiguous = _chat_span("t-amb", "amb", 30, conversation="c-a")
    late_plain = {"traceId": "t-late", "spanId": "late", "startTimeUnixNano": "40", "attributes": []}
    unnamed_spans = [
        {
            "traceId": "t-x",
            "spanId": "http-root",
            "startTimeUnixNano": "5",
            "attributes": [_attribute("http.method", "POST")],
        },
        {"traceId": "t-plain", "spanId": "db", "startTimeUnixNano": "6", "attributes": [_attribute("db.system", "x")]},
        ambiguous,
        {**ambiguous, "attributes": [*ambiguous["attributes"][:-1], _attribute("gen_ai.conversation.id", "c-b")]},
        late_plain,
        {**late_plain, "attributes": [_attribute("gen_ai.system", "neutral")]},
        {"kind": "future_span_kind", "attributes": []},
    ]
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attribute("service.name", "neutral-agent")]},
                "scopeSpans": [{"schemaUrl": SEMCONV_SCHEMA_URL, "spans": agent_spans}],
            },
            {
                "resource": {
                    "attributes": [
                        _attribute("deployment.environment", "neutral"),
                        _attribute("process.pid", 7),
                        _attribute("service.version", "2"),
                    ]
                },
                "scopeSpans": [{"spans": unnamed_spans}],
            },
        ]
    }


def test_otlp_topology_streams_with_parser_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = _topology_document()
    source = _source(tmp_path, document)
    parsed = [session.provider_session_id for session in otel_genai.parse(document, "ignored")]
    unnamed = next(identity.split(":", 1)[0] for identity in parsed if identity.startswith("resource-"))
    assert parsed == [
        "neutral-agent:conversation:conv-1",
        "neutral-agent:conversation:conv-2",
        f"{unnamed}:trace:t-amb",
        f"{unnamed}:trace:t-late",
    ]
    expected = _expected(document, source)
    by_identity = {session.provider_session_id: session for session in expected}
    assert f"{unnamed}:trace:t-amb" in by_identity
    first = by_identity["neutral-agent:conversation:conv-1"]
    evidence = [event.payload for event in first.session_events if event.event_type == "otel_span_evidence"]
    # The cross-resource root joins the conversation that started first and
    # keeps its own resource identity.
    assert [(payload["span_id"], payload.get("resource_id")) for payload in evidence][0] == ("http-root", unnamed)
    # Replayed history and the replayed tool exchange add only the new turn.
    assert [message.text for message in first.messages] == ["pinned", "q1", "a1", None, None, "a2"]
    assert first.models_used == ["request-model", "response-model"]
    assert any(
        event.event_type == "message_usage" and event.payload["model"] == "request-model"
        for event in first.session_events
    )
    assert first.unit_accounting is not None
    unknown = [event.payload for event in first.session_events if event.event_type.endswith("_unknown_input")]
    assert unknown == [{"source_index": 1, "wire_type": "future_span_kind"}]

    _refuse_whole_document(monkeypatch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.OTEL_GENAI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.shard_path is not None
    _assert_same_publication(list(artifact.iter_sessions()), expected, artifact.shard_path, tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        '{"resourceSpans": [{"scopeSpans": [], "scopeSpans": []}]}',
        '{"resourceSpans": [{"scopeSpans": [{"spans": [], "spans": []}]}]}',
    ],
)
def test_otlp_walk_refuses_repeated_scope_keys(text: str) -> None:
    conn = sqlite3.connect(":memory:")
    envelope = _otlp_envelope(BytesIO(text.encode()))
    assert envelope is not None
    assert _index_otlp_spans(BytesIO(text.encode()), envelope[1], conn) is None
    assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone() == (0,)


@pytest.mark.parametrize(
    "text",
    [
        '{"resourceSpans": {"not": "an array"}}',
        '{"resourceSpans": [], "resourceSpans": []}',
        '{"sessions": [], "resourceSpans": []}',
        '{"exportedBy": "neutral"}',
        '[{"resourceSpans": []}]',
    ],
)
def test_otlp_probe_leaves_other_shapes_to_the_object_parser(text: str) -> None:
    assert _otlp_envelope(BytesIO(text.encode())) is None


def test_repeated_scope_keys_keep_collecting_parity(tmp_path: Path) -> None:
    span = _chat_span("trace-a", "span-a", 1, conversation="neutral")
    replaced = _chat_span("trace-b", "span-b", 2, conversation="neutral")
    text = (
        '{"resourceSpans": [{"scopeSpans": [{"spans": ['
        + json.dumps(span)
        + ']}], "scopeSpans": [{"spans": ['
        + json.dumps(replaced)
        + "]}]}]}"
    )
    source = _source(tmp_path, text)
    expected = _expected(json.loads(text), source)
    assert [event.payload["span_id"] for event in expected[0].session_events] == ["span-b"]
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.OTEL_GENAI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.shard_path is not None
    _assert_same_publication(list(artifact.iter_sessions()), expected, artifact.shard_path, tmp_path)


def test_otlp_export_corrupt_suffix_leaves_no_artifact(tmp_path: Path) -> None:
    source = _source(tmp_path, json.dumps(_document(20))[:-3] + ", {broken")
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.OTEL_GENAI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


@pytest.mark.parametrize("failure", ["mutation", "parser"])
def test_otlp_export_failure_after_spill_discards_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = _source(tmp_path, _document(40))
    original_append = SqliteSessionEventSink.append
    written = 0

    def append_then_fail(self: SqliteSessionEventSink, value: Any) -> None:
        nonlocal written
        original_append(self, value)
        written += 1
        if written == 5:
            if failure == "mutation":
                source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
            else:
                raise RuntimeError("synthetic parse worker failure")

    monkeypatch.setattr(SqliteSessionEventSink, "append", append_then_fail)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.OTEL_GENAI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert written >= 5
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert artifact.deferred is (failure == "mutation")
    assert list(directory.glob("*.db")) == []


def test_retained_otlp_export_uses_streamed_replay_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.sources.revision_backfill as revision_backfill

    document = _document(60)
    source_path = str(tmp_path / "otel" / "trace.json")
    expected = _expected(document, Path(source_path))
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(document).encode())
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.OTEL_GENAI,
        blob_hash=blob_hash,
        source_path=source_path,
        file_mtime="2025-01-02T03:04:05Z",
    ) as (reader, raw_id):
        _refuse_whole_document(monkeypatch)
        artifact = revision_backfill.prepare_retained_jsonl_artifact(
            reader, raw_id, directory=BlobStore(blob_root)._ensure_private_staging_root() / "prepared"
        )
        try:
            assert artifact.error is None, artifact.error
            assert artifact.positive_evidence_filtered
            sessions = list(artifact.iter_sessions())
            assert [session.provider_session_id for session in sessions] == [
                session.provider_session_id for session in expected
            ]
            assert [len(session.session_events) for session in sessions] == [
                len(session.session_events) for session in expected
            ]
        finally:
            artifact.discard()
