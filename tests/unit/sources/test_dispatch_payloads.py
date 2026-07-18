"""Provider-dispatch payload normalization regressions."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from polylogue.archive.session.branch_type import BranchType
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import _payload_record, _payload_sequence, parse_payload, parse_stream_payload
from polylogue.sources.source_parsing import iter_source_sessions_with_raw

HERMES_ATOF_FIXTURE = Path(__file__).parents[2] / "fixtures/hermes/atof/nemo_relay_atof_v0.1_real_redacted.jsonl"


def test_payload_sequence_normalizes_streaming_decimals() -> None:
    payload = [{"whole": Decimal("2"), "fraction": Decimal("2.5"), "items": [Decimal("3")]}]

    normalized = _payload_sequence(payload)
    assert normalized == [{"whole": 2, "fraction": 2.5, "items": [3]}]
    first = normalized[0]
    assert isinstance(first, dict)
    items = first["items"]
    assert isinstance(items, list)
    assert isinstance(first["whole"], int)
    assert isinstance(first["fraction"], float)
    assert isinstance(items[0], int)


def test_payload_record_normalizes_streaming_decimals_for_chatgpt_parse() -> None:
    payload = {
        "id": "chatgpt-decimal",
        "title": "Decimal timestamp",
        "create_time": Decimal("1704995846.046526"),
        "mapping": {
            "root": {
                "id": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
                "children": [],
            }
        },
    }

    normalized = _payload_record(payload)
    assert normalized is not None
    assert normalized["create_time"] == 1704995846.046526

    sessions = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "chatgpt-decimal"


def test_parse_payload_unwraps_single_gemini_cli_record_list() -> None:
    """Full-ingest passes ``list(_iter_json_stream(...))``; a one-record gemini-cli
    file therefore arrives as a single-element list. It must still parse rather
    than silently yielding no sessions (which marked the file a permanent parse
    failure and looped retries forever)."""
    record = {
        "sessionId": "gemini-session-1",
        "projectHash": "abc",
        "startTime": "2026-04-02T09:58:03.920Z",
        "lastUpdated": "2026-04-02T10:09:41.353Z",
        "messages": [
            {"id": "m1", "timestamp": "2026-04-02T09:58:03.920Z", "type": "user", "content": [{"text": "hi"}]},
        ],
    }

    from_dict = parse_payload(Provider.GEMINI_CLI, record, "fallback")
    from_list = parse_payload(Provider.GEMINI_CLI, [record], "fallback")

    assert len(from_dict) == 1
    assert len(from_list) == 1
    assert from_list[0].provider_session_id == from_dict[0].provider_session_id


def test_parse_payload_unwraps_single_antigravity_metadata_list() -> None:
    """Antigravity ``*.metadata.json`` brain artifacts are single JSON objects;
    the list-wrapped full-ingest input must still resolve via source_path."""
    record = {
        "artifactType": "ARTIFACT_TYPE_IMPLEMENTATION_PLAN",
        "summary": "A plan summary.",
        "updatedAt": "2026-01-07T04:39:32.150534411Z",
    }
    source_path = "/x/brain/abc/plan.md.metadata.json"

    from_dict = parse_payload(Provider.ANTIGRAVITY, record, "fallback", source_path=source_path)
    from_list = parse_payload(Provider.ANTIGRAVITY, [record], "fallback", source_path=source_path)

    assert len(from_dict) == 1
    assert len(from_list) == 1


def test_source_parser_groups_real_shaped_hermes_atof_jsonl_as_one_retained_stream() -> None:
    """Exercise the filesystem emitter rather than only direct parser calls.

    Mutation: omit Hermes from grouped/stream providers or classify ATOF as a
    generic hook sidecar. The actual source walk stops producing the one
    observer session and its retained whole-file raw evidence.
    """
    source = Source(name="hermes", path=HERMES_ATOF_FIXTURE.parent)
    pairs = list(iter_source_sessions_with_raw(source, capture_raw=True))
    assert len(pairs) == 1
    raw, session = pairs[0]
    assert raw is not None
    assert raw.source_path == str(HERMES_ATOF_FIXTURE)
    assert raw.source_index is None
    assert session.provider_session_id == "observer:real-nemo-relay-session-redacted"
    assert "hermes:atof-observer" in session.ingest_flags


def test_parse_payload_splits_claude_code_aggregate_by_session_id() -> None:
    payload = [
        {"type": "summary", "leafUuid": "leaf-1", "summary": "Previous context"},
        {
            "type": "user",
            "sessionId": "first-session",
            "uuid": "first-user",
            "timestamp": "2026-06-30T01:00:00Z",
            "message": {"role": "user", "content": "first prompt"},
        },
        {
            "type": "assistant",
            "sessionId": "first-session",
            "uuid": "first-assistant",
            "timestamp": "2026-06-30T01:01:00Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "first reply"}]},
        },
        {
            "type": "user",
            "sessionId": "second-session",
            "uuid": "second-user",
            "timestamp": "2026-06-30T02:00:00Z",
            "message": {"role": "user", "content": "second prompt"},
        },
        {
            "type": "user",
            "sessionId": "first-session",
            "uuid": "first-user-later",
            "timestamp": "2026-06-30T03:00:00Z",
            "message": {"role": "user", "content": "first prompt later"},
        },
    ]

    sessions = parse_payload(Provider.CLAUDE_CODE, payload, "aggregate-file")

    assert [session.provider_session_id for session in sessions] == ["first-session", "second-session"]
    assert [len(session.messages) for session in sessions] == [4, 1]
    assert sessions[0].messages[0].message_type.value == "summary"
    assert sessions[1].messages[0].text == "second prompt"


def test_parse_stream_payload_splits_claude_code_aggregate_by_session_id() -> None:
    payload = [
        {
            "type": "user",
            "sessionId": "first-session",
            "uuid": "first-user",
            "message": {"role": "user", "content": "first prompt"},
        },
        {
            "type": "user",
            "sessionId": "second-session",
            "uuid": "second-user",
            "message": {"role": "user", "content": "second prompt"},
        },
        {
            "type": "assistant",
            "sessionId": "first-session",
            "uuid": "first-assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "first reply"}]},
        },
    ]

    sessions = parse_stream_payload(Provider.CLAUDE_CODE, iter(payload), "aggregate-file")

    assert [session.provider_session_id for session in sessions] == ["first-session", "second-session"]
    assert [len(session.messages) for session in sessions] == [2, 1]


def test_claude_acompact_classifier_matches_eager_and_memory_bounded_routes() -> None:
    """Production dependencies: first-group fallback identity and shared parser.

    Mutation: let eager grouping replace the first fallback with ``sessionId`` or
    duplicate the classifier in only one route. The model equality and acompact
    identity/topology assertions fail.
    """
    payload = [
        {
            "type": "user",
            "uuid": "task-u",
            "parentUuid": None,
            "sessionId": "parent-session",
            "isSidechain": True,
            "agentId": "task-agent",
            "promptId": "task-prompt",
            "message": {"role": "user", "content": "Task-local head."},
        },
        {
            "type": "assistant",
            "uuid": "task-a",
            "parentUuid": "task-u",
            "sessionId": "parent-session",
            "isSidechain": True,
            "message": {"role": "assistant", "content": "Task-local answer."},
        },
        {
            "type": "user",
            "uuid": "other-u",
            "sessionId": "other-session",
            "message": {"role": "user", "content": "Other session."},
        },
    ]
    fallback_id = "agent-acompact-route-parity"

    eager = parse_payload(Provider.CLAUDE_CODE, payload, fallback_id)
    streamed = parse_stream_payload(Provider.CLAUDE_CODE, iter(payload), fallback_id)

    assert [session.model_dump(mode="json") for session in streamed] == [
        session.model_dump(mode="json") for session in eager
    ]
    assert eager[0].provider_session_id == "parent-session:agent-acompact-route-parity"
    assert eager[0].parent_session_provider_id == "parent-session"
    assert eager[0].branch_type is BranchType.SIDECHAIN


def test_parse_one_source_path_treats_jsonl_text_json_wrappers_as_jsonl(tmp_path: Path) -> None:
    source_path = tmp_path / "session.jsonl.txt.json"
    source_path.write_text(
        "\n".join(
            (
                '{"type":"user","sessionId":"first-session","uuid":"u1","message":{"role":"user","content":"one"}}',
                '{"type":"user","sessionId":"second-session","uuid":"u2","message":{"role":"user","content":"two"}}',
            )
        ),
        encoding="utf-8",
    )

    rows = list(iter_source_sessions_with_raw(Source(name="claude-code", path=source_path), capture_raw=False))

    assert [session.provider_session_id for _raw, session in rows] == ["first-session", "second-session"]
