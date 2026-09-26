"""Prepared Claude Code streams preserve normalization across interleaved sessions."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.assembly_claude_code import _annotate_messages_with_history_paste
from polylogue.sources.dispatch import parse_stream_payload
from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator
from polylogue.sources.parsers.base import ParsedMessage, ParsedPasteEvidence, ParsedSession
from polylogue.sources.parsers.claude.history import HistoryEntry, HistoryPaste
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore, SqliteSessionEventSink
from polylogue.sources.sidecar_evidence import RetainedSidecarFile, RetainedSidecarScope, SiblingTranscript

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "claude-code" / "claude-normalization-main.jsonl"


def test_paste_evidence_json_round_trip_preserves_digest_bytes() -> None:
    digest = sha256(b"paste evidence").digest()
    evidence = ParsedPasteEvidence(content_hash=digest)

    assert evidence.content_hash == digest
    assert ParsedPasteEvidence.model_validate_json(evidence.model_dump_json()).content_hash == digest


def _snapshot(sessions: list[ParsedSession]) -> list[dict[str, object]]:
    return [
        {
            "session_id": session.provider_session_id,
            "title": session.title,
            "parent": session.parent_session_provider_id,
            "branch": session.branch_type,
            "leaf": session.active_leaf_message_provider_id,
            "messages": [message.model_dump(mode="json") for message in session.messages],
            "events": [event.model_dump(mode="json") for event in session.session_events],
        }
        for session in sessions
    ]


def test_prepared_multiway_stream_matches_collected_normalization(tmp_path: Path) -> None:
    records = [json.loads(line) for line in _FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    expected = _snapshot(parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "claude-normalization-main"))

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        actual_sessions = parse_stream_payload(
            Provider.CLAUDE_CODE,
            iter(records),
            "claude-normalization-main",
            message_sink_factory=store.new_sink,
            event_sink_factory=store.new_event_sink,
        )
        assert len(actual_sessions) == 2
        assert all(isinstance(session.messages, SqliteMessageSink) for session in actual_sessions)
        assert all(isinstance(session.session_events, SqliteSessionEventSink) for session in actual_sessions)
        assert _snapshot(actual_sessions) == expected
    finally:
        store.close()


def test_prepared_prefix_and_repeated_uuid_use_disk_state(tmp_path: Path) -> None:
    prefix = [
        {"type": "user", "uuid": f"prefix-{index}", "message": {"role": "user", "content": f"prefix {index}"}}
        for index in range(12)
    ]
    records = [
        *prefix,
        {"type": "user", "sessionId": "main", "uuid": "same", "message": {"role": "user", "content": "first"}},
        {"type": "user", "sessionId": "main", "uuid": "same", "message": {"role": "user", "content": "duplicate"}},
        {"type": "user", "sessionId": "other", "uuid": "other-1", "message": {"role": "user", "content": "other"}},
    ]
    expected = _snapshot(parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "main"))
    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        actual = parse_stream_payload(
            Provider.CLAUDE_CODE,
            iter(records),
            "main",
            message_sink_factory=store.new_sink,
            event_sink_factory=store.new_event_sink,
        )
        assert _snapshot(actual) == expected
        assert [len(session.messages) for session in actual] == [13, 1]
    finally:
        store.close()


def test_disk_sidecar_index_preserves_sibling_ownership() -> None:
    own = {
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": "own", "content": "Full output saved to: /tmp/own.txt"}]
        }
    }
    sibling = {"message": {"content": [{"type": "tool_result", "tool_use_id": "sibling", "content": "short"}]}}
    scope = RetainedSidecarScope(
        available=True,
        files=(
            RetainedSidecarFile("own.txt", 10, None, lambda: "full output"),
            RetainedSidecarFile("sibling.txt", 7, None, lambda: "sibling"),
        ),
        siblings=(SiblingTranscript("agent-one", lambda: iter([sibling])),),
    )
    with ToolResultIndexAccumulator() as collected:
        collected.observe(own)
        expected = collected.join_session_scoped(scope, "/sessions/main.jsonl")
    with ToolResultIndexAccumulator(disk_backed=True) as prepared:
        prepared.observe(own)
        actual = prepared.join_session_scoped(scope, "/sessions/main.jsonl")
    assert actual == expected
    assert [match.tool_use_id for match in actual.matched] == ["own"]
    assert not actual.debt


def test_prepared_background_update_replaces_start_outcome(tmp_path: Path) -> None:
    records = [
        {
            "type": "user",
            "sessionId": "background",
            "uuid": "start",
            "toolUseResult": {"backgroundTaskId": "task-1"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "Command running in background with ID: task-1.",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "user",
            "sessionId": "background",
            "uuid": "completion",
            "origin": {"kind": "task-notification"},
            "message": {
                "role": "user",
                "content": "<task-notification><task-id>task-1</task-id><tool-use-id>tool-1</tool-use-id>"
                "<output-file>/tmp/task-1.output</output-file><status>failed</status>"
                '<summary>Background command "false" failed with exit code 1</summary></task-notification>',
            },
        },
    ]
    expected = _snapshot(parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "background"))
    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        actual = parse_stream_payload(
            Provider.CLAUDE_CODE,
            iter(records),
            "background",
            message_sink_factory=store.new_sink,
            event_sink_factory=store.new_event_sink,
        )
        assert _snapshot(actual) == expected
        assert actual[0].messages[0].blocks[0].exit_code == 1
        assert [event.event_type for event in actual[0].session_events].count("background_task_completion") == 1
    finally:
        store.close()


def test_prepared_sidecar_join_replaces_block_and_appends_event(tmp_path: Path) -> None:
    records = [
        {
            "type": "user",
            "sessionId": "main",
            "uuid": "result",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "Full output saved to: /tmp/tool-1.txt",
                    }
                ],
            },
        }
    ]
    scope = RetainedSidecarScope(
        available=True,
        files=(RetainedSidecarFile("tool-1.txt", 11, 1_700_000_000_000, lambda: "full output"),),
    )

    class Resolver:
        def claude_code_scope(self, _source_path: str | Path | None) -> RetainedSidecarScope:
            return scope

        def gemini_cli_scope(self, _source_path: str | Path | None, _session_id: str | None) -> RetainedSidecarScope:
            return scope

    resolver = Resolver()
    expected = _snapshot(
        parse_stream_payload(
            Provider.CLAUDE_CODE,
            iter(records),
            "main",
            source_path="/sessions/main.jsonl",
            sidecar_resolver=resolver,
        )
    )
    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        actual = parse_stream_payload(
            Provider.CLAUDE_CODE,
            iter(records),
            "main",
            source_path="/sessions/main.jsonl",
            sidecar_resolver=resolver,
            message_sink_factory=store.new_sink,
            event_sink_factory=store.new_event_sink,
        )
        assert _snapshot(actual) == expected
        assert actual[0].messages[0].blocks[0].text == "full output"
        assert actual[0].session_events[-1].event_type == "claude_tool_result_sidecar"
    finally:
        store.close()


def test_prepared_compaction_event_keeps_boundary_coordinates(tmp_path: Path) -> None:
    records = [
        {"type": "user", "sessionId": "main", "uuid": "first", "message": {"role": "user", "content": "first"}},
        {"type": "summary", "sessionId": "main", "uuid": "compact", "summary": "Earlier context"},
        {"type": "user", "sessionId": "main", "uuid": "last", "message": {"role": "user", "content": "last"}},
    ]
    expected = parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "main")[0]
    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        actual = parse_stream_payload(
            Provider.CLAUDE_CODE,
            iter(records),
            "main",
            message_sink_factory=store.new_sink,
            event_sink_factory=store.new_event_sink,
        )[0]
        assert _snapshot([actual]) == _snapshot([expected])
        expected_boundary = next(event for event in expected.session_events if event.event_type == "compaction")
        actual_boundary = next(event for event in actual.session_events if event.event_type == "compaction")
        assert (
            actual_boundary.boundary_start_position,
            actual_boundary.boundary_end_position,
            actual_boundary.boundary_message_position,
        ) == (
            expected_boundary.boundary_start_position,
            expected_boundary.boundary_end_position,
            expected_boundary.boundary_message_position,
        )
        assert actual.active_leaf_message_provider_id == "last"
    finally:
        store.close()


def test_history_paste_enrichment_preserves_prepared_sink_and_matching_law(tmp_path: Path) -> None:
    messages = [
        ParsedMessage(
            provider_message_id="u0",
            role=Role.USER,
            text="First",
            timestamp="2026-01-01T00:00:00Z",
            paste_spans=[ParsedPasteEvidence(position=0, boundary_state="projected")],
        ),
        ParsedMessage(provider_message_id="a1", role=Role.ASSISTANT, text="Reply", timestamp="2026-01-01T00:00:05Z"),
        ParsedMessage(provider_message_id="u2", role=Role.USER, text="Second", timestamp="2026-01-01T00:00:20Z"),
        ParsedMessage(provider_message_id="u3", role=Role.USER, text="Third", timestamp="2026-01-01T00:00:23Z"),
    ]

    def entry(offset_seconds: int, paste_id: str) -> HistoryEntry:
        return HistoryEntry(
            display="",
            timestamp_ms=1_767_225_600_000 + offset_seconds * 1000,
            project=None,
            session_id="main",
            pastes=(HistoryPaste(paste_id=paste_id, paste_type="text", content=paste_id, has_content=True),),
        )

    entries = [entry(1, "first"), entry(2, "last-wins"), entry(21, "ambiguous"), entry(40, "unmatched")]
    collected = ParsedSession(source_name=Provider.CLAUDE_CODE, provider_session_id="main", messages=messages)
    expected = _annotate_messages_with_history_paste(collected, entries)
    assert collected.messages[0].paste_spans == messages[0].paste_spans
    assert [span.source_marker for span in expected.messages[0].paste_spans] == [None, "last-wins"]
    assert all(not message.paste_spans for message in expected.messages[1:])

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        sink = store.new_sink()
        sink.extend(messages)
        prepared = ParsedSession(source_name=Provider.CLAUDE_CODE, provider_session_id="main", messages=[]).model_copy(
            update={"messages": sink}
        )
        actual = _annotate_messages_with_history_paste(prepared, entries)
        assert id(actual.messages) == id(sink)
        assert [message.model_dump(mode="json") for message in actual.messages] == [
            message.model_dump(mode="json") for message in expected.messages
        ]
    finally:
        store.close()
