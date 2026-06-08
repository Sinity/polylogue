"""Codex event-stream contract tests.

These tests pin the on-disk semantics of the Codex JSONL "streaming envelope"
shape — what the persisted session records look like once a streaming response
has been realized. They are explicitly contract-shaped: if OpenAI changes the
envelope keys, payload type tokens, or call_id pairing semantics, these tests
should fail with a precise reason naming the broken field.

Catalog payloads live under ``tests/data/codex_event_stream/`` and cover three
realistic streaming shapes:

- ``text_only_stream.jsonl`` — session_meta + user/assistant text turn +
  trailing token_count.
- ``tool_call_stream.jsonl`` — single function_call paired with a
  function_call_output via ``call_id`` and a concluding assistant message.
- ``interleaved_stream.jsonl`` — multiple tool turns, mixed assistant
  messages, turn_context cwd changes, and a compaction event.

Ref #1296.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.codex import looks_like as _looks_like_impl
from polylogue.sources.parsers.codex import parse as _parse_impl
from polylogue.sources.parsers.codex import parse_stream
from polylogue.types import BlockType

CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "codex_event_stream"


# ---------------------------------------------------------------------------
# Catalog access helpers
# ---------------------------------------------------------------------------


def _load_catalog(name: str) -> list[object]:
    path = CATALOG_DIR / name
    records: list[object] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _parse(payload: list[object], fallback_id: str = "fallback") -> ParsedSession:
    return _parse_impl(payload, fallback_id)


def _looks_like(payload: list[object]) -> bool:
    return _looks_like_impl(payload)


# ---------------------------------------------------------------------------
# Catalog smoke — every shape must be detected and parsed without raising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    ["text_only_stream.jsonl", "tool_call_stream.jsonl", "interleaved_stream.jsonl"],
)
def test_catalog_streams_are_detected_as_codex(fixture: str) -> None:
    records = _load_catalog(fixture)
    assert _looks_like(records), f"{fixture} must satisfy looks_like()"


@pytest.mark.parametrize(
    "fixture",
    ["text_only_stream.jsonl", "tool_call_stream.jsonl", "interleaved_stream.jsonl"],
)
def test_catalog_streams_parse_without_raising(fixture: str) -> None:
    records = _load_catalog(fixture)
    session = _parse(records, fixture)
    assert session.source_name == "codex"
    # Each fixture must contribute at least one message — otherwise the
    # streaming envelope contract is silently dropping content.
    assert session.messages, f"{fixture} produced zero messages"


# ---------------------------------------------------------------------------
# response_item envelope shape — the "streaming envelope" contract
# ---------------------------------------------------------------------------


class TestStreamingEnvelopeShape:
    """Pin the on-disk streaming envelope contract.

    The persisted Codex JSONL wraps every realized streaming event in a
    ``response_item`` envelope with a typed ``payload``. These tests pin the
    contract: the outer ``type`` token, the inner payload ``type`` tokens we
    consume, and the message-vs-event dispatch.
    """

    def test_response_item_message_payload_routes_to_messages(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            }
        ]
        session = _parse(records)
        assert len(session.messages) == 1
        assert session.messages[0].role == Role.ASSISTANT
        assert session.session_events == []

    def test_response_item_non_message_payload_routes_to_events(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {"type": "token_count", "input_tokens": 1, "output_tokens": 2},
            }
        ]
        session = _parse(records)
        assert session.messages == []
        assert [event.event_type for event in session.session_events] == ["token_count"]

    def test_unknown_inner_payload_type_still_recorded_as_session_event(self) -> None:
        """Unknown payload type tokens must not be silently dropped.

        This is the canary for OpenAI introducing a new event class — the
        parser must still surface it as a session event so downstream
        consumers can see something is happening.
        """
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {"type": "future_event_kind", "data": 7},
            }
        ]
        session = _parse(records)
        assert [event.event_type for event in session.session_events] == ["future_event_kind"]


# ---------------------------------------------------------------------------
# response.completed equivalent — terminal message state
# ---------------------------------------------------------------------------


class TestResponseCompletedContract:
    """The on-disk equivalent of streaming ``response.completed``.

    In a live stream OpenAI emits a sequence of ``response.delta`` events
    terminated by ``response.completed``. The Codex CLI realizes those into a
    single persisted ``response_item`` whose payload is a fully-formed
    ``message`` (the completed state). The contract this pins:

    - The realized message text equals the concatenation of all output_text
      segments — there are no partial-token artifacts left over.
    - A subsequent ``token_count`` event marks the end-of-response boundary
      and does not corrupt the prior message.
    - The terminal message timestamp drives ``session.updated_at``.
    """

    def test_completed_message_concatenates_all_output_text_segments(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello "},
                        {"type": "output_text", "text": "world."},
                    ],
                },
            }
        ]
        session = _parse(records)
        assert len(session.messages) == 1
        text = session.messages[0].text or ""
        assert "Hello" in text and "world." in text

    def test_token_count_after_message_does_not_mutate_prior_message(self) -> None:
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "timestamp": "2025-01-15T10:00:08Z",
                    "content": [{"type": "output_text", "text": "Paris."}],
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "token_count", "input_tokens": 5, "output_tokens": 2},
            },
        ]
        session = _parse(records)
        assert len(session.messages) == 1
        assert session.messages[0].text == "Paris."
        assert session.messages[0].timestamp == "2025-01-15T10:00:08Z"
        assert [event.event_type for event in session.session_events] == ["token_count"]

    def test_terminal_message_drives_session_updated_at(self) -> None:
        records = _load_catalog("text_only_stream.jsonl")
        session = _parse(records)
        assert session.created_at == "2025-01-15T10:00:00Z"
        # Updated_at must come from the last assistant message, not from the
        # trailing token_count event (which has no timestamp).
        assert session.updated_at == "2025-01-15T10:00:08Z"


# ---------------------------------------------------------------------------
# Tool deltas → single tool_use block, paired by call_id
# ---------------------------------------------------------------------------


class TestToolCallStreamingContract:
    """A realized function_call envelope produces a single tool_use block.

    The streaming protocol may emit multiple tool-call deltas, but the
    persisted Codex JSONL collapses them into one ``function_call``
    response_item with the merged ``arguments`` JSON string. This pins that
    contract along with ``call_id`` round-trip semantics between
    ``function_call`` and ``function_call_output``.
    """

    def test_function_call_realizes_as_single_tool_use_block(self) -> None:
        records = _load_catalog("tool_call_stream.jsonl")
        session = _parse(records, "tool_call_stream")

        tool_use_msgs = [
            msg for msg in session.messages if any(block.type == BlockType.TOOL_USE for block in msg.content_blocks)
        ]
        assert len(tool_use_msgs) == 1
        tool_use_block = next(block for block in tool_use_msgs[0].content_blocks if block.type == BlockType.TOOL_USE)
        assert tool_use_block.tool_name == "exec_command"
        assert tool_use_block.tool_id == "call_abc"
        assert tool_use_block.tool_input == {"cmd": "ls /tmp"}

    def test_function_call_output_produces_paired_tool_result(self) -> None:
        records = _load_catalog("tool_call_stream.jsonl")
        session = _parse(records, "tool_call_stream")

        tool_results = [
            (msg, block)
            for msg in session.messages
            for block in msg.content_blocks
            if block.type == BlockType.TOOL_RESULT
        ]
        assert len(tool_results) == 1
        result_msg, result_block = tool_results[0]
        assert result_msg.role == Role.TOOL
        assert result_block.tool_id == "call_abc"
        assert result_block.text == "file1.txt\nfile2.txt"

    def test_call_id_pairs_tool_use_and_tool_result(self) -> None:
        """The cross-message contract: tool_use.tool_id == tool_result.tool_id."""
        records = _load_catalog("interleaved_stream.jsonl")
        session = _parse(records, "interleaved_stream")

        tool_use_ids = [
            block.tool_id
            for msg in session.messages
            for block in msg.content_blocks
            if block.type == BlockType.TOOL_USE
        ]
        tool_result_ids = [
            block.tool_id
            for msg in session.messages
            for block in msg.content_blocks
            if block.type == BlockType.TOOL_RESULT
        ]
        assert tool_use_ids == ["call_git", "call_read"]
        assert tool_result_ids == ["call_git", "call_read"]


# ---------------------------------------------------------------------------
# Truncated / out-of-order / malformed streams
# ---------------------------------------------------------------------------


class TestStreamResilience:
    """Contract for incomplete and disordered streams.

    The Codex parser is forgiving by design — it does not raise on truncation
    or reordering, and it must not silently drop content. These tests pin the
    observable behavior so a future change to "reject with typed error" would
    fail loudly and intentionally.
    """

    def test_truncated_stream_keeps_tool_use_without_paired_result(self) -> None:
        """A function_call with no following function_call_output must still
        appear in the parsed session. Silent drop would be a contract
        violation — the operator needs to see the partial turn."""
        records: list[object] = [
            {
                "type": "session_meta",
                "payload": {"id": "truncated", "timestamp": "2025-01-15T13:00:00Z"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "u1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Run something."}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_truncated",
                    "name": "exec_command",
                    "arguments": '{"cmd": "echo hi"}',
                },
            },
            # function_call_output intentionally absent — stream truncated.
        ]
        session = _parse(records, "truncated")

        tool_use_ids = [
            block.tool_id
            for msg in session.messages
            for block in msg.content_blocks
            if block.type == BlockType.TOOL_USE
        ]
        tool_result_ids = [
            block.tool_id
            for msg in session.messages
            for block in msg.content_blocks
            if block.type == BlockType.TOOL_RESULT
        ]
        assert tool_use_ids == ["call_truncated"]
        assert tool_result_ids == []
        # The session_events surface must record the unpaired function_call
        # so downstream consumers can detect the truncation.
        assert "function_call" in {event.event_type for event in session.session_events}

    def test_out_of_order_tool_output_before_call_is_preserved(self) -> None:
        """Output arriving before its call must still be parsed — the parser
        does not enforce ordering, but it also must not drop either side."""
        records: list[object] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_oo",
                    "output": "early-output",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "fc_oo",
                    "call_id": "call_oo",
                    "name": "exec_command",
                    "arguments": '{"cmd": "noop"}',
                },
            },
        ]
        session = _parse(records, "out-of-order")

        tool_blocks = [
            (block.type, block.tool_id)
            for msg in session.messages
            for block in msg.content_blocks
            if block.type in (BlockType.TOOL_USE, BlockType.TOOL_RESULT)
        ]
        # Both halves of the pair must survive — order may be insertion order.
        assert (BlockType.TOOL_RESULT, "call_oo") in tool_blocks
        assert (BlockType.TOOL_USE, "call_oo") in tool_blocks

    def test_malformed_record_does_not_abort_stream(self) -> None:
        records: list[object] = [
            42,  # non-dict garbage
            {"definitely": "not-a-codex-record"},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "survivor",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "still here"}],
                },
            },
        ]
        session = _parse(records, "malformed")
        assert [msg.text for msg in session.messages] == ["still here"]

    def test_parse_stream_is_equivalent_to_parse_for_truncated_input(self) -> None:
        """Streaming consumption (iterator) and list consumption must agree
        even when the stream is truncated. This is the contract that lets the
        daemon parse mid-write JSONL files safely."""
        records = _load_catalog("tool_call_stream.jsonl")[:3]  # drop output + final msg + tokens
        from_list = _parse(records, "trunc")
        from_stream = parse_stream(iter(records), "trunc")
        assert from_list == from_stream


# ---------------------------------------------------------------------------
# Cross-message references (compaction, turn_context, parent session)
# ---------------------------------------------------------------------------


class TestCrossMessageReferences:
    """Pins the cross-message reference contract carried in the stream."""

    def test_second_session_meta_becomes_parent_continuation(self) -> None:
        records: list[object] = [
            {"type": "session_meta", "payload": {"id": "child", "timestamp": "2025-01-15T14:00:00Z"}},
            {"type": "session_meta", "payload": {"id": "parent", "timestamp": "2025-01-15T13:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "u1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continued"}],
                },
            },
        ]
        session = _parse(records, "fallback")
        assert session.provider_session_id == "child"
        assert session.parent_session_provider_id == "parent"
        assert session.branch_type == BranchType.CONTINUATION

    def test_turn_context_cwd_changes_collected_into_working_directories(self) -> None:
        records = _load_catalog("interleaved_stream.jsonl")
        session = _parse(records, "interleaved_stream")
        assert session.working_directories == ["/repo/other", "/repo/polylogue"]

    def test_compaction_surfaced_with_replacement_history_flag(self) -> None:
        records = _load_catalog("interleaved_stream.jsonl")
        session = _parse(records, "interleaved_stream")
        compactions = [e for e in session.session_events if e.event_type == "compaction"]
        assert len(compactions) == 1
        assert compactions[0].payload["replacement_history_count"] == 1
        assert compactions[0].payload["summary"] == "Conversation compacted"
