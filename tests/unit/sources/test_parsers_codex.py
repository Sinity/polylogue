"""Dedicated tests for the Codex JSONL parser.

Covers format detection, envelope/direct parsing, session metadata,
branch tracking, git context, and edge cases.
"""

from __future__ import annotations

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.sources.parsers.base import ParsedConversation
from polylogue.sources.parsers.codex import looks_like as _looks_like_impl
from polylogue.sources.parsers.codex import parse as _parse_impl
from polylogue.sources.parsers.codex import parse_stream


def looks_like(payload: object) -> bool:
    if not isinstance(payload, list):
        return False
    return _looks_like_impl(payload)


def parse(payload: object, fallback_id: str) -> ParsedConversation:
    assert isinstance(payload, list)
    return _parse_impl(payload, fallback_id)


def _provider_meta(conversation: ParsedConversation) -> dict[str, object]:
    assert conversation.provider_meta is not None
    return conversation.provider_meta


def _nested_meta(conversation: ParsedConversation, key: str) -> dict[str, object]:
    value = _provider_meta(conversation).get(key)
    assert isinstance(value, dict)
    return dict(value)


# =============================================================================
# Format Detection (looks_like)
# =============================================================================


class TestLooksLike:
    def test_envelope_format_detected(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
        ]
        assert looks_like(payload)

    def test_direct_format_detected(self) -> None:
        payload = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]
        assert looks_like(payload)

    def test_state_record_detected(self) -> None:
        payload = [{"record_type": "state"}]
        assert looks_like(payload)

    def test_intermediate_format_detected(self) -> None:
        """First line with id+timestamp (no type) is intermediate format."""
        payload = [{"id": "session-123", "timestamp": "2024-01-01T10:00:00Z"}]
        assert looks_like(payload)

    def test_empty_list_rejected(self) -> None:
        assert not looks_like([])

    def test_non_list_rejected(self) -> None:
        assert not looks_like({"type": "message"})

    def test_unrecognized_records_rejected(self) -> None:
        payload = [{"random": "data", "no_type": True}]
        assert not looks_like(payload)

    def test_non_dict_items_skipped(self) -> None:
        payload = ["string", 42, None, {"type": "message", "role": "user", "content": []}]
        assert looks_like(payload)  # The dict item matches


# =============================================================================
# Session Metadata
# =============================================================================


class TestSessionMetadata:
    def test_first_session_meta_sets_conversation_id(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc", "timestamp": "2024-01-01"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_conversation_id == "conv-abc"

    def test_second_session_meta_sets_parent(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc", "timestamp": "2024-01-01"}},
            {"type": "session_meta", "payload": {"id": "parent-xyz", "timestamp": "2024-01-01"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_conversation_id == "conv-abc"
        assert result.parent_conversation_provider_id == "parent-xyz"
        assert result.branch_type == BranchType.CONTINUATION

    def test_no_session_meta_uses_fallback(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "my-fallback")
        assert result.provider_conversation_id == "my-fallback"
        assert result.parent_conversation_provider_id is None
        assert result.branch_type is None

    def test_duplicate_session_meta_id_not_counted_twice(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "same-id", "timestamp": "2024-01-01"}},
            {"type": "session_meta", "payload": {"id": "same-id", "timestamp": "2024-01-01"}},
        ]
        result = parse(payload, "fallback")
        assert result.provider_conversation_id == "same-id"
        assert result.parent_conversation_provider_id is None
        assert result.branch_type is None

    def test_intermediate_format_metadata(self) -> None:
        """Intermediate format: first line has id+timestamp."""
        payload = [
            {"id": "conv-xyz", "timestamp": "2024-01-01T12:00:00Z"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ]
        result = parse(payload, "fallback")
        assert result.provider_conversation_id == "conv-xyz"


# =============================================================================
# Message Parsing
# =============================================================================


class TestMessageParsing:
    def test_envelope_message_parsed(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is 2+2?"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].text == "What is 2+2?"

    def test_direct_message_parsed(self) -> None:
        payload = [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The answer is 4."}]},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].role == "assistant"

    def test_state_records_skipped(self) -> None:
        payload = [
            {"record_type": "state"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"record_type": "state"},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1

    def test_multiple_content_blocks(self) -> None:
        payload = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Part 1"},
                    {"type": "input_text", "text": "Part 2"},
                ],
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        # Text content should contain both parts
        text = result.messages[0].text or ""
        assert "Part 1" in text
        assert "Part 2" in text

    def test_empty_content_skipped(self) -> None:
        payload = [
            {"type": "message", "role": "user", "content": []},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 0

    def test_message_without_text_skipped(self) -> None:
        """Message with only non-text blocks is skipped."""
        payload = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "tool_use", "name": "search"},
                ],
            },
        ]
        result = parse(payload, "fallback")
        # Message has no text content → skipped
        assert len(result.messages) == 0

    def test_message_role_normalization(self) -> None:
        """Roles are normalized via Role.normalize()."""
        payload = [
            {"type": "message", "role": "User", "content": [{"type": "input_text", "text": "hello"}]},
            {"type": "message", "role": "ASSISTANT", "content": [{"type": "output_text", "text": "hi"}]},
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_envelope_payload_unwrapped(self) -> None:
        """response_item payloads are unwrapped and parsed as messages."""
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "query"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].text == "query"

    def test_messages_do_not_keep_raw_provider_meta(self) -> None:
        payload = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
                "timestamp": "2024-01-01T00:00:01Z",
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 2
        assert all(message.provider_meta is None for message in result.messages)

    def test_parse_stream_matches_list_parse(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-abc", "timestamp": "2024-01-01T00:00:00Z"}},
            {
                "type": "message",
                "id": "msg-1",
                "role": "user",
                "timestamp": "2024-01-01T00:00:01Z",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "message",
                "id": "msg-2",
                "role": "assistant",
                "timestamp": "2024-01-01T00:00:02Z",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        ]

        from_list = parse(payload, "fallback")
        from_stream = parse_stream(iter(payload), "fallback")

        assert from_stream == from_list


# =============================================================================
# Git Context and Instructions
# =============================================================================


class TestGitContextAndInstructions:
    def test_git_context_from_session_meta(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "git": {"branch": "main", "commit": "abc123"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        git_meta = _nested_meta(result, "git")
        assert git_meta["branch"] == "main"
        assert git_meta["commit"] == "abc123"

    def test_instructions_from_session_meta(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "instructions": "You are a helpful assistant.",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert _provider_meta(result)["instructions"] == "You are a helpful assistant."

    def test_git_context_from_intermediate_metadata(self) -> None:
        """Intermediate format: git context on first line."""
        payload = [
            {
                "id": "conv-xyz",
                "timestamp": "2024-01-01T12:00:00Z",
                "git": {"branch": "develop"},
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ]
        result = parse(payload, "fallback")
        assert _nested_meta(result, "git")["branch"] == "develop"

    def test_git_and_instructions_combined(self) -> None:
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s1",
                    "timestamp": "2024-01-01",
                    "git": {"branch": "feature"},
                    "instructions": "Be concise.",
                },
            },
        ]
        result = parse(payload, "fallback")
        assert _nested_meta(result, "git")["branch"] == "feature"
        assert _provider_meta(result)["instructions"] == "Be concise."

    def test_turn_context_cwd_feeds_working_directories(self) -> None:
        payload = [
            {"type": "turn_context", "payload": {"cwd": "/repo/polylogue"}},
            {"type": "turn_context", "payload": {"turn_context": {"cwd": "/repo/other"}}},
        ]

        result = parse(payload, "fallback")

        assert _provider_meta(result)["working_directories"] == ["/repo/other", "/repo/polylogue"]
        assert result.provider_events[0].payload["cwd"] == "/repo/polylogue"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_empty_payload(self) -> None:
        result = parse([], "fallback")
        assert result.provider_conversation_id == "fallback"
        assert result.messages == []
        assert result.provider_name == "codex"

    def test_all_state_records(self) -> None:
        payload = [{"record_type": "state"} for _ in range(5)]
        result = parse(payload, "fallback")
        assert len(result.messages) == 0

    def test_invalid_records_skipped(self) -> None:
        payload = [
            42,  # non-dict
            "string",  # non-dict
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1

    def test_provider_is_codex(self) -> None:
        result = parse([], "fallback")
        assert result.provider_name == "codex"

    def test_timestamp_preserved_on_messages(self) -> None:
        """Message timestamps are preserved from record."""
        payload = [
            {
                "type": "message",
                "role": "user",
                "timestamp": "2024-03-15T10:30:00Z",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2024-03-15T10:30:00Z"

    def test_numeric_epoch_timestamp_is_normalized(self) -> None:
        """Numeric epoch timestamps survive typed validation and normalize to ISO text."""
        payload = [
            {
                "type": "message",
                "role": "user",
                "timestamp": 1705312200.0,
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]

        result = parse(payload, "fallback")

        assert len(result.messages) == 1
        assert result.messages[0].timestamp == "2024-01-15T09:50:00+00:00"

    def test_conversation_updated_at_uses_latest_message_timestamp(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "conv-1", "timestamp": "2024-03-15T10:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-1",
                    "role": "user",
                    "timestamp": "2024-03-15T10:30:00Z",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-2",
                    "role": "assistant",
                    "timestamp": "2024-03-15T10:45:00Z",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            },
        ]

        result = parse(payload, "fallback")

        assert result.created_at == "2024-03-15T10:00:00Z"
        assert result.updated_at == "2024-03-15T10:45:00Z"

    def test_message_id_fallback(self) -> None:
        """Message ID falls back to f'msg-{idx}' if not provided."""
        payload = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {
                "type": "message",
                "role": "user",
                "id": "explicit-id",
                "content": [{"type": "input_text", "text": "second"}],
            },
        ]
        result = parse(payload, "fallback")
        assert len(result.messages) == 2
        # First message should have fallback ID
        assert result.messages[0].provider_message_id.startswith("msg-")
        # Second message should use explicit ID
        assert result.messages[1].provider_message_id == "explicit-id"

    def test_complex_real_world_payload(self) -> None:
        """Real-world example: envelope format with git, instructions, multiple messages."""
        payload = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "prod-session-001",
                    "timestamp": "2024-03-15T14:30:00Z",
                    "git": {"branch": "main", "commit": "f1e2d3c"},
                    "instructions": "You are an expert Python developer.",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "How do I async/await?"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Use asyncio module."}],
                },
            },
            {"record_type": "state"},  # Ignored
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Show me an example."}],
                },
            },
        ]
        result = parse(payload, "fallback")
        assert result.provider_conversation_id == "prod-session-001"
        assert len(result.messages) == 3
        assert _nested_meta(result, "git")["commit"] == "f1e2d3c"
        assert _provider_meta(result)["instructions"] == "You are an expert Python developer."
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[2].role == "user"

    def test_function_call_items_become_tool_messages_and_events(self) -> None:
        payload = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd": "git status"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "clean",
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "token_count", "input_tokens": 10, "output_tokens": 5},
            },
        ]

        result = parse(payload, "fallback")

        assert [event.event_type for event in result.provider_events] == [
            "function_call",
            "function_call_output",
            "token_count",
        ]
        assert len(result.messages) == 2
        assert result.messages[0].content_blocks[0].type == "tool_use"
        assert result.messages[0].content_blocks[0].tool_name == "exec_command"
        assert result.messages[0].content_blocks[0].tool_input == {"cmd": "git status"}
        assert result.messages[1].content_blocks[0].type == "tool_result"
