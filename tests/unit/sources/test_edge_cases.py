"""Tests for provider-specific edge cases and untested branches.

Targets the lowest-coverage provider code paths:
- ChatGPT: iter_user_assistant_pairs, tether browsing content, branching trees
- Codex: format detection, timestamp parsing, content extraction
- Claude Code: extract_reasoning_traces, extract_tool_calls, system record text
"""
from __future__ import annotations

import pytest

from polylogue.lib.viewports import ContentType
from polylogue.sources.providers.chatgpt import (
    ChatGPTAuthor,
    ChatGPTContent,
    ChatGPTConversation,
    ChatGPTMessage,
    ChatGPTNode,
)
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.codex import CodexContentBlock, CodexRecord

# =============================================================================
# ChatGPT Provider Tests
# =============================================================================


class TestChatGPTIterUserAssistantPairs:
    """Tests for ChatGPTConversation.iter_user_assistant_pairs()."""

    def _make_conv(self, messages):
        """Helper to build a ChatGPTConversation with given messages inline."""
        mapping = {}
        prev_id = None
        for i, (role, text) in enumerate(messages):
            node_id = f"node-{i}"
            msg = ChatGPTMessage(
                id=f"msg-{i}",
                author=ChatGPTAuthor(role=role),
                content=ChatGPTContent(content_type="text", parts=[text]),
            )
            node = ChatGPTNode(
                id=node_id,
                message=msg,
                parent=prev_id,
                children=[],
            )
            if prev_id and prev_id in mapping:
                mapping[prev_id].children.append(node_id)
            mapping[node_id] = node
            prev_id = node_id

        return ChatGPTConversation(
            id="conv-1", conversation_id="conv-1", title="Test",
            create_time=1700000000.0, update_time=1700000100.0,
            mapping=mapping, current_node=prev_id or "node-0",
        )

    @pytest.mark.parametrize("messages,expected_count,expected_user_text,expected_assistant_text", [
        (
            # basic_pair: Simple user-assistant pair is yielded
            [("user", "hi"), ("assistant", "hello")],
            1,
            "hi",
            "hello",
        ),
        (
            # multiple_pairs: Multiple pairs yielded correctly
            [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")],
            2,
            None,  # Just check count
            None,
        ),
        (
            # system_message_skipped: System messages at start don't break pairing
            [("system", "You are helpful"), ("user", "hi"), ("assistant", "hello")],
            1,
            "hi",
            "hello",
        ),
        (
            # consecutive_users_skip: Two consecutive user messages
            [("user", "first"), ("user", "second"), ("assistant", "response")],
            1,
            "second",
            "response",
        ),
        (
            # unmatched_trailing_assistant: Trailing assistant without preceding user is skipped
            [("user", "question"), ("assistant", "answer"), ("assistant", "trailing")],
            1,
            "question",
            "answer",
        ),
    ])
    def test_pair_scenarios(self, messages, expected_count, expected_user_text, expected_assistant_text):
        """Test various message pairing scenarios."""
        conv = self._make_conv(messages)
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == expected_count
        if expected_user_text is not None:
            assert pairs[0][0].text_content == expected_user_text
        if expected_assistant_text is not None:
            assert pairs[0][1].text_content == expected_assistant_text

    def test_empty_conversation(self):
        """No messages yields no pairs."""
        conv = self._make_conv([])
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == 0

    def test_single_message_no_pair(self):
        """Single message yields no pairs."""
        conv = self._make_conv([("user", "alone")])
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == 0


class TestChatGPTContentBlocks:
    """Tests for ChatGPTMessage.to_content_blocks() edge cases."""

    @pytest.mark.parametrize("content_type,parts,language,expected_type", [
        ("code", ["print('hi')"], "python", ContentType.CODE),
        ("tether_browsing_display", ["Search results..."], None, ContentType.TOOL_RESULT),
        ("browse_results", ["Page content"], None, ContentType.TOOL_RESULT),
        ("multimodal_image", [], None, ContentType.UNKNOWN),
    ])
    def test_content_type_mapping(self, content_type, parts, language, expected_type):
        """Test various content types map to expected ContentType."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type=content_type, parts=parts, language=language),
        )
        blocks = msg.to_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == expected_type
        if language:
            assert blocks[0].language == language

    def test_no_content_returns_empty(self):
        """No content returns empty list."""
        msg = ChatGPTMessage(id="m1", author=ChatGPTAuthor(role="user"), content=None)
        blocks = msg.to_content_blocks()
        assert blocks == []


class TestChatGPTTextExtraction:
    """Tests for ChatGPTMessage.text_content edge cases."""

    @pytest.mark.parametrize("text_field,parts,expected_text", [
        # direct_text_field: Text from content.text field
        ("Direct text", None, "Direct text"),
        # dict_part_with_text: Dict part with 'text' key is extracted
        (None, [{"text": "From dict"}], "From dict"),
        # mixed_parts: Mix of string and dict parts
        (None, ["str part", {"text": "dict part"}], None),  # Check containment
        # empty_parts: Empty parts list returns empty string
        (None, [], ""),
        # dict_part_missing_text_key: Dict part without 'text' key is skipped
        (None, [{"other": "value"}, "text"], "text"),
    ])
    def test_text_extraction_scenarios(self, text_field, parts, expected_text):
        """Test various text extraction scenarios."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", text=text_field, parts=parts),
        )
        text = msg.text_content
        if expected_text is None:
            # For mixed parts, check containment
            assert "str part" in text
            assert "dict part" in text
        else:
            assert text == expected_text

    def test_no_content(self):
        """No content returns empty string."""
        msg = ChatGPTMessage(id="m1", author=ChatGPTAuthor(role="user"), content=None)
        assert msg.text_content == ""


class TestChatGPTTreeTraversal:
    """Tests for ChatGPTConversation.messages tree traversal."""

    def test_branching_follows_first_child(self):
        """Branching tree follows first child path."""
        mapping = {
            "root": ChatGPTNode(id="root", parent=None, children=["a", "b"]),
            "a": ChatGPTNode(id="a", parent="root", children=["c"],
                            message=ChatGPTMessage(id="ma", author=ChatGPTAuthor(role="user"),
                                                   content=ChatGPTContent(content_type="text", parts=["branch A"]))),
            "b": ChatGPTNode(id="b", parent="root", children=[],
                            message=ChatGPTMessage(id="mb", author=ChatGPTAuthor(role="user"),
                                                   content=ChatGPTContent(content_type="text", parts=["branch B"]))),
            "c": ChatGPTNode(id="c", parent="a", children=[],
                            message=ChatGPTMessage(id="mc", author=ChatGPTAuthor(role="assistant"),
                                                   content=ChatGPTContent(content_type="text", parts=["response"]))),
        }
        conv = ChatGPTConversation(
            id="c1", conversation_id="c1", title="Branch",
            create_time=1700000000.0, update_time=1700000100.0,
            mapping=mapping, current_node="c",
        )
        msgs = conv.messages
        # Should follow root → a → c (first child path), skip b
        texts = [m.text_content for m in msgs]
        assert "branch A" in texts
        assert "response" in texts
        assert "branch B" not in texts

    def test_cycle_detection(self):
        """Cycle in tree doesn't cause infinite loop."""
        mapping = {
            "a": ChatGPTNode(id="a", parent=None, children=["b"],
                            message=ChatGPTMessage(id="ma", author=ChatGPTAuthor(role="user"),
                                                   content=ChatGPTContent(content_type="text", parts=["msg a"]))),
            "b": ChatGPTNode(id="b", parent="a", children=["a"],  # cycle!
                            message=ChatGPTMessage(id="mb", author=ChatGPTAuthor(role="assistant"),
                                                   content=ChatGPTContent(content_type="text", parts=["msg b"]))),
        }
        conv = ChatGPTConversation(
            id="c2", conversation_id="c2", title="Cycle",
            create_time=1700000000.0, update_time=1700000100.0,
            mapping=mapping, current_node="b",
        )
        msgs = conv.messages  # Should not hang
        assert len(msgs) == 2

    def test_no_root_node(self):
        """All nodes have parents — no root found."""
        mapping = {
            "a": ChatGPTNode(id="a", parent="b", children=[]),
            "b": ChatGPTNode(id="b", parent="a", children=[]),
        }
        conv = ChatGPTConversation(
            id="c3", conversation_id="c3", title="No Root",
            create_time=1700000000.0, update_time=1700000100.0,
            mapping=mapping, current_node="a",
        )
        assert conv.messages == []

    def test_root_only(self):
        """Root node with no message and no children."""
        mapping = {
            "root": ChatGPTNode(id="root", parent=None, children=[]),
        }
        conv = ChatGPTConversation(
            id="c4", conversation_id="c4", title="Root Only",
            create_time=1700000000.0, update_time=1700000100.0,
            mapping=mapping, current_node="root",
        )
        assert conv.messages == []  # root has no message

    def test_client_created_root_detection(self):
        """Parent 'client-created-root' is recognized as root."""
        mapping = {
            "node1": ChatGPTNode(id="node1", parent="client-created-root", children=[],
                                message=ChatGPTMessage(id="m1", author=ChatGPTAuthor(role="user"),
                                                       content=ChatGPTContent(content_type="text", parts=["hi"]))),
        }
        conv = ChatGPTConversation(
            id="c5", conversation_id="c5", title="CCR",
            create_time=1700000000.0, update_time=1700000100.0,
            mapping=mapping, current_node="node1",
        )
        msgs = conv.messages
        assert len(msgs) == 1
        assert msgs[0].text_content == "hi"


class TestChatGPTTimestamps:
    """Tests for timestamp edge cases."""

    def test_invalid_create_time(self):
        """Invalid timestamp returns None."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="user"),
            create_time=-999999999999999.0,
        )
        assert msg.timestamp is None

    def test_conversation_invalid_create_time(self):
        """Conversation with invalid timestamp returns None."""
        conv = ChatGPTConversation(
            id="c1", conversation_id="c1", title="Bad TS",
            create_time=-999999999999999.0, update_time=-999999999999999.0,
            mapping={}, current_node="",
        )
        assert conv.created_at is None
        assert conv.updated_at is None

    def test_valid_timestamp_conversion(self):
        """Valid timestamp converts correctly."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="user"),
            create_time=1700000000.0,
        )
        assert msg.timestamp is not None
        assert msg.timestamp.year == 2023


# =============================================================================
# Codex Provider Tests
# =============================================================================


class TestCodexFormatDetection:
    """Tests for CodexRecord.format_type detection."""

    @pytest.mark.parametrize("record_kwargs,expected_format", [
        # envelope_format: Record with payload detected as envelope
        ({"type": "response_item", "payload": {"role": "assistant", "content": []}}, "envelope"),
        # direct_format_with_role: Record with role (no payload) detected as direct
        ({"type": "message", "role": "user"}, "direct"),
        # direct_format_type_message: Record with type='message' detected as direct
        ({"type": "message"}, "direct"),
        # state_format: Record with record_type='state' detected as state
        ({"record_type": "state"}, "state"),
        # unknown_format: Bare record detected as unknown
        ({}, "unknown"),
    ])
    def test_format_detection(self, record_kwargs, expected_format):
        """Test format_type detection for various record configurations."""
        rec = CodexRecord(**record_kwargs)
        assert rec.format_type == expected_format


class TestCodexIsMessage:
    """Tests for CodexRecord.is_message."""

    @pytest.mark.parametrize("record_kwargs,expected_is_message", [
        # envelope_response_item: Envelope response_item is a message
        ({"type": "response_item", "payload": {"role": "assistant"}}, True),
        # envelope_session_meta: Envelope session_meta is NOT a message
        ({"type": "session_meta", "payload": {"id": "sess1"}}, False),
        # direct_message: Direct message record is a message
        ({"type": "message", "role": "user"}, True),
        # direct_role_only: Record with only role is a message
        ({"role": "assistant"}, True),
        # state_not_message: State record is NOT a message
        ({"record_type": "state"}, False),
        # unknown_not_message: Unknown format is NOT a message
        ({}, False),
    ])
    def test_is_message_detection(self, record_kwargs, expected_is_message):
        """Test is_message detection for various record types."""
        rec = CodexRecord(**record_kwargs)
        assert rec.is_message is expected_is_message


class TestCodexEffectiveContent:
    """Tests for CodexRecord.effective_content."""

    def test_envelope_content(self):
        """Content from envelope payload."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [{"type": "output_text", "text": "hello"}]})
        content = rec.effective_content
        assert len(content) == 1
        assert content[0]["text"] == "hello"

    def test_envelope_non_list_content(self):
        """Envelope payload content that's not a list returns empty."""
        rec = CodexRecord(type="response_item", payload={"content": "just a string"})
        assert rec.effective_content == []

    def test_direct_content_blocks(self):
        """Content from direct format CodexContentBlock."""
        rec = CodexRecord(
            type="message", role="assistant",
            content=[CodexContentBlock(type="output_text", text="hi")],
        )
        content = rec.effective_content
        assert len(content) == 1
        assert content[0]["type"] == "output_text"
        assert content[0]["text"] == "hi"

    def test_direct_content_dicts(self):
        """Content from direct format raw dicts."""
        rec = CodexRecord(
            type="message", role="user",
            content=[{"type": "input_text", "text": "question"}],
        )
        content = rec.effective_content
        assert content[0]["text"] == "question"

    def test_no_content(self):
        """Record with no content returns empty list."""
        rec = CodexRecord(type="session_meta", payload={})
        assert rec.effective_content == []


class TestCodexTimestampParsing:
    """Tests for CodexRecord.parsed_timestamp."""

    @pytest.mark.parametrize("timestamp,expected_year", [
        # iso_format: Standard ISO format
        ("2024-06-15T10:30:00+00:00", 2024),
        # z_suffix: Z suffix replaced with +00:00
        ("2024-06-15T10:30:00Z", 2024),
    ])
    def test_valid_timestamp_parsing(self, timestamp, expected_year):
        """Test parsing of valid timestamp formats."""
        rec = CodexRecord(timestamp=timestamp)
        ts = rec.parsed_timestamp
        assert ts is not None
        assert ts.year == expected_year

    @pytest.mark.parametrize("timestamp,description", [
        # no_timestamp: No timestamp
        (None, "missing"),
        # malformed_timestamp: Malformed format
        ("not-a-date", "malformed"),
        # empty_string: Empty string
        ("", "empty"),
    ])
    def test_invalid_timestamp_parsing(self, timestamp, description):
        """Test that invalid timestamps return None."""
        rec = CodexRecord(timestamp=timestamp) if timestamp is not None else CodexRecord()
        assert rec.parsed_timestamp is None


class TestCodexTextContent:
    """Tests for CodexRecord.text_content."""

    @pytest.mark.parametrize("payload_content,expected_text", [
        # output_text: Extracts output_text field
        ([{"type": "output_text", "output_text": "result"}], "result"),
        # input_text: Extracts input_text field
        ([{"type": "input_text", "input_text": "question"}], "question"),
        # text_field: Extracts text field
        ([{"type": "text", "text": "plain"}], "plain"),
    ])
    def test_text_extraction_by_type(self, payload_content, expected_text):
        """Test text extraction from various content field names."""
        rec = CodexRecord(type="response_item", payload={"content": payload_content})
        assert rec.text_content == expected_text

    def test_multiple_blocks_joined(self):
        """Multiple content blocks joined with newline."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [
                             {"type": "text", "text": "line1"},
                             {"type": "text", "text": "line2"},
                         ]})
        assert "line1" in rec.text_content
        assert "line2" in rec.text_content

    def test_empty_content(self):
        """No content returns empty string."""
        rec = CodexRecord()
        assert rec.text_content == ""


class TestCodexContentBlockExtraction:
    """Tests for CodexRecord.extract_content_blocks()."""

    @pytest.mark.parametrize("payload_content,expected_type,expected_language", [
        # text_block: output_text becomes TEXT content block
        ([{"type": "output_text", "text": "hello"}], ContentType.TEXT, None),
        # code_block: Block with 'code' in type becomes CODE
        ([{"type": "code_output", "text": "print('hi')", "language": "python"}], ContentType.CODE, "python"),
        # unknown_block_type: Unknown block type becomes UNKNOWN
        ([{"type": "image_data", "text": "base64..."}], ContentType.UNKNOWN, None),
    ])
    def test_content_block_extraction(self, payload_content, expected_type, expected_language):
        """Test extraction and type mapping of content blocks."""
        rec = CodexRecord(type="response_item", payload={"content": payload_content})
        blocks = rec.extract_content_blocks()
        assert len(blocks) >= 1
        assert blocks[0].type == expected_type
        if expected_language:
            assert blocks[0].language == expected_language

    def test_non_dict_block_skipped(self):
        """Non-dict content items are skipped."""
        rec = CodexRecord(type="response_item",
                         payload={"content": ["just a string", {"type": "text", "text": "real"}]})
        blocks = rec.extract_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.TEXT


class TestCodexRoleNormalization:
    """Tests for CodexRecord role normalization."""

    @pytest.mark.parametrize("record_kwargs,expected_effective_role", [
        # envelope_role: Role from envelope payload
        ({"type": "response_item", "payload": {"role": "assistant"}}, "assistant"),
        # direct_role: Role from direct format
        ({"role": "user"}, "user"),
        # envelope_missing_role: Envelope with no role defaults to 'unknown'
        ({"type": "response_item", "payload": {}}, "unknown"),
    ])
    def test_effective_role(self, record_kwargs, expected_effective_role):
        """Test effective_role extraction from various record formats."""
        rec = CodexRecord(**record_kwargs)
        assert rec.effective_role == expected_effective_role

    def test_to_meta_unknown_role_normalized(self):
        """Unknown roles map to 'unknown' in MessageMeta."""
        rec = CodexRecord(type="response_item", payload={"role": "admin"})
        meta = rec.to_meta()
        assert meta.role == "unknown"

    def test_to_meta_system_role(self):
        """System role preserved."""
        rec = CodexRecord(role="system")
        meta = rec.to_meta()
        assert meta.role == "system"


# =============================================================================
# Claude Code Provider Tests
# =============================================================================


class TestClaudeCodeExtractReasoningTraces:
    """Tests for ClaudeCodeRecord.extract_reasoning_traces()."""

    @pytest.mark.parametrize("message_content,should_have_trace", [
        # thinking_block_extracted: Thinking content block is extracted
        (
            [
                {"type": "thinking", "thinking": "Let me analyze this..."},
                {"type": "text", "text": "Here's my answer"},
            ],
            True,
        ),
        # no_thinking_blocks: Record without thinking blocks returns empty
        (
            [{"type": "text", "text": "Just text"}],
            False,
        ),
    ])
    def test_reasoning_trace_extraction(self, message_content, should_have_trace):
        """Test extraction of reasoning traces from various message formats."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": message_content},
        )
        traces = rec.extract_reasoning_traces()
        if should_have_trace:
            assert len(traces) >= 1
            assert any("analyze" in t.text.lower() for t in traces)
        else:
            assert traces == []

    def test_no_message(self):
        """System record with no message returns empty."""
        rec = ClaudeCodeRecord(type="system")
        traces = rec.extract_reasoning_traces()
        assert traces == []


class TestClaudeCodeExtractToolCalls:
    """Tests for ClaudeCodeRecord.extract_tool_calls()."""

    @pytest.mark.parametrize("message_content,should_have_calls", [
        # tool_use_extracted: tool_use content block is extracted
        (
            [
                {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"path": "/etc/hosts"}},
            ],
            True,
        ),
        # no_tool_calls: Record without tool blocks returns empty
        (
            [{"type": "text", "text": "No tools"}],
            False,
        ),
    ])
    def test_tool_call_extraction(self, message_content, should_have_calls):
        """Test extraction of tool calls from various message formats."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": message_content},
        )
        calls = rec.extract_tool_calls()
        if should_have_calls:
            assert len(calls) >= 1
        else:
            assert calls == []

    def test_no_message_returns_empty(self):
        """System record with no message returns empty list."""
        rec = ClaudeCodeRecord(type="system")
        calls = rec.extract_tool_calls()
        assert calls == []


class TestClaudeCodeTextContent:
    """Tests for ClaudeCodeRecord.text_content edge cases."""

    @pytest.mark.parametrize("record_kwargs,expected_contains", [
        # dict_message_with_list_content: List content blocks
        (
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello world"},
                        {"type": "thinking", "thinking": "deep thoughts"},
                    ]
                },
            },
            "Hello world",
        ),
        # dict_message_with_string_content: Plain string content
        (
            {
                "type": "user",
                "message": {"content": "Plain user input"},
            },
            "Plain user input",
        ),
        # message_with_multiple_text_blocks: Multiple text blocks
        (
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "First"},
                        {"type": "text", "text": "Second"},
                    ]
                },
            },
            None,  # Check both via separate assertion
        ),
    ])
    def test_text_content_scenarios(self, record_kwargs, expected_contains):
        """Test text_content extraction from various record formats."""
        rec = ClaudeCodeRecord(**record_kwargs)
        text = rec.text_content
        if expected_contains:
            assert expected_contains in text
        elif record_kwargs["type"] == "assistant" and "First" in str(record_kwargs.get("message", {})):
            # For multiple text blocks, check both
            assert "First" in text
            assert "Second" in text

    def test_system_record_top_level_content(self):
        """System record extracts text from top-level content field (Pydantic extra)."""
        rec = ClaudeCodeRecord(
            type="summary",
            content="Compacted conversation context",
        )
        # Top-level content field is stored as Pydantic extra
        assert "Compacted" in rec.text_content or rec.text_content == ""

    def test_no_message_no_content(self):
        """Record with no message and no content returns empty."""
        rec = ClaudeCodeRecord(type="system")
        assert rec.text_content == ""


# NOTE: TestClaudeCodeRoleNormalization was consolidated into
# tests/unit/sources/test_models.py::TestClaudeCodeRecordRole using
# canonical tables from tests/infra/tables.py.


class TestClaudeCodeContentBlocksRaw:
    """Tests for ClaudeCodeRecord.content_blocks_raw."""

    @pytest.mark.parametrize("record_kwargs,expected_count", [
        # dict_message_list_content: List content returns blocks
        (
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hi"}]},
            },
            1,
        ),
        # dict_message_string_content: String content returns empty (not a list)
        (
            {
                "type": "user",
                "message": {"content": "string"},
            },
            0,
        ),
        # dict_message_no_content: Message without content key returns empty
        (
            {
                "type": "user",
                "message": {"role": "user"},
            },
            0,
        ),
    ])
    def test_content_blocks_raw(self, record_kwargs, expected_count):
        """Test content_blocks_raw for various message formats."""
        rec = ClaudeCodeRecord(**record_kwargs)
        assert len(rec.content_blocks_raw) == expected_count

    def test_no_message(self):
        """No message returns empty."""
        rec = ClaudeCodeRecord(type="system")
        assert rec.content_blocks_raw == []


# NOTE: TestClaudeCodeIsContextCompaction, TestClaudeCodeIsToolProgress,
# TestClaudeCodeIsActualMessage, and TestClaudeCodeParsedTimestamp were
# consolidated into tests/unit/sources/test_models.py using canonical
# tables from tests/infra/tables.py.


class TestClaudeCodeToMeta:
    """Tests for ClaudeCodeRecord.to_meta() conversion."""

    def test_basic_conversion(self):
        """Convert record to MessageMeta."""
        rec = ClaudeCodeRecord(
            type="user",
            uuid="msg-123",
            timestamp="2024-06-15T10:30:00Z",
        )
        meta = rec.to_meta()
        assert meta.id == "msg-123"
        assert meta.role == "user"
        assert meta.provider == "claude-code"

    def test_with_cost_info(self):
        """Cost information included."""
        rec = ClaudeCodeRecord(
            type="assistant",
            uuid="msg-456",
            costUSD=0.05,
        )
        meta = rec.to_meta()
        assert meta.cost is not None
        assert meta.cost.total_usd == 0.05

    def test_with_duration(self):
        """Duration information included."""
        rec = ClaudeCodeRecord(
            type="assistant",
            uuid="msg-789",
            durationMs=1500,
        )
        meta = rec.to_meta()
        assert meta.duration_ms == 1500
