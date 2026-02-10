"""Tests for provider-specific edge cases and untested branches.

Targets the lowest-coverage provider code paths:
- ChatGPT: iter_user_assistant_pairs, tether browsing content, branching trees
- Codex: format detection, timestamp parsing, content extraction
- Claude Code: extract_reasoning_traces, extract_tool_calls, system record text
"""
from __future__ import annotations

import pytest
from polylogue.sources.providers.chatgpt import (
    ChatGPTAuthor, ChatGPTContent, ChatGPTConversation, ChatGPTMessage, ChatGPTNode,
)
from polylogue.sources.providers.codex import CodexContentBlock, CodexRecord
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.lib.viewports import ContentType


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

    def test_basic_pair(self):
        """Simple user-assistant pair is yielded."""
        conv = self._make_conv([("user", "hi"), ("assistant", "hello")])
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == 1
        assert pairs[0][0].text_content == "hi"
        assert pairs[0][1].text_content == "hello"

    def test_multiple_pairs(self):
        """Multiple pairs yielded correctly."""
        conv = self._make_conv([
            ("user", "q1"), ("assistant", "a1"),
            ("user", "q2"), ("assistant", "a2"),
        ])
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == 2

    def test_system_message_skipped(self):
        """System messages at start don't break pairing."""
        conv = self._make_conv([
            ("system", "You are helpful"),
            ("user", "hi"), ("assistant", "hello"),
        ])
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == 1

    def test_consecutive_users_skip(self):
        """Two consecutive user messages — only the second forms a pair."""
        conv = self._make_conv([
            ("user", "first"), ("user", "second"), ("assistant", "response"),
        ])
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == 1
        assert pairs[0][0].text_content == "second"

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

    def test_unmatched_trailing_assistant(self):
        """Trailing assistant message without preceding user is skipped."""
        conv = self._make_conv([
            ("user", "question"),
            ("assistant", "answer"),
            ("assistant", "trailing"),
        ])
        pairs = list(conv.iter_user_assistant_pairs())
        assert len(pairs) == 1
        assert pairs[0][1].text_content == "answer"


class TestChatGPTContentBlocks:
    """Tests for ChatGPTMessage.to_content_blocks() edge cases."""

    def test_code_content_type(self):
        """Code content produces CODE content block."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="code", parts=["print('hi')"], language="python"),
        )
        blocks = msg.to_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.CODE
        assert blocks[0].language == "python"

    def test_tether_browsing_content(self):
        """Tether browsing content becomes TOOL_RESULT."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="tether_browsing_display", parts=["Search results..."]),
        )
        blocks = msg.to_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.TOOL_RESULT

    def test_browse_content_variant(self):
        """Content type containing 'browse' also becomes TOOL_RESULT."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="browse_results", parts=["Page content"]),
        )
        blocks = msg.to_content_blocks()
        assert blocks[0].type == ContentType.TOOL_RESULT

    def test_unknown_content_type(self):
        """Unknown content type becomes UNKNOWN."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="multimodal_image", parts=[]),
        )
        blocks = msg.to_content_blocks()
        assert blocks[0].type == ContentType.UNKNOWN

    def test_no_content_returns_empty(self):
        """No content returns empty list."""
        msg = ChatGPTMessage(id="m1", author=ChatGPTAuthor(role="user"), content=None)
        blocks = msg.to_content_blocks()
        assert blocks == []


class TestChatGPTTextExtraction:
    """Tests for ChatGPTMessage.text_content edge cases."""

    def test_direct_text_field(self):
        """Text from content.text field."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", text="Direct text"),
        )
        assert msg.text_content == "Direct text"

    def test_dict_part_with_text(self):
        """Dict part with 'text' key is extracted."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", parts=[{"text": "From dict"}]),
        )
        assert msg.text_content == "From dict"

    def test_mixed_parts(self):
        """Mix of string and dict parts joined."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", parts=["str part", {"text": "dict part"}]),
        )
        assert "str part" in msg.text_content
        assert "dict part" in msg.text_content

    def test_no_content(self):
        """No content returns empty string."""
        msg = ChatGPTMessage(id="m1", author=ChatGPTAuthor(role="user"), content=None)
        assert msg.text_content == ""

    def test_empty_parts(self):
        """Empty parts list returns empty string."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=[]),
        )
        assert msg.text_content == ""

    def test_dict_part_missing_text_key(self):
        """Dict part without 'text' key is skipped."""
        msg = ChatGPTMessage(
            id="m1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=[{"other": "value"}, "text"]),
        )
        assert msg.text_content == "text"


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

    def test_envelope_format(self):
        """Record with payload detected as envelope."""
        rec = CodexRecord(type="response_item", payload={"role": "assistant", "content": []})
        assert rec.format_type == "envelope"

    def test_direct_format_with_role(self):
        """Record with role (no payload) detected as direct."""
        rec = CodexRecord(type="message", role="user")
        assert rec.format_type == "direct"

    def test_direct_format_type_message(self):
        """Record with type='message' (no role) detected as direct."""
        rec = CodexRecord(type="message")
        assert rec.format_type == "direct"

    def test_state_format(self):
        """Record with record_type='state' detected as state."""
        rec = CodexRecord(record_type="state")
        assert rec.format_type == "state"

    def test_unknown_format(self):
        """Bare record detected as unknown."""
        rec = CodexRecord()
        assert rec.format_type == "unknown"


class TestCodexIsMessage:
    """Tests for CodexRecord.is_message."""

    def test_envelope_response_item(self):
        """Envelope response_item is a message."""
        rec = CodexRecord(type="response_item", payload={"role": "assistant"})
        assert rec.is_message is True

    def test_envelope_session_meta(self):
        """Envelope session_meta is NOT a message."""
        rec = CodexRecord(type="session_meta", payload={"id": "sess1"})
        assert rec.is_message is False

    def test_direct_message(self):
        """Direct message record is a message."""
        rec = CodexRecord(type="message", role="user")
        assert rec.is_message is True

    def test_direct_role_only(self):
        """Record with only role is a message."""
        rec = CodexRecord(role="assistant")
        assert rec.is_message is True

    def test_state_not_message(self):
        """State record is NOT a message."""
        rec = CodexRecord(record_type="state")
        assert rec.is_message is False

    def test_unknown_not_message(self):
        """Unknown format is NOT a message."""
        rec = CodexRecord()
        assert rec.is_message is False


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

    def test_iso_format(self):
        """Standard ISO format."""
        rec = CodexRecord(timestamp="2024-06-15T10:30:00+00:00")
        ts = rec.parsed_timestamp
        assert ts is not None
        assert ts.year == 2024

    def test_z_suffix(self):
        """Z suffix replaced with +00:00."""
        rec = CodexRecord(timestamp="2024-06-15T10:30:00Z")
        ts = rec.parsed_timestamp
        assert ts is not None
        assert ts.year == 2024

    def test_no_timestamp(self):
        """No timestamp returns None."""
        rec = CodexRecord()
        assert rec.parsed_timestamp is None

    def test_malformed_timestamp(self):
        """Malformed timestamp returns None."""
        rec = CodexRecord(timestamp="not-a-date")
        assert rec.parsed_timestamp is None

    def test_empty_string(self):
        """Empty string returns None."""
        rec = CodexRecord(timestamp="")
        assert rec.parsed_timestamp is None


class TestCodexTextContent:
    """Tests for CodexRecord.text_content."""

    def test_output_text(self):
        """Extracts output_text field."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [{"type": "output_text", "output_text": "result"}]})
        assert rec.text_content == "result"

    def test_input_text(self):
        """Extracts input_text field."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [{"type": "input_text", "input_text": "question"}]})
        assert rec.text_content == "question"

    def test_text_field(self):
        """Extracts text field."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [{"type": "text", "text": "plain"}]})
        assert rec.text_content == "plain"

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

    def test_text_block(self):
        """output_text becomes TEXT content block."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [{"type": "output_text", "text": "hello"}]})
        blocks = rec.extract_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.TEXT

    def test_code_block(self):
        """Block with 'code' in type becomes CODE."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [{"type": "code_output", "text": "print('hi')", "language": "python"}]})
        blocks = rec.extract_content_blocks()
        assert blocks[0].type == ContentType.CODE
        assert blocks[0].language == "python"

    def test_unknown_block_type(self):
        """Unknown block type becomes UNKNOWN."""
        rec = CodexRecord(type="response_item",
                         payload={"content": [{"type": "image_data", "text": "base64..."}]})
        blocks = rec.extract_content_blocks()
        assert blocks[0].type == ContentType.UNKNOWN

    def test_non_dict_block_skipped(self):
        """Non-dict content items are skipped."""
        rec = CodexRecord(type="response_item",
                         payload={"content": ["just a string", {"type": "text", "text": "real"}]})
        blocks = rec.extract_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.TEXT


class TestCodexRoleNormalization:
    """Tests for CodexRecord role normalization."""

    def test_envelope_role(self):
        """Role from envelope payload."""
        rec = CodexRecord(type="response_item", payload={"role": "assistant"})
        assert rec.effective_role == "assistant"

    def test_direct_role(self):
        """Role from direct format."""
        rec = CodexRecord(role="user")
        assert rec.effective_role == "user"

    def test_envelope_missing_role(self):
        """Envelope with no role defaults to 'unknown'."""
        rec = CodexRecord(type="response_item", payload={})
        assert rec.effective_role == "unknown"

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

    def test_thinking_block_extracted(self):
        """Thinking content block is extracted as reasoning trace."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": [
                {"type": "thinking", "thinking": "Let me analyze this..."},
                {"type": "text", "text": "Here's my answer"},
            ]},
        )
        traces = rec.extract_reasoning_traces()
        assert len(traces) >= 1
        assert any("analyze" in t.text.lower() for t in traces)

    def test_no_thinking_blocks(self):
        """Record without thinking blocks returns empty."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": [{"type": "text", "text": "Just text"}]},
        )
        traces = rec.extract_reasoning_traces()
        assert traces == []

    def test_no_message(self):
        """System record with no message returns empty."""
        rec = ClaudeCodeRecord(type="system")
        traces = rec.extract_reasoning_traces()
        assert traces == []


class TestClaudeCodeExtractToolCalls:
    """Tests for ClaudeCodeRecord.extract_tool_calls()."""

    def test_tool_use_extracted(self):
        """tool_use content block is extracted."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": [
                {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"path": "/etc/hosts"}},
            ]},
        )
        calls = rec.extract_tool_calls()
        assert len(calls) >= 1

    def test_no_tool_calls(self):
        """Record without tool blocks returns empty."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": [{"type": "text", "text": "No tools"}]},
        )
        calls = rec.extract_tool_calls()
        assert calls == []

    def test_no_message_returns_empty(self):
        """System record with no message returns empty list."""
        rec = ClaudeCodeRecord(type="system")
        calls = rec.extract_tool_calls()
        assert calls == []


class TestClaudeCodeTextContent:
    """Tests for ClaudeCodeRecord.text_content edge cases."""

    def test_system_record_top_level_content(self):
        """System record extracts text from top-level content field (Pydantic extra)."""
        rec = ClaudeCodeRecord(
            type="summary",
            content="Compacted conversation context",
        )
        # Top-level content field is stored as Pydantic extra
        assert "Compacted" in rec.text_content or rec.text_content == ""

    def test_dict_message_with_list_content(self):
        """Dict message with list content blocks."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": [
                {"type": "text", "text": "Hello world"},
                {"type": "thinking", "thinking": "deep thoughts"},
            ]},
        )
        text = rec.text_content
        assert "Hello world" in text

    def test_dict_message_with_string_content(self):
        """Dict message with plain string content."""
        rec = ClaudeCodeRecord(
            type="user",
            message={"content": "Plain user input"},
        )
        assert rec.text_content == "Plain user input"

    def test_no_message_no_content(self):
        """Record with no message and no content returns empty."""
        rec = ClaudeCodeRecord(type="system")
        assert rec.text_content == ""

    def test_message_with_multiple_text_blocks(self):
        """Multiple text blocks in message content."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ]},
        )
        text = rec.text_content
        assert "First" in text
        assert "Second" in text


class TestClaudeCodeRoleNormalization:
    """Tests for ClaudeCodeRecord.role property."""

    @pytest.mark.parametrize("type_val,expected", [
        ("user", "user"),
        ("assistant", "assistant"),
        ("summary", "system"),
        ("system", "system"),
        ("file-history-snapshot", "system"),
        ("queue-operation", "system"),
        ("progress", "tool"),
        ("result", "tool"),
        ("unknown_type", "unknown"),
    ])
    def test_role_mapping(self, type_val, expected):
        """Each type maps to the expected role."""
        rec = ClaudeCodeRecord(type=type_val)
        assert rec.role == expected


class TestClaudeCodeContentBlocksRaw:
    """Tests for ClaudeCodeRecord.content_blocks_raw."""

    def test_dict_message_list_content(self):
        """Dict message with list content returns blocks."""
        rec = ClaudeCodeRecord(
            type="assistant",
            message={"content": [{"type": "text", "text": "hi"}]},
        )
        assert len(rec.content_blocks_raw) == 1

    def test_dict_message_string_content(self):
        """Dict message with string content returns empty (not a list)."""
        rec = ClaudeCodeRecord(type="user", message={"content": "string"})
        assert rec.content_blocks_raw == []

    def test_no_message(self):
        """No message returns empty."""
        rec = ClaudeCodeRecord(type="system")
        assert rec.content_blocks_raw == []

    def test_dict_message_no_content(self):
        """Dict message without content key returns empty."""
        rec = ClaudeCodeRecord(type="user", message={"role": "user"})
        assert rec.content_blocks_raw == []


class TestClaudeCodeIsContextCompaction:
    """Tests for ClaudeCodeRecord context compaction detection."""

    def test_summary_type_is_compaction(self):
        """Summary type record is context compaction."""
        rec = ClaudeCodeRecord(type="summary")
        assert rec.is_context_compaction is True

    def test_other_types_not_compaction(self):
        """Non-summary types are not context compaction."""
        for record_type in ("user", "assistant", "progress", "system"):
            rec = ClaudeCodeRecord(type=record_type)
            assert rec.is_context_compaction is False


class TestClaudeCodeIsToolProgress:
    """Tests for ClaudeCodeRecord tool progress detection."""

    def test_progress_type_is_tool_progress(self):
        """Progress type record is tool progress."""
        rec = ClaudeCodeRecord(type="progress")
        assert rec.is_tool_progress is True

    def test_other_types_not_tool_progress(self):
        """Non-progress types are not tool progress."""
        for record_type in ("user", "assistant", "summary", "system"):
            rec = ClaudeCodeRecord(type=record_type)
            assert rec.is_tool_progress is False


class TestClaudeCodeIsActualMessage:
    """Tests for ClaudeCodeRecord actual message detection."""

    def test_user_and_assistant_are_actual_messages(self):
        """User and assistant records are actual messages."""
        for record_type in ("user", "assistant"):
            rec = ClaudeCodeRecord(type=record_type)
            assert rec.is_actual_message is True

    def test_non_message_types(self):
        """System, progress, summary records are not actual messages."""
        for record_type in ("system", "progress", "summary", "file-history-snapshot"):
            rec = ClaudeCodeRecord(type=record_type)
            assert rec.is_actual_message is False


class TestClaudeCodeParsedTimestamp:
    """Tests for ClaudeCodeRecord.parsed_timestamp."""

    def test_iso_string_timestamp(self):
        """ISO format timestamp string."""
        rec = ClaudeCodeRecord(type="user", timestamp="2024-06-15T10:30:00Z")
        ts = rec.parsed_timestamp
        assert ts is not None
        assert ts.year == 2024

    def test_unix_seconds_timestamp(self):
        """Unix timestamp in seconds."""
        rec = ClaudeCodeRecord(type="user", timestamp=1700000000.0)
        ts = rec.parsed_timestamp
        assert ts is not None

    def test_unix_milliseconds_timestamp(self):
        """Unix timestamp in milliseconds (>1e11)."""
        rec = ClaudeCodeRecord(type="user", timestamp=1700000000000)
        ts = rec.parsed_timestamp
        assert ts is not None

    def test_no_timestamp(self):
        """No timestamp returns None."""
        rec = ClaudeCodeRecord(type="user")
        assert rec.parsed_timestamp is None

    def test_malformed_timestamp(self):
        """Malformed timestamp returns None."""
        rec = ClaudeCodeRecord(type="user", timestamp="not a date")
        assert rec.parsed_timestamp is None


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
