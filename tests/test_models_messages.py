"""Comprehensive coverage tests for models.py and messages.py uncovered lines."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic_core import core_schema

from polylogue.lib.messages import MessageCollection, MessageSource
from polylogue.lib.models import (
    Attachment,
    Conversation,
    ConversationSummary,
    DialoguePair,
    Message,
    ToolInvocation,
)
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord


# =============================================================================
# TOOLINVOCATION COVERAGE
# =============================================================================


class TestToolInvocationFileOperation:
    """Test ToolInvocation.is_file_operation property."""

    @pytest.mark.parametrize("tool_name,expected", [
        ("Read", True),
        ("Write", True),
        ("Edit", True),
        ("NotebookEdit", True),
        ("Bash", False),
    ])
    def test_is_file_operation(self, tool_name, expected):
        """Line 93: Check is_file_operation with various tools."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input={})
        assert tool.is_file_operation is expected


class TestToolInvocationGitOperation:
    """Test ToolInvocation.is_git_operation property (lines 98-101)."""

    @pytest.mark.parametrize("tool_name,input_data,expected", [
        ("Bash", {"command": "git commit -m 'test'"}, True),
        ("Read", {"command": "git status"}, False),
        ("Bash", {"command": "ls -la"}, False),
        ("Bash", {"command": 123}, False),
        ("Bash", {"command": "  git push  "}, True),
        ("Bash", {}, False),
    ], ids=["git_command", "not_bash", "non_git_bash", "non_string_cmd", "whitespace_git", "no_command"])
    def test_is_git_operation(self, tool_name, input_data, expected):
        """Line 98-101: Test is_git_operation with various inputs."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input=input_data)
        assert tool.is_git_operation is expected


class TestToolInvocationSearchOperation:
    """Test ToolInvocation.is_search_operation property (line 106)."""

    @pytest.mark.parametrize("tool_name,expected", [
        ("Glob", True),
        ("Grep", True),
        ("WebSearch", True),
        ("Bash", False),
    ])
    def test_is_search_operation(self, tool_name, expected):
        """Line 106: Test is_search_operation with various tools."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input={})
        assert tool.is_search_operation is expected


class TestToolInvocationSubagent:
    """Test ToolInvocation.is_subagent property (line 111)."""

    @pytest.mark.parametrize("tool_name,expected", [
        ("Task", True),
        ("Bash", False),
    ])
    def test_is_subagent(self, tool_name, expected):
        """Line 111: Test is_subagent with various tools."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input={})
        assert tool.is_subagent is expected


class TestToolInvocationAffectedPaths:
    """Test ToolInvocation.affected_paths property (lines 116-137)."""

    @pytest.mark.parametrize("tool_name,input_data,expected", [
        ("Read", {"file_path": "/tmp/test.txt"}, ["/tmp/test.txt"]),
        ("Write", {"file_path": "/tmp/output.txt"}, ["/tmp/output.txt"]),
        ("Edit", {"file_path": "/tmp/code.py"}, ["/tmp/code.py"]),
        ("Read", {"path": "/tmp/fallback.txt"}, ["/tmp/fallback.txt"]),
        ("Read", {"file_path": "/tmp/primary.txt", "path": "/tmp/fallback.txt"}, ["/tmp/primary.txt"]),
        ("Read", {"file_path": 123}, []),
        ("Glob", {"pattern": "**/*.py"}, ["**/*.py"]),
        ("Glob", {"pattern": ["*.py", "*.txt"]}, []),
        ("Bash", {"command": 123}, []),
        ("Task", {"prompt": "do something"}, []),
    ], ids=["read", "write", "edit", "path_fallback", "file_path_priority", "non_string_path",
            "glob", "glob_non_string", "bash_non_string", "other_tool"])
    def test_affected_paths(self, tool_name, input_data, expected):
        """Lines 116-137: Test affected_paths with various tools."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input=input_data)
        assert tool.affected_paths == expected

    def test_affected_paths_bash_extraction(self):
        """Lines 128-135: Extract paths from Bash command."""
        tool = ToolInvocation(
            tool_name="Bash",
            tool_id="bash-1",
            input={"command": "ls /tmp/file1 /tmp/file2"},
        )
        # Extracted paths containing "/"
        assert "/tmp/file1" in tool.affected_paths
        assert "/tmp/file2" in tool.affected_paths

    def test_affected_paths_bash_skip_flags(self):
        """Lines 133-135: Skip tokens starting with '-'."""
        tool = ToolInvocation(
            tool_name="Bash",
            tool_id="bash-1",
            input={"command": "ls -la /tmp/file"},
        )
        paths = tool.affected_paths
        # -la should not be included
        assert "-la" not in paths
        assert "/tmp/file" in paths


# =============================================================================
# MESSAGE CLASSIFICATION COVERAGE
# =============================================================================


class TestMessageChatGPTThinking:
    """Test Message._is_chatgpt_thinking() method (lines 300-313)."""

    @pytest.mark.parametrize("provider_meta,expected", [
        (None, False),
        ({"raw": "not a dict"}, False),
        ({"raw": {"content": {"content_type": "thoughts"}}}, True),
        ({"raw": {"content": {"content_type": "reasoning_recap"}}}, True),
        ({"raw": {"content": "not a dict"}}, False),
        ({"raw": {}}, False),
        ({"raw": {"metadata": "not a dict"}}, False),
    ], ids=["no_meta", "raw_not_dict", "thoughts", "reasoning_recap", "content_not_dict",
            "tool_no_metadata", "tool_metadata_not_dict"])
    def test_chatgpt_thinking(self, provider_meta, expected):
        """Lines 300-313: Test _is_chatgpt_thinking with various inputs."""
        msg = Message(id="m1", role="assistant", text="test", provider_meta=provider_meta)
        assert msg._is_chatgpt_thinking() is expected

    def test_chatgpt_thinking_role_tool_with_finished_text(self):
        """Lines 310-313: role is 'tool' with finished_text metadata."""
        msg = Message(
            id="m1",
            role="tool",
            text="test",
            provider_meta={
                "raw": {"metadata": {"finished_text": "some text"}}
            },
        )
        assert msg._is_chatgpt_thinking() is True


class TestMessageContextDump:
    """Test Message.is_context_dump property (lines 364-366)."""

    def test_is_context_dump_no_text(self):
        """Line 364: No text."""
        msg = Message(id="m1", role="user")
        assert msg.is_context_dump is False

    def test_is_context_dump_attachments_short_text(self):
        """Line 365-366: Has attachments and text < 100 chars."""
        att = Attachment(id="att1")
        msg = Message(
            id="m1",
            role="user",
            text="short",
            attachments=[att],
        )
        assert msg.is_context_dump is True

    def test_is_context_dump_attachments_long_text(self):
        """Line 365-366: Has attachments but text >= 100 chars."""
        att = Attachment(id="att1")
        msg = Message(
            id="m1",
            role="user",
            text="x" * 100,
            attachments=[att],
        )
        assert msg.is_context_dump is False

    def test_is_context_dump_system_prompt(self):
        """Line 368: Contains <system> tags."""
        msg = Message(
            id="m1",
            role="user",
            text="<system>system prompt content</system>",
        )
        assert msg.is_context_dump is True

    def test_is_context_dump_code_fences(self):
        """Lines 371-372: Has 3+ code fences (6+ backtick blocks)."""
        msg = Message(
            id="m1",
            role="user",
            text="```\ncode1\n```\n```\ncode2\n```\n```\ncode3\n```",
        )
        assert msg.is_context_dump is True

    def test_is_context_dump_regex_pattern(self):
        """Line 374: Matches context pattern."""
        msg = Message(
            id="m1",
            role="user",
            text="Contents of /tmp/file.txt:\nsome content",
        )
        assert msg.is_context_dump is True


class TestMessageExtractThinking:
    """Test Message.extract_thinking() method (lines 424-440)."""

    @pytest.mark.parametrize("blocks,expected", [
        ([{"type": "thinking", "text": "thinking content"}, {"type": "text", "text": "response text"}], "thinking content"),
        ([{"type": "thinking", "text": "first thought"}, {"type": "thinking", "text": "second thought"}], "first thought\n\nsecond thought"),
        (["not a dict", {"type": "thinking", "text": "thinking"}], "thinking"),
        ([{"type": "thinking", "text": 123}], None),
        ([{"type": "thinking", "text": "   \n\n   "}], None),
    ], ids=["single_block", "multiple_blocks", "mixed_blocks", "text_not_string", "empty_after_strip"])
    def test_extract_thinking_structured(self, blocks, expected):
        """Lines 423-431: Test extract_thinking with structured blocks."""
        msg = Message(
            id="m1",
            role="assistant",
            text="response",
            provider_meta={"content_blocks": blocks},
        )
        assert msg.extract_thinking() == expected

    def test_extract_thinking_structured_blocks_non_list(self):
        """Lines 424: content_blocks is not a list."""
        msg = Message(
            id="m1",
            role="assistant",
            text="response",
            provider_meta={
                "content_blocks": "not a list"
            },
        )
        # Should try XML tags next
        result = msg.extract_thinking()
        # Result depends on text content, not structured blocks
        assert result is None

    def test_extract_thinking_xml_tags(self):
        """Lines 434-437: Extract from XML thinking tags."""
        msg = Message(
            id="m1",
            role="assistant",
            text="<thinking>xml thinking content</thinking>",
        )
        assert msg.extract_thinking() == "xml thinking content"

    def test_extract_thinking_antml_tags(self):
        """Lines 434-437: Extract from antml:thinking tags."""
        msg = Message(
            id="m1",
            role="assistant",
            text="<thinking>antml thinking content</thinking>",
        )
        assert msg.extract_thinking() == "antml thinking content"

    def test_extract_thinking_xml_multiline(self):
        """Lines 434-437: XML tags with multiline content."""
        msg = Message(
            id="m1",
            role="assistant",
            text="<thinking>\nmultiline\nthinking\n</thinking>",
        )
        assert msg.extract_thinking() == "multiline\nthinking"

    def test_extract_thinking_chatgpt_thinking_role(self):
        """Lines 440: ChatGPT thinking message (role=tool with thinking metadata)."""
        msg = Message(
            id="m1",
            role="tool",
            text="thinking text content",
            provider_meta={
                "raw": {"metadata": {"finished_text": "data"}}
            },
        )
        assert msg.extract_thinking() == "thinking text content"

    def test_extract_thinking_gemini_isthought(self):
        """Lines 440: Gemini isThought marker."""
        msg = Message(
            id="m1",
            role="model",
            text="gemini thinking text",
            provider_meta={"isThought": True},
        )
        assert msg.extract_thinking() == "gemini thinking text"

    def test_extract_thinking_no_content(self):
        """Lines 443: No thinking content found."""
        msg = Message(
            id="m1",
            role="assistant",
            text="just response",
        )
        assert msg.extract_thinking() is None

    def test_extract_thinking_empty_text_with_blocks(self):
        """Line 431: Thinking block text is empty after strip."""
        msg = Message(
            id="m1",
            role="assistant",
            text="response",
            provider_meta={
                "content_blocks": [
                    {"type": "thinking", "text": "   \n\n   "},
                ]
            },
        )
        assert msg.extract_thinking() is None


# =============================================================================
# CONVERSATIONSUMMARY METADATA PROPERTIES
# =============================================================================


class TestConversationSummaryMetadata:
    """Test ConversationSummary metadata properties (lines 506-536)."""

    def test_display_title_user_title(self):
        """Line 509-511: Return user_title from metadata."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            metadata={"title": "User Title"},
        )
        assert summary.display_title == "User Title"

    def test_display_title_fallback_title(self):
        """Lines 512-513: Use title field if no user title."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            title="Auto Title",
        )
        assert summary.display_title == "Auto Title"

    def test_display_title_fallback_id(self):
        """Line 514: Use truncated ID if no title."""
        summary = ConversationSummary(
            id="c123456789abcdef",
            provider="claude",
        )
        assert summary.display_title == "c1234567"

    def test_tags_list(self):
        """Lines 519-521: Convert list of tags to strings."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            metadata={"tags": ["tag1", "tag2", 123]},
        )
        tags = summary.tags
        assert "tag1" in tags
        assert "tag2" in tags
        assert "123" in tags

    def test_tags_non_list(self):
        """Line 520: tags is not a list."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            metadata={"tags": "not a list"},
        )
        assert summary.tags == []

    def test_tags_empty(self):
        """Lines 519-522: No tags in metadata."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
        )
        assert summary.tags == []

    def test_summary_property(self):
        """Lines 527-528: Return summary from metadata."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            metadata={"summary": "Test summary"},
        )
        assert summary.summary == "Test summary"

    def test_summary_property_none(self):
        """Line 528: summary is None."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
        )
        assert summary.summary is None

    def test_is_continuation(self):
        """Line 531-532: Check continuation branch type."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            branch_type="continuation",
        )
        assert summary.is_continuation is True

    def test_is_sidechain(self):
        """Line 535-536: Check sidechain branch type."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            branch_type="sidechain",
        )
        assert summary.is_sidechain is True


# =============================================================================
# CONVERSATION FILTERING
# =============================================================================


class TestConversationFilter:
    """Test Conversation filter methods (lines 708-735)."""

    def _make_message(self, role: str, text: str, is_tool: bool = False) -> Message:
        """Helper to create test messages."""
        provider_meta = {}
        if is_tool:
            provider_meta["content_blocks"] = [{"type": "tool_use"}]
        return Message(
            id=f"m-{role}",
            role=role,
            text=text,
            provider_meta=provider_meta,
        )

    def test_filter_custom_predicate(self):
        """Line 714: Filter with custom predicate."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "hello"),
                self._make_message("assistant", "hi"),
                self._make_message("user", "bye"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        filtered = conv.filter(lambda m: m.is_user)
        assert len(filtered.messages) == 2

    def test_user_only(self):
        """Line 719: Filter user messages only."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "q1"),
                self._make_message("assistant", "a1"),
                self._make_message("user", "q2"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        user_only = conv.user_only()
        assert all(m.is_user for m in user_only.messages)

    def test_assistant_only(self):
        """Line 722-723: Filter assistant messages only."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "q1"),
                self._make_message("assistant", "a1"),
                self._make_message("user", "q2"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        assistant_only = conv.assistant_only()
        assert all(m.is_assistant for m in assistant_only.messages)

    def test_dialogue_only(self):
        """Line 727: Filter dialogue only (user + assistant)."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "q1"),
                self._make_message("assistant", "a1"),
                self._make_message("system", "sys"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        dialogue = conv.dialogue_only()
        assert all(m.is_dialogue for m in dialogue.messages)
        assert len(dialogue.messages) == 2

    def test_without_noise(self):
        """Line 731: Filter out noise (tool calls, context dumps, system)."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "real question"),
                self._make_message("assistant", "tool", is_tool=True),
                self._make_message("system", "sys msg"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        clean = conv.without_noise()
        assert all(not m.is_noise for m in clean.messages)

    def test_substantive_only(self):
        """Line 735: Filter substantive messages only."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "this is a substantive user message"),
                self._make_message("assistant", "short"),
                self._make_message("user", "x"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        substantive = conv.substantive_only()
        assert all(m.is_substantive for m in substantive.messages)


class TestConversationIterPairs:
    """Test Conversation.iter_pairs() method (lines 763-772)."""

    def _make_message(self, role: str, text: str) -> Message:
        """Helper to create test messages."""
        return Message(id=f"m-{role}", role=role, text=text)

    def test_iter_pairs_basic(self):
        """Lines 765-770: Iterate over user/assistant pairs."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "this is a substantive question"),
                self._make_message("assistant", "this is a substantive answer"),
                self._make_message("user", "another substantive question"),
                self._make_message("assistant", "another substantive answer"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        pairs = list(conv.iter_pairs())
        assert len(pairs) == 2
        assert "question" in pairs[0].user.text
        assert "answer" in pairs[0].assistant.text

    def test_iter_pairs_odd_messages(self):
        """Lines 767-772: Handle odd number of substantive messages."""
        msgs = MessageCollection(
            messages=[
                self._make_message("user", "this is a substantive question"),
                self._make_message("assistant", "this is a substantive answer"),
                self._make_message("user", "another substantive question"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        pairs = list(conv.iter_pairs())
        # Only complete pairs
        assert len(pairs) == 1

    def test_iter_pairs_out_of_order(self):
        """Lines 768-772: Skip out-of-order messages."""
        msgs = MessageCollection(
            messages=[
                self._make_message("assistant", "a1"),
                self._make_message("user", "q1"),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        pairs = list(conv.iter_pairs())
        # No valid pairs (assistant before user)
        assert len(pairs) == 0

    def test_iter_pairs_empty(self):
        """Line 767: Empty conversation."""
        msgs = MessageCollection(messages=[])
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        pairs = list(conv.iter_pairs())
        assert len(pairs) == 0


class TestConversationIterBranches:
    """Test Conversation.iter_branches() method (lines 782-808)."""

    def _make_message(self, msg_id: str, parent_id: str | None, branch_idx: int) -> Message:
        """Helper to create test messages with parent relationships."""
        return Message(
            id=msg_id,
            role="assistant",
            text=f"msg {msg_id}",
            parent_id=parent_id,
            branch_index=branch_idx,
        )

    def test_iter_branches_multiple_children(self):
        """Lines 799-808: Group messages by parent with multiple children."""
        msgs = MessageCollection(
            messages=[
                self._make_message("m1", parent_id=None, branch_idx=0),
                self._make_message("m2", parent_id="m1", branch_idx=0),
                self._make_message("m3", parent_id="m1", branch_idx=1),  # branch
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        branches_list = list(conv.iter_branches())
        assert len(branches_list) == 1
        parent_id, children = branches_list[0]
        assert parent_id == "m1"
        assert len(children) == 2

    def test_iter_branches_single_child(self):
        """Lines 804-805: Only parents with 2+ children are branches."""
        msgs = MessageCollection(
            messages=[
                self._make_message("m1", parent_id=None, branch_idx=0),
                self._make_message("m2", parent_id="m1", branch_idx=0),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        branches_list = list(conv.iter_branches())
        # No branches (only 1 child per parent)
        assert len(branches_list) == 0

    def test_iter_branches_no_parent(self):
        """Line 800: Skip messages without parent_id."""
        msgs = MessageCollection(
            messages=[
                self._make_message("m1", parent_id=None, branch_idx=0),
                self._make_message("m2", parent_id=None, branch_idx=0),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        branches_list = list(conv.iter_branches())
        # No branches (both are root messages)
        assert len(branches_list) == 0

    def test_iter_branches_sorted_by_index(self):
        """Lines 806-807: Children sorted by branch_index."""
        msgs = MessageCollection(
            messages=[
                self._make_message("m1", parent_id=None, branch_idx=0),
                self._make_message("m3", parent_id="m1", branch_idx=2),
                self._make_message("m2", parent_id="m1", branch_idx=1),
            ]
        )
        conv = Conversation(id="c1", provider="claude", messages=msgs)
        branches_list = list(conv.iter_branches())
        _, children = branches_list[0]
        # Sorted by branch_index
        assert children[0].branch_index == 1
        assert children[1].branch_index == 2


# =============================================================================
# MESSAGECOLLECTION COVERAGE
# =============================================================================


class TestMessageCollectionInitErrors:
    """Test MessageCollection.__init__ validation (lines 124, 136)."""

    def test_init_both_messages_and_source(self):
        """Line 124: Cannot specify both messages and conversation_id/source."""
        source = Mock(spec=MessageSource)
        with pytest.raises(ValueError, match="Cannot specify both"):
            MessageCollection(
                messages=[],
                conversation_id="c1",
                source=source,
            )

    @pytest.mark.parametrize("kwargs", [
        {},
        {"conversation_id": "c1"},
        {"source": Mock(spec=MessageSource)},
    ], ids=["no_args", "only_conv_id", "only_source"])
    def test_init_missing_arguments(self, kwargs):
        """Line 136: Missing required arguments."""
        with pytest.raises(ValueError, match="Must specify either"):
            MessageCollection(**kwargs)


class TestMessageCollectionIsLazy:
    """Test MessageCollection.is_lazy property (lines 147, 163)."""

    def test_is_lazy_true(self):
        """Line 147: Lazy mode (not materialized)."""
        source = Mock(spec=MessageSource)
        source.count_messages.return_value = 5
        coll = MessageCollection(conversation_id="c1", source=source)
        assert coll.is_lazy is True

    def test_is_lazy_false_eager(self):
        """Line 147: Eager mode (messages provided)."""
        coll = MessageCollection(messages=[])
        assert coll.is_lazy is False

    def test_is_lazy_false_materialized(self):
        """Line 147: Lazy mode but materialized."""
        msg = Message(id="m1", role="user", text="hello")
        source = Mock(spec=MessageSource)
        source.iter_messages.return_value = iter([msg])
        coll = MessageCollection(conversation_id="c1", source=source)
        # Use materialize() to ensure it's marked as materialized
        assert coll.is_lazy is True  # Before materialize
        coll.materialize()  # Ensure _messages is populated and _is_lazy is set False
        assert coll.is_lazy is False


class TestMessageCollectionLen:
    """Test MessageCollection.__len__ method (lines 183, 212)."""

    @pytest.mark.parametrize("messages,expected_len", [
        ([Mock(spec=Message) for _ in range(3)], 3),
        ([], 0),
    ])
    def test_len_eager(self, messages, expected_len):
        """Line 173: Eager mode returns len(list)."""
        coll = MessageCollection(messages=messages)
        assert len(coll) == expected_len

    def test_len_lazy_uncached(self):
        """Lines 179-181: Lazy mode queries source."""
        source = Mock(spec=MessageSource)
        source.count_messages.return_value = 5
        coll = MessageCollection(conversation_id="c1", source=source)
        assert len(coll) == 5

    def test_len_lazy_cached(self):
        """Lines 176-177: Use cached count."""
        source = Mock(spec=MessageSource)
        source.count_messages.return_value = 5
        coll = MessageCollection(conversation_id="c1", source=source)
        len(coll)  # First call
        len(coll)  # Should use cache
        # count_messages called only once
        source.count_messages.assert_called_once()

    def test_len_empty_edge_case(self):
        """Line 183: Return 0 if no source or conversation_id."""
        # This shouldn't happen normally but test the edge case
        coll = MessageCollection(messages=[])
        assert len(coll) == 0


class TestMessageCollectionRepr:
    """Test MessageCollection.__repr__ method (lines 215-217)."""

    def test_repr_lazy(self):
        """Lines 215-217: Representation includes 'lazy' mode."""
        source = Mock(spec=MessageSource)
        source.count_messages.return_value = 5
        coll = MessageCollection(conversation_id="c1", source=source)
        repr_str = repr(coll)
        assert "lazy" in repr_str
        assert "5" in repr_str

    def test_repr_eager(self):
        """Lines 215-217: Representation includes 'eager' mode."""
        msgs = [Mock(spec=Message) for _ in range(3)]
        coll = MessageCollection(messages=msgs)
        repr_str = repr(coll)
        assert "eager" in repr_str
        assert "3" in repr_str


class TestMessageCollectionEquality:
    """Test MessageCollection.__eq__ method (lines 225, 227)."""

    def test_eq_same_messages(self):
        """Line 227: Equal if same message content."""
        msg1 = Message(id="m1", role="user", text="hello")
        msg2 = Message(id="m1", role="user", text="hello")
        coll1 = MessageCollection(messages=[msg1])
        coll2 = MessageCollection(messages=[msg2])
        assert coll1 == coll2

    def test_eq_different_messages(self):
        """Line 227: Not equal if different messages."""
        msg1 = Message(id="m1", role="user", text="hello")
        msg2 = Message(id="m2", role="user", text="world")
        coll1 = MessageCollection(messages=[msg1])
        coll2 = MessageCollection(messages=[msg2])
        assert coll1 != coll2

    def test_eq_not_messagecollection(self):
        """Line 225: NotImplemented if comparing with non-MessageCollection."""
        coll = MessageCollection(messages=[])
        assert (coll == []) is False


class TestMessageCollectionHash:
    """Test MessageCollection.__hash__ method (line 231)."""

    def test_hash_uses_id(self):
        """Line 231: Hash uses object id."""
        coll1 = MessageCollection(messages=[])
        coll2 = MessageCollection(messages=[])
        # Different objects should have different hashes
        assert hash(coll1) != hash(coll2)

    def test_hash_same_object(self):
        """Line 231: Same object has same hash."""
        coll = MessageCollection(messages=[])
        assert hash(coll) == hash(coll)


class TestMessageCollectionToList:
    """Test MessageCollection.to_list() method (lines 244, 255-258)."""

    def test_to_list_eager(self):
        """Line 243: Eager mode returns copy of list."""
        msg1 = Message(id="m1", role="user", text="hello")
        coll = MessageCollection(messages=[msg1])
        lst = coll.to_list()
        assert len(lst) == 1
        assert lst[0] == msg1
        # Should be a copy
        lst.append(Message(id="m2", role="user", text="world"))
        assert len(coll.to_list()) == 1

    def test_to_list_lazy(self):
        """Line 244: Lazy mode materializes."""
        msg1 = Message(id="m1", role="user", text="hello")
        source = Mock(spec=MessageSource)
        source.iter_messages.return_value = iter([msg1])
        coll = MessageCollection(conversation_id="c1", source=source)
        lst = coll.to_list()
        assert len(lst) == 1

    def test_materialize_returns_self(self):
        """Lines 255-257: materialize() returns self."""
        source = Mock(spec=MessageSource)
        source.iter_messages.return_value = iter([])
        coll = MessageCollection(conversation_id="c1", source=source)
        result = coll.materialize()
        assert result is coll

    def test_materialize_sets_lazy_false(self):
        """Lines 256-257: materialize() sets is_lazy to False."""
        source = Mock(spec=MessageSource)
        source.iter_messages.return_value = iter([])
        coll = MessageCollection(conversation_id="c1", source=source)
        assert coll.is_lazy is True
        coll.materialize()
        assert coll.is_lazy is False


class TestMessageCollectionEmpty:
    """Test MessageCollection.empty() class method (line 266)."""

    def test_empty_creates_collection(self):
        """Line 266: empty() creates empty MessageCollection."""
        coll = MessageCollection.empty()
        assert len(coll) == 0
        assert isinstance(coll, MessageCollection)


class TestMessageCollectionPydanticSchema:
    """Test Pydantic schema methods (lines 288, 292, 310-313)."""

    def test_get_pydantic_core_schema(self):
        """Lines 288, 292: Core schema validation and serialization."""
        handler = Mock()
        handler.generate_schema.return_value = {"type": "object"}
        schema = MessageCollection.__get_pydantic_core_schema__(
            MessageCollection,
            handler,
        )
        # Check that it's a schema with validation/serialization
        assert schema is not None
        assert hasattr(schema, '__iter__') or isinstance(schema, dict)

    def test_get_pydantic_json_schema(self):
        """Lines 310-313: JSON schema generation."""
        handler = Mock()
        handler.generate.return_value = {"type": "object"}
        handler.resolve_ref_schema.return_value = {"type": "object"}

        json_schema = MessageCollection.__get_pydantic_json_schema__(
            Mock(),
            handler,
        )
        assert json_schema["type"] == "array"
        assert "items" in json_schema


class TestMessageCollectionBool:
    """Test MessageCollection.__bool__ method (line 212)."""

    def test_bool_empty(self):
        """Line 212: Empty collection is falsy."""
        coll = MessageCollection(messages=[])
        assert bool(coll) is False

    def test_bool_nonempty(self):
        """Line 212: Non-empty collection is truthy."""
        msg = Message(id="m1", role="user", text="hello")
        coll = MessageCollection(messages=[msg])
        assert bool(coll) is True


# =============================================================================
# DIALOGUE PAIR VALIDATION
# =============================================================================


class TestDialoguePairValidation:
    """Test DialoguePair validator."""

    def test_dialogue_pair_valid(self):
        """Valid user-assistant pair."""
        user_msg = Message(id="m1", role="user", text="question")
        assistant_msg = Message(id="m2", role="assistant", text="answer")
        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.user.is_user
        assert pair.assistant.is_assistant

    def test_dialogue_pair_invalid_user_role(self):
        """Invalid: first message not from user."""
        assistant_msg = Message(id="m1", role="assistant", text="answer")
        user_msg = Message(id="m2", role="assistant", text="question")
        with pytest.raises(ValueError, match="user message must have user role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_invalid_assistant_role(self):
        """Invalid: second message not from assistant."""
        user_msg = Message(id="m1", role="user", text="question")
        assistant_msg = Message(id="m2", role="user", text="answer")
        with pytest.raises(ValueError, match="assistant message must have assistant role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)


# =============================================================================
# MESSAGE ATTACHMENT RECORD CONVERSION
# =============================================================================


class TestAttachmentFromRecord:
    """Test Attachment.from_record() class method (lines 213-222)."""

    def test_from_record_with_name(self):
        """Lines 214-217: Extract name from provider_meta."""
        record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            provider_meta={"name": "file.txt"},
        )
        att = Attachment.from_record(record)
        assert att.name == "file.txt"

    def test_from_record_name_not_string(self):
        """Lines 217-218: Use attachment_id if name is not string."""
        record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            provider_meta={"name": 123},
        )
        att = Attachment.from_record(record)
        assert att.name == "att1"

    def test_from_record_no_provider_meta(self):
        """Line 214: No provider_meta."""
        record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
        )
        att = Attachment.from_record(record)
        assert att.name == "att1"


class TestMessageFromRecord:
    """Test Message.from_record() class method."""

    def test_from_record_empty_role(self):
        """Line 249: Empty role becomes 'unknown'."""
        record = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="",
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash1",
        )
        msg = Message.from_record(record, [])
        assert msg.role == "unknown"

    def test_from_record_whitespace_role(self):
        """Line 249: Whitespace-only role becomes 'unknown'."""
        record = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="   ",
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash1",
        )
        msg = Message.from_record(record, [])
        assert msg.role == "unknown"

    def test_from_record_normal_role(self):
        """Line 249: Normal role is stripped."""
        record = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="  assistant  ",
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash1",
        )
        msg = Message.from_record(record, [])
        assert msg.role == "assistant"


# =============================================================================
# CONVERSATIONSUMMARY.FROM_RECORD
# =============================================================================


class TestConversationSummaryFromRecord:
    """Test ConversationSummary.from_record() class method."""

    def test_from_record_complete(self):
        """Create summary with all fields populated."""
        record = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="prov-c1",
            content_hash="hash1",
            title="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            provider_meta={"key": "value"},
            metadata={"tags": ["test"]},
        )
        summary = ConversationSummary.from_record(record)
        assert summary.id == "c1"
        assert summary.provider == "claude"
        assert summary.title == "Test"


class TestConversationFromRecords:
    """Test Conversation.from_records() class method."""

    def test_from_records_with_attachments(self):
        """Build conversation with messages and attachments."""
        conv_record = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="prov-c1",
            content_hash="hash-c1",
            title="Test",
        )
        msg_record = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash-m1",
        )
        att_record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m1",
            provider_meta={"name": "file.txt"},
        )
        conv = Conversation.from_records(conv_record, [msg_record], [att_record])
        assert conv.id == "c1"
        assert len(conv.messages) == 1
        assert len(conv.messages[0].attachments) == 1


class TestConversationFromLazy:
    """Test Conversation.from_lazy() class method."""

    def test_from_lazy_creates_lazy_collection(self):
        """Create conversation with lazy message loading."""
        conv_record = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="prov-c1",
            content_hash="hash-c1",
            title="Test",
        )
        source = Mock(spec=MessageSource)
        source.count_messages.return_value = 10
        conv = Conversation.from_lazy(conv_record, source)
        assert conv.id == "c1"
        assert conv.messages.is_lazy is True
