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
# TEST DATA CONSTANTS (module-level for parametrization)
# =============================================================================

# ToolInvocation property test data
TOOLINV_FILE_OPS = [
    ("Read", True),
    ("Write", True),
    ("Edit", True),
    ("NotebookEdit", True),
    ("Bash", False),
]

TOOLINV_GIT_OPS = [
    ("Bash", {"command": "git commit -m 'test'"}, True, "git_command"),
    ("Read", {"command": "git status"}, False, "not_bash"),
    ("Bash", {"command": "ls -la"}, False, "non_git_bash"),
    ("Bash", {"command": 123}, False, "non_string_cmd"),
    ("Bash", {"command": "  git push  "}, True, "whitespace_git"),
    ("Bash", {}, False, "no_command"),
]

TOOLINV_SEARCH_OPS = [
    ("Glob", True),
    ("Grep", True),
    ("WebSearch", True),
    ("Bash", False),
]

TOOLINV_SUBAGENTS = [
    ("Task", True),
    ("Bash", False),
]

TOOLINV_AFFECTED_PATHS = [
    ("Read", {"file_path": "/tmp/test.txt"}, ["/tmp/test.txt"], "read"),
    ("Write", {"file_path": "/tmp/output.txt"}, ["/tmp/output.txt"], "write"),
    ("Edit", {"file_path": "/tmp/code.py"}, ["/tmp/code.py"], "edit"),
    ("Read", {"path": "/tmp/fallback.txt"}, ["/tmp/fallback.txt"], "path_fallback"),
    ("Read", {"file_path": "/tmp/primary.txt", "path": "/tmp/fallback.txt"}, ["/tmp/primary.txt"], "file_path_priority"),
    ("Read", {"file_path": 123}, [], "non_string_path"),
    ("Glob", {"pattern": "**/*.py"}, ["**/*.py"], "glob"),
    ("Glob", {"pattern": ["*.py", "*.txt"]}, [], "glob_non_string"),
    ("Bash", {"command": 123}, [], "bash_non_string"),
    ("Task", {"prompt": "do something"}, [], "other_tool"),
]

# Message._is_chatgpt_thinking test data
CHATGPT_THINKING_META = [
    (None, False, "no_meta"),
    ({"raw": "not a dict"}, False, "raw_not_dict"),
    ({"raw": {"content": {"content_type": "thoughts"}}}, True, "thoughts"),
    ({"raw": {"content": {"content_type": "reasoning_recap"}}}, True, "reasoning_recap"),
    ({"raw": {"content": "not a dict"}}, False, "content_not_dict"),
    ({"raw": {}}, False, "tool_no_metadata"),
    ({"raw": {"metadata": "not a dict"}}, False, "tool_metadata_not_dict"),
]

# Message.extract_thinking test data
EXTRACT_THINKING_BLOCKS = [
    ([{"type": "thinking", "text": "thinking content"}, {"type": "text", "text": "response text"}], "thinking content", "single_block"),
    ([{"type": "thinking", "text": "first thought"}, {"type": "thinking", "text": "second thought"}], "first thought\n\nsecond thought", "multiple_blocks"),
    (["not a dict", {"type": "thinking", "text": "thinking"}], "thinking", "mixed_blocks"),
    ([{"type": "thinking", "text": 123}], None, "text_not_string"),
    ([{"type": "thinking", "text": "   \n\n   "}], None, "empty_after_strip"),
]

# Message.is_context_dump test data
CONTEXT_DUMP_CASES = [
    (None, [], False, "no_text"),
    ("short", [Attachment(id="att1")], True, "attachments_short_text"),
    ("x" * 100, [Attachment(id="att1")], False, "attachments_long_text"),
    ("<system>system prompt content</system>", [], True, "system_prompt"),
    ("```\ncode1\n```\n```\ncode2\n```\n```\ncode3\n```", [], True, "code_fences"),
    ("Contents of /tmp/file.txt:\nsome content", [], True, "regex_pattern"),
]

# Message.from_record role handling test data
MESSAGE_ROLE_CASES = [
    ("", "unknown", "empty_role"),
    ("   ", "unknown", "whitespace_role"),
    ("  assistant  ", "assistant", "normal_role"),
]

# ConversationSummary metadata property test data
SUMMARY_DISPLAY_TITLE = [
    ({"title": "User Title"}, None, "User Title", "user_title"),
    ({}, "Auto Title", "Auto Title", "fallback_title"),
    ({}, None, "c1234567", "fallback_id"),
]

SUMMARY_TAGS = [
    ({"tags": ["tag1", "tag2", 123]}, ["tag1", "tag2", "123"], "list"),
    ({"tags": "not a list"}, [], "non_list"),
    ({}, [], "empty"),
]

SUMMARY_BRANCH_TYPE = [
    ("continuation", True, False, "continuation"),
    ("sidechain", False, True, "sidechain"),
    (None, False, False, "other"),
]

# Attachment.from_record name test data
ATTACHMENT_NAMES = [
    ({"name": "file.txt"}, "file.txt", "with_name"),
    ({"name": 123}, "att1", "name_not_string"),
    (None, "att1", "no_provider_meta"),
]

# Conversation filter test data
FILTER_CASES = [
    ("custom_predicate", {}, 2, "custom_predicate"),  # uses lambda
    ("user_only", {}, 2, "user_only"),
    ("assistant_only", {}, 1, "assistant_only"),
    ("dialogue_only", {}, 2, "dialogue_only"),
    ("without_noise", {}, 1, "without_noise"),
    ("substantive_only", {}, 1, "substantive_only"),
]

# Conversation.iter_pairs test data
ITER_PAIRS_CASES = [
    ("basic", 2, "basic"),
    ("odd_messages", 1, "odd_messages"),
    ("out_of_order", 0, "out_of_order"),
    ("empty", 0, "empty"),
]

# Conversation.iter_branches test data
ITER_BRANCHES_CASES = [
    ("multiple_children", 1, "multiple_children"),
    ("single_child", 0, "single_child"),
    ("no_parent", 0, "no_parent"),
    ("sorted_by_index", 2, "sorted_by_index"),
]

# DialoguePair validation test data
DIALOGUE_PAIR_CASES = [
    ("valid", "user", "assistant", True, None, "valid"),
    ("invalid_user_role", "assistant", "assistant", False, "user message must have user role", "invalid_user_role"),
    ("invalid_assistant_role", "user", "user", False, "assistant message must have assistant role", "invalid_assistant_role"),
]


# =============================================================================
# TOOLINVOCATION COVERAGE
# =============================================================================


class TestToolInvocationProperties:
    """Test ToolInvocation property methods."""

    @pytest.mark.parametrize("tool_name,expected", TOOLINV_FILE_OPS)
    def test_is_file_operation(self, tool_name, expected):
        """Line 93: Check is_file_operation with various tools."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input={})
        assert tool.is_file_operation is expected

    @pytest.mark.parametrize("tool_name,input_data,expected,test_id", TOOLINV_GIT_OPS)
    def test_is_git_operation(self, tool_name, input_data, expected, test_id):
        """Line 98-101: Test is_git_operation with various inputs."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input=input_data)
        assert tool.is_git_operation is expected

    @pytest.mark.parametrize("tool_name,expected", TOOLINV_SEARCH_OPS)
    def test_is_search_operation(self, tool_name, expected):
        """Line 106: Test is_search_operation with various tools."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input={})
        assert tool.is_search_operation is expected

    @pytest.mark.parametrize("tool_name,expected", TOOLINV_SUBAGENTS)
    def test_is_subagent(self, tool_name, expected):
        """Line 111: Test is_subagent with various tools."""
        tool = ToolInvocation(tool_name=tool_name, tool_id="t1", input={})
        assert tool.is_subagent is expected

    @pytest.mark.parametrize("tool_name,input_data,expected,test_id", TOOLINV_AFFECTED_PATHS)
    def test_affected_paths(self, tool_name, input_data, expected, test_id):
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


class TestMessageClassification:
    """Test Message classification properties."""

    @pytest.mark.parametrize("provider_meta,expected,test_id", CHATGPT_THINKING_META)
    def test_chatgpt_thinking(self, provider_meta, expected, test_id):
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

    @pytest.mark.parametrize("text,attachments,expected,test_id", CONTEXT_DUMP_CASES)
    def test_is_context_dump(self, text, attachments, expected, test_id):
        """Lines 364-374: Test is_context_dump with various inputs."""
        kwargs = {
            "id": "m1",
            "role": "user",
            "text": text,
        }
        if attachments:
            kwargs["attachments"] = attachments
        msg = Message(**kwargs)
        assert msg.is_context_dump is expected


class TestMessageExtractThinking:
    """Test Message.extract_thinking() method (lines 424-440)."""

    @pytest.mark.parametrize("blocks,expected,test_id", EXTRACT_THINKING_BLOCKS)
    def test_extract_thinking_structured(self, blocks, expected, test_id):
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


# =============================================================================
# CONVERSATIONSUMMARY METADATA PROPERTIES
# =============================================================================


class TestConversationSummaryMetadata:
    """Test ConversationSummary metadata properties (lines 506-536)."""

    @pytest.mark.parametrize("metadata,title,expected,test_id", SUMMARY_DISPLAY_TITLE)
    def test_display_title(self, metadata, title, expected, test_id):
        """Lines 509-514: Test display_title with various metadata/title."""
        summary = ConversationSummary(
            id="c123456789abcdef" if test_id == "fallback_id" else "c1",
            provider="claude",
            metadata=metadata,
            title=title,
        )
        assert summary.display_title == expected

    @pytest.mark.parametrize("metadata,expected,test_id", SUMMARY_TAGS)
    def test_tags(self, metadata, expected, test_id):
        """Lines 519-522: Test tags property."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            metadata=metadata,
        )
        tags = summary.tags
        assert tags == expected

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

    @pytest.mark.parametrize("branch_type,is_cont,is_side,test_id", SUMMARY_BRANCH_TYPE)
    def test_branch_type_properties(self, branch_type, is_cont, is_side, test_id):
        """Lines 531-536: Test is_continuation and is_sidechain."""
        summary = ConversationSummary(
            id="c1",
            provider="claude",
            branch_type=branch_type,
        )
        assert summary.is_continuation is is_cont
        assert summary.is_sidechain is is_side


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

    @pytest.mark.parametrize("filter_type,_,expected_count,test_id", FILTER_CASES)
    def test_filter_methods(self, filter_type, _, expected_count, test_id):
        """Lines 714-735: Test all filter methods."""
        if filter_type == "custom_predicate":
            msgs = MessageCollection(
                messages=[
                    self._make_message("user", "hello"),
                    self._make_message("assistant", "hi"),
                    self._make_message("user", "bye"),
                ]
            )
            conv = Conversation(id="c1", provider="claude", messages=msgs)
            filtered = conv.filter(lambda m: m.is_user)
            assert len(filtered.messages) == expected_count

        elif filter_type == "user_only":
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

        elif filter_type == "assistant_only":
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

        elif filter_type == "dialogue_only":
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
            assert len(dialogue.messages) == expected_count

        elif filter_type == "without_noise":
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

        elif filter_type == "substantive_only":
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

    @pytest.mark.parametrize("case_type,expected_pairs,test_id", ITER_PAIRS_CASES)
    def test_iter_pairs(self, case_type, expected_pairs, test_id):
        """Lines 765-772: Test iter_pairs with various message patterns."""
        if case_type == "basic":
            msgs = MessageCollection(
                messages=[
                    self._make_message("user", "this is a substantive question"),
                    self._make_message("assistant", "this is a substantive answer"),
                    self._make_message("user", "another substantive question"),
                    self._make_message("assistant", "another substantive answer"),
                ]
            )
        elif case_type == "odd_messages":
            msgs = MessageCollection(
                messages=[
                    self._make_message("user", "this is a substantive question"),
                    self._make_message("assistant", "this is a substantive answer"),
                    self._make_message("user", "another substantive question"),
                ]
            )
        elif case_type == "out_of_order":
            msgs = MessageCollection(
                messages=[
                    self._make_message("assistant", "a1"),
                    self._make_message("user", "q1"),
                ]
            )
        elif case_type == "empty":
            msgs = MessageCollection(messages=[])

        conv = Conversation(id="c1", provider="claude", messages=msgs)
        pairs = list(conv.iter_pairs())
        assert len(pairs) == expected_pairs


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

    @pytest.mark.parametrize("case_type,expected_branches,test_id", ITER_BRANCHES_CASES)
    def test_iter_branches(self, case_type, expected_branches, test_id):
        """Lines 799-808: Test iter_branches with various parent/child patterns."""
        if case_type == "multiple_children":
            msgs = MessageCollection(
                messages=[
                    self._make_message("m1", parent_id=None, branch_idx=0),
                    self._make_message("m2", parent_id="m1", branch_idx=0),
                    self._make_message("m3", parent_id="m1", branch_idx=1),  # branch
                ]
            )
            expected_len = 1
            parent_id_to_check = "m1"
            child_count = 2
        elif case_type == "single_child":
            msgs = MessageCollection(
                messages=[
                    self._make_message("m1", parent_id=None, branch_idx=0),
                    self._make_message("m2", parent_id="m1", branch_idx=0),
                ]
            )
            expected_len = 0
            parent_id_to_check = None
            child_count = None
        elif case_type == "no_parent":
            msgs = MessageCollection(
                messages=[
                    self._make_message("m1", parent_id=None, branch_idx=0),
                    self._make_message("m2", parent_id=None, branch_idx=0),
                ]
            )
            expected_len = 0
            parent_id_to_check = None
            child_count = None
        elif case_type == "sorted_by_index":
            msgs = MessageCollection(
                messages=[
                    self._make_message("m1", parent_id=None, branch_idx=0),
                    self._make_message("m3", parent_id="m1", branch_idx=2),
                    self._make_message("m2", parent_id="m1", branch_idx=1),
                ]
            )
            expected_len = 1
            parent_id_to_check = "m1"
            child_count = 2

        conv = Conversation(id="c1", provider="claude", messages=msgs)
        branches_list = list(conv.iter_branches())
        assert len(branches_list) == expected_len

        if expected_len > 0:
            parent_id, children = branches_list[0]
            if parent_id_to_check:
                assert parent_id == parent_id_to_check
            if child_count:
                assert len(children) == child_count
                if case_type == "sorted_by_index":
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
    """Test MessageCollection.__len__ method (lines 173, 212)."""

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
        coll = MessageCollection(messages=[])
        assert len(coll) == 0


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


class TestMessageCollectionToList:
    """Test MessageCollection.to_list() and materialize() methods."""

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


# =============================================================================
# DIALOGUE PAIR VALIDATION
# =============================================================================


class TestDialoguePairValidation:
    """Test DialoguePair validator."""

    @pytest.mark.parametrize("case_type,user_role,asst_role,should_pass,match_pattern,test_id", DIALOGUE_PAIR_CASES)
    def test_dialogue_pair(self, case_type, user_role, asst_role, should_pass, match_pattern, test_id):
        """Test DialoguePair validation."""
        user_msg = Message(id="m1", role=user_role, text="question")
        assistant_msg = Message(id="m2", role=asst_role, text="answer")

        if should_pass:
            pair = DialoguePair(user=user_msg, assistant=assistant_msg)
            assert pair.user.is_user
            assert pair.assistant.is_assistant
        else:
            with pytest.raises(ValueError, match=match_pattern):
                DialoguePair(user=user_msg, assistant=assistant_msg)


# =============================================================================
# MESSAGE ATTACHMENT RECORD CONVERSION
# =============================================================================


class TestAttachmentFromRecord:
    """Test Attachment.from_record() class method (lines 213-222)."""

    @pytest.mark.parametrize("provider_meta,expected,test_id", ATTACHMENT_NAMES)
    def test_from_record_name(self, provider_meta, expected, test_id):
        """Lines 214-218: Extract/derive name from provider_meta."""
        record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            provider_meta=provider_meta,
        )
        att = Attachment.from_record(record)
        assert att.name == expected


class TestMessageFromRecord:
    """Test Message.from_record() class method."""

    @pytest.mark.parametrize("role,expected,test_id", MESSAGE_ROLE_CASES)
    def test_from_record_role(self, role, expected, test_id):
        """Line 249: Test role normalization in from_record."""
        record = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role=role,
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash1",
        )
        msg = Message.from_record(record, [])
        assert msg.role == expected


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
