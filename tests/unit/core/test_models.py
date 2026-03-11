"""Comprehensive coverage tests for models.py and messages.py uncovered lines."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import (
    Attachment,
    Conversation,
    ConversationSummary,
    DialoguePair,
    Message,
)
from polylogue.lib.viewports import ToolCall, ToolCategory, classify_tool
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord

# Test data for parametrized tests
TOOL_FILE_OPS = [("Read", True), ("Write", True), ("Edit", True), ("NotebookEdit", True), ("Bash", False)]
TOOL_SEARCH_OPS = [("Glob", True), ("Grep", True), ("WebSearch", True), ("Bash", False)]
TOOL_SUBAGENTS = [("Task", True), ("Bash", False)]
TOOL_GIT_OPS = [
    ("Bash", {"command": "git commit -m 'test'"}, True, "git_command"),
    ("Read", {"command": "git status"}, False, "not_bash"),
    ("Bash", {"command": "ls -la"}, False, "non_git_bash"),
    ("Bash", {"command": 123}, False, "non_string_cmd"),
    ("Bash", {"command": "  git push  "}, True, "whitespace_git"),
    ("Bash", {}, False, "no_command"),
]
TOOL_AFFECTED_PATHS = [
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
CHATGPT_THINKING_META = [
    (None, False, "no_meta"), ({"raw": "not a dict"}, False, "raw_not_dict"),
    ({"raw": {"content": {"content_type": "thoughts"}}}, True, "thoughts"),
    ({"raw": {"content": {"content_type": "reasoning_recap"}}}, True, "reasoning_recap"),
    ({"raw": {"content": "not a dict"}}, False, "content_not_dict"),
    ({"raw": {}}, False, "tool_no_metadata"), ({"raw": {"metadata": "not a dict"}}, False, "tool_metadata_not_dict"),
]
# Unified table for extract_thinking: (provider_meta, role, text, expected_result, test_id)
EXTRACT_THINKING_CASES = [
    # Structured blocks — thinking content extracted
    (
        {"content_blocks": [{"type": "thinking", "text": "thinking content"}, {"type": "text", "text": "response text"}]},
        "assistant", "response",
        "thinking content",
        "single_block",
    ),
    (
        {"content_blocks": [{"type": "thinking", "text": "first thought"}, {"type": "thinking", "text": "second thought"}]},
        "assistant", "response",
        "first thought\n\nsecond thought",
        "multiple_blocks",
    ),
    (
        {"content_blocks": ["not a dict", {"type": "thinking", "text": "thinking"}]},
        "assistant", "response",
        "thinking",
        "mixed_blocks",
    ),
    (
        {"content_blocks": [{"type": "thinking", "text": 123}]},
        "assistant", "response",
        None,
        "text_not_string",
    ),
    (
        {"content_blocks": [{"type": "thinking", "text": "   \n\n   "}]},
        "assistant", "response",
        None,
        "empty_after_strip",
    ),
    # content_blocks not a list — falls through to XML
    (
        {"content_blocks": "not a list"},
        "assistant", "response",
        None,
        "blocks_not_list",
    ),
    # XML <thinking> tags in text
    (
        None,
        "assistant", "<thinking>xml thinking content</thinking>",
        "xml thinking content",
        "xml_tags",
    ),
    # antml:thinking tags (same regex as <thinking>)
    (
        None,
        "assistant", "<thinking>antml thinking content</thinking>",
        "antml thinking content",
        "antml_tags",
    ),
    # Multiline XML — preserved
    (
        None,
        "assistant", "<thinking>\nmultiline\nthinking\n</thinking>",
        "multiline\nthinking",
        "xml_multiline",
    ),
    # ChatGPT thinking role (role=tool with finished_text metadata)
    (
        {"raw": {"metadata": {"finished_text": "data"}}},
        "tool", "thinking text content",
        "thinking text content",
        "chatgpt_thinking_role",
    ),
    # Gemini isThought
    (
        {"isThought": True},
        "model", "gemini thinking text",
        "gemini thinking text",
        "gemini_isthought",
    ),
    # No thinking content
    (
        None,
        "assistant", "just response",
        None,
        "no_content",
    ),
]
CONTEXT_DUMP_CASES = [
    (None, [], False, "no_text"), ("short", [Attachment(id="att1")], True, "attachments_short_text"),
    ("x" * 100, [Attachment(id="att1")], False, "attachments_long_text"),
    ("<system>system prompt content</system>", [], True, "system_prompt"),
    ("```\ncode1\n```\n```\ncode2\n```\n```\ncode3\n```", [], True, "code_fences"),
    ("Contents of /tmp/file.txt:\nsome content", [], True, "regex_pattern"),
]
MESSAGE_ROLE_CASES = [("", "unknown", "empty_role"), ("   ", "unknown", "whitespace_role"), ("  assistant  ", "assistant", "normal_role")]
ATTACHMENT_NAMES = [({"name": "file.txt"}, "file.txt", "with_name"), ({"name": 123}, "att1", "name_not_string"), (None, "att1", "no_provider_meta")]
DIALOGUE_PAIR_CASES = [
    ("valid", "user", "assistant", True, None, "valid"),
    ("invalid_user_role", "assistant", "assistant", False, "user message must have user role", "invalid_user_role"),
    ("invalid_assistant_role", "user", "user", False, "assistant message must have assistant role", "invalid_assistant_role"),
]


# =============================================================================
# TOOLINVOCATION COVERAGE
# =============================================================================


def _make_tool(tool_name: str, input_data: dict | None = None) -> ToolCall:
    """Build a ToolCall with its category derived from classify_tool."""
    inp = input_data or {}
    return ToolCall(name=tool_name, id="t1", input=inp, category=classify_tool(tool_name, inp))


class TestToolCallProperties:
    """Test ToolCall property methods (replaces removed ToolInvocation)."""

    @pytest.mark.parametrize("tool_name,expected", TOOL_FILE_OPS)
    def test_is_file_operation(self, tool_name, expected):
        tool = _make_tool(tool_name)
        assert tool.is_file_operation is expected

    @pytest.mark.parametrize("tool_name,input_data,expected,test_id", TOOL_GIT_OPS)
    def test_is_git_operation(self, tool_name, input_data, expected, test_id):
        tool = _make_tool(tool_name, input_data)
        assert tool.is_git_operation is expected

    @pytest.mark.parametrize("tool_name,expected", TOOL_SEARCH_OPS)
    def test_is_search_or_web(self, tool_name, expected):
        """Search/web category covers Glob, Grep, WebSearch."""
        tool = _make_tool(tool_name)
        is_search = tool.category in (ToolCategory.SEARCH, ToolCategory.WEB)
        assert is_search is expected

    @pytest.mark.parametrize("tool_name,expected", TOOL_SUBAGENTS)
    def test_is_subagent(self, tool_name, expected):
        tool = _make_tool(tool_name)
        assert tool.is_subagent is expected

    @pytest.mark.parametrize("tool_name,input_data,expected,test_id", TOOL_AFFECTED_PATHS)
    def test_affected_paths(self, tool_name, input_data, expected, test_id):
        tool = _make_tool(tool_name, input_data)
        assert tool.affected_paths == expected

    def test_affected_paths_bash_extraction(self):
        """Extract paths from Bash command via regex."""
        tool = _make_tool("Bash", {"command": "ls /tmp/file1 /tmp/file2"})
        assert "/tmp/file1" in tool.affected_paths
        assert "/tmp/file2" in tool.affected_paths

    def test_affected_paths_bash_skip_flags(self):
        """Path regex should not match -la style flags."""
        tool = _make_tool("Bash", {"command": "ls -la /tmp/file"})
        paths = tool.affected_paths
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

    @pytest.mark.parametrize("provider_meta,role,text,expected,test_id", EXTRACT_THINKING_CASES)
    def test_extract_thinking(self, provider_meta, role, text, expected, test_id):
        """Lines 423-440: Test extract_thinking across all cases."""
        msg = Message(
            id="m1",
            role=role,
            text=text,
            provider_meta=provider_meta,
        )
        assert msg.extract_thinking() == expected

    def test_extract_thinking_prefers_db_content_blocks(self):
        """DB-loaded thinking blocks must win over provider_meta and text fallbacks."""
        msg = Message(
            id="m1",
            role="assistant",
            text="<thinking>xml fallback</thinking>",
            content_blocks=[
                {"type": "thinking", "text": "db thought 1"},
                {"type": "thinking", "text": "db thought 2"},
                {"type": "text", "text": "visible text"},
            ],
            provider_meta={
                "content_blocks": [{"type": "thinking", "text": "provider-meta thought"}],
                "isThought": True,
            },
        )

        assert msg.extract_thinking() == "db thought 1\n\ndb thought 2"

    def test_extract_thinking_db_blocks_require_thinking_type_and_string_text(self):
        """DB content_blocks should only use typed thinking blocks with string text."""
        msg = Message(
            id="m1",
            role="assistant",
            text="plain response",
            content_blocks=[
                {"type": "text", "text": "ignored"},
                {"type": "thinking", "text": "db-only thinking"},
                {"type": "thinking", "text": 123},
            ],
        )

        assert msg.extract_thinking() == "db-only thinking"




class TestConversationIterPairs:
    """Test Conversation.iter_pairs() method (lines 763-772)."""

    def test_iter_pairs_basic(self):
        """Lines 765-770: Iterate over user/assistant pairs."""
        msgs = MessageCollection(messages=[
            Message(id="m1", role="user", text="this is a substantive question"),
            Message(id="m2", role="assistant", text="this is a substantive answer"),
            Message(id="m3", role="user", text="another substantive question"),
            Message(id="m4", role="assistant", text="another substantive answer"),
        ])
        pairs = list(Conversation(id="c1", provider="claude", messages=msgs).iter_pairs())
        assert len(pairs) == 2
        assert [(pair.user.id, pair.assistant.id) for pair in pairs] == [
            ("m1", "m2"),
            ("m3", "m4"),
        ]

    def test_iter_pairs_odd_messages(self):
        """Lines 767-772: Handle odd number of substantive messages."""
        msgs = MessageCollection(messages=[
            Message(id="m1", role="user", text="this is a substantive question"),
            Message(id="m2", role="assistant", text="this is a substantive answer"),
            Message(id="m3", role="user", text="another substantive question"),
        ])
        pairs = list(Conversation(id="c1", provider="claude", messages=msgs).iter_pairs())
        assert len(pairs) == 1
        assert [(pair.user.id, pair.assistant.id) for pair in pairs] == [("m1", "m2")]

    def test_iter_pairs_out_of_order(self):
        """Lines 768-772: Skip out-of-order messages."""
        msgs = MessageCollection(messages=[
            Message(id="m1", role="assistant", text="assistant substantive answer"),
            Message(id="m2", role="user", text="user substantive question"),
            Message(id="m3", role="assistant", text="assistant substantive reply"),
        ])
        pairs = list(Conversation(id="c1", provider="claude", messages=msgs).iter_pairs())
        assert [(pair.user.id, pair.assistant.id) for pair in pairs] == [("m2", "m3")]

    def test_iter_pairs_empty(self):
        """Line 767: Empty conversation."""
        msgs = MessageCollection(messages=[])
        pairs = list(Conversation(id="c1", provider="claude", messages=msgs).iter_pairs())
        assert len(pairs) == 0

    def test_assistant_only_preserves_exact_assistant_messages(self):
        """assistant_only() should keep only assistant messages in order."""
        msgs = MessageCollection(
            messages=[
                Message(id="u1", role="user", text="user substantive question"),
                Message(id="a1", role="assistant", text="assistant substantive answer"),
                Message(id="s1", role="system", text="system note"),
                Message(id="a2", role="assistant", text="another assistant substantive answer"),
            ]
        )

        filtered = Conversation(id="c1", provider="claude", messages=msgs).assistant_only()

        assert [msg.id for msg in filtered.messages] == ["a1", "a2"]


class TestConversationIterBranches:
    """Test Conversation.iter_branches() method (lines 782-808)."""

    def _make_msg(self, msg_id: str, parent_id: str | None, branch_idx: int) -> Message:
        """Helper: create message with parent relationship."""
        return Message(id=msg_id, role="assistant", text=f"msg {msg_id}",
                      parent_id=parent_id, branch_index=branch_idx)

    def test_iter_branches_multiple_children(self):
        """Lines 799-808: Group messages by parent with multiple children."""
        msgs = MessageCollection(messages=[
            self._make_msg("m1", None, 0),
            self._make_msg("m2", "m1", 0),
            self._make_msg("m3", "m1", 1),
        ])
        branches = list(Conversation(id="c1", provider="claude", messages=msgs).iter_branches())
        assert len(branches) == 1
        assert branches[0][0] == "m1"
        assert len(branches[0][1]) == 2

    def test_iter_branches_single_child(self):
        """Lines 804-805: Only parents with 2+ children are branches."""
        msgs = MessageCollection(messages=[
            self._make_msg("m1", None, 0),
            self._make_msg("m2", "m1", 0),
        ])
        branches = list(Conversation(id="c1", provider="claude", messages=msgs).iter_branches())
        assert len(branches) == 0

    def test_iter_branches_no_parent(self):
        """Line 800: Skip messages without parent_id."""
        msgs = MessageCollection(messages=[
            self._make_msg("m1", None, 0),
            self._make_msg("m2", None, 0),
        ])
        branches = list(Conversation(id="c1", provider="claude", messages=msgs).iter_branches())
        assert len(branches) == 0

    def test_iter_branches_sorted_by_index(self):
        """Lines 806-807: Children sorted by branch_index."""
        msgs = MessageCollection(messages=[
            self._make_msg("m1", None, 0),
            self._make_msg("m3", "m1", 2),
            self._make_msg("m2", "m1", 1),
        ])
        branches = list(Conversation(id="c1", provider="claude", messages=msgs).iter_branches())
        assert len(branches) == 1
        children = branches[0][1]
        assert children[0].branch_index == 1
        assert children[1].branch_index == 2


# =============================================================================
# MESSAGECOLLECTION COVERAGE
# =============================================================================


class TestMessageCollectionConstruction:
    """Test MessageCollection construction and pydantic schema."""

    def test_is_lazy_always_false(self):
        """MessageCollection is always eager — is_lazy is always False."""
        coll = MessageCollection(messages=[])
        assert coll.is_lazy is False

    def test_materialize_is_noop(self):
        """materialize() returns self unchanged (already eager)."""
        msg = Message(id="m1", role="user", text="hello")
        coll = MessageCollection(messages=[msg])
        assert coll.materialize() is coll

    # --- pydantic schema ---

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


# --- merged from test_message_classification.py ---


# Test cases: (role_string, is_user, is_assistant, is_system, is_dialogue, description)
ROLE_TEST_CASES = [
    ("user", True, False, False, True, "user role"),
    ("human", True, False, False, True, "human alias"),
    ("USER", True, False, False, True, "uppercase user"),
    ("assistant", False, True, False, True, "assistant role"),
    ("model", False, True, False, True, "model alias (Gemini)"),
    ("ASSISTANT", False, True, False, True, "uppercase assistant"),
    ("system", False, False, True, False, "system role"),
    ("tool", False, False, False, False, "tool role"),
]


@pytest.mark.parametrize("role,exp_user,exp_asst,exp_sys,exp_dial,desc", ROLE_TEST_CASES)
def test_role_classification_comprehensive(role, exp_user, exp_asst, exp_sys, exp_dial, desc):
    """Comprehensive role classification test.

    Replaces 8 individual role tests with single parametrized test.
    """
    msg = Message(id="1", role=role, text="Test")

    assert msg.is_user == exp_user, f"Wrong is_user for {desc}"
    assert msg.is_assistant == exp_asst, f"Wrong is_assistant for {desc}"
    assert msg.is_system == exp_sys, f"Wrong is_system for {desc}"
    assert msg.is_dialogue == exp_dial, f"Wrong is_dialogue for {desc}"


THINKING_TEST_CASES = [
    # (provider_meta, expected_is_thinking, description)
    ({"content_blocks": [{"type": "thinking", "text": "Analysis..."}]}, True, "content_blocks"),
    ({"isThought": True}, True, "Gemini isThought"),
    ({"raw": {"isThought": True}}, True, "Gemini nested isThought"),
    ({"raw": {"content": {"content_type": "thoughts"}}}, True, "ChatGPT thoughts"),
    ({"raw": {"content": {"content_type": "reasoning_recap"}}}, True, "ChatGPT reasoning"),
    ({}, False, "no markers"),
    (None, False, "None provider_meta"),
]


@pytest.mark.parametrize("provider_meta,expected,desc", THINKING_TEST_CASES)
def test_is_thinking_detection(provider_meta, expected, desc):
    """Comprehensive thinking detection test.

    Replaces 5 individual thinking detection tests.
    """
    msg = Message(
        id="1",
        role="assistant",
        text="Thinking content...",
        provider_meta=provider_meta
    )
    assert msg.is_thinking == expected, f"Wrong is_thinking for {desc}"


TOOL_USE_TEST_CASES = [
    # (role, provider_meta, expected_is_tool_use, description)
    ("tool", {}, True, "role=tool"),
    ("assistant", {"content_blocks": [{"type": "tool_use"}]}, True, "content_blocks tool_use"),
    ("assistant", {"isSidechain": True}, True, "Claude sidechain"),
    ("assistant", {"isMeta": True}, True, "Claude meta marker"),
    ("assistant", {}, False, "normal assistant"),
    ("user", {}, False, "user message"),
]


@pytest.mark.parametrize("role,provider_meta,expected,desc", TOOL_USE_TEST_CASES)
def test_is_tool_use_detection(role, provider_meta, expected, desc):
    """Comprehensive tool use detection test.

    Replaces 6 individual tool use tests.
    """
    msg = Message(
        id="1",
        role=role,
        text="Tool content",
        provider_meta=provider_meta
    )
    assert msg.is_tool_use == expected, f"Wrong is_tool_use for {desc}"


def test_is_tool_use_detection_raw_claude_code():
    """Claude Code raw provider_meta marks tool usage via harmonized path."""
    msg = Message(
        id="m-tool",
        role="assistant",
        text="I will inspect the file",
        provider="claude-code",
        provider_meta={"raw": {
            "type": "assistant",
            "uuid": "m-tool",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect the file"},
                    {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"file_path": "README.md"}},
                ],
            },
        }},
    )
    assert msg.is_tool_use is True
    assert msg.harmonized is not None
    assert msg.harmonized.tool_calls[0].id == "tool-1"
    assert msg.harmonized.tool_calls[0].input == {"file_path": "README.md"}


def test_is_thinking_detection_raw_claude_code():
    """Claude Code raw provider_meta marks thinking via harmonized path."""
    msg = Message(
        id="m-think",
        role="assistant",
        text="",
        provider="claude-code",
        provider_meta={"raw": {
            "type": "assistant",
            "uuid": "m-think",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "step by step"},
                ],
            },
        }},
    )
    assert msg.is_thinking is True
    assert msg.harmonized is not None
    assert msg.harmonized.reasoning_traces[0].text == "step by step"


CONTEXT_DUMP_TEST_CASES = [
    # (text, expected_is_context_dump, description)
    ("```\n```\n```\n```\n```\n```\nCode", True, "6+ backticks (3+ code blocks)"),
    ("<system>Long context</system>\n" * 5, True, "multiple system tags"),
    ("Normal message with one ```code block```", False, "single code block"),
    ("Regular text without markers", False, "plain text"),
]


@pytest.mark.parametrize("text,expected,desc", CONTEXT_DUMP_TEST_CASES)
def test_is_context_dump_detection(text, expected, desc):
    """Comprehensive context dump detection test.

    Replaces 4 individual context dump tests.
    """
    msg = Message(id="1", role="user", text=text)
    assert msg.is_context_dump == expected, f"Wrong is_context_dump for {desc}"


NOISE_TEST_CASES = [
    # (role, text, provider_meta, expected_is_noise, expected_is_substantive, description)
    ("system", "System prompt", {}, True, False, "system message"),
    ("assistant", "A slightly longer message", {}, False, True, "text >10 chars"),
    ("assistant", "Thinking...", {"isThought": True}, False, False, "thinking block (noise via is_thinking in substantive check)"),
    ("tool", "Tool result", {}, True, False, "tool message"),
    ("assistant", "This is a substantial answer with details.", {}, False, True, "substantive"),
    ("user", "Regular question here?", {}, False, True, "user question"),
]


@pytest.mark.parametrize("role,text,meta,exp_noise,exp_subst,desc", NOISE_TEST_CASES)
def test_noise_and_substantive_classification(role, text, meta, exp_noise, exp_subst, desc):
    """Comprehensive noise/substantive classification.

    Replaces 5 individual classification tests.
    """
    msg = Message(id="1", role=role, text=text, provider_meta=meta)

    assert msg.is_noise == exp_noise, f"Wrong is_noise for {desc}"
    assert msg.is_substantive == exp_subst, f"Wrong is_substantive for {desc}"


METADATA_TEST_CASES = [
    # (provider_meta, expected_cost, expected_duration, expected_word_count, description)
    ({"costUSD": 0.005}, 0.005, None, None, "cost only"),
    ({"durationMs": 2500}, None, 2500, None, "duration only"),
    ({"costUSD": 0.01, "durationMs": 5000}, 0.01, 5000, None, "both"),
    ({}, None, None, None, "no metadata"),
]


@pytest.mark.parametrize("meta,exp_cost,exp_dur,exp_words,desc", METADATA_TEST_CASES)
def test_metadata_extraction(meta, exp_cost, exp_dur, exp_words, desc):
    """Comprehensive metadata extraction test.

    Replaces 3 individual metadata tests.
    """
    msg = Message(id="1", role="assistant", text="Response text", provider_meta=meta)

    assert msg.cost_usd == exp_cost, f"Wrong cost_usd for {desc}"
    assert msg.duration_ms == exp_dur, f"Wrong duration_ms for {desc}"
    # word_count is always calculated from text
    assert msg.word_count > 0
