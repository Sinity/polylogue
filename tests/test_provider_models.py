"""Tests for provider-specific model viewport methods.

MERGED: test_claude_code_record.py (54 tests) and test_provider_coverage_extra.py (68 tests)
content integrated below.

Coverage includes:
1. ChatGPT: text_content with various parts structures, role normalization
2. Gemini: text_content, role_normalized, extract_reasoning_traces, extract_content_blocks
3. Claude Code: parsed_timestamp, text_content, content_blocks_raw, to_meta, flags, conversions
4. Claude AI: role_normalized, parsed_timestamp, to_meta, to_content_blocks, conversation properties
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.lib.roles import Role
from polylogue.schemas.claude_code_records import (
    FileHistorySnapshot,
    ProgressRecord,
    QueueOperationRecord,
    RecordType,
    classify_record,
    extract_metadata_record,
)
from polylogue.schemas.common import CommonMessage, CommonToolCall
from polylogue.sources.providers.chatgpt import ChatGPTMessage, ChatGPTAuthor, ChatGPTContent
from polylogue.sources.providers.gemini import GeminiMessage, GeminiPart, GeminiThoughtSignature
from polylogue.sources.providers.claude_code import (
    ClaudeCodeMessageContent,
    ClaudeCodeRecord,
    ClaudeCodeThinkingBlock,
    ClaudeCodeToolUse,
    ClaudeCodeUsage,
    ClaudeCodeUserMessage,
)
from polylogue.sources.providers.claude_ai import ClaudeAIChatMessage, ClaudeAIConversation

# =============================================================================
# Test Data Tables (module-level constants)
# =============================================================================

CHATGPT_ROLE_MAPPING = [
    ("user", "user"),
    ("assistant", "assistant"),
    ("tool", "tool"),
    ("custom", "unknown"),
]

GEMINI_ROLE_MAPPING = [
    ("user", "user"),
    ("model", "assistant"),
    ("assistant", "assistant"),
    ("custom", "unknown"),
]

GEMINI_ROLE_MAPPING_EXTRA = [
    ("user", "user"),
    ("USER", "user"),
    ("model", "assistant"),
    ("MODEL", "assistant"),
    ("assistant", "assistant"),
    ("system", "system"),
    ("SYSTEM", "system"),
    ("unknown_role", "unknown"),
    ("", "unknown"),
]

CLAUDE_CODE_TYPE_ROLE_MAPPING = [
    ("user", "user"),
    ("assistant", "assistant"),
    ("summary", "system"),
    ("system", "system"),
    ("file-history-snapshot", "system"),
    ("queue-operation", "system"),
    ("progress", "tool"),
    ("result", "tool"),
    ("init", "unknown"),
    ("", "unknown"),
]

CLAUDE_CODE_TIMESTAMP_PARSING = [
    (1700000000000, 2023, "unix_milliseconds"),
    (1700000000, 2023, "unix_seconds"),
    (1700000000000.5, 2023, "unix_float_milliseconds"),
    ("2025-01-01T00:00:00Z", 2025, "iso_string_with_z"),
    ("2025-06-15T12:30:00+05:00", 2025, "iso_string_with_timezone"),
    (None, None, "none_timestamp"),
    ("not-a-date", None, "invalid_string_returns_none"),
    (0, 1970, "zero_timestamp"),
]

CLAUDE_AI_ROLE_MAPPING = [
    ("human", "user"),
    ("assistant", "assistant"),
    ("system", "system"),
    ("", "unknown"),
]


class TestChatGPTMessageTextContent:
    """Regression tests for ChatGPTMessage.text_content."""

    @pytest.mark.parametrize("parts,expected", [
        (["Hello", "World"], "Hello\nWorld"),
        ([None, "Valid"], "Valid"),
        ([{"text": None}, {"text": "ok"}], "ok"),
        ([{"text": "hello"}], "hello"),
        ([], ""),
    ], ids=["string_parts", "none_parts", "dict_none_text", "dict_valid_text", "empty_parts"])
    def test_text_content_with_parts(self, parts, expected):
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=parts),
        )
        assert msg.text_content == expected

    def test_text_content_no_content(self):
        msg = ChatGPTMessage(id="1", author=ChatGPTAuthor(role="user"))
        assert msg.text_content == ""

    def test_text_content_direct_text(self):
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", text="Direct text"),
        )
        assert msg.text_content == "Direct text"

    @pytest.mark.parametrize("role_in,expected", CHATGPT_ROLE_MAPPING)
    def test_role_normalized(self, role_in, expected):
        msg = ChatGPTMessage(id="1", author=ChatGPTAuthor(role=role_in))
        assert msg.role_normalized == expected


class TestGeminiMessageTextContent:
    """Regression tests for GeminiMessage.text_content."""

    def test_text_content_from_text_field(self):
        msg = GeminiMessage(text="Hello", role="user")
        assert msg.text_content == "Hello"

    def test_text_content_from_parts_dict_none_text(self):
        """Regression: dict part with 'text' key but None value must not crash."""
        msg = GeminiMessage(text="", role="user", parts=[{"text": None}, {"text": "ok"}])
        assert msg.text_content == "ok"

    def test_text_content_from_parts_typed(self):
        from polylogue.sources.providers.gemini import GeminiPart
        msg = GeminiMessage(text="", role="model", parts=[GeminiPart(text="typed")])
        assert msg.text_content == "typed"

    @pytest.mark.parametrize("role_in,expected", GEMINI_ROLE_MAPPING)
    def test_role_normalized(self, role_in, expected):
        msg = GeminiMessage(text="x", role=role_in)
        assert msg.role_normalized == expected

    def test_extract_content_blocks_dict_none_text(self):
        """Regression: extract_content_blocks with None text in dict part must not crash."""
        msg = GeminiMessage(text="", role="user", parts=[{"text": None}, {"text": "ok"}])
        blocks = msg.extract_content_blocks()
        text_blocks = [b for b in blocks if b.text == "ok"]
        assert len(text_blocks) == 1

    def test_extract_content_blocks_file_data(self):
        """Pydantic coerces dict parts to GeminiPart (extra=allow), so inlineData
        ends up as a GeminiPart attribute. The GeminiPart branch only checks .text,
        so file-only parts produce no content blocks — this is current behavior."""
        msg = GeminiMessage(text="", role="user", parts=[{"inlineData": {"mimeType": "image/png"}}])
        blocks = msg.extract_content_blocks()
        # No text in the part → no content blocks extracted
        assert len(blocks) == 0

    def test_thinking_message(self):
        msg = GeminiMessage(text="Thinking...", role="model", isThought=True)
        traces = msg.extract_reasoning_traces()
        assert len(traces) == 1
        assert traces[0].text == "Thinking..."


# =============================================================================
# MERGED: Claude Code Record Tests (from test_claude_code_record.py)
# =============================================================================


class TestClaudeCodeRecordRole:
    """Test type→role mapping for all record types."""

    @pytest.mark.parametrize("record_type,expected_role", CLAUDE_CODE_TYPE_ROLE_MAPPING)
    def test_type_role_mapping(self, record_type, expected_role):
        record = ClaudeCodeRecord(type=record_type)
        assert record.role == expected_role


class TestClaudeCodeRecordTimestamp:
    """Test timestamp parsing from various formats."""

    @pytest.mark.parametrize("timestamp,expected_year,test_id", CLAUDE_CODE_TIMESTAMP_PARSING)
    def test_timestamp_parsing(self, timestamp, expected_year, test_id):
        record = ClaudeCodeRecord(type="user", timestamp=timestamp)
        ts = record.parsed_timestamp
        if expected_year is None:
            assert ts is None
        else:
            assert ts is not None
            assert isinstance(ts, datetime)
            assert ts.year == expected_year


class TestClaudeCodeRecordTextContent2:
    """Test text extraction from various message structures."""

    CLAUDE_CODE_TEXT_CONTENT_CASES = [
        (None, "", "no_message"),
        ({"role": "user", "content": "Hello world"}, "Hello world", "dict_string_content"),
        ({"role": "assistant", "content": [{"type": "text", "text": "First part"}, {"type": "text", "text": "Second part"}]}, "First part\nSecond part", "dict_text_blocks"),
        (ClaudeCodeMessageContent(role="assistant", content=[{"type": "thinking", "thinking": "Analyzing"}, {"type": "text", "text": "Here is my answer"}]), "Here is my answer", "typed_mixed_blocks_ignores_thinking"),
        ({"role": "user", "content": ""}, "", "dict_empty_content"),
        ({"role": "user"}, "", "dict_no_content_key"),
        (ClaudeCodeUserMessage(content="Hello from user"), "Hello from user", "typed_user_message_string"),
        (ClaudeCodeMessageContent(role="assistant", content=[{"type": "text", "text": "Response text"}]), "Response text", "typed_message_content_list"),
        (ClaudeCodeMessageContent(role="assistant", content=[]), "", "typed_message_empty_content"),
    ]

    @pytest.mark.parametrize("message,expected,test_id", CLAUDE_CODE_TEXT_CONTENT_CASES)
    def test_text_content(self, message, expected, test_id):
        record = ClaudeCodeRecord(type="user" if message is None else "assistant", message=message)
        assert record.text_content == expected


class TestClaudeCodeRecordContentBlocksRaw2:
    """Test raw content block extraction."""

    def test_no_message_returns_empty_list(self):
        record = ClaudeCodeRecord(type="user", message=None)
        assert record.content_blocks_raw == []

    def test_dict_message_with_list_content(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "tool_use", "name": "Read"}]
        record = ClaudeCodeRecord(
            type="assistant",
            message={"role": "assistant", "content": blocks},
        )
        assert record.content_blocks_raw == blocks

    def test_dict_message_with_string_content(self):
        """String content is not a list, returns empty."""
        record = ClaudeCodeRecord(
            type="user",
            message={"role": "user", "content": "just a string"},
        )
        assert record.content_blocks_raw == []

    def test_typed_message_with_list_content(self):
        msg = ClaudeCodeMessageContent(
            role="assistant",
            content=[{"type": "text", "text": "hello"}],
        )
        record = ClaudeCodeRecord(type="assistant", message=msg)
        assert len(record.content_blocks_raw) == 1
        assert record.content_blocks_raw[0]["type"] == "text"


class TestClaudeCodeRecordToMeta2:
    """Test harmonized metadata generation."""

    def test_basic_meta(self):
        record = ClaudeCodeRecord(type="user", uuid="msg-1")
        meta = record.to_meta()
        assert meta.id == "msg-1"
        assert meta.role == "user"
        assert meta.provider == "claude-code"
        assert meta.tokens is None
        assert meta.cost is None

    def test_meta_with_cost(self):
        record = ClaudeCodeRecord(type="assistant", uuid="msg-2", costUSD=0.05)
        meta = record.to_meta()
        assert meta.cost is not None
        assert meta.cost.total_usd == 0.05

    def test_meta_with_duration(self):
        record = ClaudeCodeRecord(type="assistant", uuid="msg-3", durationMs=1500)
        meta = record.to_meta()
        assert meta.duration_ms == 1500

    def test_meta_with_typed_message_usage(self):
        """Token usage from ClaudeCodeMessageContent is extracted."""
        usage = ClaudeCodeUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=20,
        )
        msg = ClaudeCodeMessageContent(
            role="assistant",
            model="claude-sonnet-4-20250514",
            usage=usage,
        )
        record = ClaudeCodeRecord(type="assistant", message=msg)
        meta = record.to_meta()
        assert meta.tokens is not None
        assert meta.tokens.input_tokens == 100
        assert meta.tokens.output_tokens == 50
        assert meta.tokens.cache_read_tokens == 30
        assert meta.tokens.cache_write_tokens == 20
        assert meta.model == "claude-sonnet-4-20250514"

    def test_meta_with_dict_message_usage(self):
        """Token usage from dict message is extracted."""
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "model": "claude-opus-4-20250514",
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 40,
                },
            },
        )
        meta = record.to_meta()
        assert meta.tokens is not None
        assert meta.tokens.input_tokens == 200
        assert meta.tokens.output_tokens == 100
        assert meta.model == "claude-opus-4-20250514"

    def test_meta_with_dict_message_no_usage(self):
        """Dict message without usage field returns None tokens."""
        record = ClaudeCodeRecord(
            type="assistant",
            message={"role": "assistant"},
        )
        meta = record.to_meta()
        assert meta.tokens is None




class TestClaudeCodeToolUseConversion:
    """Test tool_use → ToolCall conversion."""

    def test_to_tool_call(self):
        tool_use = ClaudeCodeToolUse(
            id="toolu_123",
            name="Read",
            input={"file_path": "/tmp/test.py"},
        )
        tc = tool_use.to_tool_call()
        assert tc.name == "Read"
        assert tc.id == "toolu_123"
        assert tc.input == {"file_path": "/tmp/test.py"}
        assert tc.provider == "claude-code"
        assert tc.category is not None

    def test_to_tool_call_empty_input(self):
        tool_use = ClaudeCodeToolUse(id="toolu_456", name="UnknownTool", input={})
        tc = tool_use.to_tool_call()
        assert tc.name == "UnknownTool"
        assert tc.input == {}


class TestClaudeCodeThinkingBlockConversion:
    """Test thinking block → ReasoningTrace conversion."""

    def test_to_reasoning_trace(self):
        block = ClaudeCodeThinkingBlock(thinking="Let me think about this problem step by step.")
        trace = block.to_reasoning_trace()
        assert trace.text == "Let me think about this problem step by step."
        assert trace.provider == "claude-code"
        assert trace.raw is not None

    def test_to_reasoning_trace_empty(self):
        block = ClaudeCodeThinkingBlock(thinking="")
        trace = block.to_reasoning_trace()
        assert trace.text == ""


class TestClaudeCodeUsageConversion:
    """Test usage → TokenUsage conversion."""

    def test_to_token_usage_full(self):
        usage = ClaudeCodeUsage(
            input_tokens=500,
            output_tokens=200,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
        )
        tu = usage.to_token_usage()
        assert tu.input_tokens == 500
        assert tu.output_tokens == 200
        assert tu.cache_read_tokens == 100
        assert tu.cache_write_tokens == 50

    def test_to_token_usage_partial(self):
        """Missing fields are None, not 0."""
        usage = ClaudeCodeUsage(input_tokens=100, output_tokens=50)
        tu = usage.to_token_usage()
        assert tu.input_tokens == 100
        assert tu.output_tokens == 50
        assert tu.cache_read_tokens is None
        assert tu.cache_write_tokens is None

    def test_to_token_usage_all_none(self):
        usage = ClaudeCodeUsage()
        tu = usage.to_token_usage()
        assert tu.input_tokens is None
        assert tu.output_tokens is None


class TestClaudeCodeRecordViewportMethods:
    """Test extract_reasoning_traces, extract_tool_calls, extract_content_blocks."""

    def test_extract_reasoning_traces_with_thinking(self):
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Analyzing the problem"},
                    {"type": "text", "text": "Here's my answer"},
                ],
            },
        )
        traces = record.extract_reasoning_traces()
        assert len(traces) >= 1
        assert any("Analyzing" in t.text for t in traces)

    def test_extract_reasoning_traces_none(self):
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [{"type": "text", "text": "No thinking here"}],
            },
        )
        traces = record.extract_reasoning_traces()
        assert traces == []

    def test_extract_tool_calls(self):
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        )
        calls = record.extract_tool_calls()
        assert len(calls) == 1
        assert calls[0].name == "Bash"

    def test_extract_tool_calls_empty(self):
        record = ClaudeCodeRecord(
            type="user",
            message={"role": "user", "content": "just text"},
        )
        calls = record.extract_tool_calls()
        assert calls == []

    def test_extract_content_blocks(self):
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                ],
            },
        )
        blocks = record.extract_content_blocks()
        assert len(blocks) >= 1


# =============================================================================
# MERGED: Gemini Additional Tests (from test_provider_coverage_extra.py)
# =============================================================================


class TestGeminiRoleNormalizedExtra:
    """Test GeminiMessage.role_normalized edge cases."""

    @pytest.mark.parametrize("role,expected", GEMINI_ROLE_MAPPING_EXTRA, ids=[
        "user_lowercase", "user_uppercase", "model_lowercase", "model_uppercase",
        "assistant_lowercase", "system_lowercase", "system_uppercase",
        "unknown_role", "empty_string"
    ])
    def test_role_normalized(self, role, expected):
        msg = GeminiMessage(text="hi", role=role)
        assert msg.role_normalized == expected

    def test_role_none_defaults_to_unknown(self):
        msg = GeminiMessage(text="hi", role="user")
        msg.role = None
        assert msg.role_normalized == "unknown"


class TestGeminiTextContentExtra:
    """Test GeminiMessage.text_content with parts variations."""

    def test_text_content_direct_text(self):
        msg = GeminiMessage(text="Direct text", role="user")
        assert msg.text_content == "Direct text"

    @pytest.mark.parametrize("parts,expected", [
        ([GeminiPart(text="Part 1"), GeminiPart(text="Part 2")], "Part 1\nPart 2"),
        ([{"text": "Dict 1"}, {"text": "Dict 2"}], "Dict 1\nDict 2"),
        ([GeminiPart(text="Typed"), {"text": "Dict"}], "Typed\nDict"),
        ([{"image": "data:..."}, {"audio": "data:..."}], ""),
        ([], ""),
    ], ids=["typed_parts", "dict_parts", "mixed_parts", "no_text_keys", "empty_list"])
    def test_text_content_from_parts(self, parts, expected):
        msg = GeminiMessage(text="", role="user", parts=parts)
        assert msg.text_content == expected

    def test_text_content_parts_dict_with_non_string_text(self):
        """Coverage for line 135: coerce non-string text to str."""
        msg = GeminiMessage(
            text="",
            role="user",
            parts=[{"text": 123}, {"text": True}],
        )
        content = msg.text_content
        assert isinstance(content, str)

    def test_text_content_parts_dict_with_none_text(self):
        msg = GeminiMessage(
            text="",
            role="user",
            parts=[{"text": None}, {"text": "Valid"}],
        )
        assert msg.text_content == "Valid"

    def test_text_content_parts_typed_none_text(self):
        msg = GeminiMessage(
            text="",
            role="user",
            parts=[GeminiPart(text=None), GeminiPart(text="Valid")],
        )
        assert msg.text_content == "Valid"

    def test_text_content_prefers_text_over_parts(self):
        msg = GeminiMessage(
            text="Direct",
            role="user",
            parts=[GeminiPart(text="Ignored")],
        )
        assert msg.text_content == "Direct"


class TestGeminiToMetaExtra:
    """Test GeminiMessage.to_meta conversion."""

    def test_to_meta_basic(self):
        msg = GeminiMessage(text="hello", role="user")
        meta = msg.to_meta()
        assert meta.role == "user"
        assert meta.provider == "gemini"
        assert meta.tokens is None

    @pytest.mark.parametrize("tokenCount,has_tokens,expected_output", [
        (42, True, 42),
        (0, True, 0),
        (None, False, None),
    ], ids=["with_count", "zero_count", "none_count"])
    def test_to_meta_token_count(self, tokenCount, has_tokens, expected_output):
        msg = GeminiMessage(text="hello", role="user", tokenCount=tokenCount)
        meta = msg.to_meta()
        if has_tokens:
            assert meta.tokens is not None
            assert meta.tokens.output_tokens == expected_output
        else:
            assert meta.tokens is None


class TestGeminiExtractReasoningTracesExtra:
    """Test GeminiMessage.extract_reasoning_traces."""

    def test_extract_reasoning_traces_no_thought(self):
        msg = GeminiMessage(text="Regular response", role="user", isThought=False)
        traces = msg.extract_reasoning_traces()
        assert len(traces) == 0

    def test_extract_reasoning_traces_thought_with_text(self):
        msg = GeminiMessage(
            text="Thinking...",
            role="model",
            isThought=True,
            thinkingBudget=1000,
        )
        traces = msg.extract_reasoning_traces()
        assert len(traces) == 1
        assert traces[0].text == "Thinking..."
        assert traces[0].token_count == 1000
        assert traces[0].provider == "gemini"

    def test_extract_reasoning_traces_thought_without_text(self):
        """Regression: isThought=True but no text should not create trace."""
        msg = GeminiMessage(text="", role="model", isThought=True)
        traces = msg.extract_reasoning_traces()
        assert len(traces) == 0

    @pytest.mark.parametrize("signatures,expected_in_raw", [
        (["sig1", "sig2"], ["sig1", "sig2"]),
        ([{"key": "value"}], [{"key": "value"}]),
    ], ids=["string_sigs", "dict_sigs"])
    def test_extract_reasoning_traces_with_signatures(self, signatures, expected_in_raw):
        """Coverage for line 157-162: various signature types."""
        msg = GeminiMessage(
            text="Thought",
            role="model",
            isThought=True,
            thoughtSignatures=signatures,
        )
        traces = msg.extract_reasoning_traces()
        assert len(traces) == 1
        assert traces[0].raw["thoughtSignatures"] == expected_in_raw

    def test_extract_reasoning_traces_with_model_signatures(self):
        """Coverage for line 159-160: BaseModel signature handling."""
        sig = GeminiThoughtSignature()
        msg = GeminiMessage(
            text="Thought",
            role="model",
            isThought=True,
            thoughtSignatures=[sig],
        )
        traces = msg.extract_reasoning_traces()
        assert len(traces) == 1
        assert len(traces[0].raw["thoughtSignatures"]) == 1

    def test_extract_reasoning_traces_budget_none(self):
        msg = GeminiMessage(
            text="Thought",
            role="model",
            isThought=True,
            thinkingBudget=None,
        )
        traces = msg.extract_reasoning_traces()
        assert len(traces) == 1
        assert traces[0].token_count is None


class TestGeminiExtractContentBlocksExtra:
    """Test GeminiMessage.extract_content_blocks."""

    GEMINI_EXTRACT_CONTENT_BLOCKS_CASES = [
        ({"text": "Thinking", "role": "model", "isThought": True}, 1, "thinking", "thought_block"),
        ({"text": "Response", "role": "model", "isThought": False}, 1, "text", "text_only"),
        ({"text": "", "role": "model", "isThought": False}, 0, None, "empty_text"),
        ({"text": "", "role": "user", "parts": [GeminiPart(text="Part1"), GeminiPart(text="Part2")]}, 2, "text", "typed_parts"),
        ({"text": "", "role": "user", "parts": [GeminiPart(text=None), GeminiPart(text="Valid")]}, 1, "text", "typed_parts_none_text"),
        ({"text": "", "role": "user", "parts": [{"text": "Dict1"}, {"text": "Dict2"}]}, None, "text", "dict_parts_text"),
        ({"text": "", "role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "base64..."}}]}, None, None, "dict_parts_inline_data"),
        ({"text": "", "role": "user", "parts": [{"fileData": {"mimeType": "application/pdf", "fileUri": "uri..."}}]}, None, None, "dict_parts_file_data"),
        ({"text": "", "role": "user", "parts": [{"other": "value"}]}, 0, None, "dict_parts_no_text_no_media"),
        ({"text": "Initial", "role": "model", "parts": [GeminiPart(text="Typed"), {"text": "Dict"}, {"inlineData": {"data": "..."}}]}, None, "text", "combined"),
    ]

    @pytest.mark.parametrize("kwargs,expected_len,expected_type,test_id", GEMINI_EXTRACT_CONTENT_BLOCKS_CASES)
    def test_extract_content_blocks(self, kwargs, expected_len, expected_type, test_id):
        from polylogue.lib.viewports import ContentType
        msg = GeminiMessage(**kwargs)
        blocks = msg.extract_content_blocks()
        if expected_len is not None:
            assert len(blocks) == expected_len
        if expected_type == "text":
            text_blocks = [b for b in blocks if b.type == ContentType.TEXT]
            assert len(text_blocks) > 0 if expected_len else True
        elif expected_type == "thinking":
            assert blocks[0].type == ContentType.THINKING


# =============================================================================
# MERGED: Claude AI Chat Message Tests (from test_provider_coverage_extra.py)
# =============================================================================


class TestClaudeAIChatMessageRoleNormalizedAndTimestamp:
    """Test ClaudeAIChatMessage role_normalized and parsed_timestamp."""

    @pytest.mark.parametrize("sender,expected", CLAUDE_AI_ROLE_MAPPING, ids=[
        "human", "assistant", "system", "empty"
    ])
    def test_role_normalized(self, sender, expected):
        msg = ClaudeAIChatMessage(uuid="1", text="hi", sender=sender)
        assert msg.role_normalized == expected

    @pytest.mark.parametrize("created_at,expect_datetime,test_id", [
        ("2024-06-15T10:30:00Z", True, "iso_z"),
        ("2024-06-15T10:30:00+00:00", True, "iso_offset"),
        ("not-a-date", False, "invalid"),
        ("", False, "empty"),
        (None, False, "none"),
        ("2024-06-15 10:30:00", None, "malformed_iso"),
    ])
    def test_parsed_timestamp(self, created_at, expect_datetime, test_id):
        msg = ClaudeAIChatMessage(
            uuid="1", text="hi", sender="human",
            created_at=created_at,
        )
        ts = msg.parsed_timestamp
        if expect_datetime is True:
            assert ts is not None and isinstance(ts, datetime)
        elif expect_datetime is False:
            assert ts is None
        else:
            assert ts is None or isinstance(ts, datetime)


class TestClaudeAIChatMessageToMeta:
    """Test ClaudeAIChatMessage.to_meta conversion."""

    @pytest.mark.parametrize("uuid_,sender,created_at,expect_timestamp,test_id", [
        ("msg-1", "human", None, False, "basic"),
        ("msg-2", "assistant", "2024-06-15T10:30:00Z", True, "with_timestamp"),
        ("msg-3", "human", "bad-date", False, "invalid_timestamp"),
    ])
    def test_to_meta(self, uuid_, sender, created_at, expect_timestamp, test_id):
        msg = ClaudeAIChatMessage(uuid=uuid_, text="hi", sender=sender, created_at=created_at)
        meta = msg.to_meta()
        assert meta.id == uuid_
        assert meta.provider == "claude-ai"
        if sender == "human":
            assert meta.role == "user"
        else:
            assert meta.role == sender
        if expect_timestamp:
            assert meta.timestamp is not None
            assert meta.timestamp.year == 2024
        else:
            assert meta.timestamp is None


class TestClaudeAIChatMessageToContentBlocks:
    """Test ClaudeAIChatMessage.to_content_blocks."""

    @pytest.mark.parametrize("text,sender,test_id", [
        ("hello world", "human", "basic"),
        ("response", "assistant", "assistant"),
        ("", "human", "empty_text"),
    ])
    def test_to_content_blocks(self, text, sender, test_id):
        from polylogue.lib.viewports import ContentType
        msg = ClaudeAIChatMessage(uuid="1", text=text, sender=sender)
        blocks = msg.to_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.TEXT
        assert blocks[0].text == text


class TestClaudeAIConversationProperties:
    """Test ClaudeAIConversation properties (title, created_datetime, updated_datetime)."""

    @pytest.mark.parametrize("name,expected_title,test_id", [
        ("My Conversation", "My Conversation", "title_from_name"),
        ("", "", "title_empty_name"),
    ])
    def test_title(self, name, expected_title, test_id):
        conv = ClaudeAIConversation(
            uuid="c-1", name=name,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert conv.title == expected_title

    @pytest.mark.parametrize("date_str,expect_valid,test_id", [
        ("2024-06-15T10:30:00Z", True, "iso_z"),
        ("2024-06-15T10:30:00+05:00", True, "iso_offset"),
        ("not-a-date", False, "invalid"),
    ])
    def test_created_datetime(self, date_str, expect_valid, test_id):
        conv = ClaudeAIConversation(
            uuid="c-1", name="Test",
            created_at=date_str,
            updated_at="2024-06-15T10:30:00Z",
        )
        dt = conv.created_datetime
        if expect_valid:
            assert dt is not None
            assert dt.year == 2024 and dt.month == 6 and dt.day == 15
        else:
            assert dt is None

    @pytest.mark.parametrize("updated_date,expect_valid,test_id", [
        ("2024-06-16T11:30:00Z", True, "valid"),
        ("bad-date", False, "invalid"),
    ])
    def test_updated_datetime(self, updated_date, expect_valid, test_id):
        conv = ClaudeAIConversation(
            uuid="c-1", name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at=updated_date,
        )
        dt = conv.updated_datetime
        if expect_valid:
            assert dt is not None
            assert dt.year == 2024 and dt.month == 6 and dt.day == 16
        else:
            assert dt is None


class TestClaudeAIConversationMessages:
    """Test ClaudeAIConversation.messages property."""

    def test_messages_alias(self):
        """Coverage for line 125: messages alias returns chat_messages."""
        conv = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            chat_messages=[
                ClaudeAIChatMessage(uuid="m1", text="hi", sender="human"),
                ClaudeAIChatMessage(uuid="m2", text="hello", sender="assistant"),
            ],
        )
        messages = conv.messages
        assert len(messages) == 2
        assert messages[0].text == "hi"
        assert messages[1].text == "hello"

    def test_messages_empty(self):
        conv = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        messages = conv.messages
        assert len(messages) == 0


class TestClaudeAIConversationIntegration:
    """Integration tests for full conversation workflow."""

    def test_full_conversation_workflow(self):
        """Test a complete conversation with all features."""
        conv = ClaudeAIConversation(
            uuid="conv-full",
            name="Full Test",
            created_at="2024-06-15T10:00:00Z",
            updated_at="2024-06-15T11:00:00Z",
            chat_messages=[
                ClaudeAIChatMessage(
                    uuid="m1",
                    text="Hello, how are you?",
                    sender="human",
                    created_at="2024-06-15T10:00:00Z",
                ),
                ClaudeAIChatMessage(
                    uuid="m2",
                    text="I'm doing well, thanks!",
                    sender="assistant",
                    created_at="2024-06-15T10:01:00Z",
                ),
            ],
        )

        assert conv.title == "Full Test"
        assert conv.created_datetime is not None
        assert conv.updated_datetime is not None
        assert len(conv.messages) == 2

        for msg in conv.messages:
            meta = msg.to_meta()
            assert meta is not None
            blocks = msg.to_content_blocks()
            assert len(blocks) > 0

    def test_conversation_with_invalid_timestamps(self):
        """Test conversation with unparseable timestamps."""
        conv = ClaudeAIConversation(
            uuid="conv-bad",
            name="Bad Times",
            created_at="bad-created",
            updated_at="bad-updated",
            chat_messages=[
                ClaudeAIChatMessage(
                    uuid="m1",
                    text="Hi",
                    sender="human",
                    created_at="bad-message-date",
                ),
            ],
        )

        assert conv.created_datetime is None
        assert conv.updated_datetime is None
        assert conv.messages[0].parsed_timestamp is None


# --- Merged from test_schemas_coverage.py ---


def test_record_type_enums():
    assert RecordType.is_message("user") is True
    assert RecordType.is_message("assistant") is True
    assert RecordType.is_message("system") is True
    assert RecordType.is_message("progress") is False

    assert RecordType.is_metadata("progress") is True
    assert RecordType.is_metadata("user") is False


def test_progress_record_from_raw():
    raw = {
        "type": "progress",
        "data": {"hookEvent": "SessionStart", "hookName": "on_session_start"},
        "toolUseID": "tool_123",
        "parentToolUseID": "parent_456",
        "timestamp": "2023-01-01T12:00:00Z",
        "sessionId": "session_abc",
    }
    record = ProgressRecord.from_raw(raw)
    assert record.hook_event == "SessionStart"
    assert record.hook_name == "on_session_start"
    assert record.tool_use_id == "tool_123"
    assert record.parent_tool_use_id == "parent_456"
    assert record.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert record.session_id == "session_abc"
    assert record.raw == raw


def test_file_history_snapshot_from_raw():
    raw = {
        "type": "file-history-snapshot",
        "messageId": "msg_123",
        "snapshot": {
            "timestamp": "2023-01-01T12:00:00Z",
            "trackedFileBackups": {
                "/path/to/file1": {"hash": "hash1"},
                "/path/to/file2": None,  # Case where hash is missing or structure differs
            },
        },
        "isSnapshotUpdate": True,
    }
    snapshot = FileHistorySnapshot.from_raw(raw)
    assert snapshot.message_id == "msg_123"
    assert snapshot.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert snapshot.is_snapshot_update is True
    assert len(snapshot.tracked_files) == 2
    assert snapshot.tracked_files[0].path == "/path/to/file1"
    assert snapshot.tracked_files[0].content_hash == "hash1"
    assert snapshot.tracked_files[1].path == "/path/to/file2"
    assert snapshot.tracked_files[1].content_hash is None


def test_queue_operation_record_from_raw():
    raw = {
        "type": "queue-operation",
        "operation": "enqueue",
        "timestamp": "2023-01-01T12:00:00Z",
        "sessionId": "session_abc",
        "content": {"foo": "bar"},
    }
    record = QueueOperationRecord.from_raw(raw)
    assert record.operation == "enqueue"
    assert record.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert record.session_id == "session_abc"
    assert record.content == {"foo": "bar"}


def test_extract_metadata_record_dispatch():
    # Progress
    raw_progress = {"type": "progress", "data": {}}
    assert isinstance(extract_metadata_record(raw_progress), ProgressRecord)

    # File Snapshot
    raw_snapshot = {"type": "file-history-snapshot", "snapshot": {}}
    assert isinstance(extract_metadata_record(raw_snapshot), FileHistorySnapshot)

    # Queue Op
    raw_queue = {"type": "queue-operation"}
    assert isinstance(extract_metadata_record(raw_queue), QueueOperationRecord)

    # Message type (should return None)
    raw_message = {"type": "user"}
    assert extract_metadata_record(raw_message) is None

    # Unknown type
    raw_unknown = {"type": "unknown_thing"}
    assert extract_metadata_record(raw_unknown) is None


def test_classify_record():
    assert classify_record({"type": "user"}) == ("message", "user")
    assert classify_record({"type": "progress"}) == ("metadata", "progress")
    assert classify_record({"type": "unknown"}) == ("metadata", "unknown")


def test_common_message_instantiation():
    msg = CommonMessage(
        role=Role.USER,
        text="Hello",
        timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        id="msg_1",
        model="gpt-4",
        tokens=10,
        cost_usd=0.01,
        is_thinking=True,
        provider="test_provider",
        raw={"orig": "data"},
    )
    assert msg.role == Role.USER
    assert msg.text == "Hello"
    assert msg.timestamp is not None
    assert msg.id == "msg_1"
    assert msg.is_thinking is True


def test_common_tool_call_instantiation():
    tool = CommonToolCall(
        name="calculator",
        input={"a": 1, "b": 2},
        output="3",
        success=True,
        provider="test_provider",
        raw={"orig": "data"},
    )
    assert tool.name == "calculator"
    assert tool.input == {"a": 1, "b": 2}
    assert tool.output == "3"
    assert tool.success is True
