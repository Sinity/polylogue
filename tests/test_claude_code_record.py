"""Tests for ClaudeCodeRecord and related typed models.

Unit tests for ClaudeCodeRecord properties and viewport extraction:
- role mapping (type → role)
- parsed_timestamp (Unix ms/s, ISO strings, edge cases)
- text_content extraction (dict message, typed message, mixed blocks)
- content_blocks_raw extraction
- to_meta() harmonized metadata generation
- Boolean flag properties
- Sub-model conversions (ToolUse, ThinkingBlock, Usage)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from polylogue.sources.providers.claude_code import (
    ClaudeCodeMessageContent,
    ClaudeCodeRecord,
    ClaudeCodeThinkingBlock,
    ClaudeCodeToolUse,
    ClaudeCodeUsage,
    ClaudeCodeUserMessage,
)


# =============================================================================
# ClaudeCodeRecord.role property
# =============================================================================


class TestClaudeCodeRecordRole:
    """Test type→role mapping for all record types."""

    def test_user_type_maps_to_user(self):
        record = ClaudeCodeRecord(type="user")
        assert record.role == "user"

    def test_assistant_type_maps_to_assistant(self):
        record = ClaudeCodeRecord(type="assistant")
        assert record.role == "assistant"

    def test_summary_type_maps_to_system(self):
        record = ClaudeCodeRecord(type="summary")
        assert record.role == "system"

    def test_system_type_maps_to_system(self):
        record = ClaudeCodeRecord(type="system")
        assert record.role == "system"

    def test_file_history_snapshot_maps_to_system(self):
        record = ClaudeCodeRecord(type="file-history-snapshot")
        assert record.role == "system"

    def test_queue_operation_maps_to_system(self):
        record = ClaudeCodeRecord(type="queue-operation")
        assert record.role == "system"

    def test_progress_type_maps_to_tool(self):
        record = ClaudeCodeRecord(type="progress")
        assert record.role == "tool"

    def test_result_type_maps_to_tool(self):
        record = ClaudeCodeRecord(type="result")
        assert record.role == "tool"

    def test_unknown_type_maps_to_unknown(self):
        record = ClaudeCodeRecord(type="init")
        assert record.role == "unknown"

    def test_empty_type_maps_to_unknown(self):
        record = ClaudeCodeRecord(type="")
        assert record.role == "unknown"


# =============================================================================
# ClaudeCodeRecord.parsed_timestamp
# =============================================================================


class TestClaudeCodeRecordTimestamp:
    """Test timestamp parsing from various formats."""

    def test_unix_milliseconds(self):
        """Timestamps > 1e11 are treated as milliseconds."""
        record = ClaudeCodeRecord(type="user", timestamp=1700000000000)
        ts = record.parsed_timestamp
        assert ts is not None
        assert isinstance(ts, datetime)
        # 1700000000 seconds = 2023-11-14T22:13:20
        assert ts.year == 2023

    def test_unix_seconds(self):
        """Timestamps <= 1e11 are treated as seconds."""
        record = ClaudeCodeRecord(type="user", timestamp=1700000000)
        ts = record.parsed_timestamp
        assert ts is not None
        assert ts.year == 2023

    def test_unix_float_milliseconds(self):
        """Float timestamps > 1e11 are milliseconds."""
        record = ClaudeCodeRecord(type="user", timestamp=1700000000000.5)
        ts = record.parsed_timestamp
        assert ts is not None
        assert ts.year == 2023

    def test_iso_string_with_z(self):
        """ISO strings with Z suffix are parsed."""
        record = ClaudeCodeRecord(type="user", timestamp="2025-01-01T00:00:00Z")
        ts = record.parsed_timestamp
        assert ts is not None
        assert ts.year == 2025
        assert ts.month == 1

    def test_iso_string_with_timezone(self):
        """ISO strings with timezone offset are parsed."""
        record = ClaudeCodeRecord(type="user", timestamp="2025-06-15T12:30:00+05:00")
        ts = record.parsed_timestamp
        assert ts is not None
        assert ts.year == 2025

    def test_none_timestamp(self):
        """None timestamp returns None."""
        record = ClaudeCodeRecord(type="user", timestamp=None)
        assert record.parsed_timestamp is None

    def test_invalid_string_returns_none(self):
        """Invalid timestamp string returns None instead of crashing."""
        record = ClaudeCodeRecord(type="user", timestamp="not-a-date")
        assert record.parsed_timestamp is None

    def test_zero_timestamp(self):
        """Zero timestamp is epoch (valid)."""
        record = ClaudeCodeRecord(type="user", timestamp=0)
        ts = record.parsed_timestamp
        assert ts is not None
        assert ts.year == 1970


# =============================================================================
# ClaudeCodeRecord.text_content
# =============================================================================


class TestClaudeCodeRecordTextContent:
    """Test text extraction from various message structures."""

    def test_no_message_returns_empty(self):
        record = ClaudeCodeRecord(type="user", message=None)
        assert record.text_content == ""

    def test_dict_message_string_content(self):
        """Dict message with string content returns the string."""
        record = ClaudeCodeRecord(
            type="user",
            message={"role": "user", "content": "Hello world"},
        )
        assert record.text_content == "Hello world"

    def test_dict_message_text_blocks(self):
        """Dict message with text content blocks extracts text."""
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
            },
        )
        assert record.text_content == "First part\nSecond part"

    def test_typed_message_mixed_blocks_ignores_thinking(self):
        """Typed ClaudeCodeMessageContent only extracts text blocks, not thinking."""
        record = ClaudeCodeRecord(
            type="assistant",
            message=ClaudeCodeMessageContent(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "Analyzing problem"},
                    {"type": "text", "text": "Here is my answer"},
                ],
            ),
        )
        text = record.text_content
        # Typed message path only extracts text, not thinking blocks
        assert text == "Here is my answer"

    def test_dict_message_empty_content(self):
        record = ClaudeCodeRecord(
            type="user",
            message={"role": "user", "content": ""},
        )
        assert record.text_content == ""

    def test_dict_message_no_content_key(self):
        record = ClaudeCodeRecord(
            type="user",
            message={"role": "user"},
        )
        assert record.text_content == ""

    def test_typed_user_message_string_content(self):
        """ClaudeCodeUserMessage with string content."""
        msg = ClaudeCodeUserMessage(content="Hello from user")
        record = ClaudeCodeRecord(type="user", message=msg)
        assert record.text_content == "Hello from user"

    def test_typed_message_content_list(self):
        """ClaudeCodeMessageContent with list of content blocks."""
        msg = ClaudeCodeMessageContent(
            role="assistant",
            content=[{"type": "text", "text": "Response text"}],
        )
        record = ClaudeCodeRecord(type="assistant", message=msg)
        assert record.text_content == "Response text"

    def test_typed_message_empty_content(self):
        msg = ClaudeCodeMessageContent(role="assistant", content=[])
        record = ClaudeCodeRecord(type="assistant", message=msg)
        assert record.text_content == ""


# =============================================================================
# ClaudeCodeRecord.content_blocks_raw
# =============================================================================


class TestClaudeCodeRecordContentBlocksRaw:
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


# =============================================================================
# ClaudeCodeRecord.to_meta()
# =============================================================================


class TestClaudeCodeRecordToMeta:
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


# =============================================================================
# Boolean flag properties
# =============================================================================


class TestClaudeCodeRecordFlags:
    """Test boolean convenience properties."""

    def test_is_context_compaction(self):
        assert ClaudeCodeRecord(type="summary").is_context_compaction is True
        assert ClaudeCodeRecord(type="user").is_context_compaction is False

    def test_is_tool_progress(self):
        assert ClaudeCodeRecord(type="progress").is_tool_progress is True
        assert ClaudeCodeRecord(type="result").is_tool_progress is False

    def test_is_actual_message_user(self):
        assert ClaudeCodeRecord(type="user").is_actual_message is True

    def test_is_actual_message_assistant(self):
        assert ClaudeCodeRecord(type="assistant").is_actual_message is True

    def test_is_actual_message_false_for_others(self):
        assert ClaudeCodeRecord(type="progress").is_actual_message is False
        assert ClaudeCodeRecord(type="summary").is_actual_message is False
        assert ClaudeCodeRecord(type="result").is_actual_message is False


# =============================================================================
# Sub-model conversions
# =============================================================================


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
        assert tc.category is not None  # classify_tool should categorize "Read"

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


# =============================================================================
# Viewport extraction methods (delegate to unified extractors)
# =============================================================================


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
        assert len(blocks) >= 1  # At least the text block
