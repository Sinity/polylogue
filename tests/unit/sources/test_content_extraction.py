"""Tests for provider content extraction edge cases.

Covers genuinely unique content extraction scenarios NOT tested in test_models.py
or test_edge_cases.py. Originally 39 tests; redundant tests were removed during
consolidation — canonical provider model tests live in test_models.py.

Unique coverage:
- _text_from_blocks static method (direct call, not via property)
- ChatGPT: parts=None (not empty list), None within parts list
- Gemini: fileData parts, code execution blocks
- Multi-provider integration: complex multi-block scenarios
- Boundary: empty thinking, only-thinking messages, all-None parts,
  long text, unicode, special characters in thinking
"""

from __future__ import annotations

import pytest

from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.chatgpt import (
    ChatGPTAuthor,
    ChatGPTContent,
    ChatGPTMessage,
)
from polylogue.sources.providers.gemini import (
    GeminiMessage,
    GeminiPart,
)


# =============================================================================
# ClaudeCodeRecord: direct _text_from_blocks call (not via text_content property)
# =============================================================================


class TestClaudeCodeTextFromBlocks:
    """Test the static method directly — bypasses text_content dispatch logic."""

    def test_text_from_blocks_excludes_thinking(self):
        blocks = [
            {"type": "thinking", "thinking": "Let me reason about this..."},
            {"type": "text", "text": "Here is my answer."},
        ]
        text = ClaudeCodeRecord._text_from_blocks(blocks)
        assert "Let me reason" not in text
        assert "Here is my answer." in text


# =============================================================================
# ChatGPT: None-guard edge cases unique to this file
# =============================================================================


class TestChatGPTPartsNoneGuards:
    """Edge cases for ChatGPT parts that differ from simple empty-list."""

    def test_none_parts_returns_empty(self):
        """parts=None (not empty list) must not crash."""
        msg = ChatGPTMessage(
            id="m1",
            author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", parts=None),
        )
        assert msg.text_content == ""

    def test_none_part_in_list_skipped(self):
        """None values in parts list should not crash."""
        msg = ChatGPTMessage(
            id="m1",
            author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", parts=["text", None]),
        )
        text = msg.text_content
        assert "text" in text


# =============================================================================
# Gemini: fileData and code execution blocks
# =============================================================================


class TestGeminiSpecialParts:
    """Gemini-specific content types not covered in test_models.py."""

    def test_parts_with_file_data(self):
        """Parts with fileData should not crash."""
        msg = GeminiMessage(
            text="",
            role="model",
            parts=[{"fileData": {"mimeType": "application/pdf", "fileUri": "uri..."}}],
        )
        assert msg.text_content == ""

    def test_code_execution_blocks(self):
        msg = GeminiMessage(
            text="Here's the code",
            role="model",
            executableCode={"language": "python", "code": "print('hello')"},
            codeExecutionResult={"outcome": "OK", "output": "hello"},
        )
        blocks = msg.extract_content_blocks()
        types = [b.type.value for b in blocks]
        assert "code" in types
        assert "tool_result" in types


# =============================================================================
# Multi-provider integration: complex multi-block scenarios
# =============================================================================


class TestMultiblockIntegration:
    """Complex multiblock scenarios with mixed content types."""

    def test_claude_code_complex_blocks(self):
        """Thinking + multiple text blocks + tool use."""
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Step 1: analyze"},
                    {"type": "thinking", "thinking": "Step 2: plan"},
                    {"type": "text", "text": "I'll help with that."},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                    {"type": "text", "text": "Running command..."},
                ],
            },
        )
        text = record.text_content
        assert "Step 1" not in text
        assert "Step 2" not in text
        assert "I'll help" in text
        assert "Running command" in text
        traces = record.extract_reasoning_traces()
        assert len(traces) == 2

    def test_chatgpt_multimodal_complex(self):
        """Mixed string and dict parts with various content types."""
        msg = ChatGPTMessage(
            id="m1",
            author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(
                content_type="text",
                parts=[
                    "Text 1",
                    {"text": "Text from dict 1"},
                    None,
                    {"text": "Text from dict 2"},
                    {"image_url": "data:..."},
                    "Text 2",
                ],
            ),
        )
        text = msg.text_content
        assert "Text 1" in text
        assert "Text 2" in text
        assert "Text from dict 1" in text
        assert "Text from dict 2" in text
        assert "image_url" not in text

    def test_gemini_parts_all_formats(self):
        """Typed, dict, and None parts mixed together."""
        msg = GeminiMessage(
            text="",
            role="model",
            parts=[
                GeminiPart(text="Typed 1"),
                {"text": "Dict 1"},
                GeminiPart(text=None),
                {"text": None},
                {"inlineData": {"data": "..."}},
                GeminiPart(text="Typed 2"),
                {"text": "Dict 2"},
            ],
        )
        text = msg.text_content
        assert "Typed 1" in text
        assert "Dict 1" in text
        assert "Typed 2" in text
        assert "Dict 2" in text
        assert text.count("\n") == 3  # 4 valid parts joined by newlines


# =============================================================================
# Boundary / edge cases
# =============================================================================


class TestContentExtractionEdgeCases:
    """Boundary conditions for content extraction across providers."""

    def test_claude_code_empty_thinking(self):
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": ""},
                    {"type": "text", "text": "Response"},
                ],
            },
        )
        assert record.text_content == "Response"

    def test_claude_code_only_thinking(self):
        """Message with only thinking blocks → empty text, but traces extracted."""
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Thinking 1"},
                    {"type": "thinking", "thinking": "Thinking 2"},
                ],
            },
        )
        assert record.text_content == ""
        traces = record.extract_reasoning_traces()
        assert len(traces) == 2

    def test_chatgpt_all_none_parts(self):
        """Parts list with all None values."""
        msg = ChatGPTMessage(
            id="m1",
            author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(content_type="text", parts=[None, None, None]),
        )
        assert msg.text_content == ""

    def test_gemini_multiple_none_text_values(self):
        """Multiple parts with None text field — only valid parts extracted."""
        msg = GeminiMessage(
            text="",
            role="model",
            parts=[
                {"text": None},
                {"text": None},
                {"text": "Finally"},
                {"text": None},
            ],
        )
        assert msg.text_content == "Finally"

    def test_gemini_very_long_text(self):
        """Long text content is preserved without truncation."""
        long_text = "word " * 10000
        msg = GeminiMessage(text=long_text, role="model")
        assert msg.text_content == long_text
        assert len(msg.text_content) >= 50000

    def test_claude_code_special_characters_in_thinking(self):
        """Thinking with special characters should still be excluded."""
        record = ClaudeCodeRecord(
            type="assistant",
            message={
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Special: \n\t\r\'\"\\"},
                    {"type": "text", "text": "Normal response"},
                ],
            },
        )
        assert record.text_content == "Normal response"

    def test_chatgpt_unicode_in_dict_parts(self):
        """Dict parts with Unicode should work."""
        msg = ChatGPTMessage(
            id="m1",
            author=ChatGPTAuthor(role="assistant"),
            content=ChatGPTContent(
                content_type="text",
                parts=[
                    {"text": "Unicode: 你好 мир 🌍"},
                    "English",
                ],
            ),
        )
        text = msg.text_content
        assert "你好" in text
        assert "мир" in text
        assert "🌍" in text
