"""Pinned provider content-extraction edge cases that still add unique value.

Broad semantic equivalence/content-block laws live elsewhere. This file keeps
only direct helper, None-guard, and provider-specific weird-input cases.
"""

from __future__ import annotations

from polylogue.lib.provider_semantics import extract_claude_code_text
from polylogue.sources.providers.chatgpt import (
    ChatGPTAuthor,
    ChatGPTContent,
    ChatGPTMessage,
)
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.gemini import GeminiMessage

# =============================================================================
# Claude Code: direct text-block extraction helper call
# =============================================================================


class TestClaudeCodeTextFromBlocks:
    """Test the shared helper directly — bypasses record dispatch logic."""

    def test_text_from_blocks_excludes_thinking(self):
        blocks = [
            {"type": "thinking", "thinking": "Let me reason about this..."},
            {"type": "text", "text": "Here is my answer."},
        ]
        text = extract_claude_code_text(blocks)
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
# Boundary / edge cases
# =============================================================================


class TestContentExtractionEdgeCases:
    """Boundary conditions for content extraction across providers."""

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
