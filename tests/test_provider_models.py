"""Tests for provider-specific model viewport methods."""

import pytest

from polylogue.sources.providers.chatgpt import ChatGPTMessage, ChatGPTAuthor, ChatGPTContent
from polylogue.sources.providers.gemini import GeminiMessage


class TestChatGPTMessageTextContent:
    """Regression tests for ChatGPTMessage.text_content."""

    def test_text_content_with_string_parts(self):
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=["Hello", "World"]),
        )
        assert msg.text_content == "Hello\nWorld"

    def test_text_content_with_none_parts(self):
        """Regression: parts list can contain None values."""
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=[None, "Valid"]),
        )
        assert msg.text_content == "Valid"

    def test_text_content_with_dict_none_text(self):
        """Regression: dict part with 'text' key but None value must not crash join()."""
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=[{"text": None}, {"text": "ok"}]),
        )
        assert msg.text_content == "ok"

    def test_text_content_with_dict_valid_text(self):
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=[{"text": "hello"}]),
        )
        assert msg.text_content == "hello"

    def test_text_content_empty_parts(self):
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", parts=[]),
        )
        assert msg.text_content == ""

    def test_text_content_no_content(self):
        msg = ChatGPTMessage(id="1", author=ChatGPTAuthor(role="user"))
        assert msg.text_content == ""

    def test_text_content_direct_text(self):
        msg = ChatGPTMessage(
            id="1", author=ChatGPTAuthor(role="user"),
            content=ChatGPTContent(content_type="text", text="Direct text"),
        )
        assert msg.text_content == "Direct text"

    def test_role_normalized(self):
        for role_in, expected in [("user", "user"), ("assistant", "assistant"), ("tool", "tool"), ("custom", "unknown")]:
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

    def test_role_normalized(self):
        for role_in, expected in [("user", "user"), ("model", "assistant"), ("assistant", "assistant"), ("custom", "unknown")]:
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
