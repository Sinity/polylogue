"""Edge case tests for unified extraction functions.

Regression tests for bugs found during the deep sweep:
1. extract_chatgpt_text crash on non-list parts (int → TypeError, str → char iteration)
2. extract_codex_text with non-dict blocks
3. extract_harmonized_message provider dispatch with malformed data
"""

from __future__ import annotations

import pytest

from polylogue.schemas.unified import (
    extract_chatgpt_text,
    extract_codex_text,
    extract_harmonized_message,
)


# =============================================================================
# extract_chatgpt_text
# =============================================================================


class TestExtractChatGPTText:
    """Tests for extract_chatgpt_text with various content structures."""

    def test_normal_string_parts(self):
        """Normal case: list of string parts joined with newlines."""
        content = {"parts": ["Hello", "World"]}
        assert extract_chatgpt_text(content) == "Hello\nWorld"

    def test_single_string_part(self):
        content = {"parts": ["Just one part"]}
        assert extract_chatgpt_text(content) == "Just one part"

    def test_empty_parts_list(self):
        content = {"parts": []}
        assert extract_chatgpt_text(content) == ""

    def test_none_content(self):
        assert extract_chatgpt_text(None) == ""

    def test_empty_dict_content(self):
        assert extract_chatgpt_text({}) == ""

    def test_no_parts_key(self):
        content = {"content_type": "text"}
        assert extract_chatgpt_text(content) == ""

    def test_parts_is_integer(self):
        """REGRESSION: int parts caused TypeError (not iterable)."""
        content = {"parts": 42}
        result = extract_chatgpt_text(content)
        assert result == "42"

    def test_parts_is_string(self):
        """REGRESSION: string parts iterated characters ('h', 'e', 'l', 'l', 'o')."""
        content = {"parts": "hello world"}
        result = extract_chatgpt_text(content)
        assert result == "hello world"

    def test_parts_is_none(self):
        """parts key exists but value is None."""
        content = {"parts": None}
        result = extract_chatgpt_text(content)
        assert result == ""

    def test_parts_is_bool(self):
        """parts key with boolean value."""
        content = {"parts": True}
        result = extract_chatgpt_text(content)
        assert result == "True"

    def test_mixed_parts_with_non_strings(self):
        """Only string parts are included (non-strings like dicts are skipped)."""
        content = {"parts": ["text part", {"type": "image"}, None, "another"]}
        result = extract_chatgpt_text(content)
        assert result == "text part\nanother"

    def test_empty_string_parts(self):
        """Empty strings in parts are included (they are still strings)."""
        content = {"parts": ["", "non-empty", ""]}
        result = extract_chatgpt_text(content)
        assert result == "\nnon-empty\n"


# =============================================================================
# extract_codex_text
# =============================================================================


class TestExtractCodexText:
    """Tests for extract_codex_text with edge cases."""

    def test_normal_text_blocks(self):
        content = [{"text": "Hello"}, {"text": "World"}]
        assert extract_codex_text(content) == "Hello\nWorld"

    def test_input_text_field(self):
        content = [{"input_text": "User input"}]
        assert extract_codex_text(content) == "User input"

    def test_output_text_field(self):
        content = [{"output_text": "Model output"}]
        assert extract_codex_text(content) == "Model output"

    def test_none_content(self):
        assert extract_codex_text(None) == ""

    def test_empty_list(self):
        assert extract_codex_text([]) == ""

    def test_non_list_content(self):
        """Content that's not a list returns empty."""
        assert extract_codex_text("not a list") == ""  # type: ignore[arg-type]

    def test_non_dict_blocks_skipped(self):
        """Non-dict items in the list are silently skipped."""
        content = [{"text": "valid"}, "string block", 42, None, {"text": "also valid"}]
        assert extract_codex_text(content) == "valid\nalso valid"

    def test_block_with_no_text_fields(self):
        """Block without any text fields is skipped."""
        content = [{"type": "image", "url": "http://example.com"}]
        assert extract_codex_text(content) == ""


# =============================================================================
# extract_harmonized_message
# =============================================================================


class TestExtractHarmonizedMessage:
    """Tests for extract_harmonized_message provider dispatch."""

    def test_claude_code_basic(self):
        raw = {
            "uuid": "msg-1",
            "type": "human",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Hello Claude"}],
            },
            "timestamp": "2025-01-01T00:00:00Z",
        }
        msg = extract_harmonized_message("claude-code", raw)
        assert msg.role == "user"
        assert msg.text == "Hello Claude"
        assert msg.id == "msg-1"

    def test_claude_code_with_cost(self):
        raw = {
            "uuid": "msg-2",
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Reply"}],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            "costUSD": 0.005,
            "durationMs": 1200,
        }
        msg = extract_harmonized_message("claude-code", raw)
        assert msg.role == "assistant"
        assert msg.model == "claude-sonnet-4-20250514"
        assert msg.cost is not None
        assert msg.cost.total_usd == 0.005
        assert msg.duration_ms == 1200

    def test_chatgpt_basic(self):
        raw = {
            "id": "msg-gpt-1",
            "author": {"role": "user"},
            "content": {"content_type": "text", "parts": ["Hello GPT"]},
            "create_time": 1704067200.0,
        }
        msg = extract_harmonized_message("chatgpt", raw)
        assert msg.role == "user"
        assert msg.text == "Hello GPT"

    def test_chatgpt_non_list_parts(self):
        """REGRESSION: non-list parts in ChatGPT content must not crash."""
        raw = {
            "id": "msg-gpt-2",
            "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": 42},
        }
        msg = extract_harmonized_message("chatgpt", raw)
        assert msg.text == "42"

    def test_gemini_basic(self):
        raw = {
            "role": "model",
            "text": "Gemini response",
        }
        msg = extract_harmonized_message("gemini", raw)
        assert msg.role == "assistant"
        assert msg.text == "Gemini response"

    def test_gemini_thinking(self):
        raw = {
            "role": "model",
            "text": "Thinking deeply...",
            "isThought": True,
            "thinkingBudget": 500,
        }
        msg = extract_harmonized_message("gemini", raw)
        assert msg.has_reasoning
        assert len(msg.reasoning_traces) == 1
        assert msg.reasoning_traces[0].token_count == 500

    def test_missing_provider_raises(self):
        """Unknown provider should raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            extract_harmonized_message("unknown-provider", {"text": "test"})

    def test_empty_raw_raises_for_missing_role(self):
        """Empty raw dict raises ValueError (role is required)."""
        with pytest.raises(ValueError, match="no role"):
            extract_harmonized_message("chatgpt", {})

    def test_claude_code_type_fallback_when_no_message(self):
        """Claude Code falls back to 'type' field when message is not a dict."""
        raw = {"uuid": "m1", "type": "human", "message": "not-a-dict"}
        msg = extract_harmonized_message("claude-code", raw)
        assert msg.role == "user"  # "human" normalizes to "user"

    def test_claude_code_empty_message_raises(self):
        """Claude Code with empty message dict has no role → raises."""
        raw = {"uuid": "m1", "type": "human", "message": {}}
        with pytest.raises(ValueError, match="no role"):
            extract_harmonized_message("claude-code", raw)
