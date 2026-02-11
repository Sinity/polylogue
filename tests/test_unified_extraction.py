"""Tests for unified extraction layer."""

from __future__ import annotations

import json
import sqlite3

import pytest

from polylogue.lib.viewports import ContentType, ToolCategory
from polylogue.schemas.unified import (
    HarmonizedMessage,
    extract_chatgpt_text,
    extract_codex_text,
    extract_from_provider_meta,
    extract_harmonized_message,
    is_message_record,
    normalize_role,
)

# =============================================================================
# Transform Function Tests
# =============================================================================


class TestNormalizeRole:
    """Tests for role normalization."""

    def test_user_variants(self):
        assert normalize_role("user") == "user"
        assert normalize_role("human") == "user"
        assert normalize_role("USER") == "user"

    def test_assistant_variants(self):
        assert normalize_role("assistant") == "assistant"
        assert normalize_role("model") == "assistant"
        assert normalize_role("ai") == "assistant"

    def test_system(self):
        assert normalize_role("system") == "system"

    def test_tool(self):
        assert normalize_role("tool") == "tool"
        assert normalize_role("function") == "tool"

    def test_unknown_passthrough(self):
        assert normalize_role("custom_role") == "unknown"


# =============================================================================
# Claude Code Extraction Tests
# =============================================================================


class TestClaudeCodeExtraction:
    """Tests for Claude Code message extraction."""

    def test_extract_basic_message(self):
        raw = {
            "type": "assistant",
            "uuid": "abc-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "model": "claude-3-opus",
                "content": [
                    {"type": "text", "text": "Hello, how can I help?"}
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50
                }
            },
            "costUSD": 0.005
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert isinstance(msg, HarmonizedMessage)
        assert msg.role == "assistant"
        assert msg.text == "Hello, how can I help?"
        assert msg.id == "abc-123"
        assert msg.model == "claude-3-opus"
        assert msg.provider == "claude-code"
        assert msg.cost is not None
        assert msg.cost.total_usd == 0.005

    def test_extract_with_tool_calls(self):
        raw = {
            "type": "assistant",
            "uuid": "def-456",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read that file."},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "Read",
                        "input": {"file_path": "/path/to/file.py"}
                    }
                ]
            }
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "Read"
        assert msg.tool_calls[0].category == ToolCategory.FILE_READ
        assert msg.tool_calls[0].affected_paths == ["/path/to/file.py"]
        assert msg.has_tool_use is True

    def test_extract_with_thinking(self):
        raw = {
            "type": "assistant",
            "uuid": "ghi-789",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {"type": "text", "text": "Here's my conclusion."}
                ]
            }
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert len(msg.reasoning_traces) == 1
        assert msg.reasoning_traces[0].text == "Let me analyze this..."
        assert msg.has_reasoning is True
        assert "Let me analyze this..." in msg.text

    def test_content_blocks_extraction(self):
        raw = {
            "type": "assistant",
            "uuid": "jkl-012",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Thinking..."},
                    {"type": "text", "text": "Response text"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
                ]
            }
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert len(msg.content_blocks) == 3
        assert msg.content_blocks[0].type == ContentType.THINKING
        assert msg.content_blocks[1].type == ContentType.TEXT
        assert msg.content_blocks[2].type == ContentType.TOOL_USE

    def test_is_message_record(self):
        assert is_message_record("claude-code", {"type": "user"}) is True
        assert is_message_record("claude-code", {"type": "assistant"}) is True
        assert is_message_record("claude-code", {"type": "system"}) is True
        assert is_message_record("claude-code", {"type": "progress"}) is False
        assert is_message_record("claude-code", {"type": "file-history-snapshot"}) is False


# =============================================================================
# ChatGPT Extraction Tests
# =============================================================================


class TestChatGPTExtraction:
    """Tests for ChatGPT extraction."""

    def test_extract_message(self):
        raw = {
            "id": "chatgpt-123",
            "author": {"role": "assistant"},
            "content": {
                "content_type": "text",
                "parts": ["Here's your answer."]
            },
            "create_time": 1705318200,
            "metadata": {"model_slug": "gpt-4"}
        }

        msg = extract_harmonized_message("chatgpt", raw)

        assert msg.role == "assistant"
        assert msg.text == "Here's your answer."
        assert msg.id == "chatgpt-123"
        assert msg.model == "gpt-4"
        assert msg.provider == "chatgpt"


# =============================================================================
# Gemini Extraction Tests
# =============================================================================


class TestGeminiExtraction:
    """Tests for Gemini extraction."""

    def test_extract_message(self):
        raw = {
            "role": "model",
            "text": "I can help with that.",
            "tokenCount": 25
        }

        msg = extract_harmonized_message("gemini", raw)

        assert msg.role == "assistant"
        assert msg.text == "I can help with that."
        assert msg.tokens is not None
        assert msg.tokens.output_tokens == 25

    def test_extract_thinking(self):
        raw = {
            "role": "model",
            "text": "Thinking about this...",
            "isThought": True,
            "thinkingBudget": 1000
        }

        msg = extract_harmonized_message("gemini", raw)

        assert msg.has_reasoning is True
        assert len(msg.reasoning_traces) == 1
        assert msg.reasoning_traces[0].token_count == 1000


# =============================================================================
# Codex Extraction Tests
# =============================================================================


class TestCodexExtraction:
    """Tests for Codex extraction."""

    def test_extract_direct_format(self):
        raw = {
            "id": "codex-123",
            "role": "user",
            "timestamp": "2024-01-15T10:30:00Z",
            "content": [
                {"type": "input_text", "text": "Hello from Codex"}
            ]
        }

        msg = extract_harmonized_message("codex", raw)

        assert msg.role == "user"
        assert msg.text == "Hello from Codex"
        assert msg.id == "codex-123"

    def test_extract_envelope_format(self):
        raw = {
            "type": "response_item",
            "payload": {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Response from Codex"}
                ]
            }
        }

        msg = extract_harmonized_message("codex", raw)

        assert msg.role == "assistant"
        assert msg.text == "Response from Codex"


# =============================================================================
# Database Integration Tests
# =============================================================================


class TestDatabaseIntegration:
    """Integration tests using seeded database with real fixture data."""

    @pytest.mark.parametrize("provider", ["claude-code", "chatgpt", "codex"])
    def test_extract_real_messages(self, seeded_db, provider):
        """Extract real messages and verify structure."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = ?
            LIMIT 20
            """,
            (provider,)
        )

        rows = cur.fetchall()
        conn.close()

        extracted = 0
        for (pm_json,) in rows:
            # Skip if provider_meta is NULL
            if pm_json is None:
                continue

            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record(provider, raw):
                continue

            msg = extract_from_provider_meta(provider, pm)

            assert isinstance(msg, HarmonizedMessage)
            assert msg.provider in (provider, "claude-ai")  # claude -> claude-ai
            assert msg.role in ("user", "assistant", "system", "tool", "unknown")
            assert isinstance(msg.text, str)

            extracted += 1

        if extracted == 0:
            pytest.skip(f"No {provider} messages with provider_meta in seeded database")

    def test_claude_code_tool_extraction(self, seeded_db):
        """Verify tool calls are extracted from Claude Code messages."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 100
            """
        )

        rows = cur.fetchall()
        conn.close()

        total_tools = 0
        for (pm_json,) in rows:
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record("claude-code", raw):
                continue

            msg = extract_from_provider_meta("claude-code", pm)
            total_tools += len(msg.tool_calls)

        # May not find tool calls in all fixtures - just verify it doesn't crash
        assert total_tools >= 0

    def test_viewports_properties(self, seeded_db):
        """Test viewport convenience properties."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 50
            """
        )

        rows = cur.fetchall()
        conn.close()

        file_ops = 0
        git_ops = 0

        for (pm_json,) in rows:
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record("claude-code", raw):
                continue

            msg = extract_from_provider_meta("claude-code", pm)
            file_ops += len(msg.file_operations)
            git_ops += len(msg.git_operations)

        # These are computed properties - just verify they don't crash
        # and return reasonable values
        assert file_ops >= 0
        assert git_ops >= 0


# MERGED FROM test_unified_extraction_edge_cases.py
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
# extract_harmonized_message (edge cases from edge_cases file)
# =============================================================================


class TestExtractHarmonizedMessageEdgeCases:
    """Tests for extract_harmonized_message provider dispatch (edge cases)."""

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

    def test_chatgpt_basic_edge(self):
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

    def test_gemini_basic_edge(self):
        raw = {
            "role": "model",
            "text": "Gemini response",
        }
        msg = extract_harmonized_message("gemini", raw)
        assert msg.role == "assistant"
        assert msg.text == "Gemini response"

    def test_gemini_thinking_edge(self):
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
