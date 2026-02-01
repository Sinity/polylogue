"""Tests for unified extraction layer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.lib.viewports import ToolCategory, ContentType
from polylogue.schemas.unified import (
    HarmonizedMessage,
    extract_harmonized_message,
    extract_from_provider_meta,
    is_message_record,
    normalize_role,
    extract_tool_calls,
    extract_reasoning_traces,
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
        assert normalize_role("custom_role") == "custom_role"


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


POLYLOGUE_DB = Path.home() / ".local/state/polylogue/polylogue.db"


@pytest.fixture
def polylogue_db():
    """Fixture providing connection to polylogue database."""
    if not POLYLOGUE_DB.exists():
        pytest.skip("Polylogue database not found")
    conn = sqlite3.connect(POLYLOGUE_DB)
    yield conn
    conn.close()


class TestDatabaseIntegration:
    """Integration tests using real database."""

    @pytest.mark.parametrize("provider", ["claude-code", "claude", "chatgpt", "gemini"])
    def test_extract_real_messages(self, polylogue_db, provider):
        """Extract real messages and verify structure."""
        cur = polylogue_db.cursor()
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

        extracted = 0
        for (pm_json,) in cur.fetchall():
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

        assert extracted > 0, f"No {provider} messages extracted"

    def test_claude_code_tool_extraction(self, polylogue_db):
        """Verify tool calls are extracted from Claude Code messages."""
        cur = polylogue_db.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 100
            """
        )

        total_tools = 0
        for (pm_json,) in cur.fetchall():
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record("claude-code", raw):
                continue

            msg = extract_from_provider_meta("claude-code", pm)
            total_tools += len(msg.tool_calls)

        # Should find at least some tool calls in 100 messages
        assert total_tools > 0, "No tool calls found in Claude Code messages"

    def test_viewports_properties(self, polylogue_db):
        """Test viewport convenience properties."""
        cur = polylogue_db.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 50
            """
        )

        file_ops = 0
        git_ops = 0

        for (pm_json,) in cur.fetchall():
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
