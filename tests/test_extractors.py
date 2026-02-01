"""Tests for glom-based provider extraction."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.schemas.common import CommonMessage, Role
from polylogue.schemas.extractors import (
    GLOM_AVAILABLE,
    extract_message,
    extract_message_from_db,
    is_claude_code_message,
    parse_iso_timestamp,
    parse_unix_timestamp,
    normalize_role,
)


pytestmark = pytest.mark.skipif(not GLOM_AVAILABLE, reason="glom not installed")


# =============================================================================
# Transform Function Tests
# =============================================================================


class TestParseTimestamps:
    """Tests for timestamp parsing functions."""

    def test_parse_iso_with_z_suffix(self):
        ts = parse_iso_timestamp("2024-01-15T10:30:00Z")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15
        assert ts.hour == 10
        assert ts.minute == 30

    def test_parse_iso_with_offset(self):
        ts = parse_iso_timestamp("2024-01-15T10:30:00+05:00")
        assert ts is not None
        assert ts.year == 2024

    def test_parse_iso_none(self):
        assert parse_iso_timestamp(None) is None

    def test_parse_iso_empty(self):
        assert parse_iso_timestamp("") is None

    def test_parse_iso_invalid(self):
        assert parse_iso_timestamp("not-a-date") is None

    def test_parse_unix_valid(self):
        ts = parse_unix_timestamp(1705318200)  # 2024-01-15 10:30:00 UTC
        assert ts is not None
        assert ts.year == 2024

    def test_parse_unix_none(self):
        assert parse_unix_timestamp(None) is None

    def test_parse_unix_zero(self):
        ts = parse_unix_timestamp(0)
        assert ts is not None
        assert ts.year == 1970


class TestNormalizeRole:
    """Tests for role normalization."""

    def test_user_variants(self):
        assert normalize_role("user") == Role.USER
        assert normalize_role("human") == Role.USER
        assert normalize_role("USER") == Role.USER  # case insensitive

    def test_assistant_variants(self):
        assert normalize_role("assistant") == Role.ASSISTANT
        assert normalize_role("model") == Role.ASSISTANT
        assert normalize_role("ai") == Role.ASSISTANT

    def test_system(self):
        assert normalize_role("system") == Role.SYSTEM

    def test_tool(self):
        assert normalize_role("tool") == Role.TOOL
        assert normalize_role("function") == Role.TOOL

    def test_none_defaults_to_user(self):
        assert normalize_role(None) == Role.USER

    def test_unknown_defaults_to_user(self):
        assert normalize_role("unknown_role") == Role.USER


# =============================================================================
# Claude Code Extraction Tests
# =============================================================================


class TestClaudeCodeExtraction:
    """Tests for Claude Code message extraction."""

    def test_extract_assistant_message(self):
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

        msg = extract_message("claude-code", raw)

        assert msg.role == Role.ASSISTANT
        assert msg.text == "Hello, how can I help?"
        assert msg.id == "abc-123"
        assert msg.model == "claude-3-opus"
        assert msg.tokens == 150
        assert msg.cost_usd == 0.005
        assert msg.provider == "claude-code"

    def test_extract_user_message(self):
        raw = {
            "type": "user",
            "uuid": "def-456",
            "timestamp": "2024-01-15T10:29:00Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Write a function"}
                ]
            }
        }

        msg = extract_message("claude-code", raw)

        assert msg.role == Role.USER
        assert msg.text == "Write a function"

    def test_extract_thinking_block(self):
        raw = {
            "type": "assistant",
            "uuid": "ghi-789",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Here's my answer"}
                ]
            }
        }

        msg = extract_message("claude-code", raw)

        assert msg.is_thinking is True
        assert "Let me think..." in msg.text
        assert "Here's my answer" in msg.text

    def test_extract_non_message_record(self):
        """Non-message records should still extract but with empty content."""
        raw = {
            "type": "file-history-snapshot",
            "timestamp": "2024-01-15T10:30:00Z",
            "messageId": "xxx",
            "snapshot": {}
        }

        msg = extract_message("claude-code", raw)

        # Should not raise, but text will be empty
        assert msg.text == ""

    def test_is_claude_code_message(self):
        assert is_claude_code_message({"type": "user"}) is True
        assert is_claude_code_message({"type": "assistant"}) is True
        assert is_claude_code_message({"type": "file-history-snapshot"}) is False
        assert is_claude_code_message({"type": "progress"}) is False


# =============================================================================
# Other Provider Tests
# =============================================================================


class TestClaudeAIExtraction:
    """Tests for Claude AI (web) extraction."""

    def test_extract_message(self):
        raw = {
            "uuid": "msg-123",
            "sender": "human",
            "text": "Hello Claude",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:30:00Z",
            "attachments": [],
            "content": [],
            "files": []
        }

        msg = extract_message("claude", raw)

        assert msg.role == Role.USER
        assert msg.text == "Hello Claude"
        assert msg.id == "msg-123"
        assert msg.provider == "claude-ai"


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

        msg = extract_message("chatgpt", raw)

        assert msg.role == Role.ASSISTANT
        assert msg.text == "Here's your answer."
        assert msg.id == "chatgpt-123"
        assert msg.model == "gpt-4"
        assert msg.provider == "chatgpt"


class TestGeminiExtraction:
    """Tests for Gemini extraction."""

    def test_extract_message(self):
        raw = {
            "role": "model",
            "text": "I can help with that.",
            "tokenCount": 25,
            "isThought": False
        }

        msg = extract_message("gemini", raw)

        assert msg.role == Role.ASSISTANT
        assert msg.text == "I can help with that."
        assert msg.tokens == 25
        assert msg.is_thinking is False
        assert msg.provider == "gemini"

    def test_extract_thinking_message(self):
        raw = {
            "role": "model",
            "text": "Thinking about this...",
            "isThought": True,
            "thinkingBudget": 1000
        }

        msg = extract_message("gemini", raw)

        assert msg.is_thinking is True


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


class TestDatabaseExtraction:
    """Integration tests using real database data."""

    @pytest.mark.parametrize("provider", ["claude-code", "claude", "chatgpt", "gemini"])
    def test_extract_real_messages(self, polylogue_db, provider):
        """Extract real messages from each provider in the database."""
        cur = polylogue_db.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = ?
            LIMIT 10
            """,
            (provider,)
        )

        rows = cur.fetchall()
        if not rows:
            pytest.skip(f"No {provider} messages in database")

        extracted = 0
        for (pm_json,) in rows:
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            # Skip non-message records for claude-code
            if provider == "claude-code" and not is_claude_code_message(raw):
                continue

            msg = extract_message_from_db(provider, pm)
            assert isinstance(msg, CommonMessage)
            assert msg.provider in (provider, "claude-ai")  # claude -> claude-ai alias
            extracted += 1

        assert extracted > 0, f"No valid {provider} messages extracted"

    def test_provider_coverage(self, polylogue_db):
        """Verify major providers in DB have extraction specs."""
        from polylogue.schemas.extractors import PROVIDER_SPECS

        cur = polylogue_db.cursor()
        cur.execute("SELECT DISTINCT provider_name FROM conversations")
        db_providers = {row[0] for row in cur.fetchall()}

        # Core providers that MUST have specs
        core_providers = {"claude-code", "claude", "chatgpt", "gemini", "codex"}

        # Check core providers have specs
        for provider in core_providers & db_providers:
            assert provider in PROVIDER_SPECS, (
                f"No spec for core provider: {provider}. "
                f"Known specs: {list(PROVIDER_SPECS.keys())}"
            )

        # Report any other providers without specs (not a failure)
        extra = db_providers - core_providers - set(PROVIDER_SPECS.keys())
        if extra:
            import warnings
            warnings.warn(f"Additional providers without specs: {extra}")
