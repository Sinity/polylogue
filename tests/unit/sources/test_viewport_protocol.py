"""Cross-provider viewport interface conformance tests.

Every provider record type must implement the viewport interface:
  text_content: str
  role_normalized: str in {user, assistant, system, tool, unknown}
  parsed_timestamp: datetime | None
  to_meta() → MessageMeta
  extract_content_blocks() → list
  extract_reasoning_traces() → list

These tests verify the contract across all providers using minimal valid instances.
"""
from __future__ import annotations

import pytest

from polylogue.sources.providers.chatgpt import ChatGPTAuthor, ChatGPTContent, ChatGPTMessage
from polylogue.sources.providers.claude_ai import ClaudeAIChatMessage
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.codex import CodexRecord
from polylogue.sources.providers.gemini import GeminiMessage

VALID_ROLE_VALUES = {"user", "assistant", "system", "tool", "unknown", "model"}

# ---------------------------------------------------------------------------
# Minimal valid instances for each provider
# ---------------------------------------------------------------------------

def _make_chatgpt_record() -> ChatGPTMessage:
    return ChatGPTMessage(
        id="m1",
        author=ChatGPTAuthor(role="user"),
        content=ChatGPTContent(content_type="text", parts=["Hello"]),
    )


def _make_claude_ai_record() -> ClaudeAIChatMessage:
    return ClaudeAIChatMessage(
        uuid="m1",
        sender="human",
        text="Hello",
    )


def _make_claude_code_record() -> ClaudeCodeRecord:
    return ClaudeCodeRecord(
        type="user",
        message={"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    )


def _make_codex_record() -> CodexRecord:
    return CodexRecord(
        type="message",
        role="user",
        content=[{"type": "input_text", "text": "Hello"}],
    )


def _make_gemini_record() -> GeminiMessage:
    return GeminiMessage(text="Hello", role="user")


PROVIDER_RECORDS = [
    ("chatgpt", _make_chatgpt_record()),
    ("claude-ai", _make_claude_ai_record()),
    ("claude-code", _make_claude_code_record()),
    ("codex", _make_codex_record()),
    ("gemini", _make_gemini_record()),
]


@pytest.mark.parametrize(
    "provider,record",
    PROVIDER_RECORDS,
    ids=[p for p, _ in PROVIDER_RECORDS],
)
class TestViewportProtocol:
    """All provider records must satisfy the viewport interface contract."""

    def test_text_content_is_str(self, provider, record):
        result = record.text_content
        assert isinstance(result, str), (
            f"{provider}.text_content must be str, got {type(result).__name__}"
        )

    def test_role_normalized_is_valid(self, provider, record):
        result = record.role_normalized
        assert isinstance(result, str), (
            f"{provider}.role_normalized must be str, got {type(result).__name__}"
        )
        assert result in VALID_ROLE_VALUES, (
            f"{provider}.role_normalized={result!r} not in {VALID_ROLE_VALUES}"
        )

    def test_parsed_timestamp_is_datetime_or_none(self, provider, record):
        from datetime import datetime
        result = record.parsed_timestamp
        assert result is None or isinstance(result, datetime), (
            f"{provider}.parsed_timestamp must be datetime | None, got {type(result).__name__}"
        )

    def test_to_meta_returns_message_meta(self, provider, record):
        from polylogue.lib.viewports import MessageMeta
        meta = record.to_meta()
        assert meta is not None, f"{provider}.to_meta() must not return None"
        assert isinstance(meta, MessageMeta), (
            f"{provider}.to_meta() must return MessageMeta, got {type(meta).__name__}"
        )
        assert hasattr(meta, "role"), f"{provider}.to_meta() result must have .role"

    def test_extract_content_blocks_returns_list(self, provider, record):
        result = record.extract_content_blocks()
        assert isinstance(result, list), (
            f"{provider}.extract_content_blocks() must return list, got {type(result).__name__}"
        )

    def test_extract_reasoning_traces_returns_list(self, provider, record):
        result = record.extract_reasoning_traces()
        assert isinstance(result, list), (
            f"{provider}.extract_reasoning_traces() must return list, got {type(result).__name__}"
        )

    def test_to_meta_role_matches_role_normalized(self, provider, record):
        """to_meta().role must agree with role_normalized."""
        meta = record.to_meta()
        assert meta.role == record.role_normalized, (
            f"{provider}: to_meta().role={meta.role!r} != role_normalized={record.role_normalized!r}"
        )
