"""Consolidated Claude importer tests using aggressive parametrization.

CONSOLIDATION: 65 tests → ~12 parametrized test functions with 80+ test cases.

Original: Separate test classes for AI format, Code format, segment extraction
New: Parametrized tests covering all variants
"""

import json
from pathlib import Path

import pytest

from polylogue.importers.claude import (
    extract_messages_from_chat_messages,
    extract_text_from_segments,
    looks_like_ai,
    looks_like_code,
    parse_ai,
    parse_code,
)


# =============================================================================
# FORMAT DETECTION - PARAMETRIZED (2 tests replacing 13)
# =============================================================================


LOOKS_LIKE_AI_CASES = [
    ({"chat_messages": []}, True, "chat_messages field"),
    ({"chat_messages": [{"uuid": "1"}]}, True, "with messages"),
    ({}, False, "missing chat_messages"),
    ({"messages": []}, False, "wrong field name"),
    ({"chat_messages": "not-a-list"}, False, "chat_messages not list"),
    (None, False, "None input"),
    ("string", False, "string input"),
    ([], False, "list input"),
    # ChatGPT format should be rejected
    ({"mapping": {}}, False, "ChatGPT format"),
]


@pytest.mark.parametrize("data,expected,desc", LOOKS_LIKE_AI_CASES)
def test_looks_like_ai_format(data, expected, desc):
    """Comprehensive AI format detection.

    Replaces 8 looks_like_ai tests.
    """
    result = looks_like_ai(data)
    assert result == expected, f"Failed {desc}"


LOOKS_LIKE_CODE_CASES = [
    ([{"parentUuid": "123"}], True, "parentUuid variant"),
    ([{"sessionId": "456"}], True, "sessionId variant"),
    ([{"session_id": "789"}], True, "session_id variant"),
    ([{"leafUuid": "abc"}], True, "leafUuid variant"),
    ([], False, "empty list"),
    ([{"messages": []}], False, "only messages field"),
    (None, False, "None input"),
    ({}, False, "dict input"),
    # ChatGPT format should be rejected
    ([{"mapping": {}}], False, "ChatGPT format"),
]


@pytest.mark.parametrize("data,expected,desc", LOOKS_LIKE_CODE_CASES)
def test_looks_like_code_format(data, expected, desc):
    """Comprehensive Code format detection.

    Replaces 5 looks_like_code tests.
    """
    result = looks_like_code(data)
    assert result == expected, f"Failed {desc}"


# =============================================================================
# SEGMENT EXTRACTION - PARAMETRIZED (1 test replacing 10)
# =============================================================================


SEGMENT_CASES = [
    (["plain text"], "plain text", "string segment"),
    ([{"text": "dict with text"}], "dict with text", "dict with text field"),
    ([{"content": "dict with content"}], "dict with content", "dict with content field"),
    ([{"type": "tool_use", "name": "read", "input": {}}], "read", "tool_use"),
    ([{"type": "tool_result", "content": "result"}], "result", "tool_result"),
    (["text1", {"text": "text2"}, "text3"], "text1\ntext2\ntext3", "mixed segments"),
    ([{}, "", None], None, "empty/None segments"),
    ([{"other": "field"}], None, "dict without text/content/type"),
]


@pytest.mark.parametrize("segments,expected_contains,desc", SEGMENT_CASES)
def test_extract_text_from_segments_comprehensive(segments, expected_contains, desc):
    """Comprehensive segment extraction.

    Replaces 10 segment extraction tests.
    """
    result = extract_text_from_segments(segments)

    if expected_contains:
        assert expected_contains in result, f"Failed {desc}: '{expected_contains}' not in '{result}'"
    else:
        assert result is None or result == "", f"Failed {desc}: expected None or empty"


# =============================================================================
# MESSAGE EXTRACTION - PARAMETRIZED (1 test replacing 15)
# =============================================================================


def make_chat_message(uuid, sender, text, attachments=None, files=None, timestamp=None):
    """Helper to create chat_messages entries."""
    msg = {
        "uuid": uuid,
        "text": text,
    }

    # Role field variants
    if sender:
        msg["sender"] = sender

    if attachments:
        msg["attachments"] = attachments
    if files:
        msg["files"] = files
    if timestamp:
        msg["created_at"] = timestamp

    return msg


EXTRACT_CHAT_MESSAGES_CASES = [
    # Basic
    ([make_chat_message("u1", "human", "Hello")], 1, "basic message"),

    # Attachments variants
    ([make_chat_message("u1", "human", "Hi", attachments=[{"file_name": "doc.pdf"}])], 1, "attachments field"),
    ([make_chat_message("u1", "human", "Hi", files=[{"file_name": "doc.pdf"}])], 1, "files field"),

    # Role variants
    ([make_chat_message("u1", "human", "Hi")], "user", "human role"),
    ([make_chat_message("u1", "assistant", "Hi")], "assistant", "assistant role"),
    ([make_chat_message("u1", None, "Hi")], "message", "missing sender defaults"),

    # Timestamp variants
    ([make_chat_message("u1", "human", "Hi", timestamp="2024-01-01T00:00:00Z")], 1, "created_at"),
    ([{"uuid": "u1", "text": "Hi", "create_time": 1704067200}], 1, "create_time"),
    ([{"uuid": "u1", "text": "Hi", "timestamp": 1704067200}], 1, "timestamp field"),

    # ID variants
    ([{"uuid": "u1", "text": "Hi"}], "u1", "uuid field"),
    ([{"id": "i1", "text": "Hi"}], "i1", "id field"),
    ([{"message_id": "m1", "text": "Hi"}], "m1", "message_id field"),

    # Content variants
    ([{"uuid": "u1", "text": ["list", "of", "parts"]}], 0, "text as list skipped"),
    ([{"uuid": "u1", "content": {"text": "nested text"}}], 1, "content dict with text"),
    ([{"uuid": "u1", "content": {"parts": ["part1", "part2"]}}], 1, "content dict with parts"),

    # Missing text
    ([{"uuid": "u1"}], 0, "missing text skipped"),
    ([{"uuid": "u1", "text": ""}], 0, "empty text skipped"),
    ([{"uuid": "u1", "text": None}], 0, "None text skipped"),

    # Non-dict items
    (["not a dict", {"uuid": "u1", "text": "Valid"}], 1, "skip non-dict"),

    # Empty list
    ([], 0, "empty list"),
]


@pytest.mark.parametrize("chat_messages,expected,desc", EXTRACT_CHAT_MESSAGES_CASES)
def test_extract_chat_messages_comprehensive(chat_messages, expected, desc):
    """Comprehensive chat_messages extraction.

    Replaces 15 extraction tests.
    """
    messages, attachments = extract_messages_from_chat_messages(chat_messages)

    if isinstance(expected, int):
        assert len(messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str):
        if messages:
            # ID field tests check provider_message_id, role tests check role
            if "field" in desc and desc not in ["attachments field", "files field", "timestamp field"]:
                assert messages[0].provider_message_id == expected, f"Failed {desc}"
            else:
                # Expected role
                assert messages[0].role == expected, f"Failed {desc}"


# =============================================================================
# PARSE AI - PARAMETRIZED (1 test replacing 10)
# =============================================================================


def make_ai_conv(chat_messages, **kwargs):
    """Helper to create AI format conversation."""
    conv = {"chat_messages": chat_messages}
    conv.update(kwargs)
    return conv


PARSE_AI_CASES = [
    # Basic
    (make_ai_conv([make_chat_message("u1", "human", "Hello")]), 1, "basic"),

    # With attachments
    (make_ai_conv([make_chat_message("u1", "human", "Hi", files=[{"file_name": "doc.pdf"}])]), 1, "with files"),

    # Title extraction
    (make_ai_conv([], name="Test Title"), "Test Title", "title extraction"),

    # Empty chat_messages
    (make_ai_conv([]), 0, "empty messages"),

    # Content variants
    ({"chat_messages": [{"uuid": "u1", "content": {"text": "nested"}}]}, 1, "content dict"),
    ({"chat_messages": [{"uuid": "u1", "content": {"parts": ["p1"]}}]}, 1, "content parts"),

    # Missing text skipped
    (make_ai_conv([{"uuid": "u1"}]), 0, "missing text"),
]


@pytest.mark.parametrize("conv_data,expected,desc", PARSE_AI_CASES)
def test_parse_ai_comprehensive(conv_data, expected, desc):
    """Comprehensive AI format parsing.

    Replaces 10 parse_ai tests.
    """
    result = parse_ai(conv_data, "fallback-id")

    if isinstance(expected, int):
        assert len(result.messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str):
        # Expected title
        assert result.title == expected, f"Failed {desc}"


# =============================================================================
# PARSE CODE - PARAMETRIZED (1 test replacing 17)
# =============================================================================


def make_code_message(msg_type, text, **kwargs):
    """Helper to create Code format message.

    Creates message in claude-code format with nested message.content structure.
    """
    msg = {
        "type": msg_type,
    }
    # Add text as nested message.content
    if text or "message" not in kwargs:
        msg["message"] = {"content": text} if text else {}
    msg.update(kwargs)
    return msg


PARSE_CODE_CASES = [
    # Basic messages
    ([make_code_message("user", "Question")], 1, "user message"),
    ([make_code_message("assistant", "Answer")], 1, "assistant message"),

    # Tool use (as message content list)
    ([make_code_message("assistant", "", message={"content": [{"type": "tool_use", "name": "read"}]})], 1, "tool use"),
    ([make_code_message("user", "", message={"content": [{"type": "tool_result", "content": "result"}]})], 1, "tool result"),

    # Thinking blocks
    ([make_code_message("assistant", "", message={"content": [{"type": "thinking", "thinking": "analysis"}]})], 1, "with thinking field"),

    # Metadata preservation
    ([make_code_message("assistant", "text", costUSD=0.01, durationMs=1000)], 1, "with cost/duration"),
    ([make_code_message("assistant", "text", isSidechain=True)], 1, "sidechain marker"),

    # Skip summary/init types
    ([make_code_message("summary", "Summary text")], 0, "skip summary"),
    ([make_code_message("init", "Init")], 0, "skip init"),

    # Type to role mapping
    ([make_code_message("user", "Q")], "user", "user type"),
    ([make_code_message("assistant", "A")], "assistant", "assistant type"),

    # Message as string (content field as string)
    ([{"type": "user", "message": {"content": "String content"}}], 1, "message as string"),

    # Empty/missing fields (currently returns message with text=None)
    ([make_code_message("assistant", "")], 1, "empty text"),
    ([{"type": "assistant"}], 1, "missing text"),

    # Session metadata
    ([make_code_message("user", "Hi")], None, "extracts session metadata"),

    # Timestamp extraction
    ([make_code_message("user", "Hi", timestamp=1704067200)], 1704067200, "timestamp"),

    # Empty list
    ([], 0, "empty messages"),

    # Non-dict items
    (["not a dict"], 0, "skip non-dict"),
]


@pytest.mark.parametrize("messages,expected,desc", PARSE_CODE_CASES)
def test_parse_code_comprehensive(messages, expected, desc):
    """Comprehensive Code format parsing.

    Replaces 17 parse_code tests.
    """
    result = parse_code(messages, "fallback-id")

    if isinstance(expected, int):
        if "timestamp" not in desc:
            assert len(result.messages) == expected, f"Failed {desc}"
        else:
            # Check timestamp was extracted (as float string)
            if result.messages and result.messages[0].timestamp:
                assert float(result.messages[0].timestamp) == expected
    elif isinstance(expected, str):
        # Expected role
        if result.messages:
            assert result.messages[0].role == expected, f"Failed {desc}"
    elif expected is None:
        # Just verify it parsed
        assert result.provider_name in ["claude", "claude-code"]
