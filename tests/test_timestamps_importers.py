from __future__ import annotations

from polylogue.importers.claude import (
    extract_messages_from_chat_messages,
    normalize_timestamp,
    parse_code,
)


def test_normalize_timestamp_seconds():
    """Test that seconds are preserved."""
    ts = 1704067200  # 2024-01-01
    assert normalize_timestamp(ts) == "1704067200.0"
    assert normalize_timestamp(str(ts)) == "1704067200.0"


def test_normalize_timestamp_milliseconds():
    """Test that milliseconds are converted to seconds."""
    ts_ms = 1704067200000  # 2024-01-01 in ms
    expected = "1704067200.0"
    assert normalize_timestamp(ts_ms) == expected
    assert normalize_timestamp(str(ts_ms)) == expected


def test_normalize_timestamp_iso():
    """Test that ISO strings are preserved."""
    iso = "2024-01-01T00:00:00Z"
    assert normalize_timestamp(iso) == iso


def test_extract_messages_normalizes_timestamp():
    """Test that extract_messages_from_chat_messages normalizes timestamps."""
    chat_messages = [
        {
            "uuid": "msg-1",
            "sender": "human",
            "text": "Hello",
            "created_at": 1704067200000,  # ms
        }
    ]
    messages, _ = extract_messages_from_chat_messages(chat_messages)
    assert messages[0].timestamp == "1704067200.0"


def test_parse_code_normalizes_timestamp():
    """Test that parse_code normalizes timestamps."""
    payload = [
        {
            "type": "user",
            "uuid": "msg-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200000,  # ms
            "message": {"content": "Hello"},
        }
    ]
    convo = parse_code(payload, "fallback")
    assert convo.messages[0].timestamp == "1704067200.0"
