"""Tests for heuristic fallback parser."""
from __future__ import annotations

from polylogue.importers.fallback_parser import (
    create_degraded_markdown,
    extract_messages_heuristic,
    extract_text_recursively,
    extract_timestamps,
)


def test_extract_text_recursively():
    """Test recursive text extraction."""
    data = {
        "messages": [
            {"text": "Hello, this is a test message that should be extracted"},
            {"content": "Another message with enough text to be meaningful"},
        ],
        "metadata": {
            "short": "x",  # Too short
            "url": "https://example.com",  # URL should be skipped
        }
    }

    texts = extract_text_recursively(data, min_length=20)

    assert len(texts) == 2
    assert "Hello, this is a test message" in texts[0]
    assert "Another message" in texts[1]


def test_extract_timestamps():
    """Test timestamp extraction."""
    data = {
        "created_at": 1609459200,  # 2021-01-01
        "messages": [
            {"timestamp": 1609545600},  # 2021-01-02
            {"time": 1609632000},       # 2021-01-03
        ],
        "invalid": 999,  # Too old
        "future": 2000000000,  # Too far in future
    }

    timestamps = extract_timestamps(data)

    # Should find valid timestamps
    assert len(timestamps) >= 3
    assert 1609459200 in timestamps
    assert 1609545600 in timestamps


def test_extract_messages_from_array():
    """Test message extraction from array format."""
    data = {
        "messages": [
            {
                "text": "This is a user message with sufficient length",
                "sender": "user",
                "timestamp": 1609459200,
            },
            {
                "text": "This is an assistant response with enough text",
                "sender": "assistant",
                "timestamp": 1609545600,
            }
        ]
    }

    messages = extract_messages_heuristic(data)

    assert len(messages) >= 2
    assert any("user message" in msg["text"] for msg in messages)
    assert any("assistant response" in msg["text"] for msg in messages)


def test_extract_messages_from_mapping():
    """Test message extraction from mapping format (like ChatGPT)."""
    data = {
        "mapping": {
            "msg-1": {
                "message": {
                    "content": {
                        "parts": ["First message with enough text to be extracted"]
                    }
                },
                "create_time": 1609459200,
            },
            "msg-2": {
                "message": {
                    "content": {
                        "parts": ["Second message also with sufficient length"]
                    }
                },
                "create_time": 1609545600,
            }
        }
    }

    messages = extract_messages_heuristic(data)

    assert len(messages) >= 2


def test_create_degraded_markdown():
    """Test degraded markdown generation."""
    messages = [
        {
            "text": "Hello, this is the first message",
            "timestamp": 1609459200,
            "role": "user",
        },
        {
            "text": "This is the assistant's response",
            "timestamp": 1609545600,
            "role": "assistant",
        }
    ]

    markdown = create_degraded_markdown(messages, title="Test Chat")

    assert "Test Chat" in markdown
    assert "DEGRADED MODE" in markdown
    assert "first message" in markdown
    assert "assistant's response" in markdown
    assert "user" in markdown.lower()
    assert "assistant" in markdown.lower()
