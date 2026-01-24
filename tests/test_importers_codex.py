"""Tests for the Codex importer."""

from __future__ import annotations

from polylogue.importers.codex import looks_like, parse


def test_looks_like_valid():
    """Test looks_like returns True for valid Codex format."""
    valid_payload = [
        {"prompt": "Hello", "completion": "Hi there!"},
        {"prompt": "What is 2+2?", "completion": "4"},
    ]
    assert looks_like(valid_payload) is True


def test_looks_like_empty_list():
    """Test looks_like returns False for empty list."""
    assert looks_like([]) is False


def test_looks_like_not_list():
    """Test looks_like returns False for non-list types."""
    assert looks_like({}) is False
    assert looks_like("string") is False
    assert looks_like(None) is False
    assert looks_like(123) is False


def test_looks_like_list_without_prompt_completion():
    """Test looks_like returns False for list without prompt/completion fields."""
    invalid_payload = [
        {"question": "Hello", "answer": "Hi"},
        {"text": "Some text"},
    ]
    assert looks_like(invalid_payload) is False


def test_looks_like_mixed_content():
    """Test looks_like returns True if at least one item has prompt/completion."""
    mixed_payload = [
        {"other": "data"},
        {"prompt": "Hello", "completion": "Hi"},
        {"random": "stuff"},
    ]
    assert looks_like(mixed_payload) is True


def test_parse_basic():
    """Test parsing a basic Codex conversation."""
    payload = [
        {"prompt": "Hello", "completion": "Hi there!"},
        {"prompt": "How are you?", "completion": "I'm doing well!"},
    ]
    result = parse(payload, fallback_id="test-123")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "test-123"
    assert result.title == "test-123"
    assert result.created_at is None
    assert result.updated_at is None
    assert len(result.messages) == 4  # 2 prompts + 2 completions

    # Check first pair
    assert result.messages[0].provider_message_id == "prompt-1"
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "Hello"
    assert result.messages[1].provider_message_id == "completion-1"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].text == "Hi there!"

    # Check second pair
    assert result.messages[2].provider_message_id == "prompt-2"
    assert result.messages[2].role == "user"
    assert result.messages[2].text == "How are you?"
    assert result.messages[3].provider_message_id == "completion-2"
    assert result.messages[3].role == "assistant"
    assert result.messages[3].text == "I'm doing well!"


def test_parse_with_timestamp():
    """Test parsing Codex conversation with timestamps."""
    payload = [
        {"prompt": "Test", "completion": "Response", "timestamp": 1234567890},
    ]
    result = parse(payload, fallback_id="test-ts")

    assert len(result.messages) == 2
    assert result.messages[0].timestamp == "1234567890"
    assert result.messages[1].timestamp == "1234567890"


def test_parse_empty_strings():
    """Test that empty prompt/completion strings are skipped."""
    payload = [
        {"prompt": "", "completion": "Only completion"},
        {"prompt": "Only prompt", "completion": ""},
        {"prompt": "", "completion": ""},
    ]
    result = parse(payload, fallback_id="test-empty")

    # Should only get 2 messages (one completion, one prompt)
    assert len(result.messages) == 2
    assert result.messages[0].text == "Only completion"
    assert result.messages[0].role == "assistant"
    assert result.messages[1].text == "Only prompt"
    assert result.messages[1].role == "user"


def test_parse_missing_fields():
    """Test parsing when prompt/completion fields are missing."""
    payload = [
        {"prompt": "Has prompt only"},
        {"completion": "Has completion only"},
        {"other": "No relevant fields"},
    ]
    result = parse(payload, fallback_id="test-missing")

    assert len(result.messages) == 2
    assert result.messages[0].text == "Has prompt only"
    assert result.messages[0].role == "user"
    assert result.messages[1].text == "Has completion only"
    assert result.messages[1].role == "assistant"


def test_parse_non_dict_items():
    """Test parsing skips non-dict items in the list."""
    payload = [
        "string item",
        {"prompt": "Valid", "completion": "Valid"},
        123,
        None,
        {"prompt": "Another", "completion": "One"},
    ]
    result = parse(payload, fallback_id="test-mixed")

    assert len(result.messages) == 4
    assert result.messages[0].text == "Valid"
    assert result.messages[2].text == "Another"


def test_parse_non_string_values():
    """Test that non-string prompt/completion values are ignored."""
    payload = [
        {"prompt": 123, "completion": "text"},
        {"prompt": "text", "completion": None},
        {"prompt": ["list"], "completion": {"dict": "value"}},
    ]
    result = parse(payload, fallback_id="test-types")

    # Only the valid string messages should be included
    assert len(result.messages) == 2
    assert result.messages[0].text == "text"
    assert result.messages[0].role == "assistant"
    assert result.messages[1].text == "text"
    assert result.messages[1].role == "user"


def test_parse_empty_list():
    """Test parsing an empty list returns conversation with no messages."""
    result = parse([], fallback_id="test-empty-list")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "test-empty-list"
    assert len(result.messages) == 0
