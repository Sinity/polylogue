"""Tests for the Codex importer."""

from __future__ import annotations

from polylogue.importers.codex import looks_like, parse


def test_looks_like_envelope_format():
    """Test looks_like returns True for envelope format."""
    valid_payload = [
        {"type": "session_meta", "payload": {"id": "test-123", "timestamp": "2025-01-01"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
    ]
    assert looks_like(valid_payload) is True


def test_looks_like_intermediate_format():
    """Test looks_like returns True for intermediate format."""
    valid_payload = [
        {"id": "test-123", "timestamp": "2025-01-01", "git": {}},
        {"type": "message", "role": "user", "content": []},
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


def test_parse_empty_list():
    """Test parsing an empty list returns conversation with no messages."""
    result = parse([], fallback_id="test-empty-list")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "test-empty-list"
    assert len(result.messages) == 0


def test_parse_envelope_format():
    """Test parsing envelope format with session_meta and response_item.

    Note: The parser uses the fallback_id (filename) as the conversation ID,
    not the UUID from session_meta.payload.id. This ensures backwards
    compatibility with existing database records.
    """
    payload = [
        {"type": "session_meta", "payload": {"id": "session-123", "timestamp": "2025-01-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-1",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
                "timestamp": "2025-01-01T00:00:01Z",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-2",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "Hi there!"}],
                "timestamp": "2025-01-01T00:00:02Z",
            },
        },
    ]
    result = parse(payload, fallback_id="rollout-2025-01-01T00-00-00-session-123")

    assert result.provider_name == "codex"
    # Uses fallback_id (filename), not session_meta.payload.id
    assert result.provider_conversation_id == "rollout-2025-01-01T00-00-00-session-123"
    # Timestamp is still extracted from session_meta
    assert result.created_at == "2025-01-01T00:00:00Z"
    assert len(result.messages) == 2

    assert result.messages[0].provider_message_id == "msg-1"
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "Hello"
    assert result.messages[0].timestamp == "2025-01-01T00:00:01Z"

    assert result.messages[1].provider_message_id == "msg-2"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].text == "Hi there!"


def test_parse_intermediate_format():
    """Test parsing intermediate format with direct records.

    Note: The parser uses the fallback_id (filename) as the conversation ID,
    not the ID from the first metadata line. This ensures backwards
    compatibility with existing database records.
    """
    payload = [
        {"id": "session-456", "timestamp": "2025-01-02T00:00:00Z", "git": {}},
        {"record_type": "state"},
        {
            "type": "message",
            "id": "msg-3",
            "role": "user",
            "content": [{"type": "input_text", "text": "Test message"}],
            "timestamp": "2025-01-02T00:00:01Z",
        },
    ]
    result = parse(payload, fallback_id="rollout-2025-01-02T00-00-00-session-456")

    assert result.provider_name == "codex"
    # Uses fallback_id (filename), not intermediate format id
    assert result.provider_conversation_id == "rollout-2025-01-02T00-00-00-session-456"
    # Timestamp is still extracted from first metadata line
    assert result.created_at == "2025-01-02T00:00:00Z"
    assert len(result.messages) == 1

    assert result.messages[0].provider_message_id == "msg-3"
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "Test message"
