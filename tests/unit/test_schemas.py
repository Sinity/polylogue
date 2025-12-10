"""Tests for Pydantic provider schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from polylogue.importers.schemas import ChatGPTConversation, ChatGPTMessage, ClaudeAIConversation


def test_chatgpt_message_valid():
    """Test valid ChatGPT message parsing."""
    data = {
        "id": "msg-123",
        "author": {"role": "user"},
        "content": {
            "content_type": "text",
            "parts": ["Hello world"]
        },
        "create_time": 1609459200.0,
    }

    msg = ChatGPTMessage(**data)
    assert msg.id == "msg-123"
    assert msg.author.role == "user"
    assert msg.content.content_type == "text"


def test_chatgpt_message_missing_required_field():
    """Test that missing required fields raise ValidationError."""
    data = {
        # Missing 'id'
        "author": {"role": "user"},
        "content": {"content_type": "text", "parts": []},
    }

    with pytest.raises(ValidationError) as exc_info:
        ChatGPTMessage(**data)

    # Should clearly indicate 'id' field is missing
    assert "id" in str(exc_info.value).lower()


def test_chatgpt_conversation_valid():
    """Test valid ChatGPT conversation parsing."""
    data = {
        "title": "Test Conversation",
        "mapping": {
            "msg-1": {
                "id": "msg-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hi"]},
                }
            }
        }
    }

    conv = ChatGPTConversation(**data)
    assert conv.title == "Test Conversation"
    assert "msg-1" in conv.mapping


def test_chatgpt_conversation_with_id():
    """Test conversation ID normalization."""
    data1 = {
        "title": "Test",
        "id": "conv-123",
        "mapping": {}
    }

    data2 = {
        "title": "Test",
        "conversation_id": "conv-456",
        "mapping": {}
    }

    conv1 = ChatGPTConversation(**data1)
    conv2 = ChatGPTConversation(**data2)

    assert conv1.conversation_id_normalized == "conv-123"
    assert conv2.conversation_id_normalized == "conv-456"


def test_claude_conversation_valid():
    """Test valid Claude.ai conversation parsing."""
    data = {
        "uuid": "conv-123",
        "name": "Test Chat",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "chat_messages": [
            {
                "uuid": "msg-1",
                "text": "Hello",
                "sender": "human",
            }
        ]
    }

    conv = ClaudeAIConversation(**data)
    assert conv.uuid == "conv-123"
    assert conv.name == "Test Chat"
    assert len(conv.all_messages) == 1


def test_claude_conversation_messages_format():
    """Test both chat_messages and messages fields work."""
    data1 = {
        "uuid": "conv-1",
        "name": "Test",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "chat_messages": [
            {"uuid": "msg-1", "text": "Hi", "sender": "human"}
        ]
    }

    data2 = {
        "uuid": "conv-2",
        "name": "Test",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "messages": [
            {"uuid": "msg-2", "text": "Hi", "sender": "human"}
        ]
    }

    conv1 = ClaudeAIConversation(**data1)
    conv2 = ClaudeAIConversation(**data2)

    assert len(conv1.all_messages) == 1
    assert len(conv2.all_messages) == 1
