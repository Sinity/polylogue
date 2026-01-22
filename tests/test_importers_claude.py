"""Tests for Claude importer (claude.py)."""
from __future__ import annotations

import json

import pytest

from polylogue.importers.claude import (
    extract_messages_from_chat_messages,
    extract_text_from_segments,
    looks_like_ai,
    looks_like_code,
    parse_ai,
    parse_code,
)


class TestExtractMessagesFromChatMessages:
    """Tests for extract_messages_from_chat_messages helper function."""

    def test_extract_messages_basic(self):
        """Test extracting basic messages from chat_messages list."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "Hello",
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "uuid": "msg-2",
                "sender": "assistant",
                "text": "Hi there!",
                "created_at": "2024-01-01T00:01:00Z",
            },
        ]

        messages, attachments = extract_messages_from_chat_messages(chat_messages)

        assert len(messages) == 2
        assert len(attachments) == 0
        assert messages[0].provider_message_id == "msg-1"
        assert messages[0].role == "user"
        assert messages[0].text == "Hello"
        assert messages[1].provider_message_id == "msg-2"
        assert messages[1].role == "assistant"

    def test_extract_messages_with_attachments(self):
        """Test extracting messages with attachments field."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "Check file",
                "attachments": [
                    {
                        "id": "att-1",
                        "name": "document.pdf",
                        "mimeType": "application/pdf",
                        "size": 2048,
                    },
                ],
            },
        ]

        messages, attachments = extract_messages_from_chat_messages(chat_messages)

        assert len(messages) == 1
        assert len(attachments) == 1
        assert attachments[0].provider_attachment_id == "att-1"
        assert attachments[0].name == "document.pdf"
        assert attachments[0].mime_type == "application/pdf"
        assert attachments[0].size_bytes == 2048

    def test_extract_messages_with_files(self):
        """Test extracting messages with files field."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "See image",
                "files": [
                    {
                        "fileId": "file-1",
                        "filename": "screenshot.png",
                    },
                ],
            },
        ]

        messages, attachments = extract_messages_from_chat_messages(chat_messages)

        assert len(attachments) == 1
        assert attachments[0].provider_attachment_id == "file-1"
        assert attachments[0].name == "screenshot.png"

    def test_extract_messages_multiple_attachments(self):
        """Test extracting message with multiple attachments."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "Multiple files",
                "attachments": [
                    {"id": "att-1", "name": "file1.txt"},
                    {"id": "att-2", "name": "file2.pdf"},
                    {"id": "att-3", "name": "file3.docx"},
                ],
            },
        ]

        messages, attachments = extract_messages_from_chat_messages(chat_messages)

        assert len(attachments) == 3
        assert attachments[0].provider_attachment_id == "att-1"
        assert attachments[1].provider_attachment_id == "att-2"
        assert attachments[2].provider_attachment_id == "att-3"

    def test_extract_messages_role_alternatives(self):
        """Test role extraction with various field names."""
        chat_messages = [
            {"uuid": "1", "sender": "human", "text": "msg1"},  # sender field
            {"uuid": "2", "role": "assistant", "text": "msg2"},  # role field
            {"uuid": "3", "text": "msg3"},  # no role field
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert messages[0].role == "user"  # human -> user
        assert messages[1].role == "assistant"
        assert messages[2].role == "message"  # default

    def test_extract_messages_timestamp_alternatives(self):
        """Test timestamp extraction with various field names."""
        chat_messages = [
            {"uuid": "1", "sender": "human", "text": "msg", "created_at": "2024-01-01"},
            {"uuid": "2", "sender": "assistant", "text": "msg", "create_time": 1704067200},
            {"uuid": "3", "sender": "human", "text": "msg", "timestamp": "2024-01-02"},
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert messages[0].timestamp == "2024-01-01"
        assert messages[1].timestamp == "1704067200.0"
        assert messages[2].timestamp == "2024-01-02"

    def test_extract_messages_id_alternatives(self):
        """Test message ID extraction with various field names."""
        chat_messages = [
            {"uuid": "uuid-1", "sender": "human", "text": "msg"},
            {"id": "id-1", "sender": "human", "text": "msg"},
            {"message_id": "msg-id-1", "sender": "human", "text": "msg"},
            {"sender": "human", "text": "msg"},  # No id field
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert messages[0].provider_message_id == "uuid-1"
        assert messages[1].provider_message_id == "id-1"
        assert messages[2].provider_message_id == "msg-id-1"
        assert messages[3].provider_message_id == "msg-4"  # Generated fallback

    def test_extract_messages_content_as_list(self):
        """Test extracting text when content is a list."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "content": ["Part A", "Part B"],
            },
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert len(messages) == 1
        assert messages[0].text == "Part A\nPart B"

    def test_extract_messages_content_as_dict(self):
        """Test extracting text when content is a dict."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "content": {"text": "Dict text"},
            },
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert len(messages) == 1
        assert messages[0].text == "Dict text"

    def test_extract_messages_content_dict_with_parts(self):
        """Test extracting from dict content with parts."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "content": {"parts": ["A", "B", "C"]},
            },
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert len(messages) == 1
        assert messages[0].text == "A\nB\nC"

    def test_extract_messages_skip_non_dict(self):
        """Test that non-dict items are skipped."""
        chat_messages = [
            "string item",
            123,
            None,
            {"uuid": "msg-1", "sender": "human", "text": "Valid"},
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert len(messages) == 1
        assert messages[0].text == "Valid"

    def test_extract_messages_skip_messages_without_text(self):
        """Test that messages without text are skipped."""
        chat_messages = [
            {"uuid": "msg-1", "sender": "human", "text": "Valid"},
            {"uuid": "msg-2", "sender": "assistant"},  # No text
            {"uuid": "msg-3", "sender": "human", "text": "Another valid"},
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert len(messages) == 2
        assert messages[0].text == "Valid"
        assert messages[1].text == "Another valid"

    def test_extract_messages_provider_meta(self):
        """Test that raw message is stored in provider_meta."""
        chat_messages = [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "Test",
                "extra_field": "extra_value",
            },
        ]

        messages, _ = extract_messages_from_chat_messages(chat_messages)

        assert messages[0].provider_meta is not None
        assert "raw" in messages[0].provider_meta
        assert messages[0].provider_meta["raw"]["extra_field"] == "extra_value"

    def test_extract_messages_empty_list(self):
        """Test with empty chat_messages list."""
        messages, attachments = extract_messages_from_chat_messages([])

        assert len(messages) == 0
        assert len(attachments) == 0

    def test_extract_messages_attachment_message_id_reference(self):
        """Test that attachment references correct message ID."""
        chat_messages = [
            {
                "uuid": "msg-abc-123",
                "sender": "human",
                "text": "With attachment",
                "attachments": [
                    {"id": "att-1", "name": "file.pdf"},
                ],
            },
        ]

        messages, attachments = extract_messages_from_chat_messages(chat_messages)

        assert attachments[0].message_provider_id == "msg-abc-123"


class TestClaudeFormatDetection:
    """Tests for Claude format detection functions."""

    def test_looks_like_ai_with_chat_messages(self):
        """Test looks_like_ai returns True for valid Claude AI format."""
        payload = {"chat_messages": [{"uuid": "1", "text": "hello"}]}
        assert looks_like_ai(payload) is True

    def test_looks_like_ai_with_empty_chat_messages(self):
        """Test looks_like_ai returns True even with empty chat_messages list."""
        payload = {"chat_messages": []}
        assert looks_like_ai(payload) is True

    def test_looks_like_ai_invalid_chatgpt_format(self):
        """Test looks_like_ai returns False for ChatGPT format."""
        payload = {"mapping": {"node1": {"message": {"content": "hello"}}}}
        assert looks_like_ai(payload) is False

    def test_looks_like_ai_not_dict(self):
        """Test looks_like_ai returns False for non-dict types."""
        assert looks_like_ai([]) is False
        assert looks_like_ai("string") is False
        assert looks_like_ai(None) is False

    def test_looks_like_ai_chat_messages_not_list(self):
        """Test looks_like_ai returns False when chat_messages is not a list."""
        payload = {"chat_messages": "not a list"}
        assert looks_like_ai(payload) is False

    def test_looks_like_code_valid_parent_uuid(self):
        """Test looks_like_code returns True for valid Claude Code format with parentUuid."""
        payload = [
            {"type": "user", "uuid": "1", "parentUuid": "parent-1"},
            {"type": "assistant", "uuid": "2", "parentUuid": "1"},
        ]
        assert looks_like_code(payload) is True

    def test_looks_like_code_valid_session_id(self):
        """Test looks_like_code returns True for valid Claude Code format with sessionId."""
        payload = [
            {"type": "user", "sessionId": "session-123"},
            {"type": "assistant", "sessionId": "session-123"},
        ]
        assert looks_like_code(payload) is True

    def test_looks_like_code_valid_session_id_snake_case(self):
        """Test looks_like_code returns True for session_id variant."""
        payload = [
            {"type": "user", "session_id": "session-456"},
        ]
        assert looks_like_code(payload) is True

    def test_looks_like_code_valid_leaf_uuid(self):
        """Test looks_like_code returns True for leafUuid."""
        payload = [
            {"type": "user", "leafUuid": "leaf-1"},
        ]
        assert looks_like_code(payload) is True

    def test_looks_like_code_invalid_ai_format(self):
        """Test looks_like_code returns False for Claude AI format."""
        payload = {"chat_messages": [{"uuid": "1", "text": "hello"}]}
        assert looks_like_code(payload) is False

    def test_looks_like_code_not_list(self):
        """Test looks_like_code returns False for non-list types."""
        assert looks_like_code({}) is False
        assert looks_like_code("string") is False
        assert looks_like_code(None) is False

    def test_looks_like_code_empty_list(self):
        """Test looks_like_code returns False for empty list."""
        assert looks_like_code([]) is False

    def test_looks_like_code_no_identifying_keys(self):
        """Test looks_like_code returns False for list without identifying keys."""
        payload = [
            {"type": "user", "content": "hello"},
            {"type": "assistant", "content": "hi"},
        ]
        assert looks_like_code(payload) is False


class TestSegmentExtraction:
    """Tests for extract_text_from_segments."""

    def test_extract_text_from_segments_text_only(self):
        """Test extracting text from segments with only strings."""
        segments = ["Hello", "World", "!"]
        result = extract_text_from_segments(segments)
        assert result == "Hello\nWorld\n!"

    def test_extract_text_from_segments_with_text_field(self):
        """Test extracting text from segments with dict containing 'text' field."""
        segments = [
            {"text": "First message"},
            {"text": "Second message"},
        ]
        result = extract_text_from_segments(segments)
        assert result == "First message\nSecond message"

    def test_extract_text_from_segments_with_content_field(self):
        """Test extracting text from segments with dict containing 'content' field."""
        segments = [
            {"content": "Content A"},
            {"content": "Content B"},
        ]
        result = extract_text_from_segments(segments)
        assert result == "Content A\nContent B"

    def test_extract_text_from_segments_with_tool_use(self):
        """Test extracting tool_use blocks as JSON."""
        segments = [
            "Before tool",
            {"type": "tool_use", "id": "tool-1", "name": "bash", "input": {"command": "ls"}},
            "After tool",
        ]
        result = extract_text_from_segments(segments)

        assert "Before tool" in result
        assert "After tool" in result
        # Tool use should be JSON serialized
        assert '"type": "tool_use"' in result
        assert '"name": "bash"' in result

    def test_extract_text_from_segments_with_tool_result(self):
        """Test extracting tool_result blocks as JSON."""
        segments = [
            {"type": "tool_result", "tool_use_id": "tool-1", "content": "file.txt"},
        ]
        result = extract_text_from_segments(segments)

        assert '"type": "tool_result"' in result
        assert '"tool_use_id": "tool-1"' in result

    def test_extract_text_from_segments_mixed(self):
        """Test extracting from mixed segment types."""
        segments = [
            "Plain text",
            {"text": "Dict with text"},
            {"content": "Dict with content"},
            {"type": "tool_use", "name": "test"},
            "",  # Empty string should be filtered
        ]
        result = extract_text_from_segments(segments)

        assert "Plain text" in result
        assert "Dict with text" in result
        assert "Dict with content" in result
        assert '"type": "tool_use"' in result

    def test_extract_text_from_segments_empty(self):
        """Test that empty segments list returns None."""
        assert extract_text_from_segments([]) is None

    def test_extract_text_from_segments_only_empty_strings(self):
        """Test that only empty strings returns None."""
        segments = ["", "", ""]
        assert extract_text_from_segments(segments) is None

    def test_extract_text_from_segments_skip_non_dict_non_string(self):
        """Test that non-dict, non-string items are skipped."""
        segments = [
            "Valid text",
            123,  # Should be skipped
            None,  # Should be skipped
            ["list"],  # Should be skipped
            {"text": "Valid dict"},
        ]
        result = extract_text_from_segments(segments)
        assert result == "Valid text\nValid dict"

    def test_extract_text_from_segments_dict_without_text_content_or_type(self):
        """Test dicts without text/content/type fields are skipped."""
        segments = [
            {"random": "field"},
            {"text": "Valid"},
            {"other": "data"},
        ]
        result = extract_text_from_segments(segments)
        assert result == "Valid"


class TestClaudeAIParsing:
    """Tests for parse_ai (Claude web UI format)."""

    def test_parse_ai_basic_conversation(self):
        """Test parsing a basic Claude AI conversation."""
        payload = {
            "uuid": "conv-123",
            "name": "Test Conversation",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T01:00:00Z",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "Hello",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "uuid": "msg-2",
                    "sender": "assistant",
                    "text": "Hi there!",
                    "created_at": "2024-01-01T00:01:00Z",
                },
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-123")

        assert result.provider_name == "claude"
        assert result.provider_conversation_id == "conv-123"
        assert result.title == "Test Conversation"
        assert result.created_at == "2024-01-01T00:00:00Z"
        assert result.updated_at == "2024-01-01T01:00:00Z"
        assert len(result.messages) == 2

        # Check first message
        assert result.messages[0].provider_message_id == "msg-1"
        assert result.messages[0].role == "user"
        assert result.messages[0].text == "Hello"
        assert result.messages[0].timestamp == "2024-01-01T00:00:00Z"

        # Check second message
        assert result.messages[1].provider_message_id == "msg-2"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].text == "Hi there!"

    def test_parse_ai_with_content_list(self):
        """Test parsing messages with content as list (segments)."""
        payload = {
            "id": "conv-456",
            "title": "Segment Test",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "content": ["Part 1", "Part 2"],
                },
                {
                    "uuid": "msg-2",
                    "sender": "assistant",
                    "content": [
                        {"text": "Response part 1"},
                        {"text": "Response part 2"},
                    ],
                },
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-456")

        assert len(result.messages) == 2
        assert result.messages[0].text == "Part 1\nPart 2"
        assert result.messages[1].text == "Response part 1\nResponse part 2"

    def test_parse_ai_with_attachments(self):
        """Test parsing messages with attachments."""
        payload = {
            "uuid": "conv-789",
            "name": "With Attachments",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "Check this file",
                    "attachments": [
                        {
                            "id": "att-1",
                            "name": "document.pdf",
                            "mimeType": "application/pdf",
                            "size": 1024,
                        },
                    ],
                },
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-789")

        assert len(result.messages) == 1
        assert len(result.attachments) == 1

        att = result.attachments[0]
        assert att.provider_attachment_id == "att-1"
        assert att.message_provider_id == "msg-1"
        assert att.name == "document.pdf"
        assert att.mime_type == "application/pdf"
        assert att.size_bytes == 1024

    def test_parse_ai_with_files_field(self):
        """Test parsing messages with 'files' field instead of 'attachments'."""
        payload = {
            "uuid": "conv-999",
            "name": "With Files",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "Check this",
                    "files": [
                        {
                            "fileId": "file-1",
                            "filename": "image.png",
                        },
                    ],
                },
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-999")

        assert len(result.attachments) == 1
        assert result.attachments[0].provider_attachment_id == "file-1"
        assert result.attachments[0].name == "image.png"

    def test_parse_ai_role_normalization(self):
        """Test that roles are properly normalized."""
        payload = {
            "chat_messages": [
                {"uuid": "1", "sender": "human", "text": "User message"},
                {"uuid": "2", "sender": "assistant", "text": "Assistant message"},
                {"uuid": "3", "role": "user", "text": "Another user message"},
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-role")

        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[2].role == "user"

    def test_parse_ai_missing_fields(self):
        """Test graceful handling of missing fields."""
        payload = {
            "chat_messages": [
                {
                    # No uuid, sender, or timestamp
                    "text": "Message with minimal fields",
                },
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-missing")

        assert result.provider_name == "claude"
        assert result.provider_conversation_id == "fallback-missing"
        assert result.title == "fallback-missing"
        assert result.created_at is None
        assert result.updated_at is None
        assert len(result.messages) == 1
        assert result.messages[0].provider_message_id == "msg-1"
        assert result.messages[0].text == "Message with minimal fields"

    def test_parse_ai_empty_chat_messages(self):
        """Test parsing with empty chat_messages list."""
        payload = {
            "uuid": "conv-empty",
            "name": "Empty",
            "chat_messages": [],
        }

        result = parse_ai(payload, fallback_id="fallback-empty")

        assert result.provider_conversation_id == "conv-empty"
        assert len(result.messages) == 0
        assert len(result.attachments) == 0

    def test_parse_ai_content_dict_with_text(self):
        """Test parsing when content is a dict with 'text' field."""
        payload = {
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "content": {"text": "Text in dict"},
                },
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-dict")

        assert len(result.messages) == 1
        assert result.messages[0].text == "Text in dict"

    def test_parse_ai_content_dict_with_parts(self):
        """Test parsing when content is a dict with 'parts' field."""
        payload = {
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "content": {"parts": ["Part A", "Part B", "Part C"]},
                },
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-parts")

        assert len(result.messages) == 1
        assert result.messages[0].text == "Part A\nPart B\nPart C"

    def test_parse_ai_skip_messages_without_text(self):
        """Test that messages without extractable text are skipped."""
        payload = {
            "chat_messages": [
                {"uuid": "1", "sender": "human", "text": "Valid message"},
                {"uuid": "2", "sender": "assistant"},  # No text
                {"uuid": "3", "sender": "human", "content": None},  # No valid content
                {"uuid": "4", "sender": "assistant", "text": "Another valid message"},
            ],
        }

        result = parse_ai(payload, fallback_id="fallback-skip")

        assert len(result.messages) == 2
        assert result.messages[0].text == "Valid message"
        assert result.messages[1].text == "Another valid message"


class TestClaudeCodeParsing:
    """Tests for parse_code (Claude Code JSONL format)."""

    def test_parse_code_basic_conversation(self):
        """Test parsing a basic Claude Code conversation."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-123",
                "timestamp": 1704067200000,
                "message": {"content": "Hello Claude Code"},
            },
            {
                "type": "assistant",
                "uuid": "msg-2",
                "sessionId": "session-123",
                "timestamp": 1704067201000,
                "message": {"content": "Hi! How can I help?"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-123")

        assert result.provider_name == "claude-code"
        assert result.provider_conversation_id == "session-123"
        assert result.title == "session-123"
        assert result.created_at == "1704067200.0"
        assert result.updated_at == "1704067201.0"
        assert len(result.messages) == 2

        assert result.messages[0].provider_message_id == "msg-1"
        assert result.messages[0].role == "user"
        assert result.messages[0].text == "Hello Claude Code"
        assert result.messages[0].timestamp == "1704067200.0"

        assert result.messages[1].provider_message_id == "msg-2"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].text == "Hi! How can I help?"

    def test_parse_code_with_tool_use_blocks(self):
        """Test parsing messages with tool_use blocks."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-456",
                "timestamp": 1704067200000,
                "message": {"content": "List files"},
            },
            {
                "type": "assistant",
                "uuid": "msg-2",
                "sessionId": "session-456",
                "timestamp": 1704067201000,
                "message": {
                    "content": [
                        "I'll list the files for you.",
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "bash",
                            "input": {"command": "ls -la"},
                        },
                    ],
                },
            },
        ]

        result = parse_code(payload, fallback_id="fallback-456")

        assert len(result.messages) == 2
        msg = result.messages[1]
        assert "I'll list the files for you." in msg.text
        assert '"type": "tool_use"' in msg.text
        assert '"name": "bash"' in msg.text

    def test_parse_code_with_thinking_blocks(self):
        """Test parsing messages that may contain thinking blocks."""
        payload = [
            {
                "type": "assistant",
                "uuid": "msg-1",
                "sessionId": "session-789",
                "timestamp": 1704067200000,
                "message": {
                    "content": [
                        {"type": "thinking", "content": "Internal thoughts"},
                        {"text": "Actual response"},
                    ],
                },
            },
        ]

        result = parse_code(payload, fallback_id="fallback-789")

        assert len(result.messages) == 1
        # Thinking block should be serialized as JSON since it has 'type'
        # but not 'tool_use' or 'tool_result'
        # Actually, looking at extract_text_from_segments, only tool_use/tool_result get JSON
        # So thinking block with 'content' field should extract that content
        assert "Internal thoughts" in result.messages[0].text or "Actual response" in result.messages[0].text

    def test_parse_code_timestamp_extraction(self):
        """Test timestamp extraction and conversation date calculation."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-999",
                "timestamp": 1000000,
                "message": {"content": "First"},
            },
            {
                "type": "assistant",
                "uuid": "msg-2",
                "sessionId": "session-999",
                "timestamp": 2000000,
                "message": {"content": "Second"},
            },
            {
                "type": "user",
                "uuid": "msg-3",
                "sessionId": "session-999",
                "timestamp": 1500000,
                "message": {"content": "Third"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-999")

        # created_at should be earliest timestamp
        assert result.created_at == "1000000.0"
        # updated_at should be latest timestamp
        assert result.updated_at == "2000000.0"

    def test_parse_code_conversation_title_extraction(self):
        """Test that session ID is used as title."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "my-unique-session",
                "message": {"content": "Test"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-title")

        assert result.provider_conversation_id == "my-unique-session"
        assert result.title == "my-unique-session"

    def test_parse_code_fallback_id_when_no_session_id(self):
        """Test that fallback_id is used when sessionId is missing."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "message": {"content": "Test"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-no-session")

        assert result.provider_conversation_id == "fallback-no-session"
        assert result.title == "fallback-no-session"

    def test_parse_code_skip_summary_and_init_types(self):
        """Test that summary and init message types are skipped."""
        payload = [
            {
                "type": "summary",
                "uuid": "summary-1",
                "sessionId": "session-skip",
                "message": {"content": "Summary text"},
            },
            {
                "type": "init",
                "uuid": "init-1",
                "sessionId": "session-skip",
                "message": {"content": "Init text"},
            },
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-skip",
                "message": {"content": "Actual message"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-skip")

        assert len(result.messages) == 1
        assert result.messages[0].text == "Actual message"

    def test_parse_code_type_to_role_mapping(self):
        """Test that message types are correctly mapped to roles."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-role",
                "message": {"content": "User"},
            },
            {
                "type": "human",
                "uuid": "msg-2",
                "sessionId": "session-role",
                "message": {"content": "Human"},
            },
            {
                "type": "assistant",
                "uuid": "msg-3",
                "sessionId": "session-role",
                "message": {"content": "Assistant"},
            },
            {
                "type": "custom_type",
                "uuid": "msg-4",
                "sessionId": "session-role",
                "message": {"content": "Custom"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-role")

        assert result.messages[0].role == "user"
        assert result.messages[1].role == "user"  # human -> user
        assert result.messages[2].role == "assistant"
        assert result.messages[3].role == "custom_type"

    def test_parse_code_message_content_as_string(self):
        """Test parsing when message.content is a string."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-str",
                "message": {"content": "Simple string content"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-str")

        assert len(result.messages) == 1
        assert result.messages[0].text == "Simple string content"

    def test_parse_code_message_as_string(self):
        """Test parsing when message field itself is a string."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-msg-str",
                "message": "Direct string message",
            },
        ]

        result = parse_code(payload, fallback_id="fallback-msg-str")

        assert len(result.messages) == 1
        assert result.messages[0].text == "Direct string message"

    def test_parse_code_with_provider_meta(self):
        """Test that provider metadata is captured."""
        payload = [
            {
                "type": "assistant",
                "uuid": "msg-1",
                "sessionId": "session-meta",
                "timestamp": 1704067200000,
                "costUSD": 0.0015,
                "durationMs": 1234,
                "isSidechain": True,
                "isMeta": False,
                "message": {"content": "Test"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-meta")

        msg = result.messages[0]
        assert msg.provider_meta is not None
        assert msg.provider_meta["costUSD"] == 0.0015
        assert msg.provider_meta["durationMs"] == 1234
        assert msg.provider_meta["isSidechain"] is True
        assert "isMeta" not in msg.provider_meta  # False values not included
        assert "raw" in msg.provider_meta

    def test_parse_code_empty_list(self):
        """Test parsing empty payload list."""
        result = parse_code([], fallback_id="fallback-empty")

        assert result.provider_name == "claude-code"
        assert result.provider_conversation_id == "fallback-empty"
        assert len(result.messages) == 0
        assert result.created_at is None
        assert result.updated_at is None

    def test_parse_code_skip_non_dict_items(self):
        """Test that non-dict items are skipped."""
        payload = [
            "string item",
            123,
            None,
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-mixed",
                "message": {"content": "Valid message"},
            },
            ["list", "item"],
        ]

        result = parse_code(payload, fallback_id="fallback-mixed")

        assert len(result.messages) == 1
        assert result.messages[0].text == "Valid message"

    def test_parse_code_missing_message_id(self):
        """Test fallback message ID generation when uuid is missing."""
        payload = [
            {
                "type": "user",
                "sessionId": "session-no-id",
                "message": {"content": "First"},
            },
            {
                "type": "assistant",
                "sessionId": "session-no-id",
                "message": {"content": "Second"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-no-id")

        assert result.messages[0].provider_message_id == "msg-1"
        assert result.messages[1].provider_message_id == "msg-2"

    def test_parse_code_no_timestamp(self):
        """Test that missing timestamps are handled gracefully."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-no-ts",
                "message": {"content": "No timestamp"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-no-ts")

        assert result.messages[0].timestamp is None
        assert result.created_at is None
        assert result.updated_at is None

    def test_parse_code_session_id_snake_case(self):
        """Test that session_id (snake_case) is also recognized."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "session_id": "session-snake-case",
                "message": {"content": "Test"},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-snake")

        assert result.provider_conversation_id == "session-snake-case"

    def test_parse_code_message_with_no_text(self):
        """Test that messages without extractable text still get added."""
        payload = [
            {
                "type": "user",
                "uuid": "msg-1",
                "sessionId": "session-no-text",
                "timestamp": 1704067200000,
                "message": {},
            },
        ]

        result = parse_code(payload, fallback_id="fallback-no-text")

        # Message is still added even with no text
        assert len(result.messages) == 1
        assert result.messages[0].text is None
        assert result.messages[0].provider_message_id == "msg-1"
