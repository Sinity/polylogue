"""Tests for ChatGPT importer (chatgpt.py)."""
from __future__ import annotations

from polylogue.importers.chatgpt import (
    _coerce_float,
    extract_messages_from_mapping,
    looks_like,
    parse,
)


class TestLooksLike:
    """Tests for ChatGPT format detection."""

    def test_looks_like_valid_mapping(self):
        """Test looks_like returns True for valid ChatGPT format with mapping."""
        payload = {
            "mapping": {
                "node-1": {
                    "message": {
                        "content": {"parts": ["Hello"]},
                        "author": {"role": "user"},
                    }
                }
            }
        }
        assert looks_like(payload) is True

    def test_looks_like_empty_mapping(self):
        """Test looks_like returns True even with empty mapping dict."""
        payload = {"mapping": {}}
        assert looks_like(payload) is True

    def test_looks_like_missing_mapping(self):
        """Test looks_like returns False when no mapping key."""
        payload = {"conversation_id": "123", "title": "Test"}
        assert looks_like(payload) is False

    def test_looks_like_mapping_not_dict(self):
        """Test looks_like returns False when mapping isn't a dict."""
        payload = {"mapping": []}
        assert looks_like(payload) is False

        payload = {"mapping": "not a dict"}
        assert looks_like(payload) is False

        payload = {"mapping": None}
        assert looks_like(payload) is False

    def test_looks_like_invalid_claude_format(self):
        """Test looks_like returns False for Claude AI format."""
        payload = {"chat_messages": [{"uuid": "1", "text": "hello"}]}
        assert looks_like(payload) is False

    def test_looks_like_not_dict(self):
        """Test looks_like returns False for non-dict types."""
        assert looks_like([]) is False
        assert looks_like("string") is False
        assert looks_like(None) is False


class TestCoerceFloat:
    """Tests for _coerce_float utility function."""

    def test_coerce_float_int(self):
        """Test converts int to float."""
        assert _coerce_float(42) == 42.0
        assert _coerce_float(0) == 0.0
        assert _coerce_float(-100) == -100.0

    def test_coerce_float_float(self):
        """Test returns float as-is."""
        assert _coerce_float(3.14) == 3.14
        assert _coerce_float(0.0) == 0.0
        assert _coerce_float(-2.5) == -2.5

    def test_coerce_float_string(self):
        """Test parses numeric string to float."""
        assert _coerce_float("42") == 42.0
        assert _coerce_float("3.14") == 3.14
        assert _coerce_float("-100.5") == -100.5
        assert _coerce_float("0") == 0.0

    def test_coerce_float_invalid_string(self):
        """Test returns None for invalid string."""
        assert _coerce_float("not a number") is None
        assert _coerce_float("abc123") is None
        assert _coerce_float("") is None

    def test_coerce_float_none(self):
        """Test returns None for None input."""
        assert _coerce_float(None) is None

    def test_coerce_float_other_types(self):
        """Test returns None for other types."""
        assert _coerce_float([]) is None
        assert _coerce_float({}) is None
        assert _coerce_float(True) is None  # bool is technically int subclass, but worth testing


class TestExtractMessagesFromMapping:
    """Tests for extract_messages_from_mapping."""

    def test_extract_messages_from_mapping_basic(self):
        """Test extracts messages correctly from basic mapping."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["Hello"]},
                    "author": {"role": "user"},
                    "create_time": 1000000,
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": ["Hi there!"]},
                    "author": {"role": "assistant"},
                    "create_time": 1000001,
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 2
        assert messages[0].provider_message_id == "msg-1"
        assert messages[0].role == "user"
        assert messages[0].text == "Hello"
        assert messages[0].timestamp == "1000000"

        assert messages[1].provider_message_id == "msg-2"
        assert messages[1].role == "assistant"
        assert messages[1].text == "Hi there!"
        assert messages[1].timestamp == "1000001"

    def test_extract_messages_timestamp_sorting(self):
        """Test messages sorted by create_time."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-3",
                    "content": {"parts": ["Third"]},
                    "author": {"role": "user"},
                    "create_time": 3000000,
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["First"]},
                    "author": {"role": "user"},
                    "create_time": 1000000,
                }
            },
            "node-3": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": ["Second"]},
                    "author": {"role": "assistant"},
                    "create_time": 2000000,
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 3
        assert messages[0].provider_message_id == "msg-1"
        assert messages[1].provider_message_id == "msg-2"
        assert messages[2].provider_message_id == "msg-3"

    def test_extract_messages_null_timestamps(self):
        """Test handles None timestamps (sorted by index)."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["No timestamp 1"]},
                    "author": {"role": "user"},
                    "create_time": None,
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": ["No timestamp 2"]},
                    "author": {"role": "assistant"},
                    "create_time": None,
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 2
        # When all timestamps are None, order is preserved by index
        assert messages[0].provider_message_id == "msg-1"
        assert messages[1].provider_message_id == "msg-2"
        assert messages[0].timestamp is None
        assert messages[1].timestamp is None

    def test_extract_messages_mixed_null_and_valid_timestamps(self):
        """Test messages with mixed None and valid timestamps."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["Has timestamp"]},
                    "author": {"role": "user"},
                    "create_time": 1000000,
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": ["No timestamp"]},
                    "author": {"role": "assistant"},
                    "create_time": None,
                }
            },
            "node-3": {
                "message": {
                    "id": "msg-3",
                    "content": {"parts": ["Also has timestamp"]},
                    "author": {"role": "user"},
                    "create_time": 2000000,
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 3
        # Messages with timestamps come first, sorted by timestamp
        # Messages without timestamps come last, sorted by index
        assert messages[0].provider_message_id == "msg-1"
        assert messages[1].provider_message_id == "msg-3"
        assert messages[2].provider_message_id == "msg-2"

    def test_extract_messages_zero_timestamp(self):
        """Test that zero timestamp is handled correctly (not treated as falsy)."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-zero",
                    "content": {"parts": ["Zero timestamp"]},
                    "author": {"role": "user"},
                    "create_time": 0,
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-positive",
                    "content": {"parts": ["Positive timestamp"]},
                    "author": {"role": "assistant"},
                    "create_time": 1000000,
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 2
        # Zero should be sorted before positive timestamp
        assert messages[0].provider_message_id == "msg-zero"
        assert messages[0].timestamp == "0"
        assert messages[1].provider_message_id == "msg-positive"

    def test_extract_messages_skips_non_dict_nodes(self):
        """Test gracefully skips non-dict nodes."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["Valid"]},
                    "author": {"role": "user"},
                }
            },
            "node-2": "not a dict",
            "node-3": None,
            "node-4": ["list"],
            "node-5": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": ["Also valid"]},
                    "author": {"role": "assistant"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 2
        assert messages[0].provider_message_id == "msg-1"
        assert messages[1].provider_message_id == "msg-2"

    def test_extract_messages_missing_content(self):
        """Test handles missing content field."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    # No content field
                    "author": {"role": "user"},
                    "create_time": 1000000,
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "content": None,  # content is None
                    "author": {"role": "user"},
                }
            },
            "node-3": {
                "message": {
                    "id": "msg-3",
                    "content": "not a dict",  # content is not dict
                    "author": {"role": "user"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        # All should be skipped
        assert len(messages) == 0

    def test_extract_messages_empty_parts(self):
        """Test handles empty parts list."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": []},
                    "author": {"role": "user"},
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": None},  # parts is None
                    "author": {"role": "assistant"},
                }
            },
            "node-3": {
                "message": {
                    "id": "msg-3",
                    "content": {},  # No parts field
                    "author": {"role": "user"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        # Empty parts should still create messages with empty text
        assert len(messages) == 3
        assert messages[0].text == ""
        assert messages[1].text == ""
        assert messages[2].text == ""

    def test_extract_messages_parts_not_list(self):
        """Test handles when parts is not a list."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": "not a list"},
                    "author": {"role": "user"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        # Should be skipped
        assert len(messages) == 0

    def test_extract_messages_multiple_parts(self):
        """Test joining multiple parts with newlines."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["Part 1", "Part 2", "Part 3"]},
                    "author": {"role": "user"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 1
        assert messages[0].text == "Part 1\nPart 2\nPart 3"

    def test_extract_messages_parts_with_none_values(self):
        """Test that None/empty parts are filtered out."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["Part 1", None, "", "Part 2", False, 0, "Part 3"]},
                    "author": {"role": "user"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 1
        # None, empty string, False, and 0 should be filtered out (they are falsy)
        assert messages[0].text == "Part 1\nPart 2\nPart 3"

    def test_extract_messages_role_normalization(self):
        """Test author.role is mapped correctly."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["User message"]},
                    "author": {"role": "user"},
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": ["Assistant message"]},
                    "author": {"role": "assistant"},
                }
            },
            "node-3": {
                "message": {
                    "id": "msg-3",
                    "content": {"parts": ["System message"]},
                    "author": {"role": "system"},
                }
            },
            "node-4": {
                "message": {
                    "id": "msg-4",
                    "content": {"parts": ["Custom role"]},
                    "author": {"role": "tool"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "system"
        assert messages[3].role == "tool"

    def test_extract_messages_missing_author_role(self):
        """Test defaults to user when author or role is missing."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["No author"]},
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "content": {"parts": ["Author is None"]},
                    "author": None,
                }
            },
            "node-3": {
                "message": {
                    "id": "msg-3",
                    "content": {"parts": ["Role is None"]},
                    "author": {"role": None},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 3
        # All should default to "user"
        assert messages[0].role == "user"
        assert messages[1].role == "user"
        assert messages[2].role == "user"

    def test_extract_messages_missing_message_id(self):
        """Test fallback message ID generation when id is missing."""
        mapping = {
            "node-1": {
                "message": {
                    # No id field in message
                    "content": {"parts": ["First"]},
                    "author": {"role": "user"},
                }
            },
            "node-2": {
                "id": "node-id-2",  # id in node, not message
                "message": {
                    "content": {"parts": ["Second"]},
                    "author": {"role": "assistant"},
                }
            },
            "node-3": {
                # Neither message nor node has id
                "message": {
                    "content": {"parts": ["Third"]},
                    "author": {"role": "user"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 3
        # Should use node id if available
        assert messages[0].provider_message_id == "msg-1"
        assert messages[1].provider_message_id == "node-id-2"
        assert messages[2].provider_message_id == "msg-3"

    def test_extract_messages_provider_meta(self):
        """Test that provider metadata is captured."""
        mapping = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "content": {"parts": ["Test"]},
                    "author": {"role": "user"},
                    "create_time": 1000000,
                    "metadata": {"model": "gpt-4"},
                }
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)

        assert len(messages) == 1
        assert messages[0].provider_meta is not None
        assert "raw" in messages[0].provider_meta
        assert messages[0].provider_meta["raw"]["id"] == "msg-1"
        assert messages[0].provider_meta["raw"]["metadata"]["model"] == "gpt-4"

    def test_extract_messages_empty_mapping(self):
        """Test empty mapping returns empty lists."""
        messages, attachments = extract_messages_from_mapping({})
        assert messages == []
        assert attachments == []

    def test_extract_messages_no_message_field(self):
        """Test skips nodes without message field."""
        mapping = {
            "node-1": {
                "id": "node-1",
                # No message field
            },
            "node-2": {
                "message": None,  # message is None
            },
            "node-3": {
                "message": "not a dict",  # message is not dict
            },
        }

        messages, attachments = extract_messages_from_mapping(mapping)
        assert len(messages) == 0


class TestParse:
    """Tests for parse function."""

    def test_parse_basic_conversation(self):
        """Test parses simple 2-message conversation."""
        payload = {
            "id": "conv-123",
            "title": "Test Conversation",
            "create_time": 1000000,
            "update_time": 2000000,
            "mapping": {
                "node-1": {
                    "message": {
                        "id": "msg-1",
                        "content": {"parts": ["Hello ChatGPT"]},
                        "author": {"role": "user"},
                        "create_time": 1000000,
                    }
                },
                "node-2": {
                    "message": {
                        "id": "msg-2",
                        "content": {"parts": ["Hello! How can I help you?"]},
                        "author": {"role": "assistant"},
                        "create_time": 1000001,
                    }
                },
            },
        }

        result = parse(payload, fallback_id="fallback-123")

        assert result.provider_name == "chatgpt"
        assert result.provider_conversation_id == "conv-123"
        assert result.title == "Test Conversation"
        assert result.created_at == "1000000"
        assert result.updated_at == "2000000"
        assert len(result.messages) == 2

    def test_parse_extracts_title(self):
        """Test gets title from payload."""
        # Test title field
        payload1 = {
            "title": "My Title",
            "mapping": {},
        }
        result1 = parse(payload1, fallback_id="fallback")
        assert result1.title == "My Title"

        # Test name field as fallback
        payload2 = {
            "name": "My Name",
            "mapping": {},
        }
        result2 = parse(payload2, fallback_id="fallback")
        assert result2.title == "My Name"

        # Test title takes precedence over name
        payload3 = {
            "title": "Title Wins",
            "name": "Name Loses",
            "mapping": {},
        }
        result3 = parse(payload3, fallback_id="fallback")
        assert result3.title == "Title Wins"

    def test_parse_extracts_conversation_id(self):
        """Test gets id/uuid/conversation_id."""
        # Test id field
        payload1 = {
            "id": "conv-id-123",
            "mapping": {},
        }
        result1 = parse(payload1, fallback_id="fallback")
        assert result1.provider_conversation_id == "conv-id-123"

        # Test uuid field
        payload2 = {
            "uuid": "conv-uuid-456",
            "mapping": {},
        }
        result2 = parse(payload2, fallback_id="fallback")
        assert result2.provider_conversation_id == "conv-uuid-456"

        # Test conversation_id field
        payload3 = {
            "conversation_id": "conv-789",
            "mapping": {},
        }
        result3 = parse(payload3, fallback_id="fallback")
        assert result3.provider_conversation_id == "conv-789"

        # Test id takes precedence
        payload4 = {
            "id": "id-wins",
            "uuid": "uuid-loses",
            "conversation_id": "conv-id-loses",
            "mapping": {},
        }
        result4 = parse(payload4, fallback_id="fallback")
        assert result4.provider_conversation_id == "id-wins"

    def test_parse_extracts_timestamps(self):
        """Test create_time/update_time extraction."""
        payload = {
            "create_time": 1704067200,
            "update_time": 1704153600,
            "mapping": {},
        }

        result = parse(payload, fallback_id="fallback")

        assert result.created_at == "1704067200"
        assert result.updated_at == "1704153600"

    def test_parse_uses_fallback_id(self):
        """Test when no id in payload, uses fallback_id."""
        payload = {
            "title": "No ID Conversation",
            "mapping": {},
        }

        result = parse(payload, fallback_id="my-fallback-id")

        assert result.provider_conversation_id == "my-fallback-id"
        assert result.title == "No ID Conversation"

    def test_parse_uses_fallback_id_for_title(self):
        """Test when no title/name in payload, uses fallback_id for title."""
        payload = {
            "id": "conv-123",
            "mapping": {},
        }

        result = parse(payload, fallback_id="fallback-title")

        assert result.provider_conversation_id == "conv-123"
        assert result.title == "fallback-title"

    def test_parse_missing_timestamps(self):
        """Test handles missing timestamps."""
        payload = {
            "id": "conv-no-ts",
            "mapping": {},
        }

        result = parse(payload, fallback_id="fallback")

        assert result.created_at is None
        assert result.updated_at is None

    def test_parse_zero_timestamps(self):
        """Test handles zero timestamps (should not be treated as falsy)."""
        payload = {
            "id": "conv-zero",
            "create_time": 0,
            "update_time": 0,
            "mapping": {},
        }

        result = parse(payload, fallback_id="fallback")

        assert result.created_at == "0"
        assert result.updated_at == "0"

    def test_parse_missing_mapping(self):
        """Test handles missing mapping field."""
        payload = {
            "id": "conv-no-mapping",
            "title": "No Mapping",
        }

        result = parse(payload, fallback_id="fallback")

        assert result.provider_conversation_id == "conv-no-mapping"
        assert len(result.messages) == 0

    def test_parse_complex_conversation(self):
        """Test parsing a more complex conversation with multiple exchanges."""
        payload = {
            "id": "conv-complex",
            "title": "Complex Conversation",
            "create_time": 1000000,
            "update_time": 1000005,
            "mapping": {
                "root": {
                    "message": None,
                },
                "node-1": {
                    "message": {
                        "id": "msg-1",
                        "content": {"parts": ["What is Python?"]},
                        "author": {"role": "user"},
                        "create_time": 1000001,
                    }
                },
                "node-2": {
                    "message": {
                        "id": "msg-2",
                        "content": {"parts": ["Python is a programming language."]},
                        "author": {"role": "assistant"},
                        "create_time": 1000002,
                    }
                },
                "node-3": {
                    "message": {
                        "id": "msg-3",
                        "content": {"parts": ["Tell me more."]},
                        "author": {"role": "user"},
                        "create_time": 1000003,
                    }
                },
                "node-4": {
                    "message": {
                        "id": "msg-4",
                        "content": {
                            "parts": [
                                "Python was created by Guido van Rossum.",
                                "It is known for its readability.",
                            ]
                        },
                        "author": {"role": "assistant"},
                        "create_time": 1000004,
                    }
                },
            },
        }

        result = parse(payload, fallback_id="fallback")

        assert len(result.messages) == 4
        assert result.messages[0].text == "What is Python?"
        assert result.messages[1].text == "Python is a programming language."
        assert result.messages[2].text == "Tell me more."
        assert result.messages[3].text == "Python was created by Guido van Rossum.\nIt is known for its readability."

    def test_parse_minimal_payload(self):
        """Test parsing with minimal valid payload."""
        payload = {"mapping": {}}

        result = parse(payload, fallback_id="minimal")

        assert result.provider_name == "chatgpt"
        assert result.provider_conversation_id == "minimal"
        assert result.title == "minimal"
        assert result.created_at is None
        assert result.updated_at is None
        assert len(result.messages) == 0

    def test_parse_preserves_provider_name(self):
        """Test that provider_name is always 'chatgpt'."""
        payload = {"mapping": {}}

        result = parse(payload, fallback_id="fallback")

        assert result.provider_name == "chatgpt"
