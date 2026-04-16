"""Tests for storage record mapping functions.

Production code under test: polylogue/storage/store.py
Functions: _parse_json, _row_get, _row_to_conversation, _row_to_message, _row_to_raw_conversation
"""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.errors import DatabaseError
from polylogue.storage.backends.queries.mappers import (
    _parse_json,
    _row_get,
    _row_to_conversation,
    _row_to_message,
    _row_to_raw_conversation,
)
from polylogue.storage.store import (
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
)


def make_row(columns: dict) -> sqlite3.Row:
    """Create a sqlite3.Row from a dict of column_name→value."""
    cols = list(columns.keys())
    vals = list(columns.values())
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Use appropriate types for columns that need numeric handling
    type_map = {
        "version": "INTEGER",
        "branch_index": "INTEGER",
        "word_count": "INTEGER",
        "has_tool_use": "INTEGER",
        "has_thinking": "INTEGER",
        "source_index": "INTEGER",
        "block_index": "INTEGER",
        "validation_drift_count": "INTEGER",
    }
    col_defs = ", ".join(f'"{c}" {type_map.get(c, "TEXT")}' for c in cols)
    placeholders = ", ".join("?" * len(cols))
    conn.execute(f"CREATE TABLE t ({col_defs})")
    conn.execute(f"INSERT INTO t VALUES ({placeholders})", vals)
    row = conn.execute("SELECT * FROM t").fetchone()
    conn.close()
    return row


# =============================================================================
# _parse_json
# =============================================================================


class TestParseJson:
    """Tests for _parse_json helper."""

    def test_valid_json_string(self):
        """Valid JSON string returns parsed data."""
        result = _parse_json('{"key": "value"}', field="test_field", record_id="rec1")
        assert result == {"key": "value"}

    def test_none_returns_none(self):
        """None input returns None."""
        result = _parse_json(None, field="test_field", record_id="rec1")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        result = _parse_json("", field="test_field", record_id="rec1")
        assert result is None

    def test_corrupt_json_raises_database_error(self):
        """Corrupt JSON raises DatabaseError with diagnostic context."""
        with pytest.raises(DatabaseError, match="Corrupt JSON"):
            _parse_json("{invalid json}", field="provider_meta", record_id="conv-123")

    def test_database_error_includes_field_name(self):
        """DatabaseError message includes the field name for diagnostics."""
        with pytest.raises(DatabaseError, match="provider_meta"):
            _parse_json("not valid", field="provider_meta", record_id="conv-1")

    def test_database_error_includes_record_id(self):
        """DatabaseError message includes the record ID for diagnostics."""
        with pytest.raises(DatabaseError, match="conv-999"):
            _parse_json("{bad}", field="meta", record_id="conv-999")


# =============================================================================
# _row_get
# =============================================================================


class TestRowGet:
    """Tests for _row_get helper."""

    def test_existing_key_returns_value(self):
        """Returns value for an existing column key."""
        row = make_row({"name": "test", "value": "42"})
        assert _row_get(row, "name") == "test"
        assert _row_get(row, "value") == "42"

    def test_missing_key_returns_default(self):
        """Returns default for a missing column key."""
        row = make_row({"name": "test"})
        assert _row_get(row, "nonexistent") is None
        assert _row_get(row, "nonexistent", "fallback") == "fallback"

    def test_default_none(self):
        """Default is None when not specified."""
        row = make_row({"x": "1"})
        assert _row_get(row, "missing") is None


# =============================================================================
# _row_to_conversation
# =============================================================================


class TestRowToConversation:
    """Tests for _row_to_conversation mapper."""

    def test_maps_required_fields(self):
        """All required fields are mapped from row to ConversationRecord."""
        row = make_row(
            {
                "conversation_id": "conv-1",
                "provider_name": "claude-ai",
                "provider_conversation_id": "ext-conv-1",
                "title": "Test Chat",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "sort_key": None,
                "content_hash": "abcdef1234567890",
                "provider_meta": None,
                "metadata": None,
                "version": 1,
                "parent_conversation_id": None,
                "branch_type": None,
                "raw_id": None,
            }
        )
        result = _row_to_conversation(row)
        assert isinstance(result, ConversationRecord)
        assert result.conversation_id == "conv-1"
        assert result.provider_name == "claude-ai"
        assert result.title == "Test Chat"
        assert result.content_hash == "abcdef1234567890"

    def test_maps_json_provider_meta(self):
        """JSON provider_meta is parsed from string."""
        import json

        meta = {"model": "claude-3"}
        row = make_row(
            {
                "conversation_id": "conv-2",
                "provider_name": "claude-ai",
                "provider_conversation_id": "ext-2",
                "title": "With Meta",
                "created_at": None,
                "updated_at": None,
                "sort_key": None,
                "content_hash": "hash123456789abc",
                "provider_meta": json.dumps(meta),
                "metadata": None,
                "version": 1,
                "parent_conversation_id": None,
                "branch_type": None,
                "raw_id": None,
            }
        )
        result = _row_to_conversation(row)
        assert result.provider_meta == {"model": "claude-3"}


# =============================================================================
# _row_to_message
# =============================================================================


class TestRowToMessage:
    """Tests for _row_to_message mapper."""

    def test_maps_required_fields(self):
        """All required fields are mapped from row to MessageRecord."""
        row = make_row(
            {
                "message_id": "m-1",
                "conversation_id": "conv-1",
                "provider_message_id": "ext-m-1",
                "role": "user",
                "text": "Hello world",
                "sort_key": 1704106200.0,
                "content_hash": "msghash123456789",
                "version": 1,
                "parent_message_id": None,
                "branch_index": 0,
                "provider_name": "claude-ai",
                "word_count": 2,
                "has_tool_use": 0,
                "has_thinking": 0,
            }
        )
        result = _row_to_message(row)
        assert isinstance(result, MessageRecord)
        assert result.message_id == "m-1"
        assert result.conversation_id == "conv-1"
        assert result.role == "user"
        assert result.text == "Hello world"
        assert result.branch_index == 0

    def test_null_branch_index_defaults_to_zero(self):
        """Null branch_index from DB maps to 0."""
        row = make_row(
            {
                "message_id": "m-2",
                "conversation_id": "conv-1",
                "provider_message_id": None,
                "role": "assistant",
                "text": "Reply",
                "sort_key": None,
                "content_hash": "msghash987654321",
                "version": 1,
                "parent_message_id": None,
                "branch_index": None,
                "provider_name": "",
                "word_count": 0,
                "has_tool_use": 0,
                "has_thinking": 0,
            }
        )
        result = _row_to_message(row)
        assert result.branch_index == 0


# =============================================================================
# _row_to_raw_conversation
# =============================================================================


class TestRowToRawConversation:
    """Tests for _row_to_raw_conversation mapper."""

    def test_maps_all_fields(self):
        """All fields are mapped from row to RawConversationRecord."""
        row = make_row(
            {
                "raw_id": "sha256hash",
                "provider_name": "chatgpt",
                "payload_provider": None,
                "source_name": "inbox",
                "source_path": "/tmp/data.json",
                "source_index": "0",
                "blob_size": 14,
                "acquired_at": "2024-01-01T00:00:00Z",
                "file_mtime": "2024-01-01T00:00:00Z",
                "parsed_at": None,
                "parse_error": None,
                "validated_at": None,
                "validation_status": None,
                "validation_error": None,
                "validation_drift_count": None,
                "validation_provider": None,
                "validation_mode": None,
            }
        )
        result = _row_to_raw_conversation(row)
        assert isinstance(result, RawConversationRecord)
        assert result.raw_id == "sha256hash"
        assert result.provider_name == "chatgpt"
        assert result.source_name == "inbox"
