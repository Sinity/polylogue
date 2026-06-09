"""Tests for storage record mapping functions.

Insightion code under test: polylogue/storage/runtime/__init__.py
Functions: _parse_json, _row_get, _row_to_session, _row_to_message, _row_to_raw_session
"""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.core.enums import Origin
from polylogue.errors import DatabaseError
from polylogue.storage.runtime import (
    MessageRecord,
    RawSessionRecord,
    SessionRecord,
)
from polylogue.storage.sqlite.queries.mappers import (
    _parse_json,
    _row_get,
    _row_to_message,
    _row_to_raw_session,
    _row_to_session,
)


def make_row(columns: dict[str, object]) -> sqlite3.Row:
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
    assert row is not None
    assert isinstance(row, sqlite3.Row)
    return row


# =============================================================================
# _parse_json
# =============================================================================


class TestParseJson:
    """Tests for _parse_json helper."""

    def test_valid_json_string(self: object) -> None:
        """Valid JSON string returns parsed data."""
        result = _parse_json('{"key": "value"}', field="test_field", record_id="rec1")
        assert result == {"key": "value"}

    def test_none_returns_none(self: object) -> None:
        """None input returns None."""
        result = _parse_json(None, field="test_field", record_id="rec1")
        assert result is None

    def test_empty_string_returns_none(self: object) -> None:
        """Empty string returns None."""
        result = _parse_json("", field="test_field", record_id="rec1")
        assert result is None

    def test_corrupt_json_raises_database_error(self: object) -> None:
        """Corrupt JSON raises DatabaseError with diagnostic context."""
        with pytest.raises(DatabaseError, match="Corrupt JSON"):
            _parse_json("{invalid json}", field="provider_meta", record_id="conv-123")

    def test_database_error_includes_field_name(self: object) -> None:
        """DatabaseError message includes the field name for diagnostics."""
        with pytest.raises(DatabaseError, match="provider_meta"):
            _parse_json("not valid", field="provider_meta", record_id="conv-1")

    def test_database_error_includes_record_id(self: object) -> None:
        """DatabaseError message includes the record ID for diagnostics."""
        with pytest.raises(DatabaseError, match="conv-999"):
            _parse_json("{bad}", field="meta", record_id="conv-999")


# =============================================================================
# _row_get
# =============================================================================


class TestRowGet:
    """Tests for _row_get helper."""

    def test_existing_key_returns_value(self: object) -> None:
        """Returns value for an existing column key."""
        row = make_row({"name": "test", "value": "42"})
        assert _row_get(row, "name") == "test"
        assert _row_get(row, "value") == "42"

    def test_missing_key_returns_default(self: object) -> None:
        """Returns default for a missing column key."""
        row = make_row({"name": "test"})
        assert _row_get(row, "nonexistent") is None
        assert _row_get(row, "nonexistent", "fallback") == "fallback"

    def test_default_none(self: object) -> None:
        """Default is None when not specified."""
        row = make_row({"x": "1"})
        assert _row_get(row, "missing") is None


# =============================================================================
# _row_to_session
# =============================================================================


class TestRowToSession:
    """Tests for _row_to_session mapper."""

    def test_maps_required_fields(self: object) -> None:
        """All required fields are mapped from row to SessionRecord."""
        row = make_row(
            {
                "session_id": "conv-1",
                "origin": "claude-ai-export",
                "native_id": "ext-conv-1",
                "title": "Test Chat",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "sort_key": None,
                "content_hash": "abcdef1234567890",
                "metadata": None,
                "version": 1,
                "parent_session_id": None,
                "branch_type": None,
                "raw_id": None,
            }
        )
        result = _row_to_session(row)
        assert isinstance(result, SessionRecord)
        assert result.session_id == "conv-1"
        assert result.origin == Origin.from_string("claude-ai-export")
        assert result.title == "Test Chat"
        assert result.content_hash == "abcdef1234567890"

    def test_maps_json_metadata(self: object) -> None:
        """JSON metadata is parsed from string."""
        import json

        meta = {"model": "claude-3"}
        row = make_row(
            {
                "session_id": "conv-2",
                "origin": "claude-ai-export",
                "native_id": "ext-2",
                "title": "With Meta",
                "created_at": None,
                "updated_at": None,
                "sort_key": None,
                "content_hash": "hash123456789abc",
                "metadata": json.dumps(meta),
                "version": 1,
                "parent_session_id": None,
                "branch_type": None,
                "raw_id": None,
            }
        )
        result = _row_to_session(row)
        assert result.metadata == {"model": "claude-3"}


# =============================================================================
# _row_to_message
# =============================================================================


class TestRowToMessage:
    """Tests for _row_to_message mapper."""

    def test_maps_required_fields(self: object) -> None:
        """All required fields are mapped from row to MessageRecord."""
        row = make_row(
            {
                "message_id": "m-1",
                "session_id": "conv-1",
                "provider_message_id": "ext-m-1",
                "role": "user",
                "text": "Hello world",
                "sort_key": 1704106200.0,
                "content_hash": "msghash123456789",
                "version": 1,
                "parent_message_id": None,
                "branch_index": 0,
                "source_name": "claude-ai",
                "word_count": 2,
                "has_tool_use": 0,
                "has_thinking": 0,
            }
        )
        result = _row_to_message(row)
        assert isinstance(result, MessageRecord)
        assert result.message_id == "m-1"
        assert result.session_id == "conv-1"
        assert result.role == "user"
        assert result.text == "Hello world"
        assert result.branch_index == 0

    def test_null_branch_index_defaults_to_zero(self: object) -> None:
        """Null branch_index from DB maps to 0."""
        row = make_row(
            {
                "message_id": "m-2",
                "session_id": "conv-1",
                "provider_message_id": None,
                "role": "assistant",
                "text": "Reply",
                "sort_key": None,
                "content_hash": "msghash987654321",
                "version": 1,
                "parent_message_id": None,
                "branch_index": None,
                "source_name": "",
                "word_count": 0,
                "has_tool_use": 0,
                "has_thinking": 0,
            }
        )
        result = _row_to_message(row)
        assert result.branch_index == 0


# =============================================================================
# _row_to_raw_session
# =============================================================================


class TestRowToRawSession:
    """Tests for _row_to_raw_session mapper."""

    def test_maps_all_fields(self: object) -> None:
        """All fields are mapped from row to RawSessionRecord."""
        from polylogue.core.enums import Origin
        from polylogue.core.sources import provider_from_origin

        row = make_row(
            {
                "raw_id": "sha256hash",
                # raw_sessions carries a single origin column (#1743); source_name
                # and payload_provider project from it on read.
                "origin": "claude-ai-export",
                "source_path": "/tmp/data.json",
                "source_index": 0,
                "blob_size": 14,
                "acquired_at_ms": 1704067200000,
                "file_mtime_ms": 1704067200000,
                "parsed_at_ms": None,
                "parse_error": None,
                "validated_at_ms": None,
                "validation_status": None,
                "validation_error": None,
                "validation_drift_count": 0,
                "validation_mode": None,
            }
        )
        result = _row_to_raw_session(row)
        assert isinstance(result, RawSessionRecord)
        assert result.raw_id == "sha256hash"
        assert result.source_name == provider_from_origin(Origin.from_string("claude-ai-export")).value
