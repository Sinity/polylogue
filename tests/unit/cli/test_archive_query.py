"""Tests for pure helper functions in polylogue.cli.archive_query."""

from __future__ import annotations

import csv
import io
import json
import types
from typing import cast
from unittest.mock import MagicMock

import click
import pytest

from polylogue.archive.message.roles import Role
from polylogue.cli.archive_query import (
    _build_cursor,
    _csv,
    _csv_tokens,
    _decode_cursor,
    _ellipsize,
    _emit_delete,
    _has_value,
    _hit_line,
    _limit,
    _message_removed_by_transform,
    _message_roles,
    _message_type,
    _metadata_pairs,
    _offset,
    _optional_date_ms,
    _optional_int,
    _optional_str,
    _paginate_rows,
    _project_payload,
    _resolve_excluded_origins,
    _resolve_origins,
    _selected_fields,
    _sort,
    _stats_by_line,
    _summary_line,
    _tool_tokens,
    _transform,
    _tuple_tokens,
)
from polylogue.operations import OperationSpec, build_runtime_operation_catalog
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary
from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow


# Tests for _resolve_origins
class TestResolveOrigins:
    """Tests for _resolve_origins."""

    def test_explicit_origin_single(self) -> None:
        """Single origin returns as tuple."""
        params: dict[str, object] = {"origin": "claude-code-session"}
        result = _resolve_origins(params)
        assert result == ("claude-code-session",)

    def test_explicit_origin_csv(self) -> None:
        """CSV origins are parsed and deduplicated."""
        params: dict[str, object] = {"origin": "claude-code-session,chatgpt-export"}
        result = _resolve_origins(params)
        assert result == ("claude-code-session", "chatgpt-export")

    def test_explicit_origin_deduped(self) -> None:
        """Duplicate origins are deduplicated."""
        params: dict[str, object] = {"origin": "claude-code-session,claude-code-session"}
        result = _resolve_origins(params)
        assert result == ("claude-code-session",)

    def test_explicit_origin_stripped(self) -> None:
        """Whitespace is stripped."""
        params: dict[str, object] = {"origin": "  claude-code-session  ,  chatgpt-export  "}
        result = _resolve_origins(params)
        assert result == ("claude-code-session", "chatgpt-export")

    def test_empty_params(self) -> None:
        """Empty params returns empty tuple."""
        result = _resolve_origins({})
        assert result == ()

    def test_provider_param_is_ignored(self) -> None:
        """The public root query surface speaks origin only (#1810).

        The legacy ``provider`` -> origin fallback was removed; a provider
        token on the root query no longer resolves to any origin.
        """
        assert _resolve_origins({"provider": "claude-code"}) == ()


# Tests for _resolve_excluded_origins
class TestResolveExcludedOrigins:
    """Tests for _resolve_excluded_origins."""

    def test_explicit_excluded_origin(self) -> None:
        """Explicit excluded origins are parsed."""
        params: dict[str, object] = {"exclude_origin": "claude-code-session,chatgpt-export"}
        result = _resolve_excluded_origins(params)
        assert result == ("claude-code-session", "chatgpt-export")

    def test_exclude_provider_param_is_ignored(self) -> None:
        """Excluded-provider fallback removed: origin vocabulary only (#1810)."""
        assert _resolve_excluded_origins({"exclude_provider": "claude-code"}) == ()

    def test_empty_params(self) -> None:
        """Empty params returns empty tuple."""
        result = _resolve_excluded_origins({})
        assert result == ()


# Tests for _csv_tokens
class TestCsvTokens:
    """Tests for _csv_tokens."""

    def test_none_value(self) -> None:
        """None returns empty tuple."""
        result = _csv_tokens(None)
        assert result == ()

    def test_single_string(self) -> None:
        """Single comma-separated string is split."""
        result = _csv_tokens("a,b,c")
        assert result == ("a", "b", "c")

    def test_whitespace_stripped(self) -> None:
        """Whitespace is stripped from each token."""
        result = _csv_tokens("  a  ,  b  ,  c  ")
        assert result == ("a", "b", "c")

    def test_empty_tokens_dropped(self) -> None:
        """Empty tokens are dropped."""
        result = _csv_tokens("a,,b")
        assert result == ("a", "b")

    def test_tuple_of_strings(self) -> None:
        """Tuple of CSV strings is flattened."""
        result = _csv_tokens(("a,b", "c,d"))
        assert result == ("a", "b", "c", "d")

    def test_empty_tuple(self) -> None:
        """Empty tuple returns empty tuple (not ('()',))."""
        result = _csv_tokens(())
        assert result == ()

    def test_list_of_strings(self) -> None:
        """List of CSV strings is flattened."""
        result = _csv_tokens(["a,b", "c"])
        assert result == ("a", "b", "c")


# Tests for _tuple_tokens
class TestTupleTokens:
    """Tests for _tuple_tokens."""

    def test_none_value(self) -> None:
        """None returns empty tuple."""
        result = _tuple_tokens(None)
        assert result == ()

    def test_string_value(self) -> None:
        """Single string is wrapped in tuple."""
        result = _tuple_tokens("value")
        assert result == ("value",)

    def test_empty_string(self) -> None:
        """Empty string returns empty tuple."""
        result = _tuple_tokens("")
        assert result == ()

    def test_whitespace_string(self) -> None:
        """Whitespace-only string returns empty tuple."""
        result = _tuple_tokens("   ")
        assert result == ()

    def test_iterable(self) -> None:
        """Iterable values are stripped and collected."""
        result = _tuple_tokens(["a", "b", "c"])
        assert result == ("a", "b", "c")

    def test_iterable_with_empty_strings(self) -> None:
        """Empty strings in iterable are dropped."""
        result = _tuple_tokens(["a", "", "b"])
        assert result == ("a", "b")


# Tests for _metadata_pairs
class TestMetadataPairs:
    """Tests for _metadata_pairs."""

    def test_none_value(self) -> None:
        """None returns empty tuple."""
        result = _metadata_pairs(None)
        assert result == ()

    def test_list_of_pairs(self) -> None:
        """List of 2-tuples is converted to tuple of string pairs."""
        result = _metadata_pairs([("key1", "value1"), ("key2", "value2")])
        assert result == (("key1", "value1"), ("key2", "value2"))

    def test_list_of_lists(self) -> None:
        """List of lists with 2+ elements works."""
        result = _metadata_pairs([["key1", "value1"], ["key2", "value2"]])
        assert result == (("key1", "value1"), ("key2", "value2"))

    def test_non_iterable_raises_error(self) -> None:
        """Non-iterable value raises UsageError."""
        with pytest.raises(click.UsageError, match="expects key/value pairs"):
            _metadata_pairs("not-a-list")

    def test_non_pair_element_raises_error(self) -> None:
        """Element that is not a 2+ sequence raises UsageError."""
        with pytest.raises(click.UsageError, match="expects key/value pairs"):
            _metadata_pairs([("key", "value"), "not-a-pair"])

    def test_single_element_raises_error(self) -> None:
        """1-element sequence raises UsageError."""
        with pytest.raises(click.UsageError, match="expects key/value pairs"):
            _metadata_pairs([["single"]])


# Tests for _tool_tokens
class TestToolTokens:
    """Tests for _tool_tokens."""

    def test_lowercases_tokens(self) -> None:
        """Tool tokens are lowercased."""
        result = _tool_tokens("Tool,ANOTHER")
        assert result == ("tool", "another")

    def test_none_value(self) -> None:
        """None returns empty tuple."""
        result = _tool_tokens(None)
        assert result == ()


# Tests for _message_type
class TestMessageType:
    """Tests for _message_type."""

    def test_none_value(self) -> None:
        """None returns None."""
        result = _message_type(None)
        assert result is None

    def test_false_value(self) -> None:
        """False returns None."""
        result = _message_type(False)
        assert result is None

    def test_valid_message_type(self) -> None:
        """Valid message type returns its value."""
        result = _message_type("message")
        assert result == "message"

    def test_invalid_message_type(self) -> None:
        """Invalid message type raises UsageError."""
        with pytest.raises(click.UsageError):
            _message_type("invalid-type")


# Tests for _message_roles
class TestMessageRoles:
    """Tests for _message_roles."""

    def test_dialogue_only_true(self) -> None:
        """dialogue_only=True returns (USER, ASSISTANT)."""
        params: dict[str, object] = {"dialogue_only": True}
        result = _message_roles(params)
        assert result == (Role.USER, Role.ASSISTANT)

    def test_empty_params(self) -> None:
        """Empty params returns empty tuple."""
        result = _message_roles({})
        assert result == ()

    def test_message_role_param(self) -> None:
        """message_role param is normalized."""
        params: dict[str, object] = {"message_role": "user"}
        result = _message_roles(params)
        assert result == (Role.USER,)

    def test_invalid_message_role(self) -> None:
        """Invalid message role raises UsageError."""
        params: dict[str, object] = {"message_role": "invalid-role"}
        with pytest.raises(click.UsageError):
            _message_roles(params)


# Tests for _sort
class TestSort:
    """Tests for _sort."""

    def test_none_value(self) -> None:
        """None returns None."""
        result = _sort(None)
        assert result is None

    def test_false_value(self) -> None:
        """False returns None."""
        result = _sort(False)
        assert result is None

    def test_valid_sort_values(self) -> None:
        """Valid sort values are returned."""
        for sort_val in ["date", "messages", "words", "longest", "tokens", "random"]:
            result = _sort(sort_val)
            assert result == sort_val

    def test_invalid_sort_raises_error(self) -> None:
        """Invalid sort value raises UsageError."""
        with pytest.raises(click.UsageError, match="sort must be one of"):
            _sort("invalid-sort")


# Tests for _transform
class TestTransform:
    """Tests for _transform."""

    def test_none_value(self) -> None:
        """None returns None."""
        result = _transform(None)
        assert result is None

    def test_false_value(self) -> None:
        """False returns None."""
        result = _transform(False)
        assert result is None

    def test_valid_transforms(self) -> None:
        """Valid transform values are returned."""
        for transform_val in ["strip-tools", "strip-thinking", "strip-all"]:
            result = _transform(transform_val)
            assert result == transform_val

    def test_invalid_transform_raises_error(self) -> None:
        """Invalid transform raises UsageError."""
        with pytest.raises(click.UsageError, match="transform must be one of"):
            _transform("invalid-transform")


# Tests for _optional_int
class TestOptionalInt:
    """Tests for _optional_int."""

    def test_int_value(self) -> None:
        """Int value is returned."""
        result = _optional_int(42)
        assert result == 42

    def test_zero_is_returned(self) -> None:
        """Zero is returned (0 is not in false set)."""
        result = _optional_int(0)
        assert result == 0

    def test_non_int_returns_none(self) -> None:
        """Non-int value returns None."""
        result = _optional_int("42")
        assert result is None


# Tests for _optional_str
class TestOptionalStr:
    """Tests for _optional_str."""

    def test_none_value(self) -> None:
        """None returns None."""
        result = _optional_str(None)
        assert result is None

    def test_string_value(self) -> None:
        """String value is stripped."""
        result = _optional_str("  hello  ")
        assert result == "hello"

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        result = _optional_str("")
        assert result is None

    def test_whitespace_string_returns_none(self) -> None:
        """Whitespace-only string returns None."""
        result = _optional_str("   ")
        assert result is None


# Tests for _limit
class TestLimit:
    """Tests for _limit."""

    def test_positive_int(self) -> None:
        """Positive int is returned."""
        result = _limit({"limit": 50})
        assert result == 50

    def test_default_when_missing(self) -> None:
        """Default 20 when limit is missing."""
        result = _limit({})
        assert result == 20

    def test_default_when_non_positive(self) -> None:
        """Default 20 when limit is non-positive."""
        result = _limit({"limit": 0})
        assert result == 20
        result = _limit({"limit": -1})
        assert result == 20

    def test_default_when_non_int(self) -> None:
        """Default 20 when limit is non-int."""
        result = _limit({"limit": "50"})
        assert result == 20


# Tests for _offset
class TestOffset:
    """Tests for _offset."""

    def test_positive_int(self) -> None:
        """Positive int is returned."""
        result = _offset({"offset": 10})
        assert result == 10

    def test_default_when_missing(self) -> None:
        """Default 0 when offset is missing."""
        result = _offset({})
        assert result == 0

    def test_default_when_non_positive(self) -> None:
        """Default 0 when offset is non-positive."""
        result = _offset({"offset": 0})
        assert result == 0
        result = _offset({"offset": -1})
        assert result == 0

    def test_default_when_non_int(self) -> None:
        """Default 0 when offset is non-int."""
        result = _offset({"offset": "10"})
        assert result == 0


# Tests for _optional_date_ms
class TestOptionalDateMs:
    """Tests for _optional_date_ms."""

    def test_none_value(self) -> None:
        """None returns None."""
        result = _optional_date_ms("since", None)
        assert result is None

    def test_false_value(self) -> None:
        """False returns None."""
        result = _optional_date_ms("since", False)
        assert result is None

    def test_valid_iso_date(self) -> None:
        """Valid ISO date is parsed to milliseconds."""
        result = _optional_date_ms("since", "2026-01-15")
        assert isinstance(result, int)
        assert result > 0

    def test_invalid_date_raises_exception(self) -> None:
        """Invalid date raises ClickException."""
        with pytest.raises(click.ClickException, match="Cannot parse date"):
            _optional_date_ms("since", "not-a-date")


# Tests for _has_value
class TestHasValue:
    """Tests for _has_value."""

    def test_none_is_false(self) -> None:
        """None returns False."""
        assert _has_value(None) is False

    def test_false_is_false(self) -> None:
        """False returns False."""
        assert _has_value(False) is False

    def test_empty_string_is_false(self) -> None:
        """Empty string returns False."""
        assert _has_value("") is False

    def test_empty_tuple_is_false(self) -> None:
        """Empty tuple returns False."""
        assert _has_value(()) is False

    def test_empty_list_is_false(self) -> None:
        """Empty list returns False."""
        assert _has_value([]) is False

    def test_string_is_true(self) -> None:
        """Non-empty string returns True."""
        assert _has_value("value") is True

    def test_zero_is_true(self) -> None:
        """Zero returns True (0 is not in the false set)."""
        assert _has_value(0) is True

    def test_tuple_is_true(self) -> None:
        """Non-empty tuple returns True."""
        assert _has_value(("a",)) is True


# Tests for _selected_fields
class TestSelectedFields:
    """Tests for _selected_fields."""

    def test_none_value(self) -> None:
        """None returns None."""
        result = _selected_fields(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        result = _selected_fields("")
        assert result is None

    def test_single_field(self) -> None:
        """Single field returns frozenset."""
        result = _selected_fields("field1")
        assert result == frozenset({"field1"})

    def test_multiple_fields(self) -> None:
        """CSV fields are parsed."""
        result = _selected_fields("field1,field2,field3")
        assert result == frozenset({"field1", "field2", "field3"})

    def test_whitespace_stripped(self) -> None:
        """Whitespace is stripped from each field."""
        result = _selected_fields("  field1  ,  field2  ")
        assert result == frozenset({"field1", "field2"})

    def test_empty_fields_dropped(self) -> None:
        """Empty fields are dropped."""
        result = _selected_fields("field1,,field2")
        assert result == frozenset({"field1", "field2"})

    def test_whitespace_only_returns_none(self) -> None:
        """Whitespace-only result returns None."""
        result = _selected_fields("   ")
        assert result is None


# Tests for _project_payload
class TestProjectPayload:
    """Tests for _project_payload."""

    def test_no_fields_specified(self) -> None:
        """No fields returns copy of payload."""
        payload: dict[str, object] = {"a": 1, "b": 2, "c": 3}
        result = _project_payload(payload, None)
        assert result == payload
        assert result is not payload  # Should be a copy

    def test_select_subset_of_fields(self) -> None:
        """Selected fields are kept."""
        payload: dict[str, object] = {"a": 1, "b": 2, "c": 3}
        result = _project_payload(payload, "a,c")
        assert result == {"a": 1, "c": 3}

    def test_missing_fields_ignored(self) -> None:
        """Missing fields are safely ignored."""
        payload: dict[str, object] = {"a": 1, "b": 2}
        result = _project_payload(payload, "a,missing")
        assert result == {"a": 1}


# Tests for _csv
class TestCsv:
    """Tests for _csv."""

    def test_empty_list(self) -> None:
        """Empty list returns empty string."""
        result = _csv([])
        assert result == ""

    def test_single_row(self) -> None:
        """Single row is formatted with header."""
        items: list[dict[str, object]] = [{"id": "1", "name": "test"}]
        result = _csv(items)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "test"

    def test_multiple_rows(self) -> None:
        """Multiple rows are formatted."""
        items: list[dict[str, object]] = [
            {"id": "1", "name": "test1"},
            {"id": "2", "name": "test2"},
        ]
        result = _csv(items)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2

    def test_csv_contains_header(self) -> None:
        """CSV output contains header line."""
        items: list[dict[str, object]] = [{"key": "value"}]
        result = _csv(items)
        lines = [line.strip() for line in result.strip().split("\n")]
        assert len(lines) == 2
        assert lines[0] == "key"


# Tests for _ellipsize
class TestEllipsize:
    """Tests for _ellipsize."""

    def test_short_string_unchanged(self) -> None:
        """Short string is unchanged."""
        result = _ellipsize("hello", 10)
        assert result == "hello"

    def test_long_string_truncated(self) -> None:
        """Long string is truncated with ellipsis."""
        result = _ellipsize("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_max_width_3_or_less(self) -> None:
        """max_width <= 3 is hard slice."""
        result = _ellipsize("hello", 3)
        assert result == "hel"

    def test_exact_length_no_truncation(self) -> None:
        """String of exact length is not truncated."""
        result = _ellipsize("hello", 5)
        assert result == "hello"


# Tests for _summary_line
class TestSummaryLine:
    """Tests for _summary_line."""

    def test_summary_line_format(self) -> None:
        """Summary line contains expected fields."""
        item: dict[str, object] = {
            "id": "abc123def456",
            "title": "Test Session",
            "created_at": "2026-01-15T10:00:00Z",
            "updated_at": "2026-01-15T11:00:00Z",
            "origin": "claude-code-session",
            "message_count": 42,
        }
        result = _summary_line(item)
        assert "abc123def456" in result
        assert "2026-01-15" in result
        assert "claude-code-session" in result
        assert "Test Session" in result
        assert "42" in result

    def test_summary_line_missing_title(self) -> None:
        """Missing title falls back to session_id."""
        item: dict[str, object] = {
            "id": "abc123",
            "origin": "chatgpt-export",
            "created_at": "2026-01-15T10:00:00Z",
        }
        result = _summary_line(item)
        assert "abc123" in result


# Tests for _hit_line
class TestHitLine:
    """Tests for _hit_line."""

    def test_hit_line_format(self) -> None:
        """Hit line contains expected fields."""
        item: dict[str, object] = {
            "rank": 1,
            "origin": "claude-code-session",
            "title": "Hit Title",
            "session_id": "abc123",
            "snippet": "...relevant text...",
        }
        result = _hit_line(item)
        assert "1" in result
        assert "claude-code-session" in result
        assert "Hit Title" in result
        assert "...relevant text..." in result

    def test_hit_line_missing_title_uses_session_id(self) -> None:
        """Missing title falls back to session_id."""
        item: dict[str, object] = {
            "rank": 2,
            "origin": "chatgpt-export",
            "session_id": "xyz789",
            "snippet": "...snippet...",
        }
        result = _hit_line(item)
        assert "xyz789" in result


# Tests for _stats_by_line
class TestStatsByLine:
    """Tests for _stats_by_line."""

    def test_stats_by_line_format(self) -> None:
        """Stats line contains group and count."""
        item: dict[str, object] = {"group": "claude-code", "count": 42}
        result = _stats_by_line(item)
        assert "claude-code" in result
        assert "42" in result


# Tests for _message_removed_by_transform
class TestMessageRemovedByTransform:
    """Tests for _message_removed_by_transform."""

    def test_no_transform_keeps_message(self) -> None:
        """transform=None keeps message."""
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=()),
        )
        assert _message_removed_by_transform(message, None) is False

    def test_strip_tools_removes_tool_use(self) -> None:
        """strip-tools removes message with tool_use block."""
        block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="tool_use"),
        )
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(block,)),
        )
        assert _message_removed_by_transform(message, "strip-tools") is True

    def test_strip_tools_removes_tool_result(self) -> None:
        """strip-tools removes message with tool_result block."""
        block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="tool_result"),
        )
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(block,)),
        )
        assert _message_removed_by_transform(message, "strip-tools") is True

    def test_strip_tools_keeps_other_blocks(self) -> None:
        """strip-tools keeps message without tool blocks."""
        block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="text"),
        )
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(block,)),
        )
        assert _message_removed_by_transform(message, "strip-tools") is False

    def test_strip_thinking_removes_thinking(self) -> None:
        """strip-thinking removes message with thinking block."""
        block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="thinking"),
        )
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(block,)),
        )
        assert _message_removed_by_transform(message, "strip-thinking") is True

    def test_strip_thinking_keeps_other_blocks(self) -> None:
        """strip-thinking keeps message without thinking block."""
        block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="text"),
        )
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(block,)),
        )
        assert _message_removed_by_transform(message, "strip-thinking") is False

    def test_strip_all_removes_tool_or_thinking(self) -> None:
        """strip-all removes message with tool or thinking."""
        tool_block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="tool_use"),
        )
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(tool_block,)),
        )
        assert _message_removed_by_transform(message, "strip-all") is True

        thinking_block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="thinking"),
        )
        message2 = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(thinking_block,)),
        )
        assert _message_removed_by_transform(message2, "strip-all") is True

    def test_strip_all_keeps_prose(self) -> None:
        """strip-all keeps message with only prose."""
        block = cast(
            ArchiveBlockRow,
            types.SimpleNamespace(block_type="text"),
        )
        message = cast(
            ArchiveMessageRow,
            types.SimpleNamespace(blocks=(block,)),
        )
        assert _message_removed_by_transform(message, "strip-all") is False


# Tests for _decode_cursor and _build_cursor
class TestCursorRoundtrip:
    """Tests for cursor encoding/decoding."""

    def test_decode_none_returns_none(self) -> None:
        """_decode_cursor(None) returns None."""
        result = _decode_cursor(None)
        assert result is None

    def test_decode_invalid_cursor_raises_error(self) -> None:
        """Invalid cursor token raises UsageError."""
        with pytest.raises(click.UsageError, match="invalid --cursor"):
            _decode_cursor("invalid-cursor-token")

    def test_cursor_roundtrip(self) -> None:
        """Cursor can be built and decoded."""
        summary = cast(
            ArchiveSessionSummary,
            types.SimpleNamespace(session_id="test-session-id"),
        )
        built_cursor = _build_cursor(summary, rank=10, retrieval_lane="dialogue")
        assert isinstance(built_cursor, str)

        decoded = _decode_cursor(built_cursor)
        assert decoded is not None
        assert decoded.r == 10
        assert decoded.c == "test-session-id"
        assert decoded.lane == "dialogue"


# Tests for _paginate_rows
class TestPaginateRows:
    """Tests for _paginate_rows."""

    def test_rows_within_limit(self) -> None:
        """Rows within limit return all rows and no cursor."""
        rows = [
            cast(
                ArchiveSessionSummary,
                types.SimpleNamespace(session_id=f"session-{i}"),
            )
            for i in range(5)
        ]
        page, next_cursor = _paginate_rows(rows, limit=10, offset=0)
        assert len(page) == 5
        assert next_cursor is None

    def test_rows_exceed_limit(self) -> None:
        """Rows exceeding limit return limited page and cursor."""
        rows = [
            cast(
                ArchiveSessionSummary,
                types.SimpleNamespace(session_id=f"session-{i}"),
            )
            for i in range(30)
        ]
        page, next_cursor = _paginate_rows(rows, limit=10, offset=0)
        assert len(page) == 10
        assert next_cursor is not None

    def test_empty_rows(self) -> None:
        """Empty rows return empty page and no cursor."""
        page, next_cursor = _paginate_rows([], limit=10, offset=0)
        assert len(page) == 0
        assert next_cursor is None

    def test_offset_applied_to_cursor_rank(self) -> None:
        """Offset is applied to cursor rank."""
        rows = [
            cast(
                ArchiveSessionSummary,
                types.SimpleNamespace(session_id=f"session-{i}"),
            )
            for i in range(30)
        ]
        page, next_cursor = _paginate_rows(rows, limit=10, offset=5)
        assert len(page) == 10
        assert next_cursor is not None

        decoded = _decode_cursor(next_cursor)
        assert decoded is not None
        assert decoded.r == 15  # offset + len(page) = 5 + 10


class TestEmitDeleteMachineModeNoPrompt:
    """`_emit_delete` must never block on an interactive prompt in machine mode (#1818 P6).

    The delete verb always emits a JSON MutationResultPayload. In plain mode
    (machine output, non-TTY pipe, or POLYLOGUE_FORCE_PLAIN) it must refuse a
    forceless delete with a parseable ``aborted`` envelope rather than calling
    ``env.ui.confirm`` (which would prompt on a TTY or SystemExit on a pipe).
    """

    @staticmethod
    def _delete_spec() -> OperationSpec:
        return build_runtime_operation_catalog().by_name()["mutate-delete-session"]

    @staticmethod
    def _env(*, plain: bool) -> MagicMock:
        env = MagicMock()
        env.ui.plain = plain
        env.ui.confirm = MagicMock()
        return env

    @staticmethod
    def _archive() -> MagicMock:
        archive = MagicMock()
        archive.delete_sessions = MagicMock(return_value=0)
        return archive

    def test_plain_forceless_delete_aborts_without_prompt(self, capsys: pytest.CaptureFixture[str]) -> None:
        spec = self._delete_spec()
        assert "Destructive" in spec.effects
        assert "confirmed_before_execute" in spec.safety_guards
        assert "cli" in spec.surfaces

        env = self._env(plain=True)
        archive = self._archive()

        _emit_delete(env, archive, ("s1", "s2"), params={"force": False, "dry_run": False})

        env.ui.confirm.assert_not_called()
        archive.delete_sessions.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "aborted"
        assert payload["operation"] == "delete"
        assert payload["detail"] == "confirmation_required"
        assert payload["session_count"] == 2
        assert payload["affected_count"] == 0

    def test_dry_run_evidence_lists_matched_sessions(self, capsys: pytest.CaptureFixture[str]) -> None:
        spec = self._delete_spec()
        assert "explicit_dry_run_evidence" in spec.safety_guards

        env = self._env(plain=True)
        archive = self._archive()

        _emit_delete(env, archive, ("s1", "s2"), params={"force": False, "dry_run": True})

        env.ui.confirm.assert_not_called()
        archive.delete_sessions.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "preview"
        assert payload["operation"] == "delete"
        assert payload["session_count"] == 2
        assert payload["affected_count"] == 0
        assert payload["session_ids"] == ["s1", "s2"]

    def test_plain_forced_delete_proceeds_without_prompt(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = self._env(plain=True)
        archive = self._archive()
        archive.delete_sessions.return_value = 2

        _emit_delete(env, archive, ("s1", "s2"), params={"force": True, "dry_run": False})

        env.ui.confirm.assert_not_called()
        archive.delete_sessions.assert_called_once_with(("s1", "s2"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "deleted"
        assert payload["affected_count"] == 2

    def test_interactive_forceless_delete_still_prompts(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Human interactive use (non-plain) must keep the confirmation prompt.
        env = self._env(plain=False)
        env.ui.confirm.return_value = False
        archive = self._archive()

        _emit_delete(env, archive, ("s1", "s2"), params={"force": False, "dry_run": False})

        env.ui.confirm.assert_called_once()
        archive.delete_sessions.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "aborted"
