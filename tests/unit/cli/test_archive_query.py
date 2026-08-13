"""Tests for pure helper functions in polylogue.cli.archive_query."""

from __future__ import annotations

import csv
import io
import json
import types
from http import HTTPStatus
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.cli.archive_query import (
    _build_cursor,
    _csv,
    _csv_tokens,
    _decode_cursor,
    _emit_delete,
    _emit_no_results,
    _emit_stats,
    _has_value,
    _hit_line,
    _limit,
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
    _session_summary_text,
    _session_text,
    _sort,
    _stats_by_line,
    _summary_line,
    _summary_payload,
    _tool_tokens,
    _tuple_tokens,
    execute_delete_by_session_ids,
)
from polylogue.config import Config
from polylogue.daemon_client import DaemonMutationIndeterminateError, DaemonResponseError
from polylogue.operations import OperationSpec, build_runtime_operation_catalog
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary
from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope


def test_summary_payload_renders_read_time_display_label() -> None:
    summary = ArchiveSessionSummary(
        session_id="claude-code-session:display-label",
        native_id="display-label",
        origin="claude-code-session",
        title=None,
        created_at="2026-08-06T00:00:00+00:00",
        updated_at="2026-08-06T00:00:00+00:00",
        message_count=3,
        word_count=10,
        tags=(),
        display_label="polylogue · 2 files · 3 msgs · 2026-08-06",
    )

    payload = _summary_payload(summary)

    assert payload["title"] == "polylogue · 2 files · 3 msgs · 2026-08-06"


def test_emit_no_results_includes_convergence_warning(capsys: pytest.CaptureFixture[str]) -> None:
    warning = "Archive is converging: 3 index rebuild attempt(s) active; results may be partial."

    with patch("polylogue.cli.convergence_feedback.convergence_warning_line", return_value=warning):
        with pytest.raises(SystemExit) as exc_info:
            _emit_no_results({"mode": "find"}, output_format="text")

    assert exc_info.value.code == 2
    assert capsys.readouterr().out.splitlines()[:2] == [warning, "No sessions matched."]


def test_emit_no_results_json_includes_convergence_warning(capsys: pytest.CaptureFixture[str]) -> None:
    warning = "Archive is converging: 3 index rebuild attempt(s) active; results may be partial."

    with patch("polylogue.cli.convergence_feedback.convergence_warning_line", return_value=warning):
        with pytest.raises(SystemExit) as exc_info:
            _emit_no_results({"mode": "find"}, output_format="json")

    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["archive_converging"] is True
    assert payload["convergence_warning"] == warning
    assert payload["items"] == []
    assert payload["total"] == 0


def test_emit_stats_includes_convergence_warning(capsys: pytest.CaptureFixture[str]) -> None:
    from polylogue.archive.stats import ArchiveStats

    warning = "Archive is converging: 3 index rebuild attempt(s) active; results may be partial."
    stats = ArchiveStats(
        total_sessions=3,
        total_messages=12,
        total_attachments=0,
        origins={"codex-session": 3},
    )

    with patch("polylogue.cli.convergence_feedback.convergence_warning_line", return_value=warning):
        _emit_stats(stats, output_format="plaintext", origin=None, query="", fields=None)

    assert capsys.readouterr().out.splitlines()[:2] == [warning, "Sessions: 3"]


def test_emit_stats_json_includes_convergence_warning(capsys: pytest.CaptureFixture[str]) -> None:
    from polylogue.archive.stats import ArchiveStats

    warning = "Archive is converging: 3 index rebuild attempt(s) active; results may be partial."
    stats = ArchiveStats(
        total_sessions=3,
        total_messages=12,
        total_attachments=0,
        origins={"codex-session": 3},
    )

    with patch("polylogue.cli.convergence_feedback.convergence_warning_line", return_value=warning):
        _emit_stats(stats, output_format="json", origin=None, query="", fields=None)

    payload = json.loads(capsys.readouterr().out)
    assert payload["archive_converging"] is True
    assert payload["convergence_warning"] == warning
    assert payload["total_sessions"] == 3


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


# Tests for _resolve_excluded_origins
class TestResolveExcludedOrigins:
    """Tests for _resolve_excluded_origins."""

    def test_explicit_excluded_origin(self) -> None:
        """Explicit excluded origins are parsed."""
        params: dict[str, object] = {"exclude_origin": "claude-code-session,chatgpt-export"}
        result = _resolve_excluded_origins(params)
        assert result == ("claude-code-session", "chatgpt-export")

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


# _ellipsize was deleted as dead code (polylogue-x7d): archive_query.py's
# row-truncation duplicate (_snippet/_ellipsize) was replaced by the shared
# polylogue.archive.query.search_hits.bound_display_text primitive, which
# carries the equivalent behavior tests in tests/unit/archive/test_search_hits.py.


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

    def test_summary_line_bounds_multiline_title(self) -> None:
        item: dict[str, object] = {
            "id": "abc123",
            "title": "needle\n" + "\n".join(f"/tmp/hermes-agent/path-{index}.py" for index in range(50)),
            "origin": "claude-code-session",
            "created_at": "2026-01-15T10:00:00Z",
        }

        result = _summary_line(item)

        assert "\n" not in result
        assert "/tmp/hermes-agent/path-20.py" not in result
        assert "..." in result


# Tests for _hit_line
class TestHitLine:
    """Tests for _hit_line."""

    def test_hit_line_format(self) -> None:
        """Hit line contains expected fields."""
        item: dict[str, object] = {
            "session": {
                "id": "abc123",
                "origin": "claude-code-session",
                "title": "Hit Title",
            },
            "match": {
                "rank": 1,
                "snippet": "...relevant text...",
            },
        }
        result = _hit_line(item)
        assert "1" in result
        assert "claude-code-session" in result
        assert "Hit Title" in result
        assert "...relevant text..." in result

    def test_hit_line_missing_title_uses_session_id(self) -> None:
        """Missing title falls back to session_id."""
        item: dict[str, object] = {
            "session": {
                "id": "xyz789",
                "origin": "chatgpt-export",
            },
            "match": {
                "rank": 2,
                "snippet": "...snippet...",
            },
        }
        result = _hit_line(item)
        assert "xyz789" in result

    def test_hit_line_bounds_multiline_title_and_giant_snippet(self) -> None:
        item: dict[str, object] = {
            "session": {
                "id": "abc123",
                "origin": "claude-code-session",
                "title": "needle\n" + "\n".join(f"/tmp/hermes-agent/path-{index}.py" for index in range(50)),
            },
            "match": {
                "rank": 1,
                "snippet": "hermes " + ("full transcript payload " * 100),
            },
        }

        result = _hit_line(item)

        assert "\n" not in result
        assert "/tmp/hermes-agent/path-20.py" not in result
        assert "full transcript payload " * 20 not in result
        assert "..." in result


# Tests for _stats_by_line
class TestStatsByLine:
    """Tests for _stats_by_line."""

    def test_stats_by_line_format(self) -> None:
        """Stats line contains group and count."""
        item: dict[str, object] = {"group": "claude-code", "count": 42}
        result = _stats_by_line(item)
        assert "claude-code" in result
        assert "42" in result


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

        _emit_delete(env, ("s1", "s2"), params={"force": False, "dry_run": False})

        env.ui.confirm.assert_not_called()
        archive.delete_sessions.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "aborted"
        assert payload["operation"] == "delete"
        assert payload["detail"] == "confirmation_required"
        assert payload["session_count"] == 2
        assert payload["affected_count"] == 0

    @pytest.mark.parametrize(
        "acknowledgement",
        [
            {"status": "prepared", "preview_ref": "preview:delete"},
            {"status": "cancelled", "preview_ref": "preview:other"},
            {"status": "cancelled"},
        ],
    )
    def test_interactive_delete_refuses_unbound_cancellation_acknowledgement(
        self, acknowledgement: dict[str, object]
    ) -> None:
        env = self._env(plain=False)
        env.ui.confirm.return_value = False

        with (
            patch(
                "polylogue.cli.archive_query._submit_daemon_mutation",
                side_effect=[
                    {"status": "prepared", "preview_ref": "preview:delete", "session_ids": ["s1"]},
                    acknowledgement,
                ],
            ),
            pytest.raises(click.ClickException, match="invalid delete cancellation acknowledgement"),
        ):
            _emit_delete(env, ("s1",), params={"force": False, "dry_run": False})

    def test_dry_run_evidence_lists_matched_sessions(self, capsys: pytest.CaptureFixture[str]) -> None:
        spec = self._delete_spec()
        assert "explicit_dry_run_evidence" in spec.safety_guards

        env = self._env(plain=True)
        archive = self._archive()

        _emit_delete(env, ("s1", "s2"), params={"force": False, "dry_run": True})

        env.ui.confirm.assert_not_called()
        archive.delete_sessions.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "preview"
        assert payload["operation"] == "delete"
        assert payload["session_count"] == 2
        assert payload["affected_count"] == 0
        assert payload["session_ids"] == ["s1", "s2"]

    def test_plain_forced_delete_routes_through_daemon_without_prompt(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = self._env(plain=True)
        archive = self._archive()

        with patch(
            "polylogue.cli.archive_query._submit_daemon_mutation",
            side_effect=[
                {"status": "prepared", "preview_ref": "preview:delete", "session_ids": ["s1", "s2"]},
                {"status": "authorized", "authorization_token": "daemon-token"},
                {"status": "deleted", "affected_count": 2},
            ],
        ) as daemon_delete:
            _emit_delete(env, ("s1", "s2"), params={"force": True, "dry_run": False})

        env.ui.confirm.assert_not_called()
        archive.delete_sessions.assert_not_called()
        assert [call.args[1] for call in daemon_delete.call_args_list] == [
            "/api/cli/delete/prepare",
            "/api/cli/delete/authorize",
            "/api/cli/delete",
        ]
        assert [call.kwargs["body"] for call in daemon_delete.call_args_list] == [
            {"session_ids": ["s1", "s2"]},
            {"preview_ref": "preview:delete"},
            {"authorization_token": "daemon-token"},
        ]
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "deleted"
        assert payload["affected_count"] == 2

    def test_resolved_batch_releases_selection_snapshot_before_daemon_write(self) -> None:
        """Known IDs need no SQLite reader while the daemon owns the delete."""

        env = self._env(plain=True)
        with (
            patch(
                "polylogue.cli.archive_query.archive_read_context",
                side_effect=AssertionError("must not pin a WAL reader"),
            ),
            patch("polylogue.cli.archive_query._emit_delete") as emit_delete,
        ):
            execute_delete_by_session_ids(env, ["s1", "s2"], force=True)

        emit_delete.assert_called_once_with(
            env,
            ("s1", "s2"),
            params={"force": True, "delete_matched": True, "dry_run": False},
        )

    @pytest.mark.parametrize(
        "daemon_error",
        [
            pytest.param(
                DaemonMutationIndeterminateError(method="POST", path="/api/cli/delete"),
                id="slow-response-is-indeterminate",
            ),
            pytest.param(
                DaemonResponseError(
                    status=HTTPStatus.BAD_REQUEST,
                    code="invalid_request",
                    detail="invalid session_ids",
                ),
                id="http-refusal-is-not-offline-absence",
            ),
        ],
    )
    def test_confirmed_delete_never_falls_back_after_daemon_error(
        self,
        capsys: pytest.CaptureFixture[str],
        daemon_error: Exception,
    ) -> None:
        from polylogue.operations.durable_change_train import acquire_durable_archive_ownership

        env = self._env(plain=True)
        archive = self._archive()

        with (
            patch("polylogue.cli.archive_query._submit_daemon_mutation", side_effect=daemon_error),
            patch(
                "polylogue.operations.durable_change_train.acquire_durable_archive_ownership",
                wraps=acquire_durable_archive_ownership,
            ) as offline_ownership,
            pytest.raises(click.ClickException),
        ):
            _emit_delete(env, ("s1", "s2"), params={"force": True, "dry_run": False})

        offline_ownership.assert_not_called()
        archive.delete_sessions.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_confirmed_delete_uses_an_unbounded_daemon_wait(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        import polylogue.cli.archive_query as archive_query

        initialized: list[dict[str, object]] = []

        class Client:
            last_elapsed_ms = 250

            def __init__(self, _socket_path: Path, **kwargs: object) -> None:
                initialized.append(kwargs)

            def probe(self, **_kwargs: object) -> dict[str, object]:
                return {"ok": True}

            def request_mutation_json(self, method: str, path: str, body: dict[str, object]) -> dict[str, object]:
                assert (method, path, body) == ("POST", "/api/cli/delete/prepare", {"session_ids": ["s1"]})
                return {"status": "prepared", "preview_ref": "preview:delete", "session_ids": ["s1"]}

        config = cast(
            "Config",
            SimpleNamespace(
                archive_root=tmp_path,
                db_path=tmp_path / "index.db",
                api_auth_token=None,
                api_allow_no_auth=True,
            ),
        )
        monkeypatch.setattr(archive_query, "_daemon_disabled", lambda **_kwargs: False)
        monkeypatch.setattr("polylogue.cli.daemon_client.DaemonClient", Client)
        monkeypatch.setattr("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: tmp_path / "daemon.sock")
        monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: None)

        payload = archive_query._submit_daemon_mutation(config, "/api/cli/delete/prepare", body={"session_ids": ["s1"]})

        assert initialized == [{"timeout_s": None, "auth_token": None}]
        assert payload == {
            "status": "prepared",
            "preview_ref": "preview:delete",
            "session_ids": ["s1"],
            "_daemon_elapsed_ms": 250,
        }

    def test_confirmed_delete_routes_to_explicit_split_root_daemon(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        import polylogue.cli.archive_query as archive_query

        configured_root = tmp_path / "configured"
        selected_root = tmp_path / "selected"
        configured_root.mkdir()
        selected_root.mkdir()
        initialized: list[Path] = []
        probed: list[dict[str, object]] = []

        class Client:
            last_elapsed_ms = None

            def __init__(self, socket_path: Path, **_kwargs: object) -> None:
                initialized.append(socket_path)

            def probe(self, **kwargs: object) -> dict[str, object]:
                probed.append(kwargs)
                return {"ok": True}

            def request_mutation_json(self, _method: str, _path: str, _body: dict[str, object]) -> dict[str, object]:
                return {"status": "prepared"}

        config = cast(
            "Config",
            SimpleNamespace(
                archive_root=configured_root,
                db_path=selected_root / "index.db",
                api_auth_token=None,
                api_allow_no_auth=True,
            ),
        )
        monkeypatch.setattr(archive_query, "_daemon_disabled", lambda **_kwargs: False)
        monkeypatch.setattr("polylogue.cli.daemon_client.DaemonClient", Client)
        monkeypatch.setattr("polylogue.daemon.socket_path.daemon_socket_path", lambda root: root / "daemon.sock")
        monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: None)

        assert archive_query._submit_daemon_mutation(config, "/api/cli/delete/prepare", body={}) == {
            "status": "prepared"
        }
        assert initialized == [selected_root / "daemon.sock"]
        assert probed[0]["archive_root"] == str(selected_root)

    def test_interactive_forceless_delete_still_prompts(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Human interactive use (non-plain) must keep the confirmation prompt.
        env = self._env(plain=False)
        env.ui.confirm.return_value = False
        archive = self._archive()

        with patch(
            "polylogue.cli.archive_query._submit_daemon_mutation",
            side_effect=[
                {"status": "prepared", "preview_ref": "preview:delete", "session_ids": ["s1", "s2"]},
                {"status": "cancelled", "preview_ref": "preview:delete"},
            ],
        ) as daemon_delete:
            _emit_delete(env, ("s1", "s2"), params={"force": False, "dry_run": False})

        env.ui.confirm.assert_called_once()
        archive.delete_sessions.assert_not_called()
        assert [call.args[1] for call in daemon_delete.call_args_list] == [
            "/api/cli/delete/prepare",
            "/api/cli/delete/cancel",
        ]
        assert daemon_delete.call_args_list[-1].kwargs["body"] == {"preview_ref": "preview:delete"}
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "aborted"


class TestSessionSummaryText:
    """``read --view summary`` must render a condensed synopsis, not the full
    transcript (#analyze-perf): previously ``summary`` and ``transcript``
    both routed to the same renderer and produced byte-identical output for
    any session.
    """

    @staticmethod
    def _envelope() -> ArchiveSessionEnvelope:
        messages = (
            ArchiveMessageRow(
                message_id="s1:1",
                native_id="1",
                role="user",
                position=0,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=(ArchiveBlockRow(block_id="s1:1:0", message_id="s1:1", block_type="text", text="hello there"),),
                word_count=2,
            ),
            ArchiveMessageRow(
                message_id="s1:2",
                native_id="2",
                role="assistant",
                position=1,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=False,
                blocks=(
                    ArchiveBlockRow(block_id="s1:2:0", message_id="s1:2", block_type="text", text="using a tool now"),
                ),
                word_count=4,
                has_tool_use=True,
            ),
            ArchiveMessageRow(
                message_id="s1:3",
                native_id="3",
                role="assistant",
                position=2,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=True,
                blocks=(ArchiveBlockRow(block_id="s1:3:0", message_id="s1:3", block_type="text", text="all done"),),
                word_count=2,
            ),
        )
        return ArchiveSessionEnvelope(
            session_id="claude-code-session:abc",
            native_id="abc",
            origin="claude-code-session",
            title="Fix the thing",
            active_leaf_message_id="s1:3",
            messages=messages,
        )

    def test_summary_differs_from_transcript(self) -> None:
        envelope = self._envelope()
        summary = _session_summary_text(envelope)
        transcript = _session_text(envelope)
        assert summary != transcript
        # The summary is a synopsis (counts + first/last excerpt), not a
        # per-message transcript dump -- it must not contain every message's
        # role header the way the transcript does.
        assert transcript.count("## user") + transcript.count("## assistant") == 3
        assert "## user\n" not in summary
        assert "## assistant\n" not in summary

    def test_summary_reports_counts_and_first_last_turn(self) -> None:
        envelope = self._envelope()
        summary = _session_summary_text(envelope)
        assert "messages: 3" in summary
        assert "user: 1" in summary
        assert "assistant: 2" in summary
        assert "messages with tool use: 1" in summary
        assert "hello there" in summary  # first turn excerpt
        assert "all done" in summary  # last turn excerpt

    def test_transcript_still_renders_every_message(self) -> None:
        envelope = self._envelope()
        transcript = _session_text(envelope)
        assert "hello there" in transcript
        assert "using a tool now" in transcript
        assert "all done" in transcript
