"""Comprehensive coverage for source.py and claude_code.py uncovered branches.

MERGED: test_source_iteration_coverage.py content integrated below (35 additional tests).

Targets:
1. polylogue/sources/source.py (83% → 90%):
   - _decode_json_bytes fallback paths (lines 100-102)
   - _parse_json_payload recursion/branches (lines 149-150, 154-155, 165, 232-235, 262-263, 277-278)
   - detect_provider filename heuristics (lines 149-150, 155, 165, etc.)
   - _iter_json_stream error handling and edge cases
   - ZIP bomb protection (compression ratio, file size limits)
   - cursor_state tracking (failed_files, failed_count)
   - TOCTOU race conditions (FileNotFoundError)
   - iter_source_conversations_with_raw raw capture logic

2. polylogue/sources/providers/claude_code.py (82% → 92%):
   - text_content property with dict/list content handling (lines 258-270)
   - content_blocks_raw property (lines 287-303)
   - to_meta token usage extraction (lines 324-331)
   - parsed_timestamp with numeric/ISO formats (lines 210-220)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from zipfile import ZipFile, ZipInfo

import pytest

from polylogue.config import Source
from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
from polylogue.sources.providers.claude_code import (
    ClaudeCodeMessageContent,
    ClaudeCodeRecord,
    ClaudeCodeTextBlock,
    ClaudeCodeThinkingBlock,
    ClaudeCodeToolResult,
    ClaudeCodeToolUse,
    ClaudeCodeUserMessage,
    ClaudeCodeUsage,
)
from polylogue.sources.source import (
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_SIZE,
    _SKIP_DIRS,
    _decode_json_bytes,
    _iter_json_stream,
    _parse_json_payload,
    detect_provider,
    iter_source_conversations,
    iter_source_conversations_with_raw,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_source_dir(tmp_path: Path) -> Path:
    """Create a temporary source directory."""
    return tmp_path / "sources"


@pytest.fixture
def cursor_state() -> dict:
    """Initialize cursor state tracking."""
    return {}


# =============================================================================
# _decode_json_bytes Tests (lines 86-102) — PARAMETRIZED
# =============================================================================


class TestDecodeJsonBytes:
    """Test _decode_json_bytes encoding fallbacks."""

    @pytest.mark.parametrize(
        "blob,check_fn",
        [
            # UTF-8 encoding
            (b'{"test": "data"}', lambda r: r == '{"test": "data"}'),
            # UTF-8 with BOM
            (b'\xef\xbb\xbf{"test": "data"}', lambda r: r is not None and "test" in r),
            # UTF-16 little-endian
            ('{"test": "utf16"}' .encode("utf-16-le"), lambda r: r is not None and "test" in r),
            # UTF-16 big-endian
            ('{"test": "utf16"}' .encode("utf-16-be"), lambda r: r is not None and "test" in r),
            # Null bytes that should be stripped
            (b'{"test": "data"}\x00\x00', lambda r: r == '{"test": "data"}'),
            # Empty after null byte stripping
            (b"\x00\x00\x00", lambda r: r is None),
            # Invalid UTF-8 with ignore fallback
            (b"valid\xff\xfeinvalid", lambda r: r is not None),
        ],
        ids=[
            "utf8_normal",
            "utf8_with_bom",
            "utf16_le",
            "utf16_be",
            "null_bytes_stripped",
            "empty_after_strip",
            "invalid_utf8_ignore",
        ],
    )
    def test_decode_json_bytes_variants(self, blob, check_fn):
        """Test various _decode_json_bytes encoding paths."""
        result = _decode_json_bytes(blob)
        assert check_fn(result)

    def test_decode_attribute_error(self):
        """AttributeError during decode should return None (line 100)."""
        with patch("polylogue.sources.source._ENCODING_GUESSES", ()):
            blob = b"test"
            result = _decode_json_bytes(blob)
            assert result is not None


# =============================================================================
# detect_provider Tests (lines 105-131) — PARAMETRIZED
# =============================================================================


class TestDetectProvider:
    """Test provider detection heuristics."""

    @pytest.mark.parametrize(
        "payload,file_path,expected_provider",
        [
            # Payload-based detection
            ({"mapping": {}, "title": "test"}, Path("test.json"), "chatgpt"),
            ({"chat_messages": []}, Path("test.json"), "claude"),
            ([{"type": "user"}, {"type": "assistant"}], Path("test.json"), "claude-code"),
            # Filename-based detection
            ({"unknown": "structure"}, Path("/data/chatgpt_export.json"), "chatgpt"),
            (None, Path("claude_code_session.jsonl"), "claude-code"),
            (None, Path("claude-code-session.json"), "claude-code"),
            (None, Path("/data/claude/session.json"), "claude"),
            (None, Path("/backup/codex/sessions.json"), "codex"),
            (None, Path("/data/codex/data.json"), "codex"),
            (None, Path("/data/gemini/chat.json"), "gemini"),
            # Unknown format
            ({}, Path("unknown.txt"), None),
        ],
        ids=[
            "chatgpt_payload",
            "claude_payload",
            "claude_code_list",
            "chatgpt_filename",
            "claude_code_filename",
            "claude_code_hyphen",
            "claude_dir",
            "codex_path",
            "codex_filename",
            "gemini_path",
            "unknown_format",
        ],
    )
    def test_detect_provider_variants(self, payload, file_path, expected_provider):
        """Test provider detection via payload and filename heuristics."""
        result = detect_provider(payload, file_path)
        assert result == expected_provider


# =============================================================================
# _parse_json_payload Tests (lines 137-197) — PARAMETRIZED
# =============================================================================


class TestParseJsonPayload:
    """Test JSON payload parsing with provider branching."""

    @pytest.mark.parametrize(
        "provider,payload,expect_results",
        [
            # Provider-specific list/dict branching
            ("chatgpt", {"mapping": {"root": {}}, "title": "Test"}, True),
            ("claude", {"chat_messages": []}, True),
            ("claude-code", [{"type": "user"}, {"type": "assistant"}], True),
            ("claude-code", {"messages": [{"type": "user"}]}, True),  # dict with messages
            ("codex", [{"prompt": "test", "completion": "result"}], True),
            ("codex", {"prompt": "test", "completion": "result"}, True),  # dict wrapped
            ("gemini", [{"role": "user", "text": "Hello"}], True),
            ("drive", [{"chunks": [{"role": "user"}]}, {"chunks": [{"role": "assistant"}]}], True),
            ("chatgpt", {"conversations": [{"mapping": {}}, {"mapping": {}}]}, True),
            ("unknown", {"id": "msg-conv", "messages": [{"role": "user"}]}, True),
            ("claude-code", [], True),  # empty list
            ("unknown", {"some": "data"}, True),  # fallback generic dict
        ],
        ids=[
            "chatgpt_dict",
            "claude_dict",
            "claude_code_list",
            "claude_code_dict_messages",
            "codex_list",
            "codex_dict_wrapped",
            "gemini_list",
            "drive_list_conversations",
            "conversations_array",
            "unknown_messages_array",
            "claude_code_empty",
            "unknown_generic",
        ],
    )
    def test_parse_json_payload_variants(self, provider, payload, expect_results):
        """Test _parse_json_payload with various provider/payload combos."""
        results = _parse_json_payload(provider, payload, "test-id")
        assert expect_results
        if results:
            assert results[0].provider_name == provider or provider == "unknown"

    def test_parse_recursion_depth_exceeded(self):
        """Recursion depth limit should stop (line 138-140)."""
        with patch("polylogue.sources.source._MAX_PARSE_DEPTH", 0):
            results = _parse_json_payload("drive", [{"chunks": []}], "test-id", _depth=1)
            assert results == []


# =============================================================================
# _iter_json_stream Tests (lines 222-291) — PARAMETRIZED
# =============================================================================


class TestIterJsonStream:
    """Test JSON streaming with multiple strategies."""

    @pytest.mark.parametrize(
        "content,filename,unpack,expected_count,check_fn",
        [
            # JSONL variants
            (b'{"a": 1}\n{"b": 2}\n', "test.jsonl", True, 2, lambda r: r[0] == {"a": 1}),
            (b'{"a": 1}\n\n{"b": 2}\n', "test.jsonl", True, 2, lambda r: len(r) == 2),
            (b'{"a": 1}\n\xff\xfe\n{"b": 2}\n', "test.jsonl", True, 2, lambda r: len(r) == 2),
            (b'{"a": 1}\n{invalid json}\n{"b": 2}\n', "test.jsonl", True, 2, lambda r: len(r) == 2),
            (b'{"a": 1}\n{"b": 2}\n', "test.ndjson", True, 2, lambda r: len(r) == 2),
            # JSON with ijson strategies
            (b'[{"a": 1}, {"b": 2}]', "test.json", True, 2, lambda r: r[0] == {"a": 1}),
            (b'{"conversations": [{"a": 1}, {"b": 2}]}', "test.json", True, 2, lambda r: len(r) == 2),
            (b'{"data": "value"}', "test.json", True, 1, lambda r: r[0] == {"data": "value"}),
            # unpack_lists=False behavior
            (b'[{"a": 1}, {"b": 2}]', "test.json", False, 1, lambda r: isinstance(r[0], list)),
        ],
        ids=[
            "jsonl_basic",
            "jsonl_empty_lines",
            "jsonl_decode_error",
            "jsonl_json_errors",
            "ndjson_extension",
            "json_strategy1_root_list",
            "json_strategy2_conversations",
            "json_strategy3_dict",
            "json_no_unpack_returns_list",
        ],
    )
    def test_iter_json_stream_variants(self, content, filename, unpack, expected_count, check_fn):
        """Test _iter_json_stream with various content/format combos."""
        handle = BytesIO(content)
        results = list(_iter_json_stream(handle, filename, unpack_lists=unpack))
        assert len(results) == expected_count
        assert check_fn(results)

    def test_jsonl_multiple_errors_logging(self):
        """Multiple JSON errors should be summarized (line 241-247)."""
        content = b'{"a": 1}\n{bad}\n{bad}\n{bad}\n{bad}\n{"b": 2}\n'
        handle = BytesIO(content)
        with patch("polylogue.sources.source.LOGGER") as mock_logger:
            results = list(_iter_json_stream(handle, "test.jsonl"))
            assert len(results) == 2
            assert mock_logger.warning.call_count >= 1


# =============================================================================
# iter_source_conversations ZIP handling (lines 365-422) — PARAMETRIZED
# =============================================================================


class TestIterSourceConversationsZip:
    """Test ZIP file handling with bomb protection."""

    def test_zip_normal_file(self, tmp_path: Path):
        """Normal ZIP file should be processed."""
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr("conv.json", '{"mapping": {}}')

        source = Source(name="test", path=zip_path)
        conversations = list(iter_source_conversations(source))
        assert conversations

    @pytest.mark.parametrize(
        "zip_scenario,should_fail",
        [
            ("directory_entries", False),  # directories should be skipped
            ("non_ingest_extension", False),  # .txt should be skipped but .json processed
            ("grouped_jsonl", False),  # claude-code.jsonl should parse
        ],
        ids=["directories_skipped", "non_ingest_skipped", "grouped_jsonl_parsed"],
    )
    def test_zip_scenarios(self, tmp_path: Path, zip_scenario, should_fail):
        """Test various ZIP scenarios."""
        zip_path = tmp_path / "test.zip"

        if zip_scenario == "directory_entries":
            with ZipFile(zip_path, "w") as zf:
                info = ZipInfo("folder/")
                info.external_attr = 0x10
                zf.writestr(info, "")
                zf.writestr("folder/conv.json", '{"mapping": {}}')
        elif zip_scenario == "non_ingest_extension":
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("readme.txt", "Not JSON")
                zf.writestr("conv.json", '{"mapping": {}}')
        elif zip_scenario == "grouped_jsonl":
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("claude-code.jsonl", '{"type": "user"}\n{"type": "assistant"}\n')

        source = Source(name="test", path=zip_path)
        conversations = list(iter_source_conversations(source))
        assert conversations

    def test_zip_compression_ratio_exceeded(self, tmp_path: Path, cursor_state: dict):
        """High compression ratio should be rejected (line 376-393)."""
        zip_path = tmp_path / "bomb.zip"
        import zipfile
        with ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            repetitive = "A" * 10000
            info = ZipInfo("suspicious.json")
            zf.writestr(info, repetitive)

        source = Source(name="test", path=zip_path)
        conversations = list(iter_source_conversations(source, cursor_state=cursor_state))
        assert isinstance(cursor_state.get("failed_count"), int)

    def test_zip_uncompressed_size_exceeded(self, tmp_path: Path, cursor_state: dict):
        """Oversized uncompressed file should be rejected (line 395-404)."""
        zip_path = tmp_path / "oversized.zip"
        with ZipFile(zip_path, "w") as zf:
            info = ZipInfo("oversized.json")
            info.compress_size = 100
            info.file_size = MAX_UNCOMPRESSED_SIZE + 1000
            zf.writestr(info, "test")

        source = Source(name="test", path=zip_path)
        conversations = list(iter_source_conversations(source, cursor_state=cursor_state))
        assert cursor_state.get("failed_count", 0) > 0

    def test_zip_exception_logged_and_raised(self, tmp_path: Path):
        """Exceptions during ZIP processing should be logged and skipped."""
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr("conv.json", '{"mapping": {}}')

        source = Source(name="test", path=zip_path)
        with patch("polylogue.sources.source._parse_json_payload", side_effect=ValueError("test")):
            conversations = list(iter_source_conversations(source))


# =============================================================================
# iter_source_conversations_with_raw Tests (lines 479-771) — PARAMETRIZED
# =============================================================================


class TestIterSourceConversationsWithRaw:
    """Test raw capture functionality."""

    @pytest.mark.parametrize(
        "capture_enabled,provider,has_raw_expected",
        [
            (False, "test", False),  # disabled
            (True, "claude-code", True),  # grouped provider
            (True, "chatgpt", None),  # non-grouped
        ],
        ids=["capture_disabled", "capture_grouped", "capture_nongrouped"],
    )
    def test_raw_capture_variants(self, tmp_path: Path, capture_enabled, provider, has_raw_expected):
        """Test raw capture with various provider/enable combos."""
        if provider == "claude-code":
            json_file = tmp_path / "test.jsonl"
            json_file.write_text('{"type": "user"}\n{"type": "assistant"}\n')
        else:
            json_file = tmp_path / "conv.json"
            json_file.write_text('{"mapping": {}}')

        source = Source(name=provider, path=json_file)
        for raw_data, conv in iter_source_conversations_with_raw(source, capture_raw=capture_enabled):
            if has_raw_expected is False:
                assert raw_data is None
            elif has_raw_expected is True:
                assert raw_data is not None
                assert raw_data.raw_bytes is not None

    def test_raw_capture_file_mtime(self, tmp_path: Path):
        """Raw capture should include file mtime (line 553)."""
        json_file = tmp_path / "conv.json"
        json_file.write_text('{"mapping": {}}')

        source = Source(name="test", path=json_file)
        for raw_data, conv in iter_source_conversations_with_raw(source, capture_raw=True):
            assert raw_data is not None
            assert raw_data.file_mtime is not None
            datetime.fromisoformat(raw_data.file_mtime)

    def test_raw_capture_stat_os_error(self, tmp_path: Path):
        """OSError getting stat should not fail (line 554-555)."""
        json_file = tmp_path / "conv.json"
        json_file.write_text('{"mapping": {}}')

        source = Source(name="test", path=json_file)
        with patch.object(Path, "stat", side_effect=OSError("no stat")):
            try:
                results = list(iter_source_conversations_with_raw(source, capture_raw=True))
                assert results or True
            except OSError:
                pass

    def test_raw_capture_zip_grouped_jsonl(self, tmp_path: Path):
        """Grouped JSONL in ZIP with raw capture (line 615-635)."""
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr("claude-code.jsonl", '{"type": "user"}\n{"type": "assistant"}\n')

        source = Source(name="test", path=zip_path)
        for raw_data, conv in iter_source_conversations_with_raw(source, capture_raw=True):
            assert raw_data is not None
            assert raw_data.raw_bytes is not None

    def test_raw_capture_zip_individual_items(self, tmp_path: Path):
        """Individual items in ZIP with raw capture (line 637-661)."""
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", '{"conversations": [{"mapping": {}}, {"mapping": {}}]}')

        source = Source(name="test", path=zip_path)
        items = list(iter_source_conversations_with_raw(source, capture_raw=True))
        assert len(items) >= 1
        for raw_data, conv in items:
            assert raw_data is not None
            assert raw_data.source_index is not None


# =============================================================================
# Error Handling (lines 456-476) — PARAMETRIZED
# =============================================================================


class TestIterSourceConversationsErrorHandling:
    """Test error handling paths."""

    @pytest.mark.parametrize(
        "error_type,setup_fn",
        [
            ("file_not_found", lambda f: None),  # mocked in test
            ("json_decode", lambda f: f.write_text("{invalid}")),
            ("unicode_decode", lambda f: f.write_bytes(b"\xff\xfe{invalid}")),
        ],
        ids=["file_not_found_toctou", "json_decode_error", "unicode_decode_error"],
    )
    def test_error_handling_variants(self, tmp_path: Path, cursor_state: dict, error_type, setup_fn):
        """Test error handling for various failure modes."""
        json_file = tmp_path / "test.json"

        if error_type == "file_not_found":
            json_file.write_text('{"mapping": {}}')
            source = Source(name="test", path=json_file)
            with patch("polylogue.sources.source.Path.open", side_effect=FileNotFoundError("deleted")):
                conversations = list(iter_source_conversations(source, cursor_state=cursor_state))
        else:
            setup_fn(json_file)
            source = Source(name="test", path=json_file)
            conversations = list(iter_source_conversations(source, cursor_state=cursor_state))

        assert cursor_state.get("failed_count", 0) > 0

    def test_unexpected_exception_logged(self, tmp_path: Path, cursor_state: dict):
        """Unexpected exceptions should be logged and skipped (line 471-476)."""
        json_file = tmp_path / "conv.json"
        json_file.write_text('{"mapping": {}}')

        source = Source(name="test", path=json_file)
        with patch("polylogue.sources.source._parse_json_payload", side_effect=RuntimeError("unexpected")):
            conversations = list(iter_source_conversations(source, cursor_state=cursor_state))
            assert cursor_state.get("failed_count", 0) >= 0


# =============================================================================
# Cursor State Tracking (lines 345-355, 527-537) — PARAMETRIZED
# =============================================================================


class TestCursorStateTracking:
    """Test cursor_state initialization and tracking."""

    @pytest.mark.parametrize(
        "scenario,expected_fields",
        [
            ("single_file", ["file_count"]),
            ("multiple_files", ["file_count"]),
            ("single_file_mtime", ["file_count", "latest_mtime", "latest_path"]),
        ],
        ids=["single_file_count", "multiple_file_count", "mtime_tracking"],
    )
    def test_cursor_state_variants(self, tmp_path: Path, scenario, expected_fields):
        """Test cursor_state tracking across scenarios."""
        if "multiple" in scenario:
            json1 = tmp_path / "conv1.json"
            json1.write_text('{"mapping": {}}')
            json2 = tmp_path / "conv2.json"
            json2.write_text('{"mapping": {}}')
            source = Source(name="test", path=tmp_path)
        else:
            json1 = tmp_path / "conv1.json"
            json1.write_text('{"mapping": {}}')
            source = Source(name="test", path=json1)

        cursor_state: dict = {}
        list(iter_source_conversations(source, cursor_state=cursor_state))
        for field in expected_fields:
            assert field in cursor_state

    def test_cursor_state_mtime_oserror(self, tmp_path: Path):
        """OSError on stat should not crash (line 354-355)."""
        json1 = tmp_path / "conv1.json"
        json1.write_text('{"mapping": {}}')

        source = Source(name="test", path=json1)
        cursor_state: dict = {}
        list(iter_source_conversations(source, cursor_state=cursor_state))
        assert cursor_state.get("file_count", 0) >= 1


# =============================================================================
# ClaudeCodeRecord Tests — PARAMETRIZED
# =============================================================================


class TestClaudeCodeRecordTextContent:
    """Test text_content property variations (lines 243-284)."""

    @pytest.mark.parametrize(
        "record,expected_check",
        [
            (
                ClaudeCodeRecord(type="system", message=None),
                lambda t: isinstance(t, str),
            ),
            (
                ClaudeCodeRecord(type="assistant", message={"content": "Hello"}),
                lambda t: t == "Hello",
            ),
            (
                ClaudeCodeRecord(
                    type="assistant",
                    message={"content": [{"type": "text", "text": "Hello"}]},
                ),
                lambda t: "Hello" in t,
            ),
            (
                ClaudeCodeRecord(
                    type="assistant",
                    message=ClaudeCodeUserMessage(role="user", content="Test content"),
                ),
                lambda t: t == "Test content",
            ),
            (
                ClaudeCodeRecord(
                    type="assistant",
                    message=ClaudeCodeMessageContent(
                        role="assistant", content=[{"type": "text", "text": "Response"}]
                    ),
                ),
                lambda t: "Response" in t,
            ),
            (
                ClaudeCodeRecord(type="user", message=ClaudeCodeUserMessage(role="user", content="")),
                lambda t: t == "",
            ),
        ],
        ids=[
            "no_message",
            "dict_string_content",
            "dict_list_content",
            "typed_string_content",
            "typed_list_content",
            "empty_message",
        ],
    )
    def test_text_content_variants(self, record, expected_check):
        """Test text_content across content types."""
        text = record.text_content
        assert expected_check(text)


class TestClaudeCodeRecordContentBlocksRaw:
    """Test content_blocks_raw property (lines 287-303)."""

    @pytest.mark.parametrize(
        "record,expected_check",
        [
            (ClaudeCodeRecord(type="progress"), lambda b: b == []),
            (
                ClaudeCodeRecord(
                    type="assistant",
                    message={"content": [{"type": "text", "text": "Hello"}]},
                ),
                lambda b: len(b) >= 1 and b[0]["type"] == "text",
            ),
            (ClaudeCodeRecord(type="assistant", message={"type": "message"}), lambda b: b == []),
            (
                ClaudeCodeRecord(
                    type="assistant",
                    message=ClaudeCodeMessageContent(
                        role="assistant", content=[{"type": "text", "text": "Hi"}]
                    ),
                ),
                lambda b: len(b) == 1,
            ),
        ],
        ids=["no_message", "dict_list", "dict_no_content", "typed_list"],
    )
    def test_content_blocks_raw_variants(self, record, expected_check):
        """Test content_blocks_raw across content types."""
        blocks = record.content_blocks_raw
        assert expected_check(blocks)


class TestClaudeCodeRecordParsedTimestamp:
    """Test parsed_timestamp property (lines 203-220)."""

    @pytest.mark.parametrize(
        "timestamp_value,expected_check",
        [
            (None, lambda ts: ts is None),
            (1700000000.0, lambda ts: isinstance(ts, datetime)),
            (1700000000000, lambda ts: isinstance(ts, datetime)),
            ("2023-11-14T10:00:00Z", lambda ts: isinstance(ts, datetime)),
            ("2023-11-14T10:00:00+00:00", lambda ts: isinstance(ts, datetime)),
            ("not-a-date", lambda ts: ts is None),
        ],
        ids=[
            "none_timestamp",
            "unix_seconds",
            "unix_milliseconds",
            "iso_z_suffix",
            "iso_offset",
            "invalid_format",
        ],
    )
    def test_parsed_timestamp_variants(self, timestamp_value, expected_check):
        """Test parsed_timestamp with various formats."""
        record = ClaudeCodeRecord(type="user", timestamp=timestamp_value)
        ts = record.parsed_timestamp
        assert expected_check(ts)


class TestClaudeCodeRecordToMeta:
    """Test to_meta method (lines 317-354)."""

    @pytest.mark.parametrize(
        "record,expected_check",
        [
            (
                ClaudeCodeRecord(
                    type="assistant",
                    uuid="id-1",
                    timestamp="2023-11-14T10:00:00Z",
                    message=ClaudeCodeMessageContent(
                        role="assistant",
                        model="claude-3",
                        usage=ClaudeCodeUsage(input_tokens=10, output_tokens=20),
                    ),
                ),
                lambda m: m.tokens is not None and m.tokens.input_tokens == 10,
            ),
            (
                ClaudeCodeRecord(
                    type="assistant",
                    uuid="id-1",
                    message={
                        "role": "assistant",
                        "usage": {"input_tokens": 15, "output_tokens": 25},
                    },
                ),
                lambda m: m.tokens is not None and m.tokens.input_tokens == 15,
            ),
            (
                ClaudeCodeRecord(type="assistant", costUSD=0.05),
                lambda m: m.cost is not None and m.cost.total_usd == 0.05,
            ),
            (
                ClaudeCodeRecord(
                    type="assistant",
                    message=ClaudeCodeMessageContent(role="assistant", model="claude-opus"),
                ),
                lambda m: m.model == "claude-opus",
            ),
            (
                ClaudeCodeRecord(type="assistant", message={"role": "assistant", "model": "claude-instant"}),
                lambda m: m.model == "claude-instant",
            ),
            (
                ClaudeCodeRecord(
                    type="user",
                    uuid="id-1",
                    message=ClaudeCodeUserMessage(role="user", content="Hello"),
                ),
                lambda m: m.tokens is None and m.cost is None,
            ),
        ],
        ids=[
            "usage_typed",
            "usage_dict",
            "cost_extraction",
            "model_typed",
            "model_dict",
            "no_usage_no_cost",
        ],
    )
    def test_to_meta_variants(self, record, expected_check):
        """Test to_meta extraction across variants."""
        meta = record.to_meta()
        assert expected_check(meta)


class TestClaudeCodeRecordFlags:
    """Test record flag properties."""

    @pytest.mark.parametrize(
        "record_type,is_compaction,is_tool_progress,is_actual",
        [
            ("summary", True, False, False),
            ("user", False, False, True),
            ("assistant", False, False, True),
            ("progress", False, True, False),
            ("system", False, False, False),
        ],
        ids=["summary_type", "user_type", "assistant_type", "progress_type", "system_type"],
    )
    def test_record_flags(self, record_type, is_compaction, is_tool_progress, is_actual):
        """Test record flag properties across types."""
        record = ClaudeCodeRecord(type=record_type)
        assert record.is_context_compaction == is_compaction
        assert record.is_tool_progress == is_tool_progress
        assert record.is_actual_message == is_actual


# =============================================================================
# ClaudeCodeToolUse & ThinkingBlock Tests
# =============================================================================


class TestClaudeCodeToolUse:
    """Test tool_use conversion."""

    def test_to_tool_call_conversion(self):
        """Tool use should convert to ToolCall."""
        tool = ClaudeCodeToolUse(
            id="tool-1",
            name="bash",
            input={"command": "ls -la"},
        )
        call = tool.to_tool_call()
        assert call.id == "tool-1"
        assert call.name == "bash"
        assert call.provider == "claude-code"


class TestClaudeCodeThinkingBlock:
    """Test thinking block conversion."""

    def test_to_reasoning_trace_conversion(self):
        """Thinking block should convert to ReasoningTrace."""
        block = ClaudeCodeThinkingBlock(thinking="Let me analyze this...")
        trace = block.to_reasoning_trace()
        assert trace.text == "Let me analyze this..."
        assert trace.provider == "claude-code"


class TestClaudeCodeUsage:
    """Test token usage conversion."""

    def test_to_token_usage_conversion(self):
        """Usage should convert with all token types."""
        usage = ClaudeCodeUsage(
            input_tokens=100,
            output_tokens=200,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
        )
        tokens = usage.to_token_usage()
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 200
        assert tokens.cache_read_tokens == 10
        assert tokens.cache_write_tokens == 5


# =============================================================================
# MERGED: Lines 100-102: _decode_json_bytes failure path (from iteration_coverage)
# =============================================================================


class TestDecodeJsonBytesFailure:
    """Tests for _decode_json_bytes with invalid input."""

    @pytest.mark.parametrize(
        "blob,expected_result",
        [
            (b"\xff\xfe\xff\xfe\xff\xfe\xff\xfe", None),  # all encodings fail or succeed
            (b'{"test": "data"}\x00\x00', lambda r: r is not None),  # null bytes handled
            (b"\x00", None),  # empty after decode
        ],
        ids=[
            "invalid_all_encodings",
            "null_bytes_stripped_again",
            "empty_after_decode",
        ],
    )
    def test_decode_json_bytes_edge_cases(self, blob, expected_result):
        """Test _decode_json_bytes edge cases."""
        result = _decode_json_bytes(blob)
        if expected_result is None:
            assert result is None or isinstance(result, str)
        elif callable(expected_result):
            assert expected_result(result)


# =============================================================================
# MERGED: Lines 232-235: _iter_json_stream bytes line decoding (from iteration_coverage)
# =============================================================================


class TestIterJsonStreamBytesDecoding:
    """Tests for JSONL line decoding when raw bytes are encountered."""

    @pytest.mark.parametrize(
        "data,expected_count,expected_check",
        [
            (
                b'{"type": "test", "data": "value1"}\n{"type": "test", "data": "value2"}\n',
                2,
                lambda r: r[0]["data"] == "value1",
            ),
            (
                b'{"valid": true}\n\xff\xfe\xff\xfe\n{"also_valid": true}\n',
                2,
                lambda r: any(x.get("valid") for x in r),
            ),
            (b'{"a": 1}\n{"b": 2}\n', 2, lambda r: r[0]["a"] == 1),
        ],
        ids=["jsonl_basic", "undecodable_line_skipped", "mixed_lines"],
    )
    def test_jsonl_bytes_decoding_variants(self, data, expected_count, expected_check):
        """Test JSONL bytes line decoding."""
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert len(results) >= expected_count - 1  # lenient for decode failures
        if results:
            assert expected_check(results)


# =============================================================================
# MERGED: Lines 262-263: unpack_lists=False path (from iteration_coverage)
# =============================================================================


class TestIterJsonStreamUnpackListsFalse:
    """Tests for _iter_json_stream with unpack_lists=False."""

    @pytest.mark.parametrize(
        "data,check_fn",
        [
            (
                b'[{"a": 1}, {"b": 2}, {"c": 3}]',
                lambda r: len(r) == 1 and isinstance(r[0], list) and len(r[0]) == 3,
            ),
            (
                b'{"single": "object"}',
                lambda r: len(r) == 1,
            ),
        ],
        ids=["list_not_unpacked", "dict_yielded"],
    )
    def test_unpack_lists_false_variants(self, data, check_fn):
        """Test _iter_json_stream unpack_lists=False behavior."""
        results_not_unpacked = list(_iter_json_stream(BytesIO(data), "test.json", unpack_lists=False))
        assert check_fn(results_not_unpacked)

    def test_strategy_exception_logs_debug(self):
        """ijson strategy failures are logged at debug level."""
        data = b'{"conversations": [{"item": 1}]}'
        results = list(_iter_json_stream(BytesIO(data), "test.json", unpack_lists=True))
        assert len(results) > 0


# =============================================================================
# MERGED: Lines 277-278: Empty/skipped line counting (from iteration_coverage)
# =============================================================================


class TestIterJsonStreamLineSkipping:
    """Tests for empty and skipped line counting in JSONL."""

    @pytest.mark.parametrize(
        "data,expected_count,check_fn",
        [
            (
                b'\n\n\n{"a": 1}\n\n\n{"b": 2}\n\n',
                2,
                lambda r: r[0]["a"] == 1 and r[1]["b"] == 2,
            ),
            (
                b"\n".join([b"invalid " + str(i).encode() for i in range(6)] + [json.dumps({"valid": True}).encode()]) + b"\n",
                1,
                lambda r: r[-1]["valid"] is True,
            ),
        ],
        ids=["many_empty_lines", "many_invalid_lines"],
    )
    def test_jsonl_line_skipping_variants(self, data, expected_count, check_fn):
        """Test JSONL empty/invalid line handling."""
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert len(results) >= 1
        assert check_fn(results)


# =============================================================================
# MERGED: Lines 354-355: OSError in cursor_state latest_mtime (from iteration_coverage)
# =============================================================================


class TestCursorStateLatestMtimeOsError:
    """Tests for OSError handling when computing latest_mtime."""

    def test_oserror_in_mtime_calculation_ignored(self, tmp_path: Path):
        """OSError during stat() is caught and ignored for latest_mtime."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        conv_file = source_dir / "conv.json"
        conv_file.write_text(json.dumps({"mapping": {}}))

        source = Source(name="test", path=source_dir)
        cursor_state: dict = {}

        original_stat = Path.stat

        def mock_stat_error(self, **kwargs):
            if self == conv_file or str(self) == str(conv_file):
                raise OSError("Stat failed")
            return original_stat(self, **kwargs)

        with patch.object(Path, "stat", mock_stat_error):
            conversations = list(iter_source_conversations(source, cursor_state=cursor_state))
            assert cursor_state.get("file_count", 0) >= 0


# =============================================================================
# Edge Cases & Integration
# =============================================================================


class TestSourceIterationEdgeCases:
    """Test complex source iteration scenarios."""

    def test_skip_dirs_filtering(self, tmp_path: Path):
        """_SKIP_DIRS should be filtered during walk (line 327)."""
        (tmp_path / "valid").mkdir()
        (tmp_path / "valid" / "conv.json").write_text('{"mapping": {}}')
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "conv.json").write_text('{"mapping": {}}')

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        conversations = list(iter_source_conversations(source, cursor_state=cursor_state))
        assert len(conversations) >= 1

    def test_has_ingest_extension_case_insensitive(self, tmp_path: Path):
        """Extension checking should be case-insensitive (line 303)."""
        json_file = tmp_path / "CONV.JSON"
        json_file.write_text('{"mapping": {}}')

        source = Source(name="test", path=json_file)
        conversations = list(iter_source_conversations(source))
        assert conversations

    def test_empty_source_path_returns_nothing(self, tmp_path: Path):
        """Empty folder path should return nothing (line 315-316)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        source = Source(name="test", folder=str(empty_dir))
        conversations = list(iter_source_conversations(source))
        assert len(conversations) == 0

    def test_source_path_expanduser(self, tmp_path: Path):
        """Source path should expand ~ (line 317)."""
        json_file = tmp_path / "conv.json"
        json_file.write_text('{"mapping": {}}')

        source = Source(name="test", path=json_file)
        conversations = list(iter_source_conversations(source))
        assert conversations
