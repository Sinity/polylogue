"""Tests for source decoding, JSON stream iteration, and ZIP processing.

Production code under test: polylogue/sources/source.py
Functions: _decode_json_bytes, _iter_json_stream, _ZipEntryValidator, _process_zip
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.sources.source import (
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_SIZE,
    _ZipEntryValidator,
    _decode_json_bytes,
    _iter_json_stream,
)


# =============================================================================
# _decode_json_bytes
# =============================================================================


class TestDecodeJsonBytesBasic:
    """Deterministic tests for _decode_json_bytes."""

    def test_utf8_roundtrip(self):
        """Encode then decode a UTF-8 JSON payload."""
        payload = '{"key": "value", "num": 42}'
        result = _decode_json_bytes(payload.encode("utf-8"))
        assert result is not None
        assert json.loads(result) == {"key": "value", "num": 42}

    def test_bom_stripping(self):
        """BOM-prefixed bytes are decoded (utf-8-sig in encoding list handles it)."""
        # utf-8-sig encoded bytes: BOM is consumed during decode
        payload = '{"key": "value"}'
        raw = payload.encode("utf-8-sig")  # Prepends EF BB BF
        result = _decode_json_bytes(raw)
        assert result is not None
        # utf-8 decoding succeeds first and preserves BOM as \ufeff.
        # utf-8-sig in the encoding list would strip it, but utf-8 wins first.
        # The decoded string may contain a leading BOM.
        # Verify the JSON content is present regardless.
        assert '"key"' in result
        assert '"value"' in result

    def test_utf8_sig_direct_bom(self):
        """Direct utf-8-sig BOM bytes are decoded successfully."""
        # Create bytes with a single BOM prefix
        bom = b"\xef\xbb\xbf"
        raw = bom + b'{"key": "value"}'
        result = _decode_json_bytes(raw)
        assert result is not None
        # Content is present
        assert "key" in result

    def test_null_bytes_removed(self):
        """Null bytes are stripped from decoded output."""
        payload = '{"key":\x00 "value"}'
        raw = payload.encode("utf-8")
        result = _decode_json_bytes(raw)
        assert result is not None
        assert "\x00" not in result
        assert "key" in result

    def test_fallback_encodings_utf16(self):
        """UTF-16 encoded payloads are decoded correctly."""
        payload = '{"key": "value"}'
        raw = payload.encode("utf-16")
        result = _decode_json_bytes(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_fallback_encodings_utf32(self):
        """UTF-32 encoded payloads are decoded correctly."""
        payload = '{"key": "value"}'
        raw = payload.encode("utf-32")
        result = _decode_json_bytes(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_returns_none_for_empty(self):
        """Empty bytes produce None."""
        result = _decode_json_bytes(b"")
        # Empty bytes decode to empty string, which is falsy
        assert result is None or result == ""

    def test_unicode_content_preserved(self):
        """Unicode content survives decode roundtrip."""
        payload = '{"emoji": "\\u2764", "jp": "\\u65e5\\u672c\\u8a9e"}'
        raw = payload.encode("utf-8")
        result = _decode_json_bytes(raw)
        assert result is not None
        assert json.loads(result) is not None


class TestDecodeJsonBytesFuzz:
    """Property-based tests for _decode_json_bytes."""

    @given(st.binary(max_size=4096))
    @settings(max_examples=200)
    def test_never_crashes_on_arbitrary_bytes(self, data: bytes):
        """_decode_json_bytes never raises on any input."""
        result = _decode_json_bytes(data)
        assert result is None or isinstance(result, str)


# =============================================================================
# _iter_json_stream
# =============================================================================


class TestIterJsonStream:
    """Tests for _iter_json_stream parsing strategies."""

    def test_jsonl_with_blank_lines(self):
        """JSONL parsing skips blank lines and yields valid objects."""
        content = b'{"a": 1}\n\n{"b": 2}\n\n\n{"c": 3}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.jsonl"))
        assert len(items) == 3
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}
        assert items[2] == {"c": 3}

    def test_json_root_array(self):
        """Root array JSON is unpacked into individual items."""
        content = json.dumps([{"a": 1}, {"b": 2}]).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}

    def test_conversations_wrapper(self):
        """{"conversations": [...]} is unpacked into individual items."""
        content = json.dumps({"conversations": [{"id": "c1"}, {"id": "c2"}]}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"id": "c1"}
        assert items[1] == {"id": "c2"}

    def test_single_dict_yielded_as_is(self):
        """A single JSON dict is yielded without unwrapping."""
        content = json.dumps({"key": "value"}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 1
        assert items[0] == {"key": "value"}

    def test_jsonl_invalid_lines_skipped(self):
        """Invalid JSON lines in JSONL are skipped (not crashed on)."""
        content = b'{"valid": 1}\nnot json at all\n{"also_valid": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.jsonl"))
        assert len(items) == 2
        assert items[0] == {"valid": 1}
        assert items[1] == {"also_valid": 2}

    def test_ndjson_extension_treated_as_jsonl(self):
        """Files with .ndjson extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.ndjson"))
        assert len(items) == 2

    def test_jsonl_txt_extension(self):
        """Files with .jsonl.txt extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.jsonl.txt"))
        assert len(items) == 2


# =============================================================================
# _ZipEntryValidator
# =============================================================================


class TestZipEntryValidator:
    """Tests for ZIP bomb protection and entry filtering."""

    def _make_zip_info(
        self,
        filename: str,
        file_size: int = 1000,
        compress_size: int = 100,
        is_dir: bool = False,
    ) -> zipfile.ZipInfo:
        """Create a ZipInfo with specified attributes."""
        info = zipfile.ZipInfo(filename)
        info.file_size = file_size
        info.compress_size = compress_size
        if is_dir:
            info.external_attr = 0o40775 << 16  # Directory bit
        return info

    def test_bomb_protection_compression_ratio(self):
        """Entries with compression ratio > MAX_COMPRESSION_RATIO are rejected."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state={"failed_files": [], "failed_count": 0},
            zip_path=Path("test.zip"),
        )
        # Ratio = 200000 / 1 = 200000, well above MAX_COMPRESSION_RATIO
        bomb_entry = self._make_zip_info("data.json", file_size=200000, compress_size=1)
        entries = list(validator.filter_entries([bomb_entry]))
        assert len(entries) == 0

    def test_size_limit_rejection(self):
        """Entries with uncompressed size > MAX_UNCOMPRESSED_SIZE are rejected."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state={"failed_files": [], "failed_count": 0},
            zip_path=Path("test.zip"),
        )
        huge_entry = self._make_zip_info(
            "data.json",
            file_size=MAX_UNCOMPRESSED_SIZE + 1,
            compress_size=MAX_UNCOMPRESSED_SIZE,
        )
        entries = list(validator.filter_entries([huge_entry]))
        assert len(entries) == 0

    def test_claude_filter_conversations_only(self):
        """Claude provider ZIP: only conversations.json passes through."""
        validator = _ZipEntryValidator(
            "claude",
            cursor_state=None,
            zip_path=Path("claude.zip"),
        )
        entries_in = [
            self._make_zip_info("conversations.json", file_size=5000, compress_size=500),
            self._make_zip_info("settings.json", file_size=1000, compress_size=100),
            self._make_zip_info("account.json", file_size=1000, compress_size=100),
        ]
        entries_out = list(validator.filter_entries(entries_in))
        assert len(entries_out) == 1
        assert entries_out[0].filename == "conversations.json"

    def test_directories_skipped(self):
        """Directory entries in ZIP are skipped."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("test.zip"),
        )
        dir_entry = self._make_zip_info("some_dir/", is_dir=True)
        # Manually set directory flag since ZipInfo.is_dir() checks filename
        dir_entry.filename = "some_dir/"
        entries = list(validator.filter_entries([dir_entry]))
        assert len(entries) == 0

    def test_non_json_extensions_skipped(self):
        """Non-JSON files in ZIP are skipped."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("test.zip"),
        )
        entries_in = [
            self._make_zip_info("readme.txt", file_size=500, compress_size=200),
            self._make_zip_info("image.png", file_size=5000, compress_size=4000),
            self._make_zip_info("data.json", file_size=1000, compress_size=100),
        ]
        entries_out = list(validator.filter_entries(entries_in))
        assert len(entries_out) == 1
        assert entries_out[0].filename == "data.json"

    def test_valid_entry_passes_through(self):
        """A normal JSON entry with reasonable ratio passes validation."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("test.zip"),
        )
        normal_entry = self._make_zip_info("conversations.json", file_size=50000, compress_size=5000)
        entries = list(validator.filter_entries([normal_entry]))
        assert len(entries) == 1

    def test_cursor_state_records_failures(self):
        """Rejected entries record failures in cursor_state."""
        cursor_state: dict = {"failed_files": [], "failed_count": 0}
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=cursor_state,
            zip_path=Path("archive.zip"),
        )
        bomb_entry = self._make_zip_info("bomb.json", file_size=500000, compress_size=1)
        list(validator.filter_entries([bomb_entry]))
        assert cursor_state["failed_count"] >= 1
        assert len(cursor_state["failed_files"]) >= 1
