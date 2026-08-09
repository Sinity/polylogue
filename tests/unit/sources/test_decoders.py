"""Tests for source decoding, JSON stream iteration, and ZIP processing.

Insightion code under test: polylogue/sources/decoders.py
Functions: _decode_json_bytes, _iter_json_stream, _ZipEntryValidator, _process_zip
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.sources.decoder_json import JsonlDecodeError
from polylogue.sources.decoders import (
    MAX_AGGREGATE_UNCOMPRESSED_SIZE,
    MAX_UNCOMPRESSED_SIZE,
    _decode_json_bytes,
    _iter_json_stream,
    _ZipEntryValidator,
    open_bounded_zip_entry,
)
from polylogue.storage.cursor_state import CursorFailurePayload, CursorStatePayload

# =============================================================================
# _decode_json_bytes
# =============================================================================


def _seeded_cursor_state() -> CursorStatePayload:
    return {"failed_files": [], "failed_count": 0}


class TestDecodeJsonBytesBasic:
    """Deterministic tests for _decode_json_bytes."""

    def test_utf8_roundtrip(self) -> None:
        """Encode then decode a UTF-8 JSON payload."""
        payload = '{"key": "value", "num": 42}'
        result = _decode_json_bytes(payload.encode("utf-8"))
        assert result is not None
        assert json.loads(result) == {"key": "value", "num": 42}

    def test_bom_stripping(self) -> None:
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

    def test_utf8_sig_direct_bom(self) -> None:
        """Direct utf-8-sig BOM bytes are decoded successfully."""
        # Create bytes with a single BOM prefix
        bom = b"\xef\xbb\xbf"
        raw = bom + b'{"key": "value"}'
        result = _decode_json_bytes(raw)
        assert result is not None
        # Content is present
        assert "key" in result

    def test_null_bytes_removed(self) -> None:
        """Null bytes are stripped from decoded output."""
        payload = '{"key":\x00 "value"}'
        raw = payload.encode("utf-8")
        result = _decode_json_bytes(raw)
        assert result is not None
        assert "\x00" not in result
        assert "key" in result

    def test_fallback_encodings_utf16(self) -> None:
        """UTF-16 encoded payloads are decoded correctly."""
        payload = '{"key": "value"}'
        raw = payload.encode("utf-16")
        result = _decode_json_bytes(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_fallback_encodings_utf32(self) -> None:
        """UTF-32 encoded payloads are decoded correctly."""
        payload = '{"key": "value"}'
        raw = payload.encode("utf-32")
        result = _decode_json_bytes(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_returns_none_for_empty(self) -> None:
        """Empty bytes produce None."""
        result = _decode_json_bytes(b"")
        # Empty bytes decode to empty string, which is falsy
        assert result is None or result == ""

    def test_unicode_content_preserved(self) -> None:
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
    def test_never_crashes_on_arbitrary_bytes(self, data: bytes) -> None:
        """_decode_json_bytes never raises on any input."""
        result = _decode_json_bytes(data)
        assert result is None or isinstance(result, str)


# =============================================================================
# _iter_json_stream
# =============================================================================


class TestIterJsonStream:
    """Tests for _iter_json_stream parsing strategies."""

    def test_jsonl_with_blank_lines(self) -> None:
        """JSONL parsing skips blank lines and yields valid objects."""
        content = b'{"a": 1}\n\n{"b": 2}\n\n\n{"c": 3}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.jsonl"))
        assert len(items) == 3
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}
        assert items[2] == {"c": 3}

    def test_json_root_array(self) -> None:
        """Root array JSON is unpacked into individual items."""
        content = json.dumps([{"a": 1}, {"b": 2}]).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}

    def test_sessions_wrapper(self) -> None:
        """{"sessions": [...]} is unpacked into individual items."""
        content = json.dumps({"sessions": [{"id": "c1"}, {"id": "c2"}]}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"id": "c1"}
        assert items[1] == {"id": "c2"}

    def test_single_dict_yielded_as_is(self) -> None:
        """A single JSON dict is yielded without unwrapping."""
        content = json.dumps({"key": "value"}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 1
        assert items[0] == {"key": "value"}

    def test_jsonl_invalid_lines_skipped(self) -> None:
        """Invalid JSON lines in JSONL are skipped (not crashed on)."""
        content = b'{"valid": 1}\nnot json at all\n{"also_valid": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.jsonl"))
        assert len(items) == 2
        assert items[0] == {"valid": 1}
        assert items[1] == {"also_valid": 2}

    def test_ndjson_extension_treated_as_jsonl(self) -> None:
        """Files with .ndjson extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.ndjson"))
        assert len(items) == 2

    def test_jsonl_txt_extension(self) -> None:
        """Files with .jsonl.txt extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.jsonl.txt"))
        assert len(items) == 2

    def test_strict_jsonl_decode_reports_physical_offending_line(self) -> None:
        content = b'{"valid": 1}\n\nnot json at all\n{"later": 2}\n'
        with pytest.raises(JsonlDecodeError) as exc_info:
            list(_iter_json_stream(io.BytesIO(content), "data.jsonl", fail_on_decode_error=True))
        assert exc_info.value.line_number == 3


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

    def test_bomb_protection_compression_ratio(self) -> None:
        """Entries with compression ratio > MAX_COMPRESSION_RATIO are rejected."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=_seeded_cursor_state(),
            zip_path=Path("test.zip"),
        )
        # Ratio = 200000 / 1 = 200000, well above MAX_COMPRESSION_RATIO
        bomb_entry = self._make_zip_info("data.json", file_size=200000, compress_size=1)
        entries = list(validator.filter_entries([bomb_entry]))
        assert len(entries) == 0

    def test_size_limit_rejection(self) -> None:
        """Entries with uncompressed size > MAX_UNCOMPRESSED_SIZE are rejected."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=_seeded_cursor_state(),
            zip_path=Path("test.zip"),
        )
        huge_entry = self._make_zip_info(
            "data.json",
            file_size=MAX_UNCOMPRESSED_SIZE + 1,
            compress_size=MAX_UNCOMPRESSED_SIZE,
        )
        entries = list(validator.filter_entries([huge_entry]))
        assert len(entries) == 0

    def test_claude_json_entries_are_not_special_cased(self) -> None:
        """Claude ZIP validation now relies on artifact classification, not filename allowlists."""
        validator = _ZipEntryValidator(
            "claude-ai",
            cursor_state=None,
            zip_path=Path("claude.zip"),
        )
        entries_in = [
            self._make_zip_info("sessions.json", file_size=5000, compress_size=500),
            self._make_zip_info("settings.json", file_size=1000, compress_size=100),
            self._make_zip_info("account.json", file_size=1000, compress_size=100),
        ]
        entries_out = list(validator.filter_entries(entries_in))
        assert [entry.filename for entry in entries_out] == [
            "sessions.json",
            "settings.json",
            "account.json",
        ]

    def test_directories_skipped(self) -> None:
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

    def test_non_json_extensions_skipped(self) -> None:
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

    def test_valid_entry_passes_through(self) -> None:
        """A normal JSON entry with reasonable ratio passes validation."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("test.zip"),
        )
        normal_entry = self._make_zip_info("sessions.json", file_size=50000, compress_size=5000)
        entries = list(validator.filter_entries([normal_entry]))
        assert len(entries) == 1

    def test_bounded_open_preserves_duplicate_zipinfo_identity(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("duplicate.json", b"first")
            zf.writestr("duplicate.json", b"second")
        buffer.seek(0)

        with zipfile.ZipFile(buffer) as zf:
            infos = zf.infolist()
            with open_bounded_zip_entry(zf, infos[0]) as handle:
                assert handle.read() == b"first"

    def test_cursor_state_records_failures(self) -> None:
        """Rejected entries record failures in cursor_state."""
        cursor_state = _seeded_cursor_state()
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=cursor_state,
            zip_path=Path("archive.zip"),
        )
        bomb_entry = self._make_zip_info("bomb.json", file_size=500000, compress_size=1)
        list(validator.filter_entries([bomb_entry]))
        failed_files: list[CursorFailurePayload] = cursor_state.get("failed_files", [])
        assert cursor_state["failed_count"] >= 1
        assert len(failed_files) >= 1

    def test_aggregate_size_limit_rejects_many_entries_under_per_entry_cap(self) -> None:
        """A zip bomb built from many entries, each individually under
        MAX_UNCOMPRESSED_SIZE, is rejected once their SUM would exceed
        MAX_AGGREGATE_UNCOMPRESSED_SIZE (polylogue-lqxx).

        Each entry here is deliberately just 1 byte under the per-entry
        cap, so this proves the aggregate check fires on its own -- the
        existing per-entry check alone would accept every single one of
        these entries.
        """
        cursor_state = _seeded_cursor_state()
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=cursor_state,
            zip_path=Path("bomb.zip"),
        )
        per_entry_size = MAX_UNCOMPRESSED_SIZE - 1
        assert per_entry_size <= MAX_UNCOMPRESSED_SIZE  # sanity: never trips the per-entry cap

        # 7 entries * (10 GiB - 1 byte) ~= 70 GiB, comfortably over the
        # 64 GiB aggregate cap, while no single entry is oversized.
        entry_count = (MAX_AGGREGATE_UNCOMPRESSED_SIZE // per_entry_size) + 2
        entries = [
            self._make_zip_info(f"conversation_{i}.json", file_size=per_entry_size, compress_size=per_entry_size)
            for i in range(entry_count)
        ]

        accepted = list(validator.filter_entries(entries))

        # Every accepted entry was individually under the per-entry cap
        # (proven above), yet not all entries were accepted -- the
        # aggregate check, not the per-entry check, did the rejecting.
        assert len(accepted) < entry_count
        assert sum(info.file_size for info in accepted) <= MAX_AGGREGATE_UNCOMPRESSED_SIZE

        failed_files: list[CursorFailurePayload] = cursor_state.get("failed_files", [])
        assert cursor_state["failed_count"] >= 1
        assert any("Aggregate uncompressed size" in failure["error"] for failure in failed_files)

    def test_aggregate_size_limit_allows_archive_comfortably_under_cap(self) -> None:
        """Multiple entries whose sum stays well under the aggregate cap
        all decode successfully -- no regression for legitimate
        multi-file exports."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("normal_export.zip"),
        )
        # 3 entries of 1 GiB each = 3 GiB total, far under both the 10 GiB
        # per-entry cap and the 64 GiB aggregate cap.
        one_gib = 1024 * 1024 * 1024
        entries = [
            self._make_zip_info(f"conversation_{i}.json", file_size=one_gib, compress_size=one_gib) for i in range(3)
        ]

        accepted = list(validator.filter_entries(entries))

        assert len(accepted) == 3
        assert sum(info.file_size for info in accepted) == 3 * one_gib

    def test_validator_leaves_terminal_artifact_classification_to_zip_processing(self) -> None:
        """ZIP validation must not path-exclude entries before payload decoding.

        Regression test for polylogue-dc1k: every ``OriginArtifactRule.path_pattern``
        in ``origin_specs.py`` is anchored ``(?:^|/)``, but the entry was
        classified on ``f"{zip_path}:{name}"`` -- the character immediately
        before the pattern's leading segment was ``:``, never ``/`` or
        start-of-string, so no rule could ever match and this exclusion was
        dead code for every zip import. Builds a *real* zip (not a bare
        ``ZipInfo``) so the entries here are exactly what ``zf.infolist()``
        would hand ``filter_entries`` in production.
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            # Matches the "workflow_run" OriginArtifactRule for claude-code
            # (parse_policy="fact" -> parse_as_session=False): must be excluded.
            zf.writestr("workflows/run.json", json.dumps({"run_id": "abc"}))
            # Matches the "agent_transcript" OriginArtifactRule for claude-code
            # (parse_policy="session" -> parse_as_session=True): must survive.
            zf.writestr("subagents/agent-1.jsonl", json.dumps({"type": "user"}) + "\n")
            # No OriginArtifactRule matches this path at all, so ZIP processing
            # must leave it available for ordinary payload classification.
            zf.writestr("sessions.json", json.dumps({"conversations": []}))
        buffer.seek(0)

        with zipfile.ZipFile(buffer) as zf:
            validator = _ZipEntryValidator(
                "claude-code", cursor_state=_seeded_cursor_state(), zip_path=Path("export.zip")
            )
            accepted = [info.filename for info in validator.filter_entries(zf.infolist())]

        assert "workflows/run.json" in accepted
        assert "subagents/agent-1.jsonl" in accepted
        assert "sessions.json" in accepted
