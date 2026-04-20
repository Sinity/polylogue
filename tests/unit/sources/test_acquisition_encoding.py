"""Tests for ZIP encoding edge cases through the acquisition/parse pipeline.

These tests verify that the ZIP extraction -> JSON/JSONL parse pipeline handles
BOM markers, mixed line endings, partial corruption, and non-UTF-8 encodings
correctly. They test through iter_source_conversations (parse path) and
iter_source_raw_data (acquisition path).
"""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Source
from polylogue.sources.parsers.base import ParsedConversation, RawConversationData
from polylogue.storage.state_views import CursorFailurePayload, CursorStatePayload
from tests.infra.encoding_fixtures import EncodingFixtureBuilder


def _make_source(path: Path, name: str = "codex") -> Source:
    """Create a Source config pointing at a directory."""
    return Source(name=name, path=path)


def _collect_conversations(source_path: Path, provider: str = "codex") -> list[ParsedConversation]:
    """Run iter_source_conversations and collect results."""
    from polylogue.sources.source_parsing import iter_source_conversations

    source = _make_source(source_path, name=provider)
    return list(iter_source_conversations(source))


def _collect_conversations_with_raw(
    source_path: Path,
    provider: str = "codex",
    *,
    cursor_state: CursorStatePayload | None = None,
) -> list[tuple[RawConversationData | None, ParsedConversation]]:
    """Run iter_source_conversations_with_raw and collect results."""
    from polylogue.sources.source_parsing import iter_source_conversations_with_raw

    source = _make_source(source_path, name=provider)
    return list(iter_source_conversations_with_raw(source, cursor_state=cursor_state, capture_raw=True))


def _collect_raw_data(
    source_path: Path,
    provider: str = "codex",
    *,
    cursor_state: CursorStatePayload | None = None,
) -> list[RawConversationData]:
    """Run iter_source_raw_data and collect results."""
    from polylogue.sources.source_acquisition import iter_source_raw_data

    source = _make_source(source_path, name=provider)
    return list(iter_source_raw_data(source, cursor_state=cursor_state))


def _empty_cursor_state() -> CursorStatePayload:
    return {}


def _failed_files(cursor_state: CursorStatePayload) -> list[CursorFailurePayload]:
    return cursor_state.get("failed_files", [])


class TestZipBomHandling:
    """BOM handling through ZIP extraction -> parse pipeline."""

    def test_utf8_bom_json_in_zip_parses(self, tmp_path: Path) -> None:
        """JSON file with UTF-8 BOM inside ZIP is decoded correctly."""
        EncodingFixtureBuilder.bom_utf8_json_zip(tmp_path)
        results = _collect_conversations(tmp_path, provider="chatgpt")
        # Should parse at least one conversation despite BOM
        assert len(results) >= 1
        conv = results[0]
        assert conv.messages

    def test_bom_in_jsonl_lines_stripped(self, tmp_path: Path) -> None:
        """JSONL with BOM chars on individual lines inside ZIP."""
        EncodingFixtureBuilder.bom_in_jsonl_zip(tmp_path)
        results = _collect_conversations(tmp_path, provider="codex")
        assert len(results) >= 1
        conv = results[0]
        assert conv.messages

    def test_utf16_bom_json_in_zip_no_crash(self, tmp_path: Path) -> None:
        """UTF-16 encoded JSON inside ZIP does not raise unhandled exceptions.

        This exercises _decode_json_bytes multi-encoding fallback. The json.load()
        strategy 3 path receives the raw bytes from the ZIP handle, and UTF-16
        BOM may or may not be handled depending on the json module's byte mode.
        Either parsing succeeds or the error is caught — no unhandled crash.
        """
        EncodingFixtureBuilder.utf16_bom_json_zip(tmp_path)
        # Must not raise — either parses or silently fails
        _collect_conversations(tmp_path, provider="chatgpt")
        # If it parsed, great; if not, the error was handled gracefully
        # (json.load on a BinaryIO with UTF-16 bytes may raise JSONDecodeError
        # which is caught by the outer handler)


class TestZipLineEndings:
    """Line ending handling in ZIP-extracted JSONL."""

    def test_mixed_line_endings_jsonl_in_zip(self, tmp_path: Path) -> None:
        """JSONL with mixed CRLF/LF/CR endings inside ZIP parses correctly."""
        EncodingFixtureBuilder.mixed_line_endings_zip(tmp_path)
        results = _collect_conversations(tmp_path, provider="codex")
        assert len(results) >= 1
        conv = results[0]
        # Should have parsed messages from the JSONL despite mixed endings
        assert conv.messages


class TestZipPartialCorruption:
    """Resilience when ZIP contains mix of valid and corrupt entries."""

    def test_valid_entries_survive_corrupt_siblings(self, tmp_path: Path) -> None:
        """Valid JSON entries are parsed even when other entries are corrupt."""
        EncodingFixtureBuilder.partial_corruption_zip(tmp_path)
        cursor_state: CursorStatePayload = _empty_cursor_state()
        results = _collect_conversations_with_raw(tmp_path, provider="chatgpt", cursor_state=cursor_state)
        # At least the valid.json entry should produce a conversation
        # (it's added to the ZIP first, so it's yielded before corrupt.json fails)
        assert len(results) >= 1
        _raw, conv = results[0]
        assert conv.messages

    def test_cursor_state_records_corrupt_entries(self, tmp_path: Path) -> None:
        """cursor_state tracks failures for corrupt ZIP entries."""
        EncodingFixtureBuilder.partial_corruption_zip(tmp_path)
        cursor_state: CursorStatePayload = _empty_cursor_state()
        _results = _collect_conversations_with_raw(tmp_path, provider="chatgpt", cursor_state=cursor_state)
        # The corrupt entry should cause a recorded failure
        # _initialize_cursor_state creates failed_files as a list
        failed = _failed_files(cursor_state)
        assert len(failed) >= 1, f"Expected at least 1 failure recorded, got cursor_state={cursor_state}"


class TestAcquisitionRawPreservation:
    """Raw acquisition path preserves bytes; parse path strips BOM."""

    def test_raw_acquisition_preserves_bytes_including_bom(self, tmp_path: Path) -> None:
        """iter_source_raw_data stores raw bytes as-is (BOM not stripped at acquisition)."""
        EncodingFixtureBuilder.bom_utf8_json_zip(tmp_path)
        results = _collect_raw_data(tmp_path, provider="chatgpt")
        assert len(results) >= 1
        # Content should be in blob store with BOM preserved
        assert results[0].blob_hash is not None
        from polylogue.storage.blob_store import get_blob_store

        blob_bytes = get_blob_store().read_all(results[0].blob_hash)
        assert blob_bytes.startswith(b"\xef\xbb\xbf"), "BOM should be preserved in raw acquisition"

    def test_parse_stage_strips_bom(self, tmp_path: Path) -> None:
        """iter_source_conversations strips BOM during parsing."""
        EncodingFixtureBuilder.bom_utf8_json_zip(tmp_path)
        results = _collect_conversations(tmp_path, provider="chatgpt")
        assert len(results) >= 1
        conv = results[0]
        # The parsed title should not contain BOM artifacts
        if conv.title:
            assert "\ufeff" not in conv.title
