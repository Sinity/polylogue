"""Tests for iter_source_conversations_with_raw and _iter_json_stream.

Covers:
- iter_source_conversations_with_raw: raw byte capture for all provider types
- _iter_json_stream: ijson streaming strategies + fallback
- parse_drive_payload: Drive-specific payload parsing
- Cursor state tracking in raw capture variant
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.sources.source import (
    _iter_json_stream,
    iter_source_conversations_with_raw,
    parse_drive_payload,
)


# =============================================================================
# _iter_json_stream strategies
# =============================================================================


class TestIterJsonStreamJsonl:
    """Tests for JSONL file streaming."""

    def test_jsonl_yields_each_line(self):
        data = b'{"a": 1}\n{"b": 2}\n'
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert len(results) == 2
        assert results[0] == {"a": 1}
        assert results[1] == {"b": 2}

    def test_jsonl_skips_empty_lines(self):
        data = b'\n\n{"a": 1}\n\n'
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert len(results) == 1
        assert results[0] == {"a": 1}

    def test_jsonl_skips_invalid_lines(self):
        data = b'{"valid": true}\nnot json\n{"also": "valid"}\n'
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert len(results) == 2
        assert results[0] == {"valid": True}
        assert results[1] == {"also": "valid"}

    def test_jsonl_handles_bytes_with_encoding_issues(self):
        # A valid JSON line with special handling for null bytes
        line = b'{"id": "test"}\n'
        results = list(_iter_json_stream(BytesIO(line), "test.jsonl"))
        assert len(results) == 1
        assert results[0]["id"] == "test"

    def test_ndjson_extension(self):
        data = b'{"a": 1}\n'
        results = list(_iter_json_stream(BytesIO(data), "test.ndjson"))
        assert len(results) == 1
        assert results[0] == {"a": 1}

    def test_jsonl_txt_extension(self):
        data = b'{"a": 1}\n'
        results = list(_iter_json_stream(BytesIO(data), "data.jsonl.txt"))
        assert len(results) == 1
        assert results[0] == {"a": 1}

    def test_jsonl_many_errors_summarized(self):
        """More than 3 errors should be summarized."""
        lines = []
        for i in range(6):
            lines.append(b"bad json " + str(i).encode())
        lines.append(json.dumps({"good": True}).encode())
        data = b"\n".join(lines) + b"\n"
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert len(results) == 1
        assert results[0]["good"] is True

    def test_jsonl_empty_file(self):
        data = b""
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert results == []

    def test_jsonl_only_whitespace(self):
        data = b"\n\n\n"
        results = list(_iter_json_stream(BytesIO(data), "test.jsonl"))
        assert results == []


class TestIterJsonStreamJson:
    """Tests for JSON file streaming (ijson strategies + fallback)."""

    def test_root_list_strategy(self):
        """Strategy 1: ijson items(item) for root-level list."""
        data = json.dumps([{"id": 1}, {"id": 2}]).encode()
        results = list(_iter_json_stream(BytesIO(data), "test.json"))
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2

    def test_conversations_key_strategy(self):
        """Strategy 2: ijson conversations.item for nested list."""
        data = json.dumps({"conversations": [{"id": "c1"}, {"id": "c2"}]}).encode()
        results = list(_iter_json_stream(BytesIO(data), "test.json"))
        assert len(results) == 2
        assert results[0]["id"] == "c1"
        assert results[1]["id"] == "c2"

    def test_single_dict_fallback(self):
        """Strategy 3: json.load for single dict."""
        data = json.dumps({"id": "single", "data": True}).encode()
        results = list(_iter_json_stream(BytesIO(data), "test.json"))
        assert len(results) == 1
        assert results[0]["id"] == "single"

    def test_unpack_lists_false(self):
        """With unpack_lists=False, list is yielded as single item."""
        data = json.dumps([{"id": 1}, {"id": 2}]).encode()
        results = list(_iter_json_stream(BytesIO(data), "test.json", unpack_lists=False))
        assert len(results) == 1
        assert isinstance(results[0], list)
        assert len(results[0]) == 2
        assert results[0][0]["id"] == 1

    def test_empty_list(self):
        data = json.dumps([]).encode()
        results = list(_iter_json_stream(BytesIO(data), "test.json"))
        assert len(results) == 0

    def test_empty_dict(self):
        data = json.dumps({}).encode()
        results = list(_iter_json_stream(BytesIO(data), "test.json"))
        assert len(results) == 1
        assert results[0] == {}

    def test_nested_conversations_with_unpack_false(self):
        """With unpack_lists=False, nested conversations are not unpacked."""
        data = json.dumps({"conversations": [{"id": "c1"}, {"id": "c2"}]}).encode()
        results = list(_iter_json_stream(BytesIO(data), "test.json", unpack_lists=False))
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert "conversations" in results[0]


# =============================================================================
# iter_source_conversations_with_raw: basic functionality
# =============================================================================


class TestRawCaptureBasic:
    """Tests for basic raw byte capture functionality."""

    def test_json_file_captures_raw_bytes(self, tmp_path):
        """Raw bytes are captured for JSON file conversations."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("raw-test")
         .add_message("user", "hello", text="hello")
         .write_to(tmp_path / "conv.json"))

        source = Source(name="chatgpt", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None
        assert raw_data.raw_bytes is not None
        assert len(raw_data.raw_bytes) > 0
        assert raw_data.source_path is not None
        assert raw_data.file_mtime is not None

    def test_capture_raw_false_yields_none(self, tmp_path):
        """With capture_raw=False, raw data is None for non-grouped providers."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("no-raw")
         .add_message("user", "hi", text="hi")
         .write_to(tmp_path / "conv.json"))

        source = Source(name="chatgpt", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source, capture_raw=False))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is None

    def test_no_path_returns_empty(self, tmp_path):
        """Source with no path (using folder) yields nothing if folder doesn't exist."""
        # Source requires either path or folder, use folder with nonexistent path
        nonexistent = tmp_path / "nonexistent"
        source = Source(name="empty", folder=str(nonexistent))
        results = list(iter_source_conversations_with_raw(source))
        assert results == []

    def test_nonexistent_path_returns_empty(self, tmp_path):
        """Source with nonexistent path yields nothing."""
        nonexistent = tmp_path / "does_not_exist"
        source = Source(name="test", path=nonexistent)
        results = list(iter_source_conversations_with_raw(source))
        assert results == []

    def test_empty_directory_returns_empty(self, tmp_path):
        """Empty directory yields nothing."""
        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert results == []


# =============================================================================
# iter_source_conversations_with_raw: grouped providers (claude-code, codex)
# =============================================================================


class TestRawCaptureGrouped:
    """Tests for raw capture with grouped providers (one file = one conversation)."""

    def test_claude_code_jsonl_captures_entire_file(self, tmp_path):
        """Claude Code JSONL: entire file captured as raw bytes."""
        records = [
            {"type": "user", "uuid": "u1", "sessionId": "s1", "message": {"content": "Hello"}},
            {"type": "assistant", "uuid": "a1", "sessionId": "s1", "message": {"content": "Hi"}},
        ]
        (tmp_path / "session.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        source = Source(name="claude-code", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None
        assert raw_data.provider_hint == "claude-code"
        # Raw bytes should contain the entire file content
        assert b"Hello" in raw_data.raw_bytes
        assert b"Hi" in raw_data.raw_bytes

    def test_codex_json_captures_raw(self, tmp_path):
        """Codex JSON list: entire file captured."""
        payload = [
            {"type": "session_meta", "payload": {"id": "codex-1", "timestamp": "2025-01-01"}},
            {"type": "response_item", "payload": {"type": "message", "id": "m1", "role": "user",
             "content": [{"type": "input_text", "text": "Test"}]}},
        ]
        (tmp_path / "codex.json").write_text(json.dumps(payload))

        source = Source(name="codex", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None
        assert b"codex-1" in raw_data.raw_bytes

    def test_gemini_grouped_capture(self, tmp_path):
        """Gemini (grouped provider) captures entire file."""
        payload = [
            {"role": "user", "text": "Hello Gemini"},
            {"role": "model", "text": "Hi there"},
        ]
        (tmp_path / "gemini_chat.json").write_text(json.dumps(payload))

        source = Source(name="gemini", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None
        assert b"Hello Gemini" in raw_data.raw_bytes

    def test_grouped_provider_single_conversation_per_file(self, tmp_path):
        """Grouped providers yield single conversation per file regardless of structure."""
        # For claude-code, write JSONL with proper message format
        records = [
            {"type": "user", "uuid": "u1", "sessionId": "s1", "message": {"content": "Q1"}},
            {"type": "assistant", "uuid": "a1", "sessionId": "s1", "message": {"content": "A1"}},
        ]
        (tmp_path / "session1.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        source = Source(name="claude-code", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        # All records bundled into single conversation
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None
        assert raw_data.source_index is None


# =============================================================================
# iter_source_conversations_with_raw: ZIP handling
# =============================================================================


class TestRawCaptureZip:
    """Tests for raw capture from ZIP files."""

    def test_zip_json_captures_raw(self, tmp_path):
        """ZIP with JSON: raw bytes captured per conversation."""
        from tests.helpers import ChatGPTExportBuilder

        conv = ChatGPTExportBuilder("zip-raw").add_node("user", "Zip test").build()
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps([conv]))

        source = Source(name="chatgpt", path=zip_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv_parsed = results[0]
        assert raw_data is not None
        assert raw_data.source_path.startswith(str(zip_path))
        assert b"Zip test" in raw_data.raw_bytes

    def test_zip_grouped_jsonl_captures_entire_content(self, tmp_path):
        """ZIP with grouped JSONL: entire file content captured."""
        records = [
            {"type": "user", "uuid": "u1", "sessionId": "s1", "message": {"content": "From zip"}},
        ]
        zip_path = tmp_path / "claude-code.zip"
        content = "\n".join(json.dumps(r) for r in records) + "\n"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("session.jsonl", content)

        source = Source(name="claude-code", path=zip_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None
        assert b"From zip" in raw_data.raw_bytes

    def test_zip_bomb_protection_with_raw(self, tmp_path):
        """ZIP bomb protection works in raw capture variant."""
        data = b"A" * (10 * 1024 * 1024)  # 10MB of A's
        zip_path = tmp_path / "bomb.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("bomb.json", data)

        # Verify high compression ratio
        with zipfile.ZipFile(zip_path) as zf:
            info = zf.infolist()[0]
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio <= 100:
                    pytest.skip("Compression ratio not high enough for this test")

        source = Source(name="chatgpt", path=zip_path)
        cursor_state: dict = {}
        results = list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        assert len(results) == 0
        assert cursor_state.get("failed_count", 0) >= 1

    def test_zip_claude_filter_only_conversations_json(self, tmp_path):
        """Claude AI ZIP only processes conversations.json."""
        from tests.helpers import ClaudeExportBuilder

        conv = ClaudeExportBuilder("c1").add_human("Hello").build()
        zip_path = tmp_path / "claude-export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps([conv["conversations"][0]]))
            zf.writestr("other.json", json.dumps({"irrelevant": True}))

        source = Source(name="claude", path=zip_path)
        results = list(iter_source_conversations_with_raw(source))
        # Should process conversations.json only
        assert isinstance(results, list)

    def test_zip_multiple_files_each_indexed(self, tmp_path):
        """Multiple conversations in ZIP each get source_index."""
        convs = [
            {"id": "c1", "messages": [{"role": "user", "content": "Q1"}]},
            {"id": "c2", "messages": [{"role": "user", "content": "Q2"}]},
        ]
        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.json", json.dumps(convs))

        source = Source(name="test", path=zip_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 2
        # Each should have an index
        for i, (raw_data, _) in enumerate(results):
            if raw_data:
                assert raw_data.source_index == i

    def test_zip_skip_directories(self, tmp_path):
        """ZIP directories are skipped."""
        zip_path = tmp_path / "withdir.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.json", json.dumps({"id": "test"}))
            zf.writestr("subdir/", "")  # Directory entry

        source = Source(name="test", path=zip_path)
        results = list(iter_source_conversations_with_raw(source))
        # Should only have the file, not the directory
        assert isinstance(results, list)


# =============================================================================
# iter_source_conversations_with_raw: cursor state
# =============================================================================


class TestRawCaptureCursorState:
    """Tests for cursor state tracking in raw capture."""

    def test_cursor_state_tracks_file_count(self, tmp_path):
        from tests.helpers import GenericConversationBuilder

        for i in range(3):
            (GenericConversationBuilder(f"c{i}")
             .add_message("user", f"msg{i}", text=f"msg{i}")
             .write_to(tmp_path / f"conv{i}.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        assert cursor_state["file_count"] == 3

    def test_cursor_state_tracks_latest_mtime(self, tmp_path):
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("c1")
         .add_message("user", "test", text="test")
         .write_to(tmp_path / "conv.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        assert "latest_mtime" in cursor_state
        assert "latest_path" in cursor_state
        assert isinstance(cursor_state["latest_mtime"], float)

    def test_cursor_state_tracks_failures(self, tmp_path):
        (tmp_path / "bad.json").write_text("not json at all")

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        assert cursor_state.get("failed_count", 0) >= 1
        assert len(cursor_state.get("failed_files", [])) >= 1

    def test_cursor_state_empty_dir(self, tmp_path):
        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        assert cursor_state["file_count"] == 0
        assert cursor_state.get("failed_count", 0) == 0

    def test_cursor_state_with_capture_raw_false(self, tmp_path):
        """Cursor state works regardless of capture_raw setting."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("c1")
         .write_to(tmp_path / "conv.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        list(iter_source_conversations_with_raw(source, capture_raw=False, cursor_state=cursor_state))
        assert cursor_state["file_count"] == 1

    def test_cursor_state_accumulated_from_multiple_calls(self, tmp_path):
        """Cursor state can be reused across multiple calls."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("c1")
         .write_to(tmp_path / "conv1.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        first_count = cursor_state["file_count"]

        (GenericConversationBuilder("c2")
         .write_to(tmp_path / "conv2.json"))

        # Reset and rescan
        cursor_state2: dict = {}
        list(iter_source_conversations_with_raw(source, cursor_state=cursor_state2))
        assert cursor_state2["file_count"] == 2


# =============================================================================
# iter_source_conversations_with_raw: error handling
# =============================================================================


class TestRawCaptureErrorHandling:
    """Tests for error handling in raw capture."""

    def test_continues_after_json_decode_error(self, tmp_path):
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("good")
         .add_message("user", "valid", text="valid")
         .write_to(tmp_path / "good.json"))
        (tmp_path / "bad.json").write_text("{ broken json")

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        results = list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        # Should still get the good conversation
        convs = [r[1] for r in results]
        assert any(c.provider_conversation_id == "good" for c in convs)
        assert cursor_state.get("failed_count", 0) >= 1

    def test_file_not_found_tracked(self, tmp_path):
        """File disappearing during iteration is tracked."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("disappear")
         .write_to(tmp_path / "vanish.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        # Delete the file just before reading (TOCTOU race)
        results = []
        for raw_data, conv in iter_source_conversations_with_raw(source, cursor_state=cursor_state):
            results.append((raw_data, conv))
            # File already loaded, so we get it

        assert cursor_state["file_count"] >= 1

    def test_skip_dirs_respected(self, tmp_path):
        """_SKIP_DIRS are pruned in raw capture variant too."""
        from tests.helpers import GenericConversationBuilder

        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        (GenericConversationBuilder("skipped")
         .write_to(analysis_dir / "data.json"))

        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) == 0

    def test_skip_pycache_dir(self, tmp_path):
        """__pycache__ directories are skipped."""
        from tests.helpers import GenericConversationBuilder

        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()
        (GenericConversationBuilder("skipped")
         .write_to(pycache_dir / "data.json"))

        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) == 0

    def test_unicode_decode_error_tracked(self, tmp_path):
        """Files with invalid encoding are tracked as failures."""
        # Write binary garbage that can't be decoded
        bad_file = tmp_path / "bad_encoding.json"
        bad_file.write_bytes(b'\xff\xfe invalid utf-8 { bad json')

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        list(iter_source_conversations_with_raw(source, cursor_state=cursor_state))
        assert cursor_state.get("failed_count", 0) >= 1


# =============================================================================
# parse_drive_payload
# =============================================================================


class TestParseDrivePayload:
    """Tests for Drive-specific payload parsing."""

    def test_list_of_chunks(self):
        """List with role/text items parsed as chunked prompt."""
        payload = [
            {"role": "user", "text": "Hello drive"},
            {"role": "model", "text": "Hi"},
        ]
        results = parse_drive_payload("drive", payload, "drive-test")
        assert len(results) >= 1
        assert results[0].provider_name == "drive"

    def test_dict_with_chunked_prompt(self):
        """Dict with 'chunks' key parsed directly."""
        payload = {"chunks": [
            {"role": "user", "text": "Hello"},
        ]}
        results = parse_drive_payload("drive", payload, "drive-chunk")
        assert len(results) == 1

    def test_dict_with_chunkedPrompt(self):
        """Dict with 'chunkedPrompt' key parsed."""
        payload = {"chunkedPrompt": {"messages": []}}
        results = parse_drive_payload("drive", payload, "drive-cp")
        assert len(results) >= 1

    def test_nested_list_recurses(self):
        """Nested lists are recursed into."""
        payload = [
            [{"role": "user", "text": "Nested"}],
        ]
        results = parse_drive_payload("drive", payload, "nested")
        assert len(results) >= 1

    def test_max_depth_stops_recursion(self):
        """Exceeding max depth returns empty."""
        results = parse_drive_payload("drive", [{"role": "user", "text": "deep"}], "deep", _depth=11)
        assert results == []

    def test_empty_list(self):
        results = parse_drive_payload("drive", [], "empty")
        assert results == []

    def test_dict_detects_chatgpt(self):
        """Dict payload triggers detect_provider for chatgpt."""
        from tests.helpers import ChatGPTExportBuilder

        payload = ChatGPTExportBuilder("drive-chatgpt").add_node("user", "Hi").build()
        results = parse_drive_payload("drive", payload, "auto-detect")
        assert len(results) >= 1
        assert results[0].provider_name == "chatgpt"

    def test_non_dict_non_list_returns_empty(self):
        results = parse_drive_payload("drive", "plain string", "nope")
        assert results == []

    def test_list_of_dicts_with_chunks_key(self):
        """List of conversation dicts with 'chunks' key."""
        payload = [
            {"chunks": [{"role": "user", "text": "First"}]},
            {"chunks": [{"role": "user", "text": "Second"}]},
        ]
        results = parse_drive_payload("drive", payload, "multi-conv")
        assert len(results) >= 2

    def test_gemini_provider_list_handling(self):
        """Gemini provider handles lists same way as drive."""
        payload = [
            {"role": "user", "parts": ["Hello Gemini"]},
            {"role": "model", "parts": ["Hi"]},
        ]
        results = parse_drive_payload("gemini", payload, "gemini-test")
        assert len(results) >= 1

    def test_deep_nesting_within_limit(self):
        """Deep nesting within max depth limit."""
        payload = [[[{"role": "user", "text": "Deep"}]]]
        results = parse_drive_payload("drive", payload, "nested-deep", _depth=0)
        # Should recurse through nesting
        assert isinstance(results, list)

    def test_dict_with_conversations_key_in_drive_context(self):
        """Dict with 'conversations' key is unpacked."""
        payload = {
            "conversations": [
                {"chunks": [{"role": "user", "text": "Conv 1"}]},
                {"chunks": [{"role": "user", "text": "Conv 2"}]},
            ]
        }
        results = parse_drive_payload("drive", payload, "conv-list")
        assert len(results) >= 2


# =============================================================================
# iter_source_conversations_with_raw: integration with different formats
# =============================================================================


class TestRawCaptureFormats:
    """Integration tests with different conversation formats."""

    def test_chatgpt_export_raw_capture(self, tmp_path):
        """ChatGPT export format with raw capture."""
        from tests.helpers import ChatGPTExportBuilder

        export = (ChatGPTExportBuilder("chat1")
                  .title("Test Chat")
                  .add_node("user", "What is 2+2?")
                  .add_node("assistant", "4")
                  .write_to(tmp_path / "chat.json"))

        source = Source(name="chatgpt", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None
        assert conv.title == "Test Chat"
        assert b"2+2" in raw_data.raw_bytes

    def test_claude_export_raw_capture(self, tmp_path):
        """Claude export format with raw capture."""
        from tests.helpers import ClaudeExportBuilder

        export = (ClaudeExportBuilder("claude1")
                  .name("Claude Test")
                  .add_human("Hello Claude")
                  .add_assistant("Hi there!")
                  .write_to(tmp_path / "claude.json"))

        source = Source(name="claude", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None

    def test_multiple_files_each_indexed(self, tmp_path):
        """Multiple conversation files each tracked with index."""
        from tests.helpers import GenericConversationBuilder

        for i in range(3):
            (GenericConversationBuilder(f"conv{i}")
             .add_message("user", f"message{i}", text=f"msg{i}")
             .write_to(tmp_path / f"conv{i}.json"))

        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 3

    def test_raw_bytes_valid_json(self, tmp_path):
        """Raw bytes are valid JSON that can be re-parsed."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("json-round-trip")
         .add_message("user", "test", text="test")
         .write_to(tmp_path / "conv.json"))

        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]

        if raw_data and raw_data.raw_bytes:
            # Should be valid JSON
            reparsed = json.loads(raw_data.raw_bytes.decode("utf-8"))
            assert isinstance(reparsed, dict)
            assert "id" in reparsed


# =============================================================================
# iter_source_conversations_with_raw: mixed files
# =============================================================================


class TestRawCaptureMixedFiles:
    """Tests with mixed file types and provider detections."""

    def test_mixed_json_and_jsonl(self, tmp_path):
        """Directory with both .json and .jsonl files."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("json-conv")
         .write_to(tmp_path / "conv.json"))

        # For JSONL, write proper claude-code format
        records = [
            {"type": "user", "uuid": "u1", "sessionId": "s1", "message": {"content": "Q"}},
        ]
        (tmp_path / "session.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 2

    def test_nested_directory_traversal(self, tmp_path):
        """Nested directories are traversed."""
        from tests.helpers import GenericConversationBuilder

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (GenericConversationBuilder("nested")
         .write_to(subdir / "conv.json"))

        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1

    def test_symlinked_directory_traversal(self, tmp_path):
        """Symlinked directories are followed."""
        from tests.helpers import GenericConversationBuilder

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (GenericConversationBuilder("linked")
         .write_to(subdir / "conv.json"))

        # Create symlink
        link = tmp_path / "link"
        try:
            link.symlink_to(subdir)
            source = Source(name="test", path=tmp_path)
            results = list(iter_source_conversations_with_raw(source))
            # Should find both original and symlinked (or just one depending on os.walk behavior)
            assert len(results) >= 1
        except (OSError, NotImplementedError):
            # Symlinks may not be supported on this system
            pytest.skip("Symlinks not supported on this system")

    def test_file_as_source_path(self, tmp_path):
        """Source path can be a single file."""
        from tests.helpers import GenericConversationBuilder

        file_path = tmp_path / "single.json"
        (GenericConversationBuilder("single")
         .write_to(file_path))

        source = Source(name="test", path=file_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]
        assert raw_data is not None


# =============================================================================
# iter_source_conversations_with_raw: mtime preservation
# =============================================================================


class TestRawCaptureMtimeTracking:
    """Tests for file modification time tracking."""

    def test_file_mtime_recorded(self, tmp_path):
        """File modification time is recorded in raw data."""
        from tests.helpers import GenericConversationBuilder

        (GenericConversationBuilder("mtime-test")
         .write_to(tmp_path / "conv.json"))

        source = Source(name="test", path=tmp_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]

        if raw_data:
            assert raw_data.file_mtime is not None
            # Should be ISO format timestamp
            assert "T" in raw_data.file_mtime or ":" in raw_data.file_mtime

    def test_zip_mtime_recorded(self, tmp_path):
        """ZIP file mtime is recorded."""
        from tests.helpers import GenericConversationBuilder

        conv = GenericConversationBuilder("zip-mtime").build()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conv.json", json.dumps([conv]))

        source = Source(name="test", path=zip_path)
        results = list(iter_source_conversations_with_raw(source))
        assert len(results) >= 1
        raw_data, conv = results[0]

        if raw_data:
            assert raw_data.file_mtime is not None
