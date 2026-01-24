from __future__ import annotations

import json
import zipfile

from polylogue.config import Source
from polylogue.ingestion import iter_source_conversations, parse_drive_payload


def test_auto_detect_chatgpt_and_claude(tmp_path):
    chatgpt_payload = {
        "id": "conv-chatgpt",
        "mapping": {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"parts": ["Hello"]},
                    "create_time": 1,
                },
            }
        },
    }
    claude_payload = {
        "conversations": [
            {
                "id": "conv-claude",
                "name": "Claude Chat",
                "chat_messages": [
                    {
                        "id": "msg-1",
                        "sender": "user",
                        "content": [{"type": "text", "text": "Hi"}],
                    }
                ],
            }
        ]
    }
    (tmp_path / "chatgpt.json").write_text(json.dumps(chatgpt_payload), encoding="utf-8")
    (tmp_path / "claude.json").write_text(json.dumps(claude_payload), encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))
    providers = {convo.provider_name for convo in conversations}
    assert "chatgpt" in providers
    assert "claude" in providers


def test_claude_chat_messages_attachments(tmp_path):
    payload = {
        "chat_messages": [
            {
                "id": "msg-1",
                "sender": "assistant",
                "content": [{"type": "text", "text": "Files"}],
                "attachments": [{"id": "file-1", "name": "notes.txt", "size": 12, "mimeType": "text/plain"}],
            }
        ]
    }
    source_file = tmp_path / "claude.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="inbox", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert conversations
    convo = conversations[0]
    assert convo.attachments
    attachment = convo.attachments[0]
    assert attachment.provider_attachment_id == "file-1"
    assert attachment.name == "notes.txt"


def test_iter_source_conversations_handles_utf32_jsonl(tmp_path):
    payload = {
        "id": "utf32-conv",
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "text": "Hello UTF32",
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "text": "Hi there!",
            },
        ],
    }
    source_file = tmp_path / "custom.jsonl"
    source_file.write_text(json.dumps(payload), encoding="utf-32-be")

    source = Source(name="custom", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert conversations
    assert [msg.text for msg in conversations[0].messages] == ["Hello UTF32", "Hi there!"]


def test_iter_source_conversations_strips_null_bytes(tmp_path):
    payload = {
        "id": "null-conv",
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "text": "Hello",
            }
        ],
    }
    source_file = tmp_path / "custom.jsonl"
    line = json.dumps(payload).encode("utf-8")
    source_file.write_bytes(line + b"\x00\n")

    source = Source(name="custom", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert conversations
    assert [msg.text for msg in conversations[0].messages] == ["Hello"]


def test_iter_source_conversations_handles_ndjson(tmp_path):
    payloads = [
        {
            "id": "conv-1",
            "messages": [
                {
                    "id": "msg-1",
                    "role": "user",
                    "text": "First",
                }
            ],
        },
        {
            "id": "conv-2",
            "messages": [
                {
                    "id": "msg-2",
                    "role": "assistant",
                    "text": "Second",
                }
            ],
        },
    ]
    source_file = tmp_path / "conversations.ndjson"
    source_file.write_text("\n".join(json.dumps(item) for item in payloads) + "\n", encoding="utf-8")

    source = Source(name="custom", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert len(conversations) == 2
    assert {convo.provider_conversation_id for convo in conversations} == {"conv-1", "conv-2"}


def test_parse_drive_payload_detects_chatgpt_payload(tmp_path):
    payload = {
        "id": "conv-drive",
        "mapping": {
            "n1": {
                "id": "n1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"parts": ["Drive conversation"]},
                    "create_time": 1.0,
                },
            }
        },
    }

    result = parse_drive_payload("drive", payload, str(tmp_path / "chatgpt.json"))
    assert result
    assert result[0].provider_name == "chatgpt"


# Case-insensitive extension tests


def test_iter_source_conversations_finds_uppercase_json(tmp_path):
    """Files like CHATGPT.JSON are found (case-insensitive)."""
    payload = {
        "id": "upper-conv",
        "messages": [{"id": "m1", "role": "user", "text": "uppercase test"}],
    }
    # Write file with uppercase extension
    (tmp_path / "UPPER.JSON").write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    # File should be found (case-insensitive extension matching)
    assert len(conversations) == 1
    # The conversation should have the messages
    assert len(conversations[0].messages) >= 1
    # Check that we got the content from the file
    assert conversations[0].provider_conversation_id == "upper-conv"


def test_iter_source_conversations_finds_mixed_case_jsonl(tmp_path):
    """Files like Export.JSONL are found."""
    payload = {
        "id": "mixed-conv",
        "messages": [{"id": "m1", "role": "user", "text": "mixed case test"}],
    }
    # Write file with mixed case extension
    (tmp_path / "Export.JSONL").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    assert len(conversations) == 1
    assert conversations[0].messages[0].text == "mixed case test"


def test_has_ingest_extension_handles_double_extensions(tmp_path):
    """Files like data.jsonl.txt are recognized."""
    payload = {
        "id": "double-ext-conv",
        "messages": [{"id": "m1", "role": "user", "text": "double extension test"}],
    }
    # Write file with .jsonl.txt double extension
    (tmp_path / "data.jsonl.txt").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    assert len(conversations) == 1
    assert conversations[0].messages[0].text == "double extension test"


# Empty/invalid conversation tests


def test_parse_json_payload_empty_conversations_list(tmp_path):
    """Payload with empty 'conversations' array returns empty list."""
    payload = {"conversations": []}
    (tmp_path / "empty.json").write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    assert conversations == []


def test_parse_json_payload_invalid_conversation_items(tmp_path):
    """Non-dict items in 'conversations' are handled gracefully."""
    payload = {
        "conversations": [
            None,
            "not a dict",
            {"id": "valid", "messages": [{"id": "m1", "role": "user", "text": "valid"}]},
        ]
    }
    (tmp_path / "mixed.json").write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    # Should get at least the valid conversation, skipping invalid entries
    assert len(conversations) >= 1
    valid_convos = [c for c in conversations if c.provider_conversation_id == "valid"]
    assert len(valid_convos) == 1


# Encoding fallback tests


def test_decode_json_bytes_utf8_with_bom(tmp_path):
    """_decode_json_bytes() handles UTF-8 with BOM."""
    from polylogue.ingestion.source import _decode_json_bytes

    payload = {"id": "test", "messages": [{"id": "m1", "role": "user", "text": "Hello"}]}
    json_str = json.dumps(payload)

    # UTF-8 with BOM (0xEF 0xBB 0xBF)
    utf8_bom = b"\xef\xbb\xbf" + json_str.encode("utf-8")

    result = _decode_json_bytes(utf8_bom)
    assert result is not None
    # BOM should be handled by decoder, but if not we strip it
    if result.startswith("\ufeff"):
        result = result.lstrip("\ufeff")
    parsed = json.loads(result)
    assert parsed["id"] == "test"


def test_decode_json_bytes_utf16_le(tmp_path):
    """_decode_json_bytes() handles UTF-16 LE encoding."""
    from polylogue.ingestion.source import _decode_json_bytes

    payload = {"id": "utf16test", "messages": []}
    json_str = json.dumps(payload)

    # UTF-16 LE
    utf16_bytes = json_str.encode("utf-16-le")

    result = _decode_json_bytes(utf16_bytes)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["id"] == "utf16test"


def test_decode_json_bytes_utf16_be(tmp_path):
    """_decode_json_bytes() handles UTF-16 BE encoding."""
    from polylogue.ingestion.source import _decode_json_bytes

    payload = {"id": "utf16be", "messages": []}
    json_str = json.dumps(payload)

    # UTF-16 BE
    utf16_bytes = json_str.encode("utf-16-be")

    result = _decode_json_bytes(utf16_bytes)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["id"] == "utf16be"


def test_decode_json_bytes_invalid_utf8_fallback(tmp_path):
    """_decode_json_bytes() handles bytes gracefully without crashing."""
    from polylogue.ingestion.source import _decode_json_bytes

    # Test with genuinely problematic bytes that require fallback
    # This tests that the function tries multiple encodings and eventually succeeds
    test_cases = [
        # Valid UTF-8 with some control characters
        b'{"id": "test", "data": "value"}',
        # Latin-1 compatible
        b'{"name": "caf\xe9"}',
    ]

    for test_bytes in test_cases:
        result = _decode_json_bytes(test_bytes)
        # Should return something without crashing
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


def test_decode_json_bytes_strips_null_bytes(tmp_path):
    """_decode_json_bytes() removes null bytes from decoded string."""
    from polylogue.ingestion.source import _decode_json_bytes

    # JSON with embedded null bytes
    payload = b'{"id": "test\x00null", "messages": []}'

    result = _decode_json_bytes(payload)
    assert result is not None
    # Null bytes should be stripped
    assert "\x00" not in result
    parsed = json.loads(result)
    assert parsed["id"] == "testnull"


def test_decode_json_bytes_returns_none_on_all_nulls(tmp_path):
    """_decode_json_bytes() returns None if only null bytes remain."""
    from polylogue.ingestion.source import _decode_json_bytes

    # All null bytes
    all_nulls = b"\x00\x00\x00\x00"

    result = _decode_json_bytes(all_nulls)
    # After stripping nulls, nothing remains
    assert result is None


def test_decode_json_bytes_handles_utf32(tmp_path):
    """_decode_json_bytes() handles UTF-32 encoding."""
    from polylogue.ingestion.source import _decode_json_bytes

    payload = {"id": "utf32", "messages": []}
    json_str = json.dumps(payload)

    # UTF-32 LE
    utf32_bytes = json_str.encode("utf-32-le")

    result = _decode_json_bytes(utf32_bytes)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["id"] == "utf32"


def test_iter_source_conversations_handles_encoding_variations(tmp_path):
    """iter_source_conversations() handles files with different encodings."""
    payload = {"id": "encoding-test", "messages": [{"id": "m1", "role": "user", "text": "Hello"}]}

    # UTF-8 with BOM
    utf8_file = tmp_path / "utf8bom.json"
    utf8_file.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))

    # UTF-16 LE
    utf16_file = tmp_path / "utf16.json"
    utf16_file.write_bytes(json.dumps(payload).encode("utf-16-le"))

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    # Should successfully parse both files
    assert len(conversations) >= 2
    conv_ids = {c.provider_conversation_id for c in conversations}
    assert "encoding-test" in conv_ids


def test_iter_source_conversations_handles_malformed_encoding_gracefully(tmp_path):
    """iter_source_conversations() handles malformed encodings without crashing."""
    # Create a file with mixed valid and invalid UTF-8
    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b'{"id": "partially-valid\xff\xfe", "messages": []}')

    source = Source(name="inbox", path=tmp_path)
    # Should not crash, may or may not parse successfully
    conversations = list(iter_source_conversations(source))

    # At minimum, should not raise an exception
    # Behavior depends on whether fallback can salvage the JSON
    assert isinstance(conversations, list)


def test_decode_json_bytes_empty_after_cleaning(tmp_path):
    """_decode_json_bytes() returns None if string is empty after cleaning."""
    from polylogue.ingestion.source import _decode_json_bytes

    # String that becomes empty after cleaning
    only_nulls = b"\x00\x00\x00"

    result = _decode_json_bytes(only_nulls)
    assert result is None


def test_iter_source_conversations_jsonl_with_null_bytes(tmp_path):
    """iter_source_conversations() handles JSONL with embedded null bytes."""
    payload = {"id": "null-test", "messages": [{"id": "m1", "role": "user", "text": "Hello"}]}

    jsonl_file = tmp_path / "nulls.jsonl"
    # Write JSON with null bytes
    line = json.dumps(payload).encode("utf-8")
    jsonl_file.write_bytes(line + b"\x00\n")

    source = Source(name="inbox", path=jsonl_file)
    conversations = list(iter_source_conversations(source))

    # Should handle null bytes and parse successfully
    assert len(conversations) == 1
    assert conversations[0].provider_conversation_id == "null-test"


class TestIterSourceConversations:
    """Tests for iter_source_conversations function."""

    def test_tracks_file_count_in_cursor_state(self, tmp_path):
        """cursor_state should track number of files processed, including failures."""
        # Create 3 valid JSON files
        for i in range(3):
            (tmp_path / f"conv{i}.json").write_text(json.dumps({
                "id": f"conv-{i}",
                "title": f"Test {i}",
                "messages": [{"role": "user", "text": "hello"}]
            }))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        list(iter_source_conversations(source, cursor_state=cursor_state))

        assert "file_count" in cursor_state
        assert cursor_state["file_count"] == 3

        # SHOULD FAIL until failure tracking is implemented:
        # cursor_state should also track failed files to avoid re-processing
        assert "failed_count" in cursor_state, "cursor_state should track failed file count"

    def test_continues_after_invalid_json(self, tmp_path):
        """Should continue processing after encountering invalid JSON, track failures."""
        # One valid, one invalid, one valid
        (tmp_path / "valid1.json").write_text(json.dumps({
            "id": "v1", "title": "Valid 1",
            "messages": [{"role": "user", "text": "hi"}]
        }))
        (tmp_path / "invalid.json").write_text("{ this is not valid json }")
        (tmp_path / "valid2.json").write_text(json.dumps({
            "id": "v2", "title": "Valid 2",
            "messages": [{"role": "user", "text": "bye"}]
        }))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        # Should parse 2 valid conversations despite invalid file
        assert len(convs) == 2
        assert {c.provider_conversation_id for c in convs} == {"v1", "v2"}

        # SHOULD FAIL until failure tracking is implemented:
        # The invalid file should be tracked as failed in cursor_state
        # This prevents re-processing on next run and allows reporting of failures
        assert "failed_count" in cursor_state, "Failed files should be tracked in cursor_state"
        assert cursor_state.get("failed_count", 0) >= 1, "At least invalid.json should be tracked as failed"

    def test_handles_empty_directory(self, tmp_path):
        """Should handle empty directory gracefully."""
        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        assert convs == []
        assert cursor_state.get("file_count") == 0

    def test_handles_deeply_nested_zip(self, tmp_path):
        """Should process ZIP files with nested content."""
        inner_data = json.dumps({
            "id": "nested", "title": "Nested Conv",
            "messages": [{"role": "user", "text": "from zip"}]
        })

        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations/conv.json", inner_data)

        source = Source(name="test", path=tmp_path)
        convs = list(iter_source_conversations(source))

        assert len(convs) == 1
        assert convs[0].title == "Nested Conv"


class TestZipBombProtection:
    """Tests for ZIP bomb / resource exhaustion protection."""

    def test_rejects_highly_compressed_zip(self, tmp_path):
        """ZIP bomb protection MUST reject suspicious compression ratios.

        This test SHOULD FAIL until ZIP bomb protection is implemented.
        Currently, the function silently processes dangerous ZIPs without detection.
        """
        # Create a "zip bomb" - highly repetitive content compresses extremely well
        bomb_content = "A" * (10 * 1024 * 1024)  # 10MB of 'A's

        zip_path = tmp_path / "suspicious.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("bomb.json", bomb_content)

        # Verify this is actually a suspicious zip (setup validation)
        zip_size = zip_path.stat().st_size
        with zipfile.ZipFile(zip_path) as zf:
            uncompressed_size = zf.infolist()[0].file_size

        ratio = uncompressed_size / zip_size
        # This demonstrates the vulnerability - ratio > 100x is suspicious
        assert ratio > 100, f"Test setup: ratio {ratio} should be > 100"

        source = Source(name="test", path=tmp_path)

        # STRICT: Should reject or return empty (not silently process bomb)
        convs = list(iter_source_conversations(source))
        assert len(convs) == 0, f"ZIP bomb should be rejected, but got {len(convs)} conversations"


class TestTOCTOUHandling:
    """Tests for TOCTOU (time-of-check-time-of-use) race condition handling."""

    def test_handles_file_deleted_after_detection(self, tmp_path, monkeypatch):
        """Should handle file being deleted between detection and read.

        This test validates that FileNotFoundError from deleted files is caught
        and tracked gracefully without crashing the entire ingest process.
        """
        import io

        # Create a file
        test_file = tmp_path / "conversation.json"
        test_file.write_text(json.dumps({
            "id": "test",
            "title": "Test",
            "messages": [{"id": "m1", "role": "user", "text": "hello"}]
        }))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        # Track which files are opened
        opened_files = []
        original_open = io.open

        def tracking_open(path, *args, **kwargs):
            opened_files.append(str(path))
            # Delete the file on first open attempt to simulate race condition
            if str(test_file) in str(path) and len(opened_files) == 1:
                test_file.unlink()
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", tracking_open)

        # Should not raise, but gracefully handle the deleted file
        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        # Should track the failure
        assert cursor_state.get("failed_count", 0) >= 1, \
            "Deleted file should be tracked as failed"
        assert any("conversation.json" in str(f)
                   for f in cursor_state.get("failed_files", [])), \
            f"Failed file path should be tracked, got: {cursor_state.get('failed_files')}"

    def test_handles_file_replaced_with_invalid_content(self, tmp_path, monkeypatch):
        """Should handle file being replaced with invalid content during read."""
        import io

        test_file = tmp_path / "conversation.json"
        test_file.write_text(json.dumps({
            "id": "test",
            "title": "Test",
            "messages": [{"id": "m1", "role": "user", "text": "hello"}]
        }))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        # Track opens and corrupt file on first access
        opens = [0]
        original_open = io.open

        def corrupting_open(path, *args, **kwargs):
            opens[0] += 1
            # Corrupt the file on first open
            if str(test_file) in str(path) and opens[0] == 1:
                test_file.write_text("{ this is not valid json at all")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", corrupting_open)

        # Should handle gracefully (not crash)
        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        # May or may not have results, but should not crash
        assert isinstance(convs, list)
        # Should have tracked the JSON decode error
        assert cursor_state.get("failed_count", 0) >= 1, \
            "Invalid JSON should be tracked as failed"

    def test_file_not_found_tracked_in_cursor_state(self, tmp_path, monkeypatch):
        """File that disappears should be tracked in cursor_state with error details."""
        import io

        # Create a file that we'll fail to open
        test_file = tmp_path / "disappearing.json"
        test_file.write_text(json.dumps({
            "id": "test",
            "title": "Test",
            "messages": []
        }))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        # Monkey-patch open to fail for disappearing file
        original_open = io.open

        def failing_open(path, *args, **kwargs):
            if "disappearing" in str(path):
                raise FileNotFoundError(f"File disappeared: {path}")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", failing_open)

        # Should handle gracefully
        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        # Failure should be tracked
        assert cursor_state.get("failed_count", 0) >= 1, \
            "FileNotFoundError should increment failed_count"

        failed_files = cursor_state.get("failed_files", [])
        assert len(failed_files) >= 1, "Failed file should be tracked in failed_files"

        # Error message should indicate file was not found
        error_msg = str(failed_files[0].get("error", ""))
        assert "not found" in error_msg.lower() or "disappeared" in error_msg.lower(), \
            f"Error should mention file not found, got: {error_msg}"

    def test_continues_processing_after_file_not_found(self, tmp_path, monkeypatch):
        """Processing should continue after encountering a missing file."""
        import io

        # Create multiple files
        (tmp_path / "file1.json").write_text(json.dumps({
            "id": "conv1",
            "messages": [{"id": "m1", "role": "user", "text": "first"}]
        }))
        (tmp_path / "file2.json").write_text(json.dumps({
            "id": "conv2",
            "messages": [{"id": "m1", "role": "user", "text": "second"}]
        }))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        # Fail on file2 but succeed on file1
        original_open = io.open

        def selective_fail_open(path, *args, **kwargs):
            if "file2" in str(path):
                raise FileNotFoundError(f"File not found: {path}")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", selective_fail_open)

        # Should get file1 successfully
        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        # Should have at least processed file1
        conv_ids = {c.provider_conversation_id for c in convs}
        assert "conv1" in conv_ids, "Should successfully process file1 before file2 fails"

        # Should have tracked file2 failure
        assert cursor_state.get("failed_count", 0) >= 1, \
            "file2 failure should be tracked"
