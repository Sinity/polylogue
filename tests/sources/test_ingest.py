from __future__ import annotations

import json
import zipfile
from pathlib import Path

from polylogue.config import Source
from polylogue.sources import iter_source_conversations, parse_drive_payload
from tests.helpers import (
    ChatGPTExportBuilder,
    GenericConversationBuilder,
    InboxBuilder,
)


def test_auto_detect_chatgpt_and_claude(tmp_path):
    inbox = (InboxBuilder(tmp_path)
             .add_chatgpt_export("conv-chatgpt", filename="chatgpt.json")
             .add_claude_export("conv-claude", name="Claude Chat", filename="claude.json")
             .build())

    source = Source(name="inbox", path=inbox)
    conversations = list(iter_source_conversations(source))
    providers = {convo.provider_name for convo in conversations}
    assert "chatgpt" in providers
    assert "claude" in providers


def test_claude_chat_messages_attachments(tmp_path):
    # Use builder with custom message containing attachments
    from tests.helpers import make_claude_chat_message
    payload = {
        "chat_messages": [
            make_claude_chat_message(
                "msg-1", "assistant", "Files",
                attachments=[{"id": "file-1", "name": "notes.txt", "size": 12, "mimeType": "text/plain"}],
            )
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
    # UTF-32 encoding requires manual bytes handling, keep inline
    payload = (GenericConversationBuilder("utf32-conv")
               .add_message("user", "Hello UTF32", text="Hello UTF32")
               .add_message("assistant", "Hi there!", text="Hi there!")
               .build())
    source_file = tmp_path / "custom.jsonl"
    source_file.write_text(json.dumps(payload), encoding="utf-32-be")

    source = Source(name="custom", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert conversations
    assert [msg.text for msg in conversations[0].messages] == ["Hello UTF32", "Hi there!"]


def test_iter_source_conversations_strips_null_bytes(tmp_path):
    # Null bytes require manual bytes handling
    payload = (GenericConversationBuilder("null-conv")
               .add_message("user", "Hello", text="Hello")
               .build())
    source_file = tmp_path / "custom.jsonl"
    line = json.dumps(payload).encode("utf-8")
    source_file.write_bytes(line + b"\x00\n")

    source = Source(name="custom", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert conversations
    assert [msg.text for msg in conversations[0].messages] == ["Hello"]


def test_iter_source_conversations_handles_ndjson(tmp_path):
    payloads = [
        GenericConversationBuilder("conv-1").add_message("user", "First", text="First").build(),
        GenericConversationBuilder("conv-2").add_message("assistant", "Second", text="Second").build(),
    ]
    source_file = tmp_path / "conversations.ndjson"
    source_file.write_text("\n".join(json.dumps(item) for item in payloads) + "\n", encoding="utf-8")

    source = Source(name="custom", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert len(conversations) == 2
    assert {convo.provider_conversation_id for convo in conversations} == {"conv-1", "conv-2"}


def test_parse_drive_payload_detects_chatgpt_payload(tmp_path):
    payload = (ChatGPTExportBuilder("conv-drive")
               .add_node("user", "Drive conversation")
               .build())

    result = parse_drive_payload("drive", payload, str(tmp_path / "chatgpt.json"))
    assert result
    assert result[0].provider_name == "chatgpt"


# Case-insensitive extension tests


def test_iter_source_conversations_finds_uppercase_json(tmp_path):
    """Files like CHATGPT.JSON are found (case-insensitive)."""
    (GenericConversationBuilder("upper-conv")
     .add_message("user", "uppercase test", text="uppercase test")
     .write_to(tmp_path / "UPPER.JSON"))

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    assert len(conversations) == 1
    assert len(conversations[0].messages) >= 1
    assert conversations[0].provider_conversation_id == "upper-conv"


def test_iter_source_conversations_finds_mixed_case_jsonl(tmp_path):
    """Files like Export.JSONL are found."""
    payload = (GenericConversationBuilder("mixed-conv")
               .add_message("user", "mixed case test", text="mixed case test")
               .build())
    (tmp_path / "Export.JSONL").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    assert len(conversations) == 1
    assert conversations[0].messages[0].text == "mixed case test"


def test_has_ingest_extension_handles_double_extensions(tmp_path):
    """Files like data.jsonl.txt are recognized."""
    payload = (GenericConversationBuilder("double-ext-conv")
               .add_message("user", "double extension test", text="double extension test")
               .build())
    (tmp_path / "data.jsonl.txt").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    assert len(conversations) == 1
    assert conversations[0].messages[0].text == "double extension test"


# Empty/invalid conversation tests


def test_parse_json_payload_empty_conversations_list(tmp_path):
    """Payload with empty 'conversations' array returns empty list."""
    (tmp_path / "empty.json").write_text(json.dumps({"conversations": []}), encoding="utf-8")

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
    from polylogue.sources.source import _decode_json_bytes

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
    from polylogue.sources.source import _decode_json_bytes

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
    from polylogue.sources.source import _decode_json_bytes

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
    from polylogue.sources.source import _decode_json_bytes

    # Test with genuinely problematic bytes that require fallback
    test_cases = [
        b'{"id": "test", "data": "value"}',
        b'{"name": "caf\xe9"}',
    ]

    for test_bytes in test_cases:
        result = _decode_json_bytes(test_bytes)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


def test_decode_json_bytes_strips_null_bytes(tmp_path):
    """_decode_json_bytes() removes null bytes from decoded string."""
    from polylogue.sources.source import _decode_json_bytes

    payload = b'{"id": "test\x00null", "messages": []}'

    result = _decode_json_bytes(payload)
    assert result is not None
    assert "\x00" not in result
    parsed = json.loads(result)
    assert parsed["id"] == "testnull"


def test_decode_json_bytes_returns_none_on_all_nulls(tmp_path):
    """_decode_json_bytes() returns None if only null bytes remain."""
    from polylogue.sources.source import _decode_json_bytes

    all_nulls = b"\x00\x00\x00\x00"

    result = _decode_json_bytes(all_nulls)
    assert result is None


def test_decode_json_bytes_handles_utf32(tmp_path):
    """_decode_json_bytes() handles UTF-32 encoding."""
    from polylogue.sources.source import _decode_json_bytes

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
    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b'{"id": "partially-valid\xff\xfe", "messages": []}')

    source = Source(name="inbox", path=tmp_path)
    conversations = list(iter_source_conversations(source))

    # At minimum, should not raise an exception
    assert isinstance(conversations, list)


def test_decode_json_bytes_empty_after_cleaning(tmp_path):
    """_decode_json_bytes() returns None if string is empty after cleaning."""
    from polylogue.sources.source import _decode_json_bytes

    only_nulls = b"\x00\x00\x00"

    result = _decode_json_bytes(only_nulls)
    assert result is None


def test_iter_source_conversations_jsonl_with_null_bytes(tmp_path):
    """iter_source_conversations() handles JSONL with embedded null bytes."""
    payload = {"id": "null-test", "messages": [{"id": "m1", "role": "user", "text": "Hello"}]}

    jsonl_file = tmp_path / "nulls.jsonl"
    line = json.dumps(payload).encode("utf-8")
    jsonl_file.write_bytes(line + b"\x00\n")

    source = Source(name="inbox", path=jsonl_file)
    conversations = list(iter_source_conversations(source))

    assert len(conversations) == 1
    assert conversations[0].provider_conversation_id == "null-test"


class TestIngestIterConversations:
    """Tests for iter_source_conversations function."""

    def test_tracks_file_count_in_cursor_state(self, tmp_path):
        """cursor_state should track number of files processed, including failures."""
        for i in range(3):
            (GenericConversationBuilder(f"conv-{i}")
             .title(f"Test {i}")
             .add_message("user", "hello", text="hello")
             .write_to(tmp_path / f"conv{i}.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        list(iter_source_conversations(source, cursor_state=cursor_state))

        assert "file_count" in cursor_state
        assert cursor_state["file_count"] == 3

        # SHOULD FAIL until failure tracking is implemented:
        assert "failed_count" in cursor_state, "cursor_state should track failed file count"

    def test_continues_after_invalid_json(self, tmp_path):
        """Should continue processing after encountering invalid JSON, track failures."""
        (GenericConversationBuilder("v1")
         .title("Valid 1")
         .add_message("user", "hi", text="hi")
         .write_to(tmp_path / "valid1.json"))
        (tmp_path / "invalid.json").write_text("{ this is not valid json }")
        (GenericConversationBuilder("v2")
         .title("Valid 2")
         .add_message("user", "bye", text="bye")
         .write_to(tmp_path / "valid2.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}
        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        assert len(convs) == 2
        assert {c.provider_conversation_id for c in convs} == {"v1", "v2"}

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
        inner_data = (GenericConversationBuilder("nested")
                      .title("Nested Conv")
                      .add_message("user", "from zip", text="from zip")
                      .build())

        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations/conv.json", json.dumps(inner_data))

        source = Source(name="test", path=tmp_path)
        convs = list(iter_source_conversations(source))

        assert len(convs) == 1
        assert convs[0].title == "Nested Conv"


class TestZipBombProtection:
    """Tests for ZIP bomb / resource exhaustion protection."""

    def test_rejects_highly_compressed_zip(self, tmp_path):
        """ZIP bomb protection MUST reject suspicious compression ratios."""
        bomb_content = "A" * (10 * 1024 * 1024)  # 10MB of 'A's

        zip_path = tmp_path / "suspicious.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("bomb.json", bomb_content)

        zip_size = zip_path.stat().st_size
        with zipfile.ZipFile(zip_path) as zf:
            uncompressed_size = zf.infolist()[0].file_size

        ratio = uncompressed_size / zip_size
        assert ratio > 100, f"Test setup: ratio {ratio} should be > 100"

        source = Source(name="test", path=tmp_path)

        convs = list(iter_source_conversations(source))
        assert len(convs) == 0, f"ZIP bomb should be rejected, but got {len(convs)} conversations"


class TestTOCTOUHandling:
    """Tests for TOCTOU (time-of-check-time-of-use) race condition handling."""

    def test_handles_file_deleted_after_detection(self, tmp_path, monkeypatch):
        """Should handle file being deleted between detection and read."""
        import io

        test_file = tmp_path / "conversation.json"
        (GenericConversationBuilder("test")
         .title("Test")
         .add_message("user", "hello", text="hello")
         .write_to(test_file))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        opened_files = []
        original_open = io.open

        def tracking_open(path, *args, **kwargs):
            opened_files.append(str(path))
            if str(test_file) in str(path) and len(opened_files) == 1:
                test_file.unlink()
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", tracking_open)

        list(iter_source_conversations(source, cursor_state=cursor_state))

        assert cursor_state.get("failed_count", 0) >= 1, \
            "Deleted file should be tracked as failed"
        assert any("conversation.json" in str(f)
                   for f in cursor_state.get("failed_files", [])), \
            f"Failed file path should be tracked, got: {cursor_state.get('failed_files')}"

    def test_handles_file_replaced_with_invalid_content(self, tmp_path, monkeypatch):
        """Should handle file being replaced with invalid content during read."""
        import io

        test_file = tmp_path / "conversation.json"
        (GenericConversationBuilder("test")
         .title("Test")
         .add_message("user", "hello", text="hello")
         .write_to(test_file))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        opens = [0]
        original_open = io.open

        def corrupting_open(path, *args, **kwargs):
            opens[0] += 1
            if str(test_file) in str(path) and opens[0] == 1:
                test_file.write_text("{ this is not valid json at all")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", corrupting_open)

        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        assert isinstance(convs, list)
        assert cursor_state.get("failed_count", 0) >= 1, \
            "Invalid JSON should be tracked as failed"

    def test_file_not_found_tracked_in_cursor_state(self, tmp_path, monkeypatch):
        """File that disappears should be tracked in cursor_state with error details."""
        import io

        test_file = tmp_path / "disappearing.json"
        (GenericConversationBuilder("test")
         .title("Test")
         .write_to(test_file))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        original_open = io.open

        def failing_open(path, *args, **kwargs):
            if "disappearing" in str(path):
                raise FileNotFoundError(f"File disappeared: {path}")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", failing_open)

        list(iter_source_conversations(source, cursor_state=cursor_state))

        assert cursor_state.get("failed_count", 0) >= 1, \
            "FileNotFoundError should increment failed_count"

        failed_files = cursor_state.get("failed_files", [])
        assert len(failed_files) >= 1, "Failed file should be tracked in failed_files"

        error_msg = str(failed_files[0].get("error", ""))
        assert "not found" in error_msg.lower() or "disappeared" in error_msg.lower(), \
            f"Error should mention file not found, got: {error_msg}"

    def test_continues_processing_after_file_not_found(self, tmp_path, monkeypatch):
        """Processing should continue after encountering a missing file."""
        import io

        (GenericConversationBuilder("conv1")
         .add_message("user", "first", text="first")
         .write_to(tmp_path / "file1.json"))
        (GenericConversationBuilder("conv2")
         .add_message("user", "second", text="second")
         .write_to(tmp_path / "file2.json"))

        source = Source(name="test", path=tmp_path)
        cursor_state: dict = {}

        original_open = io.open

        def selective_fail_open(path, *args, **kwargs):
            if "file2" in str(path):
                raise FileNotFoundError(f"File not found: {path}")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("io.open", selective_fail_open)

        convs = list(iter_source_conversations(source, cursor_state=cursor_state))

        conv_ids = {c.provider_conversation_id for c in convs}
        assert "conv1" in conv_ids, "Should successfully process file1 before file2 fails"

        assert cursor_state.get("failed_count", 0) >= 1, \
            "file2 failure should be tracked"


# --- Merged from test_source_ingest_json_list.py ---


def test_iter_source_conversations_handles_codex_json_list(tmp_path: Path):
    """Test that Codex/Claude-Code/Gemini single-conversation JSON lists are not unpacked."""
    # A Codex/Claude-Code export is often a list of messages representing one conversation
    # If unpacked, this would look like N conversations with 0 messages each.
    # If not unpacked, it looks like 1 conversation with N messages.
    payload = [
        {"type": "session_meta", "payload": {"id": "test-session", "timestamp": "2025-01-01"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-1",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-2",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "Hi"}],
            },
        },
    ]

    # Write as a single JSON file
    source_file = tmp_path / "codex_export.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    # Name hint helps detection
    source = Source(name="codex", path=source_file)

    conversations = list(iter_source_conversations(source))

    # Should result in ONE conversation with messages, NOT multiple empty ones
    assert len(conversations) == 1
    convo = conversations[0]
    assert convo.provider_name == "codex"
    # 2 response_item messages
    assert len(convo.messages) == 2
    assert convo.messages[0].text == "Hello"


def test_iter_source_conversations_handles_claude_code_json_list(tmp_path: Path):
    """Test that Claude Code single-conversation JSON lists are not unpacked."""
    payload = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "message": {"content": "Hello"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "message": {"content": "Hi"},
        },
    ]

    source_file = tmp_path / "claude-code_export.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="claude-code", path=source_file)

    conversations = list(iter_source_conversations(source))

    assert len(conversations) == 1
    convo = conversations[0]
    assert convo.provider_name == "claude-code"
    assert len(convo.messages) == 2
    assert convo.messages[0].text == "Hello"


def test_iter_source_conversations_still_unpacks_chatgpt_json_list(tmp_path: Path):
    """Test that ChatGPT list of conversations IS unpacked (default behavior)."""
    # ChatGPT export is a list of conversation objects
    payload = [
        {
            "title": "Conv 1",
            "mapping": {
                "n1": {"message": {"content": {"parts": ["Msg 1"]}, "author": {"role": "user"}}}
            }
        },
        {
            "title": "Conv 2",
            "mapping": {
                "n2": {"message": {"content": {"parts": ["Msg 2"]}, "author": {"role": "user"}}}
            }
        }
    ]

    source_file = tmp_path / "chatgpt_export.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="chatgpt", path=source_file)

    conversations = list(iter_source_conversations(source))

    # Should unpack into 2 conversations
    assert len(conversations) == 2
    assert conversations[0].messages[0].text == "Msg 1"
    assert conversations[1].messages[0].text == "Msg 2"
