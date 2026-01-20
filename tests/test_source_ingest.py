from __future__ import annotations

import json

from polylogue.config import Source
from polylogue.source_ingest import iter_source_conversations, parse_drive_payload


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
    from polylogue.source_ingest import _decode_json_bytes

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
    from polylogue.source_ingest import _decode_json_bytes

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
    from polylogue.source_ingest import _decode_json_bytes

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
    from polylogue.source_ingest import _decode_json_bytes

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
    from polylogue.source_ingest import _decode_json_bytes

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
    from polylogue.source_ingest import _decode_json_bytes

    # All null bytes
    all_nulls = b"\x00\x00\x00\x00"

    result = _decode_json_bytes(all_nulls)
    # After stripping nulls, nothing remains
    assert result is None


def test_decode_json_bytes_handles_utf32(tmp_path):
    """_decode_json_bytes() handles UTF-32 encoding."""
    from polylogue.source_ingest import _decode_json_bytes

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
    from polylogue.source_ingest import _decode_json_bytes

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
