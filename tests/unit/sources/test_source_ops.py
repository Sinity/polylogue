from __future__ import annotations

import json
from pathlib import Path

from polylogue.config import Source
from polylogue.sources import iter_source_conversations
from tests.infra.helpers import GenericConversationBuilder, make_claude_chat_message


def test_claude_chat_messages_attachment_metadata_survives_source_iteration(tmp_path: Path) -> None:
    payload = {
        "chat_messages": [
            make_claude_chat_message(
                "msg-1",
                "assistant",
                "Files",
                attachments=[{"id": "file-1", "name": "notes.txt", "size": 12, "mimeType": "text/plain"}],
            )
        ]
    }
    source_file = tmp_path / "claude.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    conversations = list(iter_source_conversations(Source(name="inbox", path=source_file)))

    assert conversations
    attachment = conversations[0].attachments[0]
    assert attachment.provider_attachment_id == "file-1"
    assert attachment.name == "notes.txt"


def test_iter_source_conversations_tracks_failures_without_stopping(tmp_path: Path) -> None:
    (
        GenericConversationBuilder("v1")
        .title("Valid 1")
        .add_message("user", "hi", text="hi")
        .write_to(tmp_path / "valid1.json")
    )
    (tmp_path / "invalid.json").write_text("{ this is not valid json }", encoding="utf-8")
    (
        GenericConversationBuilder("v2")
        .title("Valid 2")
        .add_message("user", "bye", text="bye")
        .write_to(tmp_path / "valid2.json")
    )

    cursor_state: dict[str, object] = {}
    conversations = list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert {conversation.provider_conversation_id for conversation in conversations} == {"v1", "v2"}
    assert cursor_state["file_count"] == 3
    assert cursor_state["failed_count"] >= 1
    assert any("invalid.json" in item["path"] for item in cursor_state["failed_files"])


def test_iter_source_conversations_handles_empty_directory(tmp_path: Path) -> None:
    cursor_state: dict[str, object] = {}

    conversations = list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert conversations == []
    assert cursor_state["file_count"] == 0
    assert cursor_state["failed_count"] == 0


def test_iter_source_conversations_tracks_deleted_file_as_failure(tmp_path: Path, monkeypatch) -> None:
    import io

    test_file = tmp_path / "conversation.json"
    (
        GenericConversationBuilder("test")
        .title("Test")
        .add_message("user", "hello", text="hello")
        .write_to(test_file)
    )

    cursor_state: dict[str, object] = {}
    original_open = io.open
    opens = {"count": 0}

    def tracking_open(path, *args, **kwargs):
        opens["count"] += 1
        if str(test_file) in str(path) and opens["count"] == 1:
            test_file.unlink()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("io.open", tracking_open)

    list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert cursor_state["failed_count"] >= 1
    assert any("conversation.json" in item["path"] for item in cursor_state["failed_files"])


def test_iter_source_conversations_tracks_replaced_invalid_file_as_failure(tmp_path: Path, monkeypatch) -> None:
    import io

    test_file = tmp_path / "conversation.json"
    (
        GenericConversationBuilder("test")
        .title("Test")
        .add_message("user", "hello", text="hello")
        .write_to(test_file)
    )

    cursor_state: dict[str, object] = {}
    original_open = io.open
    opens = {"count": 0}

    def corrupting_open(path, *args, **kwargs):
        opens["count"] += 1
        if str(test_file) in str(path) and opens["count"] == 1:
            test_file.write_text("{ this is not valid json at all", encoding="utf-8")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("io.open", corrupting_open)

    conversations = list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert conversations == []
    assert cursor_state["failed_count"] >= 1
    assert any("conversation.json" in item["path"] for item in cursor_state["failed_files"])


def test_iter_source_conversations_continues_after_missing_file(tmp_path: Path, monkeypatch) -> None:
    import io

    (
        GenericConversationBuilder("conv1")
        .add_message("user", "first", text="first")
        .write_to(tmp_path / "file1.json")
    )
    (
        GenericConversationBuilder("conv2")
        .add_message("user", "second", text="second")
        .write_to(tmp_path / "file2.json")
    )

    cursor_state: dict[str, object] = {}
    original_open = io.open

    def selective_fail_open(path, *args, **kwargs):
        if "file2" in str(path):
            raise FileNotFoundError(f"File not found: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("io.open", selective_fail_open)

    conversations = list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert {conversation.provider_conversation_id for conversation in conversations} == {"conv1"}
    assert cursor_state["failed_count"] >= 1
    assert any("file2.json" in item["path"] for item in cursor_state["failed_files"])
