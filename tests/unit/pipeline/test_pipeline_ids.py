"""Focused tests for pipeline ID and attachment path helpers."""

from __future__ import annotations

import pytest

from polylogue.assets import asset_path
from polylogue.pipeline.ids import (
    attachment_content_id,
    conversation_content_hash,
    conversation_id,
    materialize_attachment_path,
    message_content_hash,
    move_attachment_to_archive,
)
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation, ParsedMessage


def test_attachment_content_id_returns_target_path_without_moving(tmp_path):
    archive_root = tmp_path / "archive"
    uploads = tmp_path / "uploads"
    archive_root.mkdir()
    uploads.mkdir()
    source_file = uploads / "note.txt"
    source_file.write_text("hello world", encoding="utf-8")

    attachment = ParsedAttachment(
        provider_attachment_id="file-1",
        message_provider_id="msg-1",
        name="note.txt",
        mime_type="text/plain",
        size_bytes=11,
        path=str(source_file),
        provider_meta={},
    )

    digest, updated_meta, updated_path = attachment_content_id("chatgpt", attachment, archive_root=archive_root)
    target = asset_path(archive_root, digest)

    assert digest
    assert updated_path == str(target)
    assert updated_meta is not None and "sha256" in updated_meta
    assert source_file.exists()
    assert not target.exists()


def test_materialize_attachment_path_moves_file_into_assets(tmp_path):
    archive_root = tmp_path / "archive"
    uploads = tmp_path / "uploads"
    archive_root.mkdir()
    uploads.mkdir()
    source_file = uploads / "note.txt"
    source_file.write_text("hello world", encoding="utf-8")
    target = asset_path(archive_root, "abc123")

    materialize_attachment_path(source_file, target)

    assert not source_file.exists()
    assert target.exists()


class TestAttachmentPathMove:
    def test_move_attachment_raises_on_missing_source(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            move_attachment_to_archive(tmp_path / "nonexistent.txt", tmp_path / "archive" / "dest.txt")

    def test_move_attachment_raises_on_permission_error(self, tmp_path, monkeypatch):
        import shutil

        source = tmp_path / "source.txt"
        source.write_text("content")
        dest = tmp_path / "archive" / "dest.txt"

        def failing_move(*args, **kwargs):
            raise PermissionError("Access denied")

        monkeypatch.setattr(shutil, "move", failing_move)
        with pytest.raises(PermissionError):
            move_attachment_to_archive(source, dest)

    def test_move_attachment_creates_parent_dirs(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("content")
        dest = tmp_path / "deep" / "nested" / "archive" / "dest.txt"

        move_attachment_to_archive(source, dest)

        assert dest.exists()
        assert dest.read_text() == "content"
        assert not source.exists()


class TestConversationIdValidation:
    def test_rejects_empty_provider(self):
        with pytest.raises(ValueError, match="provider"):
            conversation_id("", "conv-123")

    def test_rejects_empty_provider_conversation_id(self):
        with pytest.raises(ValueError, match="conversation"):
            conversation_id("chatgpt", "")


def test_conversation_content_hash_with_missing_message_ids():
    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="",
                role="user",
                text="Hello",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )

    digest = conversation_content_hash(conversation)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_message_hash_none_vs_empty_timestamp_distinguishable():
    msg_with_none = ParsedMessage(provider_message_id="msg-1", role="user", text="hello", timestamp=None)
    msg_with_empty = ParsedMessage(provider_message_id="msg-1", role="user", text="hello", timestamp="")

    assert message_content_hash(msg_with_none, "msg-1") != message_content_hash(msg_with_empty, "msg-1")


def test_message_hash_empty_text_is_deterministic():
    message = ParsedMessage(provider_message_id="msg-1", role="user", text="", timestamp="2024-01-01")
    first = message_content_hash(message, "msg-1")
    second = message_content_hash(message, "msg-1")
    assert first == second
    assert len(first) == 64


def test_message_hash_different_provider_id_produces_different_hash():
    message = ParsedMessage(provider_message_id="msg-1", role="user", text="hello", timestamp="2024-01-01")
    assert message_content_hash(message, "msg-1") != message_content_hash(message, "msg-2")


def test_conversation_hash_empty_messages_is_valid():
    conversation = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Empty Conv",
        created_at=None,
        updated_at=None,
        messages=[],
    )
    assert len(conversation_content_hash(conversation)) == 64


def test_conversation_hash_timestamps_affect_hash():
    message = ParsedMessage(provider_message_id="m1", role="user", text="hi", timestamp=None)
    conversation_one = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-01",
        updated_at=None,
        messages=[message],
    )
    conversation_two = ParsedConversation(
        provider_name="test",
        provider_conversation_id="conv-1",
        title="Test",
        created_at="2024-01-02",
        updated_at=None,
        messages=[message],
    )

    assert conversation_content_hash(conversation_one) != conversation_content_hash(conversation_two)
