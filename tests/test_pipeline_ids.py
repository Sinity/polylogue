from __future__ import annotations

import pytest

from polylogue.assets import asset_path
from polylogue.importers.base import ParsedAttachment, ParsedConversation, ParsedMessage
from polylogue.pipeline.ids import (
    attachment_content_id,
    conversation_content_hash,
    conversation_id,
    message_content_hash,
)


def test_attachment_content_id_moves_file_into_assets(tmp_path):
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
        path=str(source_file),  # Must set path for file to be moved
        provider_meta={},
    )

    # attachment_content_id now returns (digest, updated_meta, updated_path) without mutation
    digest, updated_meta, updated_path = attachment_content_id("chatgpt", attachment, archive_root=archive_root)
    target = asset_path(archive_root, digest)

    assert digest
    assert updated_path == str(target)  # returned path, not mutated attachment.path
    assert updated_meta is not None and "sha256" in updated_meta
    assert not source_file.exists()
    assert target.exists()


class TestConversationId:
    """Tests for conversation_id function."""

    def test_generates_deterministic_id(self):
        """Same inputs should produce same ID."""
        id1 = conversation_id("chatgpt", "conv-123")
        id2 = conversation_id("chatgpt", "conv-123")
        assert id1 == id2

    def test_different_providers_different_ids(self):
        """Different providers should produce different IDs."""
        id1 = conversation_id("chatgpt", "conv-123")
        id2 = conversation_id("claude", "conv-123")
        assert id1 != id2

    def test_rejects_empty_provider(self):
        """Empty provider_name MUST be rejected.

        This test SHOULD FAIL until validation is added to conversation_id().
        """
        with pytest.raises(ValueError, match="provider"):
            conversation_id("", "conv-123")

    def test_rejects_empty_provider_conversation_id(self):
        """Empty provider_conversation_id MUST be rejected.

        This test SHOULD FAIL until validation is added.
        """
        with pytest.raises(ValueError, match="conversation"):
            conversation_id("chatgpt", "")


class TestMessageContentHash:
    """Tests for message_content_hash function."""

    def test_deterministic_hash(self):
        """Same message content should produce same hash."""
        msg = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="hello world",
            timestamp="2024-01-01",
        )
        hash1 = message_content_hash(msg, "msg-1")
        hash2 = message_content_hash(msg, "msg-1")
        assert hash1 == hash2

    def test_different_text_different_hash(self):
        """Different text should produce different hash."""
        msg1 = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="hello",
            timestamp="2024-01-01",
        )
        msg2 = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="goodbye",
            timestamp="2024-01-01",
        )
        hash1 = message_content_hash(msg1, "msg-1")
        hash2 = message_content_hash(msg2, "msg-1")
        assert hash1 != hash2

    def test_none_timestamp_vs_missing_distinguishable(self):
        """None vs empty timestamp MUST produce different hashes.

        This test verifies hash collision prevention.
        """
        # Explicit None
        msg_with_none = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="hello",
            timestamp=None,
        )
        # Empty string
        msg_with_empty = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="hello",
            timestamp="",
        )

        hash_with_none = message_content_hash(msg_with_none, "msg-1")
        hash_with_empty = message_content_hash(msg_with_empty, "msg-1")

        # These MUST be different to prevent hash collisions
        assert hash_with_none != hash_with_empty, "Hash collision: None and empty string produce same hash!"

    def test_empty_text_hashes_consistently(self):
        """Empty text should still produce valid, deterministic hash."""
        msg = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="",
            timestamp="2024-01-01",
        )
        hash1 = message_content_hash(msg, "msg-1")
        hash2 = message_content_hash(msg, "msg-1")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_different_role_different_hash(self):
        """Different message roles should produce different hashes."""
        msg_user = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="hello",
            timestamp="2024-01-01",
        )
        msg_assistant = ParsedMessage(
            provider_message_id="msg-1",
            role="assistant",
            text="hello",
            timestamp="2024-01-01",
        )
        hash1 = message_content_hash(msg_user, "msg-1")
        hash2 = message_content_hash(msg_assistant, "msg-1")
        assert hash1 != hash2

    def test_different_provider_id_different_hash(self):
        """Different provider_message_id should produce different hash."""
        msg = ParsedMessage(
            provider_message_id="msg-1",
            role="user",
            text="hello",
            timestamp="2024-01-01",
        )
        hash1 = message_content_hash(msg, "msg-1")
        hash2 = message_content_hash(msg, "msg-2")
        assert hash1 != hash2


class TestConversationContentHash:
    """Tests for conversation_content_hash function."""

    def test_deterministic_for_same_content(self):
        """Same conversation content should produce same hash."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Test",
            created_at=None,
            updated_at=None,
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role="user",
                    text="hi",
                    timestamp=None,
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role="assistant",
                    text="hello",
                    timestamp=None,
                ),
            ],
        )
        hash1 = conversation_content_hash(convo)
        hash2 = conversation_content_hash(convo)
        assert hash1 == hash2

    def test_message_order_matters(self):
        """Different message order should produce different hash."""
        msg1 = ParsedMessage(
            provider_message_id="m1",
            role="user",
            text="first",
            timestamp=None,
        )
        msg2 = ParsedMessage(
            provider_message_id="m2",
            role="assistant",
            text="second",
            timestamp=None,
        )

        convo_order1 = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Test",
            created_at=None,
            updated_at=None,
            messages=[msg1, msg2],
        )
        convo_order2 = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Test",
            created_at=None,
            updated_at=None,
            messages=[msg2, msg1],
        )

        hash_order1 = conversation_content_hash(convo_order1)
        hash_order2 = conversation_content_hash(convo_order2)

        assert hash_order1 != hash_order2

    def test_title_affects_hash(self):
        """Different titles should produce different hashes."""
        msg = ParsedMessage(
            provider_message_id="m1",
            role="user",
            text="hi",
            timestamp=None,
        )
        convo1 = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Title A",
            created_at=None,
            updated_at=None,
            messages=[msg],
        )
        convo2 = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Title B",
            created_at=None,
            updated_at=None,
            messages=[msg],
        )
        hash1 = conversation_content_hash(convo1)
        hash2 = conversation_content_hash(convo2)
        assert hash1 != hash2

    def test_empty_messages_valid_hash(self):
        """Conversation with no messages should still produce valid hash."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Empty Conv",
            created_at=None,
            updated_at=None,
            messages=[],
        )
        hash_result = conversation_content_hash(convo)
        assert len(hash_result) == 64

    def test_attachments_affect_hash(self):
        """Different attachments should produce different hashes."""
        msg = ParsedMessage(
            provider_message_id="m1",
            role="user",
            text="hi",
            timestamp=None,
        )
        att = ParsedAttachment(
            provider_attachment_id="att-1",
            message_provider_id="m1",
            name="file.txt",
            mime_type="text/plain",
            size_bytes=100,
        )

        convo_no_att = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Test",
            created_at=None,
            updated_at=None,
            messages=[msg],
            attachments=[],
        )
        convo_with_att = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Test",
            created_at=None,
            updated_at=None,
            messages=[msg],
            attachments=[att],
        )

        hash_no_att = conversation_content_hash(convo_no_att)
        hash_with_att = conversation_content_hash(convo_with_att)

        assert hash_no_att != hash_with_att

    def test_timestamps_affect_hash(self):
        """Different created_at/updated_at should produce different hashes."""
        msg = ParsedMessage(
            provider_message_id="m1",
            role="user",
            text="hi",
            timestamp=None,
        )
        convo1 = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Test",
            created_at="2024-01-01",
            updated_at=None,
            messages=[msg],
        )
        convo2 = ParsedConversation(
            provider_name="test",
            provider_conversation_id="conv-1",
            title="Test",
            created_at="2024-01-02",
            updated_at=None,
            messages=[msg],
        )

        hash1 = conversation_content_hash(convo1)
        hash2 = conversation_content_hash(convo2)

        assert hash1 != hash2


class TestAttachmentPathMove:
    """Tests for attachment path handling."""

    def test_move_attachment_raises_on_missing_source(self, tmp_path):
        """Moving non-existent attachment should raise, not silently fail.

        This test SHOULD FAIL until error handling is added.
        """
        from polylogue.pipeline.ids import move_attachment_to_archive

        missing_source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "archive" / "dest.txt"

        with pytest.raises(FileNotFoundError):
            move_attachment_to_archive(missing_source, dest)

    def test_move_attachment_raises_on_permission_error(self, tmp_path, monkeypatch):
        """Move failure due to permissions should raise, not silently fail."""
        import shutil

        from polylogue.pipeline.ids import move_attachment_to_archive

        source = tmp_path / "source.txt"
        source.write_text("content")
        dest = tmp_path / "archive" / "dest.txt"

        # Make shutil.move raise
        def failing_move(*args, **kwargs):
            raise PermissionError("Access denied")

        monkeypatch.setattr(shutil, "move", failing_move)

        with pytest.raises(PermissionError):
            move_attachment_to_archive(source, dest)

    def test_move_attachment_creates_parent_dirs(self, tmp_path):
        """Move should create parent directories if needed."""
        from polylogue.pipeline.ids import move_attachment_to_archive

        source = tmp_path / "source.txt"
        source.write_text("content")
        dest = tmp_path / "deep" / "nested" / "archive" / "dest.txt"

        move_attachment_to_archive(source, dest)

        assert dest.exists()
        assert dest.read_text() == "content"
        assert not source.exists()  # Original should be gone
