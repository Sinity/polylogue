"""Tests for pipeline/services/indexing.py - IndexService."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.pipeline.services.indexing import IndexService
from polylogue.storage.db import open_connection
from polylogue.storage.store import ConversationRecord, MessageRecord


class TestIndexService:
    """Test IndexService functionality."""

    def test_update_index_empty_list(self, tmp_path, monkeypatch):
        """Update index with empty conversation list."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        with open_connection(None) as conn:
            config = Config(
                version=2,
                archive_root=tmp_path / "archive",
                render_root=tmp_path / "render",
                sources=[],
                path=tmp_path / "config.json",
            )
            service = IndexService(config, conn)

            result = service.update_index([])
            assert result is True

    def test_update_index_with_conversations(self, tmp_path, monkeypatch):
        """Update index with actual conversations."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        # Use isolated DB connection context for both save and index
        with open_connection(None) as conn:
            # Create a conversation directly using the connection
            conv = ConversationRecord(
                conversation_id="conv1",
                provider_name="chatgpt",
                provider_conversation_id="chat123",
                title="Test",
                created_at=None,
                updated_at=None,
                content_hash="hash123",
                provider_meta=None,
                version=1,
            )
            msg = MessageRecord(
                message_id="msg1",
                conversation_id="conv1",
                provider_message_id=None,
                role="user",
                text="Hello world",
                timestamp=None,
                content_hash="msghash1",
                provider_meta=None,
                version=1,
            )
            # Insert directly using connection
            conn.execute(
                """INSERT INTO conversations (
                    conversation_id, provider_name, provider_conversation_id,
                    title, content_hash, version
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (conv.conversation_id, conv.provider_name, conv.provider_conversation_id,
                 conv.title, conv.content_hash, conv.version),
            )
            conn.execute(
                """INSERT INTO messages (
                    message_id, conversation_id, role, text, content_hash, version
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (msg.message_id, msg.conversation_id, msg.role, msg.text,
                 msg.content_hash, msg.version),
            )
            conn.commit()

            config = Config(
                version=2,
                archive_root=tmp_path / "archive",
                render_root=tmp_path / "render",
                sources=[],
                path=tmp_path / "config.json",
            )
            service = IndexService(config, conn=conn)

            result = service.update_index(["conv1"])
            assert result is True

    def test_rebuild_index_success(self, tmp_path, monkeypatch):
        """Rebuild index from scratch."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        with open_connection(None):
            pass  # Initialize DB

        config = Config(
            version=2,
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            path=tmp_path / "config.json",
        )
        service = IndexService(config, conn=None)

        result = service.rebuild_index()
        assert result is True

    def test_ensure_index_exists_success(self, tmp_path, monkeypatch):
        """Ensure FTS5 index exists."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        with open_connection(None) as conn:
            config = Config(
                version=2,
                archive_root=tmp_path / "archive",
                render_root=tmp_path / "render",
                sources=[],
                path=tmp_path / "config.json",
            )
            service = IndexService(config, conn)

            result = service.ensure_index_exists()
            assert result is True

    def test_get_index_status(self, tmp_path, monkeypatch):
        """Get index status."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        with open_connection(None) as conn:
            config = Config(
                version=2,
                archive_root=tmp_path / "archive",
                render_root=tmp_path / "render",
                sources=[],
                path=tmp_path / "config.json",
            )
            service = IndexService(config, conn)

            status = service.get_index_status()
            assert isinstance(status, dict)
            assert "exists" in status
            assert "count" in status
