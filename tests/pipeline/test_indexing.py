"""Tests for pipeline/services/indexing.py - IndexService."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.pipeline.services.indexing import IndexService
from polylogue.storage.backends.sqlite import open_connection
from tests.helpers import make_conversation, make_message


class TestIndexService:
    """Test IndexService functionality."""

    def test_update_index_empty_list(self, workspace_env):
        """Update index with empty conversation list."""
        with open_connection(None) as conn:
            config = Config(
                archive_root=workspace_env["archive_root"],
                render_root=workspace_env["archive_root"] / "render",
                sources=[],
            )
            service = IndexService(config, conn)

            result = service.update_index([])
            assert result is True

    def test_update_index_with_conversations(self, workspace_env):
        """Update index with actual conversations."""
        # Use isolated DB connection context for both save and index
        with open_connection(None) as conn:
            # Create test data using helpers
            conv = make_conversation(
                "conv1",
                provider_name="chatgpt",
                title="Test",
                content_hash="hash123",
                created_at=None,
                updated_at=None,
                provider_meta=None,
            )
            msg = make_message(
                "msg1",
                "conv1",
                text="Hello world",
                content_hash="msghash1",
                timestamp=None,
                provider_meta=None,
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
                archive_root=workspace_env["archive_root"],
                render_root=workspace_env["archive_root"] / "render",
                sources=[],
            )
            service = IndexService(config, conn=conn)

            result = service.update_index(["conv1"])
            assert result is True

    def test_rebuild_index_success(self, workspace_env):
        """Rebuild index from scratch."""
        with open_connection(None):
            pass  # Initialize DB

        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )
        service = IndexService(config, conn=None)

        result = service.rebuild_index()
        assert result is True

    def test_ensure_index_exists_success(self, workspace_env):
        """Ensure FTS5 index exists."""
        with open_connection(None) as conn:
            config = Config(
                archive_root=workspace_env["archive_root"],
                render_root=workspace_env["archive_root"] / "render",
                sources=[],
            )
            service = IndexService(config, conn)

            result = service.ensure_index_exists()
            assert result is True

    def test_get_index_status(self, workspace_env):
        """Get index status."""
        with open_connection(None) as conn:
            config = Config(
                archive_root=workspace_env["archive_root"],
                render_root=workspace_env["archive_root"] / "render",
                sources=[],
            )
            service = IndexService(config, conn)

            status = service.get_index_status()
            assert isinstance(status, dict)
            assert "exists" in status
            assert "count" in status

    def test_get_index_status_uses_service_connection(self, workspace_env):
        """Regression: get_index_status must use the service's connection, not open_connection(None)."""
        with open_connection(None) as conn:
            config = Config(
                archive_root=workspace_env["archive_root"],
                render_root=workspace_env["archive_root"] / "render",
                sources=[],
            )
            service = IndexService(config, conn)

            # Ensure FTS table exists via this connection
            service.ensure_index_exists()

            # get_index_status should use the same connection and find the table
            status = service.get_index_status()
            assert status["exists"] is True
            assert isinstance(status["count"], int)


# --- Merged from test_supplementary_coverage.py ---


class TestIndexServiceErrors:
    """Tests for IndexService error handling paths."""

    def test_update_index_failure(self):
        """update_index should return False on exception."""
        from unittest.mock import MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config)

        with patch(
            "polylogue.pipeline.services.indexing.update_index_for_conversations",
            side_effect=Exception("db locked"),
        ):
            result = service.update_index(["conv1", "conv2"])
            assert result is False

    def test_rebuild_index_failure(self):
        """rebuild_index should return False on exception."""
        from unittest.mock import MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config)

        with patch(
            "polylogue.pipeline.services.indexing.rebuild_index",
            side_effect=Exception("disk full"),
        ):
            result = service.rebuild_index()
            assert result is False

    def test_ensure_index_failure(self):
        """ensure_index_exists should return False on exception."""
        from unittest.mock import MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        mock_conn = MagicMock()
        service = IndexService(config=config, conn=mock_conn)

        with patch(
            "polylogue.pipeline.services.indexing.ensure_index",
            side_effect=Exception("corruption"),
        ):
            result = service.ensure_index_exists()
            assert result is False

    def test_get_index_status_failure(self):
        """get_index_status should return fallback on exception."""
        from unittest.mock import MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config)

        with patch(
            "polylogue.pipeline.services.indexing.index_status",
            side_effect=Exception("no such table"),
        ):
            result = service.get_index_status()
            assert result == {"exists": False, "count": 0}

    def test_update_index_empty_ids_ensures_index(self):
        """update_index with empty list should ensure index exists."""
        from unittest.mock import MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        mock_conn = MagicMock()
        service = IndexService(config=config, conn=mock_conn)

        with patch("polylogue.pipeline.services.indexing.ensure_index") as mock_ensure:
            result = service.update_index([])
            assert result is True
            mock_ensure.assert_called_once_with(mock_conn)
