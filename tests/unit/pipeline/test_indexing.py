"""Tests for pipeline/services/indexing.py - IndexService."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.pipeline.services.indexing import IndexService
from tests.infra.helpers import make_conversation, make_message


class TestIndexService:
    """Test IndexService functionality."""

    async def test_update_index_empty_list(self, sqlite_backend):
        """Update index with empty conversation list."""
        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.update_index([])
        assert result is True

    async def test_update_index_with_conversations(self, sqlite_backend):
        """Update index with actual conversations."""
        from polylogue.storage.store import ConversationRecord, MessageRecord

        # Create test data using backend-compatible records
        conv = ConversationRecord(
            conversation_id="conv1",
            provider_name="chatgpt",
            provider_conversation_id="prov_conv1",
            title="Test",
            content_hash="hash123",
        )
        msg = MessageRecord(
            message_id="msg1",
            conversation_id="conv1",
            role="user",
            text="Hello world",
            content_hash="msghash1",
        )
        # Insert using backend API
        await sqlite_backend.save_conversation_record(conv)
        await sqlite_backend.save_messages([msg])

        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.update_index(["conv1"])
        assert result is True

    async def test_rebuild_index_success(self, sqlite_backend):
        """Rebuild index from scratch."""
        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.rebuild_index()
        assert result is True

    async def test_ensure_index_exists_success(self, sqlite_backend):
        """Ensure FTS5 index exists."""
        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.ensure_index_exists()
        assert result is True

    async def test_get_index_status(self, sqlite_backend):
        """Get index status."""
        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        status = await service.get_index_status()
        assert isinstance(status, dict)
        assert "exists" in status
        assert "count" in status

    async def test_get_index_status_uses_service_connection(self, sqlite_backend):
        """Regression: get_index_status must use the service's backend, not open_connection(None)."""
        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        # Ensure FTS table exists via this backend
        await service.ensure_index_exists()

        # get_index_status should use the same backend and find the table
        status = await service.get_index_status()
        assert status["exists"] is True
        assert isinstance(status["count"], int)


# --- Merged from test_supplementary_coverage.py ---


class TestIndexServiceErrors:
    """Tests for IndexService error handling paths."""

    async def test_update_index_failure(self):
        """update_index should return False on exception."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.async_update_index_for_conversations",
            side_effect=Exception("db locked"),
        ):
            result = await service.update_index(["conv1", "conv2"])
            assert result is False

    async def test_rebuild_index_failure(self):
        """rebuild_index should return False on exception."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.async_rebuild_index",
            side_effect=Exception("disk full"),
        ):
            result = await service.rebuild_index()
            assert result is False

    async def test_ensure_index_failure(self):
        """ensure_index_exists should return False on exception."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        mock_backend = MagicMock()
        service = IndexService(config=config, backend=mock_backend)

        with patch(
            "polylogue.pipeline.services.indexing.async_ensure_index",
            side_effect=Exception("corruption"),
        ):
            result = await service.ensure_index_exists()
            assert result is False

    async def test_get_index_status_failure(self):
        """get_index_status should return fallback on exception."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.async_index_status",
            side_effect=Exception("no such table"),
        ):
            result = await service.get_index_status()
            assert result == {"exists": False, "count": 0}

    async def test_update_index_empty_ids_ensures_index(self):
        """update_index with empty list should ensure index exists."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        mock_backend = MagicMock()
        service = IndexService(config=config, backend=mock_backend)

        with patch("polylogue.pipeline.services.indexing.async_ensure_index") as mock_ensure:
            mock_ensure.return_value = AsyncMock()
            result = await service.update_index([])
            assert result is True
            mock_ensure.assert_called_once_with(mock_backend)
