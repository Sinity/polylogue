"""Tests for pipeline/services/indexing.py - IndexService."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from polylogue.config import Config
from polylogue.pipeline.services.indexing import IndexService
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import make_content_block, make_conversation, make_message


def _config() -> Config:
    return Config(
        archive_root=Path("/tmp"),
        render_root=Path("/tmp/render"),
        sources=[],
    )


class TestIndexService:
    """Test IndexService functionality."""

    async def test_update_index_empty_list(self, sqlite_backend: SQLiteBackend) -> None:
        """Update index with empty conversation list."""
        service = IndexService(_config(), backend=sqlite_backend)

        result = await service.update_index([])
        assert result is True

    async def test_update_index_with_conversations(self, sqlite_backend: SQLiteBackend) -> None:
        """Update index with actual conversations."""

        # Create test data using backend-compatible records
        conv = make_conversation(
            conversation_id="conv1",
            provider_name="chatgpt",
            provider_conversation_id="prov_conv1",
            title="Test",
            content_hash="hash123",
        )
        msg = make_message(
            message_id="msg1",
            conversation_id="conv1",
            role="user",
            text="Hello world",
            content_hash="msghash1",
        )
        # Insert using backend API
        await sqlite_backend.save_conversation_record(conv)
        await sqlite_backend.save_messages([msg])

        service = IndexService(_config(), backend=sqlite_backend)

        result = await service.update_index(["conv1"])
        assert result is True

    async def test_update_index_accepts_async_iterable(self, sqlite_backend: SQLiteBackend) -> None:
        """Streaming conversation IDs can be indexed without prebuilding a list."""

        await sqlite_backend.save_conversation_record(
            make_conversation(
                conversation_id="conv-stream",
                provider_name="chatgpt",
                provider_conversation_id="prov-conv-stream",
                title="Stream Test",
                content_hash="hash-stream",
            )
        )
        await sqlite_backend.save_messages(
            [
                make_message(
                    message_id="msg-stream",
                    conversation_id="conv-stream",
                    role="user",
                    text="hello world",
                    content_hash="msghash-stream",
                )
            ]
        )

        service = IndexService(_config(), backend=sqlite_backend)

        async def conversation_ids() -> AsyncIterator[str]:
            yield "conv-stream"

        result = await service.update_index(conversation_ids())

        assert result is True
        status = await service.get_index_status()
        assert status["exists"] is True
        assert status["count"] >= 1

    async def test_rebuild_index_success(self, sqlite_backend: SQLiteBackend) -> None:
        """Rebuild index from scratch."""
        service = IndexService(_config(), backend=sqlite_backend)

        result = await service.rebuild_index()
        assert result is True

    async def test_rebuild_index_reports_chunk_progress(self, sqlite_backend: SQLiteBackend) -> None:
        """Full rebuild skips the action phase when no action repair is needed."""

        for index in range(3):
            conversation_id = f"conv-progress-{index}"
            await sqlite_backend.save_conversation_record(
                make_conversation(
                    conversation_id=conversation_id,
                    provider_name="chatgpt",
                    provider_conversation_id=f"prov-{index}",
                    title=f"Conversation {index}",
                    content_hash=f"hash-{index}",
                )
            )
            await sqlite_backend.save_messages(
                [
                    make_message(
                        message_id=f"msg-progress-{index}",
                        conversation_id=conversation_id,
                        role="user",
                        text=f"Hello from conversation {index}",
                        content_hash=f"message-hash-{index}",
                    )
                ]
            )

        service = IndexService(_config(), backend=sqlite_backend)
        progress_events: list[tuple[int, str | None]] = []

        def capture(amount: int, desc: str | None = None) -> None:
            progress_events.append((amount, desc))

        result = await service.rebuild_index(progress_callback=capture)

        assert result is True
        assert progress_events
        descriptions = [desc for _, desc in progress_events if desc is not None]
        assert descriptions[0] == "Indexing: full-text search 0/3"
        assert descriptions[-1] == "Indexing: full-text search 3/3"

    async def test_rebuild_index_reports_action_phase_for_missing_action_rows(
        self,
        sqlite_backend: SQLiteBackend,
    ) -> None:
        """Full rebuild still repairs action rows when tool-use blocks exist without action rows."""

        await sqlite_backend.save_conversation_record(
            make_conversation(
                conversation_id="conv-plain",
                provider_name="chatgpt",
                provider_conversation_id="prov-plain",
                title="Plain Conversation",
                content_hash="hash-plain",
            )
        )
        await sqlite_backend.save_messages(
            [
                make_message(
                    message_id="msg-plain",
                    conversation_id="conv-plain",
                    role="user",
                    text="No tool use here",
                    sort_key=0.5,
                    content_hash="message-hash-plain",
                )
            ]
        )
        await sqlite_backend.save_conversation_record(
            make_conversation(
                conversation_id="conv-action",
                provider_name="chatgpt",
                provider_conversation_id="prov-action",
                title="Action Conversation",
                content_hash="hash-action",
            )
        )
        await sqlite_backend.save_messages(
            [
                make_message(
                    message_id="msg-action",
                    conversation_id="conv-action",
                    role="assistant",
                    text="Ran rg",
                    sort_key=1.0,
                    content_hash="message-hash-action",
                )
            ]
        )
        await sqlite_backend.save_content_blocks(
            [
                make_content_block(
                    message_id="msg-action",
                    conversation_id="conv-action",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="exec_command",
                    tool_id="tool-1",
                    tool_input='{"cmd":"rg -n action_events"}',
                )
            ]
        )

        service = IndexService(_config(), backend=sqlite_backend)
        progress_events: list[tuple[int, str | None]] = []

        def capture(amount: int, desc: str | None = None) -> None:
            progress_events.append((amount, desc))

        result = await service.rebuild_index(progress_callback=capture)

        assert result is True
        descriptions = [desc for _, desc in progress_events if desc is not None]
        assert descriptions[0] == "Indexing: action events 0/3"
        assert "Indexing: full-text search 1/3" in descriptions
        assert descriptions[-1] == "Indexing: full-text search 3/3"

    async def test_ensure_index_exists_success(self, sqlite_backend: SQLiteBackend) -> None:
        """Ensure FTS5 index exists."""
        service = IndexService(_config(), backend=sqlite_backend)

        result = await service.ensure_index_exists()
        assert result is True

    async def test_get_index_status(self, sqlite_backend: SQLiteBackend) -> None:
        """Get index status."""
        service = IndexService(_config(), backend=sqlite_backend)

        status = await service.get_index_status()
        assert isinstance(status, dict)
        assert "exists" in status
        assert "count" in status

    async def test_get_index_status_uses_service_connection(self, sqlite_backend: SQLiteBackend) -> None:
        """Regression: get_index_status must use the service's backend, not open_connection(None)."""
        service = IndexService(_config(), backend=sqlite_backend)

        # Ensure FTS table exists via this backend
        await service.ensure_index_exists()

        # get_index_status should use the same backend and find the table
        status = await service.get_index_status()
        assert status["exists"] is True
        assert isinstance(status["count"], int)

    async def test_get_index_status_after_schema_init(self, sqlite_backend: SQLiteBackend) -> None:
        """Status after schema init shows index exists with zero-or-more entries."""
        service = IndexService(_config(), backend=sqlite_backend)

        from polylogue.storage.query_models import ConversationRecordQuery

        await sqlite_backend.queries.list_conversations(ConversationRecordQuery())

        status = await service.get_index_status()
        assert status["exists"] is True
        assert isinstance(status["count"], int)


# --- Merged from test_supplementary_coverage.py ---


class TestIndexServiceErrors:
    """Tests for IndexService error handling paths."""

    async def test_update_index_failure(self) -> None:
        """update_index should return False on exception."""
        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.update_index_for_conversations",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("db locked"),
        ):
            result = await service.update_index(["conv1", "conv2"])
            assert result is False

    async def test_rebuild_index_failure(self) -> None:
        """rebuild_index should return False on exception."""
        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.rebuild_index",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("disk full"),
        ):
            result = await service.rebuild_index()
            assert result is False

    async def test_ensure_index_failure(self) -> None:
        """ensure_index_exists should return False on exception."""
        config = MagicMock()
        mock_backend = MagicMock()
        service = IndexService(config=config, backend=mock_backend)

        with patch(
            "polylogue.pipeline.services.indexing.ensure_index",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("corruption"),
        ):
            result = await service.ensure_index_exists()
            assert result is False

    async def test_get_index_status_failure(self) -> None:
        """get_index_status should return fallback on exception."""
        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.index_status",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("no such table"),
        ):
            result = await service.get_index_status()
            assert result == {"exists": False, "count": 0}

    async def test_update_index_no_backend(self) -> None:
        """Update without backend returns False."""
        config = MagicMock()
        service = IndexService(config=config, backend=None)

        result = await service.update_index(["conv-1"])

        assert result is False

    async def test_rebuild_no_backend(self) -> None:
        """Rebuild without backend returns False."""
        config = MagicMock()
        service = IndexService(config=config, backend=None)

        result = await service.rebuild_index()

        assert result is False

    async def test_get_index_status_no_backend(self) -> None:
        """Status without backend returns defaults."""
        config = MagicMock()
        service = IndexService(config=config, backend=None)

        status = await service.get_index_status()

        assert status == {"exists": False, "count": 0}

    async def test_update_index_empty_ids_ensures_index(self) -> None:
        """update_index always delegates to the canonical storage helper."""
        config = MagicMock()
        mock_backend = MagicMock()
        service = IndexService(config=config, backend=mock_backend)

        with patch(
            "polylogue.pipeline.services.indexing.update_index_for_conversations", new_callable=AsyncMock
        ) as mock_update:
            result = await service.update_index([])
            assert result is True
            mock_update.assert_called_once_with([], mock_backend)
