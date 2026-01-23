"""Tests for pipeline/services/ingestion.py."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.importers.base import ParsedConversation, ParsedMessage
from polylogue.pipeline.services.ingestion import IngestResult, IngestionService


# ============================================================================
# IngestResult Tests
# ============================================================================


class TestIngestResultInit:
    """Tests for IngestResult initialization."""

    def test_counts_initialized_to_zero(self):
        """All count fields start at zero."""
        result = IngestResult()

        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0
        assert result.counts["attachments"] == 0
        assert result.counts["skipped_conversations"] == 0
        assert result.counts["skipped_messages"] == 0
        assert result.counts["skipped_attachments"] == 0

    def test_changed_counts_initialized_to_zero(self):
        """All changed_counts fields start at zero."""
        result = IngestResult()

        assert result.changed_counts["conversations"] == 0
        assert result.changed_counts["messages"] == 0
        assert result.changed_counts["attachments"] == 0

    def test_processed_ids_initialized_empty(self):
        """processed_ids starts as empty set."""
        result = IngestResult()

        assert result.processed_ids == set()
        assert isinstance(result.processed_ids, set)

    def test_lock_initialized(self):
        """_lock is a threading.Lock instance."""
        result = IngestResult()

        assert hasattr(result, "_lock")
        assert isinstance(result._lock, type(threading.Lock()))


class TestIngestResultMerge:
    """Tests for IngestResult.merge_result method."""

    def test_merge_adds_to_counts(self):
        """merge_result accumulates counts."""
        result = IngestResult()

        result.merge_result(
            conversation_id="conv1",
            result_counts={
                "conversations": 1,
                "messages": 5,
                "attachments": 2,
                "skipped_conversations": 0,
                "skipped_messages": 1,
                "skipped_attachments": 0,
            },
            content_changed=True,
        )

        assert result.counts["conversations"] == 1
        assert result.counts["messages"] == 5
        assert result.counts["attachments"] == 2
        assert result.counts["skipped_messages"] == 1

    def test_merge_multiple_conversations(self):
        """Multiple merges accumulate correctly."""
        result = IngestResult()

        result.merge_result(
            "conv1",
            {"conversations": 1, "messages": 3, "attachments": 1, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
            content_changed=True,
        )
        result.merge_result(
            "conv2",
            {"conversations": 1, "messages": 7, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 2, "skipped_attachments": 0},
            content_changed=True,
        )

        assert result.counts["conversations"] == 2
        assert result.counts["messages"] == 10
        assert result.counts["attachments"] == 1
        assert result.counts["skipped_messages"] == 2

    def test_merge_tracks_changed_conversations(self):
        """content_changed=True increments changed_counts."""
        result = IngestResult()

        result.merge_result(
            "conv1",
            {"conversations": 1, "messages": 2, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
            content_changed=True,
        )

        assert result.changed_counts["conversations"] == 1
        assert result.changed_counts["messages"] == 2

    def test_merge_unchanged_conversation_not_tracked(self):
        """content_changed=False does not increment changed_counts for conversations."""
        result = IngestResult()

        result.merge_result(
            "conv1",
            {"conversations": 0, "messages": 0, "attachments": 0, "skipped_conversations": 1, "skipped_messages": 0, "skipped_attachments": 0},
            content_changed=False,
        )

        assert result.changed_counts["conversations"] == 0
        # Skipped conversation not added to processed_ids
        assert "conv1" not in result.processed_ids

    def test_merge_adds_to_processed_ids_when_changed(self):
        """Conversation ID added to processed_ids when content changed."""
        result = IngestResult()

        result.merge_result(
            "conv-123",
            {"conversations": 1, "messages": 1, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
            content_changed=True,
        )

        assert "conv-123" in result.processed_ids

    def test_merge_adds_to_processed_ids_when_ingest_changed(self):
        """Conversation ID added when ingest counts > 0 even if content unchanged."""
        result = IngestResult()

        # New messages but content_changed=False (e.g., first import)
        result.merge_result(
            "conv-456",
            {"conversations": 1, "messages": 5, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
            content_changed=False,
        )

        assert "conv-456" in result.processed_ids

    def test_merge_skipped_not_in_processed_ids(self):
        """Skipped conversation (no changes) not in processed_ids."""
        result = IngestResult()

        result.merge_result(
            "conv-skipped",
            {"conversations": 0, "messages": 0, "attachments": 0, "skipped_conversations": 1, "skipped_messages": 5, "skipped_attachments": 0},
            content_changed=False,
        )

        assert "conv-skipped" not in result.processed_ids

    def test_merge_thread_safe(self):
        """Concurrent merges don't cause race conditions."""
        result = IngestResult()
        errors: list[Exception] = []

        def merge_batch(start_id: int) -> None:
            try:
                for i in range(100):
                    result.merge_result(
                        f"conv-{start_id}-{i}",
                        {"conversations": 1, "messages": 2, "attachments": 1, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
                        content_changed=True,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=merge_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert result.counts["conversations"] == 500
        assert result.counts["messages"] == 1000
        assert result.counts["attachments"] == 500
        assert len(result.processed_ids) == 500


# ============================================================================
# IngestionService Tests
# ============================================================================


class TestIngestionServiceInit:
    """Tests for IngestionService initialization."""

    def test_init_sets_repository(self):
        """Repository is stored on init."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        assert service.repository is mock_repo

    def test_init_sets_archive_root(self):
        """archive_root is stored on init."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)
        archive = Path("/custom/archive")

        service = IngestionService(
            repository=mock_repo,
            archive_root=archive,
            config=mock_config,
        )

        assert service.archive_root == archive

    def test_init_sets_config(self):
        """Config is stored on init."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        assert service.config is mock_config


class TestIngestionServiceIngestSources:
    """Tests for IngestionService.ingest_sources method."""

    def _make_parsed_conversation(self, conv_id: str) -> ParsedConversation:
        """Helper to create a ParsedConversation."""
        return ParsedConversation(
            provider_name="test",
            provider_conversation_id=conv_id,
            title=f"Conversation {conv_id}",
            messages=[
                ParsedMessage(
                    provider_message_id=f"msg-{conv_id}-1",
                    role="user",
                    text="Hello",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def test_ingest_empty_sources_returns_empty_result(self):
        """Empty sources list returns empty IngestResult."""
        mock_repo = MagicMock()
        mock_repo._db_path = Path("/tmp/test.db")
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        result = service.ingest_sources([])

        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0
        assert len(result.processed_ids) == 0

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    @patch("polylogue.pipeline.services.ingestion.connection_context")
    @patch("polylogue.pipeline.services.ingestion.prepare_ingest")
    def test_ingest_single_source(self, mock_prepare, mock_conn_ctx, mock_iter_source):
        """Single file source is processed correctly."""
        mock_repo = MagicMock()
        mock_repo._db_path = Path("/tmp/test.db")
        mock_config = MagicMock(spec=Config)

        # Setup mocks
        conv = self._make_parsed_conversation("conv-1")
        mock_iter_source.return_value = iter([conv])
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=None)
        mock_prepare.return_value = (
            "conv-1",
            {"conversations": 1, "messages": 1, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
            True,
        )

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))
        result = service.ingest_sources([source])

        assert result.counts["conversations"] == 1
        assert result.counts["messages"] == 1
        assert "conv-1" in result.processed_ids

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    @patch("polylogue.pipeline.services.ingestion.connection_context")
    @patch("polylogue.pipeline.services.ingestion.prepare_ingest")
    def test_ingest_multiple_sources(self, mock_prepare, mock_conn_ctx, mock_iter_source):
        """Multiple sources are processed correctly."""
        mock_repo = MagicMock()
        mock_repo._db_path = Path("/tmp/test.db")
        mock_config = MagicMock(spec=Config)

        # Each source returns one conversation
        conv1 = self._make_parsed_conversation("conv-1")
        conv2 = self._make_parsed_conversation("conv-2")

        call_count = [0]

        def iter_source_side_effect(source, cursor_state=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter([conv1])
            return iter([conv2])

        mock_iter_source.side_effect = iter_source_side_effect
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=None)

        def prepare_side_effect(convo, source_name, **kwargs):
            return (
                convo.provider_conversation_id,
                {"conversations": 1, "messages": 1, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
                True,
            )

        mock_prepare.side_effect = prepare_side_effect

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        sources = [
            Source(name="source-1", path=Path("/tmp/inbox1")),
            Source(name="source-2", path=Path("/tmp/inbox2")),
        ]
        result = service.ingest_sources(sources)

        assert result.counts["conversations"] == 2
        assert "conv-1" in result.processed_ids
        assert "conv-2" in result.processed_ids

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    @patch("polylogue.pipeline.services.ingestion.connection_context")
    @patch("polylogue.pipeline.services.ingestion.prepare_ingest")
    def test_progress_callback_called(self, mock_prepare, mock_conn_ctx, mock_iter_source):
        """Progress callback is invoked for each conversation."""
        mock_repo = MagicMock()
        mock_repo._db_path = Path("/tmp/test.db")
        mock_config = MagicMock(spec=Config)

        conv = self._make_parsed_conversation("conv-1")
        mock_iter_source.return_value = iter([conv])
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=None)
        mock_prepare.return_value = (
            "conv-1",
            {"conversations": 1, "messages": 1, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
            True,
        )

        callback = MagicMock()

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))
        service.ingest_sources([source], progress_callback=callback)

        callback.assert_called_with(1, desc="Ingesting")

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    @patch("polylogue.pipeline.services.ingestion.connection_context")
    @patch("polylogue.pipeline.services.ingestion.prepare_ingest")
    @patch("polylogue.pipeline.services.ingestion.invalidate_search_cache")
    def test_search_cache_invalidated_on_changes(self, mock_invalidate, mock_prepare, mock_conn_ctx, mock_iter_source):
        """Search cache is invalidated when conversations are processed."""
        mock_repo = MagicMock()
        mock_repo._db_path = Path("/tmp/test.db")
        mock_config = MagicMock(spec=Config)

        conv = self._make_parsed_conversation("conv-1")
        mock_iter_source.return_value = iter([conv])
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=None)
        mock_prepare.return_value = (
            "conv-1",
            {"conversations": 1, "messages": 1, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0},
            True,
        )

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))
        service.ingest_sources([source])

        mock_invalidate.assert_called_once()

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    @patch("polylogue.pipeline.services.ingestion.connection_context")
    @patch("polylogue.pipeline.services.ingestion.prepare_ingest")
    @patch("polylogue.pipeline.services.ingestion.invalidate_search_cache")
    def test_search_cache_not_invalidated_when_no_changes(self, mock_invalidate, mock_prepare, mock_conn_ctx, mock_iter_source):
        """Search cache is NOT invalidated when no conversations processed."""
        mock_repo = MagicMock()
        mock_repo._db_path = Path("/tmp/test.db")
        mock_config = MagicMock(spec=Config)

        # Empty source
        mock_iter_source.return_value = iter([])

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))
        service.ingest_sources([source])

        mock_invalidate.assert_not_called()

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    @patch("polylogue.pipeline.services.ingestion.connection_context")
    @patch("polylogue.pipeline.services.ingestion.prepare_ingest")
    def test_error_in_prepare_ingest_propagates(self, mock_prepare, mock_conn_ctx, mock_iter_source):
        """Errors in prepare_ingest are propagated."""
        mock_repo = MagicMock()
        mock_repo._db_path = Path("/tmp/test.db")
        mock_config = MagicMock(spec=Config)

        conv = self._make_parsed_conversation("conv-1")
        mock_iter_source.return_value = iter([conv])
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=None)
        mock_prepare.side_effect = ValueError("Test error")

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))

        with pytest.raises(ValueError, match="Test error"):
            service.ingest_sources([source])


class TestIterSourceConversationsSafe:
    """Tests for IngestionService._iter_source_conversations_safe method."""

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    def test_file_source_uses_iter_source_conversations(self, mock_iter_source):
        """File sources use iter_source_conversations."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)

        mock_iter_source.return_value = iter([])

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))
        list(service._iter_source_conversations_safe(source=source, ui=None, download_assets=True))

        mock_iter_source.assert_called_once()

    @patch("polylogue.pipeline.services.ingestion.iter_drive_conversations")
    def test_drive_source_uses_iter_drive_conversations(self, mock_iter_drive):
        """Drive sources use iter_drive_conversations."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)
        mock_config.drive_config = MagicMock()

        mock_iter_drive.return_value = iter([])

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="drive-source", folder="some-folder-id")
        list(service._iter_source_conversations_safe(source=source, ui=None, download_assets=True))

        mock_iter_drive.assert_called_once()

    @patch("polylogue.pipeline.services.ingestion.iter_drive_conversations")
    def test_drive_auth_error_handled_gracefully(self, mock_iter_drive):
        """DriveAuthError is caught and logged, not propagated."""
        from polylogue.ingestion import DriveAuthError

        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)
        mock_config.drive_config = MagicMock()

        mock_iter_drive.side_effect = DriveAuthError("Token expired")

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="drive-source", folder="some-folder-id")
        result = list(service._iter_source_conversations_safe(source=source, ui=None, download_assets=True))

        # No exception raised, empty result
        assert result == []

    @patch("polylogue.pipeline.services.ingestion.iter_drive_conversations")
    def test_drive_auth_error_updates_cursor_state(self, mock_iter_drive):
        """DriveAuthError updates cursor_state with error info."""
        from polylogue.ingestion import DriveAuthError

        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)
        mock_config.drive_config = MagicMock()

        mock_iter_drive.side_effect = DriveAuthError("Token expired")

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="drive-source", folder="some-folder-id")
        cursor_state: dict[str, Any] = {"error_count": 0}

        list(service._iter_source_conversations_safe(source=source, ui=None, download_assets=True, cursor_state=cursor_state))

        assert cursor_state["error_count"] == 1
        assert cursor_state["latest_error"] == "Token expired"
        assert cursor_state["latest_error_source"] == "drive-source"

    @patch("polylogue.pipeline.services.ingestion.iter_source_conversations")
    def test_cursor_state_passed_to_file_iterator(self, mock_iter_source):
        """cursor_state is passed through to iter_source_conversations."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)

        mock_iter_source.return_value = iter([])

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))
        cursor_state = {"last_processed": "conv-123"}

        list(service._iter_source_conversations_safe(source=source, ui=None, download_assets=True, cursor_state=cursor_state))

        mock_iter_source.assert_called_once_with(source, cursor_state=cursor_state)

    @patch("polylogue.pipeline.services.ingestion.iter_drive_conversations")
    def test_cursor_state_passed_to_drive_iterator(self, mock_iter_drive):
        """cursor_state is passed through to iter_drive_conversations."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)
        mock_config.drive_config = MagicMock()

        mock_iter_drive.return_value = iter([])

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="drive-source", folder="folder-id")
        cursor_state = {"last_processed": "conv-123"}

        list(service._iter_source_conversations_safe(source=source, ui=None, download_assets=True, cursor_state=cursor_state))

        call_kwargs = mock_iter_drive.call_args.kwargs
        assert call_kwargs["cursor_state"] is cursor_state


# ============================================================================
# Integration Tests
# ============================================================================


class TestIngestionServiceIntegration:
    """Integration tests for IngestionService with real database."""

    def test_ingest_with_real_database(self, cli_workspace, tmp_path):
        """Full ingestion flow with real database."""
        import json

        from polylogue.config import Config, Source
        from polylogue.storage.backends.sqlite import SQLiteBackend
        from polylogue.storage.repository import StorageRepository

        # Create test conversation file
        inbox = cli_workspace["inbox_dir"]
        conv_data = {
            "id": "test-conv-1",
            "title": "Test Conversation",
            "create_time": 1700000000,
            "update_time": 1700000100,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "children": ["msg1"],
                },
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello, world!"]},
                        "create_time": 1700000050,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        (inbox / "conversations.json").write_text(json.dumps([conv_data]))

        # Create repository with real backend (using explicit db_path)
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        repository = StorageRepository(backend=backend)

        # Create config
        config = Config(
            version=2,
            archive_root=cli_workspace["archive_root"],
            render_root=cli_workspace["render_root"],
            sources=[Source(name="test-inbox", path=inbox)],
            path=cli_workspace["config_path"],
        )

        # Create service and ingest
        service = IngestionService(
            repository=repository,
            archive_root=cli_workspace["archive_root"],
            config=config,
        )

        result = service.ingest_sources(config.sources)

        # Verify results
        assert result.counts["conversations"] >= 1
        assert len(result.processed_ids) >= 1
