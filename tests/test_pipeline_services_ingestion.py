"""Tests for pipeline/services/ingestion.py."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services.ingestion import IngestionService, IngestResult

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
    """Tests for IngestionService.ingest_sources method.

    ingest_sources() orchestrates:
    1. ACQUIRE stage via AcquisitionService.acquire_sources()
    2. PARSE stage via self.ingest_from_raw()

    These tests mock the stage boundaries to verify orchestration logic.
    """

    def test_ingest_empty_sources_returns_empty_result(self):
        """Empty sources list returns empty IngestResult (no acquisition needed)."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo._backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        with patch("polylogue.pipeline.services.acquisition.AcquisitionService") as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service
            # Empty sources → empty result from acquire
            from polylogue.pipeline.services.acquisition import AcquireResult
            mock_acquire_service.acquire_sources.return_value = AcquireResult()

            result = service.ingest_sources([])

        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0
        assert len(result.processed_ids) == 0

    def test_ingest_calls_acquire_then_parse(self):
        """ingest_sources calls acquire stage, then parse stage with returned raw_ids."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo._backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        with patch("polylogue.pipeline.services.acquisition.AcquisitionService") as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service

            # Acquisition returns some raw_ids
            from polylogue.pipeline.services.acquisition import AcquireResult
            acquire_result = AcquireResult()
            acquire_result.raw_ids = ["raw-1", "raw-2"]
            acquire_result.counts["acquired"] = 2
            mock_acquire_service.acquire_sources.return_value = acquire_result

            # Mock ingest_from_raw
            mock_ingest_result = IngestResult()
            mock_ingest_result.counts["conversations"] = 2
            mock_ingest_result.counts["messages"] = 5
            mock_ingest_result.processed_ids = {"conv-1", "conv-2"}
            with patch.object(service, "ingest_from_raw", return_value=mock_ingest_result) as mock_parse:
                source = Source(name="test-source", path=Path("/tmp/inbox"))
                result = service.ingest_sources([source])

                # Verify acquire was called
                mock_acquire_service.acquire_sources.assert_called_once()

                # Verify parse was called with the raw_ids from acquire
                mock_parse.assert_called_once_with(
                    raw_ids=["raw-1", "raw-2"],
                    progress_callback=None,
                )

        assert result.counts["conversations"] == 2
        assert result.counts["messages"] == 5
        assert result.processed_ids == {"conv-1", "conv-2"}

    def test_ingest_skips_parse_when_nothing_acquired(self):
        """If acquisition returns no raw_ids, parse stage is skipped."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo._backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        with patch("polylogue.pipeline.services.acquisition.AcquisitionService") as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service

            # Acquisition returns empty (e.g., all duplicates skipped)
            from polylogue.pipeline.services.acquisition import AcquireResult
            acquire_result = AcquireResult()
            acquire_result.counts["skipped"] = 5
            mock_acquire_service.acquire_sources.return_value = acquire_result

            # Mock ingest_from_raw to verify it's NOT called
            with patch.object(service, "ingest_from_raw") as mock_parse:
                source = Source(name="test-source", path=Path("/tmp/inbox"))
                result = service.ingest_sources([source])

                mock_parse.assert_not_called()

        # Empty result returned
        assert result.counts["conversations"] == 0

    def test_progress_callback_passed_to_both_stages(self):
        """Progress callback is forwarded to both acquire and parse stages."""
        mock_repo = MagicMock()
        mock_backend = MagicMock()
        mock_repo._backend = mock_backend
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        callback = MagicMock()

        with patch("polylogue.pipeline.services.acquisition.AcquisitionService") as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service

            from polylogue.pipeline.services.acquisition import AcquireResult
            acquire_result = AcquireResult()
            acquire_result.raw_ids = ["raw-1"]
            mock_acquire_service.acquire_sources.return_value = acquire_result

            mock_ingest_result = IngestResult()
            with patch.object(service, "ingest_from_raw", return_value=mock_ingest_result) as mock_parse:
                source = Source(name="test-source", path=Path("/tmp/inbox"))
                service.ingest_sources([source], progress_callback=callback)

                # Callback passed to acquire
                mock_acquire_service.acquire_sources.assert_called_once_with(
                    [source],
                    progress_callback=callback,
                )

                # Callback passed to parse
                mock_parse.assert_called_once_with(
                    raw_ids=["raw-1"],
                    progress_callback=callback,
                )

    def test_backend_not_initialized_raises(self):
        """RuntimeError raised if repository backend is None."""
        mock_repo = MagicMock()
        mock_repo._backend = None
        mock_config = MagicMock(spec=Config)

        service = IngestionService(
            repository=mock_repo,
            archive_root=Path("/tmp/archive"),
            config=mock_config,
        )

        source = Source(name="test-source", path=Path("/tmp/inbox"))

        with pytest.raises(RuntimeError, match="backend is not initialized"):
            service.ingest_sources([source])


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
            archive_root=cli_workspace["archive_root"],
            render_root=cli_workspace["render_root"],
            sources=[Source(name="test-inbox", path=inbox)],
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

    def test_ingest_from_raw_parses_stored_conversations(self, cli_workspace, tmp_path):
        """Full acquire → parse flow using database-driven testing."""
        import json

        from polylogue.config import Config, Source
        from polylogue.pipeline.services.acquisition import AcquisitionService
        from polylogue.storage.backends.sqlite import SQLiteBackend
        from polylogue.storage.repository import StorageRepository

        # Create test conversation file
        inbox = cli_workspace["inbox_dir"]
        conv_data = {
            "id": "test-conv-raw",
            "title": "Test Raw Conversation",
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
                        "content": {"content_type": "text", "parts": ["Hello from raw!"]},
                        "create_time": 1700000050,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        (inbox / "conversations.json").write_text(json.dumps([conv_data]))

        # Single backend instance - same connection management throughout
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        repository = StorageRepository(backend=backend)

        config = Config(
            archive_root=cli_workspace["archive_root"],
            render_root=cli_workspace["render_root"],
            sources=[Source(name="test-inbox", path=inbox)],
        )

        # Step 1: ACQUIRE - store raw bytes
        acquire_service = AcquisitionService(backend=backend)
        acquire_result = acquire_service.acquire_sources(config.sources)

        assert acquire_result.counts["acquired"] == 1
        raw_ids = acquire_result.raw_ids

        # Step 2: PARSE - read from raw_conversations, parse, store
        # Uses same backend via repository
        ingest_service = IngestionService(
            repository=repository,
            archive_root=cli_workspace["archive_root"],
            config=config,
        )

        parse_result = ingest_service.ingest_from_raw(raw_ids=raw_ids)

        # Verify parsing succeeded
        assert parse_result.counts["conversations"] >= 1
        assert len(parse_result.processed_ids) >= 1

        # Verify the raw_id link is set on the conversation
        with backend._get_connection() as conn:
            row = conn.execute(
                "SELECT raw_id FROM conversations WHERE conversation_id = ?",
                (list(parse_result.processed_ids)[0],),
            ).fetchone()

        assert row is not None
        assert row["raw_id"] == raw_ids[0]  # Linked to raw source!
