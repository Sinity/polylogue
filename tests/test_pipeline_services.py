"""Tests for pipeline service classes.

Consolidated from:
- test_pipeline_services.py (service initialization tests)
- test_pipeline_services_acquisition.py (AcquisitionService tests)
- test_pipeline_services_ingestion.py (IngestionService tests)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services import IndexService, RenderService
from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
from polylogue.pipeline.services.ingestion import IngestionService, IngestResult
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository

# ============================================================================
# RenderService Tests
# ============================================================================


class TestRenderService:
    """Tests for RenderService."""

    def test_initialization(self, tmp_path: Path):
        """RenderService should initialize with required renderer."""
        from polylogue.rendering.renderers import MarkdownRenderer

        render_root = tmp_path / "render"
        archive_root = tmp_path / "archive"
        renderer = MarkdownRenderer(archive_root=archive_root)

        service = RenderService(renderer=renderer, render_root=render_root)

        assert service.renderer is renderer
        assert service.render_root == render_root

    def test_render_conversations_empty_list(self, tmp_path: Path):
        """RenderService should handle empty conversation list."""
        from polylogue.rendering.renderers import MarkdownRenderer

        archive_root = tmp_path / "archive"
        renderer = MarkdownRenderer(archive_root=archive_root)
        service = RenderService(renderer=renderer, render_root=tmp_path / "render")

        result = service.render_conversations([])

        assert result.rendered_count == 0
        assert result.failures == []

    def test_render_conversations_tracks_failures(self, tmp_path: Path):
        """RenderService should track failures when rendering fails."""
        from polylogue.rendering.renderers import MarkdownRenderer

        tmp_path / "archive"
        mock_renderer = MagicMock(spec=MarkdownRenderer)

        def render_side_effect(conversation_id, output_path):
            if "fail" in conversation_id:
                raise ValueError("Test error")

        mock_renderer.render.side_effect = render_side_effect

        service = RenderService(renderer=mock_renderer, render_root=tmp_path / "render")

        result = service.render_conversations(["success-1", "fail-1", "success-2"])

        assert result.rendered_count == 2
        assert len(result.failures) == 1
        assert result.failures[0]["conversation_id"] == "fail-1"
        assert "Test error" in result.failures[0]["error"]


# ============================================================================
# IndexService Tests
# ============================================================================


class TestIndexService:
    """Tests for IndexService."""

    def test_initialization(self, tmp_path: Path):
        """IndexService should initialize with config."""
        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )

        service = IndexService(config)

        assert service.config is config

    def test_update_index_empty_list(self, tmp_path: Path, workspace_env):
        """IndexService should handle empty conversation list."""
        from polylogue.storage.backends.sqlite import connection_context

        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )

        with connection_context(None) as conn:
            service = IndexService(config, conn)

            # Empty list should ensure index exists and return True
            result = service.update_index([])

            assert result is True

    def test_get_index_status_when_no_index(self, tmp_path: Path):
        """IndexService should return status when index doesn't exist."""
        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        service = IndexService(config)

        # Without mocking, this will use actual index_status()
        # which should return exists=False for new database
        status = service.get_index_status()

        assert "exists" in status
        assert "count" in status


# ============================================================================
# AcquireResult Tests
# ============================================================================


class TestAcquireResult:
    """Tests for AcquireResult."""

    def test_counts_initialized_to_zero(self):
        """All count fields start at zero."""
        result = AcquireResult()

        assert result.counts["acquired"] == 0
        assert result.counts["skipped"] == 0
        assert result.counts["errors"] == 0

    def test_raw_ids_initialized_empty(self):
        """raw_ids starts as empty list."""
        result = AcquireResult()

        assert result.raw_ids == []
        assert isinstance(result.raw_ids, list)


# ============================================================================
# AcquisitionService Tests
# ============================================================================


class TestAcquisitionServiceInit:
    """Tests for AcquisitionService initialization."""

    def test_init_sets_backend(self, tmp_path: Path):
        """Backend is stored on init."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        service = AcquisitionService(backend=backend)

        assert service.backend is backend


class TestAcquisitionServiceAcquireSources:
    """Tests for AcquisitionService.acquire_sources method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a SQLiteBackend with a temp database."""
        return SQLiteBackend(db_path=tmp_path / "test.db")

    def test_acquire_empty_sources(self, backend: SQLiteBackend):
        """Empty sources list returns empty result."""
        service = AcquisitionService(backend=backend)

        result = service.acquire_sources([])

        assert result.counts["acquired"] == 0
        assert result.counts["skipped"] == 0
        assert result.counts["errors"] == 0
        assert result.raw_ids == []

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_single_conversation(self, mock_iter, backend: SQLiteBackend):
        """Single conversation is acquired correctly."""
        raw_bytes = b'{"id": "test-conv", "messages": []}'
        raw_data = RawConversationData(
            raw_bytes=raw_bytes,
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint="chatgpt",
        )
        mock_iter.return_value = iter([(raw_data, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 1
        assert result.counts["skipped"] == 0
        assert len(result.raw_ids) == 1

        # Verify stored in database
        stored = backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.raw_content == raw_bytes
        assert stored.provider_name == "chatgpt"

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_multiple_conversations(self, mock_iter, backend: SQLiteBackend):
        """Multiple conversations are acquired correctly."""
        convos = [
            RawConversationData(
                raw_bytes=f'{{"id": "conv-{i}"}}'.encode(),
                source_path="/tmp/test.json",
                source_index=i,
                provider_hint="chatgpt",
            )
            for i in range(3)
        ]
        mock_iter.return_value = iter([(r, MagicMock()) for r in convos])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 3
        assert len(result.raw_ids) == 3

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_skips_duplicates(self, mock_iter, backend: SQLiteBackend):
        """Duplicate raw_ids are skipped."""
        raw_bytes = b'{"id": "same-conv"}'
        raw_data = RawConversationData(
            raw_bytes=raw_bytes,
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint="chatgpt",
        )
        # Same conversation twice
        mock_iter.return_value = iter([(raw_data, MagicMock()), (raw_data, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 1
        assert result.counts["skipped"] == 1

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_uses_source_name_as_fallback_provider(self, mock_iter, backend: SQLiteBackend):
        """Source name is used as provider_name when provider_hint is None."""
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint=None,  # No hint
        )
        mock_iter.return_value = iter([(raw_data, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="my-inbox", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        stored = backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.provider_name == "my-inbox"

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_progress_callback_called(self, mock_iter, backend: SQLiteBackend):
        """Progress callback is invoked for each conversation."""
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
        )
        mock_iter.return_value = iter([(raw_data, MagicMock())])

        callback = MagicMock()

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        service.acquire_sources([source], progress_callback=callback)

        callback.assert_called_with(1, desc="Acquiring")

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_handles_iteration_error(self, mock_iter, backend: SQLiteBackend):
        """Errors during source iteration are counted."""
        mock_iter.side_effect = ValueError("File not found")

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["errors"] == 1
        assert result.counts["acquired"] == 0

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_handles_none_raw_data(self, mock_iter, backend: SQLiteBackend):
        """None raw_data is counted as error."""
        mock_iter.return_value = iter([(None, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["errors"] == 1
        assert result.counts["acquired"] == 0


class TestAcquisitionServiceIntegration:
    """Integration tests for AcquisitionService with real files."""

    def test_acquire_real_chatgpt_file(self, tmp_path: Path):
        """Acquire from a real ChatGPT conversations.json file."""
        # Create a realistic ChatGPT export
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        conv_data = [
            {
                "id": "conv-1",
                "title": "Test Chat",
                "create_time": 1700000000,
                "update_time": 1700000100,
                "mapping": {
                    "root": {"id": "root", "message": None, "children": ["msg1"]},
                    "msg1": {
                        "id": "msg1",
                        "message": {
                            "id": "msg1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Hello"]},
                            "create_time": 1700000050,
                        },
                        "parent": "root",
                        "children": [],
                    },
                },
            },
        ]
        (inbox / "conversations.json").write_text(json.dumps(conv_data))

        # Create backend and service
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        service = AcquisitionService(backend=backend)

        source = Source(name="chatgpt-inbox", path=inbox)

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 1
        assert result.counts["errors"] == 0
        assert len(result.raw_ids) == 1

        # Verify the raw content is stored
        stored = backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.provider_name == "chatgpt"

        # Parse the stored content to verify it's valid JSON
        stored_data = json.loads(stored.raw_content)
        assert stored_data["id"] == "conv-1"
        assert stored_data["title"] == "Test Chat"

    def test_acquire_multiple_json_files(self, tmp_path: Path):
        """Acquire from multiple JSON files in a directory."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        # Create two separate JSON files
        conv1 = {
            "id": "conv-1",
            "title": "Chat 1",
            "create_time": 1700000000,
            "update_time": 1700000100,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["msg1"]},
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                        "create_time": 1700000050,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        conv2 = {
            "id": "conv-2",
            "title": "Chat 2",
            "create_time": 1700000200,
            "update_time": 1700000300,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["msg2"]},
                "msg2": {
                    "id": "msg2",
                    "message": {
                        "id": "msg2",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["World"]},
                        "create_time": 1700000250,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }

        # Write as separate files (ChatGPT exports multiple conversations in one file)
        (inbox / "conversations.json").write_text(json.dumps([conv1, conv2]))

        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        service = AcquisitionService(backend=backend)

        source = Source(name="chatgpt-export", path=inbox)

        result = service.acquire_sources([source])

        # ChatGPT bundle: each conversation in the array is acquired separately
        assert result.counts["acquired"] == 2
        assert len(result.raw_ids) == 2

        # Verify both are stored with distinct raw_ids
        assert len(set(result.raw_ids)) == 2  # All unique


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
# IngestionService Integration Tests
# ============================================================================


class TestIngestionServiceIntegration:
    """Integration tests for IngestionService with real database."""

    def test_ingest_with_real_database(self, cli_workspace, tmp_path):
        """Full ingestion flow with real database."""
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
        repository = ConversationRepository(backend=backend)

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
        repository = ConversationRepository(backend=backend)

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
