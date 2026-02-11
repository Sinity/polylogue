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
        renderer = MarkdownRenderer(archive_root=tmp_path / "archive")
        service = RenderService(renderer=renderer, render_root=tmp_path / "render")
        assert service.renderer is renderer
        assert service.render_root == tmp_path / "render"

    def test_render_conversations_empty_list(self, tmp_path: Path):
        """RenderService should handle empty conversation list."""
        from polylogue.rendering.renderers import MarkdownRenderer
        renderer = MarkdownRenderer(archive_root=tmp_path / "archive")
        service = RenderService(renderer=renderer, render_root=tmp_path / "render")
        result = service.render_conversations([])
        assert result.rendered_count == 0
        assert result.failures == []

    def test_render_conversations_tracks_failures(self, tmp_path: Path):
        """RenderService should track failures when rendering fails."""
        from polylogue.rendering.renderers import MarkdownRenderer
        mock_renderer = MagicMock(spec=MarkdownRenderer)
        mock_renderer.render.side_effect = lambda cid, _: (_ for _ in ()).throw(ValueError("Test error")) if "fail" in cid else None
        service = RenderService(renderer=mock_renderer, render_root=tmp_path / "render")
        result = service.render_conversations(["success-1", "fail-1", "success-2"])
        assert result.rendered_count == 2 and len(result.failures) == 1
        assert result.failures[0]["conversation_id"] == "fail-1" and "Test error" in result.failures[0]["error"]


# ============================================================================
# IndexService Tests
# ============================================================================


class TestIndexService:
    """Tests for IndexService."""

    def test_initialization(self, tmp_path: Path):
        """IndexService should initialize with config."""
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        service = IndexService(config)
        assert service.config is config

    def test_update_index_empty_list(self, tmp_path: Path, workspace_env):
        """IndexService should handle empty conversation list."""
        from polylogue.storage.backends.sqlite import connection_context
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        with connection_context(None) as conn:
            service = IndexService(config, conn)
            assert service.update_index([]) is True

    def test_get_index_status_when_no_index(self, tmp_path: Path):
        """IndexService should return status when index doesn't exist."""
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        service = IndexService(config)
        status = service.get_index_status()
        assert "exists" in status and "count" in status


# ============================================================================
# AcquireResult Tests
# ============================================================================


class TestAcquireResult:
    """Tests for AcquireResult."""

    @pytest.mark.parametrize("count_field", ["acquired", "skipped", "errors"])
    def test_counts_initialized_to_zero(self, count_field):
        """All count fields start at zero."""
        assert AcquireResult().counts[count_field] == 0



# ============================================================================
# AcquisitionService Tests
# ============================================================================


class TestAcquisitionServiceAcquireSources:
    """Tests for AcquisitionService.acquire_sources method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    def test_init_sets_backend(self, backend: SQLiteBackend):
        """Backend is stored on init."""
        service = AcquisitionService(backend=backend)
        assert service.backend is backend

    def test_acquire_empty_sources(self, backend: SQLiteBackend):
        """Empty sources list returns empty result."""
        result = AcquisitionService(backend=backend).acquire_sources([])
        assert all([result.counts[k] == 0 for k in ["acquired", "skipped", "errors"]]) and result.raw_ids == []

    @pytest.mark.parametrize(
        "num_convos,expected_acquired,expected_skipped,raw_content",
        [
            (1, 1, 0, b'{"id": "test-conv", "messages": []}'),
            (3, 3, 0, b'{"id": "conv-0"}'),
            (2, 1, 1, b'{"id": "same-conv"}'),
        ],
    )
    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_conversations(
        self,
        mock_iter,
        backend: SQLiteBackend,
        num_convos: int,
        expected_acquired: int,
        expected_skipped: int,
        raw_content: bytes,
    ):
        """Acquire conversations: single, multiple, or duplicates."""
        if num_convos == 1:
            raw_data = RawConversationData(
                raw_bytes=raw_content,
                source_path="/tmp/test.json",
                source_index=0,
                provider_hint="chatgpt",
            )
            mock_iter.return_value = iter([(raw_data, MagicMock())])
        elif num_convos == 3:
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
        else:  # num_convos == 2
            raw_data = RawConversationData(
                raw_bytes=raw_content,
                source_path="/tmp/test.json",
                source_index=0,
                provider_hint="chatgpt",
            )
            mock_iter.return_value = iter([(raw_data, MagicMock()), (raw_data, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))
        result = service.acquire_sources([source])

        assert result.counts["acquired"] == expected_acquired
        assert result.counts["skipped"] == expected_skipped
        assert len(result.raw_ids) == expected_acquired

        if expected_acquired > 0:
            stored = backend.get_raw_conversation(result.raw_ids[0])
            assert stored is not None
            assert stored.raw_content == (raw_content if num_convos != 3 else b'{"id": "conv-0"}')
            assert stored.provider_name == "chatgpt"

    @pytest.mark.parametrize(
        "provider_hint,source_name,expected_provider",
        [("chatgpt", "test-source", "chatgpt"), (None, "my-inbox", "my-inbox")],
    )
    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_provider_fallback(
        self,
        mock_iter,
        backend: SQLiteBackend,
        provider_hint: str | None,
        source_name: str,
        expected_provider: str,
    ):
        """Source name is used as fallback provider when provider_hint is None."""
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint=provider_hint,
        )
        mock_iter.return_value = iter([(raw_data, MagicMock())])
        service = AcquisitionService(backend=backend)
        source = Source(name=source_name, path=Path("/tmp/inbox"))
        result = service.acquire_sources([source])
        stored = backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.provider_name == expected_provider

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

    @pytest.mark.parametrize("error_scenario", ["iteration_error", "none_raw_data"])
    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_handles_errors(self, mock_iter, backend: SQLiteBackend, error_scenario: str):
        """Errors during source iteration are counted."""
        if error_scenario == "iteration_error":
            mock_iter.side_effect = ValueError("File not found")
        else:
            mock_iter.return_value = iter([(None, MagicMock())])
        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))
        result = service.acquire_sources([source])
        assert result.counts["errors"] == 1
        assert result.counts["acquired"] == 0


class TestAcquisitionServiceIntegration:
    """Integration tests for AcquisitionService with real files."""

    def _make_conv(self, cid: str, title: str, time: int, msg: str) -> dict:
        return {"id": cid, "title": title, "create_time": time, "update_time": time+100, "mapping": {
            "root": {"id": "root", "message": None, "children": ["msg1"]},
            "msg1": {"id": "msg1", "message": {"id": "msg1", "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [msg]}, "create_time": time+50},
                "parent": "root", "children": []}}}

    def test_acquire_real_chatgpt_file(self, tmp_path: Path):
        """Acquire from a real ChatGPT conversations.json file."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "conversations.json").write_text(json.dumps([self._make_conv("conv-1", "Test Chat", 1700000000, "Hello")]))
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = AcquisitionService(backend=backend).acquire_sources([Source(name="chatgpt-inbox", path=inbox)])
        assert result.counts["acquired"] == 1 and result.counts["errors"] == 0 and len(result.raw_ids) == 1
        stored = backend.get_raw_conversation(result.raw_ids[0])
        data = json.loads(stored.raw_content)
        assert stored.provider_name == "chatgpt" and data["id"] == "conv-1" and data["title"] == "Test Chat"

    def test_acquire_multiple_json_files(self, tmp_path: Path):
        """Acquire from multiple JSON files in a directory."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        convs = [self._make_conv("conv-1", "Chat 1", 1700000000, "Hello"),
                 self._make_conv("conv-2", "Chat 2", 1700000200, "World")]
        (inbox / "conversations.json").write_text(json.dumps(convs))
        result = AcquisitionService(backend=SQLiteBackend(db_path=tmp_path / "test.db")).acquire_sources([Source(name="chatgpt-export", path=inbox)])
        assert result.counts["acquired"] == 2 and len(result.raw_ids) == 2 and len(set(result.raw_ids)) == 2


# ============================================================================
# IngestResult Tests
# ============================================================================


class TestIngestResultInit:
    """Tests for IngestResult initialization."""

    @pytest.mark.parametrize("count_field", ["conversations", "messages", "attachments", "skipped_conversations", "skipped_messages", "skipped_attachments"])
    def test_counts_initialized_to_zero(self, count_field):
        """All count fields start at zero."""
        assert IngestResult().counts[count_field] == 0

    @pytest.mark.parametrize("changed_count_field", ["conversations", "messages", "attachments"])
    def test_changed_counts_initialized_to_zero(self, changed_count_field):
        """All changed_counts fields start at zero."""
        assert IngestResult().changed_counts[changed_count_field] == 0



class TestIngestResultMerge:
    """Tests for IngestResult.merge_result method."""

    @pytest.mark.parametrize("conv_id,result_counts,content_changed,exp_c,exp_m,exp_a,exp_s,in_p", [
        ("conv1", {"conversations": 1, "messages": 5, "attachments": 2, "skipped_conversations": 0, "skipped_messages": 1, "skipped_attachments": 0}, True, 1, 5, 2, 1, True),
        ("conv1", {"conversations": 1, "messages": 2, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0}, True, 1, 2, 0, 0, True),
        ("conv1", {"conversations": 0, "messages": 0, "attachments": 0, "skipped_conversations": 1, "skipped_messages": 0, "skipped_attachments": 0}, False, 0, 0, 0, 0, False),
        ("conv-123", {"conversations": 1, "messages": 1, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0}, True, 1, 1, 0, 0, True),
        ("conv-456", {"conversations": 1, "messages": 5, "attachments": 0, "skipped_conversations": 0, "skipped_messages": 0, "skipped_attachments": 0}, False, 1, 5, 0, 0, True),
        ("conv-skipped", {"conversations": 0, "messages": 0, "attachments": 0, "skipped_conversations": 1, "skipped_messages": 5, "skipped_attachments": 0}, False, 0, 0, 0, 5, False),
    ])
    def test_merge_result_scenarios(self, conv_id: str, result_counts: dict, content_changed: bool, exp_c: int, exp_m: int, exp_a: int, exp_s: int, in_p: bool):
        """Parametrized merge scenarios."""
        result = IngestResult()
        result.merge_result(conversation_id=conv_id, result_counts=result_counts, content_changed=content_changed)
        assert result.counts["conversations"] == exp_c and result.counts["messages"] == exp_m
        assert result.counts["attachments"] == exp_a and result.counts["skipped_messages"] == exp_s
        assert (conv_id in result.processed_ids) == in_p

    def test_merge_multiple_conversations(self):
        """Multiple merges accumulate correctly."""
        result = IngestResult()
        result.merge_result(
            "conv1",
            {
                "conversations": 1,
                "messages": 3,
                "attachments": 1,
                "skipped_conversations": 0,
                "skipped_messages": 0,
                "skipped_attachments": 0,
            },
            content_changed=True,
        )
        result.merge_result(
            "conv2",
            {
                "conversations": 1,
                "messages": 7,
                "attachments": 0,
                "skipped_conversations": 0,
                "skipped_messages": 2,
                "skipped_attachments": 0,
            },
            content_changed=True,
        )
        assert result.counts["conversations"] == 2
        assert result.counts["messages"] == 10
        assert result.counts["attachments"] == 1
        assert result.counts["skipped_messages"] == 2

    def test_merge_thread_safe(self):
        """Concurrent merges don't cause race conditions."""
        result = IngestResult()
        errors: list[Exception] = []

        def merge_batch(start_id: int) -> None:
            try:
                for i in range(100):
                    result.merge_result(
                        f"conv-{start_id}-{i}",
                        {
                            "conversations": 1,
                            "messages": 2,
                            "attachments": 1,
                            "skipped_conversations": 0,
                            "skipped_messages": 0,
                            "skipped_attachments": 0,
                        },
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

    @pytest.mark.parametrize(
        "attr_name",
        ["repository", "archive_root", "config"],
    )
    def test_init_attributes(self, attr_name: str):
        """Service initialization sets required attributes."""
        mock_repo = MagicMock()
        mock_config = MagicMock(spec=Config)
        archive_path = Path("/tmp/archive")

        service = IngestionService(
            repository=mock_repo,
            archive_root=archive_path,
            config=mock_config,
        )

        if attr_name == "repository":
            assert service.repository is mock_repo
        elif attr_name == "archive_root":
            assert service.archive_root == archive_path
        elif attr_name == "config":
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

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService"
        ) as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service
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

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService"
        ) as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service

            acquire_result = AcquireResult()
            acquire_result.raw_ids = ["raw-1", "raw-2"]
            acquire_result.counts["acquired"] = 2
            mock_acquire_service.acquire_sources.return_value = acquire_result

            mock_ingest_result = IngestResult()
            mock_ingest_result.counts["conversations"] = 2
            mock_ingest_result.counts["messages"] = 5
            mock_ingest_result.processed_ids = {"conv-1", "conv-2"}
            with patch.object(
                service, "ingest_from_raw", return_value=mock_ingest_result
            ) as mock_parse:
                source = Source(name="test-source", path=Path("/tmp/inbox"))
                result = service.ingest_sources([source])

                mock_acquire_service.acquire_sources.assert_called_once()
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

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService"
        ) as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service

            acquire_result = AcquireResult()
            acquire_result.counts["skipped"] = 5
            mock_acquire_service.acquire_sources.return_value = acquire_result

            with patch.object(service, "ingest_from_raw") as mock_parse:
                source = Source(name="test-source", path=Path("/tmp/inbox"))
                result = service.ingest_sources([source])

                mock_parse.assert_not_called()

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

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService"
        ) as mock_acquire_cls:
            mock_acquire_service = MagicMock()
            mock_acquire_cls.return_value = mock_acquire_service

            acquire_result = AcquireResult()
            acquire_result.raw_ids = ["raw-1"]
            mock_acquire_service.acquire_sources.return_value = acquire_result

            mock_ingest_result = IngestResult()
            with patch.object(
                service, "ingest_from_raw", return_value=mock_ingest_result
            ) as mock_parse:
                source = Source(name="test-source", path=Path("/tmp/inbox"))
                service.ingest_sources([source], progress_callback=callback)

                mock_acquire_service.acquire_sources.assert_called_once_with(
                    [source],
                    progress_callback=callback,
                )

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

    def _conv_json(self, cid: str, title: str, msg: str) -> dict:
        return {"id": cid, "title": title, "create_time": 1700000000, "update_time": 1700000100,
            "mapping": {"root": {"id": "root", "message": None, "children": ["msg1"]},
            "msg1": {"id": "msg1", "message": {"id": "msg1", "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [msg]}, "create_time": 1700000050},
                "parent": "root", "children": []}}}

    def test_ingest_with_real_database(self, cli_workspace, tmp_path):
        """Full ingestion flow with real database."""
        inbox = cli_workspace["inbox_dir"]
        (inbox / "conversations.json").write_text(json.dumps([self._conv_json("test-conv-1", "Test Conversation", "Hello, world!")]))
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        config = Config(archive_root=cli_workspace["archive_root"], render_root=cli_workspace["render_root"],
                       sources=[Source(name="test-inbox", path=inbox)])
        result = IngestionService(repository=ConversationRepository(backend=backend),
                                 archive_root=cli_workspace["archive_root"], config=config).ingest_sources(config.sources)
        assert result.counts["conversations"] >= 1 and len(result.processed_ids) >= 1

    def test_ingest_from_raw_parses_stored_conversations(self, cli_workspace, tmp_path):
        """Full acquire → parse flow using database-driven testing."""
        inbox = cli_workspace["inbox_dir"]
        (inbox / "conversations.json").write_text(json.dumps([self._conv_json("test-conv-raw", "Test Raw Conversation", "Hello from raw!")]))
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        config = Config(archive_root=cli_workspace["archive_root"], render_root=cli_workspace["render_root"],
                       sources=[Source(name="test-inbox", path=inbox)])
        raw_ids = AcquisitionService(backend=backend).acquire_sources(config.sources).raw_ids
        assert len(raw_ids) == 1
        parse_result = IngestionService(repository=ConversationRepository(backend=backend),
                                       archive_root=cli_workspace["archive_root"], config=config).ingest_from_raw(raw_ids=raw_ids)
        assert parse_result.counts["conversations"] >= 1 and len(parse_result.processed_ids) >= 1
        with backend._get_connection() as conn:
            row = conn.execute("SELECT raw_id FROM conversations WHERE conversation_id = ?",
                             (list(parse_result.processed_ids)[0],)).fetchone()
        assert row is not None and row["raw_id"] == raw_ids[0]
