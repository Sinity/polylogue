"""Tests for async pipeline service classes.

Covers:
- AcquisitionService: acquire_sources, _store_raw
- RenderService: render_conversations, semaphore limiting
- IndexService: ensure_index_exists, rebuild_index, update_index, get_index_status
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
from polylogue.pipeline.services.indexing import IndexService
from polylogue.pipeline.services.rendering import RenderService, RenderResult
from polylogue.storage.backends.async_sqlite import SQLiteBackend


# =============================================================================
# AcquisitionService Tests
# =============================================================================


class TestAcquireResult:
    """Tests for AcquireResult data class."""

    def test_initial_counts(self):
        """Fresh result has zero counts."""
        result = AcquireResult()
        assert result.counts == {"acquired": 0, "skipped": 0, "errors": 0}
        assert result.raw_ids == []


class TestAcquisitionService:
    """Tests for AcquisitionService."""

    @pytest.mark.asyncio
    async def test_acquire_empty_sources(self):
        """Empty source list returns zero counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            service = AcquisitionService(backend)

            result = await service.acquire_sources([])

            assert result.counts["acquired"] == 0
            assert result.counts["skipped"] == 0
            assert result.counts["errors"] == 0
            assert result.raw_ids == []
            await backend.close()

    @pytest.mark.asyncio
    async def test_acquire_progress_callback(self):
        """Progress callback is invoked per conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            service = AcquisitionService(backend)

            progress_calls = []

            def progress(count, desc=""):
                progress_calls.append((count, desc))

            # Mock iter to return some data
            mock_raw = MagicMock()
            mock_raw.raw_bytes = b"test content"
            mock_raw.provider_hint = "chatgpt"
            mock_raw.source_path = "test.json"
            mock_raw.source_index = 0
            mock_raw.file_mtime = None

            with patch.object(service, "_iter_source_conversations", return_value=[(mock_raw, None)]):
                source = Source(name="test", path=Path(tmpdir))
                result = await service.acquire_sources([source], progress_callback=progress)

            assert len(progress_calls) == 1
            assert progress_calls[0] == (1, "Acquiring")
            assert result.counts["acquired"] == 1
            await backend.close()

    @pytest.mark.asyncio
    async def test_acquire_error_handling(self):
        """Source iteration error increments error count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            service = AcquisitionService(backend)

            with patch.object(service, "_iter_source_conversations", side_effect=ValueError("source broken")):
                source = Source(name="broken", path=Path(tmpdir))
                result = await service.acquire_sources([source])

            assert result.counts["errors"] == 1
            assert result.counts["acquired"] == 0
            await backend.close()

    @pytest.mark.asyncio
    async def test_acquire_duplicate_skipped(self):
        """Second acquire of same content is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            service = AcquisitionService(backend)

            mock_raw = MagicMock()
            mock_raw.raw_bytes = b"duplicate content"
            mock_raw.provider_hint = "chatgpt"
            mock_raw.source_path = "test.json"
            mock_raw.source_index = 0
            mock_raw.file_mtime = None

            source = Source(name="test", path=Path(tmpdir))

            with patch.object(service, "_iter_source_conversations", return_value=[(mock_raw, None)]):
                # First acquire
                result1 = await service.acquire_sources([source])
                assert result1.counts["acquired"] == 1

                # Second acquire of same content
                result2 = await service.acquire_sources([source])
                assert result2.counts["skipped"] == 1
                assert result2.counts["acquired"] == 0

            await backend.close()


# =============================================================================
# RenderService Tests
# =============================================================================


class TestRenderResult:
    """Tests for RenderResult data class."""

    def test_initial_state(self):
        """Fresh result has zero count and empty failures."""
        result = RenderResult()
        assert result.rendered_count == 0
        assert result.failures == []

    def test_record_success(self):
        """record_success increments count."""
        result = RenderResult()
        result.record_success()
        result.record_success()
        assert result.rendered_count == 2

    def test_record_failure(self):
        """record_failure appends failure info."""
        result = RenderResult()
        result.record_failure("conv-1", "boom")
        assert len(result.failures) == 1
        assert result.failures[0] == {"conversation_id": "conv-1", "error": "boom"}


class TestRenderService:
    """Tests for RenderService."""

    @pytest.mark.asyncio
    async def test_render_empty_ids(self):
        """No IDs returns zero rendered."""
        renderer = MagicMock()
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations([])

        assert result.rendered_count == 0
        assert result.failures == []
        renderer.render.assert_not_called()

    @pytest.mark.asyncio
    async def test_render_success(self):
        """Renders conversations, increments count."""
        renderer = MagicMock()
        renderer.render.return_value = None  # Success
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations(["conv-1", "conv-2", "conv-3"])

        assert result.rendered_count == 3
        assert result.failures == []
        assert renderer.render.call_count == 3

    @pytest.mark.asyncio
    async def test_render_failure_tracked(self):
        """Render errors collected in failures list."""
        renderer = MagicMock()

        def render_side_effect(conv_id, root):
            if conv_id == "conv-bad":
                raise ValueError("render exploded")

        renderer.render.side_effect = render_side_effect
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations(["conv-ok", "conv-bad", "conv-ok2"])

        assert result.rendered_count == 2
        assert len(result.failures) == 1
        assert result.failures[0]["conversation_id"] == "conv-bad"
        assert "render exploded" in result.failures[0]["error"]

    @pytest.mark.asyncio
    async def test_render_concurrency_limit(self):
        """Semaphore limits concurrent renders."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        renderer = MagicMock()

        async def tracked_render(conv_id, root):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1

        # Patch asyncio.to_thread to use our tracked function
        with patch("polylogue.pipeline.services.rendering.asyncio.to_thread", side_effect=tracked_render):
            service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))
            await service.render_conversations(
                [f"conv-{i}" for i in range(10)],
                max_workers=2,
            )

        assert max_concurrent <= 2


# =============================================================================
# IndexService Tests
# =============================================================================


class TestIndexService:
    """Tests for IndexService."""

    @pytest.mark.asyncio
    async def test_ensure_index_exists(self):
        """Creates FTS table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            config = Config(sources=[], archive_root=Path(tmpdir), render_root=Path(tmpdir) / "render")
            service = IndexService(config=config, backend=backend)

            result = await service.ensure_index_exists()

            assert result is True

            # Verify FTS table was created
            status = await service.get_index_status()
            assert status["exists"] is True
            assert status["count"] == 0
            await backend.close()

    @pytest.mark.asyncio
    async def test_rebuild_index(self):
        """Full rebuild succeeds on empty DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            config = Config(sources=[], archive_root=Path(tmpdir), render_root=Path(tmpdir) / "render")
            service = IndexService(config=config, backend=backend)

            result = await service.rebuild_index()

            assert result is True
            await backend.close()

    @pytest.mark.asyncio
    async def test_update_index_empty_ids(self):
        """Empty conversation_ids list still ensures index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            config = Config(sources=[], archive_root=Path(tmpdir), render_root=Path(tmpdir) / "render")
            service = IndexService(config=config, backend=backend)

            result = await service.update_index([])

            assert result is True
            # Index should exist now
            status = await service.get_index_status()
            assert status["exists"] is True
            await backend.close()

    @pytest.mark.asyncio
    async def test_update_index_no_backend(self):
        """Update without backend returns False."""
        config = Config(sources=[], archive_root=Path("/tmp"), render_root=Path("/tmp/render"))
        service = IndexService(config=config, backend=None)

        result = await service.update_index(["conv-1"])

        assert result is False

    @pytest.mark.asyncio
    async def test_rebuild_no_backend(self):
        """Rebuild without backend returns False."""
        config = Config(sources=[], archive_root=Path("/tmp"), render_root=Path("/tmp/render"))
        service = IndexService(config=config, backend=None)

        result = await service.rebuild_index()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_index_status_no_backend(self):
        """Status without backend returns defaults."""
        config = Config(sources=[], archive_root=Path("/tmp"), render_root=Path("/tmp/render"))
        service = IndexService(config=config, backend=None)

        status = await service.get_index_status()

        assert status == {"exists": False, "count": 0}

    @pytest.mark.asyncio
    async def test_get_index_status_after_schema_init(self):
        """Status after schema init shows index exists with zero entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            config = Config(sources=[], archive_root=Path(tmpdir), render_root=Path(tmpdir) / "render")
            service = IndexService(config=config, backend=backend)

            # Schema init creates FTS table automatically
            await backend.list_conversations()

            status = await service.get_index_status()
            assert status["exists"] is True
            assert status["count"] == 0
            await backend.close()
