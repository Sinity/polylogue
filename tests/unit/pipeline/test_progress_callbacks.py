"""Tests for progress callback propagation through pipeline stages.

These tests verify that progress_callback is actually invoked during
render and index stages — a gap that allowed silent pipeline execution
to ship (production failure: stages completing with zero user feedback).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.pipeline.services.rendering import RenderResult, RenderService


class TestRenderProgressCallback:
    """Verify progress_callback fires during rendering."""

    async def test_callback_called_for_each_conversation(self):
        """progress_callback is invoked once per rendered conversation."""
        from pathlib import Path

        # Mock renderer that succeeds
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        callback = MagicMock()
        result = await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=callback,
        )

        assert result.rendered_count == 3
        assert callback.call_count == 3

    async def test_callback_desc_format(self):
        """progress_callback desc follows 'Rendering: N/total' format."""
        from pathlib import Path

        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        descs = []

        def capture_callback(amount, desc=None):
            descs.append(desc)

        await service.render_conversations(
            ["conv-1", "conv-2"],
            progress_callback=capture_callback,
        )

        # All desc values should match the "Rendering: N/2" pattern
        assert all(d is not None and d.startswith("Rendering:") for d in descs)
        # The last one should show total completed
        assert "2/2" in descs[-1]

    async def test_callback_fires_on_failure_too(self):
        """progress_callback fires even when rendering fails."""
        from pathlib import Path

        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock(
            side_effect=[None, RuntimeError("render failed"), None]
        )

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        callback = MagicMock()
        result = await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=callback,
        )

        # All 3 should fire callback (2 success + 1 failure)
        assert callback.call_count == 3
        assert result.rendered_count == 2
        assert len(result.failures) == 1

    async def test_no_callback_is_safe(self):
        """Rendering works without a progress_callback."""
        from pathlib import Path

        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations(["conv-1"])
        assert result.rendered_count == 1

    async def test_callback_amount_is_one(self):
        """Each callback invocation passes amount=1."""
        from pathlib import Path

        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        amounts = []

        def capture(amount, desc=None):
            amounts.append(amount)

        await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=capture,
        )
        assert all(a == 1 for a in amounts)


class TestIndexProgressCallback:
    """Verify progress_callback firing during indexing is safe."""

    async def test_index_update_without_callback_succeeds(self, sqlite_backend):
        """Index update works without a progress_callback."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.update_index([])
        assert result is True

    async def test_index_rebuild_without_callback_succeeds(self, sqlite_backend):
        """Index rebuild works without a progress_callback."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.rebuild_index()
        assert result is True


class TestRunnerProgressPropagation:
    """Verify run_sources propagates progress_callback to render/index stages."""

    def test_render_stage_accepts_callback(self, workspace_env):
        """When stage='render', progress_callback is accepted without error."""
        import asyncio

        from polylogue.config import Config
        from polylogue.pipeline.runner import run_sources

        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)
        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=archive_root / "render",
        )

        callback_calls = []

        def track_callback(amount, desc=None):
            callback_calls.append({"amount": amount, "desc": desc})

        with patch(
            "polylogue.pipeline.runner._all_conversation_ids",
            new_callable=AsyncMock,
        ) as mock_ids:
            mock_ids.return_value = []

            result = asyncio.run(
                run_sources(
                    config=config,
                    stage="render",
                    progress_callback=track_callback,
                )
            )

            # Should complete without error when callback is provided
            assert result is not None

    def test_callback_none_when_not_provided(self, workspace_env):
        """When progress_callback is None, render still succeeds."""
        import asyncio

        from polylogue.config import Config
        from polylogue.pipeline.runner import run_sources

        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)
        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=archive_root / "render",
        )

        with patch(
            "polylogue.pipeline.runner._all_conversation_ids",
            new_callable=AsyncMock,
        ) as mock_ids:
            mock_ids.return_value = []

            # Call without progress_callback (should default to None)
            result = asyncio.run(
                run_sources(
                    config=config,
                    stage="render",
                )
            )

            # Should succeed without error
            assert result is not None

    def test_index_stage_accepts_callback(self, workspace_env):
        """When stage='index', progress_callback is accepted without error."""
        import asyncio

        from polylogue.config import Config
        from polylogue.pipeline.runner import run_sources

        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)
        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=archive_root / "render",
        )

        callback_calls = []

        def track_callback(amount, desc=None):
            callback_calls.append({"amount": amount, "desc": desc})

        result = asyncio.run(
            run_sources(
                config=config,
                stage="index",
                progress_callback=track_callback,
            )
        )

        # Should complete without error when callback is provided
        assert result is not None
        # Index stage should have called callback at least once (initial "Indexing" call)
        assert len(callback_calls) > 0
