"""Tests for progress callback propagation through pipeline stages.

These tests verify that progress_callback is actually invoked during
render and index stages — a gap that allowed silent pipeline execution
to ship (production failure: stages completing with zero user feedback).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.pipeline.services.rendering import RenderService


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


class TestRunnerProgressPropagation:
    """Verify run_sources propagates progress_callback to render/index stages."""

    @pytest.mark.parametrize(
        "stage,with_callback,expected_indexed,expected_rendered,expected_first_desc",
        [
            ("render", True, False, 0, None),
            ("render", False, False, 0, None),
            ("index", True, True, 0, "Indexing"),
        ],
    )
    def test_stage_callback_matrix(
        self,
        workspace_env,
        stage,
        with_callback,
        expected_indexed,
        expected_rendered,
        expected_first_desc,
    ):
        """run_sources accepts optional callback and emits stage-appropriate progress."""
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

        callback = track_callback if with_callback else None

        if stage == "render":
            with patch(
                "polylogue.pipeline.runner._all_conversation_ids",
                new_callable=AsyncMock,
            ) as mock_ids:
                mock_ids.return_value = []
                result = asyncio.run(
                    run_sources(
                        config=config,
                        stage=stage,
                        progress_callback=callback,
                    )
                )
        else:
            result = asyncio.run(
                run_sources(
                    config=config,
                    stage=stage,
                    progress_callback=callback,
                )
            )

        assert result.indexed is expected_indexed
        assert result.counts.get("rendered", 0) == expected_rendered
        if expected_first_desc is None:
            assert callback_calls == []
        else:
            assert callback_calls
            assert callback_calls[0]["desc"] == expected_first_desc
