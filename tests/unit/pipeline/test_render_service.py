"""Tests for render-service pipeline behavior."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.pipeline.services.rendering import (
    RenderResult,
    RenderService,
    _resolve_default_render_workers,
)


class TestRenderResult:
    """Tests for RenderResult data class."""

    def test_initial_state(self) -> None:
        """Fresh result has zero count and empty failures."""
        result = RenderResult()
        assert result.rendered_count == 0
        assert result.failures == []
        assert result.worker_count == 0
        assert result.max_current_rss_mb is None

    def test_record_success(self) -> None:
        """record_success increments count."""
        result = RenderResult()
        result.record_success()
        result.record_success()
        assert result.rendered_count == 2

    def test_record_failure(self) -> None:
        """record_failure appends failure info."""
        result = RenderResult()
        result.record_failure("conv-1", "boom")
        assert len(result.failures) == 1
        assert result.failures[0] == {"conversation_id": "conv-1", "error": "boom"}


class TestRenderService:
    """Tests for RenderService."""

    @pytest.mark.asyncio
    async def test_render_empty_ids(self) -> None:
        """No IDs returns zero rendered."""
        renderer = MagicMock()
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations([])

        assert result.rendered_count == 0
        assert result.failures == []
        renderer.render.assert_not_called()

    @pytest.mark.asyncio
    async def test_render_success(self) -> None:
        """Renders conversations, increments count."""
        renderer = AsyncMock()
        renderer.render.return_value = None  # Success
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations(["conv-1", "conv-2", "conv-3"])

        assert result.rendered_count == 3
        assert result.failures == []
        assert renderer.render.call_count == 3

    @pytest.mark.asyncio
    async def test_render_async_iterable_source(self) -> None:
        """Streaming conversation IDs are accepted without prior list materialization."""
        renderer = AsyncMock()
        renderer.render.return_value = None
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))

        async def conversation_ids() -> AsyncGenerator[str, None]:
            for convo_id in ("conv-1", "conv-2"):
                yield convo_id

        result = await service.render_conversations(conversation_ids(), total=2)

        assert result.rendered_count == 2
        assert result.failures == []
        assert renderer.render.call_count == 2

    @pytest.mark.asyncio
    async def test_render_failure_tracked(self) -> None:
        """Render errors collected in failures list."""
        renderer = AsyncMock()

        async def render_side_effect(conv_id: str, root: Path) -> None:
            del root
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
    async def test_render_concurrency_limit(self) -> None:
        """Semaphore limits concurrent renders."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        renderer = AsyncMock()

        async def tracked_render(conv_id: str, root: Path) -> None:
            nonlocal max_concurrent, current_concurrent
            del conv_id, root
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1

        renderer.render.side_effect = tracked_render
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))
        await service.render_conversations(
            [f"conv-{i}" for i in range(10)],
            max_workers=2,
        )

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_progress_callback_called_for_each_conversation(self) -> None:
        """progress_callback is invoked once per rendered conversation."""
        renderer = AsyncMock()
        renderer.render = AsyncMock()
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))
        callback = MagicMock()

        result = await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=callback,
        )

        assert result.rendered_count == 3
        assert callback.call_count == 3

    @pytest.mark.asyncio
    async def test_progress_callback_desc_format(self) -> None:
        """progress_callback descriptions should track rendered totals."""
        renderer = AsyncMock()
        renderer.render = AsyncMock()
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))
        descriptions: list[str | None] = []

        def capture(amount: int, desc: str | None = None) -> None:
            del amount
            descriptions.append(desc)

        await service.render_conversations(
            ["conv-1", "conv-2"],
            progress_callback=capture,
        )

        assert all(description is not None and description.startswith("Rendering:") for description in descriptions)
        last_description = descriptions[-1]
        assert last_description is not None
        assert "2/2" in last_description

    @pytest.mark.asyncio
    async def test_progress_callback_fires_on_failure_too(self) -> None:
        """Failures should still advance render progress callbacks."""
        renderer = AsyncMock()
        renderer.render = AsyncMock(side_effect=[None, RuntimeError("render failed"), None])
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))
        callback = MagicMock()

        result = await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=callback,
        )

        assert callback.call_count == 3
        assert result.rendered_count == 2
        assert len(result.failures) == 1

    @pytest.mark.asyncio
    async def test_render_without_progress_callback_is_safe(self) -> None:
        """RenderService should not require a callback."""
        renderer = AsyncMock()
        renderer.render = AsyncMock()
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations(["conv-1"])
        assert result.rendered_count == 1

    @pytest.mark.asyncio
    async def test_progress_callback_amount_is_one(self) -> None:
        """Each render callback increments by one completed conversation."""
        renderer = AsyncMock()
        renderer.render = AsyncMock()
        service = RenderService(renderer=renderer, render_root=Path("/tmp/render"))
        amounts: list[int] = []

        def capture(amount: int, desc: str | None = None) -> None:
            del desc
            amounts.append(amount)

        await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=capture,
        )
        assert all(amount == 1 for amount in amounts)


def test_default_render_workers_back_off_when_rss_is_high(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.rendering.os.cpu_count", lambda: 16)
    monkeypatch.setattr("polylogue.pipeline.services.rendering.read_current_rss_mb", lambda: 1536.0)

    worker_count = _resolve_default_render_workers(None)

    assert worker_count == 2


def test_default_render_workers_respect_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.rendering.read_current_rss_mb", lambda: 4096.0)

    worker_count = _resolve_default_render_workers(3)

    assert worker_count == 3
