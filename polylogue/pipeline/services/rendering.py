"""Async rendering service for pipeline operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polylogue.lib.log import get_logger
from polylogue.protocols import OutputRenderer

logger = get_logger(__name__)

__all__ = ["RenderService", "RenderResult"]


class RenderResult:
    """Result of a rendering operation."""

    def __init__(self) -> None:
        self.rendered_count: int = 0
        self.failures: list[dict[str, str]] = []

    def record_success(self) -> None:
        """Record a successful render."""
        self.rendered_count += 1

    def record_failure(self, conversation_id: str, error: str) -> None:
        """Record a rendering failure.

        Args:
            conversation_id: ID of the conversation that failed to render
            error: Error message
        """
        self.failures.append(
            {
                "conversation_id": conversation_id,
                "error": error,
            }
        )


class RenderService:
    """Service for rendering conversations to Markdown and HTML (async version)."""

    def __init__(
        self,
        renderer: OutputRenderer,
        render_root: Path,
    ):
        """Initialize the async rendering service.

        Args:
            renderer: OutputRenderer implementation for rendering conversations
            render_root: Root directory for rendered output
        """
        self.renderer = renderer
        self.render_root = render_root

    async def render_conversations(
        self,
        conversation_ids: list[str],
        *,
        max_workers: int = 4,
        progress_callback: Any | None = None,
    ) -> RenderResult:
        """Render multiple conversations concurrently.

        Args:
            conversation_ids: List of conversation IDs to render
            max_workers: Maximum number of concurrent tasks
            progress_callback: Optional callback(amount, desc=...) for progress updates

        Returns:
            RenderResult with success count and failures
        """
        import asyncio

        result = RenderResult()
        total = len(conversation_ids)
        worker_count = max(1, min(max_workers, total))
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max(worker_count * 2, 1))

        async def _render_one(convo_id: str) -> None:
            try:
                await self.renderer.render(convo_id, self.render_root)
                result.record_success()
            except Exception as exc:
                logger.warning("Failed to render conversation %s: %s", convo_id, exc)
                result.record_failure(convo_id, str(exc))
            if progress_callback is not None:
                done = result.rendered_count + len(result.failures)
                progress_callback(1, desc=f"Rendering: {done}/{total}")

        async def _worker() -> None:
            while True:
                convo_id = await queue.get()
                if convo_id is None:
                    queue.task_done()
                    return
                try:
                    await _render_one(convo_id)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]

        for convo_id in conversation_ids:
            await queue.put(convo_id)

        await queue.join()

        for _ in range(worker_count):
            await queue.put(None)

        await asyncio.gather(*workers, return_exceptions=False)

        return result
