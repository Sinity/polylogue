"""Rendering service for pipeline operations."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

from polylogue.core.log import get_logger
from polylogue.protocols import OutputRenderer

logger = get_logger(__name__)


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
    """Service for rendering conversations to Markdown and HTML."""

    def __init__(
        self,
        renderer: OutputRenderer,
        render_root: Path,
    ):
        """Initialize the rendering service.

        Args:
            renderer: OutputRenderer implementation for rendering conversations
            render_root: Root directory for rendered output
        """
        self.renderer = renderer
        self.render_root = render_root

    def render_conversations(
        self,
        conversation_ids: list[str],
        *,
        max_workers: int = 4,
    ) -> RenderResult:
        """Render multiple conversations in parallel.

        Args:
            conversation_ids: List of conversation IDs to render
            max_workers: Maximum number of parallel workers

        Returns:
            RenderResult with success count and failures
        """
        result = RenderResult()

        def _render_one(convo_id: str) -> int:
            self.renderer.render(convo_id, self.render_root)
            return 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            render_futures = {executor.submit(_render_one, cid): cid for cid in conversation_ids}
            for fut in concurrent.futures.as_completed(render_futures):
                convo_id = render_futures[fut]
                try:
                    fut.result()
                    result.record_success()
                except Exception as exc:
                    logger.warning("Failed to render conversation %s: %s", convo_id, exc)
                    result.record_failure(convo_id, str(exc))

        return result
