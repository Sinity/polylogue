"""Rendering service for pipeline operations."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

from polylogue.core.log import get_logger
from polylogue.protocols import OutputRenderer
from polylogue.render import render_conversation

logger = get_logger(__name__)


class RenderResult:
    """Result of a rendering operation."""

    def __init__(self):
        self.rendered_count = 0
        self.failures: list[dict[str, str]] = []

    def record_success(self):
        """Record a successful render."""
        self.rendered_count += 1

    def record_failure(self, conversation_id: str, error: str):
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
        template_path: Path | None,
        render_root: Path,
        archive_root: Path,
        renderer: OutputRenderer | None = None,
    ):
        """Initialize the rendering service.

        Args:
            template_path: Optional path to custom HTML template
            render_root: Root directory for rendered output
            archive_root: Root directory for archived conversations
            renderer: Optional OutputRenderer implementation (uses legacy render if None)
        """
        self.template_path = template_path
        self.render_root = render_root
        self.archive_root = archive_root
        self.renderer = renderer

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

        def _render_one(convo_id):
            if self.renderer:
                # Use new renderer abstraction
                self.renderer.render(convo_id, self.render_root)
            else:
                # Use legacy render function
                render_conversation(
                    conversation_id=convo_id,
                    archive_root=self.archive_root,
                    render_root_path=self.render_root,
                    template_path=self.template_path,
                )
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
