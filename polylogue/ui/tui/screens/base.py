from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.containers import Container
from textual.widgets import Markdown as MarkdownWidget

from polylogue.rendering.core import format_conversation_markdown

if TYPE_CHECKING:
    from polylogue.archive.conversation.models import Conversation
    from polylogue.operations.archive import ArchiveOperations


ConversationMarkdownFormatter = Callable[["Conversation"], str]


class RepositoryBoundContainer(Container):
    """Shared base for TUI screens backed by injected archive operations."""

    def __init__(
        self,
        operations: ArchiveOperations | None = None,
    ) -> None:
        super().__init__()
        self._operations = operations

    def _get_ops(self, owner_name: str) -> ArchiveOperations:
        """Return the injected operations or fail with a screen-specific message."""
        if self._operations is None:
            raise RuntimeError(f"{owner_name} widget requires injected archive operations")
        return self._operations

    async def _load_conversation_markdown(
        self,
        conversation_id: str,
        *,
        viewer_selector: str,
        formatter: ConversationMarkdownFormatter = format_conversation_markdown,
    ) -> bool:
        """Load a conversation and update the target Markdown widget."""
        ops = self._get_ops(type(self).__name__)
        viewer = self.query_one(viewer_selector, MarkdownWidget)

        conv = await ops.get_conversation(conversation_id)
        if not conv:
            await viewer.update(f"Error: Could not load {conversation_id}")
            return False

        await viewer.update(formatter(conv))
        return True


__all__ = ["RepositoryBoundContainer"]
