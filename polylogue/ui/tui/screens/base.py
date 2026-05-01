from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.containers import Container
from textual.widgets import Markdown as MarkdownWidget

from polylogue.protocols import ConversationArchiveReadStore
from polylogue.rendering.core import format_conversation_markdown

if TYPE_CHECKING:
    from polylogue.archive.conversation.models import Conversation


ConversationMarkdownFormatter = Callable[["Conversation"], str]


class RepositoryBoundContainer(Container):
    """Shared base for TUI screens backed by an injected repository."""

    def __init__(
        self,
        repository: ConversationArchiveReadStore | None = None,
    ) -> None:
        super().__init__()
        self._repository = repository

    def _get_repo(self, owner_name: str) -> ConversationArchiveReadStore:
        """Return the injected repository or fail with a screen-specific message."""
        if self._repository is None:
            raise RuntimeError(f"{owner_name} widget requires an injected repository")
        return self._repository

    async def _load_conversation_markdown(
        self,
        conversation_id: str,
        *,
        viewer_selector: str,
        formatter: ConversationMarkdownFormatter = format_conversation_markdown,
    ) -> bool:
        """Load a conversation and update the target Markdown widget."""
        repo = self._get_repo(type(self).__name__)
        viewer = self.query_one(viewer_selector, MarkdownWidget)

        conv = await repo.get_eager(conversation_id)
        if not conv:
            viewer.update(f"Error: Could not load {conversation_id}")
            return False

        viewer.update(formatter(conv))
        return True


__all__ = ["RepositoryBoundContainer"]
