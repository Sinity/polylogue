from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.containers import Container
from textual.widgets import Markdown as MarkdownWidget

from polylogue.rendering.core import format_session_markdown

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.archive.session.domain_models import Session


SessionMarkdownFormatter = Callable[["Session"], str]


class RepositoryBoundContainer(Container):
    """Shared base for TUI screens backed by an injected archive facade."""

    def __init__(
        self,
        polylogue: Polylogue | None = None,
    ) -> None:
        super().__init__()
        self._polylogue = polylogue

    def _get_facade(self, owner_name: str) -> Polylogue:
        """Return the injected archive facade or fail with a screen-specific message."""
        if self._polylogue is None:
            raise RuntimeError(f"{owner_name} widget requires an injected Polylogue facade")
        return self._polylogue

    async def _load_session_markdown(
        self,
        session_id: str,
        *,
        viewer_selector: str,
        formatter: SessionMarkdownFormatter = format_session_markdown,
    ) -> bool:
        """Load a session and update the target Markdown widget."""
        facade = self._get_facade(type(self).__name__)
        viewer = self.query_one(viewer_selector, MarkdownWidget)

        conv = await facade.get_session(session_id)
        if not conv:
            await viewer.update(f"Error: Could not load {session_id}")
            return False

        await viewer.update(formatter(conv))
        return True


__all__ = ["RepositoryBoundContainer"]
