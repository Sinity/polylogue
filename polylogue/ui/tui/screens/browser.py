from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Markdown as MarkdownWidget
from textual.widgets import Tree

from polylogue.config import Config
from polylogue.rendering.core import format_conversation_markdown

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository


class Browser(Container):
    """Browser widget for navigating conversations."""

    def __init__(
        self,
        config: Config | None = None,
        repository: ConversationRepository | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._repository = repository

    def _get_repo(self) -> ConversationRepository:
        """Get the repository, falling back to the service singleton."""
        if self._repository is not None:
            return self._repository
        from polylogue.services import get_repository
        return get_repository()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tree("Sources", id="browser-tree")
            with VerticalScroll(id="browser-viewer"):
                yield MarkdownWidget(id="markdown-viewer")

    def on_mount(self) -> None:
        self.run_worker(self._fetch_tree())

    async def _fetch_tree(self) -> None:
        """Fetch tree data asynchronously, then update DOM."""
        try:
            repo = self._get_repo()

            stats = await repo.get_archive_stats()
            providers = sorted(stats.providers.keys()) if stats.providers else []

            if not providers:
                providers = ["chatgpt", "claude"]  # Fallback for empty DB

            # Collect tree data: list of (provider_label, [(title, conv_id), ...])
            tree_data: list[tuple[str, list[tuple[str, str]]]] = []
            for provider in providers:
                summaries = await repo.list_summaries(limit=50, provider=provider)
                leaves = [(s.title or s.id, s.id) for s in summaries]
                tree_data.append((provider.capitalize(), leaves))

        except Exception as e:
            self.notify(f"Failed to load browser: {e}", severity="error")
            return

        self._apply_tree(tree_data)

    def _apply_tree(self, tree_data: list[tuple[str, list[tuple[str, str]]]]) -> None:
        """Apply fetched tree data to DOM (runs on main thread)."""
        tree = self.query_one("#browser-tree", Tree)
        tree.root.expand()

        for provider_label, leaves in tree_data:
            provider_node = tree.root.add(provider_label, expand=False)
            for label, conv_id in leaves:
                provider_node.add_leaf(label, data=conv_id)

    async def on_tree_node_selected(self, message: Tree.NodeSelected[str]) -> None:
        """Handle tree node selection."""
        if not message.node.allow_expand and message.node.data:
            conv_id = str(message.node.data)
            await self.load_conversation(conv_id)

    async def load_conversation(self, conversation_id: str) -> None:
        """Load and display conversation content."""
        repo = self._get_repo()

        conv = await repo.get_eager(conversation_id)
        if not conv:
            self.query_one("#markdown-viewer", MarkdownWidget).update(f"Error: Could not load {conversation_id}")
            return

        md_text = format_conversation_markdown(conv)
        self.query_one("#markdown-viewer", MarkdownWidget).update(md_text)
