from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Markdown as MarkdownWidget
from textual.widgets import Tree

from polylogue.ui.tui.screens.base import RepositoryBoundContainer


class Browser(RepositoryBoundContainer):
    """Browser widget for navigating conversations."""

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
            repo = self._get_repo("Browser")

            stats = await repo.get_archive_stats()
            providers = sorted(stats.providers.keys()) if stats.providers else []

            # Collect tree data: list of (provider_label, [(title, conv_id), ...])
            tree_data: list[tuple[str, list[tuple[str, str]]]] = []
            for provider in providers:
                summaries = await repo.list_summaries(limit=50, provider=provider)
                leaves = [(str(s.title or s.id), str(s.id)) for s in summaries]
                tree_data.append((provider.capitalize(), leaves))

        except Exception as e:
            self.notify(f"Failed to load browser: {e}", severity="error")
            return

        self._apply_tree(tree_data)

    def _apply_tree(self, tree_data: list[tuple[str, list[tuple[str, str]]]]) -> None:
        """Apply fetched tree data to DOM (runs on main thread)."""
        tree = self.query_one("#browser-tree", Tree)
        tree.root.expand()

        if not tree_data:
            tree.root.add_leaf("No conversations in archive")
            return

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
        await self._load_conversation_markdown(conversation_id, viewer_selector="#markdown-viewer")
