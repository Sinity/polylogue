from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Markdown as MarkdownWidget
from textual.widgets import Tree

from polylogue.config import Config
from polylogue.rendering.core import format_conversation_markdown
from polylogue.services import get_repository


class Browser(Container):
    """Browser widget for navigating conversations."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tree("Sources", id="browser-tree")
            with VerticalScroll(id="browser-viewer"):
                yield MarkdownWidget(id="markdown-viewer")

    def on_mount(self) -> None:
        self.run_worker(self.load_tree(), thread=True)

    async def load_tree(self) -> None:
        """Load sources and conversations into the tree."""
        tree = self.query_one("#browser-tree", Tree)
        tree.root.expand()

        try:
            repo = get_repository()

            # Query distinct providers from the database
            from polylogue.storage.backends.sqlite import connection_context

            with connection_context(None) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT provider_name FROM conversations ORDER BY provider_name"
                ).fetchall()
            providers = [row["provider_name"] for row in rows if row["provider_name"]]

            if not providers:
                providers = ["chatgpt", "claude"]  # Fallback for empty DB

            for provider in providers:
                provider_node = tree.root.add(provider.capitalize(), expand=False)

                # Fetch summaries for this provider
                summaries = repo.list_summaries(limit=50, provider=provider)
                for summary in summaries:
                    label = summary.title or summary.id
                    # Store ID in data for retrieval
                    provider_node.add_leaf(label, data=summary.id)

        except Exception as e:
            self.notify(f"Failed to load browser: {e}", severity="error")

    async def on_tree_node_selected(self, message: Tree.NodeSelected[str]) -> None:
        """Handle tree node selection."""
        if not message.node.allow_expand and message.node.data:
            conv_id = str(message.node.data)
            self.load_conversation(conv_id)

    def load_conversation(self, conversation_id: str) -> None:
        """Load and display conversation content."""
        repo = get_repository()

        conv = repo.get_eager(conversation_id)
        if not conv:
            self.query_one("#markdown-viewer", MarkdownWidget).update(f"Error: Could not load {conversation_id}")
            return

        md_text = format_conversation_markdown(conv)
        self.query_one("#markdown-viewer", MarkdownWidget).update(md_text)
