from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Markdown as MarkdownWidget
from textual.widgets import Tree

from polylogue.config import Config
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
        self.run_worker(self.load_tree())

    async def load_tree(self) -> None:
        """Load sources and conversations into the tree."""
        tree = self.query_one("#browser-tree", Tree)
        tree.root.expand()

        repo = get_repository()

        # Group by Source (e.g. "chatgpt", "claude")
        # In Polylogue v2, sources are defined in config.
        # We can iterate configured sources and fetch conversations for each.

        # Note: We might want to list *all* providers found in DB too,
        # but starting with config sources is safer.

        # Identify distinct providers from DB would be better but let's stick to
        # listing "Recent" or grouping by Provider for now.

        # Let's try listing by Provider since that's a primary concept.
        providers = ["chatgpt", "claude", "gemini", "codex", "test", "other"]

        for provider in providers:
            provider_node = tree.root.add(provider.capitalize(), expand=False)

            # Fetch summaries for this provider
            summaries = repo.list_summaries(limit=50, provider=provider)
            for summary in summaries:
                label = summary.title or summary.id
                # Store ID in data for retrieval
                provider_node.add_leaf(label, data=summary.id)

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

        # Render to Markdown
        # We can use our existing render logic or just simple dump for now.
        # A simple formatting:

        md_lines = [f"# {conv.title or 'Untitled'}", f"*{conv.created_at}*", ""]

        for msg in conv.messages:
            role_icon = "👤" if msg.role == "user" else "🤖"
            md_lines.append(f"### {role_icon} {msg.role.upper()}")
            md_lines.append(msg.text or "*[No content]*")
            md_lines.append("")

            if msg.attachments:
                md_lines.append(f"**Attachments:** {len(msg.attachments)}")
                md_lines.append("")

        self.query_one("#markdown-viewer", MarkdownWidget).update("\n".join(md_lines))
