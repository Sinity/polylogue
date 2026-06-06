from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Markdown as MarkdownWidget
from textual.widgets import Tree

from polylogue.api.contracts.tui_surface import TUIReadSurface
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.ui.tui.screens.base import RepositoryBoundContainer


class Browser(RepositoryBoundContainer):
    """Browser widget for navigating sessions.

    Consumes the typed :class:`SessionListResponse` envelope (and
    :class:`SessionListRowPayload` rows) through
    :class:`TUIReadSurface` so the TUI shares the web reader's payload
    contract instead of rendering from raw repository rows.
    """

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
            facade = self._get_facade("Browser")
            surface = TUIReadSurface(facade)

            stats = await facade.storage_stats()
            origin_tokens = sorted(stats.origins.keys()) if stats.origins else []

            # Collect tree data using the typed payload envelope; each row
            # carries id/title/origin as canonical fields.
            tree_data: list[tuple[str, list[tuple[str, str]]]] = []
            for origin in origin_tokens:
                spec = SessionQuerySpec(
                    origins=(origin,),
                    limit=50,
                )
                envelope = await surface.list_sessions(spec)
                leaves = [(row.title, row.id) for row in envelope.items]
                tree_data.append((origin.capitalize(), leaves))

        except Exception as e:
            self.notify(f"Failed to load browser: {e}", severity="error")
            return

        self._apply_tree(tree_data)

    def _apply_tree(self, tree_data: list[tuple[str, list[tuple[str, str]]]]) -> None:
        """Apply fetched tree data to DOM (runs on main thread)."""
        tree = self.query_one("#browser-tree", Tree)
        tree.root.expand()

        if not tree_data:
            tree.root.add_leaf("No sessions in archive")
            return

        for origin_label, leaves in tree_data:
            origin_node = tree.root.add(origin_label, expand=False)
            for label, conv_id in leaves:
                origin_node.add_leaf(label, data=conv_id)

    async def on_tree_node_selected(self, message: Tree.NodeSelected[str]) -> None:
        """Handle tree node selection."""
        if not message.node.allow_expand and message.node.data:
            conv_id = str(message.node.data)
            await self.load_session(conv_id)

    async def load_session(self, session_id: str) -> None:
        """Load and display session content."""
        await self._load_session_markdown(session_id, viewer_selector="#markdown-viewer")
