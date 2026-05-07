from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input
from textual.widgets import Markdown as MarkdownWidget

from polylogue.ui.tui.screens.base import RepositoryBoundContainer


class Search(RepositoryBoundContainer):
    """Search widget for finding conversations.

    Routes through ``ArchiveOperations.search()`` which applies the
    canonical query pipeline (FTS5, filters, result mapping) instead of
    calling the repository's ``search_summaries`` directly.
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Input(placeholder="Search conversations...", id="search-input")
            with Horizontal(id="search-split"):
                yield DataTable(id="search-results")
                with Container(id="search-preview"):
                    yield MarkdownWidget(id="search-viewer")

    def on_mount(self) -> None:
        table = self.query_one("#search-results", DataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Provider", "Title", "Date")

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle search submission."""
        query = message.value
        if not query:
            return

        ops = self._get_ops("Search")

        table = self.query_one("#search-results", DataTable)
        table.clear()

        try:
            result = await ops.search(query, limit=50)
        except Exception:
            table.add_row(
                "—",
                "—",
                "Search not ready: polylogued may need to build indexes",
                "",
            )
            return

        for hit in result.hits:
            table.add_row(
                hit.conversation_id,
                hit.provider_name,
                hit.title or "Untitled",
                str(hit.timestamp) if hit.timestamp else "",
                key=hit.conversation_id,
            )

    async def on_data_table_row_selected(self, message: DataTable.RowSelected) -> None:
        """Handle result selection."""
        if not message.row_key:
            return

        conv_id = str(message.row_key.value)
        await self.load_conversation(conv_id)

    async def load_conversation(self, conversation_id: str) -> None:
        """Load and display conversation content."""
        await self._load_conversation_markdown(conversation_id, viewer_selector="#search-viewer")
