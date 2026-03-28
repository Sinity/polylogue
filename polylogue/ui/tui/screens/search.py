from __future__ import annotations

import sqlite3

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input
from textual.widgets import Markdown as MarkdownWidget

from polylogue.errors import DatabaseError
from polylogue.ui.tui.screens.base import RepositoryBoundContainer


class Search(RepositoryBoundContainer):
    """Search widget for finding conversations."""

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

        repo = self._get_repo("Search")

        # Perform search
        table = self.query_one("#search-results", DataTable)
        table.clear()

        try:
            summaries = await repo.search_summaries(query, limit=50)
        except (sqlite3.OperationalError, DatabaseError) as exc:
            if "no such table" in str(exc) or "Search index not built" in str(exc):
                table.add_row("—", "—", "Search index not built. Run: polylogue run", "")
            else:
                table.add_row("—", "—", f"Search error: {exc}", "")
            return

        for s in summaries:
            table.add_row(
                s.id,
                s.provider,
                s.title or "Untitled",
                str(s.created_at) if s.created_at else "",
                key=s.id,  # Store ID as row key for selection
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
