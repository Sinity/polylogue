from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input
from textual.widgets import Markdown as MarkdownWidget

from polylogue.config import Config

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository


class Search(Container):
    """Search widget for finding conversations."""

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

        repo = self._get_repo()

        # Perform search
        table = self.query_one("#search-results", DataTable)
        table.clear()

        try:
            summaries = await repo.search_summaries(query, limit=50)
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
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
        """Load and display conversation content (duplicate logic from Browser)."""
        repo = self._get_repo()

        conv = await repo.get_eager(conversation_id)
        if not conv:
            self.query_one("#search-viewer", MarkdownWidget).update(f"Error: Could not load {conversation_id}")
            return

        md_lines = [f"# {conv.title or 'Untitled'}", f"*{conv.created_at}*", ""]

        for msg in conv.messages:
            role_icon = "👤" if msg.role == "user" else "🤖"
            md_lines.append(f"### {role_icon} {(msg.role or 'unknown').upper()}")
            md_lines.append(msg.text or "*[No content]*")
            md_lines.append("")

        self.query_one("#search-viewer", MarkdownWidget).update("\n".join(md_lines))
