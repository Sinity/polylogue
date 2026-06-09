from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input
from textual.widgets import Markdown as MarkdownWidget

from polylogue.api.contracts.tui_surface import TUIReadSurface
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.ui.tui.screens.base import RepositoryBoundContainer


class Search(RepositoryBoundContainer):
    """Search widget for finding sessions.

    Routes through :class:`TUIReadSurface.search_sessions` so the
    TUI consumes the same :class:`SessionListResponse` envelope as
    the web reader, CLI JSON, MCP, and Python API surfaces.  Result
    rows are rendered from the typed :class:`SessionListRowPayload`
    fields (``id``, ``origin``, ``title``, ``date``).
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Input(placeholder="Search sessions...", id="search-input")
            with Horizontal(id="search-split"):
                yield DataTable(id="search-results")
                with Container(id="search-preview"):
                    yield MarkdownWidget(id="search-viewer")

    def on_mount(self) -> None:
        table = self.query_one("#search-results", DataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Origin", "Title", "Date")

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle search submission."""
        query = message.value
        if not query:
            return

        facade = self._get_facade("Search")
        surface = TUIReadSurface(facade)

        table = self.query_one("#search-results", DataTable)
        table.clear()

        spec = SessionQuerySpec(query_terms=(query,), limit=50)
        try:
            envelope = await surface.search_sessions(spec)
        except Exception:
            table.add_row(
                "—",
                "—",
                "Search not ready: polylogued may need to build indexes",
                "",
            )
            return

        for row in envelope.items:
            table.add_row(
                row.id,
                row.origin,
                row.title or "Untitled",
                row.date or "",
                key=row.id,
            )

    async def on_data_table_row_selected(self, message: DataTable.RowSelected) -> None:
        """Handle result selection."""
        if message.row_key.value is None:
            return

        conv_id = str(message.row_key.value)
        await self.load_session(conv_id)

    async def load_session(self, session_id: str) -> None:
        """Load and display session content."""
        await self._load_session_markdown(session_id, viewer_selector="#search-viewer")
