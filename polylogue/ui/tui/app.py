from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

if TYPE_CHECKING:
    from polylogue.operations.archive import ArchiveOperations


class PolylogueApp(App[None]):
    """Polylogue dashboard TUI.

    Accepts an optional ``ArchiveOperations`` instance so screens route
    through the shared archive operations contract rather than calling
    repository methods directly (ref #860).
    """

    CSS_PATH = "css/styles.tcss"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        operations: ArchiveOperations | None = None,
    ) -> None:
        super().__init__()
        self._operations = operations

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        from textual.widgets import TabbedContent, TabPane

        from polylogue.ui.tui.screens.browser import Browser
        from polylogue.ui.tui.screens.dashboard import Dashboard
        from polylogue.ui.tui.screens.search import Search

        yield Header()

        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                yield Dashboard(operations=self._operations)

            with TabPane("Browser", id="browser"):
                yield Browser(operations=self._operations)

            with TabPane("Search", id="search"):
                yield Search(operations=self._operations)

        yield Footer()

    def action_toggle_dark(self) -> None:
        """Toggle between dark and light themes."""
        if "dark" in (self.theme or ""):
            self.theme = "textual-light"
        else:
            self.theme = "textual-dark"
