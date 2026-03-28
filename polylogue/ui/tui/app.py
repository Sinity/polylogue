from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository


class PolylogueApp(App[None]):
    """Polylogue Mission Control TUI."""

    CSS_PATH = "css/styles.tcss"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        repository: ConversationRepository | None = None,
    ) -> None:
        super().__init__()
        self._repository = repository

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        from textual.widgets import TabbedContent, TabPane

        from polylogue.ui.tui.screens.browser import Browser
        from polylogue.ui.tui.screens.dashboard import Dashboard
        from polylogue.ui.tui.screens.search import Search

        yield Header()

        with TabbedContent(initial="dashboard"):
            with TabPane("Mission Control", id="dashboard"):
                yield Dashboard(repository=self._repository)

            with TabPane("Browser", id="browser"):
                yield Browser(repository=self._repository)

            with TabPane("Search", id="search"):
                yield Search(repository=self._repository)

        yield Footer()

    def action_toggle_dark(self) -> None:
        """Toggle between dark and light themes."""
        if "dark" in (self.theme or ""):
            self.theme = "textual-light"
        else:
            self.theme = "textual-dark"
