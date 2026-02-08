from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from polylogue.config import Config


class PolylogueApp(App[None]):
    """Polylogue Mission Control TUI."""

    CSS_PATH = "css/styles.tcss"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        from textual.widgets import TabbedContent, TabPane

        from polylogue.ui.tui.screens.dashboard import Dashboard

        yield Header()

        with TabbedContent(initial="dashboard"):
            with TabPane("Mission Control", id="dashboard"):
                yield Dashboard(config=self.config)

            with TabPane("Browser", id="browser"):
                from polylogue.ui.tui.screens.browser import Browser

                yield Browser(config=self.config)

            with TabPane("Search", id="search"):
                from polylogue.ui.tui.screens.search import Search

                yield Search(config=self.config)

        yield Footer()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark  # type: ignore[has-type]
