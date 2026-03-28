from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository
    from polylogue.ui.tui.screens.base import RepositoryBoundContainer


@dataclass(frozen=True)
class ScreenSpec:
    tab_id: str
    tab_label: str
    screen_type: type[RepositoryBoundContainer]


class PolylogueApp(App[None]):
    """Polylogue Mission Control TUI."""

    CSS_PATH = "css/styles.tcss"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    _SCREEN_SPECS: tuple[ScreenSpec, ...] = ()

    def __init__(
        self,
        repository: ConversationRepository | None = None,
    ) -> None:
        super().__init__()
        self._repository = repository

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        from textual.widgets import TabbedContent, TabPane

        if not self._SCREEN_SPECS:
            from polylogue.ui.tui.screens.browser import Browser
            from polylogue.ui.tui.screens.dashboard import Dashboard
            from polylogue.ui.tui.screens.search import Search

            type(self)._SCREEN_SPECS = (
                ScreenSpec("dashboard", "Mission Control", Dashboard),
                ScreenSpec("browser", "Browser", Browser),
                ScreenSpec("search", "Search", Search),
            )

        yield Header()

        with TabbedContent(initial=self._SCREEN_SPECS[0].tab_id):
            for spec in self._SCREEN_SPECS:
                with TabPane(spec.tab_label, id=spec.tab_id):
                    yield spec.screen_type(repository=self._repository)

        yield Footer()

    def action_toggle_dark(self) -> None:
        """Toggle between dark and light themes."""
        if "dark" in (self.theme or ""):
            self.theme = "textual-light"
        else:
            self.theme = "textual-dark"
