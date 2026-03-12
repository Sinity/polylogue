from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class StatCard(Static):
    """A widget to display a single statistic."""

    DEFAULT_CSS = """
    StatCard {
        width: 1fr;
        height: 4;
        border: solid $accent;
        margin: 1;
        background: $surface;
        content-align: center middle;
    }
    """

    value: reactive[str] = reactive("", layout=False)

    def __init__(self, label: str, value: str | int, id: str | None = None) -> None:
        super().__init__(id=id)
        self.label = label
        self.value = str(value)

    def render(self) -> str:
        return f"[dim]{self.label}[/]\n[bold]{self.value or '…'}[/]"

    def watch_value(self, new_value: str) -> None:
        self.refresh()
