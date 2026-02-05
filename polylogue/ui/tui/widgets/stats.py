from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static


class StatCard(Static):
    """A widget to display a single statistic."""

    DEFAULT_CSS = """
    StatCard {
        width: 1fr;
        height: auto;
        border: solid $accent;
        padding: 1;
        margin: 1;
        background: $surface;
    }
    
    StatCard > .label {
        color: $text-muted;
        text-align: center;
    }

    StatCard > .value {
        color: $text;
        text-style: bold;
        text-align: center;
        height: auto;
        padding-top: 1;
        padding-bottom: 1;
    }
    """

    def __init__(self, label: str, value: str | int, id: str | None = None) -> None:
        super().__init__(id=id)
        self.label = label
        self.value = str(value)

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="label")
        yield Static(self.value, classes="value")
