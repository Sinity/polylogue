from __future__ import annotations

from unittest.mock import patch

from rich.errors import MarkupError
from rich.table import Table
from rich.text import Text

from polylogue.ui.facade_console import PlainConsole, _render_plain_object


def test_render_plain_object_handles_text_markup_tables_and_fallbacks() -> None:
    table = Table()
    table.add_column("Name")
    table.add_row("Value")

    assert _render_plain_object(Text("hello")) == "hello"
    assert _render_plain_object("[bold]hello[/bold]") == "hello"
    with patch("polylogue.ui.facade_console.Text.from_markup", side_effect=MarkupError("boom")):
        assert _render_plain_object("[broken") == "[broken"
    assert "Name" in _render_plain_object(table)
    assert _render_plain_object(123) == "123"


def test_plain_console_print_renders_space_joined_plain_values() -> None:
    console = PlainConsole()

    with patch("builtins.print") as mock_print:
        console.print(Text("hello"), "[bold]world[/bold]", 123)

    mock_print.assert_called_once_with("hello world 123")
