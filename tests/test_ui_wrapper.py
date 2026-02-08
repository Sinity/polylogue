from unittest.mock import MagicMock, patch

import pytest

from polylogue.ui import UI
from polylogue.ui.facade import ConsoleFacade


@pytest.fixture
def mock_facade():
    with patch("polylogue.ui.create_console_facade") as mock_create:
        facade = MagicMock(spec=ConsoleFacade)
        # MagicMock spec doesn't include fields that are initialized in __post_init__ or field(init=False)
        # unless we explicitly set them on the instance.
        console_mock = MagicMock()
        facade.console = console_mock
        facade.plain = False
        mock_create.return_value = facade
        yield facade


def test_ui_init_success(mock_facade):
    ui = UI(plain=False)
    assert ui._facade == mock_facade
    assert ui.plain is False
    assert ui.console == mock_facade.console


def test_ui_init_failure():
    with patch("polylogue.ui.create_console_facade", side_effect=RuntimeError("Test error")):
        with pytest.raises(SystemExit) as exc:
            UI(plain=False)
        assert "Test error" in str(exc.value)


def test_create_ui(mock_facade):
    from polylogue.ui import create_ui

    ui = create_ui(plain=True)
    assert isinstance(ui, UI)
    mock_facade.plain = True  # Setup specific for this test if needed


def test_ui_delegation(mock_facade):
    ui = UI(plain=False)

    ui.banner("Title", "Subtitle")
    mock_facade.banner.assert_called_with("Title", "Subtitle")

    ui.summary("Summary", ["line1"])
    mock_facade.summary.assert_called_with("Summary", ["line1"])

    ui.render_markdown("# Title")
    mock_facade.render_markdown.assert_called_with("# Title")

    ui.render_code("print('hi')")
    mock_facade.render_code.assert_called_with("print('hi')", "python")

    ui.render_diff("old", "new")
    mock_facade.render_diff.assert_called_with("old", "new", "file")


def test_ui_confirm_delegation(mock_facade):
    ui = UI(plain=False)
    mock_facade.confirm.return_value = True
    assert ui.confirm("Are you sure?") is True
    mock_facade.confirm.assert_called_with("Are you sure?", default=True)


def test_ui_confirm_plain(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)

    # Mock isatty to be True so input is used
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["y", "n", "", "foo"]):
        assert ui.confirm("Q1") is True
        assert ui.confirm("Q2") is False
        assert ui.confirm("Q3", default=True) is True  # Empty input -> default
        # "foo" -> not in {"y", "yes"} -> False (default logic logic: return response.lower() in {"y", "yes"})
        assert ui.confirm("Q4") is False


def test_ui_confirm_plain_eof(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.confirm("Q") is True  # Default


def test_ui_choose_empty(mock_facade):
    ui = UI(plain=False)
    assert ui.choose("Pick", []) is None


def test_ui_choose_plain(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    options = ["A", "B", "C"]

    with patch("sys.stdin.isatty", return_value=True):
        # 1. Valid choice "1" -> "A"
        # 2. Invalid choice "99" -> loop -> "2" -> "B"
        # 3. Non-digit -> loop -> "3" -> "C"
        # 4. Empty -> None
        with patch("builtins.input", side_effect=["1", "99", "2", "foo", "3", ""]):
            assert ui.choose("Pick", options) == "A"
            assert ui.choose("Pick", options) == "B"
            assert ui.choose("Pick", options) == "C"
            assert ui.choose("Pick", options) is None


def test_ui_choose_plain_eof(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    options = ["A"]
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.choose("Pick", options) is None


def test_ui_input_delegation(mock_facade):
    ui = UI(plain=False)
    mock_facade.input.return_value = "val"
    assert ui.input("Prompt") == "val"
    mock_facade.input.assert_called_with("Prompt", default=None)


def test_ui_input_plain(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["val", ""]):
        assert ui.input("P1") == "val"
        assert ui.input("P2", default="def") == "def"


def test_ui_input_plain_eof(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.input("P") is None
        assert ui.input("P", default="def") == "def"


def test_plain_progress_tracker():
    console_mock = MagicMock()
    from polylogue.ui import _PlainProgressTracker

    # Test with total (int)
    tracker = _PlainProgressTracker(console_mock, "Task", 10)
    tracker.advance(1)
    tracker.update(description="New Task")
    tracker.update(total=20)
    tracker.__exit__(None, None, None)

    assert console_mock.print.call_count >= 2

    # Test with None total
    tracker = _PlainProgressTracker(console_mock, "Task2", None)
    tracker.advance(1.5)  # float
    tracker.update(total=5.5)
    tracker.__exit__(None, None, None)


class TestPlainConsoleMarkupStripping:
    """Regression tests: PlainConsole must strip Rich markup for CI/plain output."""

    def test_strips_bold_markup(self, capsys):
        from polylogue.ui.facade import PlainConsole

        console = PlainConsole()
        console.print("[bold]Archive:[/bold] 1,234 conversations")
        captured = capsys.readouterr()
        assert "[bold]" not in captured.out
        assert "Archive: 1,234 conversations" in captured.out

    def test_strips_color_markup(self, capsys):
        from polylogue.ui.facade import PlainConsole

        console = PlainConsole()
        console.print("[green]✓[/green] All ok")
        captured = capsys.readouterr()
        assert "[green]" not in captured.out
        assert "✓ All ok" in captured.out

    def test_strips_hex_color_markup(self, capsys):
        from polylogue.ui.facade import PlainConsole

        console = PlainConsole()
        console.print("[#d97757]████████[/#d97757]")
        captured = capsys.readouterr()
        assert "#d97757" not in captured.out
        assert "████████" in captured.out

    def test_preserves_plain_text(self, capsys):
        from polylogue.ui.facade import PlainConsole

        console = PlainConsole()
        console.print("No markup at all")
        captured = capsys.readouterr()
        assert captured.out.strip() == "No markup at all"

    def test_handles_empty_string(self, capsys):
        from polylogue.ui.facade import PlainConsole

        console = PlainConsole()
        console.print("")
        captured = capsys.readouterr()
        assert captured.out.strip() == ""


def test_rich_progress_tracker():
    progress_mock = MagicMock()
    task_id = "task1"

    from polylogue.ui import _RichProgressTracker

    tracker = _RichProgressTracker(progress_mock, task_id)

    with tracker:
        tracker.advance(5)
        tracker.update(total=100, description="Processing")

    progress_mock.__enter__.assert_called()
    progress_mock.advance.assert_called_with(task_id, 5)
    progress_mock.update.assert_called()
    progress_mock.__exit__.assert_called()
