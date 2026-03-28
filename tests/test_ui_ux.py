"""Comprehensive tests for Extant UI/UX (ConsoleFacade).

Tests interactive prompts via mock files and rich rendering output.
"""

import json
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from polylogue.ui.facade import PlainConsoleFacade, UIError, create_console_facade


@pytest.fixture
def mock_prompt_file(tmp_path, monkeypatch):
    """Setup a mock prompt response file."""
    prompt_file = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(prompt_file))
    return prompt_file


@pytest.fixture
def console_facade():
    """Create a real ConsoleFacade with captured output."""
    facade = create_console_facade(plain=False)
    # Replace console with a capture console
    facade.console = Console(force_terminal=True, no_color=False, width=80)
    # We'll use capture context in tests
    return facade


class TestConsoleFacadeInteractions:
    """Test interactive prompts using file-based mocking."""

    def test_confirm_true(self, mock_prompt_file):
        """Test confirm() returns True from mock."""
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "value": True}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?") is True

    def test_confirm_false(self, mock_prompt_file):
        """Test confirm() returns False from mock."""
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "value": False}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?") is False

    def test_confirm_default_override(self, mock_prompt_file):
        """Test confirm() uses default from mock instruction."""
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?", default=True) is True

        # Reset and test default=False
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")
        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?", default=False) is False

    def test_input_value(self, mock_prompt_file):
        """Test input() returns mocked string."""
        mock_prompt_file.write_text(json.dumps({"type": "input", "value": "test_input"}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.input("Name?") == "test_input"

    def test_choose_value(self, mock_prompt_file):
        """Test choose() selects by value."""
        mock_prompt_file.write_text(json.dumps({"type": "choose", "value": "Option B"}) + "\n")

        facade = create_console_facade(plain=False)
        result = facade.choose("Pick one", ["Option A", "Option B", "Option C"])
        assert result == "Option B"

    def test_choose_index(self, mock_prompt_file):
        """Test choose() selects by index."""
        mock_prompt_file.write_text(json.dumps({"type": "choose", "index": 2}) + "\n")

        facade = create_console_facade(plain=False)
        result = facade.choose("Pick one", ["Option A", "Option B", "Option C"])
        assert result == "Option C"

    def test_interaction_sequence(self, mock_prompt_file):
        """Test a sequence of different interactions."""
        lines = [
            json.dumps({"type": "input", "value": "User"}),
            json.dumps({"type": "confirm", "value": True}),
            json.dumps({"type": "choose", "value": "Red"}),
        ]
        mock_prompt_file.write_text("\n".join(lines) + "\n")

        facade = create_console_facade(plain=False)

        assert facade.input("Name?") == "User"
        assert facade.confirm("Save?") is True
        assert facade.choose("Color?", ["Red", "Blue"]) == "Red"

    def test_mismatched_prompt_type_raises_error(self, mock_prompt_file):
        """Test that mismatching prompt type raises UIError."""
        mock_prompt_file.write_text(json.dumps({"type": "input", "value": "User"}) + "\n")

        facade = create_console_facade(plain=False)

        with pytest.raises(UIError, match="expected 'input' but got 'confirm'"):
            facade.confirm("Save?")


class TestConsoleFacadeRendering:
    """Test rich rendering methods."""

    def test_banner_render(self):
        """Test banner rendering."""
        facade = create_console_facade(plain=False)
        with facade.console.capture() as capture:
            facade.banner("Welcome", "To Mission Control")

        output = capture.get()
        assert "Welcome" in output
        assert "To Mission Control" in output
        # Border characters check (approximate)
        assert "◈" in output or "mission" in output.lower()

    def test_summary_render(self):
        """Test list summary rendering."""
        facade = create_console_facade(plain=False)
        items = ["Item 1", "[red]Item 2[/red]"]
        with facade.console.capture() as capture:
            facade.summary("Checklist", items)

        output = capture.get()
        assert "Checklist" in output
        assert "Item 1" in output
        assert "Item 2" in output

    def test_render_diff(self):
        """Test diff rendering."""
        facade = create_console_facade(plain=False)

        # Mock pager to avoid actual paging which interferes with capture
        facade.console.pager = MagicMock()
        facade.console.pager.return_value.__enter__ = MagicMock()
        facade.console.pager.return_value.__exit__ = MagicMock()

        old = "line 1\nline 2"
        new = "line 1\nline 3"

        with facade.console.capture() as capture:
            facade.render_diff(old, new, filename="test.txt")

        output = capture.get()
        assert "line 2" in output
        assert "line 3" in output
        # Check for diff symbols? Color codes make exact matching hard,
        # but content should be there.

    def test_status_messages(self):
        """Test success/warning/error messages."""
        facade = create_console_facade(plain=False)
        with facade.console.capture() as capture:
            facade.success("Good job")
            facade.warning("Be careful")
            facade.error("Oh no")
            facade.info("FYI")

        output = capture.get()
        assert "Good job" in output
        assert "Be careful" in output
        assert "Oh no" in output
        assert "FYI" in output


class TestPlainFacade:
    """Test plain mode fallback."""

    def test_plain_interactions_defaults(self):
        """Plain mode should return defaults without prompting."""
        facade = create_console_facade(plain=True)
        assert isinstance(facade, PlainConsoleFacade)

        assert facade.confirm("Ctx?", default=True) is True
        assert facade.confirm("Ctx?", default=False) is False
        assert facade.input("Input?", default="Default") == "Default"
        assert facade.input("Input?", default=None) is None
        assert facade.choose("Pick", ["A", "B"]) is None  # Defaults to None in code? Or None if no interaction?
        # Checked code: choose returns None in plain mode.

    def test_plain_rendering(self, capsys):
        """Plain mode should print simple text."""
        facade = create_console_facade(plain=True)

        facade.banner("Title", "Subtitle")
        captured = capsys.readouterr()
        assert "== Title ==" in captured.out
        assert "Subtitle" in captured.out

        facade.success("Done")
        captured = capsys.readouterr()
        assert "✓ Done" in captured.out
