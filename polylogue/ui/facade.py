"""Terminal UI facade.

Polylogue ships with a small UI layer. Interactive mode relies entirely on the
bundled Python stack (questionary + Rich) so it works anywhere without external
CLI binaries, while plain mode falls back to basic stdout prompting.
"""

from __future__ import annotations

import difflib
import json
import os
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import questionary
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name
from rich import box
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme


class UIError(Exception):
    """UI-related errors (prompt stubs, user interaction)."""


class ConsoleLike(Protocol):
    def print(self, *objects: object, **kwargs: object) -> None: ...


class PlainConsole:
    """Minimal Console shim for non-interactive mode."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def print(self, *objects: object, **_: object) -> None:
        text = " ".join(str(obj) for obj in objects)
        print(text)


@dataclass
class ConsoleFacade:
    """Pure Python facade for terminal UI - no external binaries required."""

    plain: bool
    console: ConsoleLike = field(init=False)

    theme: Theme = field(init=False, repr=False)
    _prompt_responses: deque[dict[str, object]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.theme = Theme(
            {
                "banner.icon": "bold #7fdbca",
                "banner.title": "bold #e0f2f1",
                "banner.subtitle": "#cdecef",
                "banner.border": "#14b8a6",
                "panel.border": "#3b82f6",
                "panel.text": "#e5e7eb",
                "summary.title": "bold #c4e0ff",
                "summary.bullet": "bold #34d399",
                "summary.text": "#d6dee8",
                "status.icon.error": "bold #ff6b6b",
                "status.icon.warning": "bold #f9a825",
                "status.icon.success": "bold #34d399",
                "status.icon.info": "bold #38bdf8",
                "status.message": "#e5e7eb",
                "code.border": "#4c1d95",
                "markdown.border": "#475569",
            }
        )
        if self.plain:
            self.console = PlainConsole()
        else:
            self.console = Console(
                no_color=False, force_terminal=True, theme=self.theme
            )
        self._panel_box = box.ROUNDED
        self._banner_box = box.DOUBLE
        self._prompt_responses = self._load_prompt_responses()

    def _load_prompt_responses(self) -> deque[dict[str, object]]:
        prompt_file = os.environ.get("POLYLOGUE_TEST_PROMPT_FILE")
        if not prompt_file:
            return deque()
        entries: deque[dict[str, object]] = deque()
        for line in Path(prompt_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    entries.append(data)
            except json.JSONDecodeError as exc:
                raise UIError(f"Invalid prompt stub entry: {line}") from exc
        return entries

    def _pop_prompt_response(self, kind: str) -> dict[str, object] | None:
        if not self._prompt_responses:
            return None
        entry = self._prompt_responses.popleft()
        expected = entry.get("type")
        if expected and expected != kind:
            raise UIError(f"Prompt stub expected '{expected}' but got '{kind}'")
        return entry

    def banner(self, title: str, subtitle: str | None = None) -> None:
        """Display a banner message."""
        if self.plain:
            self.console.print(f"== {title} ==")
            if subtitle:
                self.console.print(subtitle)
            return

        title_text = Text(style="banner.title")
        title_text.append("◈ ", style="banner.icon")
        title_text.append(title)
        if subtitle:
            title_text.append(f"\n{subtitle}", style="banner.subtitle")
        panel = Panel(
            Align.left(title_text),
            border_style="banner.border",
            box=self._banner_box,
            padding=(1, 3),
        )
        self.console.print(panel)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        """Display a summary panel."""
        text = "\n".join(lines)
        if self.plain:
            self.console.print(f"-- {title} --")
            if text:
                self.console.print(text)
            return

        summary_text = Text()
        for line in lines:
            summary_text.append("• ", style="summary.bullet")
            summary_text.append(line + "\n", style="summary.text")
        panel = Panel(
            summary_text if summary_text.plain else Text(text, style="summary.text"),
            title=f"  {title}  ",
            title_align="left",
            border_style="panel.border",
            box=self._panel_box,
            padding=(1, 2),
        )
        self.console.print(panel)

    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        """Ask for confirmation."""
        if self.plain:
            return default
        response = self._pop_prompt_response("confirm")
        if response is not None:
            if response.get("use_default"):
                return default
            value = response.get("value")
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {"y", "yes", "true", "1"}:
                    return True
                if lowered in {"n", "no", "false", "0"}:
                    return False
        result = questionary.confirm(prompt, default=default).ask()
        return default if result is None else result

    def choose(self, prompt: str, options: list[str]) -> str | None:
        """Choose from a list of options."""
        if not options:
            return None
        if self.plain:
            return None
        response = self._pop_prompt_response("choose")
        if response is not None:
            if response.get("use_default"):
                return options[0] if options else None
            if "value" in response and response["value"] in options:
                return response["value"]
            if "index" in response:
                try:
                    idx = int(response["index"])  # type: ignore[arg-type]
                    if 0 <= idx < len(options):
                        return options[idx]
                except (KeyError, ValueError, TypeError):
                    # Response missing index, or index not numeric/valid
                    pass
        if len(options) > 12:
            result = questionary.autocomplete(
                prompt, choices=options, match_middle=True
            ).ask()
        else:
            result = questionary.select(prompt, choices=options).ask()
        return result

    def input(self, prompt: str, *, default: str | None = None) -> str | None:
        """Get text input from user."""
        if self.plain:
            return default
        response = self._pop_prompt_response("input")
        if response is not None:
            if response.get("use_default"):
                return default
            if "value" in response:
                value = response["value"]
                return None if value is None else str(value)
        result = questionary.text(prompt, default=default or "").ask()
        return result if result else default

    def render_markdown(self, content: str) -> None:
        """Render Markdown content."""
        if self.plain:
            self.console.print(content)
            return

        md = Markdown(content, style="summary.text")
        panel = Panel(
            md,
            border_style="markdown.border",
            box=self._panel_box,
            padding=(1, 2),
        )
        self.console.print(panel)

    def render_code(self, code: str, language: str = "python") -> None:
        """Render syntax-highlighted code."""
        if self.plain:
            self.console.print(code)
            return

        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        panel = Panel(
            syntax,
            border_style="code.border",
            box=self._panel_box,
            padding=(0, 1),
        )
        self.console.print(panel)

    def render_diff(self, old_text: str, new_text: str, filename: str = "file") -> None:
        """Render a diff between two texts."""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )

        diff_text = "".join(diff)

        if self.plain:
            self.console.print(diff_text)
            return

        try:
            with self.console.pager():
                lexer = get_lexer_by_name("diff")
                highlighted = highlight(diff_text, lexer, TerminalFormatter())
                self.console.print(highlighted, markup=False, highlight=False)
        except (ImportError, AttributeError):
            # Pygments not available or lexer not found - fall back to basic syntax
            syntax = Syntax(diff_text, "diff", theme="ansi_dark")
            self.console.print(syntax)

    def error(self, message: str) -> None:
        """Display an error message."""
        self._status("✗", "status.icon.error", message)

    def warning(self, message: str) -> None:
        """Display a warning message."""
        self._status("!", "status.icon.warning", message)

    def success(self, message: str) -> None:
        """Display a success message."""
        self._status("✓", "status.icon.success", message)

    def info(self, message: str) -> None:
        """Display an info message."""
        if self.plain:
            self.console.print(message)
            return
        self._status("ℹ", "status.icon.info", message)

    def _status(self, icon: str, icon_style: str, message: str) -> None:
        if self.plain:
            prefix = icon if icon in {"✓", "✗"} else icon.upper()
            self.console.print(f"{prefix} {message}")
            return
        text = Text()
        text.append(f"{icon} ", style=icon_style)
        text.append(message, style="status.message")
        self.console.print(text)


@dataclass
class PlainConsoleFacade(ConsoleFacade):
    """Plain console facade for non-interactive environments."""

    def __post_init__(self) -> None:
        self.plain = True
        self.console = PlainConsole()


def create_console_facade(plain: bool) -> ConsoleFacade:
    """Create a console facade."""
    if plain:
        return PlainConsoleFacade(plain=True)
    return ConsoleFacade(plain=False)
