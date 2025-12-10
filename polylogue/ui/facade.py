"""Pure Python UI facade using Questionary and Rich (no external binaries)."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol

import questionary
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


class ConsoleLike(Protocol):
    def print(self, *objects: object, **kwargs: object) -> None:
        ...


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

    def __post_init__(self) -> None:
        if self.plain:
            self.console = PlainConsole()
        else:
            self.console = Console(no_color=False, force_terminal=True)

    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        """Display a banner message."""
        if self.plain:
            self.console.print(f"== {title} ==")
            if subtitle:
                self.console.print(subtitle)
            return

        body = title if not subtitle else f"{title}\n{subtitle}"
        panel = Panel(Text(body, style="bold cyan"), border_style="blue")
        self.console.print(panel)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        """Display a summary panel."""
        text = "\n".join(lines)
        if self.plain:
            self.console.print(f"-- {title} --")
            if text:
                self.console.print(text)
            return

        panel = Panel(Text(text), title=title, title_align="left")
        self.console.print(panel)

    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        """Ask for confirmation."""
        if self.plain:
            return default
        return questionary.confirm(prompt, default=default).ask() or default

    def choose(self, prompt: str, options: List[str]) -> Optional[str]:
        """Choose from a list of options."""
        if not options:
            return None
        if self.plain:
            return None
        result = questionary.select(prompt, choices=options).ask()
        return result

    def input(self, prompt: str, *, default: Optional[str] = None) -> Optional[str]:
        """Get text input from user."""
        if self.plain:
            return default
        result = questionary.text(prompt, default=default or "").ask()
        return result if result else default

    def render_markdown(self, content: str) -> None:
        """Render Markdown content."""
        if self.plain:
            self.console.print(content)
            return

        md = Markdown(content)
        self.console.print(md)

    def render_code(self, code: str, language: str = "python") -> None:
        """Render syntax-highlighted code."""
        if self.plain:
            self.console.print(code)
            return

        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)

    def render_diff(self, old_text: str, new_text: str, filename: str = "file") -> None:
        """Render a diff between two texts."""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm=""
        )

        diff_text = "".join(diff)

        if self.plain:
            self.console.print(diff_text)
            return

        # Use pygments for diff highlighting
        try:
            lexer = get_lexer_by_name("diff")
            highlighted = highlight(diff_text, lexer, TerminalFormatter())
            self.console.print(highlighted, markup=False, highlight=False)
        except Exception:
            # Fallback to plain diff
            self.console.print(diff_text, markup=False)

    def error(self, message: str) -> None:
        """Display an error message."""
        if self.plain:
            self.console.print(f"ERROR: {message}")
        else:
            self.console.print(f"[bold red]ERROR:[/bold red] {message}")

    def warning(self, message: str) -> None:
        """Display a warning message."""
        if self.plain:
            self.console.print(f"WARNING: {message}")
        else:
            self.console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")

    def success(self, message: str) -> None:
        """Display a success message."""
        if self.plain:
            self.console.print(f"SUCCESS: {message}")
        else:
            self.console.print(f"[bold green]✓[/bold green] {message}")

    def info(self, message: str) -> None:
        """Display an info message."""
        if self.plain:
            self.console.print(message)
        else:
            self.console.print(f"[cyan]ℹ[/cyan] {message}")


@dataclass
class PlainConsoleFacade(ConsoleFacade):
    """Plain console facade for non-interactive environments."""

    def __post_init__(self) -> None:
        self.plain = True
        self.console = PlainConsole()


@dataclass
class InteractiveConsoleFacade(ConsoleFacade):
    """Interactive console facade using pure Python libraries."""

    def __post_init__(self) -> None:
        self.plain = False
        self.console = Console(no_color=False, force_terminal=True)


def create_console_facade(plain: bool) -> ConsoleFacade:
    """Create a console facade with no external binary dependencies."""
    if plain:
        return PlainConsoleFacade(plain=True)
    return InteractiveConsoleFacade(plain=False)
