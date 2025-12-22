"""Terminal UI facade.

Polylogue ships with a small UI layer. In interactive mode we hard-require the
external helpers provided by the Nix devshell (gum, skim, bat, glow, delta).
Plain mode falls back to basic stdout prompting.
"""
from __future__ import annotations

import difflib
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol

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

    theme: Theme = field(init=False, repr=False)

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
            self.console = Console(no_color=False, force_terminal=True, theme=self.theme)
        self._panel_box = box.ROUNDED
        self._banner_box = box.DOUBLE

    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
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
            lineterm=""
        )

        diff_text = "".join(diff)

        if self.plain:
            self.console.print(diff_text)
            return

        try:
            lexer = get_lexer_by_name("diff")
            highlighted = highlight(diff_text, lexer, TerminalFormatter())
            self.console.print(highlighted, markup=False, highlight=False)
        except Exception:
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


@dataclass
class InteractiveConsoleFacade(ConsoleFacade):
    """Deprecated placeholder kept for backwards imports."""

    def __post_init__(self) -> None:  # pragma: no cover - legacy alias
        self.plain = False
        self.console = Console(no_color=False, force_terminal=True)


def create_console_facade(plain: bool) -> ConsoleFacade:
    """Create a console facade.

    Interactive mode uses gum + friends. Missing deps are treated as a hard
    failure (no graceful degradation).
    """
    if plain:
        return PlainConsoleFacade(plain=True)
    _ensure_interactive_deps()
    return GumConsoleFacade(plain=False)


_REQUIRED_INTERACTIVE_CMDS = ("gum", "sk", "bat", "glow", "delta")


def _ensure_interactive_deps() -> None:
    missing: List[str] = []
    for cmd in _REQUIRED_INTERACTIVE_CMDS:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    if missing:
        raise RuntimeError(
            "Interactive dependencies missing: "
            + ", ".join(missing)
            + ". Enter `nix develop` in the repo to load required helpers."
        )


@dataclass
class GumConsoleFacade(ConsoleFacade):
    """Interactive facade backed by gum/skim helpers."""

    def __post_init__(self) -> None:
        self.plain = False
        self.console = Console(no_color=False, force_terminal=True)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        markdown_lines = [f"## {title}"] + [f"- {line}" for line in lines]
        markdown = "\n".join(markdown_lines) + "\n"
        proc = subprocess.run(
            ["gum", "format"],
            input=markdown,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "gum format failed")
        # gum emits formatted markdown; print verbatim (no rich markup).
        self.console.print(proc.stdout.rstrip("\n"), markup=False, highlight=False)

    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        cmd = ["gum", "confirm", "--prompt", prompt]
        if default:
            cmd.append("--default")
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            return True
        if proc.stderr and "--prompt" in proc.stderr and "unknown" in proc.stderr.lower():
            # Newer gum expects prompt as a positional argument.
            retry_cmd = ["gum", "confirm"]
            if default:
                retry_cmd.append("--default")
            retry_cmd.append(prompt)
            retry = subprocess.run(retry_cmd, text=True, capture_output=True, check=False)
            return retry.returncode == 0
        return False

    def choose(self, prompt: str, options: List[str]) -> Optional[str]:
        if not options:
            return None
        proc = subprocess.run(
            ["sk", "--prompt", prompt],
            input="\n".join(options),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        choice = proc.stdout.strip()
        return choice if choice else None

    def input(self, prompt: str, *, default: Optional[str] = None) -> Optional[str]:
        cmd = ["gum", "input", "--prompt", prompt]
        if default is not None:
            cmd += ["--value", default]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            return default
        value = proc.stdout.strip()
        return value or default
