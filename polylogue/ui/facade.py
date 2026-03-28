from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Sequence

try:  # pragma: no cover - optional rich dependency
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except ImportError:  # pragma: no cover - fall back to plain console
    Console = None
    Panel = None
    Text = None


class ConsoleLike(Protocol):
    def print(self, *objects: object, **kwargs: object) -> None:
        ...


class PlainConsole:
    """Minimal Console shim when Rich is unavailable."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def print(self, *objects: object, **_: object) -> None:
        text = " ".join(str(obj) for obj in objects)
        print(text)


@dataclass
class ConsoleFacade:
    """Base facade for interacting with terminal UI."""

    plain: bool
    console: ConsoleLike = field(init=False)

    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        if self.plain:
            self.console.print(f"== {title} ==")
            if subtitle:
                self.console.print(subtitle)
            return
        self._interactive_banner(title, subtitle)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        text = "\n".join(lines)
        if self.plain:
            self.console.print(f"-- {title} --")
            if text:
                self.console.print(text)
            return
        self._interactive_summary(title, text)

    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        if self.plain:
            return default
        return self._interactive_confirm(prompt, default=default)

    def choose(self, prompt: str, options: List[str]) -> Optional[str]:
        if not options:
            return None
        if self.plain:
            return None
        return self._interactive_choose(prompt, options)

    def input(self, prompt: str, *, default: Optional[str] = None) -> Optional[str]:
        if self.plain:
            return default
        return self._interactive_input(prompt, default=default)

    # Interactive implementations -----------------------------------------
    def _interactive_banner(self, title: str, subtitle: Optional[str]) -> None:
        body = title if not subtitle else f"{title}\n{subtitle}"
        result = subprocess.run(
            [
                "gum",
                "style",
                "--border",
                "rounded",
                "--margin",
                "1",
                "--padding",
                "1",
                body,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout.strip():
            self.console.print(result.stdout.rstrip(), markup=False, highlight=False)
        else:
            self.console.print(body, markup=False, highlight=False)

    def _interactive_summary(self, title: str, text: str) -> None:
        result = subprocess.run(
            ["gum", "format"],
            input=f"## {title}\n{text}\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        output = result.stdout.rstrip()
        if output:
            self.console.print(output, markup=False, highlight=False)
            return
        renderable = Text(text) if Text is not None else text
        if Panel is not None:
            self.console.print(Panel(renderable, title=title))
        else:
            self.console.print(f"{title}\n{text}")

    def _interactive_confirm(self, prompt: str, *, default: bool) -> bool:
        cmd: List[str] = ["gum", "confirm", "--prompt", prompt]
        if default:
            cmd.append("--default")
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0

    def _interactive_choose(self, prompt: str, options: List[str]) -> Optional[str]:
        result = subprocess.run(
            ["gum", "choose", "--header", prompt, *options],
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        )
        choice = result.stdout.strip()
        return choice or None

    def _interactive_input(self, prompt: str, *, default: Optional[str]) -> Optional[str]:
        cmd: List[str] = ["gum", "input", "--placeholder", prompt]
        if default:
            cmd.extend(["--value", default])
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, text=True)
        value = result.stdout.strip()
        if value:
            return value
        return default


@dataclass
class PlainConsoleFacade(ConsoleFacade):
    def __post_init__(self) -> None:
        self.console = PlainConsole()
        self.plain = True


@dataclass
class InteractiveConsoleFacade(ConsoleFacade):
    """Rich + gum backed facade that can fall back to plain mode."""

    def __post_init__(self) -> None:
        missing = [cmd for cmd in ("gum", "sk") if shutil.which(cmd) is None]
        if Console is None or missing:
            # Fall back to plain facade when Rich or gum is not available.
            self.console = PlainConsole()
            self.plain = True
            warning = " and ".join(missing) + " command(s)" if missing else "interactive dependencies"
            self.console.print(f"[plain]Interactive UI unavailable (missing {warning}); using plain output.")
        else:
            self.console = Console(no_color=False, force_terminal=True)


def create_console_facade(plain: bool) -> ConsoleFacade:
    if plain:
        return PlainConsoleFacade(plain=True)
    facade = InteractiveConsoleFacade(plain=False)
    if facade.plain:
        return PlainConsoleFacade(plain=True)
    return facade
