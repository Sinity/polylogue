from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]


class PlainConsole:
    """Minimal Console shim when Rich is unavailable."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def print(self, *objects: object, **_: object) -> None:
        # Mimic ``rich.console.Console.print`` signature for basic usage.
        text = " ".join(str(obj) for obj in objects)
        print(text)


@dataclass
class UI:
    """Abstraction over interactive (gum) and plain terminal interactions."""

    plain: bool

    def __post_init__(self) -> None:
        if Console is None:
            self.plain = True
            self.console = PlainConsole()
        else:
            self.console = Console(no_color=self.plain, force_terminal=not self.plain)
        self._plain_warnings: set[str] = set()

    def _warn_plain(self, topic: str) -> None:
        if topic in self._plain_warnings:
            return
        self._plain_warnings.add(topic)
        self.console.print(
            f"[yellow]Plain mode cannot prompt for {topic}; using defaults."
        )

    # Presentation helpers -------------------------------------------------
    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        if self.plain:
            self.console.print(f"== {title} ==")
            if subtitle:
                self.console.print(subtitle)
            return
        body = title if not subtitle else f"{title}\n{subtitle}"
        try:
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
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.stdout.strip():
                self.console.print(result.stdout.rstrip())
            else:
                self.console.print(body)
        except Exception:
            self.console.print(body)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        text = "\n".join(lines)
        if self.plain:
            self.console.print(f"-- {title} --")
            self.console.print(text)
            return
        try:
            result = subprocess.run(
                ["gum", "format"],
                input=f"## {title}\n{text}\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.stdout.strip():
                self.console.print(result.stdout.rstrip())
                return
        except Exception:
            pass
        if Console is None or Panel is None:
            self.console.print(f"== {title} ==\n{text}")
            return
        renderable = Text(text) if Text is not None else text
        self.console.print(Panel(renderable, title=title))

    # Prompting ------------------------------------------------------------
    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        if self.plain:
            self._warn_plain("confirmation")
            return default
        try:
            result = subprocess.run(
                ["gum", "confirm", "--default", "true" if default else "false", prompt],
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return default

    def choose(self, prompt: str, options: List[str]) -> Optional[str]:
        if not options:
            return None
        if self.plain:
            self._warn_plain("selection")
            return None
        try:
            result = subprocess.run(
                ["gum", "choose", "--header", prompt, *options],
                check=False,
                stdout=subprocess.PIPE,
                text=True,
            )
            choice = result.stdout.strip()
            return choice or None
        except Exception:
            return options[0]

    def input(self, prompt: str, *, default: Optional[str] = None) -> Optional[str]:
        if self.plain:
            self._warn_plain("input")
            return default
        cmd = ["gum", "input", "--placeholder", prompt]
        if default:
            cmd.extend(["--value", default])
        try:
            result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, text=True)
            value = result.stdout.strip()
            if not value:
                return default
            return value
        except Exception:
            return default


def detect_plain(flag_plain: bool) -> bool:
    if flag_plain:
        return True
    # Require both stdout and stderr to be TTY for interactive behaviour.
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        return True
    return False


def create_ui(flag_plain: bool) -> UI:
    return UI(plain=detect_plain(flag_plain))
