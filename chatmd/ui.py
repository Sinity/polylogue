from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from rich.console import Console
from rich.panel import Panel


@dataclass
class UI:
    """Abstraction over interactive (gum) and plain terminal interactions."""

    plain: bool

    def __post_init__(self) -> None:
        self.console = Console(no_color=self.plain, force_terminal=not self.plain)

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
        self.console.print(Panel(text, title=title))

    # Prompting ------------------------------------------------------------
    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        if self.plain:
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
            return options[0]
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
