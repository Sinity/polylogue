from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

import shutil as _shutil
import subprocess as _subprocess

from .facade import Console, ConsoleFacade, ConsoleLike, create_console_facade

shutil = _shutil
subprocess = _subprocess

__all__ = [
    "UI",
    "create_ui",
    "detect_plain",
    "ConsoleFacade",
    "ConsoleLike",
    "Console",
    "shutil",
    "subprocess",
]


def detect_plain(flag_plain: bool) -> bool:
    if flag_plain:
        return True
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        return True
    return False


class UI:
    """High-level UI abstraction delegating to a console facade."""

    def __init__(self, plain: bool) -> None:
        self._facade: ConsoleFacade = create_console_facade(plain)
        self._plain_warnings: Set[str] = set()

    @property
    def plain(self) -> bool:
        return self._facade.plain

    @property
    def console(self) -> ConsoleLike:
        return self._facade.console

    @console.setter
    def console(self, value: ConsoleLike) -> None:
        self._facade.console = value

    # Presentation helpers -------------------------------------------------
    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        self._facade.banner(title, subtitle)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        self._facade.summary(title, lines)

    # Prompting ------------------------------------------------------------
    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        if self.plain:
            try:
                response = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip()
            except EOFError:
                return default
            if not response:
                return default
            return response.lower() in {"y", "yes"}
        return self._facade.confirm(prompt, default=default)

    def choose(self, prompt: str, options: List[str]) -> Optional[str]:
        if not options:
            return None
        if self.plain:
            for idx, option in enumerate(options, start=1):
                self.console.print(f"{idx}. {option}")
            while True:
                try:
                    response = input(f"{prompt} [1-{len(options)}]: ").strip()
                except EOFError:
                    return options[0]
                if not response:
                    return options[0]
                if response.isdigit():
                    value = int(response)
                    if 1 <= value <= len(options):
                        return options[value - 1]
                self.console.print("[yellow]Enter a number corresponding to your choice.")
            return None
        return self._facade.choose(prompt, options)

    def input(self, prompt: str, *, default: Optional[str] = None) -> Optional[str]:
        if self.plain:
            suffix = f" [{default}]" if default else ""
            try:
                value = input(f"{prompt}{suffix}: ").strip()
            except EOFError:
                return default
            return value or default
        return self._facade.input(prompt, default=default)

    # Internal utilities ---------------------------------------------------
    def _warn_plain(self, topic: str) -> None:
        if topic in self._plain_warnings:
            return
        self._plain_warnings.add(topic)
        self.console.print(f"[yellow]Plain mode cannot prompt for {topic}; using defaults.")


def create_ui(flag_plain: bool) -> UI:
    return UI(plain=detect_plain(flag_plain))
