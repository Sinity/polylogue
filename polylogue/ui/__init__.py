from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Optional, Set

import shutil as _shutil
import subprocess as _subprocess

from .facade import Console, ConsoleFacade, ConsoleLike, create_console_facade

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
except ImportError:
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    TaskProgressColumn = None
    TimeRemainingColumn = None

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

    # Progress bars --------------------------------------------------------
    @contextmanager
    def progress(self, description: str = "Processing", total: Optional[int] = None) -> Iterator[Optional[object]]:
        """Create a progress bar context manager.

        In plain mode or when Rich is unavailable, yields None and shows no progress.
        In interactive mode with Rich available, yields a Progress object with an active task.

        Args:
            description: Task description to display
            total: Total number of items (if known)

        Yields:
            Progress object with active task (or None in plain mode)

        Example:
            with ui.progress("Syncing chats", total=len(chats)) as progress:
                if progress:
                    task = progress.task_id
                    for item in chats:
                        # ... process item ...
                        progress.advance(task)
        """
        if self.plain or Progress is None:
            yield None
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task(description, total=total)
            # Create a simple wrapper that exposes task_id and advance
            class ProgressWrapper:
                def __init__(self, prog, tid):
                    self._progress = prog
                    self.task_id = tid

                def advance(self, task_id=None, advance=1):
                    self._progress.advance(task_id or self.task_id, advance=advance)

                def update(self, task_id=None, **kwargs):
                    self._progress.update(task_id or self.task_id, **kwargs)

            yield ProgressWrapper(progress, task_id)

    # Internal utilities ---------------------------------------------------
    def _warn_plain(self, topic: str) -> None:
        if topic in self._plain_warnings:
            return
        self._plain_warnings.add(topic)
        self.console.print(f"[yellow]Plain mode cannot prompt for {topic}; using defaults.")


def create_ui(flag_plain: bool) -> UI:
    return UI(plain=detect_plain(flag_plain))
