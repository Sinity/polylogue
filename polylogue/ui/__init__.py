from __future__ import annotations

from typing import Iterable, List, Optional, Set
import sys

import shutil as _shutil
import subprocess as _subprocess
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)

from .facade import Console, ConsoleFacade, ConsoleLike, create_console_facade

shutil = _shutil
subprocess = _subprocess

__all__ = [
    "UI",
    "create_ui",
    "ConsoleFacade",
    "ConsoleLike",
    "Console",
    "shutil",
    "subprocess",
]


class UI:
    """High-level UI abstraction delegating to a console facade."""

    def __init__(self, plain: bool) -> None:
        try:
            self._facade: ConsoleFacade = create_console_facade(plain)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
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
            if not sys.stdin.isatty():
                self._abort_plain_prompt("confirmation prompts")
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
            if not sys.stdin.isatty():
                self._abort_plain_prompt("menu selections")
            for idx, option in enumerate(options, start=1):
                self.console.print(f"{idx}. {option}")
            while True:
                try:
                    response = input(f"{prompt} [1-{len(options)}]: ").strip()
                except EOFError:
                    return None
                if not response:
                    return None
                if response.isdigit():
                    value = int(response)
                    if 1 <= value <= len(options):
                        return options[value - 1]
                self.console.print("[yellow]Enter a number corresponding to your choice.")
            return None
        return self._facade.choose(prompt, options)

    def input(self, prompt: str, *, default: Optional[str] = None) -> Optional[str]:
        if self.plain:
            if not sys.stdin.isatty():
                self._abort_plain_prompt("text input")
            suffix = f" [{default}]" if default else ""
            try:
                value = input(f"{prompt}{suffix}: ").strip()
            except EOFError:
                return default
            return value or default
        return self._facade.input(prompt, default=default)

    # Progress -------------------------------------------------------------
    def progress(self, description: str, total: Optional[int] = None):
        if self.plain:
            return _NullProgressTracker()
        columns = (
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}" if total is not None else "{task.completed}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress = Progress(*columns, console=self.console if isinstance(self.console, Console) else None, transient=True)
        task_id: TaskID = progress.add_task(description, total=total)
        return _RichProgressTracker(progress, task_id)

    # Internal utilities ---------------------------------------------------
    def _warn_plain(self, topic: str) -> None:
        if topic in self._plain_warnings:
            return
        self._plain_warnings.add(topic)
        self.console.print(f"[yellow]Plain mode cannot prompt for {topic}; rerun with --interactive or pass explicit flags.")

    def _abort_plain_prompt(self, topic: str) -> None:
        self._warn_plain(topic)
        raise SystemExit(1)


def create_ui(plain: bool) -> UI:
    return UI(plain=plain)


class _NullProgressTracker:
    def __enter__(self):
        return self

    def advance(self, *_args, **_kwargs) -> None:
        return None

    def update(self, *_args, **_kwargs) -> None:
        return None

    def __exit__(self, *_exc) -> None:
        return None


class _RichProgressTracker:
    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def __enter__(self):
        self._progress.__enter__()
        return self

    def advance(self, advance: float = 1.0) -> None:
        self._progress.advance(self._task_id, advance)

    def update(self, *, total: Optional[int] = None, description: Optional[str] = None) -> None:
        kwargs = {}
        if total is not None:
            kwargs["total"] = total
        if description is not None:
            kwargs["description"] = description
        if kwargs:
            self._progress.update(self._task_id, **kwargs)

    def __exit__(self, exc_type, exc, tb):
        self._progress.__exit__(exc_type, exc, tb)
