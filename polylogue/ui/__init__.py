from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .facade import ConsoleFacade, ConsoleLike, create_console_facade

__all__ = [
    "UI",
    "create_ui",
    "ConsoleFacade",
    "ConsoleLike",
    "Console",
]


class UI:
    """High-level UI abstraction delegating to a console facade."""

    def __init__(self, plain: bool) -> None:
        try:
            self._facade: ConsoleFacade = create_console_facade(plain)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        self._plain_warnings: set[str] = set()

    @property
    def plain(self) -> bool:
        return self._facade.plain

    @property
    def console(self) -> ConsoleLike:
        return self._facade.console  # type: ignore[return-value]

    @console.setter
    def console(self, value: ConsoleLike) -> None:
        self._facade.console = value  # type: ignore[assignment]

    # Presentation helpers -------------------------------------------------
    def banner(self, title: str, subtitle: str | None = None) -> None:
        self._facade.banner(title, subtitle)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        self._facade.summary(title, lines)

    def render_markdown(self, content: str) -> None:
        self._facade.render_markdown(content)

    def render_code(self, code: str, language: str = "python") -> None:
        self._facade.render_code(code, language)

    def render_diff(self, old_text: str, new_text: str, filename: str = "file") -> None:
        self._facade.render_diff(old_text, new_text, filename)

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

    def choose(self, prompt: str, options: list[str]) -> str | None:
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
                self._print_notice("Enter a number corresponding to your choice.")
            return None
        return self._facade.choose(prompt, options)

    def input(self, prompt: str, *, default: str | None = None) -> str | None:
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
    def progress(self, description: str, total: int | None = None) -> _PlainProgressTracker | _RichProgressTracker:
        if self.plain:
            return _PlainProgressTracker(self.console, description, total)
        count_format = "{task.completed:.0f}/{task.total:.0f}" if total is not None else "{task.completed:.0f}"
        columns = (
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn(count_format),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress = Progress(
            *columns, console=self.console if isinstance(self.console, Console) else None, transient=True
        )
        task_id: TaskID = progress.add_task(description, total=total)
        return _RichProgressTracker(progress, task_id)

    # Internal utilities ---------------------------------------------------
    def _warn_plain(self, topic: str) -> None:
        if topic in self._plain_warnings:
            return
        self._plain_warnings.add(topic)
        self._print_notice(f"Plain mode cannot prompt for {topic}; rerun with --interactive or pass explicit flags.")

    def _abort_plain_prompt(self, topic: str) -> None:
        self._warn_plain(topic)
        raise SystemExit(1)

    def _print_notice(self, text: str) -> None:
        if self.plain:
            self.console.print(text)
        else:
            self.console.print(f"[yellow]{text}[/yellow]")


def create_ui(plain: bool) -> UI:
    return UI(plain=plain)


class _PlainProgressTracker:
    def __init__(self, console: ConsoleLike, description: str, total: int | None) -> None:
        self._console = console
        self._description = description
        coerced = self._coerce_int(total) if total is not None else None
        self._total: int | float | None = coerced
        if self._total is None and total is not None:
            self._total = float(total)
            self._completed: float | int = 0.0
            self._use_float = True
        else:
            self._completed = 0
            self._use_float = False
        self._console.print(f"{description}...")

    def __enter__(self) -> _PlainProgressTracker:
        return self

    def advance(self, advance: float = 1) -> None:
        as_int = self._coerce_int(advance)
        if as_int is not None and not self._use_float:
            self._completed += as_int
        else:
            self._use_float = True
            self._completed = float(self._completed) + float(advance)

    def update(self, *, total: int | None = None, description: str | None = None) -> None:
        if description and description != self._description:
            self._description = description
            self._console.print(f"{description}...")
        if total is not None:
            as_int = self._coerce_int(total)
            if as_int is not None and not self._use_float:
                self._total = as_int
            else:
                self._total = float(total)
                self._use_float = True

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        status = "aborted" if exc_type else "complete"
        suffix = ""
        if self._total is not None:
            suffix = f" ({self._format_value(self._completed)}/{self._format_value(self._total)})"
        self._console.print(f"{self._description} {status}{suffix}")

    @staticmethod
    def _coerce_int(value: float | None) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            rounded = round(value)
            if math.isclose(value, rounded, abs_tol=1e-4):
                return int(rounded)
        return None

    @staticmethod
    def _format_value(value: float | int) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            rounded = round(value)
            if math.isclose(value, rounded, abs_tol=1e-4):
                return str(int(rounded))
        return f"{float(value):.2f}".rstrip("0").rstrip(".")


class _RichProgressTracker:
    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def __enter__(self) -> _RichProgressTracker:
        self._progress.__enter__()
        return self

    def advance(self, advance: float = 1) -> None:
        self._progress.advance(self._task_id, advance)

    def update(self, *, total: int | None = None, description: str | None = None) -> None:
        if total is not None:
            self._progress.update(self._task_id, total=total)
        if description is not None:
            self._progress.update(self._task_id, description=description)

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self._progress.__exit__(exc_type, exc, tb)
