"""Terminal UI facade."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import questionary
from rich import box
from rich.console import Console
from rich.theme import Theme

from polylogue.errors import PolylogueError
from polylogue.ui.facade_console import ConsoleLike, PlainConsole
from polylogue.ui.facade_prompts import (
    _NO_STUB_RESPONSE,
    consume_choose_stub,
    consume_confirm_stub,
    consume_input_stub,
    load_prompt_responses,
    pop_prompt_response,
    require_plain_prompt_tty,
)
from polylogue.ui.facade_rendering import (
    render_banner as _render_banner,
)
from polylogue.ui.facade_rendering import (
    render_code as _render_code,
)
from polylogue.ui.facade_rendering import (
    render_diff as _render_diff,
)
from polylogue.ui.facade_rendering import (
    render_markdown as _render_markdown,
)
from polylogue.ui.facade_rendering import (
    render_status as _render_status,
)
from polylogue.ui.facade_rendering import (
    render_summary as _render_summary,
)
from polylogue.ui.theme import rich_theme_styles

__all__ = ["ConsoleFacade", "ConsoleLike", "PlainConsole", "UIError", "create_console_facade"]


class UIError(PolylogueError):
    """UI-related errors (prompt stubs, user interaction)."""

    def __init__(self, message: str, *, prompt_topic: str | None = None) -> None:
        super().__init__(message)
        self.prompt_topic = prompt_topic


@dataclass
class ConsoleFacade:
    """Pure Python facade for terminal UI."""

    plain: bool
    console: Console | PlainConsole = field(init=False)
    theme: Theme = field(init=False, repr=False)
    _prompt_responses: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.theme = Theme(rich_theme_styles())
        if self.plain:
            self.console = PlainConsole()
        else:
            self.console = Console(no_color=False, force_terminal=True, theme=self.theme)
        self._panel_box = box.ROUNDED
        self._banner_box = box.DOUBLE
        self._prompt_responses = load_prompt_responses(UIError)

    def _load_prompt_responses(self):
        return load_prompt_responses(UIError)

    def _pop_prompt_response(self, kind: str):
        return pop_prompt_response(self._prompt_responses, kind, UIError)

    def _require_plain_prompt_tty(self, prompt_topic: str) -> None:
        require_plain_prompt_tty(prompt_topic, UIError)

    def _consume_confirm_stub(self, *, default: bool):
        return consume_confirm_stub(self._prompt_responses, default=default, ui_error_cls=UIError)

    def _consume_choose_stub(self, options: list[str]):
        return consume_choose_stub(self._prompt_responses, options, UIError)

    def _consume_input_stub(self, *, default: str | None):
        return consume_input_stub(self._prompt_responses, default=default, ui_error_cls=UIError)

    def banner(self, title: str, subtitle: str | None = None) -> None:
        _render_banner(self.console, plain=self.plain, banner_box=self._banner_box, title=title, subtitle=subtitle)

    def summary(self, title: str, lines: Iterable[str]) -> None:
        _render_summary(self.console, plain=self.plain, panel_box=self._panel_box, title=title, lines=list(lines))

    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        stub_result = self._consume_confirm_stub(default=default)
        if stub_result is not _NO_STUB_RESPONSE:
            return bool(stub_result)
        if self.plain:
            self._require_plain_prompt_tty("confirmation prompts")
            try:
                value = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip()
            except EOFError:
                return default
            if not value:
                return default
            return value.lower() in {"y", "yes"}
        result = questionary.confirm(prompt, default=default).ask()
        return default if result is None else result

    def choose(self, prompt: str, options: list[str]) -> str | None:
        if not options:
            return None
        stub_result = self._consume_choose_stub(options)
        if stub_result is not _NO_STUB_RESPONSE:
            return stub_result
        if self.plain:
            self._require_plain_prompt_tty("menu selections")
            for idx, option in enumerate(options, start=1):
                self.console.print(f"{idx}. {option}")
            while True:
                try:
                    value = input(f"{prompt} [1-{len(options)}]: ").strip()
                except EOFError:
                    return None
                if not value:
                    return None
                if value.isdigit():
                    selected = int(value)
                    if 1 <= selected <= len(options):
                        return options[selected - 1]
                self.console.print("Enter a number corresponding to your choice.")
        if len(options) > 12:
            result: str | None = questionary.autocomplete(prompt, choices=options, match_middle=True).ask()
        else:
            result = questionary.select(prompt, choices=options).ask()
        return result

    def input(self, prompt: str, *, default: str | None = None) -> str | None:
        stub_result = self._consume_input_stub(default=default)
        if stub_result is not _NO_STUB_RESPONSE:
            return stub_result
        if self.plain:
            self._require_plain_prompt_tty("text input")
            suffix = f" [{default}]" if default else ""
            try:
                value = input(f"{prompt}{suffix}: ").strip()
            except EOFError:
                return default
            return value or default
        result: str | None = questionary.text(prompt, default=default or "").ask()
        return result if result else default

    def render_markdown(self, content: str) -> None:
        _render_markdown(self.console, plain=self.plain, panel_box=self._panel_box, content=content)

    def render_code(self, code: str, language: str = "python") -> None:
        _render_code(self.console, plain=self.plain, panel_box=self._panel_box, code=code, language=language)

    def render_diff(self, old_text: str, new_text: str, filename: str = "file") -> None:
        _render_diff(self.console, plain=self.plain, filename=filename, old_text=old_text, new_text=new_text)

    def error(self, message: str) -> None:
        self._status("✗", "status.icon.error", message)

    def warning(self, message: str) -> None:
        self._status("!", "status.icon.warning", message)

    def success(self, message: str) -> None:
        self._status("✓", "status.icon.success", message)

    def info(self, message: str) -> None:
        if self.plain:
            self.console.print(message)
            return
        self._status("ℹ", "status.icon.info", message)

    def _status(self, icon: str, icon_style: str, message: str) -> None:
        _render_status(self.console, plain=self.plain, icon=icon, icon_style=icon_style, message=message)


def create_console_facade(plain: bool) -> ConsoleFacade:
    """Create a console facade."""
    return ConsoleFacade(plain=plain)
