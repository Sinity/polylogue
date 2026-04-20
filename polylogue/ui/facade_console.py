"""Console protocol and plain-console shim for the UI facade."""

from __future__ import annotations

import io
from contextlib import suppress
from typing import Protocol, runtime_checkable

from rich.console import Console as RichConsole
from rich.table import Table
from rich.text import Text


@runtime_checkable
class ConsoleLike(Protocol):
    def print(self, *objects: object, **kwargs: object) -> None: ...


def _plain_text_from_markup(text: str) -> str:
    with suppress(Exception):
        return Text.from_markup(text).plain
    return text


def _render_plain_object(obj: object) -> str:
    if isinstance(obj, Text):
        return obj.plain
    if isinstance(obj, str):
        return _plain_text_from_markup(obj)

    if isinstance(obj, Table):
        buf = io.StringIO()
        tmp = RichConsole(file=buf, highlight=False, no_color=True)
        tmp.print(obj)
        return buf.getvalue().rstrip()

    raw = str(obj)
    return _plain_text_from_markup(raw)


class PlainConsole:
    """Minimal Console shim for non-interactive mode."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def print(self, *objects: object, **_: object) -> None:
        parts = [_render_plain_object(obj) for obj in objects]
        print(" ".join(parts))


__all__ = ["ConsoleLike", "PlainConsole"]
