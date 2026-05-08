"""Console protocol and plain-console shim for the UI facade."""

from __future__ import annotations

import io
from typing import Protocol, runtime_checkable

from rich.console import Console as RichConsole
from rich.errors import MarkupError
from rich.table import Table
from rich.text import Text


@runtime_checkable
class ConsoleLike(Protocol):
    def print(self, *objects: object, **kwargs: object) -> None: ...


def _render_plain_object(obj: object, *, markup: bool = True) -> str:
    if isinstance(obj, Text):
        return obj.plain
    if isinstance(obj, str):
        if not markup:
            return obj
        try:
            return Text.from_markup(obj).plain
        except MarkupError:
            return obj

    if isinstance(obj, Table):
        buf = io.StringIO()
        tmp = RichConsole(file=buf, highlight=False, no_color=True)
        tmp.print(obj)
        return buf.getvalue().rstrip()

    return str(obj)


class PlainConsole:
    """Minimal Console shim for non-interactive mode."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def print(self, *objects: object, **kwargs: object) -> None:
        markup = bool(kwargs.get("markup", True))
        parts = [_render_plain_object(obj, markup=markup) for obj in objects]
        print(" ".join(parts))


__all__ = ["ConsoleLike", "PlainConsole"]
