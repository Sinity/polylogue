"""Console protocol and plain-console shim for the UI facade."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Protocol, runtime_checkable

from rich.text import Text


@runtime_checkable
class ConsoleLike(Protocol):
    def print(self, *objects: Any, **kwargs: Any) -> None: ...


class PlainConsole:
    """Minimal Console shim for non-interactive mode."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def print(self, *objects: object, **_: object) -> None:
        import io

        from rich.console import Console as RichConsole
        from rich.table import Table

        parts = []
        for obj in objects:
            if isinstance(obj, Table):
                buf = io.StringIO()
                tmp = RichConsole(file=buf, highlight=False, no_color=True)
                tmp.print(obj)
                parts.append(buf.getvalue().rstrip())
            else:
                raw = str(obj)
                with suppress(Exception):
                    raw = Text.from_markup(raw).plain
                parts.append(raw)
        print(" ".join(parts))
