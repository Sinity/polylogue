"""Observe actual creator close on the logical-source read route.

The production context owns both native close and reconstruction cleanup.
The proxy checks that close completed after the parser leaves that context;
file-descriptor observation independently detects retained reconstructions.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from polylogue.sources import sqlite_export

#: The prefix ``logical_source_context`` gives every reconstruction it creates.
RECONSTRUCTION_PREFIX = ".polylogue-export."


class ConnectionProbe:
    """A transparent proxy over one sqlite3 connection that records ``close``."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "closed", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_connection"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_connection"), name, value)

    def observe_closed(self) -> None:
        try:
            object.__getattribute__(self, "_connection").execute("SELECT 1")
        except sqlite3.ProgrammingError:
            object.__setattr__(self, "closed", True)


@contextmanager
def record_logical_source_connections(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> Iterator[list[ConnectionProbe]]:
    """Record every connection *module* obtains from ``logical_source_context``.

    *module* must be the production module that imported the symbol, so the
    patch sits on the real read route rather than on a test-only wrapper.
    """
    opened: list[ConnectionProbe] = []
    real_open = sqlite_export.logical_source_context

    @contextmanager
    def _open(path: Path, **kwargs: Any) -> Iterator[ConnectionProbe]:
        probe: ConnectionProbe | None = None
        try:
            with real_open(path, **kwargs) as connection:
                probe = ConnectionProbe(connection)
                opened.append(probe)
                yield probe
        finally:
            if probe is not None:
                probe.observe_closed()

    monkeypatch.setattr(module, "logical_source_context", _open)
    yield opened


def open_reconstruction_handles() -> int:
    """Count open descriptors pointing at a reconstruction."""
    handles = 0
    descriptors = Path("/proc/self/fd")
    for entry in descriptors.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        name = target.removesuffix(" (deleted)")
        if any(part.startswith(RECONSTRUCTION_PREFIX) for part in Path(name).parts):
            handles += 1
    return handles
