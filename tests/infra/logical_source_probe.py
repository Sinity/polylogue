"""Observe the lifecycle of a private logical-source connection.

``open_logical_source`` rebuilds a retained export into a temporary SQLite
file and unlinks it while the connection is open, so the connection is the
only reference keeping that inode alive. A parser that returns without
closing it leaks the inode for the rest of the process -- and that leak is
invisible to any test that only inspects the rows the parser produced.

Two independent witnesses live here:

* :func:`record_logical_source_connections` patches the production
  ``open_logical_source`` symbol a parser module imported and records whether
  each handed-out connection was closed. The probe forwards every attribute
  to the real connection, so the parser under test runs its real queries.
* :func:`open_reconstruction_handles` counts this process's open file
  descriptors that still point at a deleted reconstruction, with no patching
  at all.
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

#: The prefix ``open_logical_source`` gives every reconstruction it creates.
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

    def __enter__(self) -> ConnectionProbe:
        # sqlite3's own context manager commits or rolls back and never
        # closes. Reproduce that exactly, so a caller that relies on it
        # leaves this probe open.
        object.__getattribute__(self, "_connection").__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> Any:
        return object.__getattribute__(self, "_connection").__exit__(*exc_info)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)
        object.__getattribute__(self, "_connection").close()


@contextmanager
def record_logical_source_connections(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> Iterator[list[ConnectionProbe]]:
    """Record every connection *module* obtains from ``open_logical_source``.

    *module* must be the production module that imported the symbol, so the
    patch sits on the real read route rather than on a test-only wrapper.
    """
    opened: list[ConnectionProbe] = []
    real_open = sqlite_export.open_logical_source

    def _open(path: Path, **kwargs: Any) -> ConnectionProbe:
        probe = ConnectionProbe(real_open(path, **kwargs))
        opened.append(probe)
        return probe

    monkeypatch.setattr(module, "open_logical_source", _open)
    try:
        yield opened
    finally:
        for probe in opened:
            if not probe.closed:
                probe.close()


def open_reconstruction_handles() -> int:
    """Count open descriptors pointing at an unlinked reconstruction."""
    handles = 0
    descriptors = Path("/proc/self/fd")
    for entry in descriptors.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        name = target.removesuffix(" (deleted)")
        if Path(name).name.startswith(RECONSTRUCTION_PREFIX):
            handles += 1
    return handles
