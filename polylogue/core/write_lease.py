"""Daemon-facing adapter for the archive write lease.

The lease implementation lives with SQLite storage.  This ring-neutral adapter
keeps daemon coordination independent of the storage package while preserving
one shared authority at runtime.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _implementation() -> Any:
    return importlib.import_module("polylogue.storage.sqlite.write_lease")


def current_write_lease() -> Any:
    return _implementation().current_write_lease()


def require_write_lease(purpose: str, *, archive_root: str | Path | None = None) -> Any:
    return _implementation().require_write_lease(purpose, archive_root=archive_root)


def bind_write_lease_thread() -> None:
    _implementation().bind_write_lease_thread()


@contextmanager
def arm_write_lease_enforcement(*, armed: bool = True, process_wide: bool = False) -> Iterator[None]:
    with _implementation().arm_write_lease_enforcement(armed=armed, process_wide=process_wide):
        yield


@contextmanager
def write_lease(
    actor: str,
    *,
    max_hold_seconds: float | None = None,
    archive_root: str | Path | None = None,
    coordinator: object | None = None,
) -> Iterator[Any]:
    with _implementation().write_lease(
        actor,
        max_hold_seconds=max_hold_seconds,
        archive_root=archive_root,
        coordinator=coordinator,
    ) as lease:
        yield lease
