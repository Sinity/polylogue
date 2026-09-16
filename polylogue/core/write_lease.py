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


def grant_write_lease_thread() -> Any:
    return _implementation().grant_write_lease_thread()


def bind_write_lease_thread(grant: Any) -> None:
    _implementation().bind_write_lease_thread(grant)


@contextmanager
def install_archive_write_guard() -> Iterator[None]:
    """Intercept writable archive-tier opens for the duration of the block.

    The guard itself lives with SQLite storage; this adapter keeps the daemon's
    arming site ring-neutral, exactly as the lease functions above do.
    """
    with importlib.import_module("polylogue.storage.sqlite.write_guard").install_archive_write_guard():
        yield


@contextmanager
def declared_unguarded_write(reason: str) -> Iterator[None]:
    """Run a declared non-daemon archive authority outside the guard."""
    with importlib.import_module("polylogue.storage.sqlite.write_guard").declared_unguarded_write(reason):
        yield


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


def delegate_write_lease() -> Any:
    return _implementation().delegate_write_lease()


@contextmanager
def adopt_write_lease(delegation: Any) -> Iterator[Any]:
    with _implementation().adopt_write_lease(delegation) as lease:
        yield lease


def __getattr__(name: str) -> Any:
    """Expose the implementation's lease types without a ring-crossing import."""
    if name in {
        "WriteLease",
        "WriteLeaseDelegation",
        "WriteLeaseThreadGrant",
        "UnleasedWriteError",
        "WriteHoldExceededError",
    }:
        return getattr(_implementation(), name)
    raise AttributeError(name)
