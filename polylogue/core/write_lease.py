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


def coordinator_write_lease_active() -> bool:
    """Observe the current execution unit's actual coordinator-owned lease."""
    return bool(_implementation().coordinator_write_lease_active())


def require_write_lease(purpose: str, *, archive_root: str | Path | None = None) -> Any:
    return _implementation().require_write_lease(purpose, archive_root=archive_root)


def current_sql_custody() -> Any:
    return _implementation().current_sql_custody()


def grant_write_lease_thread() -> Any:
    return _implementation().grant_write_lease_thread()


def bind_write_lease_thread(grant: Any) -> None:
    _implementation().bind_write_lease_thread(grant)


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


def async_write_lease(
    actor: str,
    *,
    max_hold_seconds: float | None = None,
    archive_root: str | Path,
    coordinator: object | None = None,
) -> Any:
    return _implementation().async_write_lease(
        actor,
        max_hold_seconds=max_hold_seconds,
        archive_root=archive_root,
        coordinator=coordinator,
    )


def archive_write_custody(archive_root: str | Path) -> Any:
    return _implementation().archive_write_custody(archive_root)


def delegate_write_lease() -> Any:
    return _implementation().delegate_write_lease()


@contextmanager
def adopt_write_lease(delegation: Any) -> Iterator[Any]:
    with _implementation().adopt_write_lease(delegation) as lease:
        yield lease


def __getattr__(name: str) -> Any:
    """Expose the implementation's lease types without a ring-crossing import."""
    if name in {
        "ArchiveWriteCustody",
        "WriteLease",
        "WriteLeaseDelegation",
        "WriteLeaseThreadGrant",
        "UnleasedWriteError",
    }:
        return getattr(_implementation(), name)
    raise AttributeError(name)


@contextmanager
def authorized_session_removal(
    *, archive_root: Path, plan_hash: str, session_ids: tuple[str, ...], excise_assertions: bool = False
) -> Iterator[None]:
    with _implementation().authorized_session_removal(
        archive_root=archive_root, plan_hash=plan_hash, session_ids=session_ids, excise_assertions=excise_assertions
    ):
        yield
