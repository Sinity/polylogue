"""Acquire and materialize hook carriers through the production route.

Hook capture is two production steps now -- the ordinary file route acquires a
carrier's bytes, and the ``hook_events`` derivation materializes the events out
of them -- so a test that only wants "the hook events are in the archive" would
otherwise have to stand up the dispatcher itself. This is that wiring, once,
built from the real adapters rather than from a shortcut that would let a test
pass against a route production does not use.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Coroutine, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.sources.live.watcher import WatchSource

__all__ = ["acquire_hook_carriers", "hook_event_count", "materialize_hook_carriers"]

#: A pass that admits nothing is the end of the backlog. The cap only bounds a
#: test that has wired something into a loop; a real backlog drains in one pass
#: per carrier page.
_MAX_PASSES = 50


@contextmanager
def _pinned_archive_root(archive_root: Path) -> Iterator[None]:
    """Resolve the declared hook topology against *archive_root* inside the block.

    ``carrier_identity`` refuses a carrier that sits under no declared hook
    root, and the declared primary root is ``hooks_sidecar_dir()`` =
    ``archive_root() / "hooks"``. The session pins ``POLYLOGUE_ARCHIVE_ROOT``
    at a scratch archive so no test can reach the operator's, so without this
    a test's own ``tmp_path`` carriers acquire and then never materialize.
    """

    previous = {name: os.environ.get(name) for name in ("POLYLOGUE_ARCHIVE_ROOT", "POLYLOGUE_CONFIG")}
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["POLYLOGUE_CONFIG"] = str(archive_root / "polylogue.toml")
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True, slots=True)
class _ArchiveRootOwner:
    archive_root: Path


def _carrier_sources(spool_root: Path) -> tuple[WatchSource, ...]:
    from polylogue.sources.hooks import hook_carrier_provider_dir
    from polylogue.sources.live.watcher import HOOK_CARRIER_PROVIDERS, WatchSource

    return tuple(
        WatchSource(
            name=f"{provider}-hooks",
            root=hook_carrier_provider_dir(provider, spool_root),
            suffixes=(".ndjson",),
            source_id=f"primary-hook-spool:{provider}",
            role="primary-writable",
        )
        for provider in HOOK_CARRIER_PROVIDERS
    )


def _run(coro: Coroutine[None, None, int]) -> int:
    """Run *coro* to completion from sync code, async test or not.

    ``asyncio.run`` refuses to nest, and the facade contract tests that need
    hook rows are themselves ``async def``. Handing the coroutine its own loop
    on a worker thread keeps one entry point for both callers.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _acquire(archive_root: Path, spool_root: Path) -> int:
    from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
    from polylogue.sources.live.watcher import LiveWatcher

    sources = _carrier_sources(spool_root)
    watcher = LiveWatcher(_ArchiveRootOwner(archive_root), sources, intake_hints_only=True)
    context = DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=sources)
    dispatcher = FairIntakeDispatcher(
        tuple(
            IntakeClassSpec(
                name=f"hook_carrier:{source.name}",
                adapter=FileIntakeAdapter(context, source, class_name=f"hook_carrier:{source.name}"),
            )
            for source in sources
        )
    )
    admitted = 0
    for _pass in range(_MAX_PASSES):
        report = await dispatcher.run_once()
        moved = sum(int(entry.admitted) for entry in report.classes)
        admitted += moved
        if not moved:
            break
    return admitted


def acquire_hook_carriers(archive_root: Path, *, spool_root: Path | None = None) -> int:
    """Admit every carrier under the spool root; return the items admitted."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    with _pinned_archive_root(archive_root):
        initialize_active_archive_root(archive_root)
        return _run(_acquire(archive_root, spool_root or archive_root / "hooks"))


def hook_event_count(archive_root: Path) -> int:
    """Rows in ``raw_hook_events``, or 0 before the source tier exists."""

    source_db = archive_root / "source.db"
    if not source_db.exists():
        return 0
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='raw_hook_events'").fetchone() is None:
            return 0
        return int(conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone()[0])


def materialize_hook_carriers(archive_root: Path, *, spool_root: Path | None = None) -> int:
    """Acquire every pending carrier, materialize its events, and count them."""

    acquire_hook_carriers(archive_root, spool_root=spool_root)
    with _pinned_archive_root(archive_root):
        return _converge(archive_root)


def _converge(archive_root: Path) -> int:
    from polylogue.operations.hook_event_derivation import (
        converge_hook_carriers,
        discover_pending_hook_carriers,
    )

    seen: set[tuple[str, ...]] = set()
    for _round in range(_MAX_PASSES):
        pending = discover_pending_hook_carriers(archive_root, 128)
        if not pending:
            break
        selected = tuple(key for key, _cost in pending)
        if selected in seen:
            # The same page twice with no progress is a permanent refusal, not
            # a backlog. Surfacing it beats spinning until the cap.
            raise AssertionError(f"hook carriers will not materialize: {selected}")
        seen.add(selected)
        converge_hook_carriers(archive_root, raw_ids=selected, limit=len(selected))
    return hook_event_count(archive_root)
