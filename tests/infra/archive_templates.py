"""Archive templates a law builds in place, sealed and cloned by the shared route.

A template is an :class:`~tests.infra.workload_artifacts.ImmutableTreeArtifact`
whose builder ran outside the artifact cache: the consuming law owns the tree's
location and lifetime, and everything after construction -- tier validation,
sealing, authenticated detached cloning, destination-owned train admission -- is the
one publication route in :mod:`tests.infra.workload_artifacts`.
"""

from __future__ import annotations

from builtins import BaseExceptionGroup
from collections.abc import Callable
from hashlib import sha256
from math import inf
from pathlib import Path
from typing import ParamSpec, TypeVar

from tests.infra.workload_artifacts import (
    ImmutableTreeArtifact,
    clone_immutable_tree,
    seal_fixture_tree,
)

_T = TypeVar("_T")


def finalize_archive_template(root: Path) -> None:
    """Publish a reusable archive template only after a verified SQLite snapshot."""
    seal_fixture_tree(root)


def _template_key(template: Path) -> str:
    """Bind a clone to the exact tree it came from.

    A law-built template has no content-addressed cache identity; its location
    is what distinguishes it, and the resulting clone's ``source_manifest_id``
    must not read as a claim about a cached artifact.
    """
    return "archive-template:" + sha256(str(template.resolve()).encode()).hexdigest()


def clone_archive_template(template: Path, destination: Path) -> str:
    """Clone into the actual reserved destination through the shared owner."""
    artifact = ImmutableTreeArtifact.adopt(template, key=_template_key(template))
    return clone_immutable_tree(artifact, destination).clone_method


def bootstrap_archive_root(root: Path) -> Path:
    """Construct each empty fixture through the canonical baseline and train owner."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    run_off_event_loop(lambda: initialize_active_archive_root(root))
    return root


def bootstrapped_tier_path(path: Path) -> Path:
    """Bootstrap the archive a law's named tier file belongs to; return its tier file.

    ``<root>/index.db`` and ``<root>/source.db`` are tiers of ``<root>``. Any
    other name (``inline.db``, ``codex.db``) is its own archive, because a root
    owns exactly one active Index: it becomes ``<parent>/<stem>/index.db``.
    """
    path = Path(path)
    if path.name in ("index.db", "source.db"):
        bootstrap_archive_root(path.parent)
        return path
    root = path.parent / path.stem
    bootstrap_archive_root(root)
    return root / "index.db"


async def run_archive_fixture_write(root: Path, prepare: Callable[[], _T]) -> _T:
    """Prepare an async law's archive on the real admitted writer creator."""
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriterSettlementError

    root.mkdir(parents=True, exist_ok=True)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    primary: BaseException | None = None
    try:
        return await coordinator.run_sync("fixture.archive.prepare", prepare)
    except BaseException as failure:
        primary = failure
        raise
    finally:
        # Completion depends on the physical owner, without a work-duration cap.
        if not await coordinator.shutdown(timeout=inf):
            cleanup = DaemonWriterSettlementError("archive fixture writer remains unsettled")
            if primary is not None:
                raise BaseExceptionGroup("archive fixture preparation and settlement failed", [primary, cleanup])
            raise cleanup


async def run_archive_fixture_prepare(prepare: Callable[[], _T]) -> _T:
    """Run canonical preparation and its publication on one lease-free creator.

    The callback owns all SQL-backed preparation and closes its carriers before
    returning detached values. Its canonical publisher acquires writer custody
    after preparation; this outer owner supplies compute admission only.
    """
    import asyncio

    from polylogue.core.compute import BoundedComputeAdapter, SubmittedOperation

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=1)
    submitted: SubmittedOperation[_T] | None = None
    completion: asyncio.Future[_T] | None = None
    primary: BaseException | None = None
    try:
        submitted = adapter.submit(prepare)
        completion = asyncio.wrap_future(submitted.future)
        return await asyncio.shield(completion)
    except BaseException as failure:
        primary = failure
        if isinstance(failure, asyncio.CancelledError) and submitted is not None:
            submitted.cancellation.cancel()
        raise
    finally:
        # A cancelled await cannot retire the creator while its body or retained
        # SQL custody is still live. Shutdown retries on that original creator.
        cleanup = asyncio.create_task(asyncio.to_thread(adapter.shutdown, wait=True))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                if submitted is not None:
                    submitted.cancellation.cancel()
                cancelled = True
            except BaseException:
                break  # Retrieve and classify the completed cleanup below.
        try:
            cleanup.result()
        except BaseException as failure:
            if primary is not None:
                raise BaseExceptionGroup("fixture preparation and settlement failed", [primary, failure]) from failure
            raise
        if completion is not None and completion.done() and not completion.cancelled():
            completion.exception()
        if cancelled and primary is None:
            raise asyncio.CancelledError


def bootstrap_ready_archive_root(root: Path) -> Path:
    """Run the complete synchronous fixture construction on its original owner."""
    import asyncio

    return asyncio.run(bootstrap_ready_archive_root_async(root))


__all__ = [
    "bootstrap_archive_root",
    "bootstrap_ready_archive_root",
    "bootstrap_ready_archive_root_async",
    "clone_archive_template",
    "finalize_archive_template",
    "run_archive_fixture_prepare",
    "run_archive_fixture_write",
]


async def bootstrap_ready_archive_root_async(root: Path) -> Path:
    """Construct and inspect through one genuine supplied fixture owner."""
    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.live_ingest import prepared_live_convergence_owner

    root.mkdir(parents=True, exist_ok=True)
    async with prepared_live_convergence_owner(root) as owner:
        await owner._write_coordinator.run_sync("fixture.archive.bootstrap", lambda: bootstrap_archive_root(root))
        await owner.run_convergence_sync(
            "fixture.archive.frontier",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
    return root


def run_off_event_loop(operation: Callable[[], _T]) -> _T:
    """Run synchronous archive setup where a synchronous write lease may block.

    A synchronous write lease refuses to block a running event loop. Fixture
    setup called from an async test therefore runs on a dedicated thread with
    no loop and is joined before the test continues; a caller without a loop
    runs it directly.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return operation()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fixture-off-loop") as executor:
        return executor.submit(operation).result()


_P = ParamSpec("_P")


def seeds_off_event_loop(seed: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorate a synchronous seeding helper so async tests may call it.

    The helper runs through :func:`run_off_event_loop`: directly without a
    running loop, otherwise on a joined loop-free thread.
    """
    from functools import wraps

    @wraps(seed)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return run_off_event_loop(lambda: seed(*args, **kwargs))

    return run
