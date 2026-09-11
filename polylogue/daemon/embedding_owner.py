"""The daemon's lease-free owner for archive embedding computation.

Embedding is the one derivation whose cost is a network round trip the daemon
does not control. Running it from inside the writer gate meant a slow or hung
provider blocked every unrelated archive publication for as long as the call
took, so this module runs the pass the other way round: the work is admitted to
the process's bounded compute capacity, where it holds neither the writer gate
nor the embedding generation lock, and each short write it needs -- attempt
reservation, one published window, the catch-up receipt -- is admitted back
through the daemon's write coordinator for exactly that write.

Two rules make that safe and are enforced here rather than documented:

* the compute worker is never the event loop thread, because
  :meth:`DaemonWriteThreadBridge.run_sync` blocks on the loop it is scheduling
  onto, and
* no second compute pool is created; the adapter is the one published for the
  process (:func:`polylogue.daemon.execution.daemon_compute_adapter`).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import TypeVar

from polylogue.daemon.execution import daemon_compute_adapter
from polylogue.daemon.write_coordinator import (
    DaemonWriteThreadBridge,
    daemon_write_coordinator,
    daemon_write_lease_active,
)
from polylogue.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

__all__ = [
    "DaemonEmbeddingAdmission",
    "converge_archive_embeddings",
    "run_lease_free_embedding_work",
]


class DaemonEmbeddingAdmission:
    """Admit one short embedding write through the daemon's write coordinator.

    Calling this from the event loop thread would deadlock: the bridge blocks
    the calling thread until the coordinator has run the operation on that same
    loop. The guard below turns that into a typed refusal instead of a hang.
    """

    def __init__(self, bridge: DaemonWriteThreadBridge, loop: asyncio.AbstractEventLoop) -> None:
        self._bridge = bridge
        self._loop_thread_id = threading.get_ident() if loop.is_running() else None

    def __call__(self, actor: str, function: Callable[[], T], /) -> T:
        if self._loop_thread_id is not None and threading.get_ident() == self._loop_thread_id:
            raise RuntimeError(
                f"embedding phase {actor} was admitted from the daemon event loop thread; "
                "lease-free embedding work must run on a compute worker"
            )
        # A ``None`` wait keeps daemon ownership until the phase returns its
        # receipt. Abandoning a reserved attempt or a completed publication on a
        # caller-side timeout would strand exactly the state this route exists
        # to keep consistent.
        return self._bridge.run_sync_with_timeout(actor, None, function)


async def run_lease_free_embedding_work(
    function: Callable[..., T],
    /,
    *args: object,
    **kwargs: object,
) -> T:
    """Run one embedding pass on the shared compute capacity, off the writer.

    ``function`` is called with an ``admit`` keyword bound to this daemon's
    coordinator, so every write it performs is a separate short admitted
    operation while the provider call between them holds nothing.
    """
    if daemon_write_lease_active():
        # The admitted phases below queue for the writer gate this caller is
        # already holding, so running here would deadlock rather than merely
        # serialize. Callers release the gate first; this makes the mistake a
        # typed refusal instead of a hang.
        raise RuntimeError("lease-free embedding work was started while this context holds the daemon writer gate")
    loop = asyncio.get_running_loop()
    bridge = DaemonWriteThreadBridge(daemon_write_coordinator(), loop)
    admission = DaemonEmbeddingAdmission(bridge, loop)
    submitted = daemon_compute_adapter().submit(
        partial(function, *args, admit=admission, **kwargs),
        admission_class="incremental-background",
    )
    return await asyncio.wrap_future(submitted.future, loop=loop)  # type: ignore[arg-type]


async def converge_archive_embeddings(
    db_path: Path,
    *,
    paths: Sequence[Path] = (),
    session_ids: Sequence[str] = (),
) -> bool:
    """Converge embeddings for these subjects without holding writer authority."""
    if not paths and not session_ids:
        return True
    from polylogue.daemon.convergence_stages import run_archive_embedding_convergence

    result = await run_lease_free_embedding_work(
        run_archive_embedding_convergence,
        db_path,
        paths=tuple(paths),
        session_ids=tuple(session_ids),
    )
    return bool(result)
