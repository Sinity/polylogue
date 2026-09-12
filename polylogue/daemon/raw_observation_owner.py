"""Daemon composition for one exact raw-observation derivation.

Raw preparation is deliberately independent of the daemon writer.  The only
writer admission is the adapter's one-observation publication, forwarded from
the bounded compute worker through :class:`DaemonWriteThreadBridge`.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.derivation import Budget, DerivationReport
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
from polylogue.operations.raw_observation_derivation import (
    RAW_OBSERVATION_DOMAIN,
    make_raw_observation_derivation,
    raw_observation_frame,
)


class _RawPublicationAdmission:
    """Synchronously return just one publication from a compute worker."""

    def __init__(self, bridge: DaemonWriteThreadBridge, *, loop_thread_id: int) -> None:
        self._bridge = bridge
        self._loop_thread_id = loop_thread_id

    def __call__(self, domain: str, publish: Callable[[], bool]) -> bool:
        if threading.get_ident() == self._loop_thread_id:
            raise RuntimeError("raw observation publication was invoked on the daemon event loop thread")
        return self._bridge.run_sync_with_timeout(f"derivation.{domain}", None, publish)


class RawObservationConvergenceOwner:
    """Converge exact admitted raw IDs through the canonical raw adapter.

    This is intentionally a small owner rather than another raw pipeline.  It
    owns only process-local serialization and borrows the daemon-wide compute
    and write owners supplied by composition.
    """

    def __init__(
        self,
        archive_root: Path,
        *,
        compute_adapter: BoundedComputeAdapter,
        write_bridge: DaemonWriteThreadBridge,
        max_payload_bytes: int,
    ) -> None:
        self._archive_root = archive_root
        self._compute_adapter = compute_adapter
        self._write_bridge = write_bridge
        self._max_payload_bytes = max_payload_bytes
        adapter = make_raw_observation_derivation(archive_root, max_payload_bytes=max_payload_bytes)
        self._converger = DaemonConverger((), derivations=(adapter,))
        self._converge_lock = asyncio.Lock()

    async def converge_raw_id(self, raw_id: str) -> DerivationReport:
        """Prepare, revalidate, and publish exactly ``raw_id`` if still pending."""
        if not raw_id:
            raise ValueError("raw observation id must be non-empty")
        async with self._converge_lock:
            from polylogue.daemon.write_coordinator import daemon_write_lease_active

            if daemon_write_lease_active():
                raise RuntimeError("raw observation convergence must start after the daemon writer lease is released")
            frame = raw_observation_frame(
                self._archive_root,
                raw_ids=(raw_id,),
            )
            loop = asyncio.get_running_loop()
            admission = _RawPublicationAdmission(self._write_bridge, loop_thread_id=threading.get_ident())
            submitted = self._compute_adapter.submit(
                partial(
                    self._converger.converge_derivations,
                    frame,
                    # One candidate needs an initial authoritative inspection
                    # and a post-publication certification inspection.
                    budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
                    domains=(RAW_OBSERVATION_DOMAIN,),
                    resume=False,
                    publisher=admission,
                ),
                admission_class="incremental-background",
            )
            operation = asyncio.wrap_future(submitted.future, loop=loop)
            try:
                return await asyncio.shield(operation)
            except asyncio.CancelledError:
                # Never detach a compute worker which may be waiting for a
                # bridged publication; owner shutdown must be able to drain it.
                with contextlib.suppress(BaseException):
                    await asyncio.shield(operation)
                raise
