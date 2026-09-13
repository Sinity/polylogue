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
        self._convergers_by_payload_limit: dict[int, DaemonConverger] = {max_payload_bytes: self._converger}
        self._converge_lock = asyncio.Lock()

    async def converge_raw_id(self, raw_id: str, *, max_payload_bytes: int | None = None) -> DerivationReport:
        """Prepare, revalidate, and publish exactly ``raw_id`` if still pending."""
        if not raw_id:
            raise ValueError("raw observation id must be non-empty")
        payload_limit = self._max_payload_bytes if max_payload_bytes is None else max_payload_bytes
        if payload_limit < 1:
            raise ValueError("raw observation payload limit must be positive")
        async with self._converge_lock:
            from polylogue.daemon.write_coordinator import daemon_write_lease_active

            if daemon_write_lease_active():
                raise RuntimeError("raw observation convergence must start after the daemon writer lease is released")
            self._require_source_frontier_authority(raw_id)
            frame = raw_observation_frame(
                self._archive_root,
                raw_ids=(raw_id,),
            )
            converger = self._convergers_by_payload_limit.get(payload_limit)
            if converger is None:
                converger = DaemonConverger(
                    (),
                    derivations=(
                        make_raw_observation_derivation(
                            self._archive_root,
                            max_payload_bytes=payload_limit,
                            stream_safe_only=payload_limit > self._max_payload_bytes,
                        ),
                    ),
                )
                self._convergers_by_payload_limit[payload_limit] = converger
            loop = asyncio.get_running_loop()
            admission = _RawPublicationAdmission(self._write_bridge, loop_thread_id=threading.get_ident())
            submitted = self._compute_adapter.submit(
                partial(
                    converger.converge_derivations,
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

    def _require_source_frontier_authority(self, raw_id: str) -> None:
        """Refuse exactly the raw paths the durable frontier cannot authorize.

        Fair intake owns retry/isolation of this typed refusal.  The check is
        deliberately at the canonical owner boundary so periodic and whale
        callers cannot select a raw observation through a legacy scanner and
        then publish it without the source-frontier proof.
        """
        from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation
        from polylogue.readiness.capability import raw_frontier_source_selection_refusal

        refusal = raw_frontier_source_selection_refusal(self._archive_root, raw_ids=(raw_id,))
        if refusal.unattributed_reason is not None:
            raise RuntimeError(f"raw observation source-selection gate blocked: {refusal.unattributed_reason}")
        if not refusal.source_paths:
            return
        source_path = (
            make_raw_observation_derivation(
                self._archive_root,
                max_payload_bytes=self._max_payload_bytes,
            )
            .source_paths((raw_id,))
            .get(raw_id)
        )
        if source_path in refusal.source_paths:
            raise RuntimeError(
                f"raw observation source-selection gate blocked: raw {raw_id} is on refused source path {source_path}"
            )
