"""Daemon composition for one exact raw-observation derivation.

Raw preparation is deliberately independent of the daemon writer.  Writer
admissions go through :class:`DaemonWriteThreadBridge`: the adapter's
one-observation publication, forwarded from the bounded compute worker, and
the terminal receipt of a retained Codex state export admitted without one.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path

from polylogue.daemon.convergence import DaemonConverger, DerivationConvergenceOwner
from polylogue.daemon.derivation import Budget, DerivationReport
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
from polylogue.operations.raw_observation_derivation import (
    RAW_OBSERVATION_DOMAIN,
    finalize_codex_state_raw,
    make_raw_observation_derivation,
    raw_needs_codex_state_receipt,
    raw_observation_frame,
)


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
    ) -> None:
        self._archive_root = archive_root
        self._write_bridge = write_bridge
        adapter = make_raw_observation_derivation(archive_root)
        self._converger = DaemonConverger((), derivations=(adapter,))
        self._owner = DerivationConvergenceOwner(
            self._converger, compute_adapter=compute_adapter, write_bridge=write_bridge
        )
        self._converge_lock = asyncio.Lock()

    async def converge_raw_id(self, raw_id: str) -> DerivationReport:
        """Prepare, revalidate, and publish exactly ``raw_id`` if still pending."""
        if not raw_id:
            raise ValueError("raw observation id must be non-empty")
        async with self._converge_lock:
            from polylogue.daemon.write_coordinator import daemon_write_lease_active

            if daemon_write_lease_active():
                raise RuntimeError("raw observation convergence must start after the daemon writer lease is released")
            await self._finalize_retained_codex_state(raw_id)
            self._require_source_frontier_authority(raw_id)
            frame = raw_observation_frame(
                self._archive_root,
                raw_ids=(raw_id,),
            )
            return await self._owner.converge(
                frame,
                budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
                domains=(RAW_OBSERVATION_DOMAIN,),
                resume=False,
            )

    async def _finalize_retained_codex_state(self, raw_id: str) -> None:
        """Write the terminal receipt of a retained Codex state export first.

        Such a raw admitted without its receipt is an incomparable cursor row
        to the source-selection gate, and every route that could finalize it
        sits behind that gate. The receipt comes from the immutable retained
        export, published through the daemon writer.
        """
        if not raw_needs_codex_state_receipt(self._archive_root, raw_id):
            return
        await asyncio.to_thread(
            self._write_bridge.run_sync_with_timeout,
            "raw_observation.codex_state_receipt",
            None,
            functools.partial(finalize_codex_state_raw, self._archive_root, raw_id),
        )

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
        source_path = make_raw_observation_derivation(self._archive_root).source_paths((raw_id,)).get(raw_id)
        if source_path in refusal.source_paths:
            raise RuntimeError(
                f"raw observation source-selection gate blocked: raw {raw_id} is on refused source path {source_path}"
            )
