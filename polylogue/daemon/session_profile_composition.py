"""Compose the session derivation with the serving runtime's existing owners."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from polylogue.daemon.convergence import (
    DaemonConverger,
    SessionProfileConvergenceOwner,
)
from polylogue.daemon.derivation import DerivationReport
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.session_insight_maintenance import SessionInsightMaintenance, make_session_insight_maintenance
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
from polylogue.operations.session_profile_convergence import (
    make_session_profile_derivation,
    make_session_profile_frame,
)

SessionProfileCallback = Callable[[Sequence[str] | None], Awaitable[DerivationReport]]


@dataclass(frozen=True, slots=True)
class ComposedSessionProfiles:
    """Automatic convergence and exact request work share one owner and lock."""

    callback: SessionProfileCallback
    maintenance: SessionInsightMaintenance

    async def __call__(self, scope: Sequence[str] | None) -> DerivationReport:
        return await self.callback(scope)


def compose_session_profile_callback(
    archive_root: Path,
    *,
    compute_adapter: BoundedComputeAdapter,
    write_bridge: DaemonWriteThreadBridge,
    now: Callable[[], float],
) -> ComposedSessionProfiles:
    """Construct once; each invocation binds the current normalized generation."""
    # The owning factories resolve this logical anchor through ArchiveLocation
    # on every generation observation; composition never opens a pointer stub.
    index_path = archive_root / "index.db"
    adapter = make_session_profile_derivation(
        index_path,
        archive_root=archive_root,
        now=now,
    )
    owner = SessionProfileConvergenceOwner(
        DaemonConverger(stages=(), derivations=(adapter,)),
        compute_adapter=compute_adapter,
        write_bridge=write_bridge,
    )

    async def converge(scope: Sequence[str] | None) -> DerivationReport:
        frame = make_session_profile_frame(index_path, archive_root=archive_root, scope=scope)
        return await owner.converge(frame)

    return ComposedSessionProfiles(
        converge,
        make_session_insight_maintenance(owner, index_db_path=index_path, archive_root=archive_root),
    )
