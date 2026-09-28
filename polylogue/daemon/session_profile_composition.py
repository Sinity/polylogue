"""Compose the session derivation with the serving runtime's existing owners."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from polylogue.daemon.convergence import (
    DaemonConverger,
    SessionProfileConvergenceOwner,
)
from polylogue.daemon.derivation import Budget, DerivationReport
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.session_insight_maintenance import SessionInsightMaintenance, make_session_insight_maintenance
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
from polylogue.operations.session_profile_convergence import (
    make_session_marker_derivation,
    make_session_profile_derivation,
    make_session_profile_frame,
    make_session_summary_derivation,
    make_session_usage_rollup_derivation,
)

SessionProfileCallback = Callable[[Sequence[str] | None], Awaitable[DerivationReport]]


@dataclass(frozen=True, slots=True)
class ComposedSessionProfiles:
    """Automatic convergence and exact request work share one owner and lock."""

    callback: SessionProfileCallback
    promoted_callback: Callable[[], Awaitable[DerivationReport]]
    maintenance: SessionInsightMaintenance
    #: Whether the archive-wide audit sweep still has domains to finish.
    audit_pending: Callable[[], bool] = field(default=lambda: False)
    #: One bounded audit pass that must finish by the given ``time.monotonic``
    #: instant, or ``None`` when the audit has nothing left to sweep or the
    #: instant passed while it waited for the owner.
    audit_pass: Callable[[float], Awaitable[DerivationReport | None]] | None = None

    async def __call__(self, scope: Sequence[str] | None) -> DerivationReport:
        return await self.callback(scope)

    async def converge_backlog(self, budget_s: float) -> DerivationReport:
        """Run bounded passes back to back until the audit sweep finishes.

        Each pass keeps its page and publication bounds, so no writer hold
        grows; what changes is that a promoted generation's sweep no longer
        advances one bounded pass per periodic tick. At 64 keys per pass and
        three domains that was one minute of wall time per 64 sessions per
        domain, whatever the writer and compute had free.
        """
        deadline = time.monotonic() + budget_s
        report = await self.callback(None)
        if self.audit_pass is None:
            return report
        # Every further pass carries the remaining time as its derivation
        # deadline, so the kernel stops inside the pass rather than a pass
        # started near the end overrunning the tick's budget.
        while deadline > time.monotonic():
            passed = await self.audit_pass(deadline)
            if passed is None:
                break
            report = passed
        return report

    async def converge_promoted(self) -> DerivationReport:
        return await self.promoted_callback()


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
    summary = make_session_summary_derivation(index_path, archive_root=archive_root)
    # Ordered before the profile: the profile reads the canonical usage rollup,
    # so reconciling it inside profile publication would move an input the
    # prepared partition had already read (polylogue-bp12n.1).
    usage_rollup = make_session_usage_rollup_derivation(index_path, archive_root=archive_root, now=now)
    profile = make_session_profile_derivation(
        index_path,
        archive_root=archive_root,
        now=now,
    )
    # Ordered after the profile: marker lowering reads the same session
    # evidence, and the edge is one-way -- a user-tier marker failure leaves
    # this domain's work pending without invalidating the index family
    # (polylogue-ylh7v).
    markers = make_session_marker_derivation(index_path, archive_root=archive_root)
    from polylogue.daemon.convergence_stages import configured_derivation_barrier

    owner = SessionProfileConvergenceOwner(
        DaemonConverger(
            stages=(),
            derivations=(summary, usage_rollup, profile, markers),
            derivation_barrier=configured_derivation_barrier(archive_root),
        ),
        compute_adapter=compute_adapter,
        write_bridge=write_bridge,
    )
    # A completed domain restarts on its next owner pass. Keep one domain
    # active until its bounded cursor is swept so earlier sessions cannot
    # consume the budget before a later domain reaches the same archive tail.
    audit_domains = (summary.domain, usage_rollup.domain, profile.domain)
    audit_budget = Budget(discovery=128, inspection=128, compute=64, publication=64, retained_outcomes=64)
    audit_lock = asyncio.Lock()
    audit_index = 0
    audit_reset = True
    demand_reset = False

    async def audit_tick(deadline_s: float | None = None) -> DerivationReport:
        nonlocal audit_index, audit_reset, demand_reset
        domain = audit_domains[audit_index]
        frame = make_session_profile_frame(index_path, archive_root=archive_root, scope=None, profile_full_scan=True)
        report = await owner.converge(
            frame,
            budget=audit_budget if deadline_s is None else replace(audit_budget, deadline_s=deadline_s),
            domains=(domain,),
            resume=not audit_reset,
        )
        audit_reset = False
        if report.cursor.position(domain).swept:
            audit_index += 1
            if audit_index == len(audit_domains):
                demand_reset = True
        return report

    async def converge(scope: Sequence[str] | None) -> DerivationReport:
        nonlocal demand_reset
        if scope is not None:
            frame = make_session_profile_frame(
                index_path, archive_root=archive_root, scope=scope, profile_demand_only=True
            )
            return await owner.converge(frame)
        async with audit_lock:
            if audit_index < len(audit_domains):
                return await audit_tick()
            frame = make_session_profile_frame(
                index_path, archive_root=archive_root, scope=None, profile_demand_only=True
            )
            report = await owner.converge(frame, resume=not demand_reset)
            demand_reset = False
            return report

    async def audit_pass(deadline_at: float) -> DerivationReport | None:
        async with audit_lock:
            # The remaining time is taken after the owner lock is held, so a
            # wait for it is spent from the tick's budget, not added to it.
            remaining = deadline_at - time.monotonic()
            if audit_index >= len(audit_domains) or remaining <= 0:
                return None
            return await audit_tick(remaining)

    async def converge_promoted() -> DerivationReport:
        nonlocal audit_index, audit_reset, demand_reset
        async with audit_lock:
            audit_index = 0
            audit_reset = True
            demand_reset = False
            report = await audit_tick()
            # A small promoted generation can finish all prerequisites now;
            # a large one retains its cursor for later periodic ticks.
            for _ in range(len(audit_domains) - 1):
                if audit_index == len(audit_domains):
                    break
                report = await audit_tick()
            return report

    return ComposedSessionProfiles(
        converge,
        converge_promoted,
        make_session_insight_maintenance(owner, index_db_path=index_path, archive_root=archive_root),
        audit_pending=lambda: audit_index < len(audit_domains),
        audit_pass=audit_pass,
    )
