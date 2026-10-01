"""Compose the session derivation with the serving runtime's existing owners."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.daemon.convergence import (
    DaemonConverger,
    SessionProfileConvergenceOwner,
)
from polylogue.daemon.derivation import Budget, DerivationReport, DomainCursor, Outcome, PassCursor, WorkCounters
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
    #: A completed unsettled sweep yields scheduling without clearing its debt.
    audit_can_continue: Callable[[], bool] = field(default=lambda: True)

    async def __call__(self, scope: Sequence[str] | None) -> DerivationReport:
        return await self.callback(scope)

    async def converge_backlog(self, budget_s: float) -> DerivationReport:
        """Run bounded passes back to back until the audit sweep finishes.

        Failed and unchanged passes leave the audit owed for its next tick.
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
        positions: dict[str, DomainCursor] = {}
        frame_binding: tuple[str, str, dict[str, str]] | None = None
        while deadline > time.monotonic():
            passed = await self.audit_pass(deadline)
            if passed is None:
                break
            report = passed
            # A failed acquisition cannot improve by immediately acquiring the
            # same relation again. The audit keeps it owed for a later tick.
            if passed.failed or not self.audit_can_continue():
                break
            binding = (passed.frame.archive_root, passed.frame.source_revision, dict(passed.frame.recipe_versions))
            if binding != frame_binding:
                positions.clear()
                frame_binding = binding
            advanced = any(
                positions.get(domain) != cursor or (cursor.swept and not passed.pending)
                for domain, cursor in passed.cursor.positions.items()
            )
            for domain, cursor in passed.cursor.positions.items():
                if cursor.swept and not passed.pending:
                    # This obligation was discharged. A predecessor completing
                    # later can owe the domain again at the same terminal
                    # cursor; that is new work, not an unchanged pending retry.
                    positions.pop(domain, None)
                else:
                    positions[domain] = cursor
            if not advanced:
                # Counts can change while the same pending cursor is retried.
                # Walking a new cursor or discharging an owed domain keeps
                # this invocation productive; no key-count cap is involved.
                break
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
    from polylogue.operations.sinex_convergence import configured_derivation_barrier

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
    # Marker delivery is an independent durable cursor. Keep it in the bounded
    # startup audit so a long profile scan cannot starve accepted markers.
    audit_domains = (summary.domain, usage_rollup.domain, profile.domain, markers.domain)
    # A swept domain that retained quiet or blocked keys stays owed, but the
    # audit rotates past it so it cannot starve the independent domains behind
    # it. When an owed domain later completes, the domains after it read its
    # new output and are owed again.
    owed_domains = set(audit_domains)
    retried_domains: set[str] = set()
    # True means the current sweep encountered unsettled work; False keeps
    # the obligation while its next, wrapped sweep checks the whole domain.
    unsettled_sweeps: dict[str, bool] = {}
    audit_binding: tuple[str, str, dict[str, str]] | None = None
    audit_continue = True
    audit_budget = Budget(discovery=128, inspection=128, compute=64, publication=64, retained_outcomes=64)
    audit_lock = asyncio.Lock()
    audit_index = 0
    audit_reset = True
    demand_reset = False

    async def audit_tick(deadline_at: float | None = None) -> DerivationReport:
        nonlocal audit_index, audit_reset, demand_reset, audit_binding, audit_continue
        domain = audit_domains[audit_index]
        frame = make_session_profile_frame(index_path, archive_root=archive_root, scope=None, profile_full_scan=True)
        binding = (frame.archive_root, frame.source_revision, dict(frame.recipe_versions))
        if audit_binding is not None and binding != audit_binding:
            owed_domains.update(audit_domains)
            retried_domains.clear()
            unsettled_sweeps.clear()
            audit_index = 0
            domain = audit_domains[0]
            audit_reset = True
        audit_binding = binding
        report = await owner.converge(
            frame,
            budget=audit_budget if deadline_at is None else replace(audit_budget, deadline_at=deadline_at),
            domains=(domain,),
            resume=not audit_reset,
        )
        audit_reset = False
        audit_continue = True
        if domain in report.cursor_unsettled_domains:
            unsettled_sweeps[domain] = True
        if report.cursor.position(domain).swept or report.failed:
            if report.pending or report.failed or unsettled_sweeps.get(domain, False):
                retried_domains.add(domain)
            else:
                owed_domains.discard(domain)
                unsettled_sweeps.pop(domain, None)
                if domain in retried_domains:
                    retried_domains.discard(domain)
                    owed_domains.update(audit_domains[audit_index + 1 :])
            if owed_domains:
                audit_index = next(
                    index
                    for step in range(1, len(audit_domains) + 1)
                    if audit_domains[index := (audit_index + step) % len(audit_domains)] in owed_domains
                )
            else:
                audit_index = len(audit_domains)
                demand_reset = True
        # A clean tail does not certify an unsettled prefix. Keep the domain
        # owed until a whole wrapped sweep is clean.
        if report.cursor.position(domain).swept and domain in unsettled_sweeps:
            unsettled_sweeps[domain] = False
            audit_continue = False
        # The converger retains every domain's cursor. This pass reports only
        # the domain it ran, so old sibling cursors cannot imply new progress.
        return replace(report, cursor=PassCursor({domain: report.cursor.position(domain)}))

    async def converge(scope: Sequence[str] | None) -> DerivationReport:
        """Converge demanded session work; never the archive-wide audit.

        Demand runs first on every tick (polylogue-6remh): a periodic tick
        that swept the startup audit before demand starved demanded profiles
        for as long as the audit took, and an idle tick must inspect nothing.
        The audit is the startup/promotion sweep, advanced only in bounded
        slices by :meth:`ComposedSessionProfiles.converge_backlog` after this
        demand pass, and it does not repeat once swept.
        """
        nonlocal demand_reset
        if scope is not None:
            frame = make_session_profile_frame(
                index_path, archive_root=archive_root, scope=scope, profile_demand_only=True
            )
            return await owner.converge(frame)
        async with audit_lock:
            frame = make_session_profile_frame(
                index_path, archive_root=archive_root, scope=None, profile_demand_only=True
            )
            report = await owner.converge(frame, resume=not demand_reset)
            demand_reset = False
            return report

    async def audit_pass(deadline_at: float) -> DerivationReport | None:
        async with audit_lock:
            # The pass carries the absolute instant, so time spent waiting for
            # the owner's lock or a compute worker is spent from the tick's
            # budget rather than added to it.
            if audit_index >= len(audit_domains) or deadline_at <= time.monotonic():
                return None
            return await audit_tick(deadline_at)

    async def converge_promoted() -> DerivationReport:
        nonlocal audit_index, audit_reset, demand_reset, audit_binding
        async with audit_lock:
            owed_domains.clear()
            owed_domains.update(audit_domains)
            retried_domains.clear()
            unsettled_sweeps.clear()
            audit_binding = None
            audit_index = 0
            audit_reset = True
            demand_reset = False
            report = await audit_tick()
            # A small promoted generation can finish all prerequisites now;
            # a large one retains its cursor for later periodic ticks.
            for _ in range(len(audit_domains) - 1):
                if audit_index == len(audit_domains):
                    break
                report = _merge_reports(report, await audit_tick())
            return report

    return ComposedSessionProfiles(
        converge,
        converge_promoted,
        make_session_insight_maintenance(owner, index_db_path=index_path, archive_root=archive_root),
        audit_pending=lambda: audit_index < len(audit_domains),
        audit_pass=audit_pass,
        audit_can_continue=lambda: audit_continue,
    )


def _merge_reports(first: DerivationReport, second: DerivationReport) -> DerivationReport:
    """Retain bounded outcomes from every domain pass of one promotion."""
    counts = {
        outcome: first.count(outcome) + second.count(outcome)
        for outcome in Outcome
        if first.count(outcome) + second.count(outcome)
    }
    first_work, second_work = first.work, second.work
    work = WorkCounters(
        pages=first_work.pages + second_work.pages,
        discovered=first_work.discovered + second_work.discovered,
        inspected=first_work.inspected + second_work.inspected,
        prerequisites_inspected=first_work.prerequisites_inspected + second_work.prerequisites_inspected,
        computed=first_work.computed + second_work.computed,
        published=first_work.published + second_work.published,
    )
    return replace(
        second,
        outcomes=first.outcomes + second.outcomes,
        counts=counts,
        work=work,
        truncated=first.truncated or second.truncated,
        cursor_unsettled_domains=first.cursor_unsettled_domains | second.cursor_unsettled_domains,
    )
