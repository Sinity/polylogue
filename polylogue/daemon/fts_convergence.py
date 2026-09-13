"""Daemon-facing owner for the storage-owned message FTS derivation.

This module deliberately has no scheduling loop, repair route, readiness
ledger, or direct SQLite work. Composition creates one shared
``DerivationConvergenceOwner`` with the daemon's bounded compute capacity and
writer bridge; this facade only selects the FTS domain and makes a
path/generation/recipe-bound frame for each request.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from polylogue.daemon.convergence import DerivationConvergenceOwner
from polylogue.daemon.derivation import Budget, DerivationFrame, DerivationReport

__all__ = ["FtsConvergenceOwner"]


class FtsConvergenceOwner:
    """Run FTS through the common owner without making another scheduler."""

    def __init__(
        self,
        owner: DerivationConvergenceOwner,
        frame_for_scope: Callable[[Sequence[str] | None], DerivationFrame],
    ) -> None:
        self._owner = owner
        self._frame_for_scope = frame_for_scope

    async def converge(
        self,
        scope: Sequence[str] | None = None,
        *,
        budget: Budget | int | None = None,
        deadline_s: float | None = None,
        resume: bool = True,
    ) -> DerivationReport:
        """Converge one bounded FTS pass and return its typed operation result.

        Readiness remains a separate authoritative inspection of FTS output;
        this report says what the operation attempted and never certifies a
        whole archive from a process-local cursor.
        """
        return await self._owner.converge(
            self._frame_for_scope(scope),
            budget=budget,
            deadline_s=deadline_s,
            resume=resume,
            domains=("messages_fts",),
        )
