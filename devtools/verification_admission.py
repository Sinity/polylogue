"""Admission policy for ordinary affected verification.

The testmon graph is a selection oracle, not permission to run an unbounded
workload.  This module keeps the admission decision pure so callers can
record a refusal before constructing a pytest step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from devtools.worker_memory import CORPUS_MAX_WORKERS

# These are deliberately conservative local-evidence budgets.  The complete
# corpus is an explicit master-boundary operation and is not truncated or
# split to fit these limits.
AFFECTED_MAX_SELECTED_TESTS: Final = 1_000
AFFECTED_MAX_ESTIMATED_SECONDS: Final = 15 * 60

#: How wide an affected run may be admitted. The time/selection budgets above
#: are this module's own policy, but width is not: an affected run occupies the
#: same ``agentctl-pytest.slice`` as the corpus run, so the memory owner
#: (``worker_memory.width_within`` over that slice's MemoryHigh) bounds both.
#: This was a hand-set ``4`` maintained independently of that bound, free to
#: admit the *selective* run wider than the full corpus in the same cgroup.
#: Taking the minimum keeps this module free to be stricter than memory allows
#: while never claiming more than the owner says fits.
AFFECTED_POLICY_MAX_WORKERS: Final = 4
AFFECTED_MAX_WORKERS: Final = min(AFFECTED_POLICY_MAX_WORKERS, CORPUS_MAX_WORKERS)
NEXT_VERIFICATION_BOUNDARY: Final = "devtools verify --all at the explicit master/corpus boundary"

#: How many unrecorded test files the estimator will collect to price exactly
#: before it refuses instead. Counting them costs one collect-only pass over
#: those files (measured 2.2s for 10 on 2026-09-22), which is worth paying for
#: the handful a normal branch adds. Past this the graph is not a partial
#: oracle but an absent one -- a cancelled ``--all`` leaves thousands -- and at
#: the observed ~17 tests per file that many would exceed the selection cap
#: anyway, so the answer is a refusal naming the corpus boundary, not a longer
#: collection.
AFFECTED_MAX_UNRECORDED_FILES: Final = 60


@dataclass(frozen=True, slots=True)
class AffectedAdmission:
    """The typed result of admitting one affected selection."""

    status: str
    selected_count: int | None
    estimated_seconds: float | None
    reason: str
    next_boundary: str = NEXT_VERIFICATION_BOUNDARY
    max_selected_tests: int = AFFECTED_MAX_SELECTED_TESTS
    max_estimated_seconds: int = AFFECTED_MAX_ESTIMATED_SECONDS
    max_workers: int = AFFECTED_MAX_WORKERS
    #: How much of ``selected_count`` is tests the graph has never recorded
    #: and will therefore execute as unknown. Carried separately because the
    #: two halves are measured differently -- one from the graph, one from a
    #: collection -- and a receipt that folds them cannot say which.
    unrecorded_tests: int | None = None

    @property
    def admitted(self) -> bool:
        return self.status == "admitted"

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "selected_count": self.selected_count,
            "estimated_seconds": self.estimated_seconds,
            "reason": self.reason,
            "next_boundary": self.next_boundary,
            "unrecorded_tests": self.unrecorded_tests,
            "budget": {
                "max_selected_tests": self.max_selected_tests,
                "max_estimated_seconds": self.max_estimated_seconds,
                "max_workers": self.max_workers,
                "max_unrecorded_files": AFFECTED_MAX_UNRECORDED_FILES,
            },
        }


def admit_affected_selection(
    *,
    graph_status: str,
    graph_reason: str,
    full_rerun_cause: str | None,
    selected_count: int | None,
    estimated_seconds: float | None,
    unrecorded_tests: int | None = None,
) -> AffectedAdmission:
    """Decide whether an affected plan is bounded enough to execute.

    ``None`` is unknown, never zero.  Unknown or unusable graph state is
    refused rather than widened to a corpus run; ``verify --all`` is the only
    deliberate corpus boundary.  No caller should execute a prefix of a plan
    that this function refuses.

    ``selected_count`` must already include ``unrecorded_tests``: testmon
    deselects only what it has recorded, so a test the graph has never seen is
    unknown and runs, and a count taken from the graph alone describes a
    smaller plan than the one that launches.  The term is passed separately
    only so the receipt can say how much of the plan came from where.
    """

    if graph_status != "usable":
        return AffectedAdmission(
            status="unknown",
            selected_count=selected_count,
            estimated_seconds=estimated_seconds,
            unrecorded_tests=unrecorded_tests,
            reason=f"testmon graph is {graph_status}: {graph_reason}",
        )
    if full_rerun_cause:
        return AffectedAdmission(
            status="refused",
            selected_count=selected_count,
            estimated_seconds=estimated_seconds,
            unrecorded_tests=unrecorded_tests,
            reason=f"affected selection would re-run the corpus: {full_rerun_cause}",
        )
    if selected_count is None:
        return AffectedAdmission(
            status="unknown",
            selected_count=None,
            estimated_seconds=estimated_seconds,
            unrecorded_tests=unrecorded_tests,
            reason="affected selection count is unknown; refusing an unbounded pytest launch",
        )
    if selected_count < 0:
        return AffectedAdmission(
            status="unknown",
            selected_count=selected_count,
            estimated_seconds=estimated_seconds,
            unrecorded_tests=unrecorded_tests,
            reason="affected selection count is invalid; refusing an unbounded pytest launch",
        )
    if selected_count > AFFECTED_MAX_SELECTED_TESTS:
        return AffectedAdmission(
            status="refused",
            selected_count=selected_count,
            estimated_seconds=estimated_seconds,
            unrecorded_tests=unrecorded_tests,
            reason=(
                f"affected selection contains {selected_count} tests, exceeding the "
                f"cap of {AFFECTED_MAX_SELECTED_TESTS}; no partial prefix is allowed"
            ),
        )
    if estimated_seconds is not None and estimated_seconds > AFFECTED_MAX_ESTIMATED_SECONDS:
        return AffectedAdmission(
            status="refused",
            selected_count=selected_count,
            estimated_seconds=estimated_seconds,
            unrecorded_tests=unrecorded_tests,
            reason=(
                f"affected selection estimates {estimated_seconds:.1f}s, exceeding the "
                f"budget of {AFFECTED_MAX_ESTIMATED_SECONDS}s; no partial prefix is allowed"
            ),
        )
    return AffectedAdmission(
        status="admitted",
        selected_count=selected_count,
        estimated_seconds=estimated_seconds,
        unrecorded_tests=unrecorded_tests,
        reason=(
            f"affected selection contains {selected_count} tests within the {AFFECTED_MAX_SELECTED_TESTS}-test cap"
        ),
    )


__all__ = [
    "AFFECTED_MAX_ESTIMATED_SECONDS",
    "AFFECTED_MAX_SELECTED_TESTS",
    "AFFECTED_MAX_UNRECORDED_FILES",
    "AFFECTED_MAX_WORKERS",
    "AFFECTED_POLICY_MAX_WORKERS",
    "AffectedAdmission",
    "NEXT_VERIFICATION_BOUNDARY",
    "admit_affected_selection",
]
