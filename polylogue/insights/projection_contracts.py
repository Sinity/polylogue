"""Declared readiness/cost/degraded contracts for default analysis projections.

Two production smokes (polylogue-duti) exposed one missing contract: a
default analysis projection either fabricates an opaque success (``analyze
--cost-outlook`` returned a syntactically valid ``null`` with no remediation
when a plan had no ``cycle_anchor_day``) or silently blows an interactive
budget (``analyze --facets`` took 17.8s with no signal that it had exceeded
any declared budget). This module is the shared declaration: every default
analysis projection names its cost/detail class, its default interactive
deadline, and any prerequisites it needs to produce a useful result. Callers
(currently the CLI; MCP/HTTP adapters pick this up as they grow projection
tools of their own) turn the declaration plus the actual execution outcome
into a :class:`~polylogue.surfaces.payloads.ProjectionAvailabilityPayload` --
one typed envelope so "empty", "not computed", and "prerequisite missing"
are never conflated into a bare null or zero.

This intentionally does not implement bounded/resumable execution for
expensive families -- that QoS mechanism belongs to the shared query
transaction (polylogue-z9gh.9). Here, "expensive" projections are gated
opt-in (``--include-deferred``) rather than off by surprise; that satisfies
the "opt-in, or a bounded resumable reference" contract without duplicating
the transaction layer's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.surfaces.payloads import ProjectionAvailabilityPayload


class ProjectionCostClass(str, Enum):
    """Execution cost tier a default analysis projection declares."""

    CHEAP = "cheap"
    """Session-summary-scoped or otherwise bounded to an interactive budget."""

    EXPENSIVE = "expensive"
    """Full-corpus scan; only ever computed when explicitly opted into."""


@dataclass(frozen=True, slots=True)
class ProjectionPrerequisite:
    """One evidence/configuration prerequisite a projection checks before computing."""

    name: str
    description: str
    remediation: str


@dataclass(frozen=True, slots=True)
class ProjectionContract:
    """Declared readiness/cost contract for one default analysis projection."""

    name: str
    cost_class: ProjectionCostClass
    default_deadline_s: float | None
    prerequisites: tuple[ProjectionPrerequisite, ...] = ()
    notes: str = ""

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-serializable contract, for explain output (AC #4)."""

        return {
            "projection": self.name,
            "cost_class": self.cost_class.value,
            "default_deadline_s": self.default_deadline_s,
            "prerequisites": [
                {"name": p.name, "description": p.description, "remediation": p.remediation} for p in self.prerequisites
            ],
            "notes": self.notes,
        }


COST_ANCHOR_PREREQUISITE = ProjectionPrerequisite(
    name="cycle_anchor_day",
    description=("The plan must declare a fixed monthly cycle_anchor_day to compute a billing-cycle window."),
    remediation=(
        "Configure a 'cycle_anchor_day' under [[cost.subscription.plans]] in polylogue.toml, "
        "or choose a plan with a fixed monthly anchor."
    ),
)

COST_OUTLOOK_CONTRACT = ProjectionContract(
    name="cost-outlook",
    cost_class=ProjectionCostClass.CHEAP,
    default_deadline_s=2.0,
    prerequisites=(COST_ANCHOR_PREREQUISITE,),
    notes="Projects the current billing cycle from materialized session-cost insights; no archive scan.",
)

FACETS_DEFERRED_PREREQUISITE = ProjectionPrerequisite(
    name="include_deferred",
    description=(
        "Deferred families (repos, role_counts, material_origins, message_types, action_types, "
        "has_flags) scan the full messages/actions tables, not just session summaries."
    ),
    remediation="Pass --include-deferred to opt in; expect a full-corpus scan rather than an interactive response.",
)

FACETS_CONTRACT = ProjectionContract(
    name="facets",
    cost_class=ProjectionCostClass.CHEAP,
    default_deadline_s=2.0,
    notes="Default families (total_counts, origins, tags) are session-summary-scoped.",
)

FACETS_DEFERRED_CONTRACT = ProjectionContract(
    name="facets-deferred",
    cost_class=ProjectionCostClass.EXPENSIVE,
    default_deadline_s=None,
    prerequisites=(FACETS_DEFERRED_PREREQUISITE,),
    notes="Opt-in full-corpus family scan; no interactive deadline is declared.",
)

PROJECTION_CONTRACTS: dict[str, ProjectionContract] = {
    "cost-outlook": COST_OUTLOOK_CONTRACT,
    "facets": FACETS_CONTRACT,
    "facets-deferred": FACETS_DEFERRED_CONTRACT,
}
"""Registry of declared contracts, keyed by projection name (AC #1)."""


def budget_exceeded(elapsed_s: float | None, deadline_s: float | None) -> bool:
    """Return whether measured execution missed a declared deadline.

    ``None`` on either side means "no measurement" / "no declared budget" --
    never treated as exceeded.
    """

    if elapsed_s is None or deadline_s is None:
        return False
    return elapsed_s > deadline_s


def cost_outlook_availability(
    plan_name: str,
    *,
    ready: bool,
    elapsed_s: float | None = None,
) -> ProjectionAvailabilityPayload:
    """Build the typed availability envelope for ``analyze --cost-outlook`` (AC #2).

    ``ready`` is ``True`` when the plan resolved a cycle window (the caller
    already has a non-``None`` :class:`~polylogue.cost.outlook.CycleOutlook`);
    ``False`` when the plan has no ``cycle_anchor_day`` and the only honest
    response is a typed unavailable result with remediation.
    """

    from polylogue.surfaces.payloads import ProjectionAvailabilityPayload, ProjectionPrerequisitePayload

    contract = PROJECTION_CONTRACTS["cost-outlook"]
    prereq = contract.prerequisites[0]
    prerequisites = (
        ProjectionPrerequisitePayload(
            name=prereq.name,
            satisfied=ready,
            description=prereq.description,
            remediation=None if ready else prereq.remediation,
        ),
    )
    exceeded = budget_exceeded(elapsed_s, contract.default_deadline_s)
    if ready:
        return ProjectionAvailabilityPayload(
            projection=contract.name,
            state="degraded" if exceeded else "ready",
            cost_class=contract.cost_class.value,
            reason="budget_exceeded" if exceeded else None,
            deadline_s=contract.default_deadline_s,
            elapsed_s=elapsed_s,
            budget_exceeded=exceeded,
            prerequisites=prerequisites,
        )
    return ProjectionAvailabilityPayload(
        projection=contract.name,
        state="unavailable",
        cost_class=contract.cost_class.value,
        reason="no_cycle_anchor",
        detail=(f"No cycle window for plan {plan_name!r}: the plan does not declare a 'cycle_anchor_day'."),
        remediation=prereq.remediation,
        deadline_s=contract.default_deadline_s,
        elapsed_s=elapsed_s,
        budget_exceeded=exceeded,
        prerequisites=prerequisites,
    )


def facets_availability(*, include_deferred: bool, elapsed_s: float | None) -> ProjectionAvailabilityPayload:
    """Build the typed availability envelope for ``analyze --facets`` (AC #3)."""

    from polylogue.surfaces.payloads import ProjectionAvailabilityPayload, ProjectionPrerequisitePayload

    contract = PROJECTION_CONTRACTS["facets-deferred" if include_deferred else "facets"]
    exceeded = budget_exceeded(elapsed_s, contract.default_deadline_s)
    prerequisites = (
        (
            ProjectionPrerequisitePayload(
                name=contract.prerequisites[0].name,
                satisfied=True,
                description=contract.prerequisites[0].description,
                remediation=None,
            ),
        )
        if include_deferred
        else ()
    )
    return ProjectionAvailabilityPayload(
        projection="facets",
        state="degraded" if exceeded else "ready",
        cost_class=contract.cost_class.value,
        reason="budget_exceeded" if exceeded else None,
        detail=(
            f"facets took {elapsed_s:.1f}s, exceeding its declared {contract.default_deadline_s:.1f}s "
            "interactive budget"
            if exceeded and elapsed_s is not None and contract.default_deadline_s is not None
            else None
        ),
        deadline_s=contract.default_deadline_s,
        elapsed_s=elapsed_s,
        budget_exceeded=exceeded,
        prerequisites=prerequisites,
    )


__all__ = [
    "COST_ANCHOR_PREREQUISITE",
    "COST_OUTLOOK_CONTRACT",
    "FACETS_CONTRACT",
    "FACETS_DEFERRED_CONTRACT",
    "FACETS_DEFERRED_PREREQUISITE",
    "PROJECTION_CONTRACTS",
    "ProjectionContract",
    "ProjectionCostClass",
    "ProjectionPrerequisite",
    "budget_exceeded",
    "cost_outlook_availability",
    "facets_availability",
]
