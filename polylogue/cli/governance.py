"""Command governance matrix — mechanical coverage enforcement.

Every command path in the Click tree must have a governance spec. Tests
walk the tree and assert: every path has a spec, every declared capability
actually exists, and new commands cannot be added without declaring coverage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GovernanceSpec:
    """Declared coverage for a single command path."""

    has_json: bool = False
    has_seeded_exercise: bool = False


# Keyed by display_name from CommandPath (e.g. "run", "run parse", "schema audit").
GOVERNANCE: dict[str, GovernanceSpec] = {
    # --- run stages ---
    "run": GovernanceSpec(),
    "run acquire": GovernanceSpec(),
    "run parse": GovernanceSpec(),
    "run materialize": GovernanceSpec(),
    "run render": GovernanceSpec(),
    "run site": GovernanceSpec(),
    "run index": GovernanceSpec(),
    "run embed": GovernanceSpec(),
    "run schema": GovernanceSpec(),
    "run reprocess": GovernanceSpec(),
    "run all": GovernanceSpec(),
    # --- doctor ---
    "doctor": GovernanceSpec(has_json=True),
    # --- audit ---
    "audit": GovernanceSpec(has_json=True),
    "audit generate": GovernanceSpec(),
    # --- schema ---
    "schema": GovernanceSpec(),
    "schema generate": GovernanceSpec(has_json=True),
    "schema list": GovernanceSpec(has_json=True),
    "schema compare": GovernanceSpec(has_json=True),
    "schema promote": GovernanceSpec(has_json=True),
    "schema explain": GovernanceSpec(has_json=True),
    "schema audit": GovernanceSpec(has_json=True),
    # --- products (dynamically generated, but declare governance) ---
    "products": GovernanceSpec(),
    "products analytics": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products day-summaries": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products enrichments": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products phases": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products profiles": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products tags": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products threads": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products week-summaries": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    "products work-events": GovernanceSpec(has_json=True, has_seeded_exercise=True),
    # --- standalone commands ---
    "tags": GovernanceSpec(has_json=True),
    "reset": GovernanceSpec(),
    "auth": GovernanceSpec(),
    "mcp": GovernanceSpec(),
    "completions": GovernanceSpec(),
    "dashboard": GovernanceSpec(),
}


__all__ = ["GOVERNANCE", "GovernanceSpec"]
