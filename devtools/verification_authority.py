"""The single-owner boundary for verification concerns.

This is deliberately a declarative matrix, rather than another runtime
controller.  It makes the AgentCTL/Polylogue split reviewable and gives the
static tests one canonical inventory to check.
"""

from __future__ import annotations

from typing import Final, Literal

Authority = Literal["agentctl", "devtools", "pytest-child"]

# Keep each concern in exactly one row.  A concern may be observed by another
# layer, but only its owner may decide or mutate it.
VERIFICATION_AUTHORITY: Final[dict[str, Authority]] = {
    "checkout_identity": "devtools",
    "selection_graph": "devtools",
    "pytest_lanes": "devtools",
    "child_process_interpretation": "pytest-child",
    "scratch": "agentctl",
    "cgroup": "agentctl",
    "resource_admission": "agentctl",
    "deadline": "agentctl",
    "cancellation": "agentctl",
    "logs": "agentctl",
    "generic_result": "agentctl",
    "semantic_receipt": "devtools",
    "retention": "devtools",
    "diagnostics": "devtools",
}

EXPECTED_CONCERNS: Final[frozenset[str]] = frozenset(
    {
        "checkout_identity",
        "selection_graph",
        "pytest_lanes",
        "child_process_interpretation",
        "scratch",
        "cgroup",
        "resource_admission",
        "deadline",
        "cancellation",
        "logs",
        "generic_result",
        "semantic_receipt",
        "retention",
        "diagnostics",
    }
)


def validate_authority_matrix() -> None:
    """Fail closed if the required ownership inventory is edited unsafely."""
    if set(VERIFICATION_AUTHORITY) != EXPECTED_CONCERNS:
        raise AssertionError("verification authority matrix has an unexplained or missing concern")
    if any(owner not in {"agentctl", "devtools", "pytest-child"} for owner in VERIFICATION_AUTHORITY.values()):
        raise AssertionError("verification authority matrix contains an unknown owner")


validate_authority_matrix()

__all__ = ["EXPECTED_CONCERNS", "VERIFICATION_AUTHORITY", "validate_authority_matrix"]
