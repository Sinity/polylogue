"""A portfolio decides one terminal outcome from its own coverage."""

from __future__ import annotations

import pytest

from polylogue.analysis.portfolio import compile_portfolio_bundle, portfolio_outcome
from polylogue.analysis.postmortem import PostmortemScope


@pytest.mark.parametrize(
    ("scope", "state", "gaps"),
    [
        (PostmortemScope(matched_session_count=0), "empty", []),
        (
            PostmortemScope(matched_session_count=3, analyzed_session_count=0, truncated=True, dropped_session_count=1),
            "degraded",
            ["match_cap_exceeded", "session_profile_unavailable"],
        ),
        (
            PostmortemScope(matched_session_count=2, analyzed_session_count=0),
            "degraded",
            ["session_profile_unavailable"],
        ),
    ],
    ids=["empty-scope", "truncated", "unprofiled"],
)
def test_portfolio_outcome_names_every_coverage_gap(scope: PostmortemScope, state: str, gaps: list[str]) -> None:
    """A truncated or partly unprofiled scope is ``degraded``, never ``ok``.

    Anti-vacuity: return ``decide_outcome(matched=...)`` without the gap list
    and the truncated and unprofiled scopes read as complete reports.
    """
    outcome = portfolio_outcome(compile_portfolio_bundle((), {}, scope=scope))

    assert outcome.state == state
    assert outcome.detail.get("gaps", []) == gaps
