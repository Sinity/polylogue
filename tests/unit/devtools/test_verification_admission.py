"""Bounded admission rules for affected verification."""

from __future__ import annotations

import pytest

from devtools.verification_admission import (
    AFFECTED_MAX_ESTIMATED_SECONDS,
    AFFECTED_MAX_SELECTED_TESTS,
    AFFECTED_MAX_WORKERS,
    AFFECTED_POLICY_MAX_WORKERS,
    admit_affected_selection,
)
from devtools.worker_memory import CORPUS_MAX_WORKERS


def test_below_cap_is_admitted_with_exact_count_and_budget() -> None:
    decision = admit_affected_selection(
        graph_status="usable",
        graph_reason="testmon datafile present",
        full_rerun_cause=None,
        selected_count=7,
        estimated_seconds=12.5,
    )

    assert decision.admitted
    assert decision.status == "admitted"
    assert decision.selected_count == 7
    assert decision.to_payload()["budget"] == {
        "max_selected_tests": AFFECTED_MAX_SELECTED_TESTS,
        "max_estimated_seconds": AFFECTED_MAX_ESTIMATED_SECONDS,
        "max_workers": AFFECTED_MAX_WORKERS,
    }


def test_above_cap_refuses_without_a_partial_prefix() -> None:
    decision = admit_affected_selection(
        graph_status="usable",
        graph_reason="testmon datafile present",
        full_rerun_cause=None,
        selected_count=AFFECTED_MAX_SELECTED_TESTS + 1,
        estimated_seconds=1,
    )

    assert decision.status == "refused"
    assert str(AFFECTED_MAX_SELECTED_TESTS + 1) in decision.reason
    assert "no partial prefix" in decision.reason
    assert "verify --all" in decision.next_boundary


def test_unknown_graph_refuses_as_unknown_not_as_green_empty_scope() -> None:
    decision = admit_affected_selection(
        graph_status="unusable",
        graph_reason="the datafile is corrupt",
        full_rerun_cause=None,
        selected_count=None,
        estimated_seconds=None,
    )

    assert decision.status == "unknown"
    assert decision.selected_count is None
    assert "corrupt" in decision.reason
    assert "verify --all" in decision.next_boundary


@pytest.mark.parametrize("estimated", [AFFECTED_MAX_ESTIMATED_SECONDS + 0.1, float("inf")])
def test_time_budget_refuses_even_when_count_is_small(estimated: float) -> None:
    decision = admit_affected_selection(
        graph_status="usable",
        graph_reason="testmon datafile present",
        full_rerun_cause=None,
        selected_count=1,
        estimated_seconds=estimated,
    )

    assert decision.status == "refused"
    assert "budget" in decision.reason


def test_affected_width_never_exceeds_the_memory_owner() -> None:
    """The affected cap is derived from the slice's memory budget, not restated.

    An affected run shares ``agentctl-pytest.slice`` with the corpus run, so it
    may never be admitted wider than the width that budget supports.

    Anti-vacuity: restoring a hand-set literal above ``CORPUS_MAX_WORKERS``
    (the previous ``4`` against a derived corpus width of ``3``) makes this red,
    and lowering the slice's MemoryHigh in ``worker_memory`` must now move this
    cap with no edit here.
    """
    assert AFFECTED_MAX_WORKERS <= CORPUS_MAX_WORKERS
    assert min(AFFECTED_POLICY_MAX_WORKERS, CORPUS_MAX_WORKERS) == AFFECTED_MAX_WORKERS
    assert AFFECTED_MAX_WORKERS >= 1
