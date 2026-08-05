"""Derived snapshots and readiness agree across convergence schedules."""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    assert_derived_readiness_equivalent,
    build_converged_archive,
    rich_convergence_pathology,
    rotated_session_order,
)


@settings(
    max_examples=8,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=1, max_value=len(rich_convergence_pathology().sessions) - 1))
def test_convergence_property_derived_snapshot_readiness_equality(tmp_path: Path, shift: int) -> None:
    """Bulk and incremental production convergence expose the same derived state."""
    pathology = rich_convergence_pathology()
    order = rotated_session_order(pathology, shift)
    bulk = build_converged_archive(tmp_path / "bulk", pathology, session_order=order)
    incremental = build_converged_archive(tmp_path / "incremental", pathology, session_order=order, incremental=True)

    assert_derived_readiness_equivalent(bulk.root, incremental.root)
