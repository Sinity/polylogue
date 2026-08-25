"""Incremental convergence must match one bulk convergence pass."""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    rich_convergence_pathology,
    rotated_session_order,
)
from tests.infra.convergence_laws import (
    ConvergenceLaw,
    assert_projection_matches_oracle,
    expected_projection,
    generated_convergence_workload,
    read_semantic_projection,
)


@settings(
    max_examples=8,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=1, max_value=len(rich_convergence_pathology().sessions) - 1))
def test_convergence_property_incremental_equals_bulk(tmp_path: Path, shift: int) -> None:
    workload = generated_convergence_workload()
    pathology = workload.pathology
    order = rotated_session_order(pathology, shift)
    bulk = build_converged_archive(tmp_path / "bulk", pathology, session_order=order)
    incremental = build_converged_archive(tmp_path / "incremental", pathology, session_order=order, incremental=True)
    assert_archives_equivalent(bulk, incremental)
    expected = expected_projection(workload)
    for archive in (bulk, incremental):
        assert_projection_matches_oracle(
            read_semantic_projection(archive.root, probe_terms=workload.probe_terms),
            expected,
            law=ConvergenceLaw.BATCHING,
        )
