"""Incremental convergence must match one bulk convergence pass."""

from __future__ import annotations

import pytest
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    build_converged_archive,
    rotated_session_order,
)
from tests.infra.convergence_laws import (
    ConvergenceLaw,
    assert_projection_matches_oracle,
    build_convergence_run_plan,
    execute_convergence_plan,
    generated_convergence_workload,
    read_semantic_projection,
    semantic_oracle,
)


@settings(
    max_examples=8,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
    deadline=None,
)
@given(st.integers(min_value=1, max_value=len(generated_convergence_workload().sources.sessions) - 1))
def test_convergence_property_incremental_equals_bulk(tmp_path_factory: pytest.TempPathFactory, shift: int) -> None:
    tmp_path = tmp_path_factory.mktemp("convergence-example")
    workload = generated_convergence_workload()
    composed = workload.sources
    order = rotated_session_order(composed, shift)
    bulk = build_converged_archive(tmp_path / "bulk", composed, session_order=order)
    incremental = build_converged_archive(tmp_path / "incremental", composed, session_order=order, incremental=True)
    execute_convergence_plan(
        build_convergence_run_plan(workload),
        (bulk.root, incremental.root),
        law=ConvergenceLaw.BATCHING,
    )
    expected = semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms)
    for archive in (bulk, incremental):
        assert_projection_matches_oracle(
            read_semantic_projection(archive.root, probe_terms=workload.probe_terms),
            expected,
            law=ConvergenceLaw.BATCHING,
        )
