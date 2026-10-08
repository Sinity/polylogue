"""Full corpus ingestion equals a converged prefix followed by its delta."""

from __future__ import annotations

import pytest
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    ConvergenceArchive,
    build_converged_archive,
    converge_convergence_archive,
    ingest_composed_sources,
    initialize_active_archive,
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

# The full-corpus side of the law depends only on the rotation, and the law
# only reads it; each distinct rotation is built and verified once per process
# instead of once per example. The prefix-then-delta side is rebuilt fresh in
# every example.
_FULL_BY_SHIFT: dict[int, ConvergenceArchive] = {}


def _converged_full_archive(tmp_path_factory: pytest.TempPathFactory, shift: int) -> ConvergenceArchive:
    if shift not in _FULL_BY_SHIFT:
        composed = generated_convergence_workload().sources
        _FULL_BY_SHIFT[shift] = build_converged_archive(
            tmp_path_factory.mktemp(f"append-prefix-full-{shift}") / "full",
            composed,
            session_order=rotated_session_order(composed, shift),
        )
    return _FULL_BY_SHIFT[shift]


@settings(
    max_examples=8,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
    deadline=None,
)
@given(
    st.integers(min_value=1, max_value=len(generated_convergence_workload().sources.sessions) - 1),
    st.integers(min_value=1, max_value=len(generated_convergence_workload().sources.sessions) - 1),
)
def test_convergence_property_append_prefix_matches_full(
    tmp_path_factory: pytest.TempPathFactory, shift: int, split: int
) -> None:
    tmp_path = tmp_path_factory.mktemp("convergence-example")
    workload = generated_convergence_workload()
    composed = workload.sources
    order = rotated_session_order(composed, shift)
    full = _converged_full_archive(tmp_path_factory, shift)

    prefix_root = tmp_path / "prefix"
    initialize_active_archive(prefix_root)
    prefix = ingest_composed_sources(
        prefix_root,
        composed,
        session_indexes=order[:split],
        converge_after_each=False,
    )
    converge_convergence_archive(prefix)
    combined = ingest_composed_sources(
        prefix_root,
        composed,
        session_indexes=order[split:],
        converge_after_each=False,
    )
    converge_convergence_archive(combined)
    execute_convergence_plan(
        build_convergence_run_plan(workload),
        (full.root, combined.root),
        law=ConvergenceLaw.APPEND_PREFIX,
    )
    expected = semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms)
    for archive in (full, combined):
        assert_projection_matches_oracle(
            read_semantic_projection(archive.root, probe_terms=workload.probe_terms),
            expected,
            law=ConvergenceLaw.APPEND_PREFIX,
        )
