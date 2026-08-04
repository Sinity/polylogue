"""Full corpus ingestion equals a converged prefix followed by its delta."""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    converge_convergence_archive,
    ingest_convergence_pathology,
    initialize_active_archive,
    rich_convergence_pathology,
    rotated_session_order,
)


@settings(
    max_examples=1,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    st.integers(min_value=1, max_value=len(rich_convergence_pathology().sessions) - 1),
    st.integers(min_value=1, max_value=len(rich_convergence_pathology().sessions) - 1),
)
def test_convergence_property_append_prefix_matches_full(tmp_path: Path, shift: int, split: int) -> None:
    pathology = rich_convergence_pathology()
    order = rotated_session_order(pathology, shift)
    full = build_converged_archive(tmp_path / "full", pathology, session_order=order)

    prefix_root = tmp_path / "prefix"
    initialize_active_archive(prefix_root)
    prefix = ingest_convergence_pathology(
        prefix_root,
        pathology,
        session_indexes=order[:split],
        converge_after_each=False,
    )
    converge_convergence_archive(prefix)
    combined = ingest_convergence_pathology(
        prefix_root,
        pathology,
        session_indexes=order[split:],
        converge_after_each=False,
    )
    converge_convergence_archive(combined)
    assert_archives_equivalent(full, combined)
