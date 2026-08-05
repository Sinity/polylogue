"""Re-ingesting an already converged corpus has no semantic effect."""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    converge_convergence_archive,
    ingest_convergence_pathology,
    rich_convergence_pathology,
    rotated_session_order,
)


@settings(
    max_examples=8,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=1, max_value=len(rich_convergence_pathology().sessions) - 1))
def test_convergence_property_reingest_is_idempotent(tmp_path: Path, shift: int) -> None:
    pathology = rich_convergence_pathology()
    order = rotated_session_order(pathology, shift)
    archive = build_converged_archive(tmp_path / "archive", pathology, session_order=order)
    baseline = build_converged_archive(tmp_path / "baseline", pathology, session_order=order)

    reingested = ingest_convergence_pathology(
        archive.root,
        pathology,
        session_indexes=order,
        converge_after_each=False,
    )
    converge_convergence_archive(reingested)
    assert_archives_equivalent(baseline, reingested)
