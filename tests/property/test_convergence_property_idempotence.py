"""Repeated real acquisition and parsing is content-hash idempotent."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    convergence_max_examples,
    drop_one_insight_row,
    ingest_convergence_pathology,
    rich_convergence_pathology,
)


@settings(
    max_examples=convergence_max_examples(),
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.permutations((0, 1, 2)))
def test_convergence_property_reingest_is_idempotent(tmp_path: Path, order: tuple[int, ...]) -> None:
    corpus = rich_convergence_pathology()
    archive = build_converged_archive(tmp_path / "archive", corpus, session_order=order)
    baseline = build_converged_archive(tmp_path / "baseline", corpus, session_order=order)

    reingested = ingest_convergence_pathology(
        archive.root,
        corpus,
        session_indexes=order,
        converge_after_each=False,
    )
    from tests.infra.convergence_harness import converge_convergence_archive

    converge_convergence_archive(reingested)
    assert_archives_equivalent(baseline, reingested)


def test_convergence_property_idempotence_red_twin_detects_dropped_profile(tmp_path: Path) -> None:
    corpus = rich_convergence_pathology()
    baseline = build_converged_archive(tmp_path / "baseline", corpus)
    mutated = build_converged_archive(tmp_path / "mutated", corpus)
    drop_one_insight_row(mutated.root)

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(baseline, mutated)
