"""Incremental and bulk convergence share the real ingest state."""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    convergence_max_examples,
    rich_convergence_pathology,
)


@settings(
    max_examples=convergence_max_examples(),
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.data())
def test_convergence_property_incremental_equals_bulk(tmp_path: Path, data: st.DataObject) -> None:
    corpus = rich_convergence_pathology()
    order = data.draw(st.permutations(tuple(range(len(corpus.members)))))
    bulk = build_converged_archive(tmp_path / "bulk", corpus, session_order=order)
    incremental = build_converged_archive(tmp_path / "incremental", corpus, session_order=order, incremental=True)
    assert_archives_equivalent(bulk, incremental)
