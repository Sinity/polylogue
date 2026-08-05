"""Permutation and interruption laws for real provider ingestion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule

from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    converge_convergence_archive,
    convergence_max_examples,
    convergence_stateful_max_examples,
    convergence_stateful_step_count,
    drop_one_fts_posting,
    ingest_convergence_pathology,
    initialize_active_archive,
    rich_convergence_pathology,
)


@settings(
    max_examples=convergence_max_examples(),
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.permutations((0, 1, 2)))
def test_convergence_property_ingestion_order_invariance(tmp_path: Path, order: tuple[int, ...]) -> None:
    corpus = rich_convergence_pathology()
    canonical = build_converged_archive(tmp_path / "canonical", corpus)
    permuted = build_converged_archive(tmp_path / "permuted", corpus, session_order=order)
    assert_archives_equivalent(canonical, permuted)


class ConvergencePropertyInterruptionMachine(RuleBasedStateMachine):
    """Resume the real route, then compare every affected session to replay."""

    def __init__(self) -> None:
        super().__init__()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="polylogue-convergence-property-")
        self._root = Path(self._tmpdir.name)
        self._corpus = rich_convergence_pathology()
        self._canonical_round = 0
        initialize_active_archive(self._root)
        self._seen: set[int] = set()
        self._archive = ingest_convergence_pathology(
            self._root,
            self._corpus,
            session_indexes=(0,),
            converge_after_each=False,
        )
        self._seen.add(0)

    @rule(data=st.data())
    def ingest_before_interruption(self, data: st.DataObject) -> None:
        indexes = data.draw(st.permutations((0, 1, 2)))
        selected = tuple(indexes[: data.draw(st.integers(min_value=1, max_value=3))])
        self._seen.update(selected)
        self._archive = ingest_convergence_pathology(
            self._root,
            self._corpus,
            session_indexes=selected,
            converge_after_each=False,
        )

    @rule()
    def resume_and_compare_canonical_replay(self) -> None:
        converge_convergence_archive(self._archive)
        affected = type(self._corpus)(tuple(self._corpus.members[index] for index in sorted(self._seen)))
        self._canonical_round += 1
        canonical = build_converged_archive(
            self._root.parent / f"canonical-{len(self._seen)}-{self._canonical_round}", affected
        )
        assert_archives_equivalent(canonical, self._archive)

    def teardown(self) -> None:
        if self._archive.root.exists():
            converge_convergence_archive(self._archive)
        self._tmpdir.cleanup()


TestConvergencePropertyInterruptionMachine = ConvergencePropertyInterruptionMachine.TestCase
TestConvergencePropertyInterruptionMachine.settings = settings(
    max_examples=convergence_stateful_max_examples(),
    stateful_step_count=convergence_stateful_step_count(),
    deadline=None,
)


def test_convergence_property_fts_red_twin_detects_dropped_posting(tmp_path: Path) -> None:
    corpus = rich_convergence_pathology()
    baseline = build_converged_archive(tmp_path / "baseline", corpus)
    mutated = build_converged_archive(tmp_path / "mutated", corpus)
    drop_one_fts_posting(mutated.root)

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(baseline, mutated)
