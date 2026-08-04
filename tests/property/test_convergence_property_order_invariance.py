"""Order and interruption laws for the real convergence route."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule

from tests.infra.convergence_harness import (
    assert_archive_verification_green,
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
@given(st.integers(min_value=1, max_value=len(rich_convergence_pathology().sessions) - 1))
def test_convergence_property_ingestion_order_invariance(tmp_path: Path, shift: int) -> None:
    pathology = rich_convergence_pathology()
    order = rotated_session_order(pathology, shift)
    canonical = build_converged_archive(tmp_path / "canonical", pathology)
    permuted = build_converged_archive(tmp_path / "permuted", pathology, session_order=order)
    assert_archives_equivalent(canonical, permuted)


class ConvergencePropertyInterruptionMachine(RuleBasedStateMachine):
    """Explore re-ingest/converge interruption boundaries against a real SQLite archive."""

    def __init__(self) -> None:
        super().__init__()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="polylogue-convergence-property-", dir="/dev/shm")
        self._root = Path(self._tmpdir.name)
        self._pathology = rich_convergence_pathology()
        initialize_active_archive(self._root)
        self._dirty = True
        self._archive = ingest_convergence_pathology(
            self._root,
            self._pathology,
            session_indexes=(0,),
            converge_after_each=False,
        )

    @rule(data=st.data())
    def reingest_one_corpus_member(self, data: st.DataObject) -> None:
        index = data.draw(st.integers(min_value=0, max_value=len(self._pathology.sessions) - 1))
        self._archive = ingest_convergence_pathology(
            self._root,
            self._pathology,
            session_indexes=(index,),
            converge_after_each=False,
        )
        self._dirty = True

    @rule()
    def resume_convergence(self) -> None:
        converge_convergence_archive(self._archive)
        assert_archive_verification_green(self._root)
        self._dirty = False

    def teardown(self) -> None:
        if self._dirty:
            converge_convergence_archive(self._archive)
            assert_archive_verification_green(self._root)
        self._tmpdir.cleanup()


TestConvergencePropertyInterruptionMachine = ConvergencePropertyInterruptionMachine.TestCase
TestConvergencePropertyInterruptionMachine.settings = settings(max_examples=1, stateful_step_count=3, deadline=None)


def test_convergence_property_order_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing the real late-parent resolver recreates the historical order bug."""
    from polylogue.storage.sqlite.archive_tiers import write as write_module
    from tests.infra.pathology_composer import compose_fork_prefix_tail_lineage

    pathology = compose_fork_prefix_tail_lineage()
    canonical = build_converged_archive(tmp_path / "canonical", pathology, session_order=(0, 1))

    monkeypatch.setattr(write_module, "_resolve_session_graph", lambda *_args, **_kwargs: None)
    mutated = build_converged_archive(tmp_path / "mutated", pathology, session_order=(1, 0))

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(canonical, mutated)
