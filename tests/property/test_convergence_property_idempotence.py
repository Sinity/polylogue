"""Re-ingesting an already converged corpus has no semantic effect."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.archive_canonical_snapshot import (
    assert_canonical_snapshots_equal,
    capture_canonical_snapshot,
)
from tests.infra.convergence_harness import (
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
from tests.infra.durability_faults import InjectedCrash, crash_after_mutating_statements
from tests.infra.sqlite_work_counter import mutating_statements


@settings(
    max_examples=8,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=1, max_value=len(generated_convergence_workload().sources.sessions) - 1))
def test_convergence_property_reingest_is_idempotent(tmp_path: Path, shift: int) -> None:
    """Re-ingesting a converged corpus changes nothing the archive durably holds.

    ``SemanticProjection`` is FTS membership plus aggregate role counts, and
    ``execute_convergence_plan`` deliberately owns route execution rather than
    row inspection. Both are the right shape for the semantic oracle and both
    are blind to the failure this law exists to exclude: a re-ingest that
    duplicates a raw-authority, attachment or provenance row leaves indexed
    text and role totals untouched, so the projection agrees while the durable
    tiers have diverged. The canonical snapshot is the durable-identity half of
    the same law, compared between an archive ingested once and one ingested
    twice, exactly as the interrupted-resume test below compares routes.

    Anti-vacuity: make ``write_source_raw_session`` mint a fresh raw row
    instead of returning the existing content-hash match, and the projection
    still matches the oracle while the snapshot comparison reports the extra
    ``source.raw_sessions`` row. That mutation is invisible to every other
    assertion in this test.
    """
    workload = generated_convergence_workload()
    composed = workload.sources
    order = rotated_session_order(composed, shift)
    archive = build_converged_archive(tmp_path / "archive", composed, session_order=order)
    baseline = build_converged_archive(tmp_path / "baseline", composed, session_order=order)

    reingested = ingest_composed_sources(
        archive.root,
        composed,
        session_indexes=order,
        converge_after_each=False,
    )
    converge_convergence_archive(reingested)
    execute_convergence_plan(
        build_convergence_run_plan(workload),
        (baseline.root, reingested.root),
        law=ConvergenceLaw.IDEMPOTENCE,
    )
    expected = semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms)
    for archive_root in (baseline.root, reingested.root):
        assert_projection_matches_oracle(
            read_semantic_projection(archive_root, probe_terms=workload.probe_terms),
            expected,
            law=ConvergenceLaw.IDEMPOTENCE,
        )
    assert_canonical_snapshots_equal(
        capture_canonical_snapshot(baseline.root),
        capture_canonical_snapshot(reingested.root),
    )


def test_second_convergence_pass_over_unchanged_inputs_writes_nothing(tmp_path: Path) -> None:
    """The derivation law, asserted against the real route rather than a fake.

    ``DerivationReport.wrote_nothing`` states that a second pass over unchanged
    inputs must perform no work. The kernel unit tests assert that on a
    synthetic domain; this asserts it on the production FTS and session-profile
    derivations driven through ``converge_convergence_archive``.

    Anti-vacuity: the first pass is recorded with the same instrument and must
    itself write. A recorder that silently observed nothing — a missed
    ``sqlite3.connect`` seam, a renamed tier — would fail that assertion
    instead of passing this test for free.
    """
    workload = generated_convergence_workload()
    composed = workload.sources
    order = rotated_session_order(composed, 1)

    with mutating_statements() as first_pass:
        archive = build_converged_archive(tmp_path / "archive", composed, session_order=order)
    assert first_pass, "anti-vacuity: building the archive must record mutating statements"

    before = capture_canonical_snapshot(archive.root)
    with mutating_statements() as second_pass:
        converge_convergence_archive(archive)
    after = capture_canonical_snapshot(archive.root)

    assert second_pass == [], f"second convergence pass wrote {len(second_pass)} statements: {second_pass[:5]}"
    assert_canonical_snapshots_equal(before, after)


@pytest.mark.parametrize("crash_at", [3, 9, 17, 34, 61, 88])
def test_interrupted_convergence_resumes_to_the_uninterrupted_state(tmp_path: Path, crash_at: int) -> None:
    """A run killed mid-transaction and resumed must converge to the clean state.

    ``crash_at`` counts mutating statements, so the injected failure lands
    inside an open transaction rather than on a commit boundary -- the case a
    resumed rebuild actually faces after a kill. Resume re-runs the same ingest
    and convergence; the archive must then be canonically identical to one that
    was never interrupted, with no dropped, duplicated or half-written rows.

    Anti-vacuity: a crash point past the end of the run would make this vacuous,
    so the injected crash is asserted to have fired.
    """
    composed = generated_convergence_workload().sources
    order = tuple(range(len(composed.sessions)))
    reference = build_converged_archive(tmp_path / "reference", composed, session_order=order)
    expected = capture_canonical_snapshot(reference.root)

    root = tmp_path / "interrupted"
    initialize_active_archive(root)
    with pytest.raises(InjectedCrash):
        with crash_after_mutating_statements(crash_at):
            converge_convergence_archive(
                ingest_composed_sources(root, composed, session_indexes=order, converge_after_each=False)
            )

    converge_convergence_archive(
        ingest_composed_sources(root, composed, session_indexes=order, converge_after_each=False)
    )
    assert_canonical_snapshots_equal(expected, capture_canonical_snapshot(root))
