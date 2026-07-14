"""Structural invariants for the improvement-loop declaration registry (polylogue-rxdo.11)."""

from __future__ import annotations

from polylogue.insights.improvement_loops import LOOP_REGISTRY, active_loops, horizon_loops


def test_registry_keys_match_loop_ids() -> None:
    for key, spec in LOOP_REGISTRY.items():
        assert key == spec.loop_id


def test_every_spec_declares_the_full_five_tuple() -> None:
    for spec in LOOP_REGISTRY.values():
        assert spec.watch.strip()
        assert spec.measure.strip()
        assert spec.propose.strip()
        assert spec.judge.strip()
        assert spec.artifact.strip()
        assert ":<hash>" in spec.artifact, f"{spec.loop_id} artifact must be content-addressed"


def test_no_loop_is_active_until_a_shared_scheduler_exists() -> None:
    """Corrective AC: no loop may claim an active daemon schedule at this closure.

    If this test starts failing because someone flipped a spec to
    ``status="active"``, the shared scheduler/state module (and that loop's
    watch/measure/propose/judge/bump wiring) must exist and be tested first
    -- update this test deliberately, not by loosening the assertion.
    """
    assert active_loops() == ()
    assert len(horizon_loops()) == len(LOOP_REGISTRY)


def test_l1_and_l2_are_declared_as_the_first_pilot_pair() -> None:
    assert "L1" in LOOP_REGISTRY
    assert "L2" in LOOP_REGISTRY
    assert LOOP_REGISTRY["L1"].implementation_ref == "polylogue-37t.17"
