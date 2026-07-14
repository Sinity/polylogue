from __future__ import annotations

import pytest

from polylogue.insights.measurement.alert_budget import (
    AlertBudgetPolicy,
    AlertBudgetState,
    AlertCandidate,
    AlertDecision,
    evaluate_alert_candidates,
)

_POLICY = AlertBudgetPolicy(cooldown_ms=60_000, magnitude_floor=1.5, global_budget_per_window=1, window_ms=3_600_000)


def test_repeated_unchanged_deviation_fires_once_then_respects_cooldown() -> None:
    state = AlertBudgetState()
    policy = AlertBudgetPolicy(
        cooldown_ms=60_000, magnitude_floor=1.5, global_budget_per_window=10, window_ms=3_600_000
    )
    candidate = AlertCandidate(watch_ref="watch:a", standardized_deviation=3.0, detected_at_ms=0)

    first = evaluate_alert_candidates([candidate], policy=policy, state=state, now_ms=0)
    second = evaluate_alert_candidates(
        [AlertCandidate(watch_ref="watch:a", standardized_deviation=3.0, detected_at_ms=10_000)],
        policy=policy,
        state=state,
        now_ms=10_000,
    )

    assert first[0].outcome == "fired"
    assert second[0].outcome == "suppressed-cooldown"


def test_cooldown_clears_after_the_declared_window() -> None:
    state = AlertBudgetState()
    policy = AlertBudgetPolicy(
        cooldown_ms=60_000, magnitude_floor=1.5, global_budget_per_window=10, window_ms=3_600_000
    )
    candidate = AlertCandidate(watch_ref="watch:a", standardized_deviation=3.0, detected_at_ms=0)

    evaluate_alert_candidates([candidate], policy=policy, state=state, now_ms=0)
    later = evaluate_alert_candidates(
        [AlertCandidate(watch_ref="watch:a", standardized_deviation=3.0, detected_at_ms=70_000)],
        policy=policy,
        state=state,
        now_ms=70_000,
    )

    assert later[0].outcome == "fired"


def test_sub_floor_changes_never_alert() -> None:
    state = AlertBudgetState()
    candidate = AlertCandidate(watch_ref="watch:a", standardized_deviation=0.2, detected_at_ms=0)

    decisions = evaluate_alert_candidates([candidate], policy=_POLICY, state=state, now_ms=0)

    assert decisions[0].outcome == "suppressed-magnitude-floor"


def test_negative_deviation_below_floor_in_magnitude_still_suppressed() -> None:
    state = AlertBudgetState()
    candidate = AlertCandidate(watch_ref="watch:a", standardized_deviation=-0.5, detected_at_ms=0)

    decisions = evaluate_alert_candidates([candidate], policy=_POLICY, state=state, now_ms=0)

    assert decisions[0].outcome == "suppressed-magnitude-floor"


def test_budget_exhaustion_suppresses_lower_priority_candidates_with_receipts() -> None:
    state = AlertBudgetState()
    candidates = [
        AlertCandidate(watch_ref="watch:small", standardized_deviation=2.0, detected_at_ms=0),
        AlertCandidate(watch_ref="watch:big", standardized_deviation=5.0, detected_at_ms=0),
        AlertCandidate(watch_ref="watch:medium", standardized_deviation=3.0, detected_at_ms=0),
    ]

    decisions = evaluate_alert_candidates(candidates, policy=_POLICY, state=state, now_ms=0)
    by_ref = {decision.watch_ref: decision for decision in decisions}

    assert by_ref["watch:big"].outcome == "fired"
    assert by_ref["watch:medium"].outcome == "suppressed-budget"
    assert by_ref["watch:small"].outcome == "suppressed-budget"
    assert all(decision.receipt for decision in decisions)


def test_larger_valid_deviation_wins_ordering_regardless_of_input_order() -> None:
    state = AlertBudgetState()
    candidates = [
        AlertCandidate(watch_ref="watch:medium", standardized_deviation=3.0, detected_at_ms=0),
        AlertCandidate(watch_ref="watch:big", standardized_deviation=5.0, detected_at_ms=0),
    ]

    decisions = evaluate_alert_candidates(candidates, policy=_POLICY, state=state, now_ms=0)

    assert decisions[0].watch_ref == "watch:big"
    assert decisions[0].outcome == "fired"
    assert decisions[1].outcome == "suppressed-budget"


def test_frame_degradation_never_fires_even_with_the_largest_magnitude() -> None:
    state = AlertBudgetState()
    candidates = [
        AlertCandidate(
            watch_ref="watch:degraded",
            standardized_deviation=9.0,
            detected_at_ms=0,
            frame_degraded=True,
            frame_degraded_reason="index generation stale by 2 epochs",
        ),
        AlertCandidate(watch_ref="watch:trustworthy", standardized_deviation=2.0, detected_at_ms=0),
    ]

    decisions = evaluate_alert_candidates(candidates, policy=_POLICY, state=state, now_ms=0)
    by_ref = {decision.watch_ref: decision for decision in decisions}

    assert by_ref["watch:degraded"].outcome == "suppressed-frame-degraded"
    assert "stale" in by_ref["watch:degraded"].receipt
    # Degradation never silently consumes the shared budget either.
    assert by_ref["watch:trustworthy"].outcome == "fired"


def test_frame_degraded_candidate_is_never_silently_absent_from_the_decision_list() -> None:
    state = AlertBudgetState()
    candidates = [
        AlertCandidate(watch_ref="watch:degraded", standardized_deviation=9.0, detected_at_ms=0, frame_degraded=True)
    ]

    decisions = evaluate_alert_candidates(candidates, policy=_POLICY, state=state, now_ms=0)

    assert len(decisions) == 1
    assert decisions[0].watch_ref == "watch:degraded"


def test_restart_preserves_required_operational_state_via_serialization() -> None:
    state = AlertBudgetState()
    candidate = AlertCandidate(watch_ref="watch:a", standardized_deviation=3.0, detected_at_ms=0)
    evaluate_alert_candidates([candidate], policy=_POLICY, state=state, now_ms=0)

    reloaded = AlertBudgetState.from_dict(state.to_dict())
    decisions = evaluate_alert_candidates(
        [AlertCandidate(watch_ref="watch:a", standardized_deviation=3.0, detected_at_ms=10_000)],
        policy=_POLICY,
        state=reloaded,
        now_ms=10_000,
    )

    assert decisions[0].outcome == "suppressed-cooldown"


def test_no_inferential_significance_field_exists_on_candidate_or_decision() -> None:
    """Binding anti-goal: no p-values/significance on exact enumeration."""

    candidate_fields = set(AlertCandidate.__dataclass_fields__)
    decision_fields = set(AlertDecision.__dataclass_fields__)

    for forbidden in ("p_value", "pvalue", "significance", "confidence_level"):
        assert forbidden not in candidate_fields
        assert forbidden not in decision_fields


def test_policy_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="cooldown_ms"):
        AlertBudgetPolicy(cooldown_ms=-1, magnitude_floor=1.0, global_budget_per_window=1, window_ms=1000)
    with pytest.raises(ValueError, match="window_ms"):
        AlertBudgetPolicy(cooldown_ms=0, magnitude_floor=1.0, global_budget_per_window=1, window_ms=0)
