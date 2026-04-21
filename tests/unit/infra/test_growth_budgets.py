from __future__ import annotations

import pytest

from tests.infra.growth_budgets import (
    GrowthBudget,
    GrowthObservation,
    assert_growth_budgets,
    evaluate_growth_budgets,
)


def test_growth_budget_accepts_linear_envelope() -> None:
    observations = (
        GrowthObservation("small", 10, {"rss_mb": 20.0, "elapsed_ms": 100.0}),
        GrowthObservation("medium", 20, {"rss_mb": 38.0, "elapsed_ms": 190.0}),
        GrowthObservation("large", 40, {"rss_mb": 78.0, "elapsed_ms": 390.0}),
    )

    report = evaluate_growth_budgets(
        observations,
        (
            GrowthBudget("rss_mb", max_step_multiplier=2.2, max_per_unit=2.1),
            GrowthBudget("elapsed_ms", max_step_multiplier=2.2, max_per_unit=10.0),
        ),
    )

    assert report.ok
    report.assert_ok()


def test_growth_budget_reports_step_multiplier_violations() -> None:
    observations = (
        GrowthObservation("small", 10, {"elapsed_ms": 100.0}),
        GrowthObservation("large", 20, {"elapsed_ms": 500.0}),
    )

    report = evaluate_growth_budgets(observations, (GrowthBudget("elapsed_ms", max_step_multiplier=2.0),))

    assert not report.ok
    assert report.violations[0].metric == "elapsed_ms"
    assert "5.000x" in report.violations[0].message


def test_growth_budget_assertion_summarizes_violations() -> None:
    observations = (
        GrowthObservation("small", 10, {"rss_mb": 20.0}),
        GrowthObservation("large", 20, {"rss_mb": 100.0}),
    )

    with pytest.raises(AssertionError, match="exceeds budget"):
        assert_growth_budgets(observations, (GrowthBudget("rss_mb", max_per_unit=3.0),))


def test_growth_budget_rejects_duplicate_size_tiers() -> None:
    observations = (
        GrowthObservation("small-a", 10, {"rss_mb": 20.0}),
        GrowthObservation("small-b", 10, {"rss_mb": 21.0}),
    )

    with pytest.raises(ValueError, match="unique size tiers"):
        evaluate_growth_budgets(observations, (GrowthBudget("rss_mb"),))
