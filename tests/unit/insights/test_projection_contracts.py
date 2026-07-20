"""Unit tests for the declared readiness/cost/degraded projection contracts (polylogue-duti)."""

from __future__ import annotations

from polylogue.insights.projection_contracts import (
    PROJECTION_CONTRACTS,
    ProjectionCostClass,
    budget_exceeded,
    cost_outlook_availability,
    facets_availability,
)


class TestBudgetExceeded:
    def test_none_elapsed_never_exceeds(self) -> None:
        assert budget_exceeded(None, 2.0) is False

    def test_none_deadline_never_exceeds(self) -> None:
        assert budget_exceeded(5.0, None) is False

    def test_within_budget(self) -> None:
        assert budget_exceeded(1.0, 2.0) is False

    def test_exactly_at_deadline_does_not_exceed(self) -> None:
        assert budget_exceeded(2.0, 2.0) is False

    def test_over_budget(self) -> None:
        assert budget_exceeded(2.5, 2.0) is True


class TestProjectionContractRegistry:
    def test_cost_outlook_is_cheap_with_a_cycle_anchor_prerequisite(self) -> None:
        contract = PROJECTION_CONTRACTS["cost-outlook"]
        assert contract.cost_class is ProjectionCostClass.CHEAP
        assert contract.default_deadline_s == 2.0
        assert [p.name for p in contract.prerequisites] == ["cycle_anchor_day"]

    def test_facets_default_is_cheap_with_no_prerequisite(self) -> None:
        contract = PROJECTION_CONTRACTS["facets"]
        assert contract.cost_class is ProjectionCostClass.CHEAP
        assert contract.default_deadline_s == 2.0
        assert contract.prerequisites == ()

    def test_facets_deferred_is_expensive_opt_in_with_no_declared_deadline(self) -> None:
        contract = PROJECTION_CONTRACTS["facets-deferred"]
        assert contract.cost_class is ProjectionCostClass.EXPENSIVE
        assert contract.default_deadline_s is None
        assert [p.name for p in contract.prerequisites] == ["include_deferred"]


class TestCostOutlookAvailability:
    def test_ready_within_budget(self) -> None:
        availability = cost_outlook_availability("claude-pro", ready=True, elapsed_s=0.1)
        assert availability.projection == "cost-outlook"
        assert availability.state == "ready"
        assert availability.cost_class == "cheap"
        assert availability.budget_exceeded is False
        assert availability.reason is None
        assert availability.prerequisites[0].satisfied is True
        assert availability.prerequisites[0].remediation is None

    def test_ready_but_over_budget_is_degraded_not_silently_ready(self) -> None:
        availability = cost_outlook_availability("claude-pro", ready=True, elapsed_s=5.0)
        assert availability.state == "degraded"
        assert availability.budget_exceeded is True
        assert availability.reason == "budget_exceeded"

    def test_missing_cycle_anchor_is_typed_unavailable_with_remediation(self) -> None:
        availability = cost_outlook_availability("github-copilot-pro", ready=False, elapsed_s=0.05)
        assert availability.state == "unavailable"
        assert availability.reason == "no_cycle_anchor"
        assert availability.detail is not None
        assert "github-copilot-pro" in availability.detail
        assert availability.remediation is not None
        assert "cost.subscription.plans" in availability.remediation
        prereq = availability.prerequisites[0]
        assert prereq.name == "cycle_anchor_day"
        assert prereq.satisfied is False
        assert prereq.remediation == availability.remediation


class TestFacetsAvailability:
    def test_default_families_ready_within_budget(self) -> None:
        availability = facets_availability(include_deferred=False, elapsed_s=0.2)
        assert availability.projection == "facets"
        assert availability.state == "ready"
        assert availability.cost_class == "cheap"
        assert availability.deadline_s == 2.0
        assert availability.budget_exceeded is False
        assert availability.prerequisites == ()

    def test_default_families_over_budget_reports_degraded(self) -> None:
        availability = facets_availability(include_deferred=False, elapsed_s=17.8)
        assert availability.state == "degraded"
        assert availability.budget_exceeded is True
        assert availability.reason == "budget_exceeded"
        assert availability.detail is not None
        assert "17.8" in availability.detail
        assert "interactive budget" in availability.detail

    def test_deferred_families_are_expensive_opt_in_with_no_deadline(self) -> None:
        availability = facets_availability(include_deferred=True, elapsed_s=30.0)
        assert availability.cost_class == "expensive"
        assert availability.deadline_s is None
        # No declared deadline means an opt-in expensive scan is never
        # reported as "exceeding" a budget it never claimed to meet.
        assert availability.budget_exceeded is False
        assert availability.state == "ready"
        assert [p.name for p in availability.prerequisites] == ["include_deferred"]
        assert availability.prerequisites[0].satisfied is True
