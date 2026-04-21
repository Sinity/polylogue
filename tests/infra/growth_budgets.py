"""Reusable growth-envelope assertions for tiered verification workloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GrowthObservation:
    """One workload observation at a concrete archive or input size."""

    tier: str
    size: int
    metrics: Mapping[str, float]

    def metric(self, name: str) -> float:
        try:
            return float(self.metrics[name])
        except KeyError as exc:
            raise KeyError(f"growth observation {self.tier!r} is missing metric {name!r}") from exc


@dataclass(frozen=True, slots=True)
class GrowthViolation:
    """One growth-budget failure."""

    metric: str
    tier: str
    message: str


@dataclass(frozen=True, slots=True)
class GrowthBudget:
    """Envelope for one metric over increasing size tiers."""

    metric: str
    max_step_multiplier: float | None = None
    max_per_unit: float | None = None
    require_monotonic_size: bool = True

    def evaluate(self, observations: Sequence[GrowthObservation]) -> tuple[GrowthViolation, ...]:
        ordered = _ordered_observations(observations, require_monotonic_size=self.require_monotonic_size)
        violations: list[GrowthViolation] = []
        previous: GrowthObservation | None = None
        for observation in ordered:
            metric_value = observation.metric(self.metric)
            if observation.size <= 0:
                violations.append(
                    GrowthViolation(self.metric, observation.tier, f"size must be positive, got {observation.size}")
                )
                previous = observation
                continue
            if self.max_per_unit is not None:
                per_unit = metric_value / observation.size
                if per_unit > self.max_per_unit:
                    violations.append(
                        GrowthViolation(
                            self.metric,
                            observation.tier,
                            f"{per_unit:.3f} per unit exceeds budget {self.max_per_unit:.3f}",
                        )
                    )
            if previous is not None and self.max_step_multiplier is not None:
                previous_value = previous.metric(self.metric)
                if previous_value > 0:
                    multiplier = metric_value / previous_value
                    if multiplier > self.max_step_multiplier:
                        violations.append(
                            GrowthViolation(
                                self.metric,
                                observation.tier,
                                (
                                    f"{multiplier:.3f}x step growth from {previous.tier!r} exceeds "
                                    f"budget {self.max_step_multiplier:.3f}x"
                                ),
                            )
                        )
            previous = observation
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class GrowthBudgetReport:
    """Evaluation result for one or more growth budgets."""

    observations: tuple[GrowthObservation, ...]
    violations: tuple[GrowthViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def assert_ok(self) -> None:
        assert self.ok, "; ".join(violation.message for violation in self.violations)


def evaluate_growth_budgets(
    observations: Iterable[GrowthObservation],
    budgets: Iterable[GrowthBudget],
) -> GrowthBudgetReport:
    """Evaluate budgets over size-tiered workload observations."""
    observation_tuple = tuple(observations)
    violations: list[GrowthViolation] = []
    for budget in budgets:
        violations.extend(budget.evaluate(observation_tuple))
    return GrowthBudgetReport(observations=observation_tuple, violations=tuple(violations))


def assert_growth_budgets(
    observations: Iterable[GrowthObservation],
    budgets: Iterable[GrowthBudget],
) -> None:
    """Assert that all supplied growth budgets hold."""
    evaluate_growth_budgets(observations, budgets).assert_ok()


def _ordered_observations(
    observations: Sequence[GrowthObservation],
    *,
    require_monotonic_size: bool,
) -> tuple[GrowthObservation, ...]:
    ordered = tuple(sorted(observations, key=lambda observation: observation.size))
    if require_monotonic_size and len({observation.size for observation in ordered}) != len(ordered):
        raise ValueError("growth observations must have unique size tiers")
    return ordered


__all__ = [
    "GrowthBudget",
    "GrowthBudgetReport",
    "GrowthObservation",
    "GrowthViolation",
    "assert_growth_budgets",
    "evaluate_growth_budgets",
]
