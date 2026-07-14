"""Ratios as derived objects: numerator_ref + denominator_ref (rxdo.9.2).

No bare percentages: a proportion is represented as a :class:`~polylogue.
insights.measurement.metric.MetricDefinition` whose ``formula_kind`` is
``"ratio"`` and whose ``component_refs`` are ``(numerator_ref,
denominator_ref)`` -- both ``metric:<hash>`` refs (rxdo.9.1). Because
``MetricDefinition.ref`` is a pure function of content, two independently
constructed numerator/denominator pairs that describe the same ratio always
collapse to the same metric identity -- there is no second ratio registry or
table.

The computed value is an ordinary evaluation receipt
(:class:`RatioResult`), always carrying numerator, denominator, and an
explicit unknown bucket that is never dropped or coerced to zero -- the
renderer can always answer "% of WHAT".
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.insights.measurement.metric import MetricDefinition, NullPolicy


class IncompatibleRatioError(ValueError):
    """Numerator/denominator grain, frame, or authority envelope mismatch."""


def derive_ratio_metric(
    numerator: MetricDefinition,
    denominator: MetricDefinition,
    *,
    construct: str,
    null_policy: NullPolicy = "separate-unknown",
) -> MetricDefinition:
    """Build the canonical ratio ``MetricDefinition`` citing both component refs.

    Fails closed (raises :class:`IncompatibleRatioError`) when numerator and
    denominator grain, declared frame, or measurement authority envelopes
    are incompatible -- a ratio across mismatched envelopes is a
    construct-validity error, not a renderable number.
    """

    if not numerator.is_compatible_with(denominator):
        raise IncompatibleRatioError(
            f"numerator metric {numerator.ref!r} and denominator metric {denominator.ref!r} "
            "do not share a compatible grain/frame/authority envelope"
        )
    return MetricDefinition(
        construct=construct,
        unit="ratio",
        unit_source=f"{numerator.unit_source}/{denominator.unit_source}",
        aggregation="ratio",
        grain=numerator.grain,
        required_frame=numerator.required_frame or denominator.required_frame,
        null_policy=null_policy,
        provenance_mixing=(
            "mixed-declared"
            if "mixed-declared" in (numerator.provenance_mixing, denominator.provenance_mixing)
            else "single-authority"
        ),
        component_refs=(numerator.ref, denominator.ref),
        formula_kind="ratio",
    )


@dataclass(frozen=True, slots=True)
class RatioResult:
    """A ratio evaluation receipt. Always answers '% of WHAT'."""

    metric_ref: str
    numerator_value: float
    numerator_n: int
    denominator_value: float
    denominator_n: int
    unknown_n: int
    frame: str
    null_policy: NullPolicy

    @property
    def is_suppressed(self) -> bool:
        """True when ``null_policy`` refuses to render with unknowns present."""

        return self.null_policy == "suppress" and self.unknown_n > 0

    @property
    def value(self) -> float | None:
        if self.is_suppressed or self.denominator_value == 0:
            return None
        return self.numerator_value / self.denominator_value

    def render(self) -> str:
        frame_label = self.frame or "declared frame"
        if self.is_suppressed:
            return (
                f"suppressed (null_policy=suppress, unknown_n={self.unknown_n} present) "
                f"of {frame_label} "
                f"(numerator={self.numerator_value:g} n={self.numerator_n}, "
                f"denominator={self.denominator_value:g} n={self.denominator_n}, "
                f"null_policy={self.null_policy})"
            )
        pct = f"{self.value * 100:.1f}%" if self.value is not None else "n/a"
        unknown_note = f", unknown={self.unknown_n}" if self.unknown_n else ""
        return (
            f"{pct} of {frame_label} "
            f"(numerator={self.numerator_value:g} n={self.numerator_n}, "
            f"denominator={self.denominator_value:g} n={self.denominator_n}{unknown_note}, "
            f"null_policy={self.null_policy})"
        )


def evaluate_ratio(
    metric: MetricDefinition,
    *,
    numerator_value: float,
    numerator_n: int,
    denominator_value: float,
    denominator_n: int,
    unknown_n: int = 0,
    frame: str = "",
) -> RatioResult:
    """Evaluate a ratio ``MetricDefinition`` against measured components.

    ``unknown_n`` is always preserved on the result regardless of
    ``null_policy`` -- only :meth:`RatioResult.render`/``value`` change how
    a suppressed unknown bucket is presented, never whether it is recorded.
    """

    if metric.formula_kind != "ratio":
        raise IncompatibleRatioError(f"metric {metric.ref!r} is not a ratio definition")
    return RatioResult(
        metric_ref=metric.ref,
        numerator_value=numerator_value,
        numerator_n=numerator_n,
        denominator_value=denominator_value,
        denominator_n=denominator_n,
        unknown_n=unknown_n,
        frame=frame or metric.required_frame,
        null_policy=metric.null_policy,
    )


__all__ = ["IncompatibleRatioError", "RatioResult", "derive_ratio_metric", "evaluate_ratio"]
