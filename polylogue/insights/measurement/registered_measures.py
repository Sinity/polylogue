"""Production measure declarations for existing quantitative projections."""

from __future__ import annotations

from polylogue.insights.measurement.metric import MetricDefinition
from polylogue.insights.measurement.registry import MeasureRegistry, MeasureSpec


def _measure(
    name: str,
    construct: str,
    unit: str,
    source: str,
    aggregation: str,
    *,
    authority: tuple[str, ...] = ("structural",),
    tier: str = "structural",
) -> MeasureSpec:
    frame = "all captured logical sessions"
    return MeasureSpec(
        name=name,
        metric=MetricDefinition(
            construct=construct,
            unit=unit,
            unit_source=source,
            aggregation=aggregation,
            required_frame=frame,
            measurement_authority=authority,  # type: ignore[arg-type]
        ),
        evidence_tier=tier,
        sample_frame=frame,
        confounds=("capture completeness",),
    )


SESSION_COST_MEASURE = _measure(
    "session_cost_usd",
    "estimated USD cost per logical session",
    "usd",
    "session_costs",
    "sum",
    authority=("provider-reported", "catalog-estimated"),
    tier="mixed provider/catalog",
)
PLAN_COMPLETION_MEASURE = _measure(
    "plan_completion_rate",
    "completed items divided by planned Claude Code TODO items",
    "ratio",
    "todo_states",
    "mean",
)
TOOL_CALLS_MEASURE = _measure("tool_calls", "structural tool calls per logical session", "count", "actions", "count")
MESSAGE_COUNT_MEASURE = _measure("message_count", "messages per logical session", "count", "messages", "count")
WALL_DURATION_MEASURE = _measure(
    "wall_duration_ms", "observed wall duration per logical session", "milliseconds", "session_profiles", "mean"
)


DEFAULT_MEASURE_REGISTRY = MeasureRegistry()
for _spec in (
    SESSION_COST_MEASURE,
    PLAN_COMPLETION_MEASURE,
    TOOL_CALLS_MEASURE,
    MESSAGE_COUNT_MEASURE,
    WALL_DURATION_MEASURE,
):
    DEFAULT_MEASURE_REGISTRY.register(_spec)


__all__ = [
    "DEFAULT_MEASURE_REGISTRY",
    "MESSAGE_COUNT_MEASURE",
    "PLAN_COMPLETION_MEASURE",
    "SESSION_COST_MEASURE",
    "TOOL_CALLS_MEASURE",
    "WALL_DURATION_MEASURE",
]
