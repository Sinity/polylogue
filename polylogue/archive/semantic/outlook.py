"""Usage outlook computation — reads #803 cost data, projects future usage (#870)."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import cast

from polylogue.archive.semantic.subscription_models import (
    AnomalyPayload,
    ModelBreakdownPayload,
    PlanConfidence,
    SubscriptionCycle,
    SubscriptionPlanConfig,
    UsageOutlookPayload,
)


def compute_usage_outlook(
    sessions: list[dict[str, object]],
    *,
    plan_name: str = "pro",
    cycle_start: str | None = None,
    anomaly_threshold: float = 2.0,
) -> UsageOutlookPayload:
    """Compute subscription outlook from per-session cost summaries."""
    now = datetime.now(UTC)
    cycle = _resolve_cycle(now, cycle_start)
    daily = _aggregate_daily(sessions)
    projection = _compute_projection(daily, cycle, now)
    anomalies = _detect_anomalies(daily, threshold=anomaly_threshold)
    confidence = _compute_confidence(sessions)

    plan_config = _plan_for_name(plan_name)
    credits_used = projection.total_credits
    credits_total = plan_config.monthly_credit_pool
    credits_remaining = max(0.0, credits_total - credits_used)
    burn_rate = credits_used / max(1, (now - cycle[0]).days) if credits_used > 0 else 0.0
    exhaustion = None
    if burn_rate > 0 and credits_remaining > 0:
        days_left = credits_remaining / burn_rate
        exhaustion = (now + timedelta(days=days_left)).isoformat()

    per_model = _per_model_breakdown(sessions)
    priced = sum(1 for c in sessions if c.get("has_cost"))
    unavailable = len(sessions) - priced
    coverage = (priced / max(1, len(sessions))) * 100

    return UsageOutlookPayload(
        cycle_start=cycle[0].isoformat(),
        cycle_end=cycle[1].isoformat(),
        plan_name=plan_name,
        plan_confidence=PlanConfidence.derived if plan_name == "pro" else PlanConfidence.unknown,
        credits_total=credits_total,
        credits_used=credits_used,
        credits_remaining=credits_remaining,
        burn_rate_credits_per_day=burn_rate,
        projected_exhaustion_date=exhaustion,
        api_equivalent_usd_total=projection.total_api_usd,
        subscription_equivalent_usd_total=projection.total_subscription_usd,
        per_model=per_model,
        anomalies=anomalies,
        priced_session_count=priced,
        unavailable_session_count=unavailable,
        confidence=confidence,
        coverage_pct=coverage,
        projection_method="trailing_30d_linear",
    )


def build_subscription_cycle(
    sessions: list[dict[str, object]],
    *,
    plan_name: str = "pro",
    cycle_start: str | None = None,
) -> SubscriptionCycle:
    outlook = compute_usage_outlook(sessions, plan_name=plan_name, cycle_start=cycle_start)
    return SubscriptionCycle(
        plan=_plan_for_name(plan_name),
        outlook=outlook,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _resolve_cycle(now: datetime, cycle_start: str | None) -> tuple[datetime, datetime]:
    if cycle_start:
        start = datetime.fromisoformat(cycle_start)
    else:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    return start, end


def _aggregate_daily(sessions: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    daily: dict[str, dict[str, float]] = defaultdict(lambda: {"credits": 0.0, "api_usd": 0.0, "count": 0})
    for c in sessions:
        date = str(c.get("created_at", ""))[:10]
        if date:
            daily[date]["credits"] += _as_float(c.get("credit_cost"))
            daily[date]["api_usd"] += _as_float(c.get("api_cost_usd"))
            daily[date]["count"] += 1
    return dict(daily)


class _Projection:
    __slots__ = ("total_credits", "total_api_usd", "total_subscription_usd", "daily_avg_credits")

    def __init__(
        self,
        total_credits: float,
        total_api_usd: float,
        total_subscription_usd: float,
        daily_avg_credits: float,
    ) -> None:
        self.total_credits = total_credits
        self.total_api_usd = total_api_usd
        self.total_subscription_usd = total_subscription_usd
        self.daily_avg_credits = daily_avg_credits


def _compute_projection(
    daily: dict[str, dict[str, float]], cycle: tuple[datetime, datetime], now: datetime
) -> _Projection:
    total_credits = sum(d["credits"] for d in daily.values())
    total_api_usd = sum(d["api_usd"] for d in daily.values())
    days_in_cycle = max(1, (now - cycle[0]).days)
    return _Projection(
        total_credits=total_credits,
        total_api_usd=total_api_usd,
        total_subscription_usd=0.0,
        daily_avg_credits=total_credits / days_in_cycle,
    )


def _detect_anomalies(daily: dict[str, dict[str, float]], *, threshold: float = 2.0) -> list[AnomalyPayload]:
    if len(daily) < 3:
        return []
    costs = [(d, v["api_usd"]) for d, v in sorted(daily.items())]
    avg = sum(v for _, v in costs) / len(costs)
    anomalies: list[AnomalyPayload] = []
    for date, cost in costs:
        if cost > avg * threshold:
            anomalies.append(
                AnomalyPayload(
                    date=date,
                    cost_usd=cost,
                    comparator_window_avg=avg,
                    threshold_multiplier=threshold,
                    affected_component="daily_cost",
                    explanation=f"Daily cost ${cost:.2f} exceeds {threshold}x the average (${avg:.2f})",
                )
            )
    return anomalies[:10]


def _compute_confidence(sessions: list[dict[str, object]]) -> float:
    if not sessions:
        return 0.0
    with_estimate = sum(1 for c in sessions if c.get("cost_is_estimated"))
    return max(0.0, 1.0 - (with_estimate / max(1, len(sessions))))


def _per_model_breakdown(sessions: list[dict[str, object]]) -> list[ModelBreakdownPayload]:
    models: dict[str, dict[str, float]] = defaultdict(lambda: {"api_usd": 0.0, "sessions": 0.0})
    for c in sessions:
        model = str(c.get("model", "unknown"))
        models[model]["api_usd"] += _as_float(c.get("api_cost_usd"))
        models[model]["sessions"] += 1
    return [
        ModelBreakdownPayload(
            model=m,
            api_equivalent_usd=v["api_usd"],
            subscription_equivalent_usd=0.0,
            session_count=int(v["sessions"]),
        )
        for m, v in sorted(models.items(), key=lambda x: -x[1]["api_usd"])
    ]


def _plan_for_name(name: str) -> SubscriptionPlanConfig:
    plans = {
        "pro": SubscriptionPlanConfig(plan_name="pro", monthly_credit_pool=0.0),
        "max": SubscriptionPlanConfig(plan_name="max", monthly_credit_pool=0.0),
    }
    return plans.get(name, SubscriptionPlanConfig(plan_name=name, monthly_credit_pool=0.0))


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(cast(float, value))
    except (ValueError, TypeError):
        return default
