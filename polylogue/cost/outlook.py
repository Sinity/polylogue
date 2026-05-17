"""Cycle outlook engine — burn, projection, quota pressure, overage (#1137).

Pure-function projection over a typed :class:`SubscriptionPlan` and a
per-day usage sequence covering the current billing cycle. No I/O, no
storage. Callers (insights, CLI, MCP) build the daily-usage sequence and
hand it to :func:`build_cycle_outlook` to obtain a :class:`CycleOutlook`
record.

Design constraints driven by #1137 / #870:

- Cycle window comes from :func:`polylogue.cost.plans.cycle_for`. Plans
  without a ``cycle_anchor_day`` cannot produce a cycle outlook — the
  function returns ``None`` rather than fabricating a window.
- Quota pressure and overage rows are only emitted when the plan declares
  both a ``quota`` and a ``quota_basis``. Absence is reported explicitly
  as ``QuotaPressureMissing(reason='no_quota_configured')`` rather than
  defaulting to zero.
- Projection method is an explicit tag (linear, trailing-7d-mean,
  eom-naive). Callers pick the method; we never hide it.
- Coverage drops below 1.0 when the daily-usage sequence is missing a day
  inside the cycle-to-date window. ``incomplete_days`` lists the missing
  ISO dates so the cost surface can surface them.
- Confidence is bounded in ``[0.0, 1.0]`` and combines plan confidence
  with coverage to discourage projecting from sparse data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from polylogue.cost.plans import OverageRule, QuotaBasis, SubscriptionPlan, cycle_for

__all__ = [
    "CycleOutlook",
    "CycleWindow",
    "DailyUsage",
    "OverageRow",
    "ProjectionMethod",
    "QuotaPressure",
    "QuotaPressureMissing",
    "build_cycle_outlook",
    "project_linear",
    "project_trailing_mean",
]


class ProjectionMethod(str, Enum):
    """Explicit tag identifying the projection function used."""

    linear = "linear"
    trailing_7d_mean = "trailing-7d-mean"
    eom_naive = "eom-naive"


class DailyUsage(BaseModel):
    """A single day's usage for one basis within the current cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    basis: QuotaBasis | Literal["usd"]
    amount: float = Field(ge=0.0)


class CycleWindow(BaseModel):
    """Half-open cycle window plus the now-position used for projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: datetime
    end: datetime
    now: datetime
    elapsed_days: float = Field(ge=0.0)
    remaining_days: float = Field(ge=0.0)
    total_days: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _coherent(self) -> CycleWindow:
        if self.end <= self.start:
            raise ValueError("cycle end must be strictly after cycle start")
        if not (self.start <= self.now <= self.end):
            raise ValueError("now must fall within [start, end]")
        return self


class QuotaPressure(BaseModel):
    """Quota pressure for the cycle when the plan declares a quota."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    basis: QuotaBasis
    quota: float = Field(ge=0.0)
    used: float = Field(ge=0.0)
    projected: float = Field(ge=0.0)
    used_ratio: float = Field(ge=0.0)
    projected_ratio: float = Field(ge=0.0)
    breach_day: date | None = None


class QuotaPressureMissing(BaseModel):
    """Explanation for the absence of quota pressure on a plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: Literal["no_quota_configured"] = "no_quota_configured"


class OverageRow(BaseModel):
    """Projected or actual overage against the plan's quota."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    basis: QuotaBasis
    actual_overage: float = Field(ge=0.0)
    projected_overage: float = Field(ge=0.0)
    overage_rule: OverageRule
    rate_usd_per_unit: float | None = None
    projected_overage_cost_usd: float | None = None


class CycleOutlook(BaseModel):
    """Projected outlook for the current billing cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_name: str
    window: CycleWindow
    cycle_to_date: Mapping[str, float]
    burn_rate_per_day: Mapping[str, float]
    projected_total: Mapping[str, float]
    projection_method: ProjectionMethod
    quota_pressure: QuotaPressure | QuotaPressureMissing
    overage_rows: tuple[OverageRow, ...]
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    incomplete_days: tuple[date, ...]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Projection primitives
# ---------------------------------------------------------------------------


def project_linear(used: float, elapsed_days: float, total_days: float) -> float:
    """Linear projection: extrapolate average daily burn over remaining cycle."""
    if elapsed_days <= 0:
        return 0.0
    per_day = used / elapsed_days
    return per_day * total_days


def project_trailing_mean(
    daily_amounts: Sequence[float],
    elapsed_days: float,
    total_days: float,
    *,
    window: int = 7,
) -> float:
    """Trailing-window mean projection.

    The mean of the last ``window`` non-null daily amounts (cycle-to-date)
    is multiplied by the total cycle days. The cycle-to-date sum is added
    to the projection of the *remaining* days, not the full cycle, so the
    historical baseline is preserved.
    """
    if not daily_amounts or total_days <= 0:
        return 0.0
    tail = list(daily_amounts[-window:])
    mean = sum(tail) / len(tail)
    remaining = max(0.0, total_days - elapsed_days)
    return float(sum(daily_amounts)) + mean * remaining


# ---------------------------------------------------------------------------
# Outlook construction
# ---------------------------------------------------------------------------


def _cycle_window(plan: SubscriptionPlan, now: datetime) -> CycleWindow | None:
    window = cycle_for(plan, now)
    if window is None:
        return None
    start = datetime.fromisoformat(window[0].replace("Z", "+00:00"))
    end = datetime.fromisoformat(window[1].replace("Z", "+00:00"))
    now_utc = now.astimezone(UTC)
    total = (end - start).total_seconds() / 86400.0
    elapsed = max(0.0, (now_utc - start).total_seconds() / 86400.0)
    remaining = max(0.0, (end - now_utc).total_seconds() / 86400.0)
    return CycleWindow(
        start=start,
        end=end,
        now=now_utc,
        elapsed_days=elapsed,
        remaining_days=remaining,
        total_days=total,
    )


def _basis_key(basis: QuotaBasis | Literal["usd"]) -> str:
    return basis.value if isinstance(basis, QuotaBasis) else basis


def _expected_days(window: CycleWindow) -> list[date]:
    """Cycle days that should have full usage data by ``now``.

    A day is "expected" once ``now`` has advanced past the start of the
    following day. The day containing ``now`` itself is excluded — it is
    still in progress and missing observations for it are not coverage
    gaps. When ``now`` is exactly at ``start`` (elapsed_days == 0) the
    list is empty.
    """
    start_day = window.start.date()
    # Number of full days elapsed since cycle start.
    full_days = int(window.elapsed_days)
    return [start_day + timedelta(days=i) for i in range(full_days)]


def _group_by_basis(
    usage: Sequence[DailyUsage],
    window: CycleWindow,
) -> dict[str, dict[date, float]]:
    """Group cycle-window usage into ``{basis: {day: amount}}``."""
    grouped: dict[str, dict[date, float]] = {}
    start_day = window.start.date()
    end_day = window.end.date()
    for row in usage:
        if row.day < start_day or row.day > end_day:
            continue
        key = _basis_key(row.basis)
        grouped.setdefault(key, {})[row.day] = grouped.get(key, {}).get(row.day, 0.0) + row.amount
    return grouped


def _project_basis(
    daily_map: Mapping[date, float],
    window: CycleWindow,
    method: ProjectionMethod,
) -> tuple[float, float, float]:
    """Return ``(used, burn_per_day, projected_total)`` for one basis.

    ``used`` is the sum of every day in ``daily_map`` (the caller has
    already restricted entries to the cycle window). Coverage is tracked
    separately so a partial day at the end of the cycle still contributes
    to the cycle-to-date total.
    """
    used = float(sum(daily_map.values()))
    if window.elapsed_days <= 0:
        return used, 0.0, used
    burn = used / window.elapsed_days
    if method is ProjectionMethod.linear:
        projected = project_linear(used, window.elapsed_days, window.total_days)
    elif method is ProjectionMethod.trailing_7d_mean:
        ordered = [daily_map[d] for d in sorted(daily_map)]
        projected = project_trailing_mean(ordered, window.elapsed_days, window.total_days)
    else:  # eom-naive: extrapolate current daily average to cycle end (== linear here)
        projected = project_linear(used, window.elapsed_days, window.total_days)
    return used, burn, projected


def _quota_pressure(
    plan: SubscriptionPlan,
    daily_map: Mapping[date, float],
    used: float,
    projected: float,
    window: CycleWindow,
) -> QuotaPressure | QuotaPressureMissing:
    if plan.quota is None or plan.quota_basis is None:
        return QuotaPressureMissing()
    quota = plan.quota
    used_ratio = used / quota if quota > 0 else 0.0
    projected_ratio = projected / quota if quota > 0 else 0.0
    breach_day: date | None = None
    if quota > 0:
        running = 0.0
        for day in sorted(daily_map):
            running += daily_map[day]
            if running >= quota:
                breach_day = day
                break
        if breach_day is None and projected >= quota and window.elapsed_days > 0:
            per_day = used / window.elapsed_days
            if per_day > 0:
                days_until = (quota - used) / per_day
                projected_breach = window.now.date() + timedelta(days=int(max(0.0, days_until)))
                end_day = window.end.date()
                if projected_breach <= end_day:
                    breach_day = projected_breach
    return QuotaPressure(
        basis=plan.quota_basis,
        quota=quota,
        used=used,
        projected=projected,
        used_ratio=used_ratio,
        projected_ratio=projected_ratio,
        breach_day=breach_day,
    )


def _overage_rows(
    plan: SubscriptionPlan,
    used: float,
    projected: float,
) -> tuple[OverageRow, ...]:
    if plan.quota is None or plan.quota_basis is None:
        return ()
    actual = max(0.0, used - plan.quota)
    projected_over = max(0.0, projected - plan.quota)
    if actual == 0.0 and projected_over == 0.0:
        return ()
    rate = plan.overage_rate_usd_per_unit
    projected_cost = rate * projected_over if rate is not None else None
    return (
        OverageRow(
            basis=plan.quota_basis,
            actual_overage=actual,
            projected_overage=projected_over,
            overage_rule=plan.overage_rule,
            rate_usd_per_unit=rate,
            projected_overage_cost_usd=projected_cost,
        ),
    )


def _coverage(
    primary_daily_map: Mapping[date, float],
    expected_days: Sequence[date],
) -> tuple[float, tuple[date, ...]]:
    if not expected_days:
        return 1.0, ()
    missing = tuple(d for d in expected_days if d not in primary_daily_map)
    ratio = 1.0 - (len(missing) / len(expected_days))
    return ratio, missing


def build_cycle_outlook(
    plan: SubscriptionPlan,
    daily_usage: Sequence[DailyUsage],
    *,
    now: datetime,
    method: ProjectionMethod = ProjectionMethod.linear,
) -> CycleOutlook | None:
    """Project ``daily_usage`` over the plan's current cycle.

    Returns ``None`` when the plan has no cycle anchor — callers must
    surface "no cycle window" rather than fabricate one.
    """
    window = _cycle_window(plan, now)
    if window is None:
        return None
    grouped = _group_by_basis(daily_usage, window)
    expected_days = _expected_days(window)

    cycle_to_date: dict[str, float] = {}
    burn: dict[str, float] = {}
    projected_total: dict[str, float] = {}
    for basis_key, daily_map in grouped.items():
        used, per_day, projected = _project_basis(daily_map, window, method)
        cycle_to_date[basis_key] = used
        burn[basis_key] = per_day
        projected_total[basis_key] = projected

    primary_basis_key = _basis_key(plan.quota_basis) if plan.quota_basis is not None else None
    primary_map: Mapping[date, float]
    if primary_basis_key is not None and primary_basis_key in grouped:
        primary_map = grouped[primary_basis_key]
    elif grouped:
        primary_map = next(iter(grouped.values()))
    else:
        primary_map = {}
    coverage_ratio, incomplete_days = _coverage(primary_map, expected_days)

    if plan.quota_basis is not None:
        basis_used = cycle_to_date.get(_basis_key(plan.quota_basis), 0.0)
        basis_projected = projected_total.get(_basis_key(plan.quota_basis), 0.0)
        pressure = _quota_pressure(
            plan,
            primary_map,
            basis_used,
            basis_projected,
            window,
        )
        overages = _overage_rows(plan, basis_used, basis_projected)
    else:
        pressure = QuotaPressureMissing()
        overages = ()

    confidence = max(0.0, min(1.0, plan.confidence * coverage_ratio))

    return CycleOutlook(
        plan_name=plan.name,
        window=window,
        cycle_to_date=cycle_to_date,
        burn_rate_per_day=burn,
        projected_total=projected_total,
        projection_method=method,
        quota_pressure=pressure,
        overage_rows=overages,
        coverage_ratio=coverage_ratio,
        incomplete_days=incomplete_days,
        confidence=confidence,
    )
