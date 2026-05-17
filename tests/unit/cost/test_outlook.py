"""Cycle outlook engine tests (#1137)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from polylogue.cost.outlook import (
    CycleOutlook,
    DailyUsage,
    OverageRow,
    ProjectionMethod,
    QuotaPressure,
    QuotaPressureMissing,
    build_cycle_outlook,
    project_linear,
    project_trailing_mean,
)
from polylogue.cost.plans import OverageRule, QuotaBasis, SubscriptionPlan


def _plan(
    *,
    name: str = "test-plan",
    quota: float | None = 30_000.0,
    basis: QuotaBasis | None = QuotaBasis.credits,
    overage_rule: OverageRule = OverageRule.metered,
    rate: float | None = 0.001,
    cycle_anchor_day: int | None = 1,
    confidence: float = 1.0,
) -> SubscriptionPlan:
    return SubscriptionPlan(
        name=name,
        provider="test",
        display_name=name,
        monthly_cost_usd=20.0,
        cycle_anchor_day=cycle_anchor_day,
        quota=quota,
        quota_basis=basis,
        overage_rule=overage_rule,
        overage_rate_usd_per_unit=rate if overage_rule in (OverageRule.metered, OverageRule.soft) else None,
        confidence=confidence,
    )


def _full_coverage(window_start: date, days: int, per_day: float, basis: QuotaBasis) -> list[DailyUsage]:
    return [DailyUsage(day=window_start + timedelta(days=i), basis=basis, amount=per_day) for i in range(days)]


# ---------------------------------------------------------------------------
# Projection primitives
# ---------------------------------------------------------------------------


def test_project_linear_zero_elapsed() -> None:
    assert project_linear(0.0, 0.0, 30.0) == 0.0


def test_project_linear_extrapolates() -> None:
    # 1000 used in 10 days over 30-day cycle => 3000 projected
    assert project_linear(1000.0, 10.0, 30.0) == pytest.approx(3000.0)


def test_project_trailing_mean_adds_remaining_only() -> None:
    # 10 days of 100/day, trailing-7 mean = 100 => total = 1000 + 100*20 = 3000
    daily = [100.0] * 10
    projected = project_trailing_mean(daily, elapsed_days=10.0, total_days=30.0, window=7)
    assert projected == pytest.approx(3000.0)


def test_project_trailing_mean_recent_spike() -> None:
    # Slow start then last-day spike — trailing mean weights the spike
    daily = [0.0] * 9 + [700.0]
    projected = project_trailing_mean(daily, elapsed_days=10.0, total_days=30.0, window=7)
    # Used = 700, trailing-7 mean = 100, remaining = 20 days => 700 + 2000 = 2700
    assert projected == pytest.approx(2700.0)


# ---------------------------------------------------------------------------
# Cycle outlook: AC walkthrough
# ---------------------------------------------------------------------------


def test_no_cycle_anchor_returns_none() -> None:
    plan = _plan(cycle_anchor_day=None, quota=None, basis=None, overage_rule=OverageRule.unknown, rate=None)
    now = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
    assert build_cycle_outlook(plan, [], now=now) is None


def test_linear_projection_over_fixed_fixture() -> None:
    plan = _plan(quota=30_000.0, basis=QuotaBasis.credits)
    # Cycle anchor day=1; for now=May 11, cycle starts May 1, 30-day cycle => ends May 31.
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    usage = _full_coverage(date(2026, 5, 1), days=10, per_day=1000.0, basis=QuotaBasis.credits)
    outlook = build_cycle_outlook(plan, usage, now=now, method=ProjectionMethod.linear)
    assert outlook is not None
    assert outlook.cycle_to_date["credits"] == pytest.approx(10_000.0)
    assert outlook.burn_rate_per_day["credits"] == pytest.approx(1000.0)
    assert outlook.projected_total["credits"] == pytest.approx(30_000.0)
    assert outlook.projection_method is ProjectionMethod.linear
    assert outlook.coverage_ratio == 1.0
    assert outlook.incomplete_days == ()
    assert outlook.confidence == pytest.approx(1.0)


def test_missing_day_drops_coverage() -> None:
    plan = _plan()
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    usage = _full_coverage(date(2026, 5, 1), days=10, per_day=1000.0, basis=QuotaBasis.credits)
    # Drop day index 5 (2026-05-06)
    usage = [row for row in usage if row.day != date(2026, 5, 6)]
    outlook = build_cycle_outlook(plan, usage, now=now)
    assert outlook is not None
    assert outlook.incomplete_days == (date(2026, 5, 6),)
    # 1 missing day out of 10 fully-elapsed days (May 1..May 10)
    assert outlook.coverage_ratio == pytest.approx(0.9)
    # confidence reflects coverage drop
    assert outlook.confidence < 1.0


def test_plan_without_quota_returns_missing_pressure() -> None:
    plan = _plan(quota=None, basis=None, overage_rule=OverageRule.soft, rate=None)
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    usage = [DailyUsage(day=date(2026, 5, 1) + timedelta(days=i), basis="usd", amount=2.50) for i in range(10)]
    outlook = build_cycle_outlook(plan, usage, now=now)
    assert outlook is not None
    assert isinstance(outlook.quota_pressure, QuotaPressureMissing)
    assert outlook.quota_pressure.reason == "no_quota_configured"
    assert outlook.overage_rows == ()


def test_quota_crossing_threshold_mid_cycle() -> None:
    # Heavy ramp that crosses the 30k quota partway through; ensures breach_day is set
    # and overage rows are emitted with metered cost.
    plan = _plan(quota=30_000.0, basis=QuotaBasis.credits)
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    # 4000/day for 10 days = 40_000 used (exceeds 30k quota at day 8)
    usage = _full_coverage(date(2026, 5, 1), days=10, per_day=4000.0, basis=QuotaBasis.credits)
    outlook = build_cycle_outlook(plan, usage, now=now)
    assert outlook is not None
    pressure = outlook.quota_pressure
    assert isinstance(pressure, QuotaPressure)
    assert pressure.used == pytest.approx(40_000.0)
    assert pressure.used_ratio > 1.0
    # breach should be on day where running total first >= 30k: 4000*8 = 32_000 => 2026-05-08
    assert pressure.breach_day == date(2026, 5, 8)
    assert len(outlook.overage_rows) == 1
    row = outlook.overage_rows[0]
    assert isinstance(row, OverageRow)
    assert row.actual_overage == pytest.approx(10_000.0)
    assert row.overage_rule is OverageRule.metered
    assert row.rate_usd_per_unit == pytest.approx(0.001)
    assert row.projected_overage_cost_usd is not None
    assert row.projected_overage_cost_usd > 0.0


def test_quota_projected_breach_without_actual_overage() -> None:
    # Low used, but projection crosses quota before cycle end -> breach_day set.
    plan = _plan(quota=20_000.0, basis=QuotaBasis.credits)
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    # 1000/day for 10 days => used = 10_000, projected linear = 30_000 (over 20_000 quota)
    usage = _full_coverage(date(2026, 5, 1), days=10, per_day=1000.0, basis=QuotaBasis.credits)
    outlook = build_cycle_outlook(plan, usage, now=now)
    assert outlook is not None
    pressure = outlook.quota_pressure
    assert isinstance(pressure, QuotaPressure)
    assert pressure.projected_ratio > 1.0
    assert pressure.breach_day is not None
    # Projected breach: (20_000 - 10_000) / 1000 = 10 days from now (2026-05-11) => 2026-05-21
    assert pressure.breach_day == date(2026, 5, 21)
    assert pressure.breach_day <= outlook.window.end.date()
    # Actual overage is zero, projected overage is positive.
    assert len(outlook.overage_rows) == 1
    assert outlook.overage_rows[0].actual_overage == 0.0
    assert outlook.overage_rows[0].projected_overage == pytest.approx(10_000.0)


def test_multi_basis_cycle_keeps_all_keys() -> None:
    plan = _plan(quota=30_000.0, basis=QuotaBasis.credits)
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    usage: list[DailyUsage] = []
    for i in range(10):
        day = date(2026, 5, 1) + timedelta(days=i)
        usage.append(DailyUsage(day=day, basis=QuotaBasis.credits, amount=1000.0))
        usage.append(DailyUsage(day=day, basis="usd", amount=0.50))
    outlook = build_cycle_outlook(plan, usage, now=now)
    assert outlook is not None
    assert set(outlook.cycle_to_date.keys()) == {"credits", "usd"}
    assert outlook.cycle_to_date["usd"] == pytest.approx(5.0)
    # Quota pressure attaches to the plan's quota_basis (credits), not USD.
    assert isinstance(outlook.quota_pressure, QuotaPressure)
    assert outlook.quota_pressure.basis is QuotaBasis.credits


def test_zero_usage_cycle_produces_zero_projection() -> None:
    plan = _plan(quota=30_000.0, basis=QuotaBasis.credits)
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    outlook = build_cycle_outlook(plan, [], now=now)
    assert outlook is not None
    assert outlook.cycle_to_date == {}
    assert outlook.projected_total == {}
    assert outlook.overage_rows == ()
    pressure = outlook.quota_pressure
    assert isinstance(pressure, QuotaPressure)
    assert pressure.used == 0.0
    assert pressure.projected == 0.0
    assert pressure.breach_day is None
    # Coverage is 0 because no days are present (11 expected, 0 supplied)
    assert outlook.coverage_ratio == 0.0


def test_end_of_cycle_boundary() -> None:
    # now = last instant before cycle end; remaining_days near zero.
    plan = _plan(quota=30_000.0, basis=QuotaBasis.credits)
    now = datetime(2026, 5, 30, 23, 0, tzinfo=UTC)
    # Full coverage May 1..May 30 (30 days), 1000/day = 30_000 = exactly quota.
    usage = _full_coverage(date(2026, 5, 1), days=30, per_day=1000.0, basis=QuotaBasis.credits)
    outlook = build_cycle_outlook(plan, usage, now=now)
    assert outlook is not None
    assert outlook.window.remaining_days < 1.5
    assert outlook.window.elapsed_days > 29.0
    pressure = outlook.quota_pressure
    assert isinstance(pressure, QuotaPressure)
    assert pressure.used == pytest.approx(30_000.0)
    # At exactly quota: breach_day is the day running total first hits quota (May 30)
    assert pressure.breach_day == date(2026, 5, 30)


def test_trailing_mean_method_tagged_on_outlook() -> None:
    plan = _plan(quota=30_000.0, basis=QuotaBasis.credits)
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    usage = _full_coverage(date(2026, 5, 1), days=10, per_day=1000.0, basis=QuotaBasis.credits)
    outlook = build_cycle_outlook(plan, usage, now=now, method=ProjectionMethod.trailing_7d_mean)
    assert outlook is not None
    assert outlook.projection_method is ProjectionMethod.trailing_7d_mean


def test_outlook_is_frozen() -> None:
    plan = _plan()
    now = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    usage = _full_coverage(date(2026, 5, 1), days=10, per_day=1000.0, basis=QuotaBasis.credits)
    outlook = build_cycle_outlook(plan, usage, now=now)
    assert isinstance(outlook, CycleOutlook)
    with pytest.raises(Exception, match=r".*"):
        outlook.plan_name = "mutated"
