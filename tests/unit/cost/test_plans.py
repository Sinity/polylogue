"""Typed subscription-plan config tests (#1132)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from polylogue.config import load_polylogue_config
from polylogue.cost.plans import (
    CURATED_SEED_SOURCE,
    USER_CONFIG_SOURCE,
    WELL_KNOWN_PLANS,
    OverageRule,
    PlanLookupError,
    QuotaBasis,
    SubscriptionPlan,
    cycle_for,
    load_plans,
    plan_by_name,
    resolve_plan,
)


def test_well_known_plans_modeled() -> None:
    expected = {
        "claude-pro",
        "claude-max-5x",
        "claude-max-20x",
        "chatgpt-plus",
        "chatgpt-pro",
        "github-copilot-pro",
        "gemini-advanced",
    }
    assert expected.issubset(set(WELL_KNOWN_PLANS))
    for plan in WELL_KNOWN_PLANS.values():
        assert plan.source == CURATED_SEED_SOURCE
        assert plan.notice
        # The public contract is that subscription-quota math is an estimate
        # from a dated curated seed; stripping the notice would make the UX
        # look vendor-authoritative.
        assert "non-authoritative" in plan.notice.lower()


def test_plan_by_name_unknown_raises_typed() -> None:
    with pytest.raises(PlanLookupError) as exc:
        plan_by_name("not-a-plan")
    assert "not-a-plan" in str(exc.value)
    assert "claude-pro" in str(exc.value)  # known plans listed


def test_plan_by_name_returns_seed() -> None:
    plan = plan_by_name("claude-pro")
    assert plan.provider == "anthropic"
    assert plan.monthly_cost_usd == 20.0
    assert plan.quota_basis is QuotaBasis.credits


def test_user_override_beats_seed() -> None:
    rows = [
        {
            "name": "claude-pro",
            "provider": "anthropic",
            "display_name": "Claude Pro (custom)",
            "monthly_cost_usd": 25.0,
            "quota": 50_000_000,
            "quota_basis": "credits",
            "overage_rule": "metered",
            "overage_rate_usd_per_unit": 0.0001,
            "cycle_anchor_day": 5,
        }
    ]
    catalogue = load_plans(rows)
    plan = catalogue["claude-pro"]
    assert plan.source == USER_CONFIG_SOURCE
    assert plan.monthly_cost_usd == 25.0
    assert plan.overage_rule is OverageRule.metered
    # Curated seed for unrelated plans is retained.
    assert catalogue["claude-max-5x"].source == CURATED_SEED_SOURCE


def test_load_plans_invalid_row_raises_typed() -> None:
    rows = [{"name": "broken"}]  # missing required fields
    with pytest.raises(PlanLookupError) as exc:
        load_plans(rows)
    assert "broken" in str(exc.value)


def test_quota_basis_requires_quota() -> None:
    with pytest.raises(ValidationError):
        SubscriptionPlan(
            name="bad",
            provider="x",
            display_name="X",
            monthly_cost_usd=1.0,
            quota_basis=QuotaBasis.credits,
        )


def test_overage_rate_requires_metered_or_soft() -> None:
    with pytest.raises(ValidationError):
        SubscriptionPlan(
            name="bad",
            provider="x",
            display_name="X",
            monthly_cost_usd=1.0,
            overage_rule=OverageRule.block,
            overage_rate_usd_per_unit=0.01,
        )


def test_currency_normalized_to_upper() -> None:
    plan = SubscriptionPlan(
        name="usd-lower",
        provider="x",
        display_name="X",
        monthly_cost_usd=1.0,
        currency="usd",
    )
    assert plan.currency == "USD"


def test_cycle_for_missing_anchor_returns_none() -> None:
    plan = SubscriptionPlan(
        name="no-anchor",
        provider="x",
        display_name="X",
        monthly_cost_usd=1.0,
    )
    assert cycle_for(plan, datetime(2026, 5, 17, tzinfo=UTC)) is None


def test_cycle_for_anchor_mid_month() -> None:
    plan = SubscriptionPlan(
        name="anchored",
        provider="x",
        display_name="X",
        monthly_cost_usd=1.0,
        cycle_anchor_day=10,
        billing_cycle_days=30,
    )
    # now is after anchor day in May -> cycle starts May 10.
    start, end = cycle_for(plan, datetime(2026, 5, 17, 12, 0, tzinfo=UTC)) or (None, None)
    assert start is not None and end is not None
    assert start.startswith("2026-05-10T00:00:00")
    assert end.startswith("2026-06-09T00:00:00")


def test_cycle_for_anchor_before_now_rolls_back() -> None:
    plan = SubscriptionPlan(
        name="anchored",
        provider="x",
        display_name="X",
        monthly_cost_usd=1.0,
        cycle_anchor_day=20,
    )
    # now is before anchor day in May -> cycle started April 20.
    start, _ = cycle_for(plan, datetime(2026, 5, 5, tzinfo=UTC)) or (None, None)
    assert start is not None and start.startswith("2026-04-20T00:00:00")


def test_resolve_plan_none_returns_none() -> None:
    assert resolve_plan(None) is None


def test_resolve_plan_unknown_raises() -> None:
    with pytest.raises(PlanLookupError):
        resolve_plan("not-a-plan")


def test_resolve_plan_user_row() -> None:
    plan = resolve_plan(
        "custom",
        user_rows=[
            {
                "name": "custom",
                "provider": "internal",
                "display_name": "Custom",
                "monthly_cost_usd": 9.99,
            }
        ],
    )
    assert plan is not None
    assert plan.source == USER_CONFIG_SOURCE
    assert plan.provider == "internal"


def test_config_loads_plans_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text(
        """
[[cost.subscription.plans]]
name = "team-claude"
provider = "anthropic"
display_name = "Team Claude"
monthly_cost_usd = 150.0
quota = 50000000
quota_basis = "credits"
overage_rule = "block"
cycle_anchor_day = 1
""".strip()
    )
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    config = load_polylogue_config()
    rows = config.subscription_plans
    assert len(rows) == 1
    assert rows[0]["name"] == "team-claude"
    catalogue = load_plans(rows)
    assert "team-claude" in catalogue
    assert catalogue["team-claude"].source == USER_CONFIG_SOURCE
    assert catalogue["team-claude"].monthly_cost_usd == 150.0
    # Seed still present.
    assert "claude-pro" in catalogue


def test_config_subscription_plans_default_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_CONFIG", "")
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    config = load_polylogue_config()
    assert config.subscription_plans == ()


def test_subscription_plan_is_frozen() -> None:
    plan = plan_by_name("claude-pro")
    with pytest.raises(ValidationError):
        plan.monthly_cost_usd = 999.0


def test_cycle_window_handles_month_end_anchor_deterministically() -> None:
    """Cycle math is deterministic for any ``cycle_anchor_day`` in [1, 28].

    The plan model already rejects anchor days 29-31 (the documented
    month-length edge cases). For supported anchors, ``cycle_for`` must
    return a half-open ``[start, end)`` covering ``now`` and producing
    stable ISO instants regardless of the current month length.
    """
    plan = SubscriptionPlan(
        name="month-end-anchor",
        provider="anthropic",
        display_name="Edge anchor",
        monthly_cost_usd=10.0,
        cycle_anchor_day=28,
    )
    # February (28 days), March (31), and a transition into April.
    for now in (
        datetime(2026, 2, 15, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 28, 0, 0, tzinfo=UTC),
        datetime(2026, 4, 5, tzinfo=UTC),
    ):
        window = cycle_for(plan, now)
        assert window is not None
        start_iso, end_iso = window
        assert start_iso < end_iso
        # The cycle window always contains ``now`` (or starts exactly at it).
        assert start_iso <= now.isoformat().replace("+00:00", "Z") < end_iso


def test_cycle_window_dst_invariant() -> None:
    """All cycle math operates in UTC, so DST transitions are a no-op.

    The plan model normalizes ``now`` to UTC inside ``cycle_for``. Pin the
    invariant by computing the cycle across a DST transition and asserting
    the boundaries differ by exactly ``billing_cycle_days * 86400`` seconds.
    """
    plan = SubscriptionPlan(
        name="dst-anchor",
        provider="anthropic",
        display_name="DST-spanning",
        monthly_cost_usd=10.0,
        cycle_anchor_day=15,
        billing_cycle_days=30,
    )
    # US DST starts mid-March 2026. UTC math must remain stable.
    now = datetime(2026, 3, 20, 7, 0, tzinfo=UTC)
    window = cycle_for(plan, now)
    assert window is not None
    start = datetime.fromisoformat(window[0].replace("Z", "+00:00"))
    end = datetime.fromisoformat(window[1].replace("Z", "+00:00"))
    assert (end - start).total_seconds() == 30 * 86400
