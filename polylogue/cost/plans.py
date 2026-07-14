"""Typed subscription-plan configuration (#1132).

Models user- and seed-supplied subscription plans (Claude Pro/Max, ChatGPT
Plus/Pro, GitHub Copilot Pro, etc.) so per-cycle outlook, quota pressure, and
overage detection have a typed substrate.

Pricing data shipped here is a documented, dated, *non-authoritative* curated
seed. The user-provided plans in ``polylogue.toml`` always win and are tagged
``source = "user-config"``. The curated seed is tagged
``source = "polylogue-curated-seed"`` with a visible effective date and
notice. Nothing in this module scrapes vendor pages or calls billing APIs.

The cycle anchor is a day-of-month integer (1-28 to avoid month-length edge
cases at 29/30/31). ``cycle_for(plan, now)`` returns the half-open
``[cycle_start, cycle_end)`` instants in UTC ISO-8601.

Quota basis is optional. A plan without a quota produces outlook rows
*without* fabricated quota numbers — the cost cluster must treat absence as
absence, not as zero remaining capacity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from polylogue.core.errors import PolylogueError

PlanSource = Literal["user-config", "polylogue-curated-seed", "vendor-published", "inferred"]
USER_CONFIG_SOURCE: PlanSource = "user-config"
CURATED_SEED_SOURCE: PlanSource = "polylogue-curated-seed"

CURATED_SEED_EFFECTIVE_DATE = "2026-05-17"
CURATED_SEED_NOTICE = (
    "Non-authoritative curated seed. Vendor prices/quotas change without notice; "
    "override in polylogue.toml under [[cost.subscription.plans]] to supersede."
)


class OverageRule(str, Enum):
    """How the plan behaves once the included quota is exhausted."""

    block = "block"
    metered = "metered"
    soft = "soft"
    unknown = "unknown"


class QuotaBasis(str, Enum):
    """Unit of the included quota for a plan."""

    credits = "credits"
    tokens = "tokens"
    messages = "messages"
    usd = "usd"


class PlanLookupError(PolylogueError):
    """Raised when a user references an unknown plan name."""


class SubscriptionPlan(BaseModel):
    """Typed subscription plan with cycle, quota, and overage rules.

    All durations and prices are explicit; no implicit unit conversions. The
    model is frozen so plan instances flow safely through the cost rollup
    pipeline without surprise mutation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Stable identifier, e.g. 'claude-pro'.")
    provider: str = Field(min_length=1, description="Origin lab/product, e.g. 'anthropic'.")
    display_name: str = Field(min_length=1, description="Human-readable plan label.")
    monthly_cost_usd: float = Field(ge=0.0, description="Recurring monthly fee in USD.")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    billing_cycle_days: int = Field(default=30, ge=1, le=366)
    cycle_anchor_day: int | None = Field(
        default=None,
        ge=1,
        le=28,
        description=(
            "Day of month the cycle resets (1-28 to avoid month-length edge cases). "
            "None means the plan has no fixed monthly anchor."
        ),
    )
    quota: float | None = Field(default=None, ge=0.0, description="Included quota per cycle.")
    quota_basis: QuotaBasis | None = Field(default=None)
    overage_rule: OverageRule = OverageRule.unknown
    overage_rate_usd_per_unit: float | None = Field(
        default=None,
        ge=0.0,
        description="Per-unit overage rate in USD, in the same unit as quota_basis.",
    )
    source: PlanSource = USER_CONFIG_SOURCE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    effective_date: str = Field(default=CURATED_SEED_EFFECTIVE_DATE)
    notice: str = ""

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _coherent_quota(self) -> SubscriptionPlan:
        if (self.quota is None) != (self.quota_basis is None):
            raise ValueError("quota and quota_basis must be provided together (or both omitted)")
        if self.overage_rate_usd_per_unit is not None and self.overage_rule not in (
            OverageRule.metered,
            OverageRule.soft,
        ):
            raise ValueError("overage_rate_usd_per_unit only meaningful for metered/soft overage rules")
        return self


def cycle_for(plan: SubscriptionPlan, now: datetime) -> tuple[str, str] | None:
    """Return the half-open ``[start, end)`` ISO-8601 instants for the cycle
    containing ``now`` for ``plan``.

    Returns ``None`` for plans that do not declare a ``cycle_anchor_day`` —
    callers must surface "no cycle window" rather than fabricate one.
    """
    if plan.cycle_anchor_day is None:
        return None
    anchor = plan.cycle_anchor_day
    now_utc = now.astimezone(UTC)
    candidate = now_utc.replace(day=anchor, hour=0, minute=0, second=0, microsecond=0)
    if candidate > now_utc:
        # Roll back one month.
        prev_month_end = candidate.replace(day=1) - timedelta(days=1)
        candidate = prev_month_end.replace(day=anchor, hour=0, minute=0, second=0, microsecond=0)
    end = candidate + timedelta(days=plan.billing_cycle_days)
    return (candidate.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z"))


# ---------------------------------------------------------------------------
# Curated seed of well-known plans (NOT vendor-authoritative).
# ---------------------------------------------------------------------------


def _seed(**kwargs: object) -> SubscriptionPlan:
    payload: dict[str, object] = {
        "source": CURATED_SEED_SOURCE,
        "confidence": 0.6,
        "effective_date": CURATED_SEED_EFFECTIVE_DATE,
        "notice": CURATED_SEED_NOTICE,
    }
    payload.update(kwargs)
    return SubscriptionPlan.model_validate(payload)


WELL_KNOWN_PLANS: Mapping[str, SubscriptionPlan] = {
    plan.name: plan
    for plan in (
        _seed(
            name="claude-pro",
            provider="anthropic",
            display_name="Claude Pro",
            monthly_cost_usd=20.0,
            quota=21_700_000,
            quota_basis=QuotaBasis.credits,
            overage_rule=OverageRule.block,
        ),
        _seed(
            name="claude-max-5x",
            provider="anthropic",
            display_name="Claude Max 5x",
            monthly_cost_usd=100.0,
            quota=180_600_000,
            quota_basis=QuotaBasis.credits,
            overage_rule=OverageRule.block,
        ),
        _seed(
            name="claude-max-20x",
            provider="anthropic",
            display_name="Claude Max 20x",
            monthly_cost_usd=200.0,
            quota=361_100_000,
            quota_basis=QuotaBasis.credits,
            overage_rule=OverageRule.block,
        ),
        _seed(
            name="chatgpt-plus",
            provider="openai",
            display_name="ChatGPT Plus",
            monthly_cost_usd=20.0,
            quota=None,
            quota_basis=None,
            overage_rule=OverageRule.soft,
        ),
        _seed(
            name="chatgpt-pro",
            provider="openai",
            display_name="ChatGPT Pro",
            monthly_cost_usd=200.0,
            quota=None,
            quota_basis=None,
            overage_rule=OverageRule.soft,
        ),
        _seed(
            name="github-copilot-pro",
            provider="github",
            display_name="GitHub Copilot Pro",
            monthly_cost_usd=10.0,
            quota=None,
            quota_basis=None,
            overage_rule=OverageRule.unknown,
        ),
        _seed(
            name="gemini-advanced",
            provider="google",
            display_name="Gemini Advanced (Google One AI Premium)",
            monthly_cost_usd=20.0,
            quota=None,
            quota_basis=None,
            overage_rule=OverageRule.unknown,
        ),
    )
}


# ---------------------------------------------------------------------------
# Lookup / load helpers
# ---------------------------------------------------------------------------


def plan_by_name(name: str, *, plans: Mapping[str, SubscriptionPlan] | None = None) -> SubscriptionPlan:
    """Return the plan for ``name`` from ``plans`` (or the curated seed).

    Raises :class:`PlanLookupError` (typed) when the name is unknown — the
    cost CLI must surface this rather than silently defaulting.
    """
    catalogue = plans if plans is not None else WELL_KNOWN_PLANS
    if name not in catalogue:
        known = ", ".join(sorted(catalogue)) or "(none configured)"
        raise PlanLookupError(f"Unknown subscription plan {name!r}. Known plans: {known}.")
    return catalogue[name]


def load_plans(rows: Iterable[Mapping[str, object]] | None) -> dict[str, SubscriptionPlan]:
    """Build a plan catalogue from user-supplied rows merged with the curated seed.

    User rows always carry ``source = "user-config"`` and override seed entries
    with the same ``name``. Rows missing required fields raise a typed
    ``PlanLookupError`` so config bugs surface at load time.
    """
    catalogue: dict[str, SubscriptionPlan] = dict(WELL_KNOWN_PLANS)
    if not rows:
        return catalogue
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise PlanLookupError(f"Plan entry must be a table, got {type(raw).__name__}.")
        data: dict[str, object] = dict(raw)
        data["source"] = USER_CONFIG_SOURCE
        data.setdefault("confidence", 1.0)
        try:
            plan = SubscriptionPlan.model_validate(data)
        except (TypeError, ValueError) as exc:
            name = data.get("name", "<unnamed>")
            raise PlanLookupError(f"Invalid subscription plan {name!r}: {exc}") from exc
        catalogue[plan.name] = plan
    return catalogue


def resolve_plan(
    name: str | None,
    *,
    user_rows: Iterable[Mapping[str, object]] | None = None,
) -> SubscriptionPlan | None:
    """Resolve ``name`` against the merged plan catalogue.

    Returns ``None`` when ``name`` is ``None`` (no plan declared). Raises
    :class:`PlanLookupError` when the name is non-empty but unknown.
    """
    if name is None:
        return None
    catalogue = load_plans(user_rows)
    return plan_by_name(name, plans=catalogue)


__all__ = [
    "CURATED_SEED_EFFECTIVE_DATE",
    "CURATED_SEED_NOTICE",
    "CURATED_SEED_SOURCE",
    "OverageRule",
    "PlanLookupError",
    "PlanSource",
    "QuotaBasis",
    "SubscriptionPlan",
    "USER_CONFIG_SOURCE",
    "WELL_KNOWN_PLANS",
    "cycle_for",
    "load_plans",
    "plan_by_name",
    "resolve_plan",
]
