"""Cost-cluster typed config and helpers (#1132).

This package houses substrate-level cost configuration that does not depend on
the archive SQLite schema or insight pipelines. It is the first home for
typed subscription plan models so the per-cycle outlook, quota pressure, and
overage detection work tracked by #995 / #870 has a single typed source of
truth.

Nothing in this package talks to the network, the filesystem, or vendor
billing APIs. Pricing values are user-configurable; the curated seed shipped
here is explicitly marked non-authoritative.
"""

from __future__ import annotations

from polylogue.cost.plans import (
    CURATED_SEED_SOURCE,
    USER_CONFIG_SOURCE,
    WELL_KNOWN_PLANS,
    OverageRule,
    PlanLookupError,
    PlanSource,
    QuotaBasis,
    SubscriptionPlan,
    cycle_for,
    load_plans,
    plan_by_name,
    resolve_plan,
)

__all__ = [
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
