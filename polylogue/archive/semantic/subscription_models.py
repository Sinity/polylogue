"""Subscription cycle and usage outlook typed models (#870).

All subscription quota math is estimated and not vendor-authoritative.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PlanConfidence(str, Enum):
    configured = "configured"
    derived = "derived"
    unknown = "unknown"


class AnomalyPayload(BaseModel):
    date: str
    cost_usd: float
    comparator_window_avg: float
    threshold_multiplier: float = 2.0
    affected_component: str = "daily_cost"
    explanation: str = ""


class ModelBreakdownPayload(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    api_equivalent_usd: float = 0.0
    subscription_equivalent_usd: float = 0.0
    session_count: int = 0


class SubscriptionPlanConfig(BaseModel):
    plan_name: str = "pro"
    monthly_credit_pool: float = 0.0
    monthly_api_budget_usd: float | None = None
    cycle_start_day: int = 1


class UsageOutlookPayload(BaseModel):
    cycle_start: str = ""
    cycle_end: str = ""
    plan_name: str = "unknown"
    plan_confidence: PlanConfidence = PlanConfidence.unknown
    credits_total: float = 0.0
    credits_used: float = 0.0
    credits_remaining: float = 0.0
    burn_rate_credits_per_day: float = 0.0
    projected_exhaustion_date: str | None = None
    api_equivalent_usd_total: float = 0.0
    subscription_equivalent_usd_total: float = 0.0
    per_model: list[ModelBreakdownPayload] = Field(default_factory=list)
    anomalies: list[AnomalyPayload] = Field(default_factory=list)
    priced_session_count: int = 0
    unavailable_session_count: int = 0
    confidence: float = 1.0
    coverage_pct: float = 100.0
    projection_method: str = "trailing_30d_linear"


class SubscriptionCycle(BaseModel):
    plan: SubscriptionPlanConfig = Field(default_factory=SubscriptionPlanConfig)
    outlook: UsageOutlookPayload = Field(default_factory=UsageOutlookPayload)
    generated_at: str = ""


class CostOutlookRequest(BaseModel):
    plan_name: str | None = None
    cycle_start: str | None = None
    anomaly_threshold: float = 2.0
