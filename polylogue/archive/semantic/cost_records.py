"""Typed cost read models for the cost tracking substrate (#803)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

TokenProvenance = Literal["provider_reported", "tokenizer_estimated", "heuristic_estimated", "unknown"]
CostConfidence = Literal["reported", "estimated", "partial", "unknown"]
CostBasis = Literal["api_billed", "api_equivalent_estimated", "subscription_equivalent_estimated", "configured_manual", "unknown"]


class SessionCostBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)
    normalized_model: str | None = None
    provider_model_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    api_cost_usd: float = 0.0
    credit_cost: float = 0.0
    subscription_equivalent_usd: float = 0.0
    confidence: CostConfidence = "unknown"
    provenance: TokenProvenance = "unknown"


class SessionCostSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_api_cost_usd: float = 0.0
    total_credit_cost: float = 0.0
    total_subscription_equivalent_usd: float = 0.0
    cost_provenance: str = "unknown"
    cost_confidence: str = "unknown"
    tokenizer_version: str | None = None
    price_snapshot_version: str | None = None
    per_model: tuple[SessionCostBreakdown, ...] = ()


__all__ = [
    "CostBasis",
    "CostConfidence",
    "SessionCostBreakdown",
    "SessionCostSummary",
    "TokenProvenance",
]
