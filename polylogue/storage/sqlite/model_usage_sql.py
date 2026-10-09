"""Shared canonical usage dollar expressions for DDL and read owners."""

from __future__ import annotations

# Correlated and grouped session-cost readers use the same priced population.
# The alias is fixed to the model-usage relation named by those SQL owners.
MODEL_USAGE_CATALOG_SUM_SQL = """
CASE WHEN MIN(u.provider_lanes_complete) = 1
          AND MIN(u.provider_usage_observed OR u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens > 0) = 1
          AND COUNT(u.catalog_cost_usd) = COUNT(CASE
              WHEN u.provider_usage_observed OR u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens > 0
                   OR u.catalog_cost_usd IS NOT NULL THEN u.model_name END)
     THEN SUM(u.catalog_cost_usd) END
""".strip()


def model_usage_cost_estimated_sql(reported_cost_sql: str) -> str:
    """Classify the selected dollar basis; absence is not an exact zero.

    The caller supplies its session's reported amount expression. Per-model
    provider dollars outrank it, and only the canonical complete catalog sum
    can certify an estimated amount. Alias ``u`` is the existing usage owner.
    """
    return f"""
CASE WHEN SUM(u.provider_cost_usd) IS NOT NULL OR {reported_cost_sql} IS NOT NULL THEN 0
     WHEN {MODEL_USAGE_CATALOG_SUM_SQL} IS NOT NULL THEN 1
     ELSE NULL END
""".strip()
