"""Primitive semantic check builders for proof surfaces."""

from __future__ import annotations

from typing import Any

from polylogue.rendering.semantic_proof_models import SemanticMetricCheck


def _critical_or_preserved(
    *,
    metric: str,
    policy: str,
    input_value: Any,
    output_value: Any,
) -> SemanticMetricCheck:
    return SemanticMetricCheck(
        metric=metric,
        status="preserved" if input_value == output_value else "critical_loss",
        policy=policy,
        input_value=input_value,
        output_value=output_value,
    )


def _declared_loss_or_preserved(
    *,
    metric: str,
    policy: str,
    input_value: int,
    output_value: Any = 0,
) -> SemanticMetricCheck:
    return SemanticMetricCheck(
        metric=metric,
        status="declared_loss" if input_value else "preserved",
        policy=policy,
        input_value=input_value,
        output_value=output_value,
    )


def _presence_check(
    *,
    metric: str,
    policy: str,
    input_value: object,
    output_present: bool,
) -> SemanticMetricCheck:
    expected_present = input_value is not None
    return SemanticMetricCheck(
        metric=metric,
        status="preserved" if expected_present == output_present else "critical_loss",
        policy=policy,
        input_value=input_value,
        output_value=output_present,
    )


__all__ = [
    "_critical_or_preserved",
    "_declared_loss_or_preserved",
    "_presence_check",
]
