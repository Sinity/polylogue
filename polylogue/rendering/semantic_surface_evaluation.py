"""Semantic surface contract evaluation primitives."""

from __future__ import annotations

from typing import Any

from polylogue.rendering.semantic_proof_models import SemanticMetricCheck
from polylogue.rendering.semantic_surface_models import SemanticMetricContract


def transform_contract_value(
    facts: dict[str, Any],
    key: str | None,
    transform: str,
    *,
    default: Any = None,
) -> Any:
    value = default if key is None else facts.get(key, default)
    if transform == "identity":
        return value
    if transform == "presence_count":
        return 1 if value else 0
    if transform == "presence_bool":
        return bool(value)
    if transform == "date_prefix10":
        return str(value)[:10] if value else None
    if transform == "id_prefix24":
        return str(value or "")[:24]
    if transform == "summary_title_projection":
        raw_title = facts.get("title") or str(facts.get("conversation_id") or "")[:20]
        raw_title = str(raw_title)
        return f"{raw_title[:47]}..." if len(raw_title) > 50 else raw_title
    if transform == "len":
        if value is None:
            return 0
        return len(value)
    if transform == "bool":
        return bool(value)
    raise ValueError(f"Unsupported semantic contract transform: {transform}")


def evaluate_contracts(
    contracts: tuple[SemanticMetricContract, ...],
    input_facts: dict[str, Any],
    output_facts: dict[str, Any],
) -> list[SemanticMetricCheck]:
    checks: list[SemanticMetricCheck] = []
    for contract in contracts:
        input_value = transform_contract_value(
            input_facts,
            contract.input_key,
            contract.input_transform,
        )
        output_value = transform_contract_value(
            output_facts,
            contract.output_key,
            contract.output_transform,
            default=contract.default_output,
        )
        if contract.mode == "preserve":
            status = "preserved" if input_value == output_value else "critical_loss"
        elif contract.mode == "declared_loss":
            status = "declared_loss" if input_value else "preserved"
        elif contract.mode == "presence":
            status = "preserved" if bool(input_value) == bool(output_value) else "critical_loss"
        else:
            raise ValueError(f"Unsupported semantic contract mode: {contract.mode}")
        checks.append(
            SemanticMetricCheck(
                metric=contract.metric,
                status=status,
                policy=contract.policy,
                input_value=input_value,
                output_value=output_value,
            )
        )
    return checks


__all__ = ["evaluate_contracts", "transform_contract_value"]
