"""Helper constructors for semantic surface contract declarations."""

from __future__ import annotations

from typing import Any

from polylogue.rendering.semantic_surface_models import SemanticMetricContract


def preserve_contract(
    metric: str,
    policy: str,
    input_key: str,
    output_key: str | None = None,
    *,
    input_transform: str = "identity",
    output_transform: str = "identity",
) -> SemanticMetricContract:
    return SemanticMetricContract(
        metric=metric,
        mode="preserve",
        policy=policy,
        input_key=input_key,
        output_key=output_key or input_key,
        input_transform=input_transform,
        output_transform=output_transform,
    )


def declared_loss_contract(
    metric: str,
    policy: str,
    input_key: str,
    output_key: str | None = None,
    *,
    input_transform: str = "identity",
    output_transform: str = "identity",
    default_output: Any = 0,
) -> SemanticMetricContract:
    return SemanticMetricContract(
        metric=metric,
        mode="declared_loss",
        policy=policy,
        input_key=input_key,
        output_key=output_key,
        input_transform=input_transform,
        output_transform=output_transform,
        default_output=default_output,
    )


def presence_contract(
    metric: str,
    policy: str,
    input_key: str,
    output_key: str,
    *,
    input_transform: str = "presence_bool",
    output_transform: str = "identity",
) -> SemanticMetricContract:
    return SemanticMetricContract(
        metric=metric,
        mode="presence",
        policy=policy,
        input_key=input_key,
        output_key=output_key,
        input_transform=input_transform,
        output_transform=output_transform,
    )


__all__ = [
    "declared_loss_contract",
    "presence_contract",
    "preserve_contract",
]
