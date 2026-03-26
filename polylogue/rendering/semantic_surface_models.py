"""Semantic surface contract models and declaration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SemanticMetricContract:
    metric: str
    mode: str
    policy: str
    input_key: str
    output_key: str | None = None
    input_transform: str = "identity"
    output_transform: str = "identity"
    default_output: Any = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "mode": self.mode,
            "policy": self.policy,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "input_transform": self.input_transform,
            "output_transform": self.output_transform,
            "default_output": self.default_output,
        }


@dataclass(frozen=True, slots=True)
class SemanticSurfaceSpec:
    name: str
    category: str
    aliases: tuple[str, ...] = ()
    export_format: str | None = None
    stream_format: str | None = None
    contracts: tuple[SemanticMetricContract, ...] = ()


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
    "SemanticMetricContract",
    "SemanticSurfaceSpec",
    "declared_loss_contract",
    "presence_contract",
    "preserve_contract",
]
