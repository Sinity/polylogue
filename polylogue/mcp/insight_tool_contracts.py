"""Typed registration contracts for registry-driven MCP product tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from polylogue.insights.registry import InsightType
from polylogue.mcp.query_contracts import MCPToolLimit, MCPToolOffset


def _sanitize_offset(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, str, bytes, bytearray)):
        return max(0, int(value))
    return 0


@dataclass(frozen=True, slots=True)
class InsightToolSignature:
    """Normalized dynamic signature for one product list MCP tool."""

    parameters: tuple[inspect.Parameter, ...]
    annotations: dict[str, object]
    kwdefaults: dict[str, object]


@dataclass(frozen=True, slots=True)
class InsightListToolSpec:
    """Registry-derived MCP list-tool contract for one product type."""

    name: str
    doc: str
    signature: InsightToolSignature

    @classmethod
    def from_insight_type(cls, insight_type: InsightType) -> InsightListToolSpec:
        query_model = insight_type.query_model
        if query_model is None:
            raise ValueError(f"Insight type {insight_type.name} does not declare a query model")

        parameters: list[inspect.Parameter] = []
        for field_name in sorted(query_model.model_fields):
            field_info = query_model.model_fields[field_name]
            if field_name == "limit":
                parameters.append(
                    inspect.Parameter(
                        field_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=insight_type.mcp_default_limit,
                        annotation=MCPToolLimit,
                    )
                )
                continue
            if field_name == "offset":
                parameters.append(
                    inspect.Parameter(
                        field_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=0,
                        annotation=MCPToolOffset,
                    )
                )
                continue
            parameters.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None if field_info.is_required() else field_info.get_default(call_default_factory=True),
                    annotation=field_info.annotation if field_info.annotation is not None else object,
                )
            )

        annotations = {parameter.name: parameter.annotation for parameter in parameters}
        annotations["return"] = str
        signature = InsightToolSignature(
            parameters=tuple(parameters),
            annotations=annotations,
            kwdefaults={parameter.name: parameter.default for parameter in parameters},
        )
        return cls(
            name=insight_type.name,
            doc=f"List {insight_type.display_name.lower()} from the archive.",
            signature=signature,
        )

    def normalize_kwargs(
        self,
        clamp_limit: Callable[[int | object], int],
        kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        normalized = dict(kwargs)
        if "limit" in normalized:
            normalized["limit"] = clamp_limit(normalized["limit"])
        if "offset" in normalized:
            normalized["offset"] = _sanitize_offset(normalized["offset"])
        return normalized


__all__ = ["InsightListToolSpec", "InsightToolSignature"]
