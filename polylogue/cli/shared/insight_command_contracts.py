"""Typed request helpers for archive-insight CLI commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.archive.query.spec import QuerySpecError, parse_query_date, split_csv
from polylogue.insights.registry import InsightType

if TYPE_CHECKING:
    import click


def find_root_params(ctx: click.Context) -> Mapping[str, object]:
    """Walk up the Click context chain to the root query command params."""
    current = ctx
    while current.parent is not None:
        current = current.parent
    return current.params


def query_model_field_names(insight_type: InsightType) -> frozenset[str]:
    """Return the accepted query-model fields for an insight type."""
    query_model = insight_type.query_model
    if query_model is None:
        return frozenset()
    return frozenset(query_model.model_fields)


class InsightCommandInputError(ValueError):
    """Invalid insight-command filter input."""


def _normalize_product_provider(value: object) -> str | None:
    providers = split_csv(value)
    if not providers:
        return None
    if len(providers) > 1:
        joined = ", ".join(providers)
        raise InsightCommandInputError(f"insights commands accept one provider, got: {joined}")
    return providers[0]


def _normalize_product_date(field: str, value: object) -> str | None:
    if value is None:
        return None
    try:
        parsed = parse_query_date(field, str(value))
    except QuerySpecError as exc:
        raise InsightCommandInputError(str(exc)) from exc
    return parsed.isoformat() if parsed is not None else None


def normalize_insight_query_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Normalize insight query filters shared by generic/status/export commands."""
    normalized = dict(kwargs)
    if "provider" in normalized:
        normalized["provider"] = _normalize_product_provider(normalized.get("provider"))
    for field in ("since", "until"):
        if field in normalized:
            normalized[field] = _normalize_product_date(field, normalized.get(field))
    return normalized


@dataclass(frozen=True, slots=True)
class InsightCommandRequest:
    """Normalized insight command request shared by all insight subcommands."""

    query_kwargs: dict[str, object]
    output_format: str | None

    @classmethod
    def from_context(
        cls,
        ctx: click.Context,
        insight_type: InsightType,
        *,
        output_format: str | None,
        kwargs: Mapping[str, object],
        inherited_root_keys: tuple[str, ...],
    ) -> InsightCommandRequest:
        root_params = find_root_params(ctx)
        accepted_root_keys = query_model_field_names(insight_type)
        normalized_kwargs = dict(kwargs)

        for key in inherited_root_keys:
            if key not in accepted_root_keys:
                continue
            if normalized_kwargs.get(key) is not None:
                continue
            inherited_value = root_params.get(key)
            if inherited_value is not None:
                normalized_kwargs[key] = inherited_value

        resolved_output_format = output_format
        if resolved_output_format is None:
            root_output_format = root_params.get("output_format")
            if isinstance(root_output_format, str):
                resolved_output_format = root_output_format

        return cls(
            query_kwargs=normalize_insight_query_kwargs(normalized_kwargs),
            output_format=resolved_output_format,
        )

    @property
    def wants_json(self) -> bool:
        return self.output_format == "json"


__all__ = [
    "find_root_params",
    "normalize_insight_query_kwargs",
    "InsightCommandInputError",
    "InsightCommandRequest",
    "query_model_field_names",
]
