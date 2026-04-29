"""Typed request helpers for archive-product CLI commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.products.registry import ProductType

if TYPE_CHECKING:
    import click


def find_root_params(ctx: click.Context) -> Mapping[str, object]:
    """Walk up the Click context chain to the root query command params."""
    current = ctx
    while current.parent is not None:
        current = current.parent
    return current.params


def query_model_field_names(product_type: ProductType) -> frozenset[str]:
    """Return the accepted query-model fields for a product type."""
    query_model = product_type.query_model
    if query_model is None:
        return frozenset()
    return frozenset(query_model.model_fields)


@dataclass(frozen=True, slots=True)
class ProductCommandRequest:
    """Normalized product command request shared by all product subcommands."""

    query_kwargs: dict[str, object]
    output_format: str | None
    json_mode: bool

    @classmethod
    def from_context(
        cls,
        ctx: click.Context,
        product_type: ProductType,
        *,
        json_mode: bool,
        output_format: str | None,
        kwargs: Mapping[str, object],
        inherited_root_keys: tuple[str, ...],
    ) -> ProductCommandRequest:
        root_params = find_root_params(ctx)
        accepted_root_keys = query_model_field_names(product_type)
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
            query_kwargs=normalized_kwargs,
            output_format=resolved_output_format,
            json_mode=json_mode,
        )

    @property
    def wants_json(self) -> bool:
        return self.json_mode or self.output_format == "json"


__all__ = [
    "find_root_params",
    "ProductCommandRequest",
    "query_model_field_names",
]
