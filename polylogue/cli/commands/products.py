"""Archive data product inspection commands — registry-driven."""

from __future__ import annotations

from typing import Any

import click

from polylogue.cli.helper_support import fail
from polylogue.cli.types import AppEnv
from polylogue.products.registry import (
    PRODUCT_REGISTRY,
    ProductQueryError,
    ProductType,
    fetch_products,
    render_product_items,
)


def _build_click_params(pt: ProductType) -> list[click.Parameter]:
    """Build Click Option parameters from a product type's cli_options."""
    params: list[click.Parameter] = []

    for opt in pt.cli_options:
        kwargs: dict[str, Any] = {"help": opt.help}
        if opt.type is not None:
            kwargs["type"] = opt.type
        if opt.default is not None:
            kwargs["default"] = opt.default
        else:
            kwargs["default"] = None
        if opt.show_default:
            kwargs["show_default"] = True
        if opt.is_flag:
            kwargs["is_flag"] = True

        params.append(click.Option(
            opt.flags,
            **kwargs,
        ))

    # Standard options on every product command
    params.append(click.Option(
        ("--limit",),
        type=int,
        default=pt.mcp_default_limit,
        show_default=True,
        help="Maximum rows",
    ))
    params.append(click.Option(
        ("--offset",),
        type=int,
        default=0,
        show_default=True,
        help="Start offset",
    ))
    params.append(click.Option(
        ("--json", "json_mode"),
        is_flag=True,
        help="Output as JSON",
    ))

    return params


def _make_callback(pt: ProductType):
    """Create the Click callback for a product type command."""

    @click.pass_obj
    def callback(env: AppEnv, json_mode: bool = False, **kwargs: Any) -> None:
        try:
            items = fetch_products(pt, env.operations, **kwargs)
        except ProductQueryError as exc:
            fail(f"products {pt.resolved_cli_command_name}", str(exc))
        render_product_items(items, pt, json_mode=json_mode)

    return callback


def _build_product_command(pt: ProductType) -> click.Command:
    """Build a Click command for a registered product type."""
    return click.Command(
        name=pt.resolved_cli_command_name,
        callback=_make_callback(pt),
        params=_build_click_params(pt),
        help=pt.cli_help or f"List {pt.display_name.lower()}.",
    )


@click.group("products")
def products_command() -> None:
    """Inspect durable archive data products."""


# Register all product types as subcommands
for _pt in PRODUCT_REGISTRY.values():
    if _pt.query_class_path and _pt.operations_method:
        products_command.add_command(_build_product_command(_pt))


__all__ = ["products_command"]
