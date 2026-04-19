"""Archive data product inspection commands — registry-driven.

Product commands inherit ``--provider``, ``--since``, and ``--until`` from
the root CLI context so that ``polylogue --provider codex products profiles``
works without re-specifying the filter on the subcommand.
"""

from __future__ import annotations

from collections.abc import Callable

import click

from polylogue.archive_products import ArchiveProductUnavailableError
from polylogue.cli.helper_support import fail
from polylogue.cli.types import AppEnv
from polylogue.products.registry import (
    PRODUCT_REGISTRY,
    ProductQueryError,
    ProductType,
    fetch_products,
    render_product_items,
)

# Root-level filter keys that product commands inherit from the parent context.
_ROOT_FILTER_KEYS = ("provider", "since", "until")


def _build_click_params(pt: ProductType) -> list[click.Parameter]:
    """Build Click Option parameters from a product type's cli_options."""
    params: list[click.Parameter] = []

    for opt in pt.cli_options:
        params.append(
            click.Option(
                opt.flags,
                help=opt.help,
                type=opt.type,
                default=opt.default,
                show_default=opt.show_default,
                is_flag=opt.is_flag,
            )
        )

    # Standard options on every product command
    params.append(
        click.Option(
            ("--limit",),
            type=int,
            default=pt.mcp_default_limit,
            show_default=True,
            help="Maximum rows",
        )
    )
    params.append(
        click.Option(
            ("--offset",),
            type=int,
            default=0,
            show_default=True,
            help="Start offset",
        )
    )
    params.append(
        click.Option(
            ("--json", "json_mode"),
            is_flag=True,
            help="Output as JSON",
        )
    )
    params.append(
        click.Option(
            ("--format", "output_format"),
            type=click.Choice(["json"]),
            default=None,
            help="Output format",
        )
    )

    return params


def _find_root_params(ctx: click.Context) -> dict[str, object]:
    """Walk up the context chain to find the root CLI group's params."""
    cur = ctx
    while cur.parent is not None:
        cur = cur.parent
    return cur.params


def _make_callback(pt: ProductType) -> Callable[..., None]:
    """Create the Click callback for a product type command.

    Inherits ``provider``, ``since``, and ``until`` from the root CLI
    context when the product's query class accepts them.
    """
    # Pre-resolve accepted fields so we only inject keys the query class understands.
    query_model = pt.query_model
    accepted_root_keys = frozenset(
        key for key in _ROOT_FILTER_KEYS if query_model is not None and key in query_model.model_fields
    )

    @click.pass_context
    def callback(
        ctx: click.Context,
        /,
        json_mode: bool = False,
        output_format: str | None = None,
        **kwargs: object,
    ) -> None:
        env: AppEnv = ctx.obj

        # Inject root filter values the product query class can accept.
        root_params = _find_root_params(ctx)
        for key in accepted_root_keys:
            if kwargs.get(key) is None:
                root_val = root_params.get(key)
                if root_val is not None:
                    kwargs[key] = root_val

        if output_format is None:
            root_output_format = root_params.get("output_format")
            if isinstance(root_output_format, str):
                output_format = root_output_format

        try:
            items = fetch_products(pt, env.operations, **kwargs)
        except (ArchiveProductUnavailableError, ProductQueryError) as exc:
            fail(f"products {pt.resolved_cli_command_name}", str(exc))
        render_product_items(items, pt, json_mode=(json_mode or output_format == "json"))

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
    if _pt.query_model is not None and _pt.operations_method_name:
        products_command.add_command(_build_product_command(_pt))


__all__ = ["products_command"]
