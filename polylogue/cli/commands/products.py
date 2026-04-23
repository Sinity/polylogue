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
from polylogue.cli.machine_errors import emit_success
from polylogue.cli.product_command_contracts import ProductCommandRequest, query_model_field_names
from polylogue.cli.types import AppEnv
from polylogue.product_readiness import ProductReadinessQuery, ProductReadinessReport, known_product_readiness_names
from polylogue.products.registry import (
    PRODUCT_REGISTRY,
    ProductQueryError,
    ProductType,
    fetch_products,
    render_product_items,
)
from polylogue.sync_bridge import run_coroutine_sync

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


def _make_callback(pt: ProductType) -> Callable[..., None]:
    """Create the Click callback for a product type command.

    Inherits ``provider``, ``since``, and ``until`` from the root CLI
    context when the product's query class accepts them.
    """
    # Pre-resolve accepted fields so we only inject keys the query class understands.
    accepted_root_keys = tuple(key for key in _ROOT_FILTER_KEYS if key in query_model_field_names(pt))

    @click.pass_context
    def callback(
        ctx: click.Context,
        /,
        json_mode: bool = False,
        output_format: str | None = None,
        **kwargs: object,
    ) -> None:
        env: AppEnv = ctx.obj
        request = ProductCommandRequest.from_context(
            ctx,
            pt,
            json_mode=json_mode,
            output_format=output_format,
            kwargs=kwargs,
            inherited_root_keys=accepted_root_keys,
        )

        try:
            items = fetch_products(pt, env.operations, **request.query_kwargs)
        except (ArchiveProductUnavailableError, ProductQueryError) as exc:
            fail(f"products {pt.resolved_cli_command_name}", str(exc))
        render_product_items(items, pt, json_mode=request.wants_json)

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


def _status_wants_json(ctx: click.Context, *, json_mode: bool, output_format: str | None) -> bool:
    if json_mode or output_format == "json":
        return True
    root_output = ctx.find_root().params.get("output_format")
    return root_output == "json"


def _render_status_plain(report: ProductReadinessReport) -> None:
    click.echo(f"Product Readiness: {report.aggregate_verdict}")
    click.echo(f"Total conversations: {report.total_conversations}")
    if report.provider or report.since or report.until:
        click.echo(f"Scope: provider={report.provider or '-'} since={report.since or '-'} until={report.until or '-'}")
    click.echo("")
    for product in report.products:
        expected = f" expected={product.expected_row_count}" if product.expected_row_count is not None else ""
        click.echo(f"{product.product_name}: {product.verdict} rows={product.row_count}{expected}")
        if product.missing_count or product.stale_count or product.orphan_count or product.legacy_incompatible_count:
            click.echo(
                "  "
                f"missing={product.missing_count} stale={product.stale_count} "
                f"orphan={product.orphan_count} legacy={product.legacy_incompatible_count}"
            )
        if product.ready_flags:
            flags = ", ".join(f"{key}={value}" for key, value in sorted(product.ready_flags.items()))
            click.echo(f"  flags: {flags}")
        if product.provider_coverage:
            providers = ", ".join(
                f"{coverage.provider_name}={coverage.row_count}" for coverage in product.provider_coverage
            )
            click.echo(f"  providers: {providers}")
        if product.version_coverage:
            versions = ", ".join(f"{coverage.field}={dict(coverage.versions)}" for coverage in product.version_coverage)
            click.echo(f"  versions: {versions}")
        if product.schema_contract_issues:
            click.echo(f"  schema: {', '.join(product.schema_contract_issues)}")


@products_command.command("status")
@click.option("--product", "products", multiple=True, help="Product readiness target. May be repeated.")
@click.option("--provider", default=None, help="Limit provider coverage details to one provider.")
@click.option("--since", default=None, help="Limit coverage details to rows at/after this timestamp or date.")
@click.option("--until", default=None, help="Limit coverage details to rows at/before this timestamp or date.")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON.")
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.pass_context
def products_status_command(
    ctx: click.Context,
    products: tuple[str, ...],
    provider: str | None,
    since: str | None,
    until: str | None,
    json_mode: bool,
    output_format: str | None,
) -> None:
    """Report product materialization coverage and readiness."""
    env: AppEnv = ctx.obj
    root_params = ctx.find_root().params
    inherited_provider = provider if provider is not None else root_params.get("provider")
    inherited_since = since if since is not None else root_params.get("since")
    inherited_until = until if until is not None else root_params.get("until")
    try:
        query = ProductReadinessQuery(
            products=products,
            provider=str(inherited_provider) if inherited_provider is not None else None,
            since=str(inherited_since) if inherited_since is not None else None,
            until=str(inherited_until) if inherited_until is not None else None,
        )
        report = run_coroutine_sync(env.operations.get_product_readiness_report(query))
    except ValueError as exc:
        valid = ", ".join(known_product_readiness_names())
        fail("products status", f"{exc}. Known products: {valid}")
    if _status_wants_json(ctx, json_mode=json_mode, output_format=output_format):
        emit_success(report.model_dump(mode="json"))
        return
    _render_status_plain(report)


# Register all product types as subcommands
for _pt in PRODUCT_REGISTRY.values():
    if _pt.query_model is not None and _pt.operations_method_name:
        products_command.add_command(_build_product_command(_pt))


__all__ = ["products_command"]
