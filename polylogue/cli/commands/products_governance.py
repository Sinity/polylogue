"""Governance and readiness archive product commands."""

from __future__ import annotations

import click

from polylogue.cli.products_rendering import (
    render_archive_debt,
    render_maintenance_runs,
    render_products_status,
    summarize_archive_debt,
)
from polylogue.cli.products_workflow import (
    get_products_status,
    list_archive_debt_products,
    list_maintenance_run_products,
)
from polylogue.cli.types import AppEnv


def register_governance_product_commands(products_command: click.Group) -> None:
    products_command.add_command(products_status_command)
    products_command.add_command(products_maintenance_command)
    products_command.add_command(products_debt_command)


@click.command("status")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_status_command(env: AppEnv, json_mode: bool) -> None:
    """Show durable product readiness status."""
    status, debt_items = get_products_status(env)
    render_products_status(
        status=status,
        debt_summary=summarize_archive_debt(debt_items),
        json_mode=json_mode,
    )


@click.command("maintenance")
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum rows")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_maintenance_command(
    env: AppEnv,
    limit: int,
    json_mode: bool,
) -> None:
    """List durable maintenance preview/apply lineage."""
    items = list_maintenance_run_products(env, limit=limit)
    render_maintenance_runs(items, json_mode=json_mode)


@click.command("debt")
@click.option("--category", default=None, help="Limit to debt category")
@click.option("--actionable-only", is_flag=True, help="Show only debt items with outstanding issues")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_debt_command(
    env: AppEnv,
    category: str | None,
    actionable_only: bool,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List governed live archive debt items."""
    items = list_archive_debt_products(
        env,
        category=category,
        actionable_only=actionable_only,
        limit=limit,
        offset=offset,
    )
    render_archive_debt(items, json_mode=json_mode)


__all__ = ["register_governance_product_commands"]
