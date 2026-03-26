"""Governance and status archive-product CLI rendering."""

from __future__ import annotations

import click

from polylogue.cli.machine_errors import emit_success
from polylogue.cli.products_rendering_support import emit_product_list


def render_products_status(
    *,
    status: dict[str, int | bool],
    debt_summary: dict[str, int],
    json_mode: bool,
) -> None:
    if json_mode:
        emit_success({"session_products": status, "archive_debt": debt_summary})
        return
    click.echo("Session Product Status:\n")
    for key, value in sorted(status.items()):
        click.echo(f"  {key}: {value}")
    click.echo("\nArchive Debt:\n")
    click.echo(f"  tracked_items: {debt_summary['tracked_items']}")
    click.echo(f"  actionable_items: {debt_summary['actionable_items']}")
    click.echo(f"  issue_rows: {debt_summary['issue_rows']}")


def render_maintenance_runs(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="maintenance_runs", items=items)
        return
    if not items:
        click.echo("No maintenance runs recorded.")
        return
    click.echo(f"Maintenance runs: {len(items)}\n")
    for item in items:
        targets = ", ".join(item.target_names) if item.target_names else "all selected"
        click.echo(
            f"  {item.executed_at} {item.mode} success={item.success} "
            f"preview={item.preview} targets={targets}"
        )


def render_archive_debt(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="archive_debt", items=items)
        return
    if not items:
        click.echo("No archive debt matched.")
        return
    click.echo(f"Archive debt: {len(items)}\n")
    for item in items:
        click.echo(
            f"  {item.debt_name} category={item.category} healthy={item.healthy} "
            f"issue_count={item.issue_count} destructive={item.destructive} "
            f"stage={item.governance_stage}"
        )
        click.echo(
            f"    target={item.maintenance_target} detail={item.detail}"
        )


__all__ = [
    "render_archive_debt",
    "render_maintenance_runs",
    "render_products_status",
]
