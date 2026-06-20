"""Archive debt command group."""

from __future__ import annotations

import click

from polylogue.operations.archive_debt import archive_debt_list
from polylogue.paths import active_index_db_path
from polylogue.surfaces.payloads import ArchiveDebtListPayload

_DEBT_KINDS = ("archive-tier", "assertion-candidate", "convergence", "embedding", "fts")


@click.group("debt")
def debt_command() -> None:
    """Inspect archive work that needs operator attention."""


@debt_command.command("list")
@click.option(
    "--kind",
    "kinds",
    type=click.Choice(_DEBT_KINDS),
    multiple=True,
    help="Restrict rows to one debt kind. Repeatable.",
)
@click.option("--only-actionable", is_flag=True, help="Only show rows with a direct operator action.")
@click.option("--limit", "-l", type=int, default=None, help="Maximum rows to emit.")
@click.option("--exact-fts", is_flag=True, help="Run exact FTS reconciliation counts.")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
def debt_list_command(
    kinds: tuple[str, ...],
    only_actionable: bool,
    limit: int | None,
    exact_fts: bool,
    output_format: str,
) -> None:
    """List archive debt across assertions, tiers, convergence, embedding, and FTS signals."""
    archive_root = active_index_db_path().parent
    payload = archive_debt_list(
        archive_root=archive_root,
        kinds=kinds,
        only_actionable=only_actionable,
        limit=limit,
        exact_fts=exact_fts,
    )
    if output_format == "json":
        click.echo(payload.to_json(exclude_none=True))
        return
    _render_text(payload)


def _render_text(payload: ArchiveDebtListPayload) -> None:
    click.echo(f"Archive debt: {payload.totals.total} row(s)")
    click.echo(
        f"  critical={payload.totals.critical} warning={payload.totals.warning} "
        f"actionable={payload.totals.actionable} blocked={payload.totals.blocked}"
    )
    if not payload.rows:
        click.echo("  No archive debt detected.")
        return
    for row in payload.rows:
        click.echo(f"\n[{row.severity}] {row.debt_ref}")
        click.echo(f"  {row.summary}")
        click.echo(f"  kind={row.kind} stage={row.stage} subject={row.subject_ref} status={row.status}")
        if row.details:
            click.echo(f"  detail: {row.details}")
        for action in row.actions:
            command = " ".join(action.command)
            click.echo(f"  action: {action.label}" + (f" ({command})" if command else ""))


__all__ = ["debt_command"]
