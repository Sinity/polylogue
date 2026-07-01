"""Archive debt command group."""

from __future__ import annotations

import click

from polylogue.operations.archive_debt import archive_debt_list
from polylogue.paths import active_index_db_path
from polylogue.surfaces.payloads import ArchiveDebtListPayload

_DEBT_KINDS = ("archive-tier", "assertion-candidate", "convergence", "embedding", "fts", "raw-materialization")
_DEBT_STATUSES = ("open", "actionable", "blocked", "classified")


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
@click.option(
    "--status",
    "statuses",
    type=click.Choice(_DEBT_STATUSES),
    multiple=True,
    help="Restrict rows to one debt status. Repeatable.",
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
    statuses: tuple[str, ...],
    only_actionable: bool,
    limit: int | None,
    exact_fts: bool,
    output_format: str,
) -> None:
    """List archive debt across assertions, tiers, raw ingest, convergence, embedding, and FTS signals."""
    archive_root = active_index_db_path().parent
    payload = archive_debt_list(
        archive_root=archive_root,
        kinds=kinds,
        statuses=statuses,
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
        f"actionable={payload.totals.actionable} blocked={payload.totals.blocked} "
        f"classified={payload.totals.classified}"
    )
    if payload.totals.affected_total:
        click.echo(
            f"  affected={payload.totals.affected_total} "
            f"affected_warning={payload.totals.affected_warning} "
            f"affected_actionable={payload.totals.affected_actionable} "
            f"affected_open={payload.totals.affected_open} "
            f"affected_blocked={payload.totals.affected_blocked} "
            f"affected_classified={payload.totals.affected_classified}"
        )
    if not payload.rows:
        click.echo("  No archive debt detected.")
        return
    for row in payload.rows:
        click.echo(f"\n[{row.severity}] {row.debt_ref}")
        click.echo(f"  {row.summary}")
        click.echo(f"  kind={row.kind} stage={row.stage} subject={row.subject_ref} status={row.status}")
        if row.affected_count is not None:
            click.echo(f"  affected_count={row.affected_count}")
        if row.details:
            click.echo(f"  detail: {row.details}")
        for evidence_ref in row.evidence_refs:
            click.echo(f"  evidence: {evidence_ref}")
        for caveat in row.caveats:
            click.echo(f"  caveat: {caveat}")
        for action in row.actions:
            command = " ".join(action.command)
            click.echo(f"  action: {action.label}" + (f" ({command})" if command else ""))


__all__ = ["debt_command"]
