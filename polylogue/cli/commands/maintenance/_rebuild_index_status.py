"""``maintenance rebuild-index-status``: consolidated raw-replay rebuild status.

polylogue-b5l.1 AC5: one command reports lease ownership, the active
generation, the resumable transaction's cursor/delta, and explicit
stale-lock/failed-transaction recovery guidance -- see
``polylogue.maintenance.rebuild_index.rebuild_status`` for the assembled
payload this command renders. Entirely read-only.
"""

from __future__ import annotations

import json

import click

from polylogue.logging import configure_logging
from polylogue.paths import archive_root


@click.command("rebuild-index-status")
@click.option(
    "--operation-id",
    "operation_id",
    type=str,
    default=None,
    help=(
        "Rebuild transaction to report. Omit to resolve the daemon's own well-known "
        "bulk-rebuild operation id (the ops reset --index && polylogued run case)."
    ),
)
@click.option(
    "--no-daemon-fallback",
    "no_daemon_fallback",
    is_flag=True,
    help="Do not fall back to the daemon's well-known bulk-rebuild operation id when --operation-id is omitted.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def rebuild_index_status_command(
    operation_id: str | None,
    no_daemon_fallback: bool,
    output_format: str,
) -> None:
    """Report consolidated raw-replay rebuild status. Read-only."""
    from polylogue.maintenance.rebuild_index import rebuild_status

    configure_logging()
    root = archive_root()
    status = rebuild_status(root, operation_id=operation_id, include_daemon_bulk_rebuild=not no_daemon_fallback)

    if output_format == "json":
        click.echo(json.dumps(status, indent=2, sort_keys=True))
        return

    click.echo(f"Archive root: {status['archive_root']}")
    lease = status["lease"]
    assert isinstance(lease, dict)
    click.echo(
        f"Lease:        held={lease['held']} holder_pid={lease['holder_pid']} "
        f"holder_host={lease['holder_host']} holder_alive={lease['holder_alive']} stale={lease['stale']}"
    )
    generation = status["generation"]
    if isinstance(generation, dict):
        click.echo(
            f"Generation:   id={generation['generation_id']} state={generation['state']} "
            f"created_at_ms={generation['created_at_ms']}"
        )
    else:
        click.echo("Generation:   none")
    click.echo(f"Schema:       user_version={status['schema_version']}")
    click.echo(f"Operation id: {status['operation_id']}")
    transaction = status["transaction"]
    if isinstance(transaction, dict):
        click.echo(
            f"Transaction:  status={transaction['status']} "
            f"processed_raw_count={transaction['processed_raw_count']:,} "
            f"processed_blob_bytes={transaction['processed_blob_bytes']:,} "
            f"last_raw_id={transaction['last_raw_id']} updated_at_ms={transaction['updated_at_ms']} "
            f"heartbeat_at_ms={transaction.get('heartbeat_at_ms')}"
        )
    else:
        click.echo("Transaction:  none")
    operation = status["operation"]
    assert isinstance(operation, dict)
    owner = operation["owner"]
    assert isinstance(owner, dict)
    click.echo(
        f"Operation:    owner={owner.get('generation_owner_id')} pid={owner.get('pid')} "
        f"host={owner.get('host')} cursor={operation.get('cursor')} "
        f"recovery_state={operation.get('recovery_state')}"
    )
    delta = status["delta"]
    if isinstance(delta, dict):
        click.echo(f"Delta:        source_snapshot_matches={delta['source_snapshot_matches']}")
    recovery = status["recovery"]
    assert isinstance(recovery, list)
    if recovery:
        click.echo("Recovery:")
        for message in recovery:
            click.echo(f"  - {message}")
    else:
        click.echo("Recovery:     none")


__all__ = ["rebuild_index_status_command"]
