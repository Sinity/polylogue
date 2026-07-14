"""Blob garbage collection: ``blob-gc`` and ``gc-history``."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.logging import configure_logging
from polylogue.paths import archive_root, render_root


@click.command("blob-gc")
@click.option(
    "--max-batch",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of eligible blobs to delete or preview.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Actually delete eligible blobs. Without this flag the command is a dry-run preview.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def blob_gc_command(max_batch: int, yes: bool, output_format: str) -> None:
    """Preview or run lease-safe blob garbage collection.

    The default is a dry-run report. Pass ``--yes`` to reclaim eligible blobs.
    """
    from polylogue.storage.blob_gc import run_blob_gc_report

    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )
    result = run_blob_gc_report(
        config.db_path,
        config.archive_root / "blob",
        max_batch=max_batch,
        dry_run=not yes,
    )
    payload = {
        "ok": True,
        "mode": "blob_gc",
        "mutates": bool(yes),
        **result.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    action = "would delete" if result.dry_run else "deleted"
    affected = result.would_delete_count if result.dry_run else result.deleted_count
    click.echo("Blob GC dry-run" if result.dry_run else "Blob GC")
    click.echo(f"Archive DB: {result.db_path}")
    click.echo(f"Blob root:  {result.blob_dir}")
    click.echo(f"Candidates: {result.candidate_count:,}")
    click.echo(f"Inspected:  {result.inspected_count:,}")
    click.echo(f"Result:     {action} {affected:,} blob(s)")
    click.echo(
        "Skipped:    "
        f"referenced={result.skipped_referenced:,} "
        f"reserved={result.skipped_reserved:,} "
        f"missing={result.skipped_missing:,} "
        f"unlink_error={result.skipped_unlink_error:,}"
    )
    if not result.dry_run:
        click.echo(f"Reclaimed:  {result.reclaimed_bytes:,} byte(s)")
        if result.generation_id is not None:
            click.echo(f"Generation: {result.generation_id}")


@click.command("gc-history")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of recent GC passes to display.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.pass_obj
def gc_history_command(env: AppEnv, limit: int, output_format: str) -> None:
    """Show recent blob-GC passes recorded in ``gc_generations``.

    Surfaces the typed reclaim counters (``reclaimed_count`` /
    ``reclaimed_bytes``) and start/completion timestamps written by
    ``run_blob_gc`` so operators can audit GC reclamation over time
    without bespoke SQLite tooling.

    A pass whose ``completed_at_ms`` is null crashed mid-run; the row is
    still surfaced so operators can see it happened.
    """
    from polylogue.storage.blob_gc import read_gc_history

    configure_logging()
    config = Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )
    history = read_gc_history(config.db_path, limit=limit)

    def _iso_ms(epoch_ms: int | None) -> str | None:
        if epoch_ms is None:
            return None
        return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()

    if output_format == "json":
        payload = [
            {
                "generation_id": row.generation_id,
                "started_at_ms": row.started_at_ms,
                "started_at_iso": _iso_ms(row.started_at_ms),
                "completed_at_ms": row.completed_at_ms,
                "completed_at_iso": _iso_ms(row.completed_at_ms),
                "reclaimed_count": row.reclaimed_count,
                "reclaimed_bytes": row.reclaimed_bytes,
            }
            for row in history
        ]
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not history:
        click.echo("No GC generations recorded yet.")
        return

    click.echo(f"Recent blob-GC passes (newest first, limit={limit}):")
    click.echo("")
    for row in history:
        when = (
            datetime.fromtimestamp(row.completed_at_ms / 1000, tz=UTC).isoformat(timespec="seconds")
            if row.completed_at_ms is not None
            else "unknown (crashed mid-pass)"
        )
        click.echo(f"  generation={row.generation_id}  completed_at={when}")
        click.echo(f"    reclaimed_count={row.reclaimed_count}  reclaimed_bytes={row.reclaimed_bytes}")
