"""Run a bounded source-authoritative reindex canary."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("reindex-canary")
@click.option("--archive-root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--input-index", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--sessions-per-origin", type=int, default=100, show_default=True)
@click.option("--report", "report_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option(
    "--schema-inference-receipt",
    "schema_inference_receipt_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
@click.option("--no-promote", is_flag=True)
def reindex_canary_command(
    archive_root: Path,
    input_index: Path | None,
    sessions_per_origin: int,
    report_path: Path,
    schema_inference_receipt_path: Path,
    output_format: str,
    no_promote: bool,
) -> None:
    """Build an inactive candidate and write bounded forensic evidence."""
    if not no_promote:
        raise click.UsageError("reindex-canary requires --no-promote")
    if sessions_per_origin <= 0:
        raise click.BadParameter("must be positive", param_hint="--sessions-per-origin")
    from polylogue.maintenance.reindex_canary import run_reindex_canary, write_canary_report

    try:
        result = run_reindex_canary(
            archive_root,
            input_index=input_index,
            schema_inference_receipt_path=schema_inference_receipt_path,
            sessions_per_origin=sessions_per_origin,
            no_promote=True,
        )
        payload = write_canary_report(
            report_path,
            selection=result.selection,
            comparison=result.comparison,
            rebuild_receipt=result.rebuild_receipt,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo(f"Selected:     {len(result.selection.selected_raw_ids):,} raw row(s)")
        click.echo(f"Compared:     {len(result.comparison.differences):,} difference(s), forensic only")
        click.echo(f"Report:       {report_path}")


__all__ = ["reindex_canary_command"]
