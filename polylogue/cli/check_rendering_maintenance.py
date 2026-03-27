"""Maintenance rendering helpers for the check command."""

from __future__ import annotations

import click

from polylogue.cli.check_support import run_vacuum
from polylogue.cli.check_workflow import CheckCommandOptions, CheckCommandResult
from polylogue.cli.types import AppEnv


def emit_maintenance_output(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> None:
    """Render maintenance/correction output after the health report."""
    if result.maintenance_results is not None:
        click.echo("")
        mode_label = "Preview of maintenance" if options.preview else "Running maintenance"
        click.echo(f"{mode_label}...")
        if options.maintenance_targets:
            click.echo(f"  Targets: {', '.join(options.maintenance_targets)}")
        total_repaired = 0
        for repair in result.maintenance_results:
            if repair.repaired_count > 0 or not repair.success:
                status = "[green]✓[/green]" if repair.success else "[red]✗[/red]"
                if env.ui.plain:
                    status = "OK" if repair.success else "FAIL"
                mode = f"{repair.category.value}{' destructive' if repair.destructive else ''}"
                env.ui.console.print(f"  {status} {repair.name} [{mode}]: {repair.detail}")
                total_repaired += repair.repaired_count

        if total_repaired > 0:
            action = "Would change" if options.preview else "Changed"
            click.echo(f"\n{action} {total_repaired} issue(s)")
        else:
            click.echo("  No selected maintenance work was needed.")
    elif options.repair or options.cleanup:
        env.ui.console.print("No maintenance operations were selected.")

    if (options.repair or options.cleanup) and options.vacuum and options.preview:
        env.ui.console.print("")
        env.ui.console.print("Preview mode: VACUUM skipped.")
    elif (options.repair or options.cleanup) and options.vacuum:
        run_vacuum(env)


__all__ = ["emit_maintenance_output"]
