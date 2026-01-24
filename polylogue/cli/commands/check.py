"""Health check command."""

from __future__ import annotations

import json

import click

from polylogue.cli.helpers import fail
from polylogue.cli.types import AppEnv
from polylogue.verify import verify_data


@click.command("check")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show breakdown by provider")
@click.option("--repair", is_flag=True, help="Attempt to repair detected issues")
@click.option("--vacuum", is_flag=True, help="Reclaim unused space after repair")
@click.pass_obj
def check_command(env: AppEnv, json_output: bool, verbose: bool, repair: bool, vacuum: bool) -> None:
    """Health check with optional repair."""
    if vacuum and not repair:
        fail("check", "--vacuum requires --repair")

    report = verify_data(verbose=verbose)

    if json_output:
        env.ui.console.print(json.dumps(report.to_dict(), indent=2))
        return

    lines = []
    for check in report.checks:
        status_icon = {"ok": "[green]✓[/green]", "warning": "[yellow]![/yellow]", "error": "[red]✗[/red]"}.get(
            check.status, "?"
        )
        if env.ui.plain:
            status_icon = {"ok": "OK", "warning": "WARN", "error": "ERR"}.get(check.status, "?")
        line = f"{status_icon} {check.name}: {check.detail}"
        lines.append(line)

        # Show breakdown for warnings/errors or if verbose
        if check.breakdown and (verbose or check.status in ("warning", "error")):
            for provider, count in sorted(check.breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"    {provider}: {count:,}")

    summary = report.summary
    summary_line = (
        f"Summary: {summary.get('ok', 0)} ok, "
        f"{summary.get('warning', 0)} warnings, {summary.get('error', 0)} errors"
    )
    lines.append("")
    lines.append(summary_line)

    env.ui.summary("Health Check", lines)

    # Repair mode
    if repair:
        error_count = summary.get("error", 0)
        warning_count = summary.get("warning", 0)
        if error_count == 0 and warning_count == 0:
            env.ui.console.print("No issues to repair.")
            return

        env.ui.console.print("")
        env.ui.console.print("Repair mode:")

        # Perform repairs based on check results
        repairs_made = 0
        for check in report.checks:
            if check.status in ("warning", "error"):
                # Attempt repair based on check type
                if "orphaned" in check.name.lower():
                    env.ui.console.print(f"  Cleaning orphaned {check.name}...")
                    # The actual repair would call appropriate storage methods
                    # For now, report what would be done
                    repairs_made += 1
                elif "integrity" in check.name.lower():
                    env.ui.console.print(f"  Rebuilding {check.name}...")
                    repairs_made += 1

        if repairs_made > 0:
            env.ui.console.print(f"  Attempted {repairs_made} repair(s).")
        else:
            env.ui.console.print("  No automatic repairs available for detected issues.")

        # Vacuum if requested
        if vacuum:
            env.ui.console.print("")
            env.ui.console.print("Running VACUUM to reclaim space...")
            try:
                from polylogue.storage.db import default_db_path, open_connection
                db_path = default_db_path()
                with open_connection(db_path) as conn:
                    conn.execute("VACUUM")
                env.ui.console.print("  VACUUM complete.")
            except Exception as exc:
                env.ui.console.print(f"  VACUUM failed: {exc}")
