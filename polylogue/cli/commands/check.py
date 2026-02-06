"""Health check command."""

from __future__ import annotations

import json

import click

from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.health import VerifyStatus, get_health, run_all_repairs


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

    config = load_effective_config(env)
    report = get_health(config)

    if json_output:
        env.ui.console.print(json.dumps(report.to_dict(), indent=2))
        return

    lines = []
    for check in report.checks:
        status_icon = {
            VerifyStatus.OK: "[green]✓[/green]",
            VerifyStatus.WARNING: "[yellow]![/yellow]",
            VerifyStatus.ERROR: "[red]✗[/red]",
        }.get(check.status, "?")
        if env.ui.plain:
            status_icon = {
                VerifyStatus.OK: "OK",
                VerifyStatus.WARNING: "WARN",
                VerifyStatus.ERROR: "ERR",
            }.get(check.status, "?")
        line = f"{status_icon} {check.name}: {check.detail}"
        lines.append(line)

        # Show breakdown for warnings/errors or if verbose
        if check.breakdown and (verbose or check.status in (VerifyStatus.WARNING, VerifyStatus.ERROR)):
            for provider, count in sorted(check.breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"    {provider}: {count:,}")

    summary = report.summary
    summary_line = (
        f"Summary: {summary.get('ok', 0)} ok, {summary.get('warning', 0)} warnings, {summary.get('error', 0)} errors"
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
        else:
            click.echo("")
            click.echo("Running repairs...")

            # Run all repair functions
            results = run_all_repairs(config)

            total_repaired = 0
            for result in results:
                if result.repaired_count > 0 or not result.success:
                    status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                    if env.ui.plain:
                        status = "OK" if result.success else "FAIL"
                    env.ui.console.print(f"  {status} {result.name}: {result.detail}")
                    total_repaired += result.repaired_count

            if total_repaired > 0:
                click.echo(f"\nRepaired {total_repaired} issue(s)")
            else:
                click.echo("  No issues found that could be automatically repaired.")

        # Vacuum if requested (always run when --vacuum is specified)
        if vacuum:
            env.ui.console.print("")
            env.ui.console.print("Running VACUUM to reclaim space...")
            try:
                from polylogue.storage.backends.sqlite import default_db_path, open_connection

                db_path = default_db_path()
                with open_connection(db_path) as conn:
                    conn.execute("VACUUM")
                env.ui.console.print("  VACUUM complete.")
            except Exception as exc:
                env.ui.console.print(f"  VACUUM failed: {exc}")
