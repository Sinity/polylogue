"""Data verification command."""

from __future__ import annotations

import json

import click

from polylogue.cli.types import AppEnv
from polylogue.verify import verify_data


@click.command("verify")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show breakdown by provider")
@click.pass_obj
def verify_command(env: AppEnv, json_output: bool, verbose: bool) -> None:
    """Verify data quality and integrity."""
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
    summary_line = f"Summary: {summary.get('ok', 0)} ok, {summary.get('warning', 0)} warnings, {summary.get('error', 0)} errors"
    lines.append("")
    lines.append(summary_line)

    env.ui.summary("Data Verification", lines)
