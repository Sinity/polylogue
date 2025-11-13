from __future__ import annotations

import json
from typing import Any, Dict, List

from rich.panel import Panel
from rich.table import Table

from ..commands import CommandEnv, status_command


def _provider_table(provider_summary: Dict[str, Dict[str, Any]]) -> Table:
    table = Table(title="Provider Summary", show_lines=False)
    table.add_column("Provider", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Items", justify="right")
    table.add_column("Attachments", justify="right")
    table.add_column("Failures", justify="right")
    for name, info in sorted(provider_summary.items()):
        table.add_row(
            name,
            str(info.get("runs", 0)),
            str(info.get("count", 0)),
            str(info.get("attachments", 0)),
            str(info.get("failures", 0)),
        )
    return table


def _recent_runs_table(runs: List[Dict[str, Any]]) -> Table:
    table = Table(title="Recent Runs", show_lines=False)
    table.add_column("Timestamp")
    table.add_column("Command")
    table.add_column("Provider")
    table.add_column("Count", justify="right")
    table.add_column("Duration", justify="right")
    for entry in runs[-10:]:
        table.add_row(
            entry.get("timestamp", ""),
            entry.get("cmd", ""),
            entry.get("provider", "") or "-",
            str(entry.get("count", 0)),
            f"{entry.get('duration', 0) or 0:.1f}s",
        )
    return table


def run_dashboards_cli(args, env: CommandEnv) -> None:
    result = status_command(env, runs_limit=args.runs_limit)
    if getattr(args, "json", False):
        payload = {
            "providerSummary": result.provider_summary,
            "recentRuns": result.recent_runs[-args.runs_limit :] if args.runs_limit else result.recent_runs,
        }
        print(json.dumps(payload, indent=2))
        return

    if env.ui.plain:
        env.ui.summary(
            "Dashboard",
            [
                f"Providers tracked: {', '.join(sorted(result.provider_summary)) or '<none>'}",
                f"Recent runs: {len(result.recent_runs)}",
            ],
        )
        return

    provider_panel = Panel(_provider_table(result.provider_summary))
    runs_panel = Panel(_recent_runs_table(result.recent_runs))
    env.ui.console.print(provider_panel)
    env.ui.console.print(runs_panel)
