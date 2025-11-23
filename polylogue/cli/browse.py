"""Browse command - consolidated exploration interface."""

from __future__ import annotations

import argparse
from ..commands import CommandEnv


def run_browse_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    """Dispatch to appropriate browse subcommand."""
    from .app import (
        run_inspect_branches,
        run_stats_cli,
        run_status_cli,
        run_dashboards_cli,
        run_runs_cli,
    )

    browse_cmd = getattr(args, "browse_cmd", None)

    if browse_cmd == "branches":
        run_inspect_branches(args, env)
    elif browse_cmd == "stats":
        run_stats_cli(args, env)
    elif browse_cmd == "status":
        run_status_cli(args, env)
    elif browse_cmd == "dashboards":
        run_dashboards_cli(args, env)
    elif browse_cmd == "runs":
        run_runs_cli(args, env)
    else:
        raise SystemExit("browse requires a subcommand: branches, stats, status, dashboards, runs")


__all__ = ["run_browse_cli"]
