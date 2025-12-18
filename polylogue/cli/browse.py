"""Browse command - consolidated exploration interface."""

from __future__ import annotations

from types import SimpleNamespace
from ..commands import CommandEnv


def run_browse_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    """Dispatch to appropriate browse subcommand."""
    from .analytics import run_analytics_cli
    from .branches_cli import run_branches_cli
    from .inbox import run_inbox_cli
    from .metrics import run_metrics_cli
    from .runs import run_runs_cli
    from .status import run_stats_cli, run_status_cli
    from .timeline import run_timeline_cli

    browse_cmd = args.browse_cmd

    if browse_cmd == "branches":
        run_branches_cli(args, env)
    elif browse_cmd == "stats":
        try:
            run_stats_cli(args, env)
        except SystemExit:
            return
    elif browse_cmd == "status":
        run_status_cli(args, env)
    elif browse_cmd == "runs":
        run_runs_cli(args, env)
    elif browse_cmd == "inbox":
        run_inbox_cli(args, env)
    elif browse_cmd == "metrics":
        run_metrics_cli(args, env)
    elif browse_cmd == "timeline":
        run_timeline_cli(args, env)
    elif browse_cmd == "analytics":
        run_analytics_cli(args, env)
    else:
        raise SystemExit(f"Unknown browse sub-command: {browse_cmd}")


__all__ = ["run_browse_cli"]
