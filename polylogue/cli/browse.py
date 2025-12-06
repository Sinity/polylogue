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
        run_runs_cli,
    )

    browse_cmd = args.browse_cmd  # required=True enforced by argparse

    if browse_cmd == "branches":
        run_inspect_branches(args, env)
    elif browse_cmd == "stats":
        run_stats_cli(args, env)
    elif browse_cmd == "status":
        run_status_cli(args, env)
    elif browse_cmd == "runs":
        run_runs_cli(args, env)
    else:
        raise SystemExit(f"Unknown browse sub-command: {browse_cmd}")


__all__ = ["run_browse_cli"]
