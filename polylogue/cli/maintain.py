"""Maintain command - consolidated maintenance interface."""

from __future__ import annotations

import argparse
from ..commands import CommandEnv


def run_maintain_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    """Dispatch to appropriate maintenance subcommand."""
    from .app import run_prune_cli, run_doctor_cli
    from .index_cli import run_index_cli

    maintain_cmd = getattr(args, "maintain_cmd", None)

    if maintain_cmd == "prune":
        run_prune_cli(args, env)
    elif maintain_cmd == "doctor":
        run_doctor_cli(args, env)
    elif maintain_cmd == "index":
        run_index_cli(args, env)
    else:
        raise SystemExit("maintain requires a subcommand: prune, doctor, index")


__all__ = ["run_maintain_cli"]
