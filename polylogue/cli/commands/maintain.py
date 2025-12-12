"""Maintain command - system maintenance and diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add maintain command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    p = _add_command_parser(
        subparsers,
        "maintain",
        help="System maintenance (prune/doctor/index)",
        description="System maintenance and diagnostics",
        epilog=add_helpers["examples_epilog"]("maintain"),
    )
    maintain_sub = p.add_subparsers(dest="maintain_cmd", required=True)

    # maintain prune
    p_maintain_prune = _add_command_parser(
        maintain_sub,
        "prune",
        help="Remove legacy single-file outputs and attachments",
        description="Remove legacy single-file outputs and attachments"
    )
    p_maintain_prune.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        type=Path,
        help="Root directory to prune (repeatable). Defaults to all configured output directories.",
    )
    p_maintain_prune.add_argument("--dry-run", action="store_true", help="Print planned actions without deleting files")
    p_maintain_prune.add_argument("--max-disk", type=float, default=None, help="Abort if projected snapshot size exceeds this many GiB")

    # maintain doctor
    p_maintain_doctor = _add_command_parser(
        maintain_sub,
        "doctor",
        help="Check local data directories for common issues",
        description="Check local data directories for common issues"
    )
    p_maintain_doctor.add_argument("--codex-dir", type=Path, default=None, help="Override Codex sessions directory")
    p_maintain_doctor.add_argument("--claude-code-dir", type=Path, default=None, help="Override Claude Code projects directory")
    p_maintain_doctor.add_argument("--limit", type=int, default=None, help="Limit number of files inspected per provider")
    p_maintain_doctor.add_argument("--json", action="store_true", help="Emit machine-readable report")
    p_maintain_doctor.add_argument("--json-verbose", action="store_true", help="Emit JSON with verbose details")

    # maintain index
    p_maintain_index = _add_command_parser(
        maintain_sub,
        "index",
        help="Index maintenance helpers",
        description="Inspect/repair Polylogue indexes"
    )
    index_sub = p_maintain_index.add_subparsers(dest="subcmd", required=True)
    p_index_check = index_sub.add_parser("check", help="Validate SQLite/Qdrant indexes")
    p_index_check.add_argument("--repair", action="store_true", help="Attempt to rebuild missing SQLite FTS data")
    p_index_check.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant validation even when configured")
    p_index_check.add_argument("--json", action="store_true", help="Emit validation results as JSON")
    p_index_check.add_argument("--json-verbose", action="store_true", help="Emit JSON with verbose details")

    # maintain restore
    p_maintain_restore = _add_command_parser(
        maintain_sub,
        "restore",
        help="Restore a snapshot directory",
        description="Restore a previously snapshotted output directory",
    )
    p_maintain_restore.add_argument("--from", dest="src", type=Path, required=True, help="Snapshot directory to restore from")
    p_maintain_restore.add_argument("--to", dest="dest", type=Path, required=True, help="Destination output directory")
    p_maintain_restore.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    p_maintain_restore.add_argument("--json", action="store_true", help="Emit restoration summary as JSON")


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute maintain command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..maintain import run_maintain_cli

    run_maintain_cli(args, env)
