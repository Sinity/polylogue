"""Status command - show cached Drive info and recent runs."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv


def _configure_status_parser(parser: argparse.ArgumentParser, *, require_provider: bool = False) -> None:
    """Configure status parser arguments.

    Args:
        parser: Parser to configure
        require_provider: Whether to require --providers argument
    """
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    parser.add_argument(
        "--json-lines",
        action="store_true",
        help="Stream newline-delimited JSON records (auto-enables --json, useful with --watch)",
    )
    parser.add_argument(
        "--json-verbose",
        action="store_true",
        help="Allow status to print tables/logs alongside JSON/JSONL output",
    )
    status_mode_group = parser.add_mutually_exclusive_group()
    status_mode_group.add_argument("--watch", action="store_true", help="Continuously refresh the status output")
    status_mode_group.add_argument("--dump-only", action="store_true", help="Only perform the dump action without printing summaries")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between refresh while watching")
    parser.add_argument("--dump", type=str, default=None, help="Write recent runs to a file ('-' for stdout)")
    parser.add_argument("--dump-limit", type=int, default=100, help="Number of runs to include when dumping")
    parser.add_argument("--runs-limit", type=int, default=200, help="Number of historical runs to include in summaries")
    parser.add_argument("--top", type=int, default=0, help="Show top runs by attachments/tokens")
    parser.add_argument("--inbox", action="store_true", help="Include inbox coverage counts in summaries")
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        required=require_provider,
        help="Comma-separated provider filter (limits summaries, dumps, and JSON output)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress table output (useful with --json-lines)")
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Write aggregated provider/run summary JSON to a file ('-' for stdout)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only emit the summary JSON without printing tables",
    )


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add status command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    p = _add_command_parser(
        subparsers,
        "status",
        help="Show cached Drive info and recent runs",
        description="Show cached Drive info and recent runs",
    )
    _configure_status_parser(p, require_provider=False)


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute status command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..status import run_status_cli

    run_status_cli(args, env)
