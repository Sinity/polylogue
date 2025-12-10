"""Browse command - explore rendered data and system status."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add browse command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    from ..arg_helpers import create_output_parent, create_filter_parent
    from .status import _configure_status_parser

    p = _add_command_parser(
        subparsers,
        "browse",
        help="Browse data (branches/stats/status/runs)",
        description="Explore rendered data and system status",
        epilog=add_helpers["examples_epilog"]("browse"),
    )
    browse_sub = p.add_subparsers(dest="browse_cmd", required=True)

    # browse branches
    p_browse_branches = _add_command_parser(
        browse_sub,
        "branches",
        help="Explore branch graphs for conversations",
        description="Explore branch graphs for conversations"
    )
    p_browse_branches.add_argument("--provider", type=str, default=None, help="Filter by provider slug")
    p_browse_branches.add_argument("--slug", type=str, default=None, help="Filter by conversation slug")
    p_browse_branches.add_argument("--conversation-id", type=str, default=None, help="Filter by provider conversation id")
    p_browse_branches.add_argument("--min-branches", type=int, default=1, help="Only include conversations with at least this many branches")
    p_browse_branches.add_argument("--branch", type=str, default=None, help="Branch ID to inspect or diff against the canonical path")
    p_browse_branches.add_argument("--diff", action="store_true", help="Display a unified diff between a branch and canonical transcript")
    p_browse_branches.add_argument(
        "--html",
        dest="html_mode",
        nargs="?",
        const="on",
        default="auto",
        choices=["auto", "on", "off"],
        metavar="MODE",
        help="Branch HTML mode: on/off/auto (default auto)",
    )
    p_browse_branches.add_argument("--out", type=Path, default=None, help="Write the branch explorer HTML to this path")
    p_browse_branches.add_argument("--theme", type=str, default=None, choices=["light", "dark"], help="Override HTML explorer theme")
    p_browse_branches.add_argument("--no-picker", action="store_true", help="Skip interactive selection even when skim/gum are available")
    p_browse_branches.add_argument("--open", action="store_true", help="Open result in $EDITOR after command completes")

    # browse stats
    output_parent = create_output_parent()
    filter_parent = create_filter_parent()

    p_browse_stats = _add_command_parser(
        browse_sub,
        "stats",
        parents=[output_parent, filter_parent],
        help="Summarize Markdown output directories",
        description="Summarize Markdown output directories"
    )
    p_browse_stats.add_argument("--dir", type=Path, default=None, help="Directory containing Markdown exports")
    p_browse_stats.add_argument("--ignore-legacy", action="store_true", help="Ignore legacy *.md files alongside conversation.md")
    p_browse_stats.add_argument(
        "--sort",
        choices=["tokens", "attachments", "attachment-bytes", "words", "recent"],
        default="tokens",
        help="Sort per-file rows before display/export",
    )
    p_browse_stats.add_argument("--limit", type=int, default=0, help="Limit the number of file rows shown/exported (0 shows all)")
    p_browse_stats.add_argument("--csv", type=str, default=None, help="Write per-file rows to CSV ('-' for stdout)")
    p_browse_stats.add_argument("--json-verbose", action="store_true", help="Print warnings/logs alongside --json/--json-lines output")

    # browse status
    p_browse_status = _add_command_parser(
        browse_sub,
        "status",
        help="Show cached Drive info and recent runs",
        description="Show cached Drive info and recent runs",
    )
    _configure_status_parser(p_browse_status)

    # browse runs
    p_browse_runs = _add_command_parser(
        browse_sub,
        "runs",
        help="List recent runs",
        description="List run history with filters"
    )
    p_browse_runs.add_argument("--limit", type=int, default=50, help="Number of runs to display")
    p_browse_runs.add_argument("--providers", type=str, default=None, help="Comma-separated provider filter")
    p_browse_runs.add_argument("--commands", type=str, default=None, help="Comma-separated command filter")
    p_browse_runs.add_argument("--since", type=str, default=None, help="Only include runs on/after this timestamp (YYYY-MM-DD or ISO)")
    p_browse_runs.add_argument("--until", type=str, default=None, help="Only include runs on/before this timestamp")
    p_browse_runs.add_argument("--json", action="store_true", help="Emit runs as JSON")
    p_browse_runs.add_argument("--json-verbose", action="store_true", help="Print logs alongside --json output")

    # browse inbox
    p_browse_inbox = _add_command_parser(
        browse_sub,
        "inbox",
        help="List pending inbox exports and quarantine malformed items",
        description="List pending inbox exports and quarantine malformed items",
    )
    p_browse_inbox.add_argument("--providers", type=str, default="chatgpt,claude", help="Comma-separated provider filter (default: chatgpt,claude)")
    p_browse_inbox.add_argument("--dir", type=Path, default=None, help="Override inbox root for a generic scan")
    p_browse_inbox.add_argument("--quarantine", action="store_true", help="Move unknown/malformed inbox items into a quarantine folder")
    p_browse_inbox.add_argument("--quarantine-dir", type=Path, default=None, help="Target directory for quarantined items (default: <inbox>/quarantine)")
    p_browse_inbox.add_argument("--json", action="store_true", help="Emit machine-readable inbox status")


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute browse command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..browse import run_browse_cli

    run_browse_cli(args, env)
