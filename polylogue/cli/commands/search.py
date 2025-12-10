"""Search command - search rendered transcripts."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add search command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    p = _add_command_parser(
        subparsers,
        "search",
        help="Search rendered transcripts",
        description="Search rendered transcripts",
        epilog=add_helpers["examples_epilog"]("search"),
    )

    # Query
    p.add_argument("query", type=str, help="FTS search query (SQLite syntax); use --from-stdin to read from stdin")
    p.add_argument("--limit", type=int, default=20, help="Maximum number of hits to return")

    # Filters
    p.add_argument("--provider", type=str, default=None, help="Filter by provider slug")
    p.add_argument("--slug", type=str, default=None, help="Filter by conversation slug")
    p.add_argument("--conversation-id", type=str, default=None, help="Filter by provider conversation id")
    p.add_argument("--branch", type=str, default=None, help="Restrict to a single branch ID")
    p.add_argument("--model", type=str, default=None, help="Filter by source model when recorded")
    p.add_argument("--since", type=str, default=None, help="Only include messages on/after this timestamp")
    p.add_argument("--until", type=str, default=None, help="Only include messages on/before this timestamp")

    # Attachment filters
    search_attachment_group = p.add_mutually_exclusive_group()
    search_attachment_group.add_argument("--with-attachments", action="store_true", help="Limit to messages with extracted attachments")
    search_attachment_group.add_argument("--without-attachments", action="store_true", help="Limit to messages without attachments")
    p.add_argument("--in-attachments", action="store_true", help="Search within attachment text when indexed")
    p.add_argument("--attachment-name", type=str, default=None, help="Filter by attachment filename substring")

    # Output options
    p.add_argument("--no-picker", action="store_true", help="Skip skim picker preview even when interactive")
    p.add_argument("--json", action="store_true", help="Emit machine-readable search results")
    p.add_argument("--json-lines", action="store_true", help="Emit newline-delimited JSON hits (implies --json and disables tables)")
    p.add_argument("--csv", type=str, default=None, help="Write search hits to CSV ('-' for stdout)")
    p.add_argument(
        "--fields",
        type=str,
        default="provider,conversationId,slug,branchId,messageId,position,timestamp,score,model,attachments,snippet,conversationPath,branchPath",
        help="Comma-separated fields to include in CSV/JSONL output",
    )
    p.add_argument("--from-stdin", action="store_true", help="Read the search query from stdin (ignores positional query if present)")
    p.add_argument("--open", action="store_true", help="Open result file in $EDITOR after search")


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute search command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..inspect import run_inspect_search

    run_inspect_search(args, env)
