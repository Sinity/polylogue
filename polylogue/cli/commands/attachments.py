"""Attachments command - manage and inspect attachments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add attachments command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    p = _add_command_parser(
        subparsers,
        "attachments",
        help="Attachment utilities (stats/extract)",
        description="Inspect and extract attachments",
        epilog=add_helpers["examples_epilog"]("attachments"),
    )
    attachments_sub = p.add_subparsers(dest="attachments_cmd", required=True)

    # attachments stats
    p_att_stats = _add_command_parser(
        attachments_sub,
        "stats",
        help="Summarize attachments",
        description="Summarize attachment counts/bytes"
    )
    p_att_stats.add_argument("--dir", type=Path, default=None, help="Root directory containing archives (defaults to all output roots)")
    p_att_stats.add_argument("--ext", type=str, default=None, help="Filter by file extension (e.g., .png)")
    p_att_stats.add_argument("--hash", action="store_true", help="Hash attachments to compute deduped totals")
    p_att_stats.add_argument("--sort", choices=["size", "name"], default="size", help="Sort field for top rows")
    p_att_stats.add_argument("--limit", type=int, default=10, help="Limit number of files displayed (0 for all)")
    p_att_stats.add_argument("--csv", type=str, default=None, help="Write attachment rows to CSV ('-' for stdout)")
    p_att_stats.add_argument("--json", action="store_true", help="Emit stats as JSON")
    p_att_stats.add_argument("--json-lines", action="store_true", help="Emit per-attachment JSONL (implies --json)")
    p_att_stats.add_argument(
        "--from-index",
        action="store_true",
        help="Read attachment metadata from the index DB (includes text/OCR stats when available)",
    )

    # attachments extract
    p_att_extract = _add_command_parser(
        attachments_sub,
        "extract",
        help="Copy attachments to a directory",
        description="Extract attachments by extension"
    )
    p_att_extract.add_argument("--dir", type=Path, default=None, help="Root directory containing archives (defaults to all output roots)")
    p_att_extract.add_argument("--ext", type=str, required=True, help="File extension to extract (e.g., .pdf)")
    p_att_extract.add_argument("--out", type=Path, required=True, help="Destination directory for extracted files")
    p_att_extract.add_argument("--limit", type=int, default=0, help="Limit number of files extracted (0 for all)")
    p_att_extract.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files in destination")


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute attachments command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..attachments import run_attachments_cli

    if not getattr(args, "attachments_cmd", None):
        env.ui.console.print("[red]attachments requires a sub-command (stats/extract)")
        raise SystemExit(1)

    run_attachments_cli(args, env)
