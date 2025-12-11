"""Import command - import provider exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add import command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    p = _add_command_parser(
        subparsers,
        "import",
        help="Import provider exports",
        description="Import provider exports",
        epilog=add_helpers["examples_epilog"]("import"),
    )

    # Required arguments
    p.add_argument(
        "provider",
        choices=["chatgpt", "claude", "claude-code", "codex"],
        help="Provider export format"
    )
    p.add_argument(
        "source",
        nargs="*",
        help="Export path or session identifier (depends on provider); use 'pick', '?', or '-' to trigger interactive picker"
    )

    # Output options
    add_helpers["out_option"](
        p,
        default_path=Path("(provider-specific)"),
        help_text="Override output directory"
    )

    # Common flags
    add_helpers["collapse_option"](p)
    add_helpers["html_option"](p)
    p.add_argument(
        "--attachment-ocr",
        action="store_true",
        help="Attempt OCR on image attachments when importing",
    )
    add_helpers["dry_run_option"](p)
    add_helpers["force_option"](p, help_text="Rewrite even if conversations appear up-to-date")
    p.add_argument("--print-paths", action="store_true", help="List written files after import")

    # Selection options
    import_selection_group = p.add_mutually_exclusive_group()
    import_selection_group.add_argument(
        "--all",
        action="store_true",
        help="Process all available items without interactive selection"
    )
    import_selection_group.add_argument(
        "--conversation-id",
        dest="conversation_ids",
        action="append",
        help="Specific conversation ID to import (repeatable)"
    )

    # Provider-specific options
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override source directory for codex/claude-code sessions"
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p.add_argument("--to-clipboard", action="store_true", help="Copy a single imported Markdown file to the clipboard")


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute import command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..imports import run_import_cli

    run_import_cli(args, env)
