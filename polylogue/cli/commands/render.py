"""Render command - convert exports to Markdown/HTML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv

# Constants from app.py
DEFAULT_RENDER_OUT = None  # Will use provider-specific defaults from config


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add render command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    # Create parent parsers for render options
    render_parent = argparse.ArgumentParser(add_help=False)
    add_helpers["collapse_option"](render_parent)
    add_helpers["html_option"](render_parent)

    write_parent = argparse.ArgumentParser(add_help=False)
    write_parent.add_argument(
        "--write",
        action="store_true",
        help="Write updated Markdown files (default is dry-run)"
    )

    p = _add_command_parser(
        subparsers,
        "render",
        parents=[render_parent, write_parent],
        help="Render JSON exports to Markdown/HTML",
        description="Render provider exports to Markdown/HTML",
        epilog=add_helpers["examples_epilog"]("render"),
    )

    # Input and output
    p.add_argument("input", type=Path, help="Input JSON file or directory containing exports")
    add_helpers["out_option"](
        p,
        default_path=DEFAULT_RENDER_OUT,
        help_text="Output directory for rendered Markdown/HTML",
    )

    # Attachment options
    p.add_argument(
        "--links-only",
        action="store_true",
        help="Link attachments without downloading"
    )
    p.add_argument(
        "--attachment-ocr",
        action="store_true",
        help="Attempt OCR on image attachments when indexing attachment text",
    )

    # Output options
    add_helpers["diff_option"](p)
    p.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p.add_argument("--print-paths", action="store_true", help="List written files after rendering")
    p.add_argument("--to-clipboard", action="store_true", help="Copy single rendered file to clipboard")
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate markdown from database instead of reading source files"
    )


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute render command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..render_force import run_render_force
    from ..render import run_render_cli

    # Check if --force flag is set to regenerate from database
    if getattr(args, "force", False):
        # Extract provider and conversation_id from input if provided
        provider = None
        conversation_id = None
        output_dir = getattr(args, "out", None)

        # Call render_force command
        exit_code = run_render_force(
            env,
            provider=provider,
            conversation_id=conversation_id,
            output_dir=output_dir
        )
        raise SystemExit(exit_code)

    # Normal render flow
    run_render_cli(args, env, json_output=getattr(args, "json", False))
