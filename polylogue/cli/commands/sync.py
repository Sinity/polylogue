"""Sync command - synchronize provider archives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv

# Constants from app.py
DEFAULT_FOLDER_NAME = "AI Chat"
DEFAULT_SYNC_OUT = None  # Will use provider-specific defaults from config
LOCAL_SYNC_PROVIDER_NAMES = ["codex", "claude-code", "claude_code", "chatgpt", "claude"]


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add sync command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    p = _add_command_parser(
        subparsers,
        "sync",
        help="Synchronize provider archives",
        description="Synchronize provider archives",
        epilog=add_helpers["examples_epilog"]("sync"),
    )

    # Required arguments
    p.add_argument(
        "provider",
        choices=["drive", *LOCAL_SYNC_PROVIDER_NAMES],
        help="Provider to synchronize",
    )

    # Output options
    add_helpers["out_option"](
        p,
        default_path=DEFAULT_SYNC_OUT,
        help_text="Override output directory (provider defaults from config are used otherwise)",
    )

    # Attachment options
    p.add_argument(
        "--links-only",
        action="store_true",
        help="Link attachments instead of downloading (Drive only)"
    )
    p.add_argument(
        "--attachment-ocr",
        action="store_true",
        help="Attempt OCR on image attachments when indexing attachment text",
    )

    # Common flags
    add_helpers["dry_run_option"](p)
    add_helpers["force_option"](p, help_text="Re-render even if conversations are up-to-date")
    p.add_argument("--prune", action="store_true", help="Remove outputs for conversations that vanished upstream")
    add_helpers["collapse_option"](p)
    add_helpers["html_option"](p)
    add_helpers["diff_option"](p, help_text="Write delta diff alongside updated Markdown")

    # Output options
    p.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    p.add_argument("--print-paths", action="store_true", help="List written files after sync")

    # Selection options
    p.add_argument(
        "--chat-id",
        dest="chat_ids",
        action="append",
        help="Drive chat/file ID to sync (repeatable)",
    )

    sync_selection_group = p.add_mutually_exclusive_group()
    sync_selection_group.add_argument(
        "--session",
        dest="sessions",
        action="append",
        type=Path,
        help="Local session/export path to sync (repeatable; local providers)",
    )
    sync_selection_group.add_argument(
        "--all",
        action="store_true",
        help="Process all available items without interactive selection"
    )

    # Provider-specific options
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override local session/export directory",
    )
    p.add_argument(
        "--folder-name",
        type=str,
        default=DEFAULT_FOLDER_NAME,
        help="Drive folder name (drive provider)"
    )
    p.add_argument("--folder-id", type=str, default=None, help="Drive folder ID override")
    p.add_argument("--since", type=str, default=None, help="Only include Drive chats updated on/after this timestamp")
    p.add_argument("--until", type=str, default=None, help="Only include Drive chats updated on/before this timestamp")
    p.add_argument("--name-filter", type=str, default=None, help="Regex filter for Drive chat names")
    p.add_argument("--list-only", action="store_true", help="List Drive chats without syncing")
    p.add_argument("--offline", action="store_true", help="Skip network-dependent steps (Drive disallowed)")

    # Watch mode flags (local providers only)
    p.add_argument(
        "--watch",
        action="store_true",
        help="Watch for changes and sync continuously (local providers only)"
    )
    p.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Minimal seconds between sync runs in watch mode (default: 2.0)"
    )
    p.add_argument(
        "--stall-seconds",
        type=float,
        default=60.0,
        help="Warn when watch makes no progress for this many seconds"
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="In watch mode, run a single sync pass and exit"
    )
    p.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a rollback snapshot of the output directory before watching"
    )


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute sync command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    from ..watch import run_watch_cli
    from ..sync import run_sync_cli

    # Handle watch mode
    if getattr(args, "watch", False):
        if args.provider == "drive":
            raise SystemExit("Drive does not support --watch; use local providers like codex/claude-code/chatgpt.")

        # Validate provider supports watch mode
        from ...local_sync import get_local_provider
        provider = get_local_provider(args.provider)
        if not provider.supports_watch:
            raise SystemExit(
                f"{provider.title} does not support watch mode "
                f"(use --watch with codex, claude-code, or chatgpt)"
            )
        run_watch_cli(args, env)
        return

    # Validate offline mode
    if getattr(args, "offline", False) and args.provider == "drive":
        raise SystemExit("--offline is not supported for Drive; run without it or target a local provider.")

    # Handle root label override
    if getattr(args, "root", None):
        label = args.root
        defaults = env.config.defaults
        roots = getattr(defaults, "roots", {}) or {}
        paths = roots.get(label)
        if not paths:
            raise SystemExit(
                f"Unknown root label '{label}'. Define it in config or use a known label."
            )
        env.config.defaults.output_dirs = paths

    # Validate Drive credentials
    if args.provider == "drive":
        from ...drive_client import DEFAULT_CREDENTIALS
        cred_path = DEFAULT_CREDENTIALS
        if env.config.drive and env.config.drive.credentials_path:
            cred_path = env.config.drive.credentials_path
        if not cred_path.exists():
            raise SystemExit(
                f"Drive sync requires credentials.json at {cred_path} "
                f"(set drive.credentials_path in config)."
            )

    # Validate exports directory for ChatGPT/Claude
    if args.provider in {"chatgpt", "claude"}:
        exports_root = (
            env.config.exports.chatgpt if args.provider == "chatgpt"
            else env.config.exports.claude
        )
        if not exports_root.exists():
            raise SystemExit(
                f"{args.provider} exports directory not found: {exports_root} "
                f"(set exports.{args.provider} in config)."
            )

    # Execute sync
    run_sync_cli(args, env)
