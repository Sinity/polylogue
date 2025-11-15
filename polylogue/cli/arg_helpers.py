from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def add_html_option(parser: ArgumentParser, *, description: str = "HTML preview mode: on/off/auto (default auto)") -> None:
    parser.add_argument(
        "--html",
        dest="html_mode",
        nargs="?",
        const="on",
        default="auto",
        choices=["auto", "on", "off"],
        metavar="MODE",
        help=description,
    )


def add_collapse_option(parser: ArgumentParser, *, help_text: str = "Override collapse threshold") -> None:
    parser.add_argument("--collapse-threshold", type=int, default=None, help=help_text)


def add_out_option(parser: ArgumentParser, *, default_path: Path, help_text: str | None = None) -> None:
    description = help_text or f"Override output directory (default {default_path})"
    parser.add_argument("--out", type=Path, default=None, help=description)


def add_diff_option(parser: ArgumentParser, *, help_text: str = "Write delta diff alongside updated Markdown") -> None:
    parser.add_argument("--diff", action="store_true", help=help_text)


def add_force_option(parser: ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--force", action="store_true", help=help_text)


def add_allow_dirty_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow overwriting files with local edits (requires --force)",
    )


def add_dry_run_option(parser: ArgumentParser, *, help_text: str = "Report actions without writing files") -> None:
    parser.add_argument("--dry-run", action="store_true", help=help_text)


def resolve_base_dir(value: Path | None, default: Path) -> Path:
    """Resolve base directory with expanduser, falling back to default.

    Args:
        value: User-provided path or None
        default: Default path to use if value is None

    Returns:
        Resolved path with ~ expanded
    """
    return Path(value).expanduser() if value else default


def ensure_directory(path: Path, *, create: bool = True) -> Path:
    """Ensure directory exists, optionally creating it.

    Args:
        path: Directory path to ensure
        create: If True, create directory if missing

    Returns:
        The same path (for chaining)
    """
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
